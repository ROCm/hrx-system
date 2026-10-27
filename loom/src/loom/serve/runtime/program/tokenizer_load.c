// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/tokenizer_load.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loom/model/card.h"
#include "loomc/iree.h"

struct loom_serve_tokenizer_load_t {
  // Accepted task ownership ends at its final publication callback.
  loomc_task_t task;
  // Allocator owning this operation and the transferred tokenizer.
  iree_allocator_t host_allocator;
  // Borrowed shared cache owned beyond every accepted load.
  loom_serve_tokenizer_cache_t* cache;
  // Immutable source/package views borrowed through final take.
  loom_serve_sources_t sources;
  // Copied locator stored immediately after this record.
  iree_string_view_t locator;
  // Serializes final publication and wake with the owner's take/destruction.
  iree_slim_mutex_t mutex;
  // Intrusive cache subscription retained until terminal publication.
  loom_serve_tokenizer_subscription_t subscription;
  // The scheduled task has relinquished all input and output borrows.
  bool task_retired;
  // The cache acquisition has delivered its terminal result.
  bool cache_completed;
  // Both asynchronous owners retired and the result may be taken.
  bool ready;
  // Worker outcome transferred to the load owner on take.
  iree_status_t status;
  // Shared tokenizer reference transferred even when another branch failed.
  loom_serve_tokenizer_t* tokenizer;
  // Nonblocking notification scheduling the serialized load owner.
  void (*wake)(void* user_data);
  // Notification payload borrowed through terminal take.
  void* user_data;
};

static void loom_serve_tokenizer_load_publish(
    loom_serve_tokenizer_load_t* load) {
  if (!load->ready && load->task_retired && load->cache_completed) {
    load->ready = true;
    load->wake(load->user_data);
  }
}

static void loom_serve_tokenizer_load_cache_complete(
    void* user_data, iree_status_t status, loom_serve_tokenizer_t* tokenizer) {
  loom_serve_tokenizer_load_t* load = user_data;
  iree_slim_mutex_lock(&load->mutex);
  load->status = status;
  load->tokenizer = tokenizer;
  load->cache_completed = true;
  loom_serve_tokenizer_load_publish(load);
  iree_slim_mutex_unlock(&load->mutex);
}

static void loom_serve_tokenizer_load_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  loom_serve_tokenizer_load_t* load = (loom_serve_tokenizer_load_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, load->locator.data, load->locator.size);
  loomc_source_t* contents = NULL;
  iree_status_t status = loom_serve_sources_open(
      &load->sources, load->locator, load->host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t bytes = loomc_source_contents(contents);
    loom_serve_tokenizer_acquisition_t acquisition;
    status = loom_serve_tokenizer_cache_acquire(load->cache, contents,
                                                &load->subscription,
                                                &acquisition, &load->tokenizer);
    if (iree_status_is_ok(status) &&
        acquisition == LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER) {
      iree_tokenizer_t* value = NULL;
      status = iree_tokenizer_from_huggingface_json(
          iree_make_string_view((const char*)bytes.data, bytes.data_length),
          load->host_allocator, &value);
      const iree_status_t publication_status =
          loom_serve_tokenizer_cache_complete(load->cache, contents, status,
                                              value, &load->tokenizer);
      status = iree_status_join(status, publication_status);
    }
    if (iree_status_is_ok(status) &&
        acquisition == LOOM_SERVE_TOKENIZER_ACQUISITION_PENDING) {
      loomc_source_release(contents);
      IREE_TRACE_ZONE_END(z0);
      return;
    }
  }
  loomc_source_release(contents);
  load->status = status;
  load->cache_completed = true;
  IREE_TRACE_ZONE_END(z0);
}

static void loom_serve_tokenizer_load_release(loomc_task_t* task) {
  loom_serve_tokenizer_load_t* load = (loom_serve_tokenizer_load_t*)task;
  iree_slim_mutex_lock(&load->mutex);
  load->task_retired = true;
  loom_serve_tokenizer_load_publish(load);
  iree_slim_mutex_unlock(&load->mutex);
}

static const loomc_task_vtable_t tokenizer_load_vtable = {
    .execute = loom_serve_tokenizer_load_execute,
    .destroy = loom_serve_tokenizer_load_release,
};

static iree_status_t loom_serve_tokenizer_load_schedule(
    loom_serve_tokenizer_cache_t* cache, loomc_task_sink_t sink,
    const loom_serve_sources_t* sources, iree_string_view_t locator,
    void (*wake)(void* user_data), void* user_data,
    iree_allocator_t host_allocator, loom_serve_tokenizer_load_t** out_load) {
  *out_load = NULL;
  loom_serve_tokenizer_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*load) + locator.size, (void**)&load));
  loomc_task_initialize(&tokenizer_load_vtable, &load->task);
  load->host_allocator = host_allocator;
  load->cache = cache;
  load->sources = *sources;
  char* path = (char*)(load + 1);
  if (locator.size) {
    memcpy(path, locator.data, locator.size);
  }
  load->locator = iree_make_string_view(path, locator.size);
  load->wake = wake;
  load->user_data = user_data;
  load->subscription.completion = loom_serve_tokenizer_load_cache_complete;
  load->subscription.user_data = load;
  iree_slim_mutex_initialize(&load->mutex);
  iree_status_t status =
      iree_status_from_loomc(loomc_task_sink_submit(sink, &load->task));
  if (iree_status_is_ok(status)) {
    *out_load = load;
  } else {
    iree_slim_mutex_deinitialize(&load->mutex);
    iree_allocator_free(host_allocator, load);
  }
  return status;
}

iree_status_t loom_serve_tokenizer_load_start(
    loom_serve_tokenizer_cache_t* cache, loomc_task_sink_t sink,
    const loom_serve_sources_t* sources, iree_string_view_t directory,
    iree_string_view_t name, void (*wake)(void* user_data), void* user_data,
    iree_allocator_t host_allocator, loom_serve_tokenizer_load_t** out_load) {
  *out_load = NULL;
  if (sources->card) {
    const loom_model_tokenizer_t* tokenizer = loom_model_selection_tokenizer(
        loom_model_card_selection(sources->card), name);
    if (!tokenizer) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "model does not declare tokenizers.%.*s",
                              (int)name.size, name.data);
    }
    const loom_model_file_t* file = &tokenizer->file;
    if ((file->format.size &&
         !iree_string_view_equal(file->format, IREE_SV("huggingface-json"))) ||
        (!file->format.size &&
         !iree_string_view_ends_with(file->path, IREE_SV(".json")))) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "tokenizers.%.*s requires huggingface-json format; specify format "
          "for a renamed file",
          (int)name.size, name.data);
    }
    return loom_serve_tokenizer_load_schedule(cache, sink, sources, file->path,
                                              wake, user_data, host_allocator,
                                              out_load);
  }
  if (iree_string_view_is_empty(directory)) {
    return loom_serve_tokenizer_load_schedule(
        cache, sink, sources, name, wake, user_data, host_allocator, out_load);
  }
  char* path = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_path_resolve(directory, name, host_allocator, &path));
  iree_status_t status = loom_serve_tokenizer_load_schedule(
      cache, sink, sources, iree_make_cstring_view(path), wake, user_data,
      host_allocator, out_load);
  iree_allocator_free(host_allocator, path);
  return status;
}

bool loom_serve_tokenizer_load_take(loom_serve_tokenizer_load_t** load_ptr,
                                    iree_status_t* status,
                                    loom_serve_tokenizer_t** out_tokenizer) {
  loom_serve_tokenizer_load_t* load = *load_ptr;
  if (!load) {
    return true;
  }
  iree_slim_mutex_lock(&load->mutex);
  const bool ready = load->ready;
  iree_slim_mutex_unlock(&load->mutex);
  if (!ready) {
    return false;
  }
  *status = iree_status_join(*status, load->status);
  *out_tokenizer = load->tokenizer;
  iree_slim_mutex_deinitialize(&load->mutex);
  iree_allocator_free(load->host_allocator, load);
  *load_ptr = NULL;
  return true;
}
