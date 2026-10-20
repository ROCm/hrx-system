// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/execution_reconfiguration.h"

typedef struct loom_serve_execution_reconfiguration_job_t {
  // Next compilation awaiting admission or actual producer collection.
  struct loom_serve_execution_reconfiguration_job_t* next;
  // Allocator retained independently of service retirement.
  iree_allocator_t host_allocator;
  // Host claim retained until the owner collects terminal producer publication.
  loom_serve_execution_permit_t permit;
  // Retained entry through owner collection and job destruction.
  loom_serve_variant_entry_t* entry;
  // Owned provider locators, filesystem base and retained package, independent
  // of HTTP. Resolution stays with the source reader so relative file bases
  // and package namespaces keep their original meaning.
  loom_serve_sources_t* sources;
} loom_serve_execution_reconfiguration_job_t;

struct loom_serve_execution_reconfiguration_t {
  // Allocator owning this policy and captured native-provider locators.
  iree_allocator_t host_allocator;
  // Borrowed live model through every candidate job and payload release.
  loom_serve_execution_update_t update;
  // Borrowed application policy over the model's actual shared host executor.
  loom_serve_execution_scheduler_t* scheduler;
  // Owned compilation jobs, including unstarted offers and accepted producers.
  loom_serve_execution_reconfiguration_job_t* jobs;
  // Owned immutable compiled choices; contains no service selection policy.
  loom_serve_variant_pool_t* pool;
  // Borrowed wake target, installed on the same serialized owner.
  loom_serve_wake_callback_t observer;
  // Acquired choice currently used by subsequent service invocations.
  loom_serve_variant_entry_t* selected;
  // One accepted policy change, applied only at the service's quiescent
  // boundary.
  struct {
    // Acquired ready candidate; NULL means no accepted selection.
    loom_serve_variant_entry_t* entry;
    // Terminal owner-thread notification, never a model callback thread.
    loom_serve_variant_selection_fn_t completion;
    // Borrowed caller state until completion, including shutdown cancellation.
    void* user_data;
  } selection;
  // Irrevocable shutdown intent; compilation retirement is independent.
  bool stopping;
};

static void loom_serve_execution_reconfiguration_job_free(
    loom_serve_execution_reconfiguration_job_t* job) {
  loom_serve_execution_permit_deinitialize(&job->permit);
  loom_serve_variant_entry_release(job->entry);
  loom_serve_sources_free(job->sources, job->host_allocator);
  iree_allocator_free(job->host_allocator, job);
}

static void loom_serve_execution_reconfiguration_complete(
    void* user_data, iree_status_t status, void* variant,
    const loom_serve_compilation_report_t* report) {
  loom_serve_execution_reconfiguration_job_t* job = user_data;
  loom_serve_variant_entry_complete(job->entry, status, variant, report);
}

static bool loom_serve_execution_reconfiguration_submit(
    loom_serve_execution_reconfiguration_t* reconfiguration,
    loom_serve_execution_reconfiguration_job_t* job) {
  if (!loom_serve_execution_permit_try_acquire(&job->permit)) {
    return false;
  }
  iree_status_t status = reconfiguration->update.compile(
      reconfiguration->update.self, job->sources,
      loom_serve_execution_reconfiguration_complete, job);
  if (!iree_status_is_ok(status)) {
    loom_serve_execution_reconfiguration_complete(job, status, NULL, NULL);
  }
  return true;
}

static iree_status_t loom_serve_execution_reconfiguration_stage(
    void* self, loom_serve_variant_id_t base,
    const loom_serve_sources_t* sources, loom_serve_variant_id_t* out_id) {
  loom_serve_execution_reconfiguration_t* reconfiguration = self;
  *out_id = 0;
  if (reconfiguration->stopping) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "service is stopping");
  }
  if (!sources->count || sources->prepare.size || sources->control.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native reconfiguration requires providers and "
                            "preserves prepare/control");
  }
  loom_serve_execution_reconfiguration_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(reconfiguration->host_allocator,
                                             sizeof(*job), (void**)&job));
  job->host_allocator = reconfiguration->host_allocator;
  iree_status_t status =
      loom_serve_sources_clone(sources, job->host_allocator, &job->sources);
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_variant_pool_begin(reconfiguration->pool, base, &job->entry);
  }
  if (iree_status_is_ok(status)) {
    *out_id = loom_serve_variant_entry_id(job->entry);
    loom_serve_variant_entry_retain(job->entry);
    job->next = reconfiguration->jobs;
    reconfiguration->jobs = job;
    loom_serve_execution_permit_initialize(reconfiguration->scheduler,
                                           LOOM_SERVE_EXECUTION_RESOURCE_HOST,
                                           &job->permit);
    loom_serve_execution_permit_request(&job->permit,
                                        LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
    // From here the pool owns terminal observation, including submission
    // rejection. The caller receives a stable ID for every accepted attempt.
    loom_serve_execution_reconfiguration_submit(reconfiguration, job);
  } else {
    loom_serve_execution_reconfiguration_job_free(job);
  }
  return status;
}

static loom_serve_variant_id_t loom_serve_execution_reconfiguration_selected(
    void* self) {
  const loom_serve_execution_reconfiguration_t* reconfiguration = self;
  return loom_serve_variant_entry_id(reconfiguration->selected);
}

static iree_status_t loom_serve_execution_reconfiguration_select(
    void* self, loom_serve_variant_id_t id, loom_serve_variant_id_t expected,
    loom_serve_variant_selection_fn_t completion, void* user_data) {
  loom_serve_execution_reconfiguration_t* reconfiguration = self;
  if (reconfiguration->stopping || reconfiguration->selection.entry ||
      loom_serve_variant_entry_id(reconfiguration->selected) != expected) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "execution selection conflicts with shutdown, a pending "
        "selection or its base");
  }
  loom_serve_variant_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_variant_pool_acquire(reconfiguration->pool, id, &entry));
  reconfiguration->selection.entry = entry;
  reconfiguration->selection.completion = completion;
  reconfiguration->selection.user_data = user_data;
  if (reconfiguration->observer.wake) {
    reconfiguration->observer.wake(reconfiguration->observer.user_data);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_execution_reconfiguration_allocate(
    loom_serve_execution_update_t update,
    loom_serve_execution_scheduler_t* scheduler, iree_host_size_t capacity,
    iree_allocator_t host_allocator,
    loom_serve_execution_reconfiguration_t** out_reconfiguration) {
  *out_reconfiguration = NULL;
  if (!capacity || !update.compile) {
    return iree_ok_status();
  }
  loom_serve_execution_reconfiguration_t* reconfiguration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*reconfiguration), (void**)&reconfiguration));
  reconfiguration->host_allocator = host_allocator;
  reconfiguration->update = update;
  reconfiguration->scheduler = scheduler;
  iree_status_t status = loom_serve_variant_pool_allocate(
      capacity, NULL, update.dispose, host_allocator,
      &reconfiguration->selected, &reconfiguration->pool);
  if (iree_status_is_ok(status)) {
    loom_serve_variant_entry_retain(reconfiguration->selected);
    *out_reconfiguration = reconfiguration;
  } else {
    loom_serve_execution_reconfiguration_free(reconfiguration);
  }
  return status;
}

void loom_serve_execution_reconfiguration_free(
    loom_serve_execution_reconfiguration_t* reconfiguration) {
  if (!reconfiguration) {
    return;
  }
  IREE_ASSERT(!reconfiguration->selection.entry);
  IREE_ASSERT(!reconfiguration->jobs);
  loom_serve_variant_entry_release(reconfiguration->selected);
  loom_serve_variant_pool_free(reconfiguration->pool);
  iree_allocator_free(reconfiguration->host_allocator, reconfiguration);
}

void loom_serve_execution_reconfiguration_set_observer(
    loom_serve_execution_reconfiguration_t* reconfiguration,
    loom_serve_wake_callback_t observer) {
  if (!reconfiguration) {
    return;
  }
  reconfiguration->observer = observer;
  loom_serve_variant_pool_set_observer(reconfiguration->pool, observer);
}

bool loom_serve_execution_reconfiguration_pending(
    const loom_serve_execution_reconfiguration_t* reconfiguration) {
  return reconfiguration &&
         (reconfiguration->selection.entry || reconfiguration->jobs);
}

bool loom_serve_execution_reconfiguration_advance(
    loom_serve_execution_reconfiguration_t* reconfiguration, bool quiescent,
    bool stopping) {
  if (!reconfiguration) {
    return false;
  }
  reconfiguration->stopping |= stopping;
  bool progress = loom_serve_variant_pool_collect(reconfiguration->pool);
  loom_serve_execution_reconfiguration_job_t** next = &reconfiguration->jobs;
  while (*next) {
    loom_serve_execution_reconfiguration_job_t* job = *next;
    if (loom_serve_variant_entry_snapshot(job->entry).phase !=
        LOOM_SERVE_VARIANT_COMPILING) {
      *next = job->next;
      loom_serve_execution_reconfiguration_job_free(job);
      progress = true;
      continue;
    }
    if (job->permit.phase == LOOM_SERVE_EXECUTION_PERMIT_WAITING) {
      if (reconfiguration->stopping) {
        loom_serve_execution_permit_cancel(&job->permit);
        loom_serve_execution_reconfiguration_complete(
            job, iree_make_status(IREE_STATUS_CANCELLED, "service is stopping"),
            NULL, NULL);
        progress = true;
      } else {
        progress |=
            loom_serve_execution_reconfiguration_submit(reconfiguration, job);
      }
    }
    next = &job->next;
  }
  loom_serve_variant_entry_t* entry = reconfiguration->selection.entry;
  if (!entry || (!quiescent && !reconfiguration->stopping)) {
    return progress;
  }
  iree_status_t status =
      reconfiguration->stopping
          ? iree_make_status(IREE_STATUS_CANCELLED, "service is stopping")
          : reconfiguration->update.select(
                reconfiguration->update.self,
                loom_serve_variant_entry_value(entry));
  if (iree_status_is_ok(status)) {
    loom_serve_variant_entry_release(reconfiguration->selected);
    reconfiguration->selected = entry;
  } else {
    loom_serve_variant_entry_release(entry);
  }
  const loom_serve_variant_selection_fn_t completion =
      reconfiguration->selection.completion;
  void* user_data = reconfiguration->selection.user_data;
  reconfiguration->selection.entry = NULL;
  reconfiguration->selection.completion = NULL;
  reconfiguration->selection.user_data = NULL;
  completion(user_data, status,
             loom_serve_variant_entry_id(reconfiguration->selected));
  return true;
}

loom_serve_reconfiguration_t loom_serve_execution_reconfiguration_interface(
    loom_serve_execution_reconfiguration_t* reconfiguration) {
  if (!reconfiguration) {
    return (loom_serve_reconfiguration_t){0};
  }
  return (loom_serve_reconfiguration_t){
      .self = reconfiguration,
      .pool = reconfiguration->pool,
      .stage_native = loom_serve_execution_reconfiguration_stage,
      .select = loom_serve_execution_reconfiguration_select,
      .selected = loom_serve_execution_reconfiguration_selected};
}
