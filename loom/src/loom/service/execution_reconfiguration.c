// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/execution_reconfiguration.h"

typedef struct loom_service_execution_reconfiguration_job_t {
  // Allocator retained through the producer's final source access.
  iree_allocator_t host_allocator;
  // Stable entry retained until this owner's terminal collection.
  loom_runtime_variant_entry_t* entry;
  // Owned provider locators and package, relinquished before completion.
  loom_runtime_sources_t* sources;
  // Owner-only link in either the active or available slot list.
  struct loom_service_execution_reconfiguration_job_t* next;
} loom_service_execution_reconfiguration_job_t;

struct loom_service_execution_reconfiguration_t {
  // Allocator owning this policy and captured native-provider locators.
  iree_allocator_t host_allocator;
  // Borrowed live model through every candidate job and payload release.
  loom_service_execution_update_t update;
  // Owned immutable compiled choices; contains no service selection policy.
  loom_runtime_variant_pool_t* pool;
  // Borrowed wake target, installed on the same serialized owner.
  loom_runtime_wake_callback_t observer;
  // Acquired choice currently used by subsequent service invocations.
  loom_runtime_variant_entry_t* selected;
  // One accepted policy change, applied only at the service's quiescent
  // boundary.
  struct {
    // Acquired ready candidate; NULL means no accepted selection.
    loom_runtime_variant_entry_t* entry;
    // Terminal owner-thread notification, never a model callback thread.
    loom_service_variant_selection_fn_t completion;
    // Borrowed caller state until completion, including shutdown cancellation.
    void* user_data;
  } selection;
  // Irrevocable shutdown intent; compilation retirement is independent.
  bool stopping;
  // Independent compiler owners, retained through terminal collection.
  loom_service_execution_reconfiguration_job_t* active_jobs;
  // Available stable slots; the selected entry bounds jobs to capacity - 1.
  loom_service_execution_reconfiguration_job_t* available_jobs;
  // Tail storage for bounded jobs; live jobs never move or share an ordinal.
  loom_service_execution_reconfiguration_job_t jobs[];
};

static void loom_service_execution_reconfiguration_complete(
    void* user_data, iree_status_t status, void* variant,
    const loom_runtime_compilation_report_t* report) {
  loom_service_execution_reconfiguration_job_t* job = user_data;
  loom_runtime_variant_entry_t* entry = job->entry;
  loom_runtime_sources_free(job->sources, job->host_allocator);
  job->sources = NULL;
  // Publication is the producer's final job/entry/pool access. Collection can
  // immediately recycle the slot, including when completion precedes return.
  loom_runtime_variant_entry_complete(entry, status, variant, report);
}

static iree_status_t loom_service_execution_reconfiguration_stage(
    void* self, loom_runtime_variant_id_t base,
    const loom_runtime_sources_t* sources, loom_runtime_variant_id_t* out_id) {
  loom_service_execution_reconfiguration_t* reconfiguration = self;
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
  // Provider capture is independent of HTTP. Resolution stays with the source
  // reader so filesystem bases and package namespaces keep their meaning.
  loom_runtime_sources_t* captured_sources = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_sources_clone(
      sources, reconfiguration->host_allocator, &captured_sources));
  loom_runtime_variant_entry_t* entry = NULL;
  iree_status_t status =
      loom_runtime_variant_pool_begin(reconfiguration->pool, base, &entry);
  if (iree_status_is_ok(status)) {
    // One selected entry always consumes pool capacity. Every active job also
    // retains its entry, so a successful reservation has an available slot.
    loom_service_execution_reconfiguration_job_t* job =
        reconfiguration->available_jobs;
    IREE_ASSERT(job);
    reconfiguration->available_jobs = job->next;
    job->entry = entry;
    loom_runtime_variant_entry_retain(entry);
    job->sources = captured_sources;
    job->next = reconfiguration->active_jobs;
    reconfiguration->active_jobs = job;
    *out_id = loom_runtime_variant_entry_id(entry);
    // From here the pool owns terminal observation, including submission
    // rejection. The caller receives a stable ID for every accepted attempt.
    status = reconfiguration->update.compile(
        reconfiguration->update.self, job->sources,
        loom_service_execution_reconfiguration_complete, job);
    if (!iree_status_is_ok(status)) {
      loom_service_execution_reconfiguration_complete(job, status, NULL, NULL);
      status = iree_ok_status();
    }
  } else {
    loom_runtime_sources_free(captured_sources,
                              reconfiguration->host_allocator);
  }
  return status;
}

static loom_runtime_variant_id_t
loom_service_execution_reconfiguration_selected(void* self) {
  const loom_service_execution_reconfiguration_t* reconfiguration = self;
  return loom_runtime_variant_entry_id(reconfiguration->selected);
}

static iree_status_t loom_service_execution_reconfiguration_select(
    void* self, loom_runtime_variant_id_t id,
    loom_runtime_variant_id_t expected,
    loom_service_variant_selection_fn_t completion, void* user_data) {
  loom_service_execution_reconfiguration_t* reconfiguration = self;
  if (reconfiguration->stopping || reconfiguration->selection.entry ||
      loom_runtime_variant_entry_id(reconfiguration->selected) != expected) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "execution selection conflicts with shutdown, a pending "
        "selection or its base");
  }
  loom_runtime_variant_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_runtime_variant_pool_acquire(reconfiguration->pool, id, &entry));
  reconfiguration->selection.entry = entry;
  reconfiguration->selection.completion = completion;
  reconfiguration->selection.user_data = user_data;
  if (reconfiguration->observer.wake) {
    reconfiguration->observer.wake(reconfiguration->observer.user_data);
  }
  return iree_ok_status();
}

iree_status_t loom_service_execution_reconfiguration_allocate(
    loom_service_execution_update_t update, iree_host_size_t capacity,
    iree_allocator_t host_allocator,
    loom_service_execution_reconfiguration_t** out_reconfiguration) {
  *out_reconfiguration = NULL;
  if (!capacity || !update.compile) {
    return iree_ok_status();
  }
  loom_service_execution_reconfiguration_t* reconfiguration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*reconfiguration), capacity - 1,
      sizeof(*reconfiguration->jobs), (void**)&reconfiguration));
  reconfiguration->host_allocator = host_allocator;
  reconfiguration->update = update;
  for (iree_host_size_t i = capacity - 1; i > 0; --i) {
    loom_service_execution_reconfiguration_job_t* job =
        &reconfiguration->jobs[i - 1];
    job->host_allocator = host_allocator;
    job->next = reconfiguration->available_jobs;
    reconfiguration->available_jobs = job;
  }
  iree_status_t status = loom_runtime_variant_pool_allocate(
      capacity, NULL, update.dispose, host_allocator,
      &reconfiguration->selected, &reconfiguration->pool);
  if (iree_status_is_ok(status)) {
    loom_runtime_variant_entry_retain(reconfiguration->selected);
    *out_reconfiguration = reconfiguration;
  } else {
    loom_service_execution_reconfiguration_free(reconfiguration);
  }
  return status;
}

void loom_service_execution_reconfiguration_free(
    loom_service_execution_reconfiguration_t* reconfiguration) {
  if (!reconfiguration) {
    return;
  }
  IREE_ASSERT(!reconfiguration->selection.entry);
  IREE_ASSERT(!reconfiguration->active_jobs);
  loom_runtime_variant_entry_release(reconfiguration->selected);
  loom_runtime_variant_pool_free(reconfiguration->pool);
  iree_allocator_free(reconfiguration->host_allocator, reconfiguration);
}

void loom_service_execution_reconfiguration_set_observer(
    loom_service_execution_reconfiguration_t* reconfiguration,
    loom_runtime_wake_callback_t observer) {
  if (!reconfiguration) {
    return;
  }
  reconfiguration->observer = observer;
  loom_runtime_variant_pool_set_observer(reconfiguration->pool, observer);
}

bool loom_service_execution_reconfiguration_pending(
    const loom_service_execution_reconfiguration_t* reconfiguration) {
  return reconfiguration &&
         (reconfiguration->selection.entry || reconfiguration->active_jobs);
}

bool loom_service_execution_reconfiguration_advance(
    loom_service_execution_reconfiguration_t* reconfiguration, bool quiescent,
    bool stopping) {
  if (!reconfiguration) {
    return false;
  }
  reconfiguration->stopping |= stopping;
  bool progress = false;
  loom_service_execution_reconfiguration_job_t** link =
      &reconfiguration->active_jobs;
  while (*link) {
    loom_service_execution_reconfiguration_job_t* job = *link;
    loom_runtime_variant_entry_collect(job->entry);
    if (loom_runtime_variant_entry_snapshot(job->entry).phase ==
        LOOM_RUNTIME_VARIANT_COMPILING) {
      link = &job->next;
    } else {
      loom_runtime_variant_entry_release(job->entry);
      job->entry = NULL;
      *link = job->next;
      job->next = reconfiguration->available_jobs;
      reconfiguration->available_jobs = job;
      progress = true;
    }
  }
  loom_runtime_variant_entry_t* entry = reconfiguration->selection.entry;
  if (!entry || (!quiescent && !reconfiguration->stopping)) {
    return progress;
  }
  iree_status_t status =
      reconfiguration->stopping
          ? iree_make_status(IREE_STATUS_CANCELLED, "service is stopping")
          : reconfiguration->update.select(
                reconfiguration->update.self,
                loom_runtime_variant_entry_value(entry));
  if (iree_status_is_ok(status)) {
    loom_runtime_variant_entry_release(reconfiguration->selected);
    reconfiguration->selected = entry;
  } else {
    loom_runtime_variant_entry_release(entry);
  }
  const loom_service_variant_selection_fn_t completion =
      reconfiguration->selection.completion;
  void* user_data = reconfiguration->selection.user_data;
  reconfiguration->selection.entry = NULL;
  reconfiguration->selection.completion = NULL;
  reconfiguration->selection.user_data = NULL;
  completion(user_data, status,
             loom_runtime_variant_entry_id(reconfiguration->selected));
  return true;
}

loom_service_reconfiguration_t loom_service_execution_reconfiguration_interface(
    loom_service_execution_reconfiguration_t* reconfiguration) {
  if (!reconfiguration) {
    return (loom_service_reconfiguration_t){0};
  }
  return (loom_service_reconfiguration_t){
      .self = reconfiguration,
      .pool = reconfiguration->pool,
      .stage_native = loom_service_execution_reconfiguration_stage,
      .select = loom_service_execution_reconfiguration_select,
      .selected = loom_service_execution_reconfiguration_selected};
}
