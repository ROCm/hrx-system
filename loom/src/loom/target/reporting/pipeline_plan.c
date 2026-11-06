// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/pipeline_plan.h"

#include <string.h>

#include "loom/target/reporting/report.h"

static iree_status_t loom_target_compile_report_pipeline_plan_copy_rows(
    const void* source, iree_host_size_t count, iree_host_size_t row_size,
    iree_allocator_t host_allocator, void** out_rows) {
  *out_rows = NULL;
  if (count == 0 || iree_allocator_is_null(host_allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t byte_count = 0;
  if (!iree_host_size_checked_mul(count, row_size, &byte_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline report row storage is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, byte_count, out_rows));
  memcpy(*out_rows, source, byte_count);
  return iree_ok_status();
}

static void loom_target_compile_report_pipeline_plan_deinitialize(
    loom_target_compile_report_pipeline_plan_t* plan,
    iree_allocator_t host_allocator) {
  if (plan == NULL) {
    return;
  }
  iree_allocator_free(host_allocator, (void*)plan->worker_rows);
  iree_allocator_free(host_allocator, (void*)plan->channel_rows);
  iree_allocator_free(host_allocator, (void*)plan->memory_rows);
  *plan = (loom_target_compile_report_pipeline_plan_t){0};
}

static iree_status_t loom_target_compile_report_pipeline_plan_clone(
    const loom_target_compile_report_pipeline_plan_t* source,
    iree_allocator_t host_allocator,
    loom_target_compile_report_pipeline_plan_t* out_target) {
  *out_target = (loom_target_compile_report_pipeline_plan_t){
      .summary = source->summary,
  };
  void* worker_rows = NULL;
  iree_status_t status = loom_target_compile_report_pipeline_plan_copy_rows(
      source->worker_rows, source->worker_row_count,
      sizeof(*source->worker_rows), host_allocator, &worker_rows);
  if (iree_status_is_ok(status)) {
    out_target->worker_rows = worker_rows;
    out_target->worker_row_count =
        worker_rows != NULL ? source->worker_row_count : 0;
    void* channel_rows = NULL;
    status = loom_target_compile_report_pipeline_plan_copy_rows(
        source->channel_rows, source->channel_row_count,
        sizeof(*source->channel_rows), host_allocator, &channel_rows);
    if (iree_status_is_ok(status)) {
      out_target->channel_rows = channel_rows;
      out_target->channel_row_count =
          channel_rows != NULL ? source->channel_row_count : 0;
    }
  }
  if (iree_status_is_ok(status)) {
    void* memory_rows = NULL;
    status = loom_target_compile_report_pipeline_plan_copy_rows(
        source->memory_rows, source->memory_row_count,
        sizeof(*source->memory_rows), host_allocator, &memory_rows);
    if (iree_status_is_ok(status)) {
      out_target->memory_rows = memory_rows;
      out_target->memory_row_count = memory_rows ? source->memory_row_count : 0;
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_target_compile_report_pipeline_plan_deinitialize(out_target,
                                                          host_allocator);
  }
  return status;
}

void loom_target_compile_report_pipeline_plan_list_deinitialize(
    loom_target_compile_report_row_list_t* list,
    iree_allocator_t host_allocator) {
  for (loom_target_compile_report_vec_t* block = list->head; block;
       block = block->next) {
    loom_target_compile_report_pipeline_plan_t* plans =
        loom_target_compile_report_vec_rows(block);
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      loom_target_compile_report_pipeline_plan_deinitialize(&plans[i],
                                                            host_allocator);
    }
  }
  loom_target_compile_report_row_list_deinitialize(host_allocator, list);
}

iree_status_t loom_target_compile_report_record_pipeline_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pipeline_plan_t* plan) {
  if (report == NULL || iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }

  loom_target_compile_report_pipeline_plan_t copied_plan = {
      .summary = plan->summary,
  };
  if (loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_pipeline_plan_clone(
        plan, report->allocator, &copied_plan));
  }
  iree_status_t status = loom_target_compile_report_row_list_append(
      &report->pipeline_plans, sizeof(copied_plan), report->allocator,
      &copied_plan);
  if (!iree_status_is_ok(status)) {
    loom_target_compile_report_pipeline_plan_deinitialize(&copied_plan,
                                                          report->allocator);
    return status;
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN;
  if (copied_plan.worker_row_count != 0 || copied_plan.channel_row_count != 0 ||
      copied_plan.memory_row_count != 0) {
    report->detail_flags |=
        LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN_ROWS;
  }
  return iree_ok_status();
}

iree_string_view_t loom_target_compile_report_pipeline_endpoint_owner_name(
    loom_target_compile_report_pipeline_endpoint_owner_t owner) {
  switch (owner) {
    case LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_BINDING:
      return IREE_SV("binding");
    case LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_WORKER:
      return IREE_SV("worker");
    default:
      return IREE_SV("unknown");
  }
}
