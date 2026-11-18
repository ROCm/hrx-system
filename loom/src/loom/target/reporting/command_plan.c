// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/command_plan.h"

#include <string.h>

#include "loom/target/reporting/report.h"

static iree_status_t loom_target_compile_report_command_copy_rows(
    const void* source, iree_host_size_t count, iree_host_size_t row_size,
    iree_allocator_t allocator, const void** out_rows) {
  *out_rows = NULL;
  if (source == NULL || count == 0) {
    return iree_ok_status();
  }
  void* rows = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(allocator, count, row_size, &rows));
  memcpy(rows, source, count * row_size);
  *out_rows = rows;
  return iree_ok_status();
}

static void loom_target_compile_report_command_plan_deinitialize(
    loom_target_compile_report_command_plan_t* plan,
    iree_allocator_t allocator) {
  iree_allocator_free(allocator, (void*)plan->commands);
  iree_allocator_free(allocator, (void*)plan->bindings);
  iree_allocator_free(allocator, (void*)plan->parameter_roots);
  iree_allocator_free(allocator, (void*)plan->parameters);
}

void loom_target_compile_report_command_plan_list_deinitialize(
    loom_target_compile_report_row_list_t* list,
    iree_allocator_t host_allocator) {
  for (loom_target_compile_report_vec_t* block = list->head; block;
       block = block->next) {
    loom_target_compile_report_command_plan_t* plans =
        loom_target_compile_report_vec_rows(block);
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      loom_target_compile_report_command_plan_deinitialize(&plans[i],
                                                           host_allocator);
    }
  }
  loom_target_compile_report_row_list_deinitialize(host_allocator, list);
}

iree_status_t loom_target_compile_report_record_command_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_command_plan_t* plan) {
  if (iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  loom_target_compile_report_command_plan_t copy = *plan;
  copy.commands = NULL;
  copy.bindings = NULL;
  copy.parameter_roots = NULL;
  copy.parameters = NULL;
  iree_status_t status = iree_ok_status();
  if (loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_COMMAND_PLAN_ROWS)) {
    status = loom_target_compile_report_command_copy_rows(
        plan->commands, plan->command_count, sizeof(*plan->commands),
        report->allocator, (const void**)&copy.commands);
    if (iree_status_is_ok(status)) {
      status = loom_target_compile_report_command_copy_rows(
          plan->bindings, plan->binding_count, sizeof(*plan->bindings),
          report->allocator, (const void**)&copy.bindings);
    }
    if (iree_status_is_ok(status)) {
      status = loom_target_compile_report_command_copy_rows(
          plan->parameter_roots, plan->parameter_root_count,
          sizeof(*plan->parameter_roots), report->allocator,
          (const void**)&copy.parameter_roots);
    }
    if (iree_status_is_ok(status)) {
      status = loom_target_compile_report_command_copy_rows(
          plan->parameters, plan->parameter_count, sizeof(*plan->parameters),
          report->allocator, (const void**)&copy.parameters);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_append(
        &report->command_plans, sizeof(copy), report->allocator, &copy);
  }
  if (iree_status_is_ok(status) && (copy.commands || copy.bindings ||
                                    copy.parameter_roots || copy.parameters)) {
    report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_COMMAND_PLAN_ROWS;
  }
  if (!iree_status_is_ok(status)) {
    loom_target_compile_report_command_plan_deinitialize(&copy,
                                                         report->allocator);
  }
  return status;
}

iree_string_view_t loom_target_compile_report_command_kind_name(
    loom_target_compile_report_command_kind_t kind) {
  static const iree_string_view_t names[] = {
      IREE_SVL("fill"),
      IREE_SVL("copy"),
      IREE_SVL("dispatch_direct"),
      IREE_SVL("dispatch_indirect_static"),
      IREE_SVL("dispatch_indirect_dynamic"),
      IREE_SVL("barrier"),
      IREE_SVL("timestamp"),
  };
  static_assert(
      IREE_ARRAYSIZE(names) == LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT,
      "every command kind has a diagnostic name");
  return names[kind];
}

iree_string_view_t loom_target_compile_report_command_buffer_role_name(
    loom_target_compile_report_command_buffer_role_t role) {
  return role == LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_FIXED
             ? IREE_SV("fixed")
             : IREE_SV("rebindable");
}
