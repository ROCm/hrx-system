// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task.h"

#include "iree/base/api.h"

void loomc_task_initialize(const loomc_task_vtable_t* vtable,
                           loomc_task_t* out_task) {
  out_task->vtable = vtable;
  out_task->next = NULL;
}

void loomc_task_execute(loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, worker_ordinal);
  const loomc_task_vtable_t* vtable = task->vtable;
  vtable->execute(task, worker_ordinal);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_destroy, "loomc_task_destroy");
  vtable->destroy(task);
  IREE_TRACE_ZONE_END(z_destroy);
  IREE_TRACE_ZONE_END(z0);
}

void loomc_task_destroy(loomc_task_t* task) {
  IREE_TRACE_ZONE_BEGIN(z0);
  task->vtable->destroy(task);
  IREE_TRACE_ZONE_END(z0);
}

loomc_status_t loomc_task_sink_submit(loomc_task_sink_t sink,
                                      loomc_task_t* task) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (sink.submit == NULL || task == NULL || task->vtable == NULL ||
      task->vtable->execute == NULL || task->vtable->destroy == NULL ||
      task->next != NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                          "task sink and initialized task are required");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  loomc_status_t return_status = sink.submit(sink.user_data, task);
  IREE_TRACE_ZONE_END(z0);
  return return_status;
}
