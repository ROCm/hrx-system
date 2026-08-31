// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task.h"

void loomc_task_initialize(const loomc_task_vtable_t* vtable,
                           loomc_task_t* out_task) {
  out_task->vtable = vtable;
  out_task->next = NULL;
}

void loomc_task_execute(loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  const loomc_task_vtable_t* vtable = task->vtable;
  vtable->execute(task, worker_ordinal);
  vtable->destroy(task);
}

void loomc_task_destroy(loomc_task_t* task) { task->vtable->destroy(task); }

loomc_status_t loomc_task_sink_submit(loomc_task_sink_t sink,
                                      loomc_task_t* task) {
  if (sink.submit == NULL || task == NULL || task->vtable == NULL ||
      task->vtable->execute == NULL || task->vtable->destroy == NULL ||
      task->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "task sink and initialized task are required");
  }
  return sink.submit(sink.user_data, task);
}
