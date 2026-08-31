// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task.h"

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

typedef struct test_task_t {
  // Generic task base.
  loomc_task_t base;

  // External execution count that outlives this task.
  int* execution_count;

  // External destruction count that outlives this task.
  int* destruction_count;

  // External worker ordinal record that outlives this task.
  loomc_host_size_t* worker_ordinal;
} test_task_t;

static void ExecuteTestTask(loomc_task_t* base_task,
                            loomc_host_size_t worker_ordinal) {
  test_task_t* task = reinterpret_cast<test_task_t*>(base_task);
  ++*task->execution_count;
  *task->worker_ordinal = worker_ordinal;
}

static void DestroyTestTask(loomc_task_t* base_task) {
  test_task_t* task = reinterpret_cast<test_task_t*>(base_task);
  ++*task->destruction_count;
  delete task;
}

static const loomc_task_vtable_t kTestTaskVtable = {
    /*.execute=*/ExecuteTestTask,
    /*.destroy=*/DestroyTestTask,
};

static loomc_task_t* AllocateTestTask(int* execution_count,
                                      int* destruction_count,
                                      loomc_host_size_t* worker_ordinal) {
  test_task_t* task = new test_task_t;
  loomc_task_initialize(&kTestTaskVtable, &task->base);
  task->execution_count = execution_count;
  task->destruction_count = destruction_count;
  task->worker_ordinal = worker_ordinal;
  return &task->base;
}

typedef struct test_sink_t {
  // Whether submissions are accepted.
  bool accepts_tasks;

  // Whether accepted tasks execute before submission returns.
  bool executes_inline;

  // Worker ordinal supplied to executed tasks.
  loomc_host_size_t worker_ordinal;

  // Accepted task waiting for explicit execution.
  loomc_task_t* pending_task;
} test_sink_t;

static loomc_status_t SubmitTestTask(void* user_data, loomc_task_t* task) {
  test_sink_t* sink = static_cast<test_sink_t*>(user_data);
  if (!sink->accepts_tasks) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE, "queue stopped");
  }
  if (sink->executes_inline) {
    loomc_task_execute(task, sink->worker_ordinal);
  } else {
    sink->pending_task = task;
  }
  return loomc_ok_status();
}

static loomc_task_sink_t TestTaskSink(test_sink_t* sink) {
  return (loomc_task_sink_t){
      /*.submit=*/SubmitTestTask,
      /*.user_data=*/sink,
  };
}

TEST(TaskTest, AcceptedSubmissionTransfersExecutionAndDestruction) {
  int execution_count = 0;
  int destruction_count = 0;
  loomc_host_size_t worker_ordinal = 0;
  test_sink_t sink = {
      /*.accepts_tasks=*/true,
      /*.executes_inline=*/false,
      /*.worker_ordinal=*/7,
      /*.pending_task=*/nullptr,
  };
  loomc_task_t* task =
      AllocateTestTask(&execution_count, &destruction_count, &worker_ordinal);

  LOOMC_ASSERT_OK(loomc_task_sink_submit(TestTaskSink(&sink), task));
  EXPECT_EQ(execution_count, 0);
  EXPECT_EQ(destruction_count, 0);
  ASSERT_NE(sink.pending_task, nullptr);

  loomc_task_execute(sink.pending_task, sink.worker_ordinal);
  sink.pending_task = nullptr;
  EXPECT_EQ(execution_count, 1);
  EXPECT_EQ(destruction_count, 1);
  EXPECT_EQ(worker_ordinal, 7u);
}

TEST(TaskTest, RejectedSubmissionLeavesOwnershipWithCaller) {
  int execution_count = 0;
  int destruction_count = 0;
  loomc_host_size_t worker_ordinal = 0;
  test_sink_t sink = {
      /*.accepts_tasks=*/false,
      /*.executes_inline=*/false,
      /*.worker_ordinal=*/0,
      /*.pending_task=*/nullptr,
  };
  loomc_task_t* task =
      AllocateTestTask(&execution_count, &destruction_count, &worker_ordinal);

  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_UNAVAILABLE,
                         loomc_task_sink_submit(TestTaskSink(&sink), task));
  EXPECT_EQ(execution_count, 0);
  EXPECT_EQ(destruction_count, 0);
  EXPECT_EQ(sink.pending_task, nullptr);

  loomc_task_destroy(task);
  EXPECT_EQ(execution_count, 0);
  EXPECT_EQ(destruction_count, 1);
}

TEST(TaskTest, InlineSubmissionMayDestroyTaskBeforeReturning) {
  int execution_count = 0;
  int destruction_count = 0;
  loomc_host_size_t worker_ordinal = 0;
  test_sink_t sink = {
      /*.accepts_tasks=*/true,
      /*.executes_inline=*/true,
      /*.worker_ordinal=*/3,
      /*.pending_task=*/nullptr,
  };

  LOOMC_ASSERT_OK(loomc_task_sink_submit(
      TestTaskSink(&sink),
      AllocateTestTask(&execution_count, &destruction_count, &worker_ordinal)));
  EXPECT_EQ(execution_count, 1);
  EXPECT_EQ(destruction_count, 1);
  EXPECT_EQ(worker_ordinal, 3u);
  EXPECT_EQ(sink.pending_task, nullptr);
}

TEST(TaskTest, InvalidProtocolIsRejectedBeforeEnteringSink) {
  int execution_count = 0;
  int destruction_count = 0;
  loomc_host_size_t worker_ordinal = 0;
  test_sink_t sink = {
      /*.accepts_tasks=*/true,
      /*.executes_inline=*/false,
      /*.worker_ordinal=*/0,
      /*.pending_task=*/nullptr,
  };
  loomc_task_t invalid_task = {};

  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_task_sink_submit(TestTaskSink(&sink), &invalid_task));
  EXPECT_EQ(sink.pending_task, nullptr);
  loomc_task_t* rejected_task =
      AllocateTestTask(&execution_count, &destruction_count, &worker_ordinal);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_task_sink_submit((loomc_task_sink_t){0}, rejected_task));
  rejected_task->next = rejected_task;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_task_sink_submit(TestTaskSink(&sink), rejected_task));
  rejected_task->next = nullptr;
  EXPECT_EQ(execution_count, 0);
  EXPECT_EQ(destruction_count, 0);
  loomc_task_destroy(rejected_task);
  EXPECT_EQ(destruction_count, 1);
}

}  // namespace
