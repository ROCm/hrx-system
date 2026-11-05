// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/workers.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_loop_nest.h"

namespace {

class PipelineCopyOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  void ExpectOrder(const loom_pipeline_transport_t& transport,
                   const loom_pipeline_transport_batch_t& batch,
                   const std::vector<size_t>& expected) {
    loom_pipeline_transport_copy_cursor_t cursor;
    IREE_ASSERT_OK(loom_pipeline_transport_copy_cursor_initialize(
        &transport, &batch, &arena_, &cursor));
    for (size_t index : expected) {
      EXPECT_EQ(loom_pipeline_transport_copy_cursor_next(&cursor),
                &batch.copies[index]);
    }
    EXPECT_EQ(loom_pipeline_transport_copy_cursor_next(&cursor), nullptr);
    EXPECT_EQ(loom_pipeline_transport_copy_cursor_next(&cursor), nullptr);
  }
  // Storage lifetime of the cursor's mutable loop counters.
  iree_arena_block_pool_t blocks_;
  // Independent cursors borrow immutable transport facts from their caller.
  iree_arena_allocator_t arena_;
};

TEST_F(PipelineCopyOrderTest, StraightLinePreservesRevisits) {
  const loom_pipeline_transport_cohort_t records[] = {
      {
          .admission_scope = LOOM_CFG_LOOP_NEST_NONE,
      },
      {
          .admission_scope = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_copy_t copies[] = {
      {
          .source = &records[0],
          .scope = LOOM_CFG_LOOP_NEST_NONE,
      },
      {
          .source = &records[1],
          .scope = LOOM_CFG_LOOP_NEST_NONE,
      },
      {
          .source = &records[0],
          .scope = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_batch_t batch = {
      .records = records,
      .record_count = 2,
      .copies = copies,
      .copy_count = 3,
  };
  ExpectOrder({}, batch, {0, 1, 2});
}

TEST_F(PipelineCopyOrderTest,
       NestedLoopsRepeatBodiesInsteadOfIndividualCopies) {
  const loom_pipeline_transport_cohort_t record = {
      .admission_scope = LOOM_CFG_LOOP_NEST_NONE,
  };
  const loom_pipeline_transport_copy_t copies[] = {
      {
          .source = &record,
          .scope = 1,
      },
      {
          .source = &record,
          .scope = 0,
      },
      {
          .source = &record,
          .scope = 0,
      },
      {
          .source = &record,
          .scope = 1,
      },
      {
          .source = &record,
          .scope = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_loop_t loops[] = {
      {
          .begin = copies + 1,
          .end = copies + 3,
          .iterations = 3,
          .parent = 1,
      },
      {
          .begin = copies,
          .end = copies + 4,
          .iterations = 2,
          .parent = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_t transport = {
      .copy_loops = loops,
      .copy_loop_count = 2,
  };
  const loom_pipeline_transport_batch_t batch = {
      .records = &record,
      .record_count = 1,
      .copies = copies,
      .copy_count = 5,
  };
  const std::vector<size_t> expected = {0, 1, 2, 1, 2, 1, 2, 3, 0,
                                        1, 2, 1, 2, 1, 2, 3, 4};
  ExpectOrder(transport, batch, expected);
  ExpectOrder(transport, batch, expected);
}

TEST_F(PipelineCopyOrderTest, AdmissionScopeIsNotExpandedInsideItsLifetime) {
  const loom_pipeline_transport_cohort_t record = {
      .admission_scope = 1,
  };
  const loom_pipeline_transport_copy_t copies[] = {
      {
          .source = &record,
          .scope = 1,
      },
      {
          .source = &record,
          .scope = 0,
      },
      {
          .source = &record,
          .scope = 0,
      },
      {
          .source = &record,
          .scope = 1,
      },
  };
  const loom_pipeline_transport_loop_t loops[] = {
      {
          .begin = copies + 1,
          .end = copies + 3,
          .iterations = 2,
          .parent = 1,
      },
      {
          .begin = copies,
          .end = copies + 4,
          .iterations = 1024,
          .parent = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_t transport = {
      .copy_loops = loops,
      .copy_loop_count = 2,
  };
  const loom_pipeline_transport_batch_t batch = {
      .records = &record,
      .record_count = 1,
      .copies = copies,
      .copy_count = 4,
  };
  ExpectOrder(transport, batch, {0, 1, 2, 1, 2, 3});
}

TEST_F(PipelineCopyOrderTest, SiblingLoopsKeepIndependentCounts) {
  const loom_pipeline_transport_cohort_t record = {
      .admission_scope = LOOM_CFG_LOOP_NEST_NONE,
  };
  const loom_pipeline_transport_copy_t copies[] = {
      {
          .source = &record,
          .scope = 0,
      },
      {
          .source = &record,
          .scope = 1,
      },
      {
          .source = &record,
          .scope = 1,
      },
  };
  const loom_pipeline_transport_loop_t loops[] = {
      {
          .begin = copies,
          .end = copies + 1,
          .iterations = 2,
          .parent = LOOM_CFG_LOOP_NEST_NONE,
      },
      {
          .begin = copies + 1,
          .end = copies + 3,
          .iterations = 3,
          .parent = LOOM_CFG_LOOP_NEST_NONE,
      },
  };
  const loom_pipeline_transport_t transport = {
      .copy_loops = loops,
      .copy_loop_count = 2,
  };
  const loom_pipeline_transport_batch_t batch = {
      .records = &record,
      .record_count = 1,
      .copies = copies,
      .copy_count = 3,
  };
  ExpectOrder(transport, batch, {0, 0, 1, 2, 1, 2, 1, 2});
}

}  // namespace
