// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame_test_fixture.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LowEmissionStateFrameTest : public ::loom::testing::LowEmissionFrameTest {
 protected:
  void ExpectStatePredecessors(
      const loom_low_emission_frame_t& frame, uint32_t consumer_node,
      std::initializer_list<uint32_t> expected_producer_nodes) {
    std::vector<uint32_t> actual_producer_nodes;
    for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies,
                                                i);
      if (dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_STATE &&
          dependency->consumer_node == consumer_node) {
        actual_producer_nodes.push_back(dependency->producer_node);
      }
    }
    std::sort(actual_producer_nodes.begin(), actual_producer_nodes.end());
    actual_producer_nodes.erase(
        std::unique(actual_producer_nodes.begin(), actual_producer_nodes.end()),
        actual_producer_nodes.end());
    std::vector<uint32_t> sorted_expected_producer_nodes(
        expected_producer_nodes);
    std::sort(sorted_expected_producer_nodes.begin(),
              sorted_expected_producer_nodes.end());
    EXPECT_EQ(actual_producer_nodes, sorted_expected_producer_nodes);
  }

  void ExpectNoStateDependencies(const loom_low_emission_frame_t& frame) {
    for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies,
                                                i);
      EXPECT_NE(dependency->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE);
    }
  }
};

TEST_F(LowEmissionStateFrameTest, CommutativeStateAccessesRemainIndependent) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @ready_update(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %slow = test.event.slow.i32 %lhs, %rhs
  %first = test.state.update.i32 %slow
  %ready = test.state.update.i32 %lhs
  return %first, %ready
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectNoStateDependencies(frame);
    EXPECT_LT(frame.schedule.nodes[2].scheduled_ordinal,
              frame.schedule.nodes[1].scheduled_ordinal);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @independent_assignment(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.schedule_state.update.i32 %value
  test.state.assign.i32.immediate 12
  %second = test.schedule_state.update.i32 %value
  return %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectNoStateDependencies(frame);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @read_only(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.read.i32 %value
  %second = test.state.read.i32 %value
  return %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectNoStateDependencies(frame);
  }
}

TEST_F(LowEmissionStateFrameTest, ReadersAndWritersFormMinimalStateChains) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @reader_before(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %observed = test.state.read.i32 %value
  %first = test.state.update.i32 %value
  %second = test.state.update.i32 %value
  return %observed, %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @reader_between(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  %observed = test.state.read.i32 %value
  %second = test.state.update.i32 %value
  return %observed, %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {0, 1});
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @reader_after(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  %second = test.state.update.i32 %value
  %observed = test.state.read.i32 %value
  return %observed, %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @replacement(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  test.state.assign.i32.immediate 0
  %second = test.state.update.i32 %value
  return %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  }
}

TEST_F(LowEmissionStateFrameTest, ExplicitStateValuesPrecedeClobbers) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @structural_operand(%state: reg<test.special>, %value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %snapshot = copy %state : reg<test.special> -> reg<test.i32>
  %first = test.state.update.i32 %value
  %second = test.state.update.i32 %value
  return %snapshot, %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @state_result(%state_lhs: reg<test.schedule_state>, %state_rhs: reg<test.schedule_state>, %value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %snapshot = test.add.schedule_state %state_lhs, %state_rhs
  %first = test.schedule_state.update.i32 %value
  %second = test.schedule_state.update.i32 %value
  return %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @local_snapshot(%state_lhs: reg<test.schedule_state>, %state_rhs: reg<test.schedule_state>) asm {
  %snapshot = test.add.schedule_state %state_lhs, %state_rhs
  %first = test.state.add.schedule_state %state_lhs, %state_rhs
  %second = test.state.add.schedule_state %state_lhs, %state_rhs
  %read = test.explicit.state.add.schedule_state %state_lhs, %snapshot
  return
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                              LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {3});
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
    EXPECT_LT(frame.schedule.nodes[3].scheduled_ordinal,
              frame.schedule.nodes[1].scheduled_ordinal);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @live_in_snapshot(%state: reg<test.schedule_state>, %rhs: reg<test.schedule_state>) asm {
  %first = test.state.add.schedule_state %state, %rhs
  %second = test.state.add.schedule_state %state, %rhs
  %read = test.explicit.state.add.schedule_state %rhs, %state
  return
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                              LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));
    ExpectStatePredecessors(frame, /*consumer_node=*/0, {2});
    ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
    EXPECT_LT(frame.schedule.nodes[2].scheduled_ordinal,
              frame.schedule.nodes[0].scheduled_ordinal);
  }
}

TEST_F(LowEmissionStateFrameTest, OpaqueOperationsFenceStateUpdates) {
  const auto expect_fence = [this](const char* source) {
    ModulePtr module = ParseModule(source);
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {0});
  };

  expect_fence(R"(
low.func.def target<test.low.core> @opaque_descriptor(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  %effect = test.projectable_effect.i32 %value
  %second = test.state.update.i32 %value
  return %first, %effect, %second
}
)");
  expect_fence(R"(
low.func.def target<test.low.core> @convergent_boundary(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  %participating = test.convergent.i32 %value
  %second = test.state.update.i32 %value
  return %first, %participating, %second
}
)");
  expect_fence(R"(
low.func.def target<test.low.core> @nested_region_boundary(%condition: reg<test.i32>, %value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  low.scf.if %condition {
    test.state.assign.i32.immediate 0
  } else {
    test.state.assign.i32.immediate 1
  }
  %second = test.state.update.i32 %value
  return %first, %second
}
)");
}

TEST_F(LowEmissionStateFrameTest, CallsFenceStateUpdatesRegardlessOfPurity) {
  const auto expect_call_fence = [this](const char* source) {
    ModulePtr module = ParseModule(source);
    loom_block_t* module_block = loom_module_block(module.get());
    ASSERT_EQ(module_block->op_count, 2u);
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFunctionFrame(
        module.get(), loom_block_op(module_block, 1), {}, &frame));
    ExpectStatePredecessors(frame, /*consumer_node=*/2, {0});
  };

  expect_call_fence(R"(
low.func.decl pure target<test.low.core> @helper()
low.func.def target<test.low.core> @pure_call(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  low.func.call pure @helper() : ()
  %second = test.state.update.i32 %value
  return %first, %second
}
)");
  expect_call_fence(R"(
low.func.decl target<test.low.core> @helper()
low.func.def target<test.low.core> @unknown_call(%value: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.state.update.i32 %value
  low.func.call @helper() : ()
  %second = test.state.update.i32 %value
  return %first, %second
}
)");
}

TEST_F(LowEmissionStateFrameTest,
       ScheduleBoundariesDoNotInventStateDependencies) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fenced_updates(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %slow = test.event.slow.i32 %lhs, %rhs
  %first = test.state.update.i32 %slow
  low.schedule.fence
  %ready = test.state.update.i32 %lhs
  return %first, %ready
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectNoStateDependencies(frame);
    ASSERT_EQ(frame.schedule.node_count, 5u);
    for (uint32_t i = 0; i < frame.schedule.node_count; ++i) {
      EXPECT_EQ(frame.schedule.nodes[i].scheduled_ordinal, i);
    }
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[2].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[4].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def schedule(locked) target<test.low.core> @locked_updates(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %slow = test.event.slow.i32 %lhs, %rhs
  %first = test.state.update.i32 %slow
  %ready = test.state.update.i32 %lhs
  return %first, %ready
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ExpectNoStateDependencies(frame);
    ASSERT_EQ(frame.schedule.node_count, 4u);
    for (uint32_t i = 0; i < frame.schedule.node_count; ++i) {
      EXPECT_EQ(frame.schedule.nodes[i].scheduled_ordinal, i);
      EXPECT_TRUE(
          iree_any_bit_set(frame.schedule.nodes[i].flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    }
  }
}

TEST_F(LowEmissionStateFrameTest, StateObservationsAreBlockLocal) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @block_local(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %observed = test.state.read.i32 %lhs
  %before0 = test.state.update.i32 %lhs
  %before1 = test.state.update.i32 %rhs
  low.br ^next
^next:
  %slow = test.event.slow.i32 %lhs, %rhs
  %after0 = test.state.update.i32 %slow
  %after1 = test.state.update.i32 %lhs
  return %observed, %before0, %before1, %after0, %after1
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStatePredecessors(frame, /*consumer_node=*/2, {1});
  ExpectStatePredecessors(frame, /*consumer_node=*/6, {});
  EXPECT_LT(frame.schedule.nodes[6].scheduled_ordinal,
            frame.schedule.nodes[5].scheduled_ordinal);
}

TEST_F(LowEmissionStateFrameTest, ConvergentStateReadersRetainOrder) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @convergent_fanout(%state_lhs: reg<test.schedule_state>, %state_rhs: reg<test.schedule_state>, %value0: reg<test.i32>, %value1: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %state = test.state.add.schedule_state %state_lhs, %state_rhs
  %read0 = test.convergent.explicit.state.read.i32 %value0, %state
  %read1 = test.convergent.explicit.state.read.i32 %value1, %state
  return %read0, %read1
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStatePredecessors(frame, /*consumer_node=*/1, {0});
  ExpectStatePredecessors(frame, /*consumer_node=*/2, {0, 1});
  EXPECT_LT(frame.schedule.nodes[1].scheduled_ordinal,
            frame.schedule.nodes[2].scheduled_ordinal);
  ExpectDependencyTiming(frame, 1, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE, 0,
                         LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                         LOOM_LOW_MODEL_QUALITY_EXACT);
}

}  // namespace
}  // namespace loom
