// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/sgpr_read_wait.h"

#include <random>
#include <unordered_set>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

using Event = loom_amdgpu_sgpr_read_wait_event_t;
using Program = std::vector<std::vector<Event>>;
using Edges = std::vector<std::vector<uint16_t>>;

Event Instruction(uint8_t kind, std::initializer_list<unsigned> reads,
                  std::initializer_list<unsigned> writes = {}) {
  Event event = {.kind = kind};
  for (unsigned index : reads) {
    event.reads[index / 64] |= UINT64_C(1) << (index % 64);
  }
  for (unsigned index : writes) {
    event.writes[index / 64] |= UINT64_C(1) << (index % 64);
  }
  return event;
}

Event Scalar(std::initializer_list<unsigned> reads,
             std::initializer_list<unsigned> writes = {}) {
  return Instruction(LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR, reads, writes);
}

Event Vector(std::initializer_list<unsigned> reads,
             std::initializer_list<unsigned> writes = {}) {
  return Instruction(LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR, reads, writes);
}

Event Drain(uint8_t fields = 7) {
  return Event{
      .kind = LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN,
      .drain_fields = fields,
  };
}

class AmdgpuSgprReadWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Plan(const Program& program, const Edges& edges,
            bool compact_single_block = false) {
    iree_arena_reset(&arena_);
    waits_.clear();
    testing::CfgGraph fixture(edges);
    std::vector<loom_low_schedule_node_t> nodes;
    std::vector<uint32_t> indices;
    for (size_t block = 0; block < program.size(); ++block) {
      for (size_t index = 0; index < program[block].size(); ++index) {
        loom_low_schedule_node_t node = {};
        node.block_index = block;
        node.scheduled_ordinal = index;
        indices.push_back(nodes.size());
        nodes.push_back(node);
      }
    }
    loom_low_schedule_table_t schedule = {.cfg_graph = *fixture.get()};
    if (compact_single_block) {
      // The function model omits graph storage for an edge-free single block.
      schedule.cfg_graph = {};
      schedule.cfg_graph.block_count = 1;
    }
    schedule.block_count = program.size();
    schedule.nodes = nodes.data();
    schedule.node_count = nodes.size();
    schedule.scheduled_node_indices = indices.data();
    schedule.scheduled_node_count = indices.size();
    loom_amdgpu_sgpr_read_wait_t* state = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_sgpr_read_wait_allocate(&schedule, &arena_, &state));
    uint32_t packet_index = 0;
    for (uint16_t block = 0; block < program.size(); ++block) {
      for (Event event : program[block]) {
        event.packet_index = packet_index++;
        IREE_ASSERT_OK(
            loom_amdgpu_sgpr_read_wait_collect(state, block, &event));
      }
    }
    IREE_ASSERT_OK(loom_amdgpu_sgpr_read_wait_resolve(
        state,
        [](void* user_data, const loom_amdgpu_wait_state_t* wait) {
          static_cast<AmdgpuSgprReadWaitTest*>(user_data)->waits_.push_back(
              *wait);
          return iree_ok_status();
        },
        this));
  }

  // Every state in this oracle belongs to one concrete path, not a join of
  // predecessor facts. The small domain permits exploring loops to closure.
  void CheckConcretePaths(const Program& program, const Edges& edges) {
    Plan(program, edges);
    ASSERT_FALSE(HasFatalFailure());
    std::vector<std::vector<uint8_t>> drains(program.size());
    for (size_t block = 0; block < program.size(); ++block) {
      drains[block].resize(program[block].size());
    }
    for (const auto& wait : waits_) {
      const uint16_t immediate = wait.immediate;
      drains[wait.block_index][wait.scheduled_ordinal] =
          ((immediate & 1) == 0 ? 1 : 0) | ((immediate & 0x0e00) == 0 ? 2 : 0) |
          ((immediate & 2) == 0 ? 4 : 0);
    }
    // Packed states: block, tracked pairs, SALU writes, VALU writes. Register
    // bits 0:1 name s0:s1; bits 2:3 name vcc_lo:vcc_hi.
    std::vector<uint32_t> worklist{0};
    std::unordered_set<uint32_t> visited{0};
    for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const uint32_t state = worklist[cursor];
      const unsigned block = state & 255;
      unsigned tracked = (state >> 8) & 15;
      unsigned scalar = (state >> 12) & 15;
      unsigned vector = (state >> 16) & 15;
      for (size_t index = 0; index < program[block].size(); ++index) {
        const Event& event = program[block][index];
        const unsigned fields = drains[block][index] | event.drain_fields;
        if (fields & 1) {
          scalar = 0;
        }
        if (fields & 2) {
          vector &= 12;
        }
        if (fields & 4) {
          vector &= 3;
        }
        const unsigned reads =
            (event.reads[0] & 3) | (((event.reads[1] >> 42) & 3) << 2);
        const unsigned writes =
            (event.writes[0] & 3) | (((event.writes[1] >> 42) & 3) << 2);
        ASSERT_EQ(reads & scalar, 0u)
            << "block=" << block << " index=" << index;
        if (event.kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR) {
          ASSERT_EQ(reads & vector, 0u)
              << "block=" << block << " index=" << index;
          for (unsigned bit = 0; bit < 4; ++bit) {
            if (reads & (1u << bit)) {
              tracked |= 3u << (bit & ~1u);
            }
          }
          vector |= writes & tracked;
        } else {
          vector &= ~reads;
          scalar |= writes & tracked;
        }
      }
      for (unsigned successor : edges[block]) {
        const uint32_t next =
            successor | (tracked << 8) | (scalar << 12) | (vector << 16);
        if (visited.insert(next).second) {
          worklist.push_back(next);
        }
      }
    }
  }

  // Recycled compiler-invocation storage.
  iree_arena_block_pool_t pool_;
  // Analysis lifetime, reset between independent programs.
  iree_arena_allocator_t arena_;
  // Actual emitted dependency actions, in final native order.
  std::vector<loom_amdgpu_wait_state_t> waits_;
};

TEST_F(AmdgpuSgprReadWaitTest, PhysicalPairAndProducerConsumerMatrix) {
  for (unsigned physical = 0; physical < 124; ++physical) {
    for (bool vector_write : {false, true}) {
      for (bool vector_read : {false, true}) {
        SCOPED_TRACE(physical);
        const uint8_t writer = vector_write ? LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR
                                            : LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR;
        const uint8_t reader = vector_read ? LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR
                                           : LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR;
        Plan({{Vector({physical ^ 1}), Instruction(writer, {}, {physical}),
               Instruction(reader, {physical})}},
             {{}}, /*compact_single_block=*/true);
        if (vector_write && !vector_read) {
          EXPECT_TRUE(waits_.empty());
        } else {
          ASSERT_EQ(waits_.size(), 1u);
          const uint16_t expected = !vector_write ? 0xfffe
                                    : physical == 106 || physical == 107
                                        ? 0xfffd
                                        : 0xf1ff;
          EXPECT_EQ(waits_[0].immediate, expected);
          EXPECT_EQ(waits_[0].scheduled_ordinal, 2u);
        }
      }
    }
  }
}

TEST_F(AmdgpuSgprReadWaitTest, HistoryIsNotRetroactiveOrSharedBetweenPairs) {
  Plan({{Scalar({}, {0}), Vector({0}), Scalar({}, {2}), Vector({2})}}, {{}});
  EXPECT_TRUE(waits_.empty());
}

TEST_F(AmdgpuSgprReadWaitTest, DrainPreservesHistoryAndSelectsOnlyItsField) {
  Plan({{Vector({0, 106}), Scalar({}, {0}), Vector({}, {106}), Drain(1),
         Vector({0}), Vector({106}), Scalar({}, {1}), Scalar({1})}},
       {{}});
  ASSERT_EQ(waits_.size(), 2u);
  EXPECT_EQ(waits_[0].scheduled_ordinal, 5u);
  EXPECT_EQ(waits_[0].immediate, 0xfffd);
  EXPECT_EQ(waits_[1].scheduled_ordinal, 7u);
  EXPECT_EQ(waits_[1].immediate, 0xfffe);
}

TEST_F(AmdgpuSgprReadWaitTest, DifferentWritersRetainBothDependencies) {
  Plan({{Vector({0}), Scalar({}, {0}), Vector({}, {0}), Vector({0})}}, {{}});
  ASSERT_EQ(waits_.size(), 1u);
  EXPECT_EQ(waits_[0].immediate, 0xf1fe);
}

TEST_F(AmdgpuSgprReadWaitTest, PropagatesHistoryAndWritesAcrossEmptyBlocks) {
  Plan({{Vector({1})}, {}, {Scalar({}, {0})}, {}, {Scalar({0})}},
       {{1}, {2}, {3}, {4}, {}});
  ASSERT_EQ(waits_.size(), 1u);
  EXPECT_EQ(waits_[0].block_index, 4u);
}

TEST_F(AmdgpuSgprReadWaitTest, ResolvedPredecessorDrainAvoidsDuplicateWait) {
  Plan({{Vector({0, 2}), Scalar({}, {0, 2}), Scalar({0})}, {Scalar({2})}},
       {{1}, {}});
  ASSERT_EQ(waits_.size(), 1u);
  EXPECT_EQ(waits_[0].block_index, 0u);
}

TEST_F(AmdgpuSgprReadWaitTest, UnreachablePredecessorDoesNotIntroduceHistory) {
  Plan({{}, {Scalar({}, {0}), Scalar({0})}, {Vector({0})}}, {{1}, {}, {1}});
  EXPECT_TRUE(waits_.empty());
}

TEST_F(AmdgpuSgprReadWaitTest, GeneratedPlansProtectEveryConcretePath) {
  std::mt19937 random(20261019);
  constexpr unsigned kPhysical[] = {0, 1, 106, 107};
  for (unsigned trial = 0; trial < 256; ++trial) {
    SCOPED_TRACE(trial);
    const unsigned block_count = 1 + random() % 6;
    Program program(block_count);
    Edges edges(block_count);
    for (unsigned block = 0; block < block_count; ++block) {
      for (unsigned count = random() % 3; count != 0; --count) {
        edges[block].push_back(random() % block_count);
      }
      for (unsigned count = 1 + random() % 6; count != 0; --count) {
        Event event = {};
        event.kind = 1 + random() % 3;
        if (event.kind == LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN) {
          event.drain_fields = 1 + random() % 7;
        } else {
          const unsigned reads = random() % 16;
          const unsigned writes = random() % 16;
          for (unsigned bit = 0; bit < 4; ++bit) {
            const unsigned physical = kPhysical[bit];
            if (reads & (1u << bit)) {
              event.reads[physical / 64] |= UINT64_C(1) << (physical % 64);
            }
            if (writes & (1u << bit)) {
              event.writes[physical / 64] |= UINT64_C(1) << (physical % 64);
            }
          }
        }
        program[block].push_back(event);
      }
    }
    CheckConcretePaths(program, edges);
    ASSERT_FALSE(HasFatalFailure());
  }
}

TEST_F(AmdgpuSgprReadWaitTest, LargeCollectionUsesPooledSegments) {
  Program program(1);
  for (unsigned i = 0; i < 32768; ++i) {
    program[0].push_back(Vector({0}));
  }
  Plan(program, {{}}, /*compact_single_block=*/true);
  EXPECT_TRUE(waits_.empty());
  EXPECT_EQ(arena_.allocation_head, nullptr);
  EXPECT_LT(arena_.used_allocation_size, 2u * 1024 * 1024);
}

}  // namespace
}  // namespace loom
