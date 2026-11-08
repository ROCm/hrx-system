// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/setup_order.h"

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/context.h"

namespace loom {
namespace {

TEST(SetupMembershipTest, StructuralSetupRequiresPolicyOrRetainedRegister) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_value_id_t value_ids[] = {7, 19};
  loom_local_value_domain_t domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
  };
  const struct {
    // Allocation retained private placement for the result.
    bool retained;
    // Allocation requires the result to remain in a register.
    bool required;
    // The caller requested ordering for all structural setup.
    bool order_structural;
    // Whether this node belongs to an ordered setup chain.
    bool ordered;
  } cases[] = {
      {false, false, false, false}, {true, false, false, false},
      {false, true, false, false},  {true, true, false, true},
      {false, false, true, true},   {true, false, true, true},
      {false, true, true, true},    {true, true, true, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "retained=" << test_case.retained
                 << " required=" << test_case.required
                 << " structural=" << test_case.order_structural);
    uint64_t placement_word = 0;
    uint64_t required_word = 0;
    loom_low_schedule_options_t options = {
        .required_register_values = {64, &required_word},
        .per_user_placement_values = {64, &placement_word},
    };
    if (test_case.retained) {
      iree_bitmap_set(options.per_user_placement_values, value_ids[1]);
    }
    if (test_case.required) {
      iree_bitmap_set(options.required_register_values, value_ids[1]);
    }
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = 1, .register_class_id = 0},
        {.unit_count = 1, .register_class_id = 0},
    };
    if (test_case.required) {
      values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .value_domain = &domain,
        .values = values,
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count = 1,
        .result_count = 1,
        .flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    state.nodes = &node;
    loom_low_schedule_setup_order_classify_node(&state, 0,
                                                test_case.order_structural);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

TEST(SetupMembershipTest, SingletonRequiredStorageNeedsExclusionOrdering) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const loom_low_schedule_options_t options = {};
  const struct {
    // Available physical units in the completion domain, or zero if unbounded.
    uint32_t capacity;
    // Physical units owned by the result.
    uint32_t result_units;
    // The result cannot spill or rematerialize.
    bool required;
    // The structural operation constructs storage for its result.
    bool storage_setup;
    // Whether the singleton storage requires exclusion ordering.
    bool ordered;
  } cases[] = {
      {1, 1, true, true, true},   {1, 1, false, true, false},
      {1, 1, true, false, false}, {2, 1, true, true, false},
      {2, 2, true, true, false},  {0, 1, true, true, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "capacity=" << test_case.capacity << " units="
                 << test_case.result_units << " required=" << test_case.required
                 << " storage setup=" << test_case.storage_setup);
    loom_low_schedule_completion_domain_t completion = {
        .capacity = test_case.capacity,
        .reg_class_id = 0,
    };
    uint16_t domain_id = test_case.capacity == 0 ? UINT16_MAX : 0;
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = 1, .register_class_id = 0},
        {.unit_count = test_case.result_units, .register_class_id = 0},
    };
    if (test_case.required) {
      values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .values = values,
        .pressure_limits =
            {
                .constrained_completion_domains = &completion,
                .constrained_completion_domain_ids_by_reg_class = &domain_id,
                .constrained_completion_domain_count =
                    test_case.capacity == 0 ? uint16_t{0} : uint16_t{1},
            },
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count = 1,
        .result_count = 1,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    if (test_case.storage_setup) {
      node.flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP;
    }
    state.nodes = &node;
    loom_low_schedule_setup_order_classify_node(
        &state, 0, /*order_structural_setup=*/false);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

TEST(SetupMembershipTest, CapturedRequiredStorageMustPreserveItsFootprint) {
  const loom_low_reg_class_t classes[] = {
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 1},
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 1},
      {.alloc_unit_bits = 32, .allocatable_count = 8},
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 2},
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  const loom_low_schedule_options_t options = {};
  const struct {
    // Human-readable invariant represented by this row.
    const char* name;
    // Units captured from the first operand in class zero.
    uint32_t first_units;
    // Units captured from a second operand, or zero for a unary conversion.
    uint32_t second_units;
    // Units owned by the result.
    uint32_t result_units;
    // Result class in the same bank, a shared bank, or a disjoint bank.
    uint16_t result_class;
    // Whether the captured operands require physical registers.
    bool inputs_required;
    // Whether the result requires physical registers.
    bool result_required;
    // Whether retaining the inputs preserves required physical storage.
    bool ordered;
  } cases[] = {
      {"same class", 4, 0, 4, 0, true, true, true},
      {"shared bank", 1, 0, 1, 1, true, true, true},
      {"different bank", 1, 0, 1, 2, true, true, false},
      {"different shared bank", 1, 0, 1, 3, true, true, false},
      {"unary contraction", 4, 0, 1, 0, true, true, false},
      {"combined contraction", 4, 4, 4, 0, true, true, false},
      {"expansion", 1, 0, 4, 0, true, true, true},
      {"composition", 1, 1, 2, 0, true, true, true},
      {"spillable result", 4, 0, 4, 0, true, false, false},
      {"spillable inputs", 4, 0, 1, 2, false, true, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = test_case.first_units, .register_class_id = 0},
        {.unit_count = test_case.second_units, .register_class_id = 0},
        {
            .unit_count = test_case.result_units,
            .register_class_id = test_case.result_class,
        },
    };
    if (test_case.inputs_required) {
      values[0].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
      if (test_case.second_units != 0) {
        values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
      }
    }
    if (test_case.result_required) {
      values[2].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .values = values,
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count =
            test_case.second_units == 0 ? uint16_t{1} : uint16_t{2},
        .result_count = 1,
        .flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    loom_low_schedule_node_result_ordinals(&node)[0] = 2;
    state.nodes = &node;
    loom_low_schedule_setup_order_classify_node(
        &state, 0, /*order_structural_setup=*/true);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

TEST(SetupMembershipTest, DescriptorMembershipPreservesRequiredReadPhases) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const struct {
    // Retained packet read/write phase information.
    loom_low_schedule_node_flags_t flags;
    // Physical units held by the captured input.
    uint32_t input_units;
    // Whether the input requires physical storage.
    bool input_required;
    // Result arity of the descriptor packet.
    uint8_t result_count;
    // Whether setup ordering preserves required storage.
    bool ordered;
  } cases[] = {
      {0, 1, true, 0, false},
      {0, 1, true, 2, false},
      {0, 1, true, 1, true},
      {0, 4, true, 1, false},
      {LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER, 1, true, 1, false},
      {LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS, 1, true, 1, false},
      {LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER, 1, false, 1, true},
      {LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS, 1, false, 1, true},
  };
  loom_value_id_t value_ids[] = {7, 19, 29};
  loom_local_value_domain_t domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "flags=" << test_case.flags
                 << " units=" << test_case.input_units
                 << " required=" << test_case.input_required
                 << " results=" << test_case.result_count);
    loom_low_descriptor_t descriptor = {
        .operand_count = static_cast<uint8_t>(test_case.result_count + 1),
        .result_count = test_case.result_count,
    };
    if (test_case.flags & LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER) {
      descriptor.flags |= LOOM_LOW_DESCRIPTOR_FLAG_EARLY_CLOBBER;
    }
    if (test_case.flags & LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS) {
      descriptor.flags |= LOOM_LOW_DESCRIPTOR_FLAG_LATE_READ;
    }
    uint64_t placement_word = 0;
    uint64_t required_word = 0;
    loom_low_schedule_options_t options = {
        .required_register_values = {64, &required_word},
        .per_user_placement_values = {64, &placement_word},
    };
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = test_case.input_units, .register_class_id = 0},
        {.unit_count = 1, .register_class_id = 0},
        {.unit_count = 1, .register_class_id = 0},
    };
    if (test_case.input_required) {
      iree_bitmap_set(options.required_register_values, value_ids[0]);
      values[0].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
    }
    for (uint16_t i = 1; i <= test_case.result_count; ++i) {
      iree_bitmap_set(options.per_user_placement_values, value_ids[i]);
      iree_bitmap_set(options.required_register_values, value_ids[i]);
      values[i].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
    }
    loom_low_schedule_node_t node = {
        .descriptor = &descriptor,
        .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
        .operand_count = 1,
        .result_count = test_case.result_count,
        .flags = test_case.flags,
        .value_ordinals = {.inline_value_ordinals = {0, 1, 2}},
    };
    if (test_case.flags & LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS) {
      node.value_ordinals.inline_value_ordinals[2] = 1;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .value_domain = &domain,
        .values = values,
        .nodes = &node,
    };
    loom_low_schedule_setup_order_classify_node(
        &state, 0, /*order_structural_setup=*/false);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

class SetupOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Initialize(std::initializer_list<uint32_t> block_sizes) {
    nodes_.clear();
    blocks_.clear();
    for (uint32_t size : block_sizes) {
      const uint32_t start = static_cast<uint32_t>(nodes_.size());
      nodes_.resize(start + size);
      for (uint32_t i = start; i < nodes_.size(); ++i) {
        nodes_[i].block_index = static_cast<uint32_t>(blocks_.size());
      }
      blocks_.push_back({.node_start = start, .node_count = size});
    }
    options_ = {};
    state_ = {
        .options = &options_,
        .scratch_arena = &arena_,
        .blocks = blocks_.data(),
        .nodes = nodes_.data(),
        .dependency_arena = &arena_,
    };
    loom_low_schedule_dependency_graph_initialize(&state_.dependencies);
    IREE_ASSERT_OK(loom_low_schedule_setup_order_initialize(
        static_cast<uint32_t>(nodes_.size()), &arena_, &state_.setup_order));
  }

  void Append(uint32_t producer, uint32_t consumer,
              loom_low_schedule_dependency_kind_t kind =
                  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA,
              int32_t minimum_issue_separation_cycles = 0) {
    const loom_low_schedule_dependency_t edge = {
        .producer_node = producer,
        .consumer_node = consumer,
        .minimum_issue_separation_cycles = minimum_issue_separation_cycles,
        .producer_attachment_index = LOOM_LOW_ID_NONE,
        .consumer_attachment_index = LOOM_LOW_ID_NONE,
        .producer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
        .consumer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
        .value_operand_index = LOOM_LOW_ID_NONE,
        .kind = kind,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &state_.dependencies, edge, &arena_));
    loom_low_schedule_setup_order_record_dependency(
        &state_.setup_order, producer, consumer,
        kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA);
  }

  void ExpectAddedOrder(
      std::initializer_list<std::pair<uint32_t, uint32_t>> expected) {
    const auto original_count = state_.dependencies.count;
    IREE_ASSERT_OK(loom_low_schedule_setup_order_finish(
        &state_, static_cast<uint32_t>(nodes_.size()),
        static_cast<uint16_t>(blocks_.size())));
    std::vector<std::pair<uint32_t, uint32_t>> actual;
    for (uint32_t i = original_count; i < state_.dependencies.count; ++i) {
      const auto* edge =
          loom_low_schedule_dependency_graph_at(&state_.dependencies, i);
      EXPECT_EQ(edge->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
      EXPECT_EQ(edge->minimum_issue_separation_cycles, 0);
      EXPECT_EQ(edge->separation_source,
                LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_STRUCTURAL);
      EXPECT_EQ(edge->producer_event_id, LOOM_LOW_TIMING_EVENT_NONE);
      EXPECT_EQ(edge->consumer_event_id, LOOM_LOW_TIMING_EVENT_NONE);
      actual.emplace_back(edge->producer_node, edge->consumer_node);
    }
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, (std::vector<std::pair<uint32_t, uint32_t>>(expected)));
  }

  // Backing storage for the native graph and temporary closure proof.
  iree_arena_block_pool_t pool_;
  // Graph construction and closure scratch share the invocation lifetime.
  iree_arena_allocator_t arena_;
  // Native source-order node metadata; no operations or module are needed.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Disjoint source-order ranges covering every node exactly once.
  std::vector<loom_low_schedule_block_t> blocks_;
  // Production policy controlling implied-successor repair.
  loom_low_schedule_options_t options_ = {};
  // Native closure inputs and outputs, borrowing the arrays above.
  loom_low_schedule_build_state_t state_ = {};
};

TEST_F(SetupOrderTest, PrivatePlacementRetainsOnlyItsNearestLocalCall) {
  for (bool chain_scratch : {false, true}) {
    for (bool private_placement : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "chain scratch=" << chain_scratch
                                        << " private=" << private_placement);
      Initialize({8, 3, 1});
      if (!chain_scratch) {
        state_.setup_order = {};
      }
      loom_value_id_t value_ids[14];
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
        value_ids[i] = i + 7;
      }
      const loom_local_value_domain_t domain = {
          .value_ids = value_ids,
          .value_count = IREE_ARRAYSIZE(value_ids),
      };
      state_.value_domain = &domain;
      for (uint32_t i = 0; i < nodes_.size(); ++i) {
        nodes_[i].result_count = 1;
        nodes_[i].value_ordinals.inline_value_ordinals[0] = i;
      }
      uint32_t calls[] = {1, 4, 5, 9};
      state_.call_node_indices = calls;
      state_.call_node_count = IREE_ARRAYSIZE(calls);
      nodes_[4].result_count = 0;
      nodes_[9].result_count = 2;
      nodes_[9].value_ordinals.inline_value_ordinals[1] = 12;
      // Setup can capture inputs or be input-free. A private result from a
      // multi-result producer does not establish this single-value boundary.
      nodes_[6].operand_count = 2;
      nodes_[6].value_ordinals = {.inline_value_ordinals = {0, 2, 6}};
      nodes_[7].result_count = 2;
      nodes_[7].value_ordinals.inline_value_ordinals[1] = 13;
      uint64_t placements = 0;
      if (private_placement) {
        options_.per_user_placement_values = {64, &placements};
        for (uint32_t ordinal : {0u, 2u, 6u, 7u, 8u, 10u, 11u}) {
          iree_bitmap_set(options_.per_user_placement_values,
                          value_ids[ordinal]);
        }
      }

      if (private_placement) {
        ExpectAddedOrder({{1, 2}, {5, 6}, {9, 10}});
      } else {
        ExpectAddedOrder({});
      }
    }
  }
}

TEST_F(SetupOrderTest, ContinuationsConsumeRetainedStorageIdentity) {
  enum SourceKind {
    kComputed,
    kCrossBank,
    kConstant,
    kObservable,
    kNotRematerializable,
    kOtherBlock,
  };
  constexpr uint32_t kSource = 1u;
  constexpr uint32_t kResult = 2u;
  const struct {
    // Producer behavior whose native facts were retained during graph building.
    SourceKind source_kind;
    // Whether the partial value must retain physical storage.
    bool source_required;
    // Whether the completing packet belongs to a rematerialization recipe.
    bool result_rematerializable;
    // Nodes with allocation-owned private placement, indexed by node ordinal.
    uint32_t retained;
    // Expected setup members, indexed by node ordinal.
    uint32_t members;
  } cases[] = {
      {kComputed, true, false, 0, 0},
      {kComputed, true, false, kSource, kSource | kResult},
      {kComputed, false, false, kSource, kSource | kResult},
      {kComputed, true, false, kResult, kSource | kResult},
      {kComputed, false, false, kResult, kResult},
      {kCrossBank, true, false, 0, 0},
      {kCrossBank, true, false, kSource, kSource | kResult},
      {kCrossBank, true, false, kResult, kResult},
      {kNotRematerializable, true, false, kResult, kResult},
      {kObservable, true, false, kResult, kResult},
      {kOtherBlock, true, false, kResult, kResult},
      {kConstant, false, true, kSource, 0},
      {kConstant, true, true, kSource, kSource},
      {kConstant, true, true, kResult, kResult},
      {kConstant, true, true, kSource | kResult, kSource | kResult},
      {kConstant, true, false, kSource, kSource | kResult},
  };
  const loom_low_reg_class_t reg_classes[] = {
      {.alloc_unit_bits = 32, .allocatable_count = 8},
      {.alloc_unit_bits = 256, .allocatable_count = 8},
  };
  loom_segmented_storage_t relations;
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_storage_relation_t),
      alignof(loom_low_schedule_storage_relation_t), &relations);
  loom_low_schedule_storage_relation_t* relation = nullptr;
  IREE_ASSERT_OK(loom_segmented_storage_append(
      &relations, &arena_, reinterpret_cast<void**>(&relation)));
  uint32_t relation_starts[] = {0, 0, 1};
  loom_value_id_t value_ids[] = {5, 11, 19, 29};
  loom_local_value_domain_t domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
  };
  uint32_t producers[] = {LOOM_LOW_SCHEDULE_NODE_NONE,
                          LOOM_LOW_SCHEDULE_NODE_NONE, 0, 1};
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "source=" << test_case.source_kind
                 << " required=" << test_case.source_required
                 << " rematerializable=" << test_case.result_rematerializable
                 << " retained=" << test_case.retained);
    const bool constant = test_case.source_kind == kConstant;
    const bool cross_bank = test_case.source_kind == kCrossBank;
    const uint8_t source_operand_count = constant ? 0 : 1;
    const uint8_t result_operand_count = constant ? 1 : 2;
    const loom_low_descriptor_t descriptors[] = {
        {
            .operand_count = static_cast<uint8_t>(source_operand_count + 1),
            .result_count = 1,
        },
        {
            .operand_start = 2,
            .operand_count = static_cast<uint8_t>(result_operand_count + 1),
            .result_count = 1,
            .layout_index = 1,
        },
    };
    loom_low_operand_t operands[] = {
        {.role = LOOM_LOW_OPERAND_ROLE_RESULT, .unit_count = 1},
        {
            .role = LOOM_LOW_OPERAND_ROLE_RESULT,
            .flags = LOOM_LOW_OPERAND_FLAG_TIED |
                     LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION,
            .unit_count = 1,
        },
        {
            .role = LOOM_LOW_OPERAND_ROLE_OPERAND,
            .unit_count = uint16_t(cross_bank ? 4 : 1),
        },
        {
            .source_value_index = uint16_t(result_operand_count - 1),
            .role = LOOM_LOW_OPERAND_ROLE_OPERAND,
            .flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT |
                     LOOM_LOW_OPERAND_FLAG_TIED |
                     LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION,
            .unit_count = 1,
        },
        {.role = LOOM_LOW_OPERAND_ROLE_OPERAND, .unit_count = 1},
    };
    if (test_case.source_kind != kNotRematerializable) {
      operands[0].flags |= LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE;
    }
    if (test_case.result_rematerializable) {
      operands[1].flags |= LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE;
    }
    uint16_t operand_indices[] = {0, 2, 1, uint16_t(constant ? 3 : 4), 3};
    const loom_low_descriptor_layout_t layouts[] = {
        {
            .minimum_packet_operand_count = source_operand_count,
            .op_kind = static_cast<loom_low_descriptor_op_kind_t>(
                constant ? LOOM_LOW_DESCRIPTOR_OP_KIND_CONST
                         : LOOM_LOW_DESCRIPTOR_OP_KIND_OP),
        },
        {
            .minimum_packet_operand_count = result_operand_count,
            .op_kind = LOOM_LOW_DESCRIPTOR_OP_KIND_OP,
        },
    };
    const loom_low_descriptor_set_t descriptor_set = {
        .descriptors = descriptors,
        .descriptor_layouts = layouts,
        .descriptor_layout_count = IREE_ARRAYSIZE(layouts),
        .descriptor_count = IREE_ARRAYSIZE(descriptors),
        .descriptor_ordinal_count = IREE_ARRAYSIZE(descriptors),
        .operands = operands,
        .operand_count = IREE_ARRAYSIZE(operands),
        .operand_indices = operand_indices,
        .operand_index_count = IREE_ARRAYSIZE(operand_indices),
        .reg_classes = reg_classes,
        .reg_class_count = IREE_ARRAYSIZE(reg_classes),
    };
    uint64_t placement_word = 0;
    uint64_t required_word = 0;
    loom_low_schedule_options_t options = {
        .required_register_values = {64, &required_word},
        .per_user_placement_values = {64, &placement_word},
    };
    loom_low_schedule_value_record_t values[4] = {};
    for (auto& value : values) {
      value.unit_count = 1;
    }
    if (cross_bank) {
      values[0].unit_count = 4;
      values[0].register_class_id = 1;
      values[0].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
      iree_bitmap_set(options.required_register_values, value_ids[0]);
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (test_case.retained & (1u << i)) {
        iree_bitmap_set(options.per_user_placement_values, value_ids[i + 2]);
      }
      if (i == 1 || test_case.source_required) {
        iree_bitmap_set(options.required_register_values, value_ids[i + 2]);
        values[i + 2].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
      }
    }
    loom_low_schedule_node_t nodes[] = {
        {
            .descriptor = &descriptors[0],
            .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
            .operand_count = source_operand_count,
            .result_count = 1,
            .value_ordinals = {.inline_value_ordinals = {0, 2}},
        },
        {
            .descriptor = &descriptors[1],
            .source_descriptor_ordinal = 1,
            .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
            .operand_count = result_operand_count,
            .result_count = 1,
            .storage_relation_count = 1,
            .value_ordinals = {.inline_value_ordinals = {1, 2, 3}},
        },
    };
    if (constant) {
      loom_low_schedule_node_result_ordinals(&nodes[0])[0] = 2;
      loom_low_schedule_node_operand_ordinals(&nodes[1])[0] = 2;
      loom_low_schedule_node_result_ordinals(&nodes[1])[0] = 3;
    }
    if (test_case.source_kind == kObservable) {
      nodes[0].traits = LOOM_TRAIT_OBSERVABLE_EFFECT;
    }
    if (test_case.source_kind == kOtherBlock) {
      nodes[1].block_index = 1;
    }
    *relation = {
        .destination_ordinal = 3,
        .source_ordinal = 2,
        .unit_count = 1,
        .source_operand_index = uint16_t(result_operand_count - 1),
        .flags = LOOM_LOW_STORAGE_RELATION_FLAG_HARD |
                 LOOM_LOW_STORAGE_RELATION_FLAG_WRITES_STORAGE,
        .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
        .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT,
    };
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptor_set},
        .value_domain = &domain,
        .values = values,
        .value_producer_nodes = producers,
        .nodes = nodes,
        .storage_relations = {relation_starts, &relations, 2, 1},
    };
    for (bool order_structural : {false, true}) {
      SCOPED_TRACE(order_structural);
      for (auto& node : nodes) {
        node.flags = 0;
      }
      state.setup_order.has_members = false;
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(nodes); ++i) {
        loom_low_schedule_setup_order_classify_node(&state, i,
                                                    order_structural);
      }
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(nodes); ++i) {
        EXPECT_EQ(iree_any_bit_set(nodes[i].flags,
                                   LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
                  (test_case.members & (1u << i)) != 0);
      }
      EXPECT_EQ(state.setup_order.has_members, test_case.members != 0);
    }
  }
}

TEST_F(SetupOrderTest, OrdersSharedCompletionAfterItsPrerequisites) {
  Initialize({4});
  nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  Append(0, 3);
  Append(1, 3);
  Append(2, 3);

  // The prerequisite precedes both setup members, which retain source order.
  ExpectAddedOrder({{0, 1}, {1, 2}});
  EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
  EXPECT_EQ(state_.setup_order.completion_nodes[1], 3u);
  EXPECT_EQ(state_.setup_order.completion_nodes[2], 3u);
  EXPECT_EQ(state_.setup_order.entry_nodes[3], 1u);
}

TEST_F(SetupOrderTest, ComposedSetupSharesItsTerminalConsumer) {
  Initialize({5});
  for (uint32_t i = 0; i < 3; ++i) {
    nodes_[i].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  }
  Append(0, 2);
  Append(1, 2);
  Append(2, 4);
  Append(3, 4);

  ExpectAddedOrder({{0, 1}, {3, 0}});
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(state_.setup_order.completion_nodes[i], 4u);
  }
  EXPECT_EQ(state_.setup_order.entry_nodes[4], 0u);
}

TEST_F(SetupOrderTest, NonValueSuccessorsDoNotCompleteSetup) {
  for (auto kind :
       {LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER}) {
    SCOPED_TRACE(kind);
    Initialize({3});
    nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    Append(0, 2, kind);
    Append(1, 2);

    ExpectAddedOrder({});
    EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    EXPECT_EQ(state_.setup_order.entry_nodes[2], 2u);
  }
}

TEST_F(SetupOrderTest, DistinctSuccessorBreaksSetupRegardlessOfSourcePosition) {
  for (bool side_effect_first : {false, true}) {
    for (bool side_edge_first : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "side effect first=" << side_effect_first
                   << " side edge first=" << side_edge_first);
      Initialize({4});
      nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
      const uint32_t consumer = side_effect_first ? 3 : 2;
      const uint32_t prerequisite = side_effect_first ? 2 : 1;
      const uint32_t side_effect = side_effect_first ? 1 : 3;
      if (side_edge_first) {
        Append(0, side_effect, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE);
      }
      Append(0, consumer);
      Append(prerequisite, consumer);
      if (!side_edge_first) {
        Append(0, side_effect, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE);
      }

      ExpectAddedOrder({});
      EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    }
  }
}

TEST_F(SetupOrderTest, FanoutEndsOnlyTheEarlierPartOfASetupChain) {
  for (auto kind :
       {LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE}) {
    SCOPED_TRACE(kind);
    Initialize({5});
    nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    Append(0, 1);
    Append(1, 4);
    Append(0, 2, kind);
    Append(2, 3, kind);
    Append(3, 4);

    ExpectAddedOrder({{3, 1}});
    EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    EXPECT_EQ(state_.setup_order.completion_nodes[1], 4u);
  }
}

TEST_F(SetupOrderTest, RepairRequiresASoleValueConsumerAndAnImpliedSideEdge) {
  for (bool repair : {false, true}) {
    for (bool implied : {false, true}) {
      for (auto side_kind : {LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
                             LOOM_LOW_SCHEDULE_DEPENDENCY_SSA}) {
        SCOPED_TRACE(::testing::Message()
                     << "repair=" << repair << " implied=" << implied
                     << " side kind=" << side_kind);
        Initialize({4});
        options_.flags =
            repair ? LOOM_LOW_SCHEDULE_FLAG_REPAIR_PRIVATE_SETUP : 0;
        nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
        Append(0, 2);
        Append(1, 2);
        Append(1, 3, side_kind);
        if (implied) {
          Append(2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
        }

        if (repair && implied &&
            side_kind != LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          ExpectAddedOrder({{0, 1}});
        } else {
          ExpectAddedOrder({});
        }
      }
    }
  }
}

TEST_F(SetupOrderTest, SourceBoundaryDoesNotAcquireAnEarlierSetupChain) {
  Initialize({4});
  nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
  Append(0, 2);
  Append(1, 2);
  Append(2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);

  ExpectAddedOrder({});
  EXPECT_EQ(state_.setup_order.completion_nodes[1], 1u);
}

TEST_F(SetupOrderTest, RepeatedValueEdgesRetainTimingThroughRepair) {
  Initialize({4});
  options_.flags = LOOM_LOW_SCHEDULE_FLAG_REPAIR_PRIVATE_SETUP;
  nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[3].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
  Append(0, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 3);
  Append(0, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 5);
  Append(1, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 7);
  Append(0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
  Append(1, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
  Append(2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);

  // Repeated operands still have one consumer. The shared boundary permits
  // delaying setup until its peer, without changing either input's timing.
  ExpectAddedOrder({{1, 0}});
  EXPECT_EQ(state_.setup_order.completion_nodes[0], 2u);
  const int32_t separations[] = {3, 5, 7};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(separations); ++i) {
    const auto* edge =
        loom_low_schedule_dependency_graph_at(&state_.dependencies, i);
    EXPECT_EQ(edge->producer_node, i < 2 ? 0u : 1u);
    EXPECT_EQ(edge->consumer_node, 2u);
    EXPECT_EQ(edge->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA);
    EXPECT_EQ(edge->minimum_issue_separation_cycles, separations[i]);
  }
}

TEST_F(SetupOrderTest, ClosesIndependentChainsInEveryBlock) {
  Initialize({3, 3});
  nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[3].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  Append(0, 2);
  Append(1, 2);
  Append(3, 5);
  Append(4, 5);

  ExpectAddedOrder({{1, 0}, {4, 3}});
  EXPECT_EQ(state_.setup_order.entry_nodes[2], 0u);
  EXPECT_EQ(state_.setup_order.entry_nodes[5], 3u);
}

}  // namespace
}  // namespace loom
