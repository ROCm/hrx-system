// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_reference_summary.h"

#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class SymbolReferenceSummaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(),
                                     &module_pool_);
    iree_arena_block_pool_initialize(32768, iree_allocator_system(),
                                     &summary_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("references"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    iree_arena_initialize(&summary_pool_, &arena_);
    loom_symbol_reference_summary_initialize(module_, &arena_, &summary_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&summary_pool_);
    iree_arena_block_pool_deinitialize(&module_pool_);
  }

  loom_symbol_ref_t Symbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_ref_t result = {};
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &result.symbol_id));
    return result;
  }

  loom_type_id_t Scalar() {
    loom_type_id_t result = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &result));
    return result;
  }

  uint16_t Encoding(iree_string_view_t family, loom_attribute_t value) {
    loom_named_attr_t parameter = {};
    IREE_CHECK_OK(loom_module_intern_string(module_, IREE_SV("value"),
                                            &parameter.name_id));
    parameter.value = value;
    loom_encoding_t encoding = {};
    IREE_CHECK_OK(
        loom_module_intern_string(module_, family, &encoding.name_id));
    encoding.alias_id = LOOM_STRING_ID_INVALID;
    encoding.attribute_count = 1;
    encoding.attributes = &parameter;
    uint16_t result = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &encoding, &result));
    return result;
  }

  loom_type_id_t ReferenceType(loom_symbol_ref_t target) {
    uint16_t encoding =
        Encoding(IREE_SV("dependent"), loom_attr_symbol(target));
    loom_type_id_t result = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(
        module_,
        loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_static(4), encoding),
        &result));
    return result;
  }

  loom_type_id_t Function(loom_type_id_t first, loom_type_id_t second) {
    alignas(loom_func_type_data_t) uint8_t
        storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
    auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
    data->arg_count = 2;
    data->types[0] = loom_type_table_get(&module_->types, first);
    data->types[1] = loom_type_table_get(&module_->types, second);
    const loom_type_id_t dependencies[] = {first, second};
    loom_type_id_t result = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_topological_type_id(
        module_, loom_type_function(data), dependencies, 2, &result));
    return result;
  }

  loom_attribute_t Dictionary(loom_attribute_t value) {
    loom_named_attr_t entry = {};
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("value"), &entry.name_id));
    entry.value = value;
    loom_attribute_t result = {};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module_, loom_make_named_attr_slice(&entry, 1), &result));
    return result;
  }

  std::vector<loom_symbol_id_t> Targets(
      loom_symbol_reference_occurrence_kind_t expected_kind) {
    std::vector<loom_symbol_id_t> targets;
    loom_symbol_reference_summary_span_t span;
    while (loom_symbol_reference_summary_next(&summary_, &span)) {
      EXPECT_EQ(span.kind, expected_kind);
      EXPECT_EQ(span.role, LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY);
      EXPECT_EQ(span.interfaces, 0u);
      for (iree_host_size_t i = 0; i < span.count; ++i) {
        targets.push_back(span.targets[i].symbol_id);
      }
    }
    return targets;
  }

  // Construction storage excluded from summary-allocation observations.
  iree_arena_block_pool_t module_pool_ = {};
  // Independent invocation scratch pool.
  iree_arena_block_pool_t summary_pool_ = {};
  // Permissive context for the public type/encoding construction APIs.
  loom_context_t context_ = {};
  // Immutable structural payload owner while a query is active.
  loom_module_t* module_ = nullptr;
  // Invocation-owned summary storage.
  iree_arena_allocator_t arena_ = {};
  // Pinned structural analysis owner and current cursor.
  loom_symbol_reference_summary_t summary_ = {};
};

TEST_F(SymbolReferenceSummaryTest, PlainScalarsNeedNoScratchStorage) {
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
      &summary_, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
  EXPECT_TRUE(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE).empty());
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&summary_pool_, &statistics);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);
}

TEST_F(SymbolReferenceSummaryTest, UntypedRegistersNeedNoScratchStorage) {
  for (uint64_t count : {1, 2, 4, 8}) {
    const auto type = loom_type_register_payload(123, count);
    EXPECT_FALSE(loom_symbol_reference_type_may_contain_ref(type));
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
        &summary_, type, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
    EXPECT_TRUE(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE).empty());
  }
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&summary_pool_, &statistics);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);
}

TEST_F(SymbolReferenceSummaryTest, TypedRegistersRetainValueReferences) {
  const auto target = Symbol(IREE_SV("target"));
  const auto value_type = ReferenceType(target);
  loom_register_type_data_t data = {
      .carrier_payload0 = 123,
      .carrier_payload1 = 4,
      .value_type = loom_type_table_get(&module_->types, value_type),
  };
  loom_type_id_t register_type = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_register_payload_with_value_type(&data),
      &register_type));
  const auto type = loom_type_table_get(&module_->types, register_type);
  EXPECT_TRUE(loom_symbol_reference_type_may_contain_ref(type));
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
      &summary_, type, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
            std::vector<loom_symbol_id_t>{target.symbol_id});
}

TEST_F(SymbolReferenceSummaryTest, SharedArrayPayloadRetainsFieldSemantics) {
  const auto first = Symbol(IREE_SV("first"));
  const auto second = Symbol(IREE_SV("second"));
  loom_symbol_ref_t* targets = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate_array(&module_->arena, 3, sizeof(*targets),
                                           reinterpret_cast<void**>(&targets)));
  targets[0] = first;
  targets[1] = second;
  targets[2] = first;
  const auto attr = loom_attr_symbol_array(targets, 3);
  loom_symbol_reference_descriptor_t dependency = {
      .interfaces = LOOM_SYMBOL_INTERFACE_GLOBAL,
  };
  loom_symbol_reference_descriptor_t availability = {
      .interfaces = LOOM_SYMBOL_INTERFACE_CALLABLE,
      .role = LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY,
  };
  for (const auto* semantics : {&dependency, &availability, &dependency}) {
    loom_attr_descriptor_t descriptor = {.attr_kind = LOOM_ATTR_SYMBOL_ARRAY};
    descriptor.reference.symbol_ref = semantics;
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
        &summary_, attr, &descriptor,
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR));
    loom_symbol_reference_summary_span_t span;
    ASSERT_TRUE(loom_symbol_reference_summary_next(&summary_, &span));
    EXPECT_EQ(span.targets, targets);
    ASSERT_EQ(span.count, 3u);
    EXPECT_EQ(span.targets[0].symbol_id, first.symbol_id);
    EXPECT_EQ(span.targets[1].symbol_id, second.symbol_id);
    EXPECT_EQ(span.targets[2].symbol_id, first.symbol_id);
    EXPECT_EQ(span.kind, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
    EXPECT_EQ(span.role, semantics->role);
    EXPECT_EQ(span.interfaces, semantics->interfaces);
    EXPECT_FALSE(loom_symbol_reference_summary_next(&summary_, &span));
  }
}

TEST_F(SymbolReferenceSummaryTest, SharedEmptyGraphDoesNotExpandPerPath) {
  loom_type_id_t type = Scalar();
  std::array<loom_type_id_t, 64> roots;
  for (auto& root : roots) {
    type = Function(type, type);
    root = type;
  }
  for (auto root : roots) {
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
        &summary_, loom_type_table_get(&module_->types, root),
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
    EXPECT_TRUE(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE).empty());
  }
}

TEST_F(SymbolReferenceSummaryTest, SharedArraysFilterOnlyModuleLocalTargets) {
  const auto first = Symbol(IREE_SV("first"));
  const auto second = Symbol(IREE_SV("second"));
  loom_symbol_ref_t* targets = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate_array(&module_->arena, 6, sizeof(*targets),
                                           reinterpret_cast<void**>(&targets)));
  targets[0] = first;
  targets[1] = {1, first.symbol_id};
  targets[2] = {0, LOOM_SYMBOL_ID_INVALID};
  targets[3] = second;
  targets[4] = first;
  targets[5] = {1, second.symbol_id};
  const auto attr = loom_attr_symbol_array(targets, 6);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
        &summary_, attr, nullptr,
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR));
    EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR),
              (std::vector<loom_symbol_id_t>{first.symbol_id, second.symbol_id,
                                             first.symbol_id}));
  }
}

TEST_F(SymbolReferenceSummaryTest, RejectsOutOfRangeLocalTargetsAtDiscovery) {
  const loom_symbol_ref_t target = {0, 0};
  iree_status_t status = loom_symbol_reference_summary_query_attr(
      &summary_, loom_attr_symbol(target), nullptr,
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SymbolReferenceSummaryTest, EncodingEdgesResetAggregateDepth) {
  const auto target = Symbol(IREE_SV("target"));
  auto nested = loom_attr_symbol(target);
  // The encoding's parameter dictionary consumes one aggregate level.
  for (iree_host_size_t i = 1; i < LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH; ++i) {
    nested = Dictionary(nested);
  }
  const auto encoding = Encoding(IREE_SV("nested"), nested);
  auto root = loom_attr_encoding(encoding);
  for (iree_host_size_t i = 0; i < LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH; ++i) {
    root = Dictionary(root);
  }
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
      &summary_, root, nullptr, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_ENCODING_ATTR),
            std::vector<loom_symbol_id_t>{target.symbol_id});
}

TEST_F(SymbolReferenceSummaryTest,
       SharedNonemptyGraphPreservesOrderedMultiplicity) {
  const auto first = Symbol(IREE_SV("first"));
  const auto second = Symbol(IREE_SV("second"));
  loom_type_id_t type = Function(ReferenceType(first), ReferenceType(second));
  for (iree_host_size_t i = 0; i < 10; ++i) {
    type = Function(type, type);
  }
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
      &summary_, loom_type_table_get(&module_->types, type),
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
  const auto targets = Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE);
  ASSERT_EQ(targets.size(), 2048u);
  for (iree_host_size_t i = 0; i < targets.size(); i += 2) {
    EXPECT_EQ(targets[i], first.symbol_id);
    EXPECT_EQ(targets[i + 1], second.symbol_id);
  }
}

TEST_F(SymbolReferenceSummaryTest, UnaryBypassComposesInnerKindOverrides) {
  const auto target = Symbol(IREE_SV("target"));
  const auto type = ReferenceType(target);
  const uint16_t encoding = Encoding(IREE_SV("typed"), loom_attr_type(type));
  const auto root = Dictionary(loom_attr_encoding(encoding));
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
      &summary_, root, nullptr, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR),
            std::vector<loom_symbol_id_t>{target.symbol_id});
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_encoding(
      &summary_, encoding, LOOM_SYMBOL_REFERENCE_OCCURRENCE_MODULE_ENCODING));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR),
            std::vector<loom_symbol_id_t>{target.symbol_id});
}

TEST_F(SymbolReferenceSummaryTest, CachedDictionariesKeepTheQueryKind) {
  const auto target = Symbol(IREE_SV("target"));
  const auto root = Dictionary(Dictionary(loom_attr_symbol(target)));
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
      &summary_, root, nullptr, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            std::vector<loom_symbol_id_t>{target.symbol_id});
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_attr(
      &summary_, root, nullptr, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
  EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
            std::vector<loom_symbol_id_t>{target.symbol_id});
}

TEST_F(SymbolReferenceSummaryTest, UnaryPrefixesReuseTheCollapsedLeaf) {
  const auto target = Symbol(IREE_SV("target"));
  const auto scalar = Scalar();
  loom_type_id_t type = ReferenceType(target);
  std::array<loom_type_id_t, 2048> roots;
  for (auto& root : roots) {
    type = Function(type, scalar);
    root = type;
  }
  const loom_symbol_ref_t* retained_target = nullptr;
  for (auto root : roots) {
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
        &summary_, loom_type_table_get(&module_->types, root),
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
    loom_symbol_reference_summary_span_t span;
    ASSERT_TRUE(loom_symbol_reference_summary_next(&summary_, &span));
    ASSERT_EQ(span.count, 1u);
    EXPECT_EQ(span.targets[0].symbol_id, target.symbol_id);
    EXPECT_EQ(span.kind, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE);
    if (retained_target) {
      EXPECT_EQ(span.targets, retained_target);
    } else {
      retained_target = span.targets;
    }
    EXPECT_FALSE(loom_symbol_reference_summary_next(&summary_, &span));
  }
}

TEST_F(SymbolReferenceSummaryTest, IndexGrowthPreservesIndependentTargets) {
  std::array<loom_symbol_ref_t, 128> targets;
  std::array<loom_type_id_t, 128> types;
  for (iree_host_size_t i = 0; i < targets.size(); ++i) {
    const auto name = "target_" + std::to_string(i);
    targets[i] = Symbol(iree_make_string_view(name.data(), name.size()));
    types[i] = ReferenceType(targets[i]);
  }
  for (iree_host_size_t i = 0; i < types.size() * 2; ++i) {
    const auto index = i < types.size() ? i : types.size() * 2 - i - 1;
    IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
        &summary_, loom_type_table_get(&module_->types, types[index]),
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
    EXPECT_EQ(Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
              std::vector<loom_symbol_id_t>{targets[index].symbol_id});
  }
}

TEST_F(SymbolReferenceSummaryTest,
       GrowingPrefixesRetainGraphNotExpandedVectors) {
  const auto target = Symbol(IREE_SV("target"));
  const loom_type_id_t leaf = ReferenceType(target);
  loom_type_id_t type = leaf;
  constexpr iree_host_size_t kReferenceCount = 2048;
  for (iree_host_size_t i = 1; i < kReferenceCount; ++i) {
    type = Function(type, leaf);
  }
  IREE_ASSERT_OK(loom_symbol_reference_summary_query_type(
      &summary_, loom_type_table_get(&module_->types, type),
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE));
  const auto targets = Targets(LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE);
  ASSERT_EQ(targets.size(), kReferenceCount);
  for (auto actual : targets) {
    EXPECT_EQ(actual, target.symbol_id);
  }
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&summary_pool_, &statistics);
  // The bound includes radix entries, geometric arrays, and pool slack. An
  // expanded vector for every prefix alone would retain over eight megabytes.
  EXPECT_LT(statistics.block_system_allocation_bytes +
                statistics.oversized_allocation_bytes,
            512 * kReferenceCount);
}

}  // namespace
}  // namespace loom
