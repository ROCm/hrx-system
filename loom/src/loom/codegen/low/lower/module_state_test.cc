// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/module_state.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/target/test/alt_descriptors.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowLowerModuleStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(
        loom_low_lower_module_state_allocate(&arena_, &module_state_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Blocks shared by pass-local state and optional module storage.
  iree_arena_block_pool_t block_pool_;
  // Pass-local planning storage.
  iree_arena_allocator_t arena_;
  // State owned by arena_.
  loom_low_lower_module_state_t* module_state_ = nullptr;
};

TEST_F(LowLowerModuleStateTest, SharesDescriptorBindingsAcrossFunctions) {
  // Two valid rule tables refer to disjoint generated descriptor sets. Their
  // ordering is part of each policy's rule-set ordinal contract.
  const iree_string_view_t keys[] = {IREE_SV("test.add.i32"),
                                     IREE_SV("test.alt.neg.i32")};
  loom_low_lower_rule_descriptor_ref_t refs[2] = {};
  loom_low_lower_rule_set_t rules[2] = {};
  for (uint16_t i = 0; i < 2; ++i) {
    refs[i].stable_id = loom_low_descriptor_stable_id_from_key(keys[i]);
    rules[i].descriptor_refs = &refs[i];
    rules[i].descriptor_ref_count = 1;
  }
  const loom_low_lower_rule_set_t* ordered[] = {&rules[0], &rules[1]};
  const loom_low_lower_rule_set_t* reversed[] = {&rules[1], &rules[0]};
  const loom_low_lower_rule_set_list_t lists[] = {{2, ordered}, {2, reversed}};
  const loom_low_descriptor_set_t* sets[] = {
      loom_test_low_core_descriptor_set(), loom_test_low_alt_descriptor_set()};
  loom_low_lower_rule_descriptor_cache_t* caches[2][2] = {};
  for (uint16_t list = 0; list < 2; ++list) {
    for (uint16_t set = 0; set < 2; ++set) {
      IREE_ASSERT_OK(loom_low_lower_module_state_rule_descriptor_cache(
          module_state_, lists[list], sets[set], &caches[list][set]));
      for (uint16_t index = 0; index < 2; ++index) {
        const loom_low_descriptor_t* descriptor = nullptr;
        IREE_ASSERT_OK(loom_low_lower_rule_descriptor_cache_resolve(
            caches[list][set], index, 0, &descriptor));
        const loom_low_lower_rule_match_context_t match_context = {
            .descriptor_set = sets[set],
        };
        const loom_low_descriptor_t* uncached_descriptor = nullptr;
        IREE_ASSERT_OK(loom_low_lower_rule_resolve_descriptor_ref(
            &match_context, lists[list].values[index], 0,
            &uncached_descriptor));
        EXPECT_EQ(uncached_descriptor, descriptor);
        if (index == (list ^ set)) {
          ASSERT_NE(descriptor, nullptr);
          EXPECT_TRUE(iree_string_view_equal(
              loom_low_descriptor_set_string(
                  sets[set],
                  loom_low_descriptor_key_string_ref(sets[set], descriptor)),
              keys[set]));
        } else {
          EXPECT_EQ(descriptor, nullptr);
        }
        IREE_ASSERT_OK(loom_low_lower_rule_resolve_descriptor_ref(
            &match_context, lists[list].values[index],
            LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE, &uncached_descriptor));
        EXPECT_EQ(uncached_descriptor, nullptr);
      }
    }
  }

  // Revisit all table pairs in another order, as independent function plans
  // do. Neither successful bindings nor cached misses allocate another map.
  const auto used_bytes = arena_.used_allocation_size;
  for (uint16_t set = 0; set < 2; ++set) {
    for (uint16_t list = 0; list < 2; ++list) {
      loom_low_lower_rule_descriptor_cache_t* cache = nullptr;
      IREE_ASSERT_OK(loom_low_lower_module_state_rule_descriptor_cache(
          module_state_, lists[list], sets[set], &cache));
      EXPECT_EQ(cache, caches[list][set]);
      for (uint16_t index = 0; index < 2; ++index) {
        const loom_low_descriptor_t* descriptor = nullptr;
        IREE_ASSERT_OK(loom_low_lower_rule_descriptor_cache_resolve(
            cache, index, 0, &descriptor));
        EXPECT_EQ(descriptor != nullptr, index == (list ^ set));
      }
    }
  }
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
}

class LowLowerReadOnlyDataTest : public LowLowerModuleStateTest {
 protected:
  void SetUp() override {
    LowLowerModuleStateTest::SetUp();
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_global_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_GLOBAL, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("resources"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    LowLowerModuleStateTest::TearDown();
  }

  // Registry containing only the global dialect used for payload definitions.
  loom_context_t context_;
  // Module that receives symbols only after payload planning is complete.
  loom_module_t* module_ = nullptr;
};

TEST_F(LowLowerReadOnlyDataTest, RetainsPayloadsUntilSymbolPublication) {
  loom_low_lower_read_only_data_id_t ids[32];
  const auto initial_symbol_count = module_->symbols.count;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    uint8_t bytes[] = {(uint8_t)i, 0x80, 0xFF};
    IREE_ASSERT_OK(loom_low_lower_module_state_intern_read_only_data(
        module_state_, iree_make_const_byte_span(bytes, sizeof(bytes)), 8,
        LOOM_LOCATION_UNKNOWN, &ids[i]));
    // Callers may reuse projection scratch immediately after interning.
    bytes[0] = 0xFF;
  }
  EXPECT_EQ(module_->symbols.count, initial_symbol_count);
  EXPECT_EQ(loom_module_block(module_)->op_count, 0u);

  // Revisit every identity after record growth and raise its alignment.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    const uint8_t bytes[] = {(uint8_t)i, 0x80, 0xFF};
    loom_low_lower_read_only_data_id_t duplicate_id;
    IREE_ASSERT_OK(loom_low_lower_module_state_intern_read_only_data(
        module_state_, iree_make_const_byte_span(bytes, sizeof(bytes)), 64,
        LOOM_LOCATION_UNKNOWN, &duplicate_id));
    EXPECT_EQ(duplicate_id, ids[i]);
  }
  EXPECT_EQ(module_->symbols.count, initial_symbol_count);

  loom_symbol_ref_t first_symbol;
  IREE_ASSERT_OK(loom_low_lower_module_state_reference_read_only_data(
      module_state_, module_, ids[0], &first_symbol));
  loom_symbol_ref_t repeated_symbol;
  IREE_ASSERT_OK(loom_low_lower_module_state_reference_read_only_data(
      module_state_, module_, ids[0], &repeated_symbol));
  EXPECT_EQ(repeated_symbol.symbol_id, first_symbol.symbol_id);
  EXPECT_EQ(module_->symbols.count, initial_symbol_count + 1);
  EXPECT_EQ(loom_module_block(module_)->op_count, 0u);

  // Finalization also publishes resources with no earlier symbol reference.
  IREE_ASSERT_OK(loom_low_lower_module_state_finalize(module_state_, module_));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    loom_symbol_ref_t symbol;
    IREE_ASSERT_OK(loom_low_lower_module_state_reference_read_only_data(
        module_state_, module_, ids[i], &symbol));
    const loom_op_t* definition =
        module_->symbols.entries[symbol.symbol_id].defining_op;
    ASSERT_NE(definition, nullptr);
    ASSERT_TRUE(loom_global_rodata_def_isa(definition));
    EXPECT_EQ(loom_global_rodata_def_alignment(definition), 64);
    const iree_const_byte_span_t contents =
        loom_global_rodata_def_contents(definition);
    ASSERT_EQ(contents.data_length, 3u);
    EXPECT_EQ(contents.data[0], i);
    EXPECT_EQ(contents.data[1], 0x80);
    EXPECT_EQ(contents.data[2], 0xFF);
  }
  EXPECT_EQ(module_->symbols.count, initial_symbol_count + IREE_ARRAYSIZE(ids));
  EXPECT_EQ(loom_module_block(module_)->op_count, IREE_ARRAYSIZE(ids));
  IREE_ASSERT_OK(loom_low_lower_module_state_finalize(module_state_, module_));
  EXPECT_EQ(loom_module_block(module_)->op_count, IREE_ARRAYSIZE(ids));
}

TEST_F(LowLowerModuleStateTest, InternsZeroInitializedStateByStaticKey) {
  static const uint8_t kFirstKey = 0;
  static const uint8_t kSecondKey = 0;

  uint32_t* first = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kFirstKey, sizeof(*first), (void**)&first));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(*first, 0u);
  *first = 42;

  uint32_t* first_again = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kFirstKey, sizeof(*first_again), (void**)&first_again));
  EXPECT_EQ(first_again, first);
  EXPECT_EQ(*first_again, 42u);

  uint32_t* second = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kSecondKey, sizeof(*second), (void**)&second));
  ASSERT_NE(second, nullptr);
  EXPECT_NE(second, first);
  EXPECT_EQ(*second, 0u);
}

TEST_F(LowLowerModuleStateTest, PreservesStateAcrossRecordGrowth) {
  static const uint8_t kKeys[32] = {0};
  uint32_t* values[IREE_ARRAYSIZE(kKeys)] = {nullptr};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
        module_state_, &kKeys[i], sizeof(*values[i]), (void**)&values[i]));
    *values[i] = (uint32_t)(i + 1);
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    uint32_t* value = nullptr;
    IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
        module_state_, &kKeys[i], sizeof(*value), (void**)&value));
    EXPECT_EQ(value, values[i]);
    EXPECT_EQ(*value, (uint32_t)(i + 1));
  }
}

TEST_F(LowLowerModuleStateTest, AllocatesPassLocalStorage) {
  void* bytes = nullptr;
  IREE_ASSERT_OK(
      loom_low_lower_module_state_allocate_bytes(module_state_, 17, &bytes));
  EXPECT_NE(bytes, nullptr);

  uint32_t* values = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_allocate_array(
      module_state_, 8, sizeof(*values), (void**)&values));
  EXPECT_NE(values, nullptr);
}

TEST(LowLowerModuleStateStandaloneTest, RequiresModuleStateForNonemptyStorage) {
  static const uint8_t kKey = 0;
  void* data = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_get_or_allocate(nullptr, &kKey, 1, &data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_allocate_bytes(nullptr, 1, &data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_allocate_array(nullptr, 1, 1, &data));
}

}  // namespace
}  // namespace loom
