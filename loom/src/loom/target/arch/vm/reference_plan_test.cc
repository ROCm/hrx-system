// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/reference_plan.h"

#include <algorithm>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type/ops.h"
#include "loom/ops/type_builders.h"
#include "loom/ops/type_registry.h"

namespace {

class VMReferencePlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_type_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TYPE,
                                                 vtables, (uint16_t)count));
    iree_arena_block_pool_initialize(32 * 1024, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  void CreateTypes(uint32_t type_count, uint32_t key_count) {
    names_.resize(type_count);
    spellings_.resize(type_count);
    descriptors_.resize(type_count);
    keys_.resize(key_count);
    key_names_.resize(key_count);
    for (uint32_t i = 0; i < key_count; ++i) {
      key_names_[i] = "object_" + std::to_string(key_count - i);
      keys_[i] = {i % 2 ? IREE_SV("alpha") : IREE_SV("zeta"),
                  {key_names_[i].data(), key_names_[i].size()}};
    }
    std::vector<loom_type_registry_entry_t> entries(type_count);
    for (uint32_t i = 0; i < type_count; ++i) {
      names_[i] = "custom.type_" + std::to_string(i);
      auto& spelling = spellings_[i];
      spelling.push_back(static_cast<uint8_t>(names_[i].size()));
      spelling.insert(spelling.end(), names_[i].begin(), names_[i].end());
      spelling.push_back('<');
      auto& descriptor = descriptors_[i];
      descriptor.name = spelling.data();
      descriptor.ir_kind = LOOM_TYPE_DIALECT;
      descriptor.semantics.semantic = LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE;
      descriptor.reference = &keys_[i % key_count];
      entries[i] = {{names_[i].data(), names_[i].size()}, &descriptor};
    }
    IREE_ASSERT_OK(loom_type_registry_register_types(&context_, entries.data(),
                                                     entries.size()));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("references"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  loom_type_t InternType(uint32_t index) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, {names_[index].data(), names_[index].size()}, &name));
    return loom_type_dialect_opaque(name);
  }

  loom_type_t Declare(iree_string_view_t name,
                      const loom_type_reference_key_t& key) {
    loom_string_id_t name_id, namespace_id, payload_id;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, key.namespace_name, &namespace_id));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, key.type_name, &payload_id));
    loom_symbol_ref_t symbol = {};
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol.symbol_id));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_type_decl_build(&builder, namespace_id, payload_id,
                                       symbol, LOOM_LOCATION_UNKNOWN, &op));
    loom_type_t type;
    IREE_CHECK_OK(loom_ref_type_make(module_, symbol, &type));
    return type;
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* params, void** pointer) {
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        static_cast<VMReferencePlanTest*>(self)->fail_allocations_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "reference storage allocation rejected");
    }
    const auto system = iree_allocator_system();
    return system.ctl(system.self, command, params, pointer);
  }

  // Injects failure only after source declarations and prior rows are prepared.
  bool fail_allocations_ = false;
  // Registry owning declaration lookup state while metadata stays borrowed.
  loom_context_t context_ = {};
  // Shared fixed blocks for module storage and independent planner scratch.
  iree_arena_block_pool_t pool_ = {};
  // Planner-only allocation scope used for exact storage accounting.
  iree_arena_allocator_t arena_ = {};
  // Module owning the full-width nominal source IDs.
  loom_module_t* module_ = nullptr;
  // Stable source type names registered with the context.
  std::vector<std::string> names_;
  // Descriptor spelling bytes whose addresses remain stable after registration.
  std::vector<std::vector<uint8_t>> spellings_;
  // Immutable registered declarations, retained through context teardown.
  std::vector<loom_type_descriptor_t> descriptors_;
  // Shared canonical identities, including aliases across distinct
  // declarations.
  std::vector<loom_type_reference_key_t> keys_;
  // Stable name bytes borrowed by canonical identities.
  std::vector<std::string> key_names_;
};

static bool KeyLess(const loom_type_reference_key_t* lhs,
                    const loom_type_reference_key_t* rhs) {
  const int comparison =
      iree_string_view_compare(lhs->namespace_name, rhs->namespace_name);
  return comparison
             ? comparison < 0
             : iree_string_view_compare(lhs->type_name, rhs->type_name) < 0;
}

static void ExpectKeyEqual(const loom_type_reference_key_t* actual,
                           const loom_type_reference_key_t* expected) {
  EXPECT_TRUE(
      iree_string_view_equal(actual->namespace_name, expected->namespace_name));
  EXPECT_TRUE(iree_string_view_equal(actual->type_name, expected->type_name));
}

TEST_F(VMReferencePlanTest, EmptyPlanAllocatesNothing) {
  loom_vm_reference_plan_t plan = {};
  EXPECT_FALSE(loom_vm_reference_plan_finalize(&plan));
  EXPECT_EQ(plan.count, 0u);
  EXPECT_EQ(plan.group_count, 0u);
  EXPECT_EQ(arena_.used_allocation_size, 0u);
}

TEST_F(VMReferencePlanTest, InlineModesAndStaticAliasesShareExternalKeys) {
  ASSERT_NO_FATAL_FAILURE(CreateTypes(1, 1));
  const auto nominal = InternType(0);
  const loom_type_reference_key_t other_key = {IREE_SVL("different"),
                                               IREE_SVL("payload")};
  const auto owner = Declare(IREE_SV("Object"), keys_[0]);
  const auto other = Declare(IREE_SV("Other"), other_key);
  const auto alias = Declare(IREE_SV("Alias"), keys_[0]);
  loom_type_t borrowed;
  IREE_ASSERT_OK(loom_ptr_type_make(
      module_, {.symbol_id = loom_type_reference_declaration_id(owner)},
      &borrowed));
  // Equal numeric IDs in the two source domains must not alias their keys.
  ASSERT_EQ(loom_type_dialect_name_id(nominal),
            loom_type_reference_declaration_id(other));
  loom_vm_reference_plan_t plan = {};
  const loom_type_t types[] = {nominal, other, owner, borrowed, alias};
  uint16_t ordinals[IREE_ARRAYSIZE(types)];
  for (size_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, types[i],
                                               &arena_, &ordinals[i]));
  }
  EXPECT_EQ(plan.count, 2u);
  EXPECT_NE(ordinals[0], ordinals[1]);
  EXPECT_EQ(ordinals[0], ordinals[2]);
  EXPECT_EQ(ordinals[0], ordinals[3]);
  EXPECT_EQ(ordinals[0], ordinals[4]);
  const auto bytes = arena_.used_allocation_size;
  for (size_t i = 0; i < 10000; ++i) {
    const auto index = i % IREE_ARRAYSIZE(types);
    uint16_t ordinal;
    IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, types[index],
                                               &arena_, &ordinal));
    EXPECT_EQ(ordinal, ordinals[index]);
  }
  EXPECT_EQ(arena_.used_allocation_size, bytes);
  loom_vm_reference_plan_finalize(&plan);
  ExpectKeyEqual(loom_vm_reference_plan_key(
                     &plan, loom_vm_reference_plan_ordinal(&plan, ordinals[0])),
                 &keys_[0]);
  ExpectKeyEqual(loom_vm_reference_plan_key(
                     &plan, loom_vm_reference_plan_ordinal(&plan, ordinals[1])),
                 &other_key);
}

TEST_F(VMReferencePlanTest, AliasesAndRepeatedUsesShareCanonicalOrdinals) {
  ASSERT_NO_FATAL_FAILURE(CreateTypes(129, 43));
  loom_vm_reference_plan_t plan = {};
  std::vector<uint16_t> ordinals(names_.size());
  for (uint32_t i = 0; i < names_.size(); ++i) {
    IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, InternType(i),
                                               &arena_, &ordinals[i]));
  }
  EXPECT_EQ(plan.count, keys_.size());
  // 172 index entries use 256 buckets; 35 overflow keys use two 32-row chunks.
  EXPECT_EQ(arena_.used_allocation_size,
            256 * sizeof(loom_intern_bucket_t) +
                64 * sizeof(loom_vm_reference_entry_t));
  const auto bytes = arena_.used_allocation_size;
  for (uint32_t i = 0; i < 100000; ++i) {
    uint16_t ordinal = 0;
    const uint32_t index = i % names_.size();
    IREE_ASSERT_OK(loom_vm_reference_plan_bind(
        &plan, module_, InternType(index), &arena_, &ordinal));
    EXPECT_EQ(ordinal, ordinals[index]);
  }
  EXPECT_EQ(arena_.used_allocation_size, bytes);
  EXPECT_TRUE(loom_vm_reference_plan_finalize(&plan));
  EXPECT_EQ(plan.group_count, 2u);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
  for (uint32_t i = 0; i < names_.size(); ++i) {
    const uint16_t ordinal = loom_vm_reference_plan_ordinal(&plan, ordinals[i]);
    ExpectKeyEqual(loom_vm_reference_plan_key(&plan, ordinal),
                   &keys_[i % keys_.size()]);
  }
}

TEST_F(VMReferencePlanTest, CanonicalOrderCrossesBothSegmentDirectories) {
  ASSERT_NO_FATAL_FAILURE(CreateTypes(4097, 4097));
  loom_vm_reference_plan_t plan = {};
  std::vector<uint16_t> ordinals(keys_.size());
  std::vector<const loom_type_reference_key_t*> expected;
  for (uint32_t i = 0; i < keys_.size(); ++i) {
    IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, InternType(i),
                                               &arena_, &ordinals[i]));
    expected.push_back(&keys_[i]);
  }
  const auto bytes = arena_.used_allocation_size;
  EXPECT_EQ(bytes, 16384 * sizeof(loom_intern_bucket_t) +
                       128 * 32 * sizeof(loom_vm_reference_entry_t) +
                       2 * sizeof(loom_segmented_storage_page_t));
  EXPECT_TRUE(loom_vm_reference_plan_finalize(&plan));
  std::sort(expected.begin(), expected.end(), KeyLess);
  for (uint32_t i = 0; i < expected.size(); ++i) {
    ExpectKeyEqual(loom_vm_reference_plan_key(&plan, i), expected[i]);
    ExpectKeyEqual(
        loom_vm_reference_plan_key(
            &plan, loom_vm_reference_plan_ordinal(&plan, ordinals[i])),
        &keys_[i]);
  }
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(VMReferencePlanTest, SourceNamesRetainTheirFullWidth) {
  ASSERT_NO_FATAL_FAILURE(CreateTypes(2, 2));
  const loom_type_t first = InternType(0);
  const uint32_t first_name = loom_type_dialect_name_id(first);
  while (module_->strings.count < first_name + 65536u) {
    const std::string name =
        "unrelated_" + std::to_string(module_->strings.count);
    loom_string_id_t unused = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, {name.data(), name.size()}, &unused));
  }
  const loom_type_t second = InternType(1);
  ASSERT_EQ(loom_type_dialect_name_id(second) - first_name, 65536u);
  loom_vm_reference_plan_t plan = {};
  uint16_t first_ordinal = 0, second_ordinal = 0, buffer_ordinal = 0;
  IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, first, &arena_,
                                             &first_ordinal));
  IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, second, &arena_,
                                             &second_ordinal));
  IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, loom_type_buffer(),
                                             &arena_, &buffer_ordinal));
  EXPECT_NE(first_ordinal, second_ordinal);
  EXPECT_NE(first_ordinal, buffer_ordinal);
  EXPECT_NE(second_ordinal, buffer_ordinal);
  EXPECT_EQ(plan.count, 3u);
  EXPECT_EQ(arena_.used_allocation_size, 1024u);
}

TEST_F(VMReferencePlanTest, AllocationFailureAtEachStorageGrowthBoundary) {
  ASSERT_NO_FATAL_FAILURE(CreateTypes(769, 769));
  std::vector<loom_type_t> types;
  for (uint32_t i = 0; i < names_.size(); ++i) {
    types.push_back(InternType(i));
  }
  // Initial bucket, first key chunk, bucket growth, and both directories.
  for (uint32_t prior_count : {0u, 8u, 48u, 520u, 768u}) {
    SCOPED_TRACE(prior_count);
    loom_vm_reference_plan_t plan = {};
    uint16_t ordinal = 0;
    for (uint32_t i = 0; i < prior_count; ++i) {
      IREE_ASSERT_OK(loom_vm_reference_plan_bind(&plan, module_, types[i],
                                                 &arena_, &ordinal));
    }
    // Consume the current block's usable tail and release cached free blocks,
    // making the next pooled allocation reach the failing system allocator.
    const auto remaining =
        arena_.block_bytes_remaining & ~(iree_max_align_t - 1);
    if (remaining) {
      void* padding = nullptr;
      IREE_ASSERT_OK(iree_arena_allocate(&arena_, remaining, &padding));
    }
    iree_arena_block_pool_trim(&pool_);
    fail_allocations_ = true;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_vm_reference_plan_bind(&plan, module_, types[prior_count], &arena_,
                                    &ordinal));
    fail_allocations_ = false;
    // A failed plan is discarded; a fresh invocation reuses the same pool.
    iree_arena_reset(&arena_);
  }
}

}  // namespace
