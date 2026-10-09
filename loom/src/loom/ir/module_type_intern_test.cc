// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/types.h"

namespace loom {
namespace {

class TypeImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(65536, iree_allocator_system(), &pool_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("types"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  struct Pair {
    // Temporary ABI payload with two argument slots.
    alignas(loom_func_type_data_t) uint8_t
        storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};

    loom_type_t Set(loom_type_t child) {
      auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
      data->arg_count = 2;
      data->types[0] = data->types[1] = child;
      return loom_type_function(data);
    }
  };

  void ExpectSharedChain(loom_type_t type, size_t depth) {
    for (size_t i = 0; i < depth; ++i) {
      const auto* data = loom_type_func_data(type);
      ASSERT_NE(data, nullptr);
      ASSERT_EQ(data->arg_count, 2);
      ASSERT_EQ(data->result_count, 0);
      EXPECT_EQ(
          std::memcmp(&data->types[0], &data->types[1], sizeof(loom_type_t)),
          0);
      type = data->types[0];
    }
    EXPECT_EQ(loom_type_kind(type), LOOM_TYPE_SCALAR);
  }

  // Minimal production type context, with no text/parser dependency.
  loom_context_t context_ = {};
  // Shared pool owns both persistent module and temporary import blocks.
  iree_arena_block_pool_t pool_ = {};
  // Canonical type owner under test.
  loom_module_t* module_ = nullptr;
};

TEST_F(TypeImportTest, DeepTemporaryGraphRetainsCanonicalChildren) {
  std::array<Pair, 2048> temporary;
  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  for (auto& pair : temporary) {
    type = pair.Set(type);
  }
  const auto used_before = module_->arena.used_allocation_size;
  loom_type_id_t root;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &root));
  EXPECT_EQ(module_->types.count, temporary.size() + 1);
  ASSERT_NO_FATAL_FAILURE(ExpectSharedChain(
      loom_type_table_get(&module_->types, root), temporary.size()));
  EXPECT_LT(module_->arena.used_allocation_size - used_before,
            256 * temporary.size());
  const auto used_after = module_->arena.used_allocation_size;
  loom_type_id_t duplicate;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &duplicate));
  EXPECT_EQ(duplicate, root);
  EXPECT_EQ(module_->arena.used_allocation_size, used_after);
}

TEST_F(TypeImportTest, CanonicalIdentitySurvivesRecentCacheAndTableGrowth) {
  std::array<loom_type_t, 96> roots;
  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  for (auto& root : roots) {
    const loom_type_t children[] = {type, type};
    IREE_ASSERT_OK(loom_module_intern_function_type(module_, children, 2,
                                                    nullptr, 0, &root));
    type = root;
  }
  const auto used = module_->arena.used_allocation_size;
  for (size_t i = 0; i < roots.size(); ++i) {
    loom_type_id_t id;
    IREE_ASSERT_OK(loom_module_intern_type_id(module_, roots[i], &id));
    EXPECT_EQ(id, i + 1);
  }
  EXPECT_EQ(module_->arena.used_allocation_size, used);
  ASSERT_NO_FATAL_FAILURE(ExpectSharedChain(type, roots.size()));
}

TEST_F(TypeImportTest, StableRowsAndFactsAcrossChunkAndDirectoryGrowth) {
  EXPECT_EQ(module_->types.count, 0u);
  EXPECT_EQ(loom_type_table_capacity(&module_->types), 0u);
  const auto used_before = module_->arena.used_allocation_size;
  loom_type_id_t scalar_id;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &scalar_id));
  ASSERT_EQ(scalar_id, 0u);
  EXPECT_EQ(module_->arena.used_allocation_size - used_before,
            sizeof(loom_type_segment_t));

  // Borrow rows from every inline segment, then grow both the directory and
  // another segment inside its already-published primary page.
  constexpr uint32_t kSegmentCount =
      LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT + 2;
  std::array<const loom_type_t*, kSegmentCount> rows = {};
  rows[0] = loom_type_table_entry(&module_->types, scalar_id);
  for (uint32_t id = 1; id < kSegmentCount * LOOM_TYPE_SEGMENT_CAPACITY; ++id) {
    const auto type = loom_type_shaped_2d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(id),
        loom_dim_pack_static(4), /*encoding_id=*/0);
    loom_type_id_t actual;
    IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &actual));
    ASSERT_EQ(actual, id);
    EXPECT_EQ(loom_type_table_hash(&module_->types, id), loom_type_hash(type));
    EXPECT_EQ(loom_type_table_dependencies(&module_->types, id), 0u);
    if (id % LOOM_TYPE_SEGMENT_CAPACITY == 0) {
      rows[id / LOOM_TYPE_SEGMENT_CAPACITY] =
          loom_type_table_entry(&module_->types, id);
    }
  }
  EXPECT_EQ(module_->types.segments.segment_count, kSegmentCount);
  ASSERT_NE(module_->types.segments.primary_page, nullptr);
  const auto used_after = module_->arena.used_allocation_size;
  for (uint32_t segment = 0; segment < kSegmentCount; ++segment) {
    const auto id = segment * LOOM_TYPE_SEGMENT_CAPACITY;
    EXPECT_EQ(loom_type_table_entry(&module_->types, id), rows[segment]);
    loom_type_id_t duplicate;
    IREE_ASSERT_OK(
        loom_module_intern_type_id(module_, *rows[segment], &duplicate));
    EXPECT_EQ(duplicate, id);
  }
  EXPECT_EQ(module_->arena.used_allocation_size, used_after);
}

TEST_F(TypeImportTest, FunctionResultAndDialectChildrenAreCanonical) {
  Pair argument;
  Pair result;
  const auto argument_type =
      argument.Set(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  const auto result_type = result.Set(loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  loom_type_t function;
  IREE_ASSERT_OK(loom_module_intern_function_type(module_, &argument_type, 1,
                                                  &result_type, 1, &function));
  const auto* data = loom_type_func_data(function);
  ASSERT_EQ(data->arg_count, 1);
  ASSERT_EQ(data->result_count, 1);
  loom_type_id_t argument_id;
  loom_type_id_t result_id;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module_, argument_type, &argument_id));
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, result_type, &result_id));
  EXPECT_EQ(data->types[0].dims[0],
            loom_type_table_get(&module_->types, argument_id).dims[0]);
  EXPECT_EQ(data->types[1].dims[0],
            loom_type_table_get(&module_->types, result_id).dims[0]);
  loom_string_id_t name;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("test.pair"), &name));
  const loom_type_t children[] = {argument_type, result_type};
  loom_type_t dialect;
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, loom_type_dialect(name, 2, children), &dialect));
  EXPECT_EQ(loom_type_dialect_params(dialect)[0].dims[0],
            data->types[0].dims[0]);
  EXPECT_EQ(loom_type_dialect_params(dialect)[1].dims[0],
            data->types[1].dims[0]);
  loom_type_id_t function_id;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, function, &function_id));
  EXPECT_EQ(loom_type_table_hash(&module_->types, function_id),
            loom_type_hash(function));
}

TEST_F(TypeImportTest, WideSignatureSharesImportsAcrossArgumentAndResultSpans) {
  Pair integer;
  Pair floating;
  const loom_type_t children[] = {
      integer.Set(loom_type_scalar(LOOM_SCALAR_TYPE_I32)),
      floating.Set(loom_type_scalar(LOOM_SCALAR_TYPE_F32)),
  };
  std::array<loom_type_t, 33> arguments;
  std::array<loom_type_t, 17> results;
  for (size_t i = 0; i < arguments.size(); ++i) {
    arguments[i] = children[i % 2];
  }
  for (size_t i = 0; i < results.size(); ++i) {
    results[i] = children[(i + 1) % 2];
  }
  loom_type_t function;
  IREE_ASSERT_OK(loom_module_intern_function_type(
      module_, arguments.data(), arguments.size(), results.data(),
      results.size(), &function));
  EXPECT_EQ(module_->types.count, 5u);
  const auto* data = loom_type_func_data(function);
  ASSERT_EQ(data->arg_count, arguments.size());
  ASSERT_EQ(data->result_count, results.size());
  EXPECT_NE(data->types[0].dims[0], children[0].dims[0]);
  EXPECT_NE(data->types[1].dims[0], children[1].dims[0]);
  for (size_t i = 0; i < arguments.size(); ++i) {
    EXPECT_EQ(data->types[i].dims[0], data->types[i % 2].dims[0]);
  }
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_EQ(data->types[arguments.size() + i].dims[0],
              data->types[(i + 1) % 2].dims[0]);
  }
  const auto used_bytes = module_->arena.used_allocation_size;
  loom_type_t duplicate;
  IREE_ASSERT_OK(loom_module_intern_function_type(
      module_, arguments.data(), arguments.size(), results.data(),
      results.size(), &duplicate));
  EXPECT_EQ(duplicate.dims[0], function.dims[0]);
  EXPECT_EQ(module_->arena.used_allocation_size, used_bytes);
}

TEST_F(TypeImportTest, RegisterCacheDoesNotExpandSharedSemanticTypes) {
  std::array<Pair, 48> temporary;
  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  for (auto& pair : temporary) {
    type = pair.Set(type);
  }
  loom_type_t canonical;
  IREE_ASSERT_OK(loom_module_intern_type(module_, type, &canonical));
  loom_type_t first;
  loom_type_t second;
  IREE_ASSERT_OK(
      loom_module_intern_register_type(module_, 1, 2, canonical, &first));
  IREE_ASSERT_OK(
      loom_module_intern_register_type(module_, 1, 2, type, &second));
  EXPECT_EQ(first.dims[0], second.dims[0]);
  EXPECT_EQ(loom_type_register_value_type(first)->dims[0], canonical.dims[0]);
}

TEST_F(TypeImportTest, TemporaryPayloadIdentityExpiresAtCallBoundary) {
  Pair temporary;
  loom_type_t first;
  loom_type_t second;
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, temporary.Set(loom_type_scalar(LOOM_SCALAR_TYPE_I32)), &first));
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, temporary.Set(loom_type_scalar(LOOM_SCALAR_TYPE_F32)), &second));
  EXPECT_NE(first.dims[0], second.dims[0]);
  EXPECT_EQ(loom_type_element_type(loom_type_func_data(first)->types[0]),
            LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(loom_type_element_type(loom_type_func_data(second)->types[0]),
            LOOM_SCALAR_TYPE_F32);
}

TEST_F(TypeImportTest, TemporaryMemoDistinguishesNamesSharingOneChildArray) {
  loom_string_id_t first_name;
  loom_string_id_t second_name;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("test.first"), &first_name));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("test.second"), &second_name));
  Pair child;
  const loom_type_t parameters[] = {
      child.Set(loom_type_scalar(LOOM_SCALAR_TYPE_F32))};
  const loom_type_t arguments[] = {
      loom_type_dialect(first_name, 1, parameters),
      loom_type_dialect(second_name, 1, parameters),
      loom_type_dialect(first_name, 1, parameters)};
  loom_type_t function;
  IREE_ASSERT_OK(loom_module_intern_function_type(module_, arguments, 3,
                                                  nullptr, 0, &function));
  const auto* data = loom_type_func_data(function);
  EXPECT_EQ(data->types[0].dims[0], data->types[2].dims[0]);
  EXPECT_NE(data->types[0].dims[0], data->types[1].dims[0]);
  EXPECT_EQ(loom_type_dialect_name_id(data->types[0]), first_name);
  EXPECT_EQ(loom_type_dialect_name_id(data->types[1]), second_name);
  EXPECT_EQ(loom_type_dialect_params(data->types[0])[0].dims[0],
            loom_type_dialect_params(data->types[1])[0].dims[0]);
}

TEST_F(TypeImportTest, SharedPayloadWithChangedOuterTypeGetsDistinctIdentity) {
  const loom_overflow_dim_t dimensions[] = {1, 2, 3};
  loom_type_t type = {};  // NOLINT(iree-cpp-designated-initializer) --
                          // Assignment sequencing spans intervening work.
  type.header = loom_type_make_header(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 3,
                                      LOOM_TYPE_FLAG_ALL_STATIC);
  type.dims[0] = reinterpret_cast<uintptr_t>(dimensions);
  loom_type_id_t tile;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &tile));
  type = loom_type_table_get(&module_->types, tile);
  type.header = loom_type_make_header(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, 3,
                                      LOOM_TYPE_FLAG_ALL_STATIC);
  loom_type_id_t tensor;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &tensor));
  EXPECT_NE(tile, tensor);
  EXPECT_NE(loom_type_table_get(&module_->types, tile).dims[0],
            loom_type_table_get(&module_->types, tensor).dims[0]);
  EXPECT_EQ(loom_type_table_hash(&module_->types, tensor),
            loom_type_hash(type));
}

TEST_F(TypeImportTest, ImportedForeignPayloadSurvivesSourceDestruction) {
  loom_module_t* source = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"), &pool_,
                                      nullptr, iree_allocator_system(),
                                      &source));
  const loom_type_t scalar = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t child;
  IREE_ASSERT_OK(
      loom_module_intern_function_type(source, &scalar, 1, &scalar, 1, &child));
  const loom_type_t children[] = {child, child};
  loom_type_t root;
  IREE_ASSERT_OK(
      loom_module_intern_function_type(source, children, 2, nullptr, 0, &root));
  loom_type_t imported;
  IREE_ASSERT_OK(loom_module_intern_type(module_, root, &imported));
  EXPECT_NE(imported.dims[0], root.dims[0]);
  EXPECT_NE(loom_type_func_data(imported)->types[0].dims[0], child.dims[0]);
  loom_module_free(source);
  const auto* imported_child =
      loom_type_func_data(loom_type_func_data(imported)->types[0]);
  EXPECT_EQ(imported_child->arg_count, 1);
  EXPECT_EQ(imported_child->result_count, 1);
  EXPECT_TRUE(loom_type_equal(imported_child->types[0], scalar));
  loom_type_id_t id;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, imported, &id));
  EXPECT_EQ(loom_type_table_get(&module_->types, id).dims[0], imported.dims[0]);
}

class PooledTypeRowsTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<PooledTypeRowsTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      const auto* allocation =
          static_cast<const iree_allocator_alloc_params_t*>(parameters);
      EXPECT_EQ(allocation->byte_length, test->GetParam());
      ++test->allocation_count_;
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(GetParam(), {this, Allocate}, &pool_);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  // Backing requests observed independently of optional pool statistics.
  iree_host_size_t allocation_count_ = 0;
  // Pool retained across successive module lifetimes.
  iree_arena_block_pool_t pool_ = {};
  // Minimal context used by the real type interner.
  loom_context_t context_ = {};
  // Current canonical type owner, or null between invocations.
  loom_module_t* module_ = nullptr;
};

TEST_P(PooledTypeRowsTest, WarmConstructionUsesOnlyRecycledFixedBlocks) {
  iree_host_size_t warm_allocation_count = 0;
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("pooled_types"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    // Both canonical rows and hash buckets exceed one block; every allocation
    // still comes from the pool and is reusable by the next module.
    constexpr uint32_t kDimensionCount = 8192;
    for (uint32_t dimension = 1; dimension <= kDimensionCount; ++dimension) {
      const auto type = loom_type_shaped_2d(
          LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
          loom_dim_pack_static(dimension), loom_dim_pack_static(4),
          /*encoding_id=*/0);
      loom_type_id_t type_id;
      IREE_ASSERT_OK(loom_module_intern_type_id(module_, type, &type_id));
      EXPECT_EQ(type_id, dimension);
    }
    EXPECT_EQ(module_->types.count, kDimensionCount + 1);
    EXPECT_EQ(module_->types.segments.segment_count,
              (kDimensionCount + LOOM_TYPE_SEGMENT_CAPACITY) /
                  LOOM_TYPE_SEGMENT_CAPACITY);
    if (iteration == 0) {
      ASSERT_GT(allocation_count_, 0u);
      warm_allocation_count = allocation_count_;
    } else {
      EXPECT_EQ(allocation_count_, warm_allocation_count);
    }
    loom_module_free(module_);
    module_ = nullptr;
  }
}

INSTANTIATE_TEST_SUITE_P(PoolSize, PooledTypeRowsTest,
                         ::testing::Values(32768, 131072));

struct TypeGrowthBoundary {
  // Number of unique types inserted before fault injection.
  iree_host_size_t type_count;
  // Expected row capacity at the boundary, before the next insertion.
  iree_host_size_t row_capacity;
  // Expected hash capacity at the boundary, before the next insertion.
  iree_host_size_t hash_capacity;
};

class TypeInternerFailureTest
    : public ::testing::TestWithParam<TypeGrowthBoundary> {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<TypeInternerFailureTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      if (test->allocation_count_++ == test->failure_index_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(128, {this, Allocate}, &pool_);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  void PrepareBoundary() {
    failure_index_ = SIZE_MAX;
    loom_module_free(module_);
    module_ = nullptr;
    // Each attempt starts with the same empty pool, not blocks cached by the
    // previous attempt's rollback or successful retry.
    iree_arena_block_pool_trim(&pool_);
    loom_module_size_hints_t hints = {};
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("type_growth"),
                                        &pool_, &hints, iree_allocator_system(),
                                        &module_));
    loom_type_id_t element_id = LOOM_TYPE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &element_id));
    ASSERT_EQ(element_id, 0u);
    for (iree_host_size_t alignment = 1; alignment < GetParam().type_count;
         ++alignment) {
      loom_type_t type = {};
      loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
      IREE_ASSERT_OK(MakeType(alignment, &type, &type_id));
      ASSERT_EQ(type_id, alignment);
    }
    ASSERT_EQ(loom_type_table_capacity(&module_->types),
              GetParam().row_capacity);
    ASSERT_EQ(module_->type_intern.capacity, GetParam().hash_capacity);
    allocation_count_ = 0;
  }

  iree_status_t MakeType(iree_host_size_t alignment, loom_type_t* out_type,
                         loom_type_id_t* out_type_id) {
    const loom_attribute_t parameters[] = {
        loom_attr_type(0), loom_attr_i64(alignment), loom_attr_absent()};
    return loom_module_make_parameterized_type(
        module_, &loom_test_array_type_parameterized_descriptor, parameters,
        IREE_ARRAYSIZE(parameters), out_type, out_type_id);
  }

  void ExpectTypeStorageUnchanged(const loom_type_table_t& types) {
    EXPECT_EQ(module_->types.count, types.count);
    EXPECT_EQ(module_->types.segments.segment_count,
              types.segments.segment_count);
    EXPECT_EQ(module_->types.segments.primary_page,
              types.segments.primary_page);
    EXPECT_EQ(module_->types.segments.page_directory,
              types.segments.page_directory);
    for (uint32_t i = 0; i < types.segments.segment_count; ++i) {
      EXPECT_EQ(
          loom_segmented_storage_const_segment(&module_->types.segments, i),
          loom_segmented_storage_const_segment(&types.segments, i));
    }
  }

  void ExpectInternerStorageUnchanged(const loom_intern_table_t& interner) {
    EXPECT_EQ(module_->type_intern.count, interner.count);
    EXPECT_EQ(module_->type_intern.capacity, interner.capacity);
    EXPECT_EQ(module_->type_intern.segments.segment_count,
              interner.segments.segment_count);
    EXPECT_EQ(module_->type_intern.segments.primary_page,
              interner.segments.primary_page);
    EXPECT_EQ(module_->type_intern.segments.page_directory,
              interner.segments.page_directory);
    for (uint32_t i = 0; i < interner.segments.segment_count; ++i) {
      EXPECT_EQ(loom_segmented_storage_const_segment(
                    &module_->type_intern.segments, i),
                loom_segmented_storage_const_segment(&interner.segments, i));
    }
  }

  // Backing allocation ordinal to fail, or SIZE_MAX when failure is disabled.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Backing allocation requests since the latest preparation completed.
  iree_host_size_t allocation_count_ = 0;
  // Small blocks expose table allocations to the fault-injecting allocator.
  iree_arena_block_pool_t pool_ = {};
  // Minimal context for type construction through the production module API.
  loom_context_t context_ = {};
  // Module owning the canonical rows, payloads, and interner under test.
  loom_module_t* module_ = nullptr;
};

TEST_P(TypeInternerFailureTest,
       GrowthFailurePreservesCanonicalTypesAndRetries) {
  ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
  loom_type_t type = {};
  loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(MakeType(GetParam().type_count, &type, &type_id));
  // Logical bucket growth within one chunk may need no backing allocation.
  // A new row or bucket chunk always allocates with this deliberately tiny
  // pool.
  if (GetParam().type_count == GetParam().row_capacity ||
      GetParam().hash_capacity >= LOOM_INTERN_SEGMENT_CAPACITY) {
    ASSERT_GT(allocation_count_, 0u);
  }

  // Page-directory growth depends on the backing allocator's address layout.
  // Exhaust failure ordinals until a complete attempt makes no failing call.
  for (iree_host_size_t failure_index = 0;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
    const loom_type_table_t types = module_->types;
    const loom_intern_table_t interner = module_->type_intern;
    const auto identity = module_->type_identity;
    const iree_host_size_t used_bytes = module_->arena.used_allocation_size;
    const iree_host_size_t owned_bytes = module_->arena.total_allocation_size;
    failure_index_ = failure_index;
    iree_status_t status = MakeType(GetParam().type_count, &type, &type_id);
    failure_index_ = SIZE_MAX;
    if (iree_status_is_ok(status)) {
      EXPECT_LE(allocation_count_, failure_index);
      EXPECT_EQ(type_id, GetParam().type_count);
      EXPECT_EQ(module_->types.count, types.count + 1);
      break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(allocation_count_, failure_index + 1);

    ExpectTypeStorageUnchanged(types);
    ExpectInternerStorageUnchanged(interner);
    EXPECT_EQ(module_->type_identity.root, identity.root);
    EXPECT_EQ(module_->type_identity.recent_page, identity.recent_page);
    EXPECT_EQ(module_->arena.used_allocation_size, used_bytes);
    EXPECT_EQ(module_->arena.total_allocation_size, owned_bytes);

    for (iree_host_size_t alignment = 1; alignment < types.count; ++alignment) {
      IREE_ASSERT_OK(MakeType(alignment, &type, &type_id));
      EXPECT_EQ(type_id, alignment);
      EXPECT_EQ(loom_test_array_type_alignment(type), alignment);
      EXPECT_EQ(loom_type_parameterized_parameters(type),
                loom_type_parameterized_parameters(
                    loom_type_table_get(&types, type_id)));
      EXPECT_EQ(loom_type_table_hash(&module_->types, type_id),
                loom_type_hash(type));
      loom_type_id_t canonical_id;
      IREE_ASSERT_OK(loom_module_intern_type_id(
          module_, loom_type_table_get(&types, type_id), &canonical_id));
      EXPECT_EQ(canonical_id, type_id);
    }
    IREE_ASSERT_OK(MakeType(GetParam().type_count, &type, &type_id));
    EXPECT_EQ(type_id, GetParam().type_count);
    EXPECT_EQ(module_->types.count, types.count + 1);
    EXPECT_EQ(module_->type_intern.count, interner.count + 1);
  }
}

TEST_P(TypeInternerFailureTest, InvalidParameterRollsBackAndRetries) {
  for (iree_host_size_t parameter_index = 0; parameter_index < 3;
       ++parameter_index) {
    SCOPED_TRACE(parameter_index);
    ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
    const iree_host_size_t used_bytes = module_->arena.used_allocation_size;
    const iree_host_size_t owned_bytes = module_->arena.total_allocation_size;
    const loom_type_table_t types = module_->types;
    const loom_intern_table_t interner = module_->type_intern;
    loom_attribute_t parameters[] = {loom_attr_type(0),
                                     loom_attr_i64(GetParam().type_count),
                                     loom_attr_absent()};
    // Each slot has a non-string descriptor. Later failures follow one or two
    // successful canonicalizations before the shared payload builder returns.
    parameters[parameter_index] = loom_attr_string(LOOM_STRING_ID_INVALID);
    loom_type_t type = {};
    loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_module_make_parameterized_type(
            module_, &loom_test_array_type_parameterized_descriptor, parameters,
            IREE_ARRAYSIZE(parameters), &type, &type_id));
    EXPECT_EQ(module_->arena.used_allocation_size, used_bytes);
    EXPECT_EQ(module_->arena.total_allocation_size, owned_bytes);
    ExpectTypeStorageUnchanged(types);
    ExpectInternerStorageUnchanged(interner);
    IREE_ASSERT_OK(MakeType(GetParam().type_count, &type, &type_id));
    EXPECT_EQ(type_id, GetParam().type_count);
  }
}

TEST_P(TypeInternerFailureTest,
       ImportFailureRetainsCompleteChildrenAndRetries) {
  std::array<loom_register_type_data_t, 32> payloads;
  loom_type_t source = loom_type_scalar(LOOM_SCALAR_TYPE_BF16);
  for (size_t i = 0; i < payloads.size(); ++i) {
    payloads[i] = {i + 1, 4, source};
    source = loom_type_register_payload_with_value_type(&payloads[i]);
  }
  ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
  loom_type_id_t root;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, source, &root));
  ASSERT_GT(allocation_count_, 0u);
  for (size_t failure = 0;; ++failure) {
    SCOPED_TRACE(failure);
    ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
    const auto initial_count = module_->types.count;
    failure_index_ = failure;
    iree_status_t status = loom_module_intern_type_id(module_, source, &root);
    failure_index_ = SIZE_MAX;
    if (iree_status_is_ok(status)) {
      EXPECT_LE(allocation_count_, failure);
      EXPECT_EQ(module_->types.count, initial_count + payloads.size());
      EXPECT_EQ(root, module_->types.count - 1);
      break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(allocation_count_, failure + 1);
    EXPECT_EQ(root, LOOM_TYPE_ID_INVALID);
    // Completed children may remain canonical after a parent fails. Every
    // published identity must still resolve after temporary scratch is
    // released.
    const auto published_count = module_->types.count;
    const auto used_bytes = module_->arena.used_allocation_size;
    for (loom_type_id_t id = 0; id < published_count; ++id) {
      loom_type_id_t actual;
      IREE_ASSERT_OK(loom_module_intern_type_id(
          module_, loom_type_table_get(&module_->types, id), &actual));
      EXPECT_EQ(actual, id);
    }
    EXPECT_EQ(module_->types.count, published_count);
    EXPECT_EQ(module_->arena.used_allocation_size, used_bytes);
    IREE_ASSERT_OK(loom_module_intern_type_id(module_, source, &root));
    EXPECT_EQ(module_->types.count, initial_count + payloads.size());
    EXPECT_EQ(root, module_->types.count - 1);
  }
}

TEST_P(TypeInternerFailureTest, GenericInternOwnsCanonicalParameterPayload) {
  ASSERT_NO_FATAL_FAILURE(PrepareBoundary());
  loom_attribute_t parameters[] = {loom_attr_type(0),
                                   loom_attr_i64(GetParam().type_count),
                                   loom_attr_absent()};
  const loom_type_t temporary_type =
      loom_type_parameterized(&loom_test_array_type_parameterized_descriptor,
                              IREE_ARRAYSIZE(parameters), parameters);
  loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, temporary_type, &type_id));
  ASSERT_EQ(type_id, GetParam().type_count);
  const loom_type_t canonical_type =
      loom_type_table_get(&module_->types, type_id);
  EXPECT_NE(loom_type_parameterized_parameters(canonical_type), parameters);
  parameters[1] = loom_attr_i64(0);
  EXPECT_EQ(loom_test_array_type_alignment(canonical_type),
            GetParam().type_count);
  const iree_host_size_t used_bytes = module_->arena.used_allocation_size;
  loom_type_t duplicate_type = {};
  loom_type_id_t duplicate_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      MakeType(GetParam().type_count, &duplicate_type, &duplicate_id));
  EXPECT_EQ(duplicate_id, type_id);
  EXPECT_EQ(loom_type_parameterized_parameters(duplicate_type),
            loom_type_parameterized_parameters(canonical_type));
  EXPECT_EQ(module_->arena.used_allocation_size, used_bytes);
}

// Bucket capacity before the interner needs its first shared pointer page.
constexpr iree_host_size_t kBucketDirectoryCapacity =
    LOOM_INTERN_SEGMENT_CAPACITY * LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT;

// Exercise row-only, logical hash-only, and simultaneous row/hash growth
// through distinct type construction, without altering table metadata. The last
// case promotes the bucket directory while appending to shared row storage.
INSTANTIATE_TEST_SUITE_P(TypeGrowth, TypeInternerFailureTest,
                         ::testing::Values(TypeGrowthBoundary{32, 32, 64},
                                           TypeGrowthBoundary{48, 64, 64},
                                           TypeGrowthBoundary{96, 96, 128},
                                           TypeGrowthBoundary{
                                               kBucketDirectoryCapacity * 3 / 4,
                                               kBucketDirectoryCapacity * 3 / 4,
                                               kBucketDirectoryCapacity}));

}  // namespace
}  // namespace loom
