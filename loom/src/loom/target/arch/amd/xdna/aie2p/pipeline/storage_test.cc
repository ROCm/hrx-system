// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/storage.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/types.h"
#include "loom/util/fact_extensions.h"

namespace {

class NativeStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    Register(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    Register(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("storage"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    realization_.worker_versions = versions_;
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.pass = &pass_;
    pass_.arena = &arena_;
    owner_.coordinate = {0, 2};
    owner_.facts =
        loom_xdna_array_tile_facts(context_.family, owner_.coordinate);
    context_.pool_tiles = pools_;
    consumer_ = owner_;
    worker_.tile = &consumer_;

    loom_string_id_t name, strides_name;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, IREE_SV("encoding.layout.strided"), &name));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("strides"), &strides_name));
    int64_t strides[] = {4, 1};
    const loom_named_attr_t parameter = {
        .name_id = strides_name,
        .value = loom_attr_i64_array(strides, 2),
    };
    const loom_encoding_t encoding = {
        .name_id = name,
        .attribute_count = 1,
        .attributes = &parameter,
    };
    uint16_t encoding_id;
    IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));
    types_[0] = loom_type_buffer();
    types_[1] = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                    loom_dim_pack_static(16), 0);
    types_[2] = loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                    loom_dim_pack_static(2),
                                    loom_dim_pack_static(2), encoding_id);
    types_[3] = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I16,
                                    loom_dim_pack_static(1), 0);
    types_[4] = types_[5] = types_[1];
    source_.function = Function(&source_return_);
    uint16_t count;
    arguments_ = loom_func_like_arg_ids(source_.function, &count);
    allocation_ = {
        .root_value_id = arguments_[0],
        .pool_index = 0,
        .byte_length = 64,
        .byte_alignment = 64,
    };
    placement_ = {.source = &allocation_, .byte_offset = 128};
    binding_ = {arguments_[0], &placement_};
    invocation_.allocations = &binding_;
    invocation_.allocation_count = 1;
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&source_.facts, &arena_,
                                                    module_->values.count));
    loom_type_registry_configure_fact_context(&source_.facts.context);
    const loom_value_fact_buffer_reference_t buffer = {
        .maximum_byte_extent = loom_value_facts_exact_i64(64),
        .minimum_alignment = 64,
        .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
        .root_value_id = arguments_[0],
    };
    loom_value_facts_t facts;
    IREE_ASSERT_OK(loom_value_facts_make_buffer_reference(
        &source_.facts.context, buffer, &facts));
    IREE_ASSERT_OK(
        loom_value_fact_table_define(&source_.facts, arguments_[0], facts));
    const int64_t offsets[] = {0, 20, 6, 0};
    const int64_t lengths[] = {64, 24, 2, 64};
    const uint64_t alignments[] = {UINT64_MAX, 4, 2, UINT64_MAX};
    for (size_t i = 1; i < 5; ++i) {
      const int16_t static_element_byte_count = i == 3 ? 2 : 4;
      const uint8_t memory_space = i == 4
                                       ? LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL
                                       : LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
      const loom_value_fact_view_reference_t view = {
          .base_byte_offset = loom_value_facts_exact_i64(offsets[i - 1]),
          .footprint_byte_length = loom_value_facts_exact_i64(lengths[i - 1]),
          .minimum_alignment = alignments[i - 1],
          .root_minimum_alignment = 64,
          .root_value_id = i == 4 ? arguments_[4] : arguments_[0],
          .buffer_value_id = i == 4 ? LOOM_VALUE_ID_INVALID : arguments_[0],
          .static_element_byte_count = static_element_byte_count,
          .memory_space = memory_space,
      };
      IREE_ASSERT_OK(loom_value_facts_make_view_reference(
          &source_.facts.context, view, &facts));
      IREE_ASSERT_OK(
          loom_value_fact_table_define(&source_.facts, arguments_[i], facts));
    }
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* tables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&registry_, dialect, tables, count));
  }

  loom_func_like_t Function(loom_op_t** out_return) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_symbol_ref_t symbol;
    IREE_CHECK_OK(loom_aie2p_worker_symbol(&context_.code, &symbol));
    loom_op_t* function;
    // Returning five formals gives each a real use; the sixth is deliberately
    // unused and requires neither reference facts nor physical placement.
    IREE_CHECK_OK(loom_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, types_,
        6, types_, 5, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function));
    loom_builder_enter_region(&builder, function, loom_func_def_body(function));
    const auto* entry = loom_region_entry_block(loom_func_def_body(function));
    IREE_CHECK_OK(loom_func_return_build(&builder, entry->arg_ids, 5,
                                         LOOM_LOCATION_UNKNOWN, out_return));
    return loom_func_like_cast(module_, function);
  }

  void Select() {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_select_storage_accesses(
        &context_, &invocation_, &source_, &worker_, &valid));
    ASSERT_TRUE(valid);
    ASSERT_EQ(worker_.storage.count, 4u);
  }

  // IR and retained compiler products share the test's arena backing.
  iree_arena_block_pool_t blocks_;
  // Per-test plan storage.
  iree_arena_allocator_t arena_;
  // Dialects needed by ordinary typed capture reconstruction.
  loom_context_t registry_;
  // Owns source and consuming worker functions.
  loom_module_t* module_ = nullptr;
  // Rewrites only the consuming function.
  loom_rewriter_t rewriter_;
  // Allocation context used by production selection.
  loom_pass_t pass_ = {};
  // Native selection and helper emission state.
  loom_aie2p_native_context_t context_ = {};
  // Helper registration has no optional function versions in this fixture.
  loom_pipeline_realization_t realization_ = {};
  // One nullable function-version entry for the consuming worker.
  loom_target_function_version_t* versions_[1] = {};
  // Physical owner of the shared allocation.
  loom_aie2p_native_tile_t owner_ = {};
  // Local or adjacent consuming core.
  loom_aie2p_native_tile_t consumer_ = {};
  // Native pool-to-tile inventory.
  loom_aie2p_native_tile_t* pools_[1] = {&owner_};
  // Selected capture records.
  loom_aie2p_native_worker_t worker_ = {};
  // Shared owner of source arguments and already inferred facts.
  loom_pipeline_worker_t source_ = {};
  // Buffer, dense/strided/subword views, external view, and dead view.
  loom_type_t types_[6] = {};
  // Original immutable formal identities.
  const loom_value_id_t* arguments_ = nullptr;
  // Source uses must remain untouched by consuming materialization.
  loom_op_t* source_return_ = nullptr;
  // Source-owned allocation extent and alignment.
  loom_pipeline_resource_allocation_t allocation_ = {};
  // Final placement includes a nonzero offset in its pool.
  loom_pipeline_storage_allocation_t placement_ = {};
  // Indexed root-to-placement binding.
  loom_pipeline_invocation_allocation_t binding_ = {};
  // Construction occurrence owns the root binding.
  loom_pipeline_invocation_t invocation_ = {};
};

TEST_F(NativeStorageTest, LocalAliasesRetainDisplacementAndAlignment) {
  Select();
  const uint32_t offsets[] = {0, 0, 20, 6};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(worker_.storage.accesses[i].argument, arguments_[i]);
    EXPECT_EQ(worker_.storage.accesses[i].address, 0x70000u + 128);
    EXPECT_EQ(worker_.storage.accesses[i].alignment, 64u);
    EXPECT_EQ(worker_.storage.accesses[i].byte_offset, offsets[i]);
  }
}

TEST_F(NativeStorageTest, NeighborChangesApertureWithoutMovingStorage) {
  consumer_.coordinate = {0, 3};
  Select();
  EXPECT_EQ(worker_.storage.accesses[0].address, 0x40000u + 128);
  EXPECT_EQ(worker_.storage.accesses[2].address, 0x40000u + 128);
  EXPECT_EQ(worker_.storage.accesses[2].byte_offset, 20u);
  EXPECT_EQ(worker_.storage.accesses[3].address, 0x40000u + 128);
  EXPECT_EQ(worker_.storage.accesses[3].byte_offset, 6u);
  EXPECT_EQ(worker_.storage.accesses[3].alignment, 64u);
}

TEST_F(NativeStorageTest, ReconstructionPreservesTypesAndOnlyRewritesInstance) {
  Select();
  loom_op_t* result;
  const auto function = Function(&result);
  uint16_t count;
  const auto* arguments = loom_func_like_arg_ids(function, &count);
  loom_pipeline_worker_materialization_t instance = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(module_, module_, &arena_, nullptr,
                                          &instance.channels.remap));
  IREE_ASSERT_OK(loom_ir_remap_map_values(&instance.channels.remap, arguments_,
                                          arguments, count));
  loom_builder_enter_region(&rewriter_.builder, function.op,
                            loom_func_like_body(function));
  loom_builder_set_before(&rewriter_.builder, result);
  IREE_ASSERT_OK(loom_aie2p_native_materialize_storage_accesses(
      &context_, &rewriter_, &worker_, &instance));
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(loom_op_operands(source_return_)[i], arguments_[i]);
    const auto replacement = loom_op_operands(result)[i];
    EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, replacement),
                                types_[i]));
    if (i == 4) {
      EXPECT_EQ(replacement, arguments[i]);
      continue;
    }
    EXPECT_FALSE(loom_module_value_has_uses(module_, arguments[i]));
    auto* aligned = loom_value_def_op(loom_module_value(module_, replacement));
    if (i != 0) {
      ASSERT_TRUE(loom_buffer_view_isa(aligned));
      const auto* offset = loom_value_def_op(
          loom_module_value(module_, loom_buffer_view_byte_offset(aligned)));
      EXPECT_EQ(loom_index_constant_value(offset).i64,
                worker_.storage.accesses[i].byte_offset);
      aligned = loom_value_def_op(
          loom_module_value(module_, loom_buffer_view_buffer(aligned)));
    }
    ASSERT_TRUE(loom_buffer_assume_alignment_isa(aligned));
    EXPECT_EQ(loom_buffer_assume_alignment_minimum_alignment(aligned),
              worker_.storage.accesses[i].alignment);
  }
}

}  // namespace
