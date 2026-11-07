// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_requirements.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/read_only_data.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowFunctionRequirementsTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("requirements"),
                                        &block_pool_, /*archive=*/nullptr,
                                        iree_allocator_system(), &module_));
    descriptor_set_ = loom_test_low_core_descriptor_set();
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("test.low.core"),
                                             &descriptor_set_name_));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("data"), &data_name_));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("z_other"), &other_name_));
    IREE_ASSERT_OK(loom_module_intern_type_id(module_, loom_type_buffer(),
                                              &buffer_type_id_));
    i32_type_ = RegisterType(IREE_SV("test.i32"));
    ptr_type_ = RegisterType(IREE_SV("test.ptr"));
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_type_t RegisterType(iree_string_view_t register_class_name) {
    uint16_t register_class = LOOM_LOW_REG_CLASS_NONE;
    const bool found = loom_low_descriptor_set_lookup_register_class(
        descriptor_set_, register_class_name, &register_class,
        /*out_descriptor_register_class=*/nullptr);
    IREE_ASSERT(found);
    return loom_low_register_type(descriptor_set_->stable_id, register_class,
                                  /*unit_count=*/1);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{/*module_id=*/0, /*symbol_id=*/symbol_id};
  }

  loom_op_t* BuildFunction(loom_symbol_ref_t function_ref) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_op_t* function = nullptr;
    IREE_CHECK_OK(loom_low_func_def_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*effect_contract=*/0, /*temperature=*/0,
        /*inline_policy=*/0,
        /*allocation=*/0, /*schedule=*/0, descriptor_set_name_,
        /*target=*/loom_symbol_ref_t{}, /*abi=*/0,
        loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), function_ref,
        /*arg_types=*/nullptr, /*arg_types_count=*/0, /*result_types=*/nullptr,
        /*result_count=*/0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function));
    return function;
  }

  loom_op_t* BuildFunction(iree_string_view_t name) {
    return BuildFunction(AddSymbol(name));
  }

  loom_builder_t BodyBuilder(loom_op_t* function) {
    loom_builder_t builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_low_func_def_body(function)), &builder);
    loom_builder_enter_region(&builder, function,
                              loom_low_func_def_body(function));
    return builder;
  }

  loom_builder_t BlockBuilder(loom_op_t* function, loom_block_t* block) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena, block, &builder);
    builder.ip.parent_op = function;
    return builder;
  }

  void BuildReturn(loom_builder_t* builder) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_return_build(builder, /*values=*/nullptr,
                                        /*values_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_symbol_ref_t BuildReadOnlyData(iree_string_view_t name,
                                      int64_t alignment, uint8_t contents) {
    const loom_symbol_ref_t symbol = AddSymbol(name);
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    const loom_global_rodata_def_build_flags_t flags =
        alignment > 1 ? LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT : 0;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_global_rodata_def_build(
        &builder, flags, /*visibility=*/0, /*linkage=*/0, symbol, alignment,
        loom_symbol_ref_array_empty(),
        iree_make_const_byte_span(&contents, sizeof(contents)),
        LOOM_LOCATION_UNKNOWN, &op));
    return symbol;
  }

  const loom_low_descriptor_t* Descriptor(iree_string_view_t key) {
    const uint32_t ordinal =
        loom_low_descriptor_set_lookup_descriptor(descriptor_set_, key);
    IREE_ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    return loom_low_descriptor_set_descriptor_at(descriptor_set_, ordinal);
  }

  void BuildReadOnlyDataPacket(loom_builder_t* builder, bool is_const,
                               loom_symbol_ref_t other,
                               loom_symbol_ref_t data) {
    const loom_named_attr_t attrs[] = {
        {.name_id = data_name_, .value = loom_attr_symbol(data)},
        {.name_id = other_name_, .value = loom_attr_symbol(other)},
    };
    const loom_named_attr_slice_t attr_slice =
        loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs));
    loom_op_t* op = nullptr;
    if (is_const) {
      IREE_CHECK_OK(loom_low_build_resolved_descriptor_const(
          builder, descriptor_set_,
          Descriptor(IREE_SV("test.requirements.rodata.const.i32")), attr_slice,
          i32_type_, LOOM_LOCATION_UNKNOWN, &op));
    } else {
      IREE_CHECK_OK(loom_low_build_resolved_descriptor_op(
          builder, descriptor_set_,
          Descriptor(IREE_SV("test.requirements.rodata.op.i32")),
          /*access_flags=*/0, /*operands=*/nullptr, /*operand_count=*/0,
          attr_slice, &i32_type_, /*result_count=*/1,
          /*tied_results=*/nullptr, /*tied_result_count=*/0,
          LOOM_LOCATION_UNKNOWN, &op));
    }
  }

  loom_low_function_requirements_t Analyze(loom_op_t* function) {
    loom_low_function_requirements_t requirements = {};
    IREE_CHECK_OK(loom_low_function_requirements_build(
        module_, descriptor_set_, loom_low_func_def_body(function),
        &analysis_arena_, &requirements));
    return requirements;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_string_id_t descriptor_set_name_ = LOOM_STRING_ID_INVALID;
  loom_string_id_t data_name_ = LOOM_STRING_ID_INVALID;
  loom_string_id_t other_name_ = LOOM_STRING_ID_INVALID;
  loom_type_id_t buffer_type_id_ = LOOM_TYPE_ID_INVALID;
  loom_type_t i32_type_ = loom_type_none();
  loom_type_t ptr_type_ = loom_type_none();
};

TEST_F(LowFunctionRequirementsTest, ImportsPreserveBodyOrder) {
  loom_op_t* function = BuildFunction(IREE_SV("imports"));
  loom_builder_t builder = BodyBuilder(function);
  loom_op_t* input = nullptr;
  IREE_ASSERT_OK(loom_low_resource_build(
      &builder, LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT,
      LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER, LOOM_VALUE_ID_INVALID,
      /*index=*/7, buffer_type_id_, /*extent=*/64,
      /*cache_swizzle_stride=*/0, ptr_type_, LOOM_LOCATION_UNKNOWN, &input));
  loom_op_t* output = nullptr;
  IREE_ASSERT_OK(loom_low_resource_build(
      &builder, /*build_flags=*/0, LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
      LOOM_VALUE_ID_INVALID,
      /*index=*/2, buffer_type_id_, /*extent=*/0,
      /*cache_swizzle_stride=*/0, ptr_type_, LOOM_LOCATION_UNKNOWN, &output));
  loom_op_t* storage = nullptr;
  IREE_ASSERT_OK(loom_low_storage_reserve_build(
      &builder, /*byte_length=*/8, /*byte_alignment=*/8,
      loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), LOOM_LOCATION_UNKNOWN,
      &storage));
  loom_op_t* spill = nullptr;
  IREE_ASSERT_OK(
      loom_low_spill_build(&builder, loom_low_resource_result(output),
                           loom_low_storage_reserve_storage(storage),
                           /*offset=*/0, LOOM_LOCATION_UNKNOWN, &spill));
  BuildReturn(&builder);

  const auto requirements = Analyze(function);
  ASSERT_EQ(requirements.resource_count, 2u);
  EXPECT_EQ(requirements.resources[0], input);
  EXPECT_EQ(loom_low_resource_index(requirements.resources[0]), 7);
  EXPECT_EQ(loom_low_resource_extent(requirements.resources[0]), 64);
  EXPECT_EQ(requirements.resources[1], output);
  EXPECT_EQ(loom_low_resource_index(requirements.resources[1]), 2);
  EXPECT_EQ(requirements.entry_preamble_end, output);
  EXPECT_EQ(requirements.entry_traffic_end, spill);
}

TEST_F(LowFunctionRequirementsTest, CollectsStorageAcrossBlocks) {
  loom_op_t* function = BuildFunction(IREE_SV("storage"));
  loom_region_t* body = loom_low_func_def_body(function);
  loom_builder_t entry_builder = BodyBuilder(function);
  loom_op_t* stack = nullptr;
  IREE_ASSERT_OK(loom_low_storage_reserve_build(
      &entry_builder, /*byte_length=*/8, /*byte_alignment=*/4,
      loom_type_storage(LOOM_STORAGE_SPACE_STACK), LOOM_LOCATION_UNKNOWN,
      &stack));
  loom_op_t* private_storage = nullptr;
  IREE_ASSERT_OK(loom_low_storage_reserve_build(
      &entry_builder, /*byte_length=*/12, /*byte_alignment=*/8,
      loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), LOOM_LOCATION_UNKNOWN,
      &private_storage));
  loom_op_t* stack_address = nullptr;
  IREE_ASSERT_OK(loom_low_storage_address_build(
      &entry_builder, loom_low_storage_reserve_storage(stack), /*offset=*/0,
      ptr_type_, LOOM_LOCATION_UNKNOWN, &stack_address));

  loom_block_t* left = nullptr;
  loom_block_t* right = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &left));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &right));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&entry_builder, left, /*args=*/nullptr,
                                   /*args_count=*/0, LOOM_LOCATION_UNKNOWN,
                                   &branch));
  loom_builder_t left_builder = BlockBuilder(function, left);
  loom_op_t* later = nullptr;
  IREE_ASSERT_OK(loom_low_storage_reserve_build(
      &left_builder, /*byte_length=*/16, /*byte_alignment=*/16,
      loom_type_storage(LOOM_STORAGE_SPACE_STACK), LOOM_LOCATION_UNKNOWN,
      &later));
  loom_op_t* later_address = nullptr;
  IREE_ASSERT_OK(loom_low_storage_address_build(
      &left_builder, loom_low_storage_reserve_storage(later), /*offset=*/0,
      ptr_type_, LOOM_LOCATION_UNKNOWN, &later_address));
  BuildReturn(&left_builder);
  loom_builder_t right_builder = BlockBuilder(function, right);
  BuildReturn(&right_builder);

  const auto requirements = Analyze(function);
  EXPECT_EQ(requirements.node_count, 8u);
  EXPECT_EQ(requirements.return_count, 2u);
  EXPECT_EQ(requirements.storage_address_count, 2u);
  const auto& layout = requirements.storage_layout;
  EXPECT_EQ(layout.space_sizes.stack_bytes, 32u);
  EXPECT_EQ(layout.space_sizes.private_bytes, 12u);
  ASSERT_EQ(layout.record_count, 3u);
  EXPECT_EQ(layout.records[2].reservation.byte_offset, 16u);
}

TEST_F(LowFunctionRequirementsTest, EmptyInterface) {
  loom_op_t* function = BuildFunction(IREE_SV("empty"));
  loom_builder_t builder = BodyBuilder(function);
  BuildReturn(&builder);

  const auto requirements = Analyze(function);
  EXPECT_EQ(requirements.resource_count, 0u);
  EXPECT_EQ(requirements.resources, nullptr);
  EXPECT_EQ(requirements.storage_layout.record_count, 0u);
  EXPECT_EQ(requirements.read_only_data.count, 0u);
  EXPECT_EQ(requirements.read_only_data.values, nullptr);
  EXPECT_EQ(requirements.read_only_data.slot_count, 0u);
  EXPECT_EQ(requirements.read_only_data.ordinal_slots, nullptr);
  EXPECT_EQ(requirements.node_count, 1u);
  EXPECT_EQ(requirements.return_count, 1u);
  EXPECT_EQ(requirements.storage_address_count, 0u);
}

TEST_F(LowFunctionRequirementsTest,
       RetainsDescriptorSelectedDataAcrossPacketKinds) {
  const loom_symbol_ref_t data0 =
      BuildReadOnlyData(IREE_SV("data0"), /*alignment=*/16, 0x00);
  const loom_symbol_ref_t data1 =
      BuildReadOnlyData(IREE_SV("data1"), /*alignment=*/1, 0x11);
  const loom_symbol_ref_t data2 =
      BuildReadOnlyData(IREE_SV("data2"), /*alignment=*/1, 0x22);
  const loom_symbol_ref_t data3 =
      BuildReadOnlyData(IREE_SV("data3"), /*alignment=*/1, 0x33);
  const loom_symbol_ref_t data4 =
      BuildReadOnlyData(IREE_SV("data4"), /*alignment=*/32, 0x44);
  const loom_symbol_ref_t unused =
      BuildReadOnlyData(IREE_SV("unused"), /*alignment=*/1, 0xFF);
  loom_op_t* function = BuildFunction(IREE_SV("uses_data"));
  loom_builder_t builder = BodyBuilder(function);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/false, unused, data4);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/true, unused, data0);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/true, unused, data4);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/false, unused, data2);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/true, unused, data1);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/false, unused, data3);
  BuildReturn(&builder);

  const auto requirements = Analyze(function);
  ASSERT_EQ(requirements.read_only_data.count, 5u);
  EXPECT_EQ(requirements.read_only_data.slot_count, 16u);
  EXPECT_EQ(requirements.read_only_data.values[0].symbol.symbol_id,
            data4.symbol_id);
  EXPECT_EQ(requirements.read_only_data.values[0].minimum_alignment, 32u);
  EXPECT_EQ(requirements.read_only_data.values[1].symbol.symbol_id,
            data0.symbol_id);
  EXPECT_EQ(requirements.read_only_data.values[1].minimum_alignment, 16u);
  EXPECT_EQ(requirements.read_only_data.values[2].symbol.symbol_id,
            data2.symbol_id);
  EXPECT_EQ(requirements.read_only_data.values[3].symbol.symbol_id,
            data1.symbol_id);
  EXPECT_EQ(requirements.read_only_data.values[4].symbol.symbol_id,
            data3.symbol_id);
  const loom_symbol_ref_t expected_order[] = {data4, data0, data2, data1,
                                              data3};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(expected_order); ++i) {
    EXPECT_EQ(loom_low_read_only_data_ordinal(&requirements.read_only_data,
                                              expected_order[i]),
              i);
  }
  EXPECT_EQ(
      loom_low_read_only_data_ordinal(&requirements.read_only_data, unused),
      UINT32_MAX);
}

TEST_F(LowFunctionRequirementsTest, SmallDataSetNeedsNoOrdinalIndex) {
  const loom_symbol_ref_t first =
      BuildReadOnlyData(IREE_SV("first"), /*alignment=*/1, 0x11);
  const loom_symbol_ref_t second =
      BuildReadOnlyData(IREE_SV("second"), /*alignment=*/1, 0x22);
  loom_op_t* function = BuildFunction(IREE_SV("small_data"));
  loom_builder_t builder = BodyBuilder(function);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/true, second, first);
  BuildReadOnlyDataPacket(&builder, /*is_const=*/false, first, second);
  BuildReturn(&builder);

  const auto requirements = Analyze(function);
  ASSERT_EQ(requirements.read_only_data.count, 2u);
  EXPECT_EQ(requirements.read_only_data.slot_count, 0u);
  EXPECT_EQ(requirements.read_only_data.ordinal_slots, nullptr);
  EXPECT_EQ(
      loom_low_read_only_data_ordinal(&requirements.read_only_data, first), 0u);
  EXPECT_EQ(
      loom_low_read_only_data_ordinal(&requirements.read_only_data, second),
      1u);
}

TEST_F(LowFunctionRequirementsTest, UnionsInventoriesWithoutChangingOrdinals) {
  const loom_symbol_ref_t symbols[] = {
      BuildReadOnlyData(IREE_SV("entry"), 16, 0x11),
      BuildReadOnlyData(IREE_SV("shared"), 32, 0x22),
      BuildReadOnlyData(IREE_SV("helper_a"), 64, 0x33),
      BuildReadOnlyData(IREE_SV("helper_b"), 1, 0x44),
      BuildReadOnlyData(IREE_SV("helper_c"), 8, 0x55),
  };
  loom_op_t* entry = BuildFunction(IREE_SV("entry_function"));
  loom_builder_t entry_builder = BodyBuilder(entry);
  for (unsigned i : {0, 1, 2}) {
    BuildReadOnlyDataPacket(&entry_builder, true, symbols[0], symbols[i]);
  }
  BuildReturn(&entry_builder);
  const auto entry_requirements = Analyze(entry);

  loom_op_t* helper = BuildFunction(IREE_SV("helper_function"));
  loom_builder_t helper_builder = BodyBuilder(helper);
  for (unsigned i : {1, 4, 3}) {
    BuildReadOnlyDataPacket(&helper_builder, true, symbols[0], symbols[i]);
  }
  BuildReturn(&helper_builder);
  const auto helper_requirements = Analyze(helper);

  loom_low_read_only_data_builder_t data_builder = {};
  IREE_ASSERT_OK(loom_low_read_only_data_builder_extend(
      &data_builder, &entry_requirements.read_only_data, &analysis_arena_));
  IREE_ASSERT_OK(loom_low_read_only_data_builder_extend(
      &data_builder, &helper_requirements.read_only_data, &analysis_arena_));
  IREE_ASSERT_OK(loom_low_read_only_data_builder_extend(
      &data_builder, &entry_requirements.read_only_data, &analysis_arena_));
  const auto data = loom_low_read_only_data_builder_finish(&data_builder);
  ASSERT_EQ(data.count, 5u);
  const unsigned expected_order[] = {0, 1, 2, 4, 3};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(expected_order); ++i) {
    const auto symbol = symbols[expected_order[i]];
    EXPECT_EQ(loom_low_read_only_data_ordinal(&data, symbol), i);
    EXPECT_EQ(data.values[i].definition,
              module_->symbols.entries[symbol.symbol_id].defining_op);
    EXPECT_EQ(data.values[i].contents.data[0], (expected_order[i] + 1) * 0x11);
  }
  EXPECT_EQ(data.values[1].minimum_alignment, 32u);
  EXPECT_EQ(entry_requirements.read_only_data.count, 3u);
  EXPECT_EQ(helper_requirements.read_only_data.count, 3u);
  EXPECT_EQ(loom_low_read_only_data_ordinal(&helper_requirements.read_only_data,
                                            symbols[1]),
            0u);
}

TEST_F(LowFunctionRequirementsTest, ResidentControlFlowHasNoReturn) {
  loom_op_t* function = BuildFunction(IREE_SV("resident"));
  loom_region_t* body = loom_low_func_def_body(function);
  loom_builder_t entry_builder = BodyBuilder(function);
  loom_block_t* wait = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &wait));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&entry_builder, wait, /*args=*/nullptr,
                                   /*args_count=*/0, LOOM_LOCATION_UNKNOWN,
                                   &branch));
  loom_builder_t wait_builder = BlockBuilder(function, wait);
  IREE_ASSERT_OK(loom_low_br_build(&wait_builder, wait, /*args=*/nullptr,
                                   /*args_count=*/0, LOOM_LOCATION_UNKNOWN,
                                   &branch));

  const auto requirements = Analyze(function);
  EXPECT_EQ(requirements.node_count, 2u);
  EXPECT_EQ(requirements.return_count, 0u);
}

TEST_F(LowFunctionRequirementsTest, RetainsUniqueDirectCalleesInCallOrder) {
  const loom_symbol_ref_t entry_ref = AddSymbol(IREE_SV("entry"));
  const loom_symbol_ref_t first_ref = AddSymbol(IREE_SV("first"));
  const loom_symbol_ref_t second_ref = AddSymbol(IREE_SV("second"));
  loom_op_t* entry = BuildFunction(entry_ref);
  BuildFunction(first_ref);
  BuildFunction(second_ref);
  loom_builder_t builder = BodyBuilder(entry);
  loom_op_t* first_second_call = nullptr;
  loom_op_t* first_first_call = nullptr;
  for (loom_symbol_ref_t callee : {second_ref, first_ref, second_ref}) {
    loom_op_t* call = nullptr;
    IREE_ASSERT_OK(loom_low_func_call_build(
        &builder, /*build_flags=*/0, /*effect_contract=*/0,
        /*temperature=*/0, /*inline_policy=*/0, callee,
        /*operands=*/nullptr,
        /*operands_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &call));
    if (callee.symbol_id == second_ref.symbol_id && !first_second_call) {
      first_second_call = call;
    } else if (callee.symbol_id == first_ref.symbol_id) {
      first_first_call = call;
    }
  }
  BuildReturn(&builder);

  const auto requirements = Analyze(entry);
  ASSERT_EQ(requirements.direct_callees.count, 2u);
  EXPECT_EQ(requirements.direct_callees.values[0].symbol.symbol_id,
            second_ref.symbol_id);
  EXPECT_EQ(requirements.direct_callees.values[0].first_call_op,
            first_second_call);
  EXPECT_EQ(requirements.direct_callees.values[1].symbol.symbol_id,
            first_ref.symbol_id);
  EXPECT_EQ(requirements.direct_callees.values[1].first_call_op,
            first_first_call);
}

TEST_F(LowFunctionRequirementsTest, RetainsManyUniqueDirectCalleesInCallOrder) {
  const loom_symbol_ref_t entry_ref = AddSymbol(IREE_SV("entry"));
  const loom_symbol_ref_t callees[] = {
      AddSymbol(IREE_SV("callee0")), AddSymbol(IREE_SV("callee1")),
      AddSymbol(IREE_SV("callee2")), AddSymbol(IREE_SV("callee3")),
      AddSymbol(IREE_SV("callee4")), AddSymbol(IREE_SV("callee5")),
      AddSymbol(IREE_SV("callee6")), AddSymbol(IREE_SV("callee7")),
  };
  loom_op_t* entry = BuildFunction(entry_ref);
  for (loom_symbol_ref_t callee : callees) {
    BuildFunction(callee);
  }
  loom_symbol_ref_t remote_callee = callees[5];
  remote_callee.module_id = 1;
  const loom_symbol_ref_t call_order[] = {
      callees[5], callees[0],    callees[6], callees[1],    callees[4],
      callees[2], callees[3],    callees[7], remote_callee, callees[5],
      callees[0], remote_callee, callees[7],
  };
  loom_builder_t builder = BodyBuilder(entry);
  for (loom_symbol_ref_t callee : call_order) {
    loom_op_t* call = nullptr;
    IREE_ASSERT_OK(loom_low_func_call_build(
        &builder, /*build_flags=*/0, /*effect_contract=*/0,
        /*temperature=*/0, /*inline_policy=*/0, callee,
        /*operands=*/nullptr,
        /*operands_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &call));
  }
  BuildReturn(&builder);

  const auto requirements = Analyze(entry);
  const loom_symbol_ref_t expected_order[] = {
      callees[5], callees[0], callees[6], callees[1],    callees[4],
      callees[2], callees[3], callees[7], remote_callee,
  };
  ASSERT_EQ(requirements.direct_callees.count, IREE_ARRAYSIZE(expected_order));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(expected_order); ++i) {
    EXPECT_EQ(requirements.direct_callees.values[i].symbol.module_id,
              expected_order[i].module_id);
    EXPECT_EQ(requirements.direct_callees.values[i].symbol.symbol_id,
              expected_order[i].symbol_id);
  }
}

}  // namespace
}  // namespace loom
