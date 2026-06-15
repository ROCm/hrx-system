// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/varint.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* error_ids = static_cast<std::vector<std::string>*>(user_data);
  error_ids->push_back(diagnostic->error->error_id);
  return iree_ok_status();
}

class ReaderTest : public ::testing::Test {
 protected:
  struct SectionEntry {
    // Section kind from loom_bytecode_section_kind_t.
    uint16_t kind = 0;
    // Byte offset of this entry in the module section directory.
    size_t directory_entry_offset = 0;
    // Byte offset from the start of the module.
    uint64_t offset = 0;
    // Byte length of the section payload.
    uint64_t length = 0;
  };

  struct ValueDefOffsets {
    // Byte offset of the value definition's dynamic-dim binding count.
    size_t dim_binding_count = 0;
    // Byte offset of the value definition's SSA encoding binding.
    size_t encoding_binding = 0;
  };

  struct BodyOpAttrOffsets {
    // Byte offset of the operation attribute value kind byte.
    size_t attr_kind = 0;
    // Byte offset of the first key inside the nested dict value.
    size_t nested_dict_first_key = 0;
    // Byte offset of the second key inside the nested dict value.
    size_t nested_dict_second_key = 0;
  };

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    InitializeBytecodeTestContext(&context_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t *
                                                            out_count);

  void InitializeBytecodeTestContext(loom_context_t* context) {
    loom_context_initialize(iree_allocator_system(), context);
    RegisterDialect(context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(context, LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables,
                    loom_global_dialect_op_semantics);
    RegisterDialect(context, LOOM_DIALECT_TEST, loom_test_dialect_vtables,
                    loom_test_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(context));
  }

  void RegisterDialect(loom_context_t* context, loom_dialect_id_t dialect_id,
                       DialectVtablesFn vtables_fn,
                       DialectSemanticsFn semantics_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(context, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics = semantics_fn(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        context, dialect_id, semantics, (uint16_t)semantics_count));
  }

  loom_module_t* CreateModule(const char* name) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, iree_make_cstring_view(name),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module));
    return module;
  }

  loom_op_t* AddSimpleFunction(loom_module_t* module, const char* name) {
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder, iree_make_cstring_view(name), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[1] = {i32_type};
    loom_type_t result_types[1] = {i32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return func_op;
    }
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_op_t* addi_op = nullptr;
    IREE_CHECK_OK(loom_test_addi_build(&body_builder, arg_ids[0], arg_ids[0],
                                       i32_type, LOOM_LOCATION_UNKNOWN,
                                       &addi_op));
    loom_value_id_t addi_result = loom_test_addi_result(addi_op);
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, &addi_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return func_op;
  }

  loom_module_t* CreateFunctionModule() {
    loom_module_t* module = CreateModule("reader_func");
    AddSimpleFunction(module, "f");
    return module;
  }

  loom_module_t* CreateImportedFunctionModule() {
    loom_module_t* module = CreateModule("reader_import");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("decl"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_string_id_t import_module_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("kernel_lib"),
                                             &import_module_id));
    loom_string_id_t import_symbol_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("extern_f"),
                                             &import_symbol_id));
    loom_type_t arg_types[1] = {i32_type};
    loom_type_t result_types[1] = {i32_type};
    loom_op_t* decl_op = nullptr;
    IREE_CHECK_OK(loom_func_decl_build(
        &builder,
        LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE |
            LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL,
        /*visibility=*/0, import_module_id, import_symbol_id, /*cc=*/0,
        /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), callee,
        arg_types, IREE_ARRAYSIZE(arg_types), result_types,
        IREE_ARRAYSIZE(result_types), nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &decl_op));
    return module;
  }

  loom_module_t* CreateTestRecordWithFutureEnumOrdinal() {
    loom_module_t* module = CreateModule("reader_future_test_enum");
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("future"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* record_op = nullptr;
    IREE_CHECK_OK(loom_test_record_build(
        &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_KIND,
        LOOM_TEST_RECORD_KIND_ARTIFACT, symbol, loom_named_attr_slice_empty(),
        LOOM_LOCATION_UNKNOWN, &record_op));
    loom_op_attrs(record_op)[loom_test_record_kind_ATTR_INDEX] =
        loom_attr_enum(250);
    return module;
  }

  loom_module_t* CreateGlobalModule() {
    loom_module_t* module = CreateModule("reader_global");
    loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    IREE_CHECK_OK(loom_module_intern_type(module, index_type, &index_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("answer"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* global_op = nullptr;
    IREE_CHECK_OK(loom_global_constant_build(
        &builder, symbol, index_type, /*predicates=*/nullptr,
        /*predicates_count=*/0, loom_attr_i64(42), LOOM_LOCATION_UNKNOWN,
        &global_op));
    return module;
  }

  loom_module_t* CreateRegisterDeclModule() {
    loom_module_t* module = CreateModule("reader_register_decl");
    loom_type_t reg_type =
        loom_low_register_type(/*descriptor_set_stable_id=*/1,
                               /*register_class_id=*/0, /*unit_count=*/4);
    IREE_CHECK_OK(loom_module_intern_type(module, reg_type, &reg_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("reg_decl"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* decl_op = nullptr;
    IREE_CHECK_OK(loom_test_decl_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, &reg_type,
        /*arg_types_count=*/1, &reg_type, /*result_count=*/1, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &decl_op));
    return module;
  }

  loom_module_t* CreateDynamicGlobalModule() {
    loom_module_t* module = CreateModule("reader_dynamic_global");
    loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    IREE_CHECK_OK(loom_module_intern_type(module, index_type, &index_type));

    loom_value_id_t dim_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module, index_type, &dim_id));
    loom_string_id_t dim_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module, IREE_SV("n"), &dim_name_id));
    IREE_CHECK_OK(loom_module_set_value_name(module, dim_id, dim_name_id));

    loom_type_t tile_type =
        loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(dim_id), /*encoding_id=*/0);

    loom_predicate_t* predicates = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module->arena, 1, sizeof(loom_predicate_t), (void**)&predicates));
    predicates[0] = loom_predicate_t{
        /*.kind=*/LOOM_PREDICATE_MUL,
        /*.arg_count=*/2,
        /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
        /*.reserved=*/{},
        /*.args=*/{(int64_t)dim_id, 16},
    };

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("weights"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* global_op = nullptr;
    IREE_CHECK_OK(loom_global_constant_build(
        &builder, symbol, tile_type, predicates, 1, loom_attr_absent(),
        LOOM_LOCATION_UNKNOWN, &global_op));
    return module;
  }

  loom_module_t* CreateTwoFunctionModule() {
    loom_module_t* module = CreateModule("reader_two_funcs");
    AddSimpleFunction(module, "f0");
    AddSimpleFunction(module, "f1");
    return module;
  }

  loom_module_t* CreateMultiBlockFunctionModule() {
    loom_module_t* module = CreateModule("reader_multi_block");
    loom_op_t* func_op = AddSimpleFunction(module, "f");
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }

    loom_region_t* body = loom_func_like_body(func_like);
    loom_block_t* second_block = nullptr;
    IREE_CHECK_OK(loom_region_append_block(module, body, &second_block));
    loom_builder_t block_builder;
    loom_builder_initialize(module, &module->arena, second_block,
                            &block_builder);
    loom_op_t* use_op = nullptr;
    IREE_CHECK_OK(loom_test_use_build(&block_builder, arg_ids, arg_count,
                                      LOOM_LOCATION_UNKNOWN, &use_op));
    return module;
  }

  loom_module_t* CreateSuccessorFunctionModule() {
    loom_module_t* module = CreateModule("reader_successor");

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("cfg"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee,
        /*arg_types=*/nullptr, 0, /*result_types=*/nullptr, 0,
        /*arg_names=*/nullptr, 0, /*result_names=*/nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));

    loom_region_t* body = loom_test_func_body(func_op);
    loom_block_t* entry_block = loom_region_entry_block(body);
    loom_block_t* exit_block = nullptr;
    IREE_CHECK_OK(loom_region_append_block(module, body, &exit_block));

    loom_builder_t entry_builder;
    loom_builder_initialize(module, &module->arena, entry_block,
                            &entry_builder);
    loom_op_t* br_op = nullptr;
    IREE_CHECK_OK(loom_test_br_build(&entry_builder, exit_block,
                                     LOOM_LOCATION_UNKNOWN, &br_op));

    loom_builder_t exit_builder;
    loom_builder_initialize(module, &module->arena, exit_block, &exit_builder);
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&exit_builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateDynamicDimFunctionModule() {
    loom_module_t* module = CreateModule("reader_dynamic_dim");
    loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_type_t vector_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(4), /*encoding_id=*/0);
    loom_type_t arg_types[2] = {index_type, vector_type};

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 2) {
      ADD_FAILURE() << "expected two function arguments";
      return module;
    }
    loom_type_t rebound_vector =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_dynamic(arg_ids[0]),
                            /*encoding_id=*/0);
    IREE_CHECK_OK(
        loom_module_set_value_type(module, arg_ids[1], rebound_vector));
    return module;
  }

  loom_module_t* CreateSsaEncodingFunctionModule() {
    loom_module_t* module = CreateModule("reader_ssa_encoding");
    loom_type_t encoding_type =
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
    loom_type_t view_type =
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_static(4), /*encoding_id=*/0);
    loom_type_t arg_types[2] = {encoding_type, view_type};

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 2) {
      ADD_FAILURE() << "expected two function arguments";
      return module;
    }
    loom_type_t rebound_view =
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_static(4), (uint16_t)arg_ids[0]);
    rebound_view.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    IREE_CHECK_OK(loom_module_set_value_type(module, arg_ids[1], rebound_view));
    return module;
  }

  loom_module_t* CreateCoResultDimFunctionModule() {
    loom_module_t* module = CreateModule("reader_co_result_dim");
    loom_type_t input_type = loom_type_shaped_1d(
        LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4),
        /*encoding_id=*/0);
    loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_type_t arg_types[1] = {input_type};

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }

    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_value_id_t result_ids[2] = {};
    IREE_CHECK_OK(loom_builder_reserve_results(
        &body_builder, IREE_ARRAYSIZE(result_ids), result_ids));
    loom_type_t output_type =
        loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(result_ids[1]),
                            /*encoding_id=*/0);
    loom_type_t result_types[2] = {output_type, index_type};
    loom_op_t* deflate_op = nullptr;
    IREE_CHECK_OK(loom_test_deflate_build(
        &body_builder, arg_ids[0], result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &deflate_op));
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateAttributeFunctionModule() {
    loom_module_t* module = CreateModule("reader_attrs");
    loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    IREE_CHECK_OK(loom_module_intern_type(module, f32_type, &f32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[1] = {f32_type};
    loom_type_t result_types[1] = {f32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }

    loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t meta_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t opt_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t phase_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t link_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("axis"), &axis_id));
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("meta"), &meta_id));
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("opt"), &opt_id));
    IREE_CHECK_OK(
        loom_module_intern_string(module, IREE_SV("phase"), &phase_id));
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("link"), &link_id));

    loom_named_attr_t meta_entries[2] = {
        {
            /*.name_id=*/opt_id,
            /*.reserved=*/{},
            /*.value=*/loom_attr_i64(3),
        },
        {
            /*.name_id=*/phase_id,
            /*.reserved=*/{},
            /*.value=*/loom_attr_string(link_id),
        },
    };
    loom_attribute_t meta_attr = {0};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module,
        loom_make_named_attr_slice(meta_entries, IREE_ARRAYSIZE(meta_entries)),
        &meta_attr));

    loom_named_attr_t entries[2] = {
        {
            /*.name_id=*/axis_id,
            /*.reserved=*/{},
            /*.value=*/loom_attr_i64(0),
        },
        {
            /*.name_id=*/meta_id,
            /*.reserved=*/{},
            /*.value=*/meta_attr,
        },
    };
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_op_t* attrs_op = nullptr;
    IREE_CHECK_OK(loom_test_attrs_build(
        &body_builder, arg_ids[0],
        loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32_type,
        LOOM_LOCATION_UNKNOWN, &attrs_op));
    loom_value_id_t result_id = loom_op_results(attrs_op)[0];
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, &result_id, 1,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreatePredicateFunctionModule() {
    loom_module_t* module = CreateModule("reader_predicates");
    loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    IREE_CHECK_OK(loom_module_intern_type(module, f32_type, &f32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[1] = {f32_type};
    loom_type_t result_types[1] = {f32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }

    loom_predicate_t* predicates = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module->arena, 2, sizeof(loom_predicate_t), (void**)&predicates));
    predicates[0] = loom_predicate_t{
        /*.kind=*/LOOM_PREDICATE_MUL,
        /*.arg_count=*/2,
        /*.arg_tags=*/
        {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_NONE},
        /*.reserved=*/{},
        /*.args=*/{(int64_t)arg_ids[0], 16, 0},
    };
    predicates[1] = loom_predicate_t{
        /*.kind=*/LOOM_PREDICATE_RANGE,
        /*.arg_count=*/3,
        /*.arg_tags=*/
        {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
        /*.reserved=*/{},
        /*.args=*/{(int64_t)arg_ids[0], 32, 512},
    };
    loom_op_attrs(func_op)[func_like.vtable->predicates_attr_index] =
        loom_attr_predicate_list(predicates, 2);
    return module;
  }

  loom_module_t* CreateTiedBodyOpModule() {
    loom_module_t* module = CreateModule("reader_tied_body_op");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_type_t arg_types[1] = {i32_type};

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_tied_result_t tied_result = {/*.result_index=*/0,
                                      /*.operand_index=*/0};
    loom_op_t* map_op = nullptr;
    IREE_CHECK_OK(loom_test_map_build(&body_builder, arg_ids, arg_count,
                                      i32_type, &tied_result, 1,
                                      LOOM_LOCATION_UNKNOWN, &map_op));
    loom_region_t* map_body = loom_test_map_body(map_op);
    loom_builder_t map_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(map_body), &map_builder);
    loom_op_t* use_op = nullptr;
    IREE_CHECK_OK(loom_test_use_build(&map_builder, arg_ids, arg_count,
                                      LOOM_LOCATION_UNKNOWN, &use_op));
    return module;
  }

  loom_module_t* CreateDeepNestedFunctionModule(uint32_t nested_region_count) {
    loom_module_t* module = CreateModule("reader_deep_nested");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[1] = {i32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }

    loom_builder_t body_builder;
    loom_region_t* body = loom_func_like_body(func_like);
    loom_block_t* current_block = loom_region_entry_block(body);
    loom_builder_initialize(module, &module->arena, current_block,
                            &body_builder);
    loom_value_id_t current_value = arg_ids[0];
    for (uint32_t i = 0; i < nested_region_count; ++i) {
      loom_op_t* map_op = nullptr;
      IREE_CHECK_OK(loom_test_map_build(&body_builder, &current_value, 1,
                                        i32_type, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &map_op));
      loom_region_t* map_body = loom_test_map_body(map_op);
      current_block = loom_region_entry_block(map_body);
      current_value = loom_block_arg_id(current_block, 0);
      loom_builder_initialize(module, &module->arena, current_block,
                              &body_builder);
    }
    loom_op_t* use_op = nullptr;
    IREE_CHECK_OK(loom_test_use_build(&body_builder, &current_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &use_op));
    return module;
  }

  loom_module_t* CreateLocatedModule() {
    loom_module_t* module = CreateModule("located");
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_register_source(module, IREE_SV("model.loom"), &source_id));
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module, loom_location_file_range(source_id, 1, 1, 1, 2), &location_id));
    return module;
  }

  loom_module_t* CreateSegmentedModule() {
    loom_module_t* module = CreateModule("reader_segmented");
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32, &i32));

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&module_builder,
                                             IREE_SV("segmented"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[5] = {i32, i32, i32, i32, i32};
    loom_type_t result_types[1] = {i32};
    loom_op_t* func_op = NULL;
    IREE_CHECK_OK(loom_test_func_build(
        &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != IREE_ARRAYSIZE(arg_types)) {
      ADD_FAILURE() << "expected five segmented function arguments";
      return module;
    }
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);

    loom_value_id_t lhs[] = {args[2], args[3]};
    loom_value_id_t rhs_values[] = {args[4]};
    loom_op_t* segmented_op = NULL;
    IREE_CHECK_OK(loom_test_segmented_build(
        &body_builder, LOOM_TEST_SEGMENTED_BUILD_FLAG_HAS_GUARD, args[0],
        args[1], lhs, IREE_ARRAYSIZE(lhs), rhs_values,
        IREE_ARRAYSIZE(rhs_values), i32, LOOM_LOCATION_UNKNOWN, &segmented_op));
    loom_value_id_t result = loom_test_segmented_result(segmented_op);
    loom_op_t* yield_op = NULL;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  std::vector<uint8_t> WriteModule(
      const loom_module_t* module,
      const loom_bytecode_write_options_t* options = nullptr) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, options, &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  loom_bytecode_read_result_t ReadMetadata(
      const std::vector<uint8_t>& bytes, loom_context_t* context,
      std::vector<std::string>* error_ids) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
    };
    IREE_CHECK_OK(loom_bytecode_read_metadata(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), context, &block_pool_, &options, &result));
    return result;
  }

  loom_bytecode_read_result_t ReadMetadata(
      const std::vector<uint8_t>& bytes, std::vector<std::string>* error_ids) {
    return ReadMetadata(bytes, &context_, error_ids);
  }

  loom_bytecode_read_result_t ReadIndex(
      const std::vector<uint8_t>& bytes, iree_arena_allocator_t* metadata_arena,
      loom_bytecode_file_metadata_t* out_metadata,
      std::vector<std::string>* error_ids) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
    };
    IREE_CHECK_OK(loom_bytecode_read_index(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), &context_, &block_pool_, metadata_arena,
        &options, &result, out_metadata));
    return result;
  }

  loom_bytecode_read_result_t ReadModule(const std::vector<uint8_t>& bytes,
                                         loom_context_t* context,
                                         loom_module_t** out_module,
                                         std::vector<std::string>* error_ids,
                                         bool verify_module = false) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
        /*.verify_module=*/verify_module,
    };
    IREE_CHECK_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), context, &block_pool_, &options, &result,
        out_module, iree_allocator_system()));
    return result;
  }

  loom_bytecode_read_result_t ReadModule(const std::vector<uint8_t>& bytes,
                                         loom_module_t** out_module,
                                         std::vector<std::string>* error_ids,
                                         bool verify_module = false) {
    return ReadModule(bytes, &context_, out_module, error_ids, verify_module);
  }

  loom_bytecode_read_result_t ReadModuleOrdinal(
      const std::vector<uint8_t>& bytes, uint16_t module_ordinal,
      loom_module_t** out_module, std::vector<std::string>* error_ids,
      bool verify_module = false) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
        /*.verify_module=*/verify_module,
    };
    IREE_CHECK_OK(loom_bytecode_read_module_ordinal(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), &context_, &block_pool_, module_ordinal,
        &options, &result, out_module, iree_allocator_system()));
    return result;
  }

  uint16_t ReadU16LE(const std::vector<uint8_t>& bytes, size_t offset) {
    return (uint16_t)bytes[offset] | ((uint16_t)bytes[offset + 1] << 8);
  }

  uint32_t ReadU32LE(const std::vector<uint8_t>& bytes, size_t offset) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      value |= (uint32_t)bytes[offset + i] << (i * 8);
    }
    return value;
  }

  uint64_t ReadU64LE(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= (uint64_t)bytes[offset + i] << (i * 8);
    }
    return value;
  }

  void WriteU16LE(std::vector<uint8_t>* bytes, size_t offset, uint16_t value) {
    (*bytes)[offset] = (uint8_t)value;
    (*bytes)[offset + 1] = (uint8_t)(value >> 8);
  }

  void WriteU32LE(std::vector<uint8_t>* bytes, size_t offset, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      (*bytes)[offset + i] = (uint8_t)(value >> (i * 8));
    }
  }

  void WriteU64LE(std::vector<uint8_t>* bytes, size_t offset, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      (*bytes)[offset + i] = (uint8_t)(value >> (i * 8));
    }
  }

  void AppendU8(std::vector<uint8_t>* bytes, uint8_t value) {
    bytes->push_back(value);
  }

  void AppendU16LE(std::vector<uint8_t>* bytes, uint16_t value) {
    bytes->push_back((uint8_t)value);
    bytes->push_back((uint8_t)(value >> 8));
  }

  void AppendU32LE(std::vector<uint8_t>* bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      bytes->push_back((uint8_t)(value >> (i * 8)));
    }
  }

  void AppendU64LE(std::vector<uint8_t>* bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      bytes->push_back((uint8_t)(value >> (i * 8)));
    }
  }

  void AppendString(std::vector<uint8_t>* bytes, iree_string_view_t string) {
    const auto* data = reinterpret_cast<const uint8_t*>(string.data);
    bytes->insert(bytes->end(), data, data + string.size);
  }

  void PadToAlignment(std::vector<uint8_t>* bytes, size_t alignment) {
    while ((bytes->size() & (alignment - 1)) != 0) {
      bytes->push_back(0);
    }
  }

  uint64_t ReadUVarint(const std::vector<uint8_t>& bytes, size_t* offset) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(bytes.data() + *offset,
                                    bytes.size() - *offset, &cursor);
    uint64_t value = 0;
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &value));
    *offset += cursor.position;
    return value;
  }

  ValueDefOffsets ReadValueDefOffsets(const std::vector<uint8_t>& bytes,
                                      size_t* offset) {
    ValueDefOffsets value_def;
    ReadUVarint(bytes, offset);  // name_id
    ReadUVarint(bytes, offset);  // type_id
    value_def.dim_binding_count = *offset;
    uint64_t dim_binding_count = ReadUVarint(bytes, offset);
    for (uint64_t i = 0; i < dim_binding_count; ++i) {
      ReadUVarint(bytes, offset);
    }
    value_def.encoding_binding = *offset;
    ReadUVarint(bytes, offset);
    return value_def;
  }

  void SkipCommentList(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t comment_count = ReadUVarint(bytes, offset);
    for (uint64_t i = 0; i < comment_count; ++i) {
      uint64_t comment_length = ReadUVarint(bytes, offset);
      *offset += (size_t)comment_length;
    }
  }

  void SkipPredicateList(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t predicate_count = ReadUVarint(bytes, offset);
    for (uint64_t i = 0; i < predicate_count; ++i) {
      *offset += 1;  // predicate kind
      uint8_t arg_count = bytes[(*offset)++];
      for (uint8_t arg_index = 0; arg_index < arg_count; ++arg_index) {
        *offset += 1;  // argument tag
        ReadUVarint(bytes, offset);
      }
    }
  }

  void SkipAttributeValue(const std::vector<uint8_t>& bytes, size_t* offset,
                          uint8_t kind) {
    switch (kind) {
      case 0:  // I64.
        ReadUVarint(bytes, offset);
        break;
      case 1:  // F64.
        *offset += sizeof(double);
        break;
      case 2:   // STRING.
      case 4:   // ENUM.
      case 6:   // SYMBOL.
      case 7:   // TYPE.
      case 10:  // ENCODING.
        ReadUVarint(bytes, offset);
        break;
      case 3:  // BOOL.
        *offset += 1;
        break;
      case 5: {  // I64_ARRAY.
        uint64_t count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < count; ++i) {
          ReadUVarint(bytes, offset);
        }
        break;
      }
      case 8:  // PREDICATE_LIST.
        SkipPredicateList(bytes, offset);
        break;
      case 9: {  // DICT.
        uint64_t entry_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < entry_count; ++i) {
          ReadUVarint(bytes, offset);
          uint8_t value_kind = bytes[(*offset)++];
          SkipAttributeValue(bytes, offset, value_kind);
        }
        break;
      }
      default:
        ADD_FAILURE() << "unknown attribute kind in test helper: "
                      << (unsigned)kind;
        break;
    }
  }

  void SkipAttributeEntries(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t attr_count = ReadUVarint(bytes, offset);
    for (uint64_t i = 0; i < attr_count; ++i) {
      ReadUVarint(bytes, offset);  // key_id
      uint8_t value_kind = bytes[(*offset)++];
      SkipAttributeValue(bytes, offset, value_kind);
    }
  }

  size_t FunctionBodyLengthOffset(const std::vector<uint8_t>& bytes,
                                  uint64_t symbol_index) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
    uint64_t symbol_count = ReadUVarint(bytes, &offset);
    EXPECT_GT(symbol_count, symbol_index);
    uint64_t import_count = ReadUVarint(bytes, &offset);
    uint64_t export_count = ReadUVarint(bytes, &offset);
    offset += (import_count + export_count) * sizeof(uint64_t);
    for (uint64_t i = 0; i < symbol_count; ++i) {
      ReadUVarint(bytes, &offset);  // name_id
      uint8_t kind = bytes[offset++];
      offset += 1;  // visibility
      uint16_t flags = ReadU16LE(bytes, offset);
      offset += 2;
      if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
        ReadUVarint(bytes, &offset);
        ReadUVarint(bytes, &offset);
      }
      EXPECT_LE(kind, LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL);
      ReadUVarint(bytes, &offset);  // def_op_table_index_plus1
      SkipCommentList(bytes, &offset);
      offset += 1;  // calling_convention
      offset += 1;  // purity
      uint64_t arg_count = ReadUVarint(bytes, &offset);
      uint64_t result_count = ReadUVarint(bytes, &offset);
      for (uint64_t arg = 0; arg < arg_count; ++arg) {
        ReadValueDefOffsets(bytes, &offset);
      }
      for (uint64_t result = 0; result < result_count; ++result) {
        uint8_t is_tied = bytes[offset++];
        ReadValueDefOffsets(bytes, &offset);
        if (is_tied) ReadUVarint(bytes, &offset);
      }
      ReadUVarint(bytes, &offset);  // tied_result_count
      SkipPredicateList(bytes, &offset);
      if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
          kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
        ReadUVarint(bytes, &offset);  // implements_op_name
        ReadUVarint(bytes, &offset);  // priority
      }
      SkipAttributeEntries(bytes, &offset);
      uint8_t has_body = bytes[offset++];
      if (!has_body) continue;
      offset += sizeof(uint64_t);  // ir_offset
      if (i == symbol_index) return offset;
      offset += sizeof(uint32_t);  // ir_length
    }
    ADD_FAILURE() << "symbol has no body: " << symbol_index;
    return 0;
  }

  size_t FileHeaderEnd(const std::vector<uint8_t>& bytes) {
    size_t offset = 16;
    while (offset < bytes.size() && bytes[offset] != 0) {
      ++offset;
    }
    ++offset;
    return (offset + 7) & ~(size_t)7;
  }

  size_t ModuleDirectoryOffset(const std::vector<uint8_t>& bytes) {
    return FileHeaderEnd(bytes);
  }

  uint64_t ModuleOffset(const std::vector<uint8_t>& bytes) {
    return ReadU64LE(bytes, ModuleDirectoryOffset(bytes) + 8);
  }

  uint64_t ModuleLength(const std::vector<uint8_t>& bytes) {
    return ReadU64LE(bytes, ModuleDirectoryOffset(bytes) + 16);
  }

  std::vector<uint8_t> CombineModuleBytecode(const std::vector<uint8_t>& first,
                                             iree_string_view_t first_name,
                                             const std::vector<uint8_t>& second,
                                             iree_string_view_t second_name) {
    uint64_t first_module_offset = ModuleOffset(first);
    uint64_t first_module_length = ModuleLength(first);
    uint64_t second_module_offset = ModuleOffset(second);
    uint64_t second_module_length = ModuleLength(second);
    EXPECT_LE(first_module_offset + first_module_length, first.size());
    EXPECT_LE(second_module_offset + second_module_length, second.size());

    std::vector<uint8_t> bytes;
    const auto* magic = reinterpret_cast<const uint8_t*>(LOOM_BYTECODE_MAGIC);
    bytes.insert(bytes.end(), magic, magic + LOOM_BYTECODE_MAGIC_LENGTH);
    AppendU8(&bytes, LOOM_BYTECODE_FORMAT_VERSION);
    AppendU8(&bytes, first[5]);
    AppendU16LE(&bytes, 2);
    uint32_t string_pool_length =
        (uint32_t)(first_name.size + second_name.size);
    AppendU32LE(&bytes, string_pool_length);
    AppendU32LE(&bytes, 0);
    AppendString(&bytes, IREE_SV("reader-test"));
    AppendU8(&bytes, 0);
    PadToAlignment(&bytes, 8);

    size_t module_directory_offset = bytes.size();
    bytes.resize(bytes.size() + 2 * sizeof(loom_bytecode_module_dir_entry_t));

    AppendString(&bytes, first_name);
    AppendString(&bytes, second_name);
    PadToAlignment(&bytes, 8);

    uint64_t output_first_module_offset = bytes.size();
    const uint8_t* first_module_data = first.data() + first_module_offset;
    bytes.insert(bytes.end(), first_module_data,
                 first_module_data + first_module_length);
    uint64_t output_second_module_offset = bytes.size();
    const uint8_t* second_module_data = second.data() + second_module_offset;
    bytes.insert(bytes.end(), second_module_data,
                 second_module_data + second_module_length);

    WriteU32LE(&bytes, module_directory_offset, 0);
    WriteU16LE(&bytes, module_directory_offset + 4, (uint16_t)first_name.size);
    WriteU16LE(&bytes, module_directory_offset + 6, 0);
    WriteU64LE(&bytes, module_directory_offset + 8, output_first_module_offset);
    WriteU64LE(&bytes, module_directory_offset + 16, first_module_length);
    WriteU32LE(&bytes, module_directory_offset + 24, (uint32_t)first_name.size);
    WriteU16LE(&bytes, module_directory_offset + 28,
               (uint16_t)second_name.size);
    WriteU16LE(&bytes, module_directory_offset + 30, 0);
    WriteU64LE(&bytes, module_directory_offset + 32,
               output_second_module_offset);
    WriteU64LE(&bytes, module_directory_offset + 40, second_module_length);
    return bytes;
  }

  std::vector<SectionEntry> ReadSectionDirectory(
      const std::vector<uint8_t>& bytes) {
    uint64_t module_offset = ModuleOffset(bytes);
    size_t section_offset = (size_t)module_offset;
    uint64_t section_count = ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);

    std::vector<SectionEntry> entries;
    entries.reserve((size_t)section_count);
    for (uint64_t i = 0; i < section_count; ++i) {
      entries.push_back(SectionEntry{
          /*.kind=*/ReadU16LE(bytes, section_offset),
          /*.directory_entry_offset=*/section_offset,
          /*.offset=*/ReadU64LE(bytes, section_offset + 8),
          /*.length=*/ReadU64LE(bytes, section_offset + 16),
      });
      section_offset += sizeof(loom_bytecode_section_dir_entry_t);
    }
    return entries;
  }

  SectionEntry FindSection(const std::vector<uint8_t>& bytes, uint16_t kind) {
    for (SectionEntry entry : ReadSectionDirectory(bytes)) {
      if (entry.kind == kind) return entry;
    }
    return SectionEntry{};
  }

  size_t SectionPayloadOffset(const std::vector<uint8_t>& bytes,
                              uint16_t kind) {
    SectionEntry entry = FindSection(bytes, kind);
    return (size_t)ModuleOffset(bytes) + (size_t)entry.offset;
  }

  size_t RootBlockValueListOffset(const std::vector<uint8_t>& bytes,
                                  uint64_t* out_arg_count) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_IR);
    ReadUVarint(bytes, &offset);  // value_count
    ReadUVarint(bytes, &offset);  // region_count
    ReadUVarint(bytes, &offset);  // block_count
    ReadUVarint(bytes, &offset);  // op_count
    uint64_t root_region_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_region_count, 1u);
    uint64_t root_region_index = ReadUVarint(bytes, &offset);
    EXPECT_EQ(root_region_index, 0u);
    uint64_t root_block_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_block_count, 1u);
    uint8_t has_label = bytes[offset++];
    if (has_label) {
      ReadUVarint(bytes, &offset);
    }
    SkipCommentList(bytes, &offset);
    *out_arg_count = ReadUVarint(bytes, &offset);
    return offset;
  }

  ValueDefOffsets RootBlockArgValueDefOffsets(const std::vector<uint8_t>& bytes,
                                              uint64_t arg_index) {
    uint64_t arg_count = 0;
    size_t offset = RootBlockValueListOffset(bytes, &arg_count);
    EXPECT_GT(arg_count, arg_index);
    ValueDefOffsets value_def;
    for (uint64_t i = 0; i <= arg_index; ++i) {
      value_def = ReadValueDefOffsets(bytes, &offset);
    }
    return value_def;
  }

  size_t FirstBodyOperandRefOffset(const std::vector<uint8_t>& bytes) {
    uint64_t arg_count = 0;
    size_t offset = RootBlockValueListOffset(bytes, &arg_count);
    for (uint64_t i = 0; i < arg_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t op_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(op_count, 1u);
    ReadUVarint(bytes, &offset);  // op_table_index_plus1
    ++offset;                     // flags
    ReadUVarint(bytes, &offset);  // location_id
    SkipCommentList(bytes, &offset);
    uint64_t operand_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(operand_count, 1u);
    return offset;
  }

  size_t FirstBodySegmentCountOffset(const std::vector<uint8_t>& bytes,
                                     uint8_t segment_index) {
    uint64_t arg_count = 0;
    size_t offset = RootBlockValueListOffset(bytes, &arg_count);
    for (uint64_t i = 0; i < arg_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t op_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(op_count, 1u);
    ReadUVarint(bytes, &offset);  // op_table_index_plus1
    ++offset;                     // flags
    ReadUVarint(bytes, &offset);  // location_id
    SkipCommentList(bytes, &offset);
    uint64_t operand_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < operand_count; ++i) {
      ReadUVarint(bytes, &offset);
    }
    for (uint8_t i = 0; i < segment_index; ++i) {
      ReadUVarint(bytes, &offset);
    }
    return offset;
  }

  size_t FirstBodyOpTiedOperandOffset(const std::vector<uint8_t>& bytes) {
    uint64_t arg_count = 0;
    size_t offset = RootBlockValueListOffset(bytes, &arg_count);
    for (uint64_t i = 0; i < arg_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t op_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(op_count, 1u);
    ReadUVarint(bytes, &offset);  // op_table_index_plus1
    ++offset;                     // flags
    ReadUVarint(bytes, &offset);  // location_id
    SkipCommentList(bytes, &offset);
    uint64_t operand_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < operand_count; ++i) {
      ReadUVarint(bytes, &offset);
    }
    uint64_t successor_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < successor_count; ++i) {
      ReadUVarint(bytes, &offset);
    }
    uint64_t result_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < result_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t tied_result_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(tied_result_count, 1u);
    ReadUVarint(bytes, &offset);  // result_index
    return offset;
  }

  BodyOpAttrOffsets FirstBodyOpAttrOffsets(const std::vector<uint8_t>& bytes) {
    uint64_t arg_count = 0;
    size_t offset = RootBlockValueListOffset(bytes, &arg_count);
    for (uint64_t i = 0; i < arg_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t op_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(op_count, 1u);
    ReadUVarint(bytes, &offset);  // op_table_index_plus1
    ++offset;                     // flags
    ReadUVarint(bytes, &offset);  // location_id
    SkipCommentList(bytes, &offset);
    uint64_t operand_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < operand_count; ++i) {
      ReadUVarint(bytes, &offset);
    }
    uint64_t successor_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < successor_count; ++i) {
      ReadUVarint(bytes, &offset);
    }
    uint64_t result_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < result_count; ++i) {
      ReadValueDefOffsets(bytes, &offset);
    }
    uint64_t tied_result_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < tied_result_count; ++i) {
      ReadUVarint(bytes, &offset);
      ReadUVarint(bytes, &offset);
    }

    BodyOpAttrOffsets attr_offsets;
    uint64_t attr_count = ReadUVarint(bytes, &offset);
    EXPECT_EQ(attr_count, 1u);

    ReadUVarint(bytes, &offset);
    attr_offsets.attr_kind = offset;
    uint8_t attr_kind = bytes[offset++];
    EXPECT_EQ(attr_kind, 9u);
    uint64_t dict_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(dict_count, 2u);

    ReadUVarint(bytes, &offset);
    uint8_t dict_first_kind = bytes[offset++];
    SkipAttributeValue(bytes, &offset, dict_first_kind);

    ReadUVarint(bytes, &offset);
    uint8_t dict_second_kind = bytes[offset++];
    EXPECT_EQ(dict_second_kind, 9u);
    uint64_t nested_dict_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(nested_dict_count, 2u);

    attr_offsets.nested_dict_first_key = offset;
    ReadUVarint(bytes, &offset);
    uint8_t nested_dict_first_kind = bytes[offset++];
    SkipAttributeValue(bytes, &offset, nested_dict_first_kind);
    attr_offsets.nested_dict_second_key = offset;
    return attr_offsets;
  }

  struct GlobalPayloadOffsets {
    // Byte offset of the defining op table index plus one.
    size_t op_table_index_plus1 = 0;
    // Byte offset of the serialized global result count.
    size_t result_count = 0;
    // Byte offset of the serialized global local value count.
    size_t local_value_count = 0;
  };

  GlobalPayloadOffsets FirstGlobalPayloadOffsets(
      const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
    uint64_t symbol_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(symbol_count, 1u);
    uint64_t import_count = ReadUVarint(bytes, &offset);
    uint64_t export_count = ReadUVarint(bytes, &offset);
    offset += (import_count + export_count) * sizeof(uint64_t);

    ReadUVarint(bytes, &offset);  // name_id
    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_SYMBOL_GLOBAL);
    offset += 1;                 // visibility
    offset += sizeof(uint16_t);  // flags

    GlobalPayloadOffsets payload_offsets;
    payload_offsets.op_table_index_plus1 = offset;
    ReadUVarint(bytes, &offset);
    SkipCommentList(bytes, &offset);
    payload_offsets.result_count = offset;
    ReadUVarint(bytes, &offset);
    payload_offsets.local_value_count = offset;
    return payload_offsets;
  }

  size_t LastOpRegionCountOffsetInRegion(const std::vector<uint8_t>& bytes,
                                         size_t* offset) {
    size_t last_region_count_offset = 0;
    uint64_t block_count = ReadUVarint(bytes, offset);
    for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
      uint8_t has_label = bytes[(*offset)++];
      if (has_label) {
        ReadUVarint(bytes, offset);
      }
      SkipCommentList(bytes, offset);
      uint64_t arg_count = ReadUVarint(bytes, offset);
      for (uint64_t arg_index = 0; arg_index < arg_count; ++arg_index) {
        ReadValueDefOffsets(bytes, offset);
      }
      uint64_t op_count = ReadUVarint(bytes, offset);
      for (uint64_t op_index = 0; op_index < op_count; ++op_index) {
        ReadUVarint(bytes, offset);  // op_table_index_plus1
        *offset += 1;                // flags
        ReadUVarint(bytes, offset);  // location_id
        SkipCommentList(bytes, offset);
        uint64_t operand_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < operand_count; ++i) {
          ReadUVarint(bytes, offset);
        }
        uint64_t successor_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < successor_count; ++i) {
          ReadUVarint(bytes, offset);
        }
        uint64_t result_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < result_count; ++i) {
          ReadValueDefOffsets(bytes, offset);
        }
        uint64_t tied_result_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < tied_result_count; ++i) {
          ReadUVarint(bytes, offset);
          ReadUVarint(bytes, offset);
        }
        uint64_t attr_count = ReadUVarint(bytes, offset);
        EXPECT_EQ(attr_count, 0u);
        last_region_count_offset = *offset;
        uint64_t region_count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < region_count; ++i) {
          size_t nested_last = LastOpRegionCountOffsetInRegion(bytes, offset);
          if (nested_last != 0) {
            last_region_count_offset = nested_last;
          }
        }
      }
    }
    return last_region_count_offset;
  }

  size_t LastOpRegionCountOffset(const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_IR);
    ReadUVarint(bytes, &offset);  // value_count
    ReadUVarint(bytes, &offset);  // region_count
    ReadUVarint(bytes, &offset);  // block_count
    ReadUVarint(bytes, &offset);  // op_count
    uint64_t root_region_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_region_count, 1u);
    uint64_t root_region_index = ReadUVarint(bytes, &offset);
    EXPECT_EQ(root_region_index, 0u);
    return LastOpRegionCountOffsetInRegion(bytes, &offset);
  }

  void ReplaceBytes(std::vector<uint8_t>* bytes, const char* from,
                    const char* to) {
    size_t length = std::strlen(from);
    ASSERT_EQ(length, std::strlen(to));
    auto it = std::search(bytes->begin(), bytes->end(), from, from + length);
    ASSERT_NE(it, bytes->end());
    std::copy(to, to + length, it);
  }

  void ExpectReadError(const std::vector<uint8_t>& bytes,
                       const char* expected_error_id) {
    std::vector<std::string> error_ids;
    loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);
    EXPECT_GT(result.error_count, 0u);
    ASSERT_FALSE(error_ids.empty());
    EXPECT_EQ(error_ids.front(), expected_error_id);
  }

  void ExpectReadModuleError(const std::vector<uint8_t>& bytes,
                             const char* expected_error_id) {
    std::vector<std::string> error_ids;
    loom_module_t* module = nullptr;
    loom_bytecode_read_result_t result = ReadModule(bytes, &module, &error_ids);
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(module, nullptr);
    if (module) {
      loom_module_free(module);
    }
    ASSERT_FALSE(error_ids.empty());
    EXPECT_EQ(error_ids.front(), expected_error_id);
  }

  void ExpectCanonicalBytecodeRoundTrip(loom_module_t* module) {
    auto first = WriteModule(module);
    loom_module_t* read_module = nullptr;
    std::vector<std::string> error_ids;
    loom_bytecode_read_result_t result =
        ReadModule(first, &read_module, &error_ids);
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_TRUE(error_ids.empty());
    ASSERT_NE(read_module, nullptr);

    auto second = WriteModule(read_module);
    EXPECT_EQ(first, second);

    loom_module_free(read_module);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ReaderTest, AcceptsEmptyModuleMetadata) {
  loom_module_t* module = CreateModule("empty");
  auto bytes = WriteModule(module);

  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(result.module_count, 1u);
  EXPECT_EQ(result.location_mode, LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS);
  EXPECT_EQ(result.first_module.string_count, 2u);
  EXPECT_EQ(result.first_module.symbol_count, 0u);

  loom_module_free(module);
}

TEST_F(ReaderTest, AcceptsFunctionMetadata) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);

  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(result.first_module.symbol_count, 1u);
  EXPECT_GT(result.first_module.type_count, 0u);
  EXPECT_GT(result.first_module.op_name_count, 0u);

  loom_module_free(module);
}

TEST_F(ReaderTest, AcceptsGlobalMetadata) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);

  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(result.first_module.symbol_count, 1u);
  EXPECT_EQ(result.first_module.value_count, 1u);
  EXPECT_GT(result.first_module.type_count, 0u);
  EXPECT_GT(result.first_module.op_name_count, 0u);

  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsFunctionModuleIndex) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(metadata.format_version, LOOM_BYTECODE_FORMAT_VERSION);
  EXPECT_EQ(metadata.location_mode,
            LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS);
  EXPECT_EQ(metadata.module_count, 1u);
  ASSERT_NE(metadata.modules, nullptr);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  EXPECT_TRUE(
      iree_string_view_equal(module_metadata.name, IREE_SV("reader_func")));
  EXPECT_GT(module_metadata.section_count, 0u);
  ASSERT_NE(module_metadata.sections, nullptr);
  EXPECT_EQ(module_metadata.summary.symbol_count, 1u);
  EXPECT_EQ(module_metadata.symbol_count, 1u);
  EXPECT_EQ(module_metadata.import_count, 0u);
  EXPECT_EQ(module_metadata.export_count, 1u);
  ASSERT_NE(module_metadata.export_symbol_indices, nullptr);
  EXPECT_EQ(module_metadata.export_symbol_indices[0], 0u);

  ASSERT_NE(module_metadata.symbols, nullptr);
  const loom_bytecode_symbol_metadata_t& symbol = module_metadata.symbols[0];
  EXPECT_TRUE(iree_string_view_equal(symbol.name, IREE_SV("f")));
  EXPECT_EQ(symbol.kind, LOOM_BYTECODE_SYMBOL_FUNC_DEF);
  EXPECT_EQ(symbol.visibility, LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC);
  EXPECT_TRUE(
      iree_all_bits_set(symbol.flags, LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC));
  EXPECT_TRUE(
      iree_string_view_equal(symbol.defining_op_name, IREE_SV("test.func")));
  EXPECT_EQ(symbol.argument_count, 1u);
  EXPECT_EQ(symbol.result_count, 1u);
  EXPECT_TRUE(symbol.has_body);
  EXPECT_GT(symbol.body_length, 0u);
  EXPECT_GE(symbol.body_absolute_offset, module_metadata.offset);
  EXPECT_GT(symbol.entry_length, 0u);

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsImportOffsetTableInIndex) {
  loom_module_t* module = CreateImportedFunctionModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  EXPECT_EQ(module_metadata.symbol_count, 1u);
  EXPECT_EQ(module_metadata.import_count, 1u);
  EXPECT_EQ(module_metadata.export_count, 0u);
  ASSERT_NE(module_metadata.import_symbol_indices, nullptr);
  EXPECT_EQ(module_metadata.import_symbol_indices[0], 0u);

  ASSERT_NE(module_metadata.symbols, nullptr);
  const loom_bytecode_symbol_metadata_t& symbol = module_metadata.symbols[0];
  EXPECT_TRUE(iree_string_view_equal(symbol.name, IREE_SV("decl")));
  EXPECT_EQ(symbol.kind, LOOM_BYTECODE_SYMBOL_FUNC_DECL);
  EXPECT_TRUE(
      iree_all_bits_set(symbol.flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT));
  EXPECT_TRUE(
      iree_all_bits_set(symbol.flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL));
  EXPECT_TRUE(
      iree_string_view_equal(symbol.import_module, IREE_SV("kernel_lib")));
  EXPECT_TRUE(
      iree_string_view_equal(symbol.import_symbol, IREE_SV("extern_f")));
  EXPECT_TRUE(
      iree_string_view_equal(symbol.defining_op_name, IREE_SV("func.decl")));
  EXPECT_EQ(symbol.argument_count, 1u);
  EXPECT_EQ(symbol.result_count, 1u);
  EXPECT_FALSE(symbol.has_body);

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsImportOffsetTableMismatchedEntry) {
  loom_module_t* module = CreateImportedFunctionModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  uint64_t symbol_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(symbol_count, 1u);
  uint64_t import_count = ReadUVarint(bytes, &offset);
  uint64_t export_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(import_count, 1u);
  ASSERT_EQ(export_count, 0u);
  WriteU64LE(&bytes, offset, 1);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsMultiModuleIndexAndMaterializesByOrdinal) {
  loom_module_t* first_module = CreateModule("first_original");
  loom_module_t* second_module = CreateFunctionModule();
  auto first_bytes = WriteModule(first_module);
  auto second_bytes = WriteModule(second_module);
  auto bytes = CombineModuleBytecode(first_bytes, IREE_SV("module_a"),
                                     second_bytes, IREE_SV("module_b"));

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(metadata.modules[0].name, IREE_SV("module_a")));
  EXPECT_TRUE(
      iree_string_view_equal(metadata.modules[1].name, IREE_SV("module_b")));
  EXPECT_EQ(metadata.modules[0].symbol_count, 0u);
  EXPECT_EQ(metadata.modules[1].symbol_count, 1u);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> ordinal_error_ids;
  loom_bytecode_read_result_t ordinal_result = ReadModuleOrdinal(
      bytes, 1, &read_module, &ordinal_error_ids, /*verify_module=*/true);
  EXPECT_EQ(ordinal_result.error_count, 0u);
  EXPECT_TRUE(ordinal_error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[read_module->name_id], IREE_SV("module_b")));
  ASSERT_EQ(read_module->symbols.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[read_module->symbols.entries[0].name_id],
      IREE_SV("f")));
  loom_module_free(read_module);

  loom_module_t* rejected_module = nullptr;
  std::vector<std::string> single_error_ids;
  loom_bytecode_read_result_t single_result =
      ReadModule(bytes, &rejected_module, &single_error_ids);
  EXPECT_GT(single_result.error_count, 0u);
  EXPECT_EQ(rejected_module, nullptr);
  ASSERT_FALSE(single_error_ids.empty());
  EXPECT_EQ(single_error_ids.front(), "ERR_BYTECODE_006");

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(first_module);
  loom_module_free(second_module);
}

TEST_F(ReaderTest, ReadsFunctionBodyModule) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids,
                 /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  EXPECT_EQ(read_module->symbols.entries[0].kind, LOOM_SYMBOL_FUNC_DEF);
  ASSERT_NE(read_module->symbols.entries[0].defining_op, nullptr);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  loom_block_t* entry = loom_region_entry_block(body);
  ASSERT_EQ(entry->arg_count, 1u);
  ASSERT_EQ(entry->op_count, 2u);
  loom_op_t* body_op = entry->first_op;
  ASSERT_NE(body_op, nullptr);
  EXPECT_TRUE(loom_test_addi_isa(body_op));
  EXPECT_EQ(loom_test_addi_lhs(body_op), loom_test_addi_rhs(body_op));
  ASSERT_NE(entry->last_op, nullptr);
  EXPECT_TRUE(loom_test_yield_isa(entry->last_op));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsSegmentedOperandCounts) {
  loom_module_t* module = CreateSegmentedModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids,
                 /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);

  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(func_op, nullptr);
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_op_t* op = loom_region_entry_block(body)->first_op;
  ASSERT_NE(op, nullptr);
  ASSERT_TRUE(loom_test_segmented_isa(op));
  const uint16_t* counts = loom_op_const_operand_segment_counts(op);
  EXPECT_EQ(counts[0], 1u);
  EXPECT_EQ(counts[1], 1u);
  EXPECT_EQ(counts[2], 2u);
  EXPECT_EQ(counts[3], 1u);
  EXPECT_EQ(loom_test_segmented_lhs(op).count, 2u);
  EXPECT_EQ(loom_test_segmented_rhs(op).count, 1u);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, EnumAttributesPreserveFutureOrdinals) {
  loom_module_t* module = CreateTestRecordWithFutureEnumOrdinal();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  if (!read_module) {
    loom_module_free(module);
    FAIL() << ::testing::PrintToString(error_ids);
  }
  ASSERT_EQ(read_module->symbols.count, 1u);

  loom_op_t* read_record = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(read_record, nullptr);
  ASSERT_TRUE(loom_test_record_isa(read_record));
  EXPECT_EQ(loom_test_record_kind(read_record), 250u);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsGlobalSymbolModule) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids,
                 /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  const loom_symbol_t& symbol = read_module->symbols.entries[0];
  EXPECT_EQ(symbol.kind, LOOM_SYMBOL_GLOBAL);
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[symbol.name_id], IREE_SV("answer")));
  ASSERT_NE(symbol.defining_op, nullptr);
  ASSERT_TRUE(loom_global_constant_isa(symbol.defining_op));
  loom_attribute_t initializer =
      loom_global_constant_initializer(symbol.defining_op);
  ASSERT_EQ(initializer.kind, LOOM_ATTR_I64);
  EXPECT_EQ(initializer.i64, 42);
  loom_type_t type = loom_module_value_type(
      read_module, loom_global_constant_type(symbol.defining_op));
  EXPECT_TRUE(loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsDynamicGlobalSymbolModule) {
  loom_module_t* module = CreateDynamicGlobalModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids,
                 /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  const loom_symbol_t& symbol = read_module->symbols.entries[0];
  ASSERT_NE(symbol.defining_op, nullptr);
  ASSERT_TRUE(loom_global_constant_isa(symbol.defining_op));

  loom_type_t type = loom_module_value_type(
      read_module, loom_global_constant_type(symbol.defining_op));
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(type, 0));
  loom_value_id_t dim_id = loom_type_dim_value_id_at(type, 0);
  ASSERT_LT(dim_id, read_module->values.count);
  const loom_value_t& dim_value = read_module->values.entries[dim_id];
  EXPECT_TRUE(loom_type_equal(dim_value.type,
                              loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
  ASSERT_NE(dim_value.name_id, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[dim_value.name_id], IREE_SV("n")));

  const loom_attribute_t* attrs = loom_op_attrs(symbol.defining_op);
  ASSERT_EQ(attrs[1].kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_EQ(attrs[1].count, 1u);
  const loom_predicate_t& predicate = attrs[1].predicate_list[0];
  EXPECT_EQ(predicate.kind, LOOM_PREDICATE_MUL);
  EXPECT_EQ(predicate.arg_count, 2u);
  EXPECT_EQ(predicate.arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ(predicate.args[0], (int64_t)dim_id);
  EXPECT_EQ(predicate.arg_tags[1], LOOM_PRED_ARG_CONST);
  EXPECT_EQ(predicate.args[1], 16);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsLocationTablesWithModuleSources) {
  loom_module_t* module = CreateLocatedModule();
  auto bytes = WriteModule(module);

  loom_context_t read_context;
  InitializeBytecodeTestContext(&read_context);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_context, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(read_module->sources.entries[0],
                                     IREE_SV("model.loom")));
  ASSERT_EQ(read_module->locations.count, 2u);
  const loom_location_entry_t& file_location =
      read_module->locations.entries[1];
  EXPECT_EQ(file_location.kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(file_location.file.source_id, 0u);
  EXPECT_EQ(file_location.file.start_line, 1u);
  EXPECT_EQ(file_location.file.end_col, 2u);

  loom_module_free(read_module);
  loom_context_deinitialize(&read_context);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsMultiBlockFunctionBodyModule) {
  loom_module_t* module = CreateMultiBlockFunctionModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 2u);
  loom_block_t* second_block = loom_region_block(body, 1);
  ASSERT_EQ(second_block->op_count, 1u);
  EXPECT_TRUE(loom_test_use_isa(second_block->first_op));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsFunctionBodySuccessorReferences) {
  loom_module_t* module = CreateSuccessorFunctionModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 2u);
  loom_block_t* entry_block = loom_region_entry_block(body);
  ASSERT_NE(entry_block, nullptr);
  ASSERT_EQ(entry_block->op_count, 1u);
  loom_op_t* br_op = loom_block_op(entry_block, 0);
  ASSERT_TRUE(loom_test_br_isa(br_op));
  ASSERT_EQ(br_op->successor_count, 1u);
  EXPECT_EQ(loom_test_br_dest(br_op), loom_region_block(body, 1));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsNestedRegionBodyModule) {
  loom_module_t* module = CreateTiedBodyOpModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry = loom_region_entry_block(body);
  ASSERT_EQ(entry->op_count, 1u);
  ASSERT_TRUE(loom_test_map_isa(entry->first_op));
  loom_region_t* map_body = loom_test_map_body(entry->first_op);
  ASSERT_NE(map_body, nullptr);
  EXPECT_EQ(map_body->block_count, 1u);
  loom_block_t* map_entry = loom_region_entry_block(map_body);
  ASSERT_EQ(map_entry->op_count, 1u);
  EXPECT_TRUE(loom_test_use_isa(map_entry->first_op));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsDynamicDimBindings) {
  loom_module_t* dynamic_module = CreateDynamicDimFunctionModule();
  auto dynamic_bytes = WriteModule(dynamic_module);

  loom_module_t* read_dynamic_module = nullptr;
  std::vector<std::string> dynamic_errors;
  loom_bytecode_read_result_t dynamic_result =
      ReadModule(dynamic_bytes, &read_dynamic_module, &dynamic_errors);

  EXPECT_EQ(dynamic_result.error_count, 0u);
  EXPECT_TRUE(dynamic_errors.empty());
  ASSERT_NE(read_dynamic_module, nullptr);
  loom_op_t* dynamic_func = read_dynamic_module->symbols.entries[0].defining_op;
  loom_func_like_t dynamic_func_like =
      loom_func_like_cast(read_dynamic_module, dynamic_func);
  uint16_t dynamic_arg_count = 0;
  const loom_value_id_t* dynamic_arg_ids =
      loom_func_like_arg_ids(dynamic_func_like, &dynamic_arg_count);
  ASSERT_EQ(dynamic_arg_count, 2u);
  loom_type_t vector_type =
      loom_module_value_type(read_dynamic_module, dynamic_arg_ids[1]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(vector_type, 0));
  EXPECT_EQ(loom_dim_value_id(loom_type_dim(vector_type, 0)),
            dynamic_arg_ids[0]);

  loom_module_free(read_dynamic_module);
  loom_module_free(dynamic_module);
}

TEST_F(ReaderTest, ReadsSsaEncodingBindings) {
  loom_module_t* encoding_module = CreateSsaEncodingFunctionModule();
  auto encoding_bytes = WriteModule(encoding_module);

  loom_module_t* read_encoding_module = nullptr;
  std::vector<std::string> encoding_errors;
  loom_bytecode_read_result_t encoding_result =
      ReadModule(encoding_bytes, &read_encoding_module, &encoding_errors);

  EXPECT_EQ(encoding_result.error_count, 0u);
  EXPECT_TRUE(encoding_errors.empty());
  ASSERT_NE(read_encoding_module, nullptr);
  loom_op_t* encoding_func =
      read_encoding_module->symbols.entries[0].defining_op;
  loom_func_like_t encoding_func_like =
      loom_func_like_cast(read_encoding_module, encoding_func);
  uint16_t encoding_arg_count = 0;
  const loom_value_id_t* encoding_arg_ids =
      loom_func_like_arg_ids(encoding_func_like, &encoding_arg_count);
  ASSERT_EQ(encoding_arg_count, 2u);
  loom_type_t view_type =
      loom_module_value_type(read_encoding_module, encoding_arg_ids[1]);
  ASSERT_TRUE(loom_type_has_ssa_encoding(view_type));
  EXPECT_EQ(loom_type_encoding_value_id(view_type), encoding_arg_ids[0]);

  loom_module_free(read_encoding_module);
  loom_module_free(encoding_module);
}

TEST_F(ReaderTest, ReadsCoResultDynamicDimBindings) {
  loom_module_t* module = CreateCoResultDimFunctionModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry = loom_region_entry_block(body);
  ASSERT_GE(entry->op_count, 1u);
  loom_op_t* deflate_op = entry->first_op;
  ASSERT_TRUE(loom_test_deflate_isa(deflate_op));
  loom_value_slice_t results = loom_test_deflate_results(deflate_op);
  ASSERT_EQ(results.count, 2u);
  loom_type_t output_type =
      loom_module_value_type(read_module, results.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(output_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(output_type, 0), results.values[1]);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, CanonicalRoundTripPreservesBodyShapes) {
  loom_module_t* simple_module = CreateFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(simple_module);
  loom_module_free(simple_module);

  loom_module_t* multi_block_module = CreateMultiBlockFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(multi_block_module);
  loom_module_free(multi_block_module);

  loom_module_t* successor_module = CreateSuccessorFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(successor_module);
  loom_module_free(successor_module);

  loom_module_t* nested_module = CreateTiedBodyOpModule();
  ExpectCanonicalBytecodeRoundTrip(nested_module);
  loom_module_free(nested_module);

  loom_module_t* co_result_module = CreateCoResultDimFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(co_result_module);
  loom_module_free(co_result_module);
}

TEST_F(ReaderTest, CanonicalRoundTripPreservesTypesAttrsAndPredicates) {
  loom_module_t* global_module = CreateGlobalModule();
  ExpectCanonicalBytecodeRoundTrip(global_module);
  loom_module_free(global_module);

  loom_module_t* dynamic_global_module = CreateDynamicGlobalModule();
  ExpectCanonicalBytecodeRoundTrip(dynamic_global_module);
  loom_module_free(dynamic_global_module);

  loom_module_t* register_decl_module = CreateRegisterDeclModule();
  ExpectCanonicalBytecodeRoundTrip(register_decl_module);
  loom_module_free(register_decl_module);

  loom_module_t* dynamic_module = CreateDynamicDimFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(dynamic_module);
  loom_module_free(dynamic_module);

  loom_module_t* encoding_module = CreateSsaEncodingFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(encoding_module);
  loom_module_free(encoding_module);

  loom_module_t* attr_module = CreateAttributeFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(attr_module);
  loom_module_free(attr_module);

  loom_module_t* predicate_module = CreatePredicateFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(predicate_module);
  loom_module_free(predicate_module);
}

TEST_F(ReaderTest, CanonicalRoundTripPreservesLocations) {
  loom_module_t* module = CreateLocatedModule();
  ExpectCanonicalBytecodeRoundTrip(module);
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidBodyValueReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t operand_offset = FirstBodyOperandRefOffset(bytes);
  bytes[operand_offset] = 0x7F;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidSegmentedOperandCount) {
  loom_module_t* module = CreateSegmentedModule();
  auto bytes = WriteModule(module);
  bytes[FirstBodySegmentCountOffset(bytes, /*segment_index=*/0)] = 0;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidGlobalDefiningOpReference) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);
  GlobalPayloadOffsets payload_offsets = FirstGlobalPayloadOffsets(bytes);
  ASSERT_LT(payload_offsets.op_table_index_plus1, bytes.size());
  bytes[payload_offsets.op_table_index_plus1] = 0;

  ExpectReadError(bytes, "ERR_BYTECODE_012");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsZeroGlobalResultCount) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);
  GlobalPayloadOffsets payload_offsets = FirstGlobalPayloadOffsets(bytes);
  ASSERT_LT(payload_offsets.result_count, bytes.size());
  ASSERT_EQ(bytes[payload_offsets.result_count], 1u);
  bytes[payload_offsets.result_count] = 0;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsGlobalLocalValueCountBelowResultCount) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);
  GlobalPayloadOffsets payload_offsets = FirstGlobalPayloadOffsets(bytes);
  ASSERT_LT(payload_offsets.local_value_count, bytes.size());
  ASSERT_EQ(bytes[payload_offsets.local_value_count], 1u);
  bytes[payload_offsets.local_value_count] = 0;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsBodySummaryCountExceedingBodyLength) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t value_count_offset =
      SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_IR);
  SectionEntry ir_section = FindSection(bytes, LOOM_BYTECODE_SECTION_IR);
  ASSERT_GT(0x7Fu, ir_section.length);
  ASSERT_LT(value_count_offset, bytes.size());
  bytes[value_count_offset] = 0x7F;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsMissingDynamicDimBinding) {
  loom_module_t* module = CreateDynamicDimFunctionModule();
  auto bytes = WriteModule(module);
  ValueDefOffsets vector_arg = RootBlockArgValueDefOffsets(bytes, 1);
  ASSERT_EQ(bytes[vector_arg.dim_binding_count], 1u);
  bytes[vector_arg.dim_binding_count] = 0;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsMissingSsaEncodingBinding) {
  loom_module_t* module = CreateSsaEncodingFunctionModule();
  auto bytes = WriteModule(module);
  ValueDefOffsets view_arg = RootBlockArgValueDefOffsets(bytes, 1);
  ASSERT_EQ(bytes[view_arg.encoding_binding], 1u);
  bytes[view_arg.encoding_binding] = 0;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsOutOfRangeBodyTiedResult) {
  loom_module_t* module = CreateTiedBodyOpModule();
  auto bytes = WriteModule(module);
  size_t tied_operand_offset = FirstBodyOpTiedOperandOffset(bytes);
  ASSERT_EQ(bytes[tied_operand_offset], 0u);
  bytes[tied_operand_offset] = 2;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidBodyAttributeKind) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.attr_kind] = 0x7F;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_011");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsNestedDictAttributeKeyOrder) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  ASSERT_LT(bytes[attr_offsets.nested_dict_first_key], 0x80u);
  ASSERT_LT(bytes[attr_offsets.nested_dict_second_key], 0x80u);
  bytes[attr_offsets.nested_dict_second_key] =
      bytes[attr_offsets.nested_dict_first_key];

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsRegionNestingLimit) {
  static constexpr uint32_t kBytecodeMaxRegionDepth = 256;
  loom_module_t* module =
      CreateDeepNestedFunctionModule(kBytecodeMaxRegionDepth - 1);
  auto bytes = WriteModule(module);
  size_t region_count_offset = LastOpRegionCountOffset(bytes);
  ASSERT_NE(region_count_offset, 0u);
  ASSERT_EQ(bytes[region_count_offset], 0u);
  bytes[region_count_offset] = 1;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsFunctionBodyTrailingBytes) {
  loom_module_t* module = CreateTwoFunctionModule();
  auto bytes = WriteModule(module);
  size_t first_length_offset = FunctionBodyLengthOffset(bytes, 0);
  uint32_t first_length = ReadU32LE(bytes, first_length_offset);
  WriteU32LE(&bytes, first_length_offset, first_length + 1);

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsTruncatedHeader) {
  std::vector<uint8_t> bytes(8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_003");
}

TEST_F(ReaderTest, RejectsInvalidMagic) {
  loom_module_t* module = CreateModule("magic");
  auto bytes = WriteModule(module);
  bytes[0] = 0;

  ExpectReadError(bytes, "ERR_BYTECODE_001");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnsupportedVersion) {
  loom_module_t* module = CreateModule("version");
  auto bytes = WriteModule(module);
  bytes[4] = LOOM_BYTECODE_FORMAT_VERSION + 1;

  ExpectReadError(bytes, "ERR_BYTECODE_002");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsFullLocationsModeUntilFieldSpansExist) {
  loom_module_t* module = CreateModule("full");
  auto bytes = WriteModule(module);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsModuleRangeBeforeMetadataEnd) {
  loom_module_t* module = CreateModule("range");
  auto bytes = WriteModule(module);
  WriteU64LE(&bytes, ModuleDirectoryOffset(bytes) + 8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsDuplicateSectionKind) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  auto sections = ReadSectionDirectory(bytes);
  ASSERT_GE(sections.size(), 2u);
  WriteU16LE(&bytes, sections[1].directory_entry_offset, sections[0].kind);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnsortedSectionRange) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  auto sections = ReadSectionDirectory(bytes);
  ASSERT_GE(sections.size(), 2u);
  WriteU64LE(&bytes, sections[1].directory_entry_offset + 8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsMissingRequiredSection) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  SectionEntry strings = FindSection(bytes, LOOM_BYTECODE_SECTION_STRINGS);
  ASSERT_NE(strings.length, 0u);
  WriteU16LE(&bytes, strings.directory_entry_offset,
             LOOM_BYTECODE_SECTION_RESOURCES);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidUtf8StringPayload) {
  loom_module_t* module = CreateModule("utf8");
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_STRINGS);
  ReadUVarint(bytes, &offset);
  uint64_t first_length = ReadUVarint(bytes, &offset);
  ASSERT_EQ(first_length, 0u);
  uint64_t second_length = ReadUVarint(bytes, &offset);
  ASSERT_GT(second_length, 0u);
  bytes[offset] = 0xFF;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidOpNameStringReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_OPS);
  uint64_t op_count = ReadUVarint(bytes, &offset);
  ASSERT_GT(op_count, 0u);
  bytes[offset] = 0x7F;

  ExpectReadError(bytes, "ERR_BYTECODE_010");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownOpName) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  ReplaceBytes(&bytes, "test.addi", "bogus.add");

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownEncodingFamily) {
  loom_context_t permissive_context;
  loom_context_initialize(iree_allocator_system(), &permissive_context);
  IREE_ASSERT_OK(loom_context_finalize(&permissive_context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&permissive_context, IREE_SV("encoding"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  loom_string_id_t encoding_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("mystery"), &encoding_name));
  loom_encoding_t encoding = {
      /*.name_id=*/encoding_name,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
  auto bytes = WriteModule(module);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
  loom_context_deinitialize(&permissive_context);
}

TEST_F(ReaderTest, RejectsNoLocationsHeaderWithLocationsSection) {
  loom_module_t* module = CreateModule("locations");
  auto bytes = WriteModule(module);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSourceLocationsHeaderWithoutLocationsSection) {
  loom_module_t* module = CreateModule("locations");
  loom_bytecode_write_options_t options = {{0}};
  options.location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;
  auto bytes = WriteModule(module, &options);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidLocationTableReference) {
  loom_module_t* module = CreateLocatedModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_LOCATIONS);
  uint64_t location_count = ReadUVarint(bytes, &offset);
  ASSERT_GE(location_count, 2u);
  offset += 2;  // Entry 0: kind=NONE, flags=0.
  ASSERT_EQ(bytes[offset], LOOM_LOCATION_FILE);
  offset += 2;  // Entry 1: kind=FILE, flags=0.
  bytes[offset] = 0x7F;

  ExpectReadError(bytes, "ERR_BYTECODE_012");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidSymbolOffsetTableReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  SectionEntry symbols = FindSection(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  ASSERT_NE(symbols.length, 0u);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  uint64_t symbol_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(symbol_count, 1u);
  uint64_t import_count = ReadUVarint(bytes, &offset);
  uint64_t export_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(import_count, 0u);
  ASSERT_EQ(export_count, 1u);
  WriteU64LE(&bytes, offset, symbols.length);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
