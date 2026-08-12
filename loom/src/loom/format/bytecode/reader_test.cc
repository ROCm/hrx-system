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
#include "loom/ops/module/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/test/types.h"

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
    // Byte offset of the first value kind inside the nested dict value.
    size_t nested_dict_first_value_kind = 0;
    // Byte offset of the second key inside the nested dict value.
    size_t nested_dict_second_key = 0;
  };

  struct AttributeValueOffsets {
    // Byte offset of the attribute value kind byte.
    size_t kind = 0;
    // Byte offset of the attribute payload after the kind byte.
    size_t payload = 0;
  };

  struct RegisterTypeOffsets {
    // Byte offset of the register value-type presence tag.
    size_t has_value_type = 0;
    // Byte offset of the register value-type table reference.
    size_t value_type = 0;
  };

  struct ProviderImportLayout {
    // Byte offset of the declared total anchor count.
    size_t total_anchor_count = 0;
    // Byte offsets of provider string IDs in canonical record order.
    std::vector<size_t> provider_ids;
    // Byte offsets of each provider's anchor symbol ordinals.
    std::vector<std::vector<size_t>> anchors;
  };

  struct SymbolReferenceLayout {
    // Byte offset of the declared total dependency occurrence count.
    size_t total_dependency_count = 0;
    // Byte offset of the declared total contract demand count.
    size_t total_contract_demand_count = 0;
    // Byte offsets of module-root dependency target ordinals.
    std::vector<size_t> module_dependencies;
    // Byte offsets of dependency target ordinals by source symbol.
    std::vector<std::vector<size_t>> symbol_dependencies;
    // Byte offsets of contract string IDs by source symbol.
    std::vector<std::vector<size_t>> symbol_contract_demands;
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
    RegisterDialect(context, LOOM_DIALECT_MODULE, loom_module_dialect_vtables,
                    loom_module_dialect_op_semantics);
    IREE_ASSERT_OK(loom_test_dialect_register(context));
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

  loom_op_t* AddSimpleFunction(loom_module_t* module, const char* name,
                               loom_test_func_build_flags_t build_flags = 0) {
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
        &builder, build_flags, /*visibility=*/0, /*cc=*/0, callee, arg_types,
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

  loom_module_t* CreateEmptyPredicateFunctionModule() {
    loom_module_t* module = CreateModule("reader_empty_predicates");
    AddSimpleFunction(module, "f", LOOM_TEST_FUNC_BUILD_FLAG_HAS_PREDICATES);
    return module;
  }

  loom_module_t* CreateRetainedFunctionModule() {
    loom_module_t* module = CreateModule("reader_retained_func");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("retained"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_type_t arg_types[1] = {i32_type};
    loom_type_t result_types[1] = {i32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_RETAIN, /*visibility=*/0,
        LOOM_FUNC_RETAIN_RETAIN, /*cc=*/0, /*purity=*/0, /*temperature=*/0,
        /*inline_policy=*/0, loom_symbol_ref_null(), /*abi=*/0,
        loom_named_attr_slice_empty(), /*export_symbol=*/LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one retained function argument";
      return module;
    }
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, arg_ids, arg_count,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
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
        /*visibility=*/0, /*retain=*/0, import_module_id, import_symbol_id,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
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

  loom_module_t* CreateEnumArrayModule() {
    loom_module_t* module = CreateModule("reader_enum_array");
    loom_region_t* body = AddVoidFunction(module, "f");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    const uint8_t required_values[] = {
        LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_LOW,
        LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_HIGH,
        LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_LOW,
    };
    const uint8_t optional_values[] = {
        LOOM_TEST_ENUM_ARRAY_ATTRS_OPTIONAL_VALUES_MIDDLE,
        42,
        LOOM_TEST_ENUM_ARRAY_ATTRS_OPTIONAL_VALUES_MIDDLE,
    };
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_enum_array_attrs_build(
        &body_builder,
        LOOM_TEST_ENUM_ARRAY_ATTRS_BUILD_FLAG_HAS_OPTIONAL_VALUES,
        loom_make_enum_array(required_values, IREE_ARRAYSIZE(required_values)),
        loom_make_enum_array(optional_values, IREE_ARRAYSIZE(optional_values)),
        loom_named_attr_slice_empty(), LOOM_LOCATION_UNKNOWN, &op));
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateSignedEnumSetModule() {
    loom_module_t* module = CreateModule("reader_signed_enum_set");
    loom_region_t* body = AddVoidFunction(module, "f");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);

    uint64_t required_words[8] = {0};
    required_words[0] =
        UINT64_C(1) << LOOM_TEST_SIGNED_ENUM_SET_ATTRS_REQUIRED_FEATURES_LOW;
    required_words[3] = UINT64_C(1) << 63;
    required_words[4] =
        UINT64_C(1) << LOOM_TEST_SIGNED_ENUM_SET_ATTRS_REQUIRED_FEATURES_MIDDLE;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_signed_enum_set_attrs_build(
        &body_builder,
        LOOM_TEST_SIGNED_ENUM_SET_ATTRS_BUILD_FLAG_HAS_OPTIONAL_FEATURES,
        loom_make_signed_enum_set(required_words, 4),
        loom_signed_enum_set_empty(), loom_named_attr_slice_empty(),
        LOOM_LOCATION_UNKNOWN, &op));
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_symbol_ref_t AddRecord(loom_module_t* module, loom_builder_t* builder,
                              iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(builder, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {/*.module_id=*/0,
                                /*.symbol_id=*/symbol_id};
    loom_op_t* record_op = nullptr;
    IREE_CHECK_OK(loom_test_record_build(builder, /*build_flags=*/0, /*kind=*/0,
                                         symbol, loom_named_attr_slice_empty(),
                                         LOOM_LOCATION_UNKNOWN, &record_op));
    return symbol;
  }

  loom_module_t* CreateSymbolArrayModule() {
    loom_module_t* module = CreateModule("reader_symbol_array");
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    const loom_symbol_ref_t a = AddRecord(module, &builder, IREE_SV("a"));
    const loom_symbol_ref_t b = AddRecord(module, &builder, IREE_SV("b"));

    loom_region_t* body = AddVoidFunction(module, "symbol_arrays");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    const loom_symbol_ref_t dependencies[] = {b, a, b};
    const loom_symbol_ref_t available[] = {a};
    loom_op_t* mixed_op = nullptr;
    IREE_CHECK_OK(loom_test_symbol_array_attrs_build(
        &body_builder, LOOM_TEST_SYMBOL_ARRAY_ATTRS_BUILD_FLAG_HAS_AVAILABLE,
        loom_make_symbol_ref_array(dependencies, IREE_ARRAYSIZE(dependencies)),
        loom_make_symbol_ref_array(available, IREE_ARRAYSIZE(available)),
        LOOM_LOCATION_UNKNOWN, &mixed_op));
    loom_op_t* present_empty_op = nullptr;
    IREE_CHECK_OK(loom_test_symbol_array_attrs_build(
        &body_builder, LOOM_TEST_SYMBOL_ARRAY_ATTRS_BUILD_FLAG_HAS_AVAILABLE,
        loom_symbol_ref_array_empty(), loom_symbol_ref_array_empty(),
        LOOM_LOCATION_UNKNOWN, &present_empty_op));
    loom_op_t* absent_op = nullptr;
    IREE_CHECK_OK(loom_test_symbol_array_attrs_build(
        &body_builder, /*build_flags=*/0, loom_symbol_ref_array_empty(),
        loom_symbol_ref_array_empty(), LOOM_LOCATION_UNKNOWN, &absent_op));
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateContractDemandModule() {
    loom_module_t* module = CreateModule("reader_contract_demands");
    loom_region_t* body = AddVoidFunction(module, "entry");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_string_id_t contract_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("test.contract"),
                                            &contract_id));
    for (int i = 0; i < 2; ++i) {
      loom_op_t* apply_op = nullptr;
      IREE_CHECK_OK(loom_func_apply_build(
          &body_builder, /*build_flags=*/0, contract_id,
          /*operands=*/nullptr, /*operands_count=*/0, /*purity=*/0,
          /*temperature=*/0, /*result_types=*/nullptr, /*result_count=*/0,
          /*tied_results=*/nullptr, /*tied_result_count=*/0,
          LOOM_LOCATION_UNKNOWN, &apply_op));
    }
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateSymbolSetModule() {
    loom_module_t* module = CreateModule("reader_symbol_set");
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    const loom_symbol_ref_t zeta = AddRecord(module, &builder, IREE_SV("zeta"));
    const loom_symbol_ref_t alpha =
        AddRecord(module, &builder, IREE_SV("alpha"));

    loom_region_t* body = AddVoidFunction(module, "symbol_sets");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    const loom_symbol_ref_t unsorted_symbols[] = {zeta, alpha};
    loom_op_t* populated_op = nullptr;
    IREE_CHECK_OK(loom_test_symbol_set_attrs_build(
        &body_builder,
        loom_make_symbol_ref_array(unsorted_symbols,
                                   IREE_ARRAYSIZE(unsorted_symbols)),
        LOOM_LOCATION_UNKNOWN, &populated_op));
    loom_op_t* empty_op = nullptr;
    IREE_CHECK_OK(loom_test_symbol_set_attrs_build(
        &body_builder, loom_symbol_ref_array_empty(), LOOM_LOCATION_UNKNOWN,
        &empty_op));
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateParameterizedAttrModule() {
    loom_module_t* module = CreateModule("reader_parameterized_attr");
    loom_region_t* body = AddVoidFunction(module, "parameterized");
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);

    loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));
    loom_attribute_t tile = loom_attr_absent();
    IREE_CHECK_OK(loom_test_tile_attr_make(module, 16, &tile));
    const uint8_t scopes[] = {
        LOOM_TEST_OPTIONS_SCOPES_SUBGROUP,
        254,
    };
    loom_attribute_t full_options = loom_attr_absent();
    IREE_CHECK_OK(loom_test_options_attr_make(
        module,
        LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES |
            LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
            LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILE,
        LOOM_TEST_OPTIONS_MODE_FAST,
        loom_make_enum_array(scopes, IREE_ARRAYSIZE(scopes)), bf16_type_id,
        tile, loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
        &full_options));
    loom_op_t* full_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_build(
        &body_builder, full_options, LOOM_LOCATION_UNKNOWN, &full_op));

    loom_attribute_t empty_options = loom_attr_absent();
    IREE_CHECK_OK(loom_test_options_attr_make(
        module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES,
        LOOM_TEST_OPTIONS_MODE_PRECISE, loom_enum_array_empty(),
        LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(),
        loom_parameterized_attr_array_empty(), &empty_options));
    loom_op_t* empty_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_build(
        &body_builder, empty_options, LOOM_LOCATION_UNKNOWN, &empty_op));

    loom_attribute_t absent_options = loom_attr_absent();
    IREE_CHECK_OK(loom_test_options_attr_make(
        module, /*build_flags=*/0, LOOM_TEST_OPTIONS_MODE_FAST,
        loom_enum_array_empty(), LOOM_TYPE_ID_INVALID, loom_attr_absent(),
        loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
        &absent_options));
    loom_op_t* absent_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_build(
        &body_builder, absent_options, LOOM_LOCATION_UNKNOWN, &absent_op));

    loom_string_id_t label_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module, IREE_SV("wave"), &label_id));
    loom_attribute_t compact = loom_attr_absent();
    IREE_CHECK_OK(loom_test_compact_attr_make(
        module, LOOM_TEST_COMPACT_ATTR_BUILD_FLAG_HAS_LABEL, label_id, 64,
        &compact));
    loom_op_t* compact_op = nullptr;
    IREE_CHECK_OK(loom_test_compact_parameterized_attr_build(
        &body_builder, compact, LOOM_LOCATION_UNKNOWN, &compact_op));

    loom_attribute_t array_values[] = {tile, full_options, tile};
    loom_attribute_t exact_tiles[] = {tile};
    loom_op_t* array_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_array_build(
        &body_builder, LOOM_TEST_PARAMETERIZED_ATTR_ARRAY_BUILD_FLAG_HAS_TILES,
        loom_make_parameterized_attr_array(array_values,
                                           IREE_ARRAYSIZE(array_values)),
        loom_make_parameterized_attr_array(exact_tiles,
                                           IREE_ARRAYSIZE(exact_tiles)),
        LOOM_LOCATION_UNKNOWN, &array_op));
    loom_op_t* present_empty_array_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_array_build(
        &body_builder, LOOM_TEST_PARAMETERIZED_ATTR_ARRAY_BUILD_FLAG_HAS_TILES,
        loom_parameterized_attr_array_empty(),
        loom_parameterized_attr_array_empty(), LOOM_LOCATION_UNKNOWN,
        &present_empty_array_op));
    loom_op_t* absent_array_op = nullptr;
    IREE_CHECK_OK(loom_test_parameterized_attr_array_build(
        &body_builder, /*build_flags=*/0, loom_parameterized_attr_array_empty(),
        loom_parameterized_attr_array_empty(), LOOM_LOCATION_UNKNOWN,
        &absent_array_op));

    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                        /*value_count=*/0,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_region_t* AddVoidFunction(loom_module_t* module, const char* name) {
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder, iree_make_cstring_view(name), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee,
        /*arg_types=*/nullptr, /*arg_type_count=*/0,
        /*result_types=*/nullptr, /*result_type_count=*/0,
        /*result_dims=*/nullptr, /*result_dim_count=*/0,
        /*result_encodings=*/nullptr, /*result_encoding_count=*/0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;
    return loom_func_like_body(loom_func_like_cast(module, func_op));
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
        &builder, 0, symbol, index_type, /*predicates=*/nullptr,
        /*predicates_count=*/0, loom_attr_i64(42), LOOM_LOCATION_UNKNOWN,
        &global_op));
    return module;
  }

  loom_module_t* CreateRegisterDeclModule(
      uint64_t carrier_payload1 = (uint64_t)4 << 16) {
    loom_module_t* module = CreateModule("reader_register_decl");
    loom_type_t vector_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
    loom_string_id_t value_type_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("test.payload"),
                                            &value_type_name_id));
    loom_type_t value_type =
        loom_type_dialect(value_type_name_id, 1, &vector_type);
    loom_type_t reg_type = loom_type_none();
    IREE_CHECK_OK(loom_module_intern_register_type(
        module, /*carrier_payload0=*/1, carrier_payload1, value_type,
        &reg_type));

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

  loom_module_t* CreateParameterizedTypeDeclModule() {
    loom_module_t* module = CreateModule("reader_parameterized_type_decl");
    loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));

    loom_type_t argument_types[6] = {};
    IREE_CHECK_OK(loom_test_scope_type_make(
        module, LOOM_TEST_SCOPE_TYPE_SCOPE_SUBGROUP, &argument_types[0]));
    IREE_CHECK_OK(
        loom_test_matrix_type_make(module, /*build_flags=*/0, bf16_type_id,
                                   LOOM_TEST_MATRIX_TYPE_SCOPE_WORKGROUP, 16,
                                   loom_symbol_ref_null(), &argument_types[1]));
    IREE_CHECK_OK(loom_test_array_type_make(
        module, /*build_flags=*/0, bf16_type_id, /*alignment=*/0,
        loom_named_attr_slice_empty(), &argument_types[2]));
    IREE_CHECK_OK(loom_test_array_type_make(
        module, LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_ALIGNMENT, bf16_type_id,
        /*alignment=*/32, loom_named_attr_slice_empty(), &argument_types[3]));
    loom_attribute_t tile = loom_attr_absent();
    IREE_CHECK_OK(loom_test_tile_attr_make(module, 8, &tile));
    loom_attribute_t options = loom_attr_absent();
    IREE_CHECK_OK(loom_test_options_attr_make(
        module, /*build_flags=*/0, LOOM_TEST_OPTIONS_MODE_FAST,
        loom_enum_array_empty(), LOOM_TYPE_ID_INVALID, loom_attr_absent(),
        loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
        &options));
    loom_attribute_t variants[] = {tile, options, tile};
    IREE_CHECK_OK(loom_test_variant_set_type_make(
        module, LOOM_TEST_VARIANT_SET_TYPE_BUILD_FLAG_HAS_ALTERNATIVES,
        loom_make_parameterized_attr_array(variants, IREE_ARRAYSIZE(variants)),
        loom_parameterized_attr_array_empty(), &argument_types[4]));
    IREE_CHECK_OK(loom_test_variant_set_type_make(
        module, /*build_flags=*/0, loom_parameterized_attr_array_empty(),
        loom_parameterized_attr_array_empty(), &argument_types[5]));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder, IREE_SV("parameterized_types"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* decl_op = nullptr;
    IREE_CHECK_OK(loom_test_decl_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, argument_types,
        IREE_ARRAYSIZE(argument_types), /*result_types=*/nullptr,
        /*result_count=*/0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &decl_op));
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
        &builder, LOOM_GLOBAL_CONSTANT_BUILD_FLAG_HAS_PREDICATES, symbol,
        tile_type, predicates, 1, loom_attr_absent(), LOOM_LOCATION_UNKNOWN,
        &global_op));
    return module;
  }

  loom_module_t* CreateTwoFunctionModule() {
    loom_module_t* module = CreateModule("reader_two_funcs");
    AddSimpleFunction(module, "f0");
    AddSimpleFunction(module, "f1");
    return module;
  }

  loom_module_t* CreateProviderImportModule() {
    loom_module_t* module = CreateModule("reader_provider_import");
    AddSimpleFunction(module, "resolved");
    const uint16_t resolved_symbol_id = (uint16_t)(module->symbols.count - 1);

    loom_string_id_t missing_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("missing"),
                                            &missing_name_id));
    uint16_t missing_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_add_symbol(module, missing_name_id, &missing_symbol_id));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module, IREE_SV("motif/provider.loom"), &provider_id));
    loom_symbol_ref_t anchors[] = {
        {/*.module_id=*/0, /*.symbol_id=*/resolved_symbol_id},
        {/*.module_id=*/0, /*.symbol_id=*/missing_symbol_id},
    };
    loom_op_t* import_op = nullptr;
    IREE_CHECK_OK(loom_module_import_build(
        &builder, provider_id,
        loom_make_symbol_ref_array(anchors, IREE_ARRAYSIZE(anchors)),
        LOOM_LOCATION_NONE, &import_op));
    import_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
    const iree_string_view_t comments[] = {IREE_SV(" provider comment")};
    IREE_CHECK_OK(loom_module_attach_op_comments(module, import_op, comments,
                                                 IREE_ARRAYSIZE(comments)));
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
        &body_builder, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT, arg_ids[0],
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
    loom_bytecode_index_options_t options = {
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
    loom_bytecode_index_options_t options = {
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

  size_t FirstSymbolFlagsOffset(const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
    EXPECT_GT(ReadUVarint(bytes, &offset), 0u);  // symbol_count
    uint64_t import_count = ReadUVarint(bytes, &offset);
    uint64_t export_count = ReadUVarint(bytes, &offset);
    offset += (import_count + export_count) * sizeof(uint64_t);
    ReadUVarint(bytes, &offset);  // name_id
    offset += 1;                  // kind
    offset += 1;                  // visibility
    return offset;
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

  RegisterTypeOffsets ReadRegisterDeclTypeOffsets(
      const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_TYPES);
    EXPECT_EQ(ReadUVarint(bytes, &offset), 4u);

    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_TYPE_SCALAR);
    offset += 1;  // Scalar element type.

    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_TYPE_VECTOR);
    offset += 1;                     // Vector element type.
    EXPECT_EQ(bytes[offset++], 1u);  // Rank.
    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
    EXPECT_EQ(ReadUVarint(bytes, &offset), 0u);  // Encoding instance.
    EXPECT_EQ(bytes[offset++], 0u);              // Static dimension.
    EXPECT_EQ(ReadUVarint(bytes, &offset), 4u);  // Dimension size.

    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_TYPE_DIALECT);
    (void)ReadUVarint(bytes, &offset);           // Type-name string ID.
    EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);  // Parameter count.
    EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);  // Vector type reference.

    EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_TYPE_REGISTER);
    EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);                 // Payload 0.
    EXPECT_EQ(ReadUVarint(bytes, &offset), (uint64_t)4 << 16);  // Payload 1.
    RegisterTypeOffsets result;
    result.has_value_type = offset++;
    result.value_type = offset;
    EXPECT_EQ(bytes[result.has_value_type], 1u);
    EXPECT_EQ(ReadUVarint(bytes, &offset), 2u);
    return result;
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

  void SkipSourceTrivia(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t source_trivia = ReadUVarint(bytes, offset);
    uint64_t comment_count =
        source_trivia >> LOOM_BYTECODE_SOURCE_TRIVIA_COMMENT_COUNT_SHIFT;
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
      case 6:   // SYMBOL.
      case 7:   // TYPE.
      case 10:  // ENCODING.
      case 12:  // SCOPED_ENUM.
        ReadUVarint(bytes, offset);
        break;
      case 3:  // BOOL.
      case 4:  // ENUM.
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
      case 11:    // BYTES.
      case 13: {  // ENUM_ARRAY.
        uint64_t count = ReadUVarint(bytes, offset);
        *offset += count;
        break;
      }
      case 16: {  // SIGNED_ENUM_SET.
        uint8_t word_count = bytes[(*offset)++];
        *offset += (size_t)word_count * 2 * sizeof(uint64_t);
        break;
      }
      case 17:    // SYMBOL_ARRAY.
      case 18: {  // SYMBOL_SET.
        uint64_t count = ReadUVarint(bytes, offset);
        for (uint64_t i = 0; i < count; ++i) {
          ReadUVarint(bytes, offset);
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
      SkipSourceTrivia(bytes, &offset);
      offset += 1;  // calling_convention
      offset += 1;  // purity
      uint64_t workload_arg_count = ReadUVarint(bytes, &offset);
      uint64_t arg_count = ReadUVarint(bytes, &offset);
      uint64_t result_count = ReadUVarint(bytes, &offset);
      for (uint64_t arg = 0; arg < workload_arg_count; ++arg) {
        ReadValueDefOffsets(bytes, &offset);
      }
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

  ProviderImportLayout ReadProviderImportLayout(
      const std::vector<uint8_t>& bytes) {
    SectionEntry section =
        FindSection(bytes, LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS);
    size_t offset = (size_t)ModuleOffset(bytes) + (size_t)section.offset;
    const size_t end_offset = offset + (size_t)section.length;
    const uint64_t provider_count = ReadUVarint(bytes, &offset);
    ProviderImportLayout layout;
    layout.total_anchor_count = offset;
    ReadUVarint(bytes, &offset);
    layout.provider_ids.reserve((size_t)provider_count);
    layout.anchors.reserve((size_t)provider_count);
    for (uint64_t i = 0; i < provider_count; ++i) {
      layout.provider_ids.push_back(offset);
      ReadUVarint(bytes, &offset);
      const uint64_t anchor_count = ReadUVarint(bytes, &offset);
      std::vector<size_t>& anchor_offsets = layout.anchors.emplace_back();
      anchor_offsets.reserve((size_t)anchor_count);
      for (uint64_t j = 0; j < anchor_count; ++j) {
        anchor_offsets.push_back(offset);
        ReadUVarint(bytes, &offset);
      }
      SkipSourceTrivia(bytes, &offset);
    }
    EXPECT_EQ(offset, end_offset);
    return layout;
  }

  SymbolReferenceLayout ReadSymbolReferenceLayout(
      const std::vector<uint8_t>& bytes) {
    SectionEntry section =
        FindSection(bytes, LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES);
    size_t offset = (size_t)ModuleOffset(bytes) + (size_t)section.offset;
    const size_t end_offset = offset + (size_t)section.length;
    const uint64_t symbol_count = ReadUVarint(bytes, &offset);
    SymbolReferenceLayout layout;
    layout.total_dependency_count = offset;
    ReadUVarint(bytes, &offset);
    layout.total_contract_demand_count = offset;
    ReadUVarint(bytes, &offset);

    const uint64_t module_dependency_count = ReadUVarint(bytes, &offset);
    layout.module_dependencies.reserve((size_t)module_dependency_count);
    for (uint64_t i = 0; i < module_dependency_count; ++i) {
      layout.module_dependencies.push_back(offset);
      ReadUVarint(bytes, &offset);
    }

    layout.symbol_dependencies.reserve((size_t)symbol_count);
    layout.symbol_contract_demands.reserve((size_t)symbol_count);
    for (uint64_t i = 0; i < symbol_count; ++i) {
      const uint64_t dependency_count = ReadUVarint(bytes, &offset);
      std::vector<size_t>& dependency_offsets =
          layout.symbol_dependencies.emplace_back();
      dependency_offsets.reserve((size_t)dependency_count);
      for (uint64_t j = 0; j < dependency_count; ++j) {
        dependency_offsets.push_back(offset);
        ReadUVarint(bytes, &offset);
      }

      const uint64_t contract_demand_count = ReadUVarint(bytes, &offset);
      std::vector<size_t>& contract_offsets =
          layout.symbol_contract_demands.emplace_back();
      contract_offsets.reserve((size_t)contract_demand_count);
      for (uint64_t j = 0; j < contract_demand_count; ++j) {
        contract_offsets.push_back(offset);
        ReadUVarint(bytes, &offset);
      }
    }
    EXPECT_EQ(offset, end_offset);
    return layout;
  }

  size_t RootRegionSourceFlagsOffset(const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_IR);
    ReadUVarint(bytes, &offset);  // value_count
    ReadUVarint(bytes, &offset);  // region_count
    ReadUVarint(bytes, &offset);  // block_count
    ReadUVarint(bytes, &offset);  // op_count
    uint64_t root_region_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_region_count, 1u);
    uint64_t root_region_index = ReadUVarint(bytes, &offset);
    EXPECT_EQ(root_region_index, 0u);
    return offset;
  }

  size_t RootBlockSourceTriviaOffset(const std::vector<uint8_t>& bytes) {
    size_t offset = RootRegionSourceFlagsOffset(bytes);
    ReadUVarint(bytes, &offset);                 // region source_flags
    EXPECT_GE(ReadUVarint(bytes, &offset), 1u);  // block_count
    uint8_t has_label = bytes[offset++];
    if (has_label) {
      ReadUVarint(bytes, &offset);
    }
    return offset;
  }

  size_t RootBlockValueListOffset(const std::vector<uint8_t>& bytes,
                                  uint64_t* out_arg_count) {
    size_t offset = RootRegionSourceFlagsOffset(bytes);
    uint64_t root_source_flags = ReadUVarint(bytes, &offset);
    EXPECT_EQ(root_source_flags, 0u);
    uint64_t root_block_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_block_count, 1u);
    uint8_t has_label = bytes[offset++];
    if (has_label) {
      ReadUVarint(bytes, &offset);
    }
    SkipSourceTrivia(bytes, &offset);
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
    SkipSourceTrivia(bytes, &offset);
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
    SkipSourceTrivia(bytes, &offset);
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
    SkipSourceTrivia(bytes, &offset);
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

  size_t FirstBodyOpAttributeListOffset(const std::vector<uint8_t>& bytes) {
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
    SkipSourceTrivia(bytes, &offset);
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
    return offset;
  }

  AttributeValueOffsets FirstBodyOpFirstAttributeValueOffsets(
      const std::vector<uint8_t>& bytes) {
    size_t offset = FirstBodyOpAttributeListOffset(bytes);
    uint64_t attr_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(attr_count, 1u);
    ReadUVarint(bytes, &offset);  // key_id
    AttributeValueOffsets value_offsets = {
        /*.kind=*/offset,
        /*.payload=*/offset + 1,
    };
    return value_offsets;
  }

  BodyOpAttrOffsets FirstBodyOpAttrOffsets(const std::vector<uint8_t>& bytes) {
    AttributeValueOffsets value_offsets =
        FirstBodyOpFirstAttributeValueOffsets(bytes);
    BodyOpAttrOffsets attr_offsets;
    attr_offsets.attr_kind = value_offsets.kind;
    size_t offset = value_offsets.kind;
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
    attr_offsets.nested_dict_first_value_kind = offset;
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
    SkipSourceTrivia(bytes, &offset);
    payload_offsets.result_count = offset;
    ReadUVarint(bytes, &offset);
    payload_offsets.local_value_count = offset;
    return payload_offsets;
  }

  size_t LastOpRegionCountOffsetInRegion(const std::vector<uint8_t>& bytes,
                                         size_t* offset) {
    size_t last_region_count_offset = 0;
    ReadUVarint(bytes, offset);  // source_flags
    uint64_t block_count = ReadUVarint(bytes, offset);
    for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
      uint8_t has_label = bytes[(*offset)++];
      if (has_label) {
        ReadUVarint(bytes, offset);
      }
      SkipSourceTrivia(bytes, offset);
      uint64_t arg_count = ReadUVarint(bytes, offset);
      for (uint64_t arg_index = 0; arg_index < arg_count; ++arg_index) {
        ReadValueDefOffsets(bytes, offset);
      }
      uint64_t op_count = ReadUVarint(bytes, offset);
      for (uint64_t op_index = 0; op_index < op_count; ++op_index) {
        ReadUVarint(bytes, offset);  // op_table_index_plus1
        *offset += 1;                // flags
        ReadUVarint(bytes, offset);  // location_id
        SkipSourceTrivia(bytes, offset);
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
    EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
    EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
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

TEST_F(ReaderTest, PreservesRegionSourceFlags) {
  loom_module_t* module = CreateFunctionModule();
  loom_op_t* func_op = module->symbols.entries[0].defining_op;
  loom_region_t* body =
      loom_func_like_body(loom_func_like_cast(module, func_op));
  ASSERT_NE(body, nullptr);
  body->source_flags = LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM;
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* read_func_op = read_module->symbols.entries[0].defining_op;
  loom_region_t* read_body =
      loom_func_like_body(loom_func_like_cast(read_module, read_func_op));
  ASSERT_NE(read_body, nullptr);
  EXPECT_EQ(read_body->source_flags, LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, PreservesSourceTrivia) {
  loom_module_t* module = CreateFunctionModule();
  loom_op_t* func_op = module->symbols.entries[0].defining_op;
  loom_region_t* body =
      loom_func_like_body(loom_func_like_cast(module, func_op));
  ASSERT_NE(body, nullptr);
  loom_block_t* entry_block = loom_region_entry_block(body);
  ASSERT_NE(entry_block->first_op, nullptr);

  func_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  entry_block->flags |= LOOM_BLOCK_FLAG_LEADING_BLANK_LINE;
  entry_block->first_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  const iree_string_view_t symbol_comments[] = {IREE_SV("symbol")};
  const iree_string_view_t block_comments[] = {IREE_SV("block")};
  const iree_string_view_t op_comments[] = {IREE_SV("operation")};
  const iree_string_view_t file_header[] = {
      IREE_SV("File-level overview."),
      IREE_SV("Second header line."),
  };
  IREE_ASSERT_OK(loom_module_attach_file_header(module, file_header,
                                                IREE_ARRAYSIZE(file_header)));
  IREE_ASSERT_OK(loom_module_attach_op_comments(
      module, func_op, symbol_comments, IREE_ARRAYSIZE(symbol_comments)));
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      module, entry_block, block_comments, IREE_ARRAYSIZE(block_comments)));
  IREE_ASSERT_OK(loom_module_attach_op_comments(
      module, entry_block->first_op, op_comments, IREE_ARRAYSIZE(op_comments)));

  auto bytes = WriteModule(module);
  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);

  loom_op_t* read_func_op = read_module->symbols.entries[0].defining_op;
  loom_region_t* read_body =
      loom_func_like_body(loom_func_like_cast(read_module, read_func_op));
  ASSERT_NE(read_body, nullptr);
  loom_block_t* read_entry_block = loom_region_entry_block(read_body);
  ASSERT_NE(read_entry_block->first_op, nullptr);
  EXPECT_TRUE(
      iree_any_bit_set(read_func_op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE));
  EXPECT_TRUE(iree_any_bit_set(read_entry_block->flags,
                               LOOM_BLOCK_FLAG_LEADING_BLANK_LINE));
  EXPECT_TRUE(iree_any_bit_set(read_entry_block->first_op->flags,
                               LOOM_OP_FLAG_LEADING_BLANK_LINE));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_file_header(read_module, &comment_count);
  ASSERT_EQ(comment_count, IREE_ARRAYSIZE(file_header));
  EXPECT_TRUE(iree_string_view_equal(comments[0], file_header[0]));
  EXPECT_TRUE(iree_string_view_equal(comments[1], file_header[1]));
  comments = loom_module_op_comments(read_module, read_func_op, &comment_count);
  ASSERT_EQ(comment_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(comments[0], symbol_comments[0]));
  comments =
      loom_module_block_comments(read_module, read_entry_block, &comment_count);
  ASSERT_EQ(comment_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(comments[0], block_comments[0]));
  comments = loom_module_op_comments(read_module, read_entry_block->first_op,
                                     &comment_count);
  ASSERT_EQ(comment_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(comments[0], op_comments[0]));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsFileHeaderLeadingBlankLine) {
  loom_module_t* module = CreateModule("file_header");
  const iree_string_view_t file_header[] = {IREE_SV("File overview.")};
  IREE_ASSERT_OK(loom_module_attach_file_header(module, file_header,
                                                IREE_ARRAYSIZE(file_header)));
  auto bytes = WriteModule(module);
  size_t offset =
      SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SOURCE_TRIVIA);
  ASSERT_EQ(bytes[offset], 2u);
  bytes[offset] |= LOOM_BYTECODE_SOURCE_TRIVIA_LEADING_BLANK_LINE;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidUtf8FileHeader) {
  loom_module_t* module = CreateModule("file_header");
  const iree_string_view_t file_header[] = {IREE_SV("File overview.")};
  IREE_ASSERT_OK(loom_module_attach_file_header(module, file_header,
                                                IREE_ARRAYSIZE(file_header)));
  auto bytes = WriteModule(module);
  size_t offset =
      SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SOURCE_TRIVIA);
  ASSERT_EQ(ReadUVarint(bytes, &offset), 2u);
  ASSERT_GT(ReadUVarint(bytes, &offset), 1u);
  ASSERT_EQ(bytes[offset], ' ');
  bytes[offset + 1] = 0xFF;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsFileHeaderTrailingBytes) {
  loom_module_t* module = CreateModule("file_header");
  const iree_string_view_t file_header[] = {IREE_SV("File overview.")};
  IREE_ASSERT_OK(loom_module_attach_file_header(module, file_header,
                                                IREE_ARRAYSIZE(file_header)));
  auto bytes = WriteModule(module);
  SectionEntry source_trivia =
      FindSection(bytes, LOOM_BYTECODE_SECTION_SOURCE_TRIVIA);
  ASSERT_NE(source_trivia.length, 0u);
  ASSERT_EQ((uint64_t)bytes.size(),
            ModuleOffset(bytes) + source_trivia.offset + source_trivia.length);
  bytes.push_back(0);
  WriteU64LE(&bytes, source_trivia.directory_entry_offset + 16,
             source_trivia.length + 1);
  WriteU64LE(&bytes, ModuleDirectoryOffset(bytes) + 16,
             ModuleLength(bytes) + 1);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

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
  EXPECT_EQ(module_metadata.strings.count,
            module_metadata.summary.string_count);
  ASSERT_NE(module_metadata.strings.values, nullptr);
  EXPECT_EQ(module_metadata.sources.count,
            module_metadata.summary.source_count);
  EXPECT_EQ(module_metadata.types.count, module_metadata.summary.type_count);
  ASSERT_NE(module_metadata.types.entries, nullptr);
  EXPECT_EQ(module_metadata.encodings.count,
            module_metadata.summary.encoding_count);
  EXPECT_EQ(module_metadata.ops.count, module_metadata.summary.op_name_count);
  ASSERT_NE(module_metadata.ops.entries, nullptr);
  EXPECT_EQ(module_metadata.locations.count,
            module_metadata.summary.location_count);
  if (module_metadata.locations.count > 0) {
    ASSERT_NE(module_metadata.locations.entries, nullptr);
  }
  EXPECT_EQ(module_metadata.symbol_count, 1u);
  EXPECT_EQ(module_metadata.import_count, 0u);
  EXPECT_EQ(module_metadata.export_count, 1u);
  ASSERT_NE(module_metadata.export_symbol_indices, nullptr);
  EXPECT_EQ(module_metadata.export_symbol_indices[0], 0u);

  ASSERT_NE(module_metadata.symbols, nullptr);
  const loom_bytecode_symbol_metadata_t& symbol = module_metadata.symbols[0];
  EXPECT_TRUE(iree_string_view_equal(symbol.name, IREE_SV("f")));
  ASSERT_LT(symbol.name_string_index, module_metadata.strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      module_metadata.strings.values[symbol.name_string_index], symbol.name));
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

  const auto* type_section =
      std::find_if(module_metadata.sections,
                   module_metadata.sections + module_metadata.section_count,
                   [](const loom_bytecode_section_metadata_t& section) {
                     return section.kind == LOOM_BYTECODE_SECTION_TYPES;
                   });
  ASSERT_NE(type_section,
            module_metadata.sections + module_metadata.section_count);
  for (iree_host_size_t i = 0; i < module_metadata.types.count; ++i) {
    const auto& entry = module_metadata.types.entries[i];
    EXPECT_GE(entry.entry_offset, type_section->absolute_offset);
    EXPECT_GT(entry.entry_length, 0u);
    EXPECT_LE(entry.entry_offset + entry.entry_length,
              type_section->absolute_offset + type_section->length);
  }
  EXPECT_FALSE(iree_string_view_is_empty(module_metadata.ops.entries[0].name));

  if (module_metadata.locations.count > 0) {
    const auto* location_section =
        std::find_if(module_metadata.sections,
                     module_metadata.sections + module_metadata.section_count,
                     [](const loom_bytecode_section_metadata_t& section) {
                       return section.kind == LOOM_BYTECODE_SECTION_LOCATIONS;
                     });
    ASSERT_NE(location_section,
              module_metadata.sections + module_metadata.section_count);
    for (iree_host_size_t i = 0; i < module_metadata.locations.count; ++i) {
      const auto& entry = module_metadata.locations.entries[i];
      EXPECT_GE(entry.entry_offset, location_section->absolute_offset);
      EXPECT_GT(entry.entry_length, 0u);
      EXPECT_LE(entry.entry_offset + entry.entry_length,
                location_section->absolute_offset + location_section->length);
    }
  }

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, IndexesRawDependencyOccurrencesBySourceSymbol) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  ASSERT_EQ(result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  ASSERT_EQ(module_metadata.symbol_count, 3u);
  EXPECT_EQ(module_metadata.summary.dependency_count, 3u);
  EXPECT_EQ(module_metadata.summary.contract_demand_count, 0u);
  EXPECT_EQ(module_metadata.module_dependency_count, 0u);
  ASSERT_EQ(module_metadata.dependency_count, 3u);
  ASSERT_NE(module_metadata.dependency_symbol_indices, nullptr);
  ASSERT_NE(module_metadata.symbol_references, nullptr);
  EXPECT_EQ(module_metadata.contract_demand_count, 0u);
  EXPECT_EQ(module_metadata.contract_demands, nullptr);

  EXPECT_EQ(module_metadata.symbol_references[0].dependency_count, 0u);
  EXPECT_EQ(module_metadata.symbol_references[1].dependency_count, 0u);
  EXPECT_EQ(module_metadata.symbol_references[2].first_dependency_index, 0u);
  EXPECT_EQ(module_metadata.symbol_references[2].dependency_count, 3u);
  EXPECT_EQ(module_metadata.dependency_symbol_indices[0], 1u);
  EXPECT_EQ(module_metadata.dependency_symbol_indices[1], 0u);
  EXPECT_EQ(module_metadata.dependency_symbol_indices[2], 1u);

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, IndexesEveryAbstractProviderDemand) {
  loom_module_t* module = CreateContractDemandModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  ASSERT_EQ(result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  ASSERT_EQ(module_metadata.symbol_count, 1u);
  EXPECT_EQ(module_metadata.summary.dependency_count, 0u);
  EXPECT_EQ(module_metadata.summary.contract_demand_count, 2u);
  EXPECT_EQ(module_metadata.dependency_count, 0u);
  EXPECT_EQ(module_metadata.dependency_symbol_indices, nullptr);
  ASSERT_EQ(module_metadata.contract_demand_count, 2u);
  ASSERT_NE(module_metadata.contract_demands, nullptr);
  ASSERT_NE(module_metadata.symbol_references, nullptr);
  EXPECT_EQ(module_metadata.symbol_references[0].first_contract_demand_index,
            0u);
  EXPECT_EQ(module_metadata.symbol_references[0].contract_demand_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(module_metadata.contract_demands[0],
                                     IREE_SV("test.contract")));
  EXPECT_TRUE(iree_string_view_equal(module_metadata.contract_demands[1],
                                     IREE_SV("test.contract")));

  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, IndexesAndMaterializesProviderImports) {
  loom_module_t* module = CreateProviderImportModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  ASSERT_EQ(result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  EXPECT_EQ(module_metadata.summary.symbol_count, 2u);
  EXPECT_EQ(module_metadata.summary.provider_import_count, 1u);
  EXPECT_EQ(module_metadata.summary.provider_import_anchor_count, 2u);
  ASSERT_EQ(module_metadata.provider_import_count, 1u);
  ASSERT_EQ(module_metadata.provider_import_anchor_count, 2u);
  ASSERT_NE(module_metadata.provider_imports, nullptr);
  ASSERT_NE(module_metadata.provider_import_anchor_symbol_indices, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(module_metadata.provider_imports[0].provider,
                             IREE_SV("motif/provider.loom")));
  EXPECT_EQ(module_metadata.provider_imports[0].first_anchor_index, 0u);
  EXPECT_EQ(module_metadata.provider_imports[0].anchor_count, 2u);
  EXPECT_TRUE(module_metadata.provider_imports[0].leading_blank_line);
  ASSERT_EQ(module_metadata.provider_imports[0].comment_count, 1u);
  ASSERT_NE(module_metadata.provider_imports[0].comments, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(module_metadata.provider_imports[0].comments[0],
                             IREE_SV(" provider comment")));
  EXPECT_EQ(module_metadata.provider_import_anchor_symbol_indices[0], 1u);
  EXPECT_EQ(module_metadata.provider_import_anchor_symbol_indices[1], 0u);
  ASSERT_NE(module_metadata.symbols, nullptr);
  EXPECT_EQ(module_metadata.symbols[0].kind, LOOM_BYTECODE_SYMBOL_FUNC_DEF);
  EXPECT_EQ(module_metadata.symbols[1].kind, LOOM_BYTECODE_SYMBOL_ANCHOR);
  iree_arena_deinitialize(&metadata_arena);

  loom_module_t* read_module = nullptr;
  error_ids.clear();
  result = ReadModule(bytes, &read_module, &error_ids,
                      /*verify_module=*/true);
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 2u);
  EXPECT_EQ(read_module->symbols.entries[1].kind, LOOM_SYMBOL_NONE);
  EXPECT_EQ(read_module->symbols.entries[1].defining_op, nullptr);

  const loom_op_t* import_op = loom_module_block(read_module)->first_op;
  ASSERT_NE(import_op, nullptr);
  ASSERT_TRUE(loom_module_import_isa(import_op));
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[loom_module_import_provider(import_op)],
      IREE_SV("motif/provider.loom")));
  loom_symbol_ref_array_t anchors = loom_module_import_symbols(import_op);
  ASSERT_EQ(anchors.count, 2u);
  EXPECT_EQ(anchors.values[0].symbol_id, 1u);
  EXPECT_EQ(anchors.values[1].symbol_id, 0u);
  EXPECT_TRUE(
      iree_all_bits_set(import_op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE));
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(read_module, import_op, &comment_count);
  ASSERT_EQ(comment_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(comments[0], IREE_SV(" provider comment")));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsProviderImportAnchorCountMismatch) {
  loom_module_t* module = CreateProviderImportModule();
  auto bytes = WriteModule(module);
  ProviderImportLayout layout = ReadProviderImportLayout(bytes);

  size_t offset = layout.total_anchor_count;
  ASSERT_EQ(ReadUVarint(bytes, &offset), 2u);
  bytes[layout.total_anchor_count] = 3;

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsOutOfRangeProviderImportAnchor) {
  loom_module_t* module = CreateProviderImportModule();
  auto bytes = WriteModule(module);
  ProviderImportLayout layout = ReadProviderImportLayout(bytes);
  ASSERT_EQ(layout.anchors.size(), 1u);
  ASSERT_EQ(layout.anchors[0].size(), 2u);

  bytes[layout.anchors[0][0]] = 127;

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsNoncanonicalProviderImportAnchorOrder) {
  loom_module_t* module = CreateProviderImportModule();
  auto bytes = WriteModule(module);
  ProviderImportLayout layout = ReadProviderImportLayout(bytes);
  ASSERT_EQ(layout.anchors.size(), 1u);
  ASSERT_EQ(layout.anchors[0].size(), 2u);
  const size_t first_offset = layout.anchors[0][0];
  const size_t second_offset = layout.anchors[0][1];
  size_t next_offset = first_offset;
  const uint64_t first_ordinal = ReadUVarint(bytes, &next_offset);
  ASSERT_EQ(next_offset, first_offset + 1);
  next_offset = second_offset;
  const uint64_t second_ordinal = ReadUVarint(bytes, &next_offset);
  ASSERT_EQ(next_offset, second_offset + 1);
  bytes[first_offset] = (uint8_t)second_ordinal;
  bytes[second_offset] = (uint8_t)first_ordinal;

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsNoncanonicalProviderImportOrder) {
  loom_module_t* module = CreateProviderImportModule();
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module, IREE_SV("zeta/provider.loom"), &provider_id));
  loom_symbol_ref_t anchor = {/*.module_id=*/0, /*.symbol_id=*/0};
  loom_op_t* import_op = nullptr;
  IREE_ASSERT_OK(loom_module_import_build(
      &builder, provider_id, loom_make_symbol_ref_array(&anchor, 1),
      LOOM_LOCATION_NONE, &import_op));
  auto bytes = WriteModule(module);
  ProviderImportLayout layout = ReadProviderImportLayout(bytes);
  ASSERT_EQ(layout.provider_ids.size(), 2u);
  size_t first_end = layout.provider_ids[0];
  const uint64_t first_provider_id = ReadUVarint(bytes, &first_end);
  ASSERT_EQ(first_end, layout.provider_ids[0] + 1);
  size_t second_end = layout.provider_ids[1];
  ReadUVarint(bytes, &second_end);
  ASSERT_EQ(second_end, layout.provider_ids[1] + 1);
  bytes[layout.provider_ids[1]] = (uint8_t)first_provider_id;

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsDuplicateSymbolIdentity) {
  loom_module_t* module = CreateFunctionModule();
  AddSimpleFunction(module, "g");
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  ASSERT_EQ(metadata.modules[0].symbol_count, 2u);
  const size_t first_entry_offset =
      (size_t)metadata.modules[0].symbols[0].entry_offset;
  const size_t second_entry_offset =
      (size_t)metadata.modules[0].symbols[1].entry_offset;
  iree_arena_deinitialize(&metadata_arena);

  size_t first_name_offset = first_entry_offset;
  size_t second_name_offset = second_entry_offset;
  const uint64_t first_name_id = ReadUVarint(bytes, &first_name_offset);
  const uint64_t second_name_id = ReadUVarint(bytes, &second_name_offset);
  ASSERT_LT(first_name_id, 128u);
  ASSERT_LT(second_name_id, 128u);
  ASSERT_NE(first_name_id, second_name_id);
  bytes[second_entry_offset] = bytes[first_entry_offset];

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsDefinitionRolesNotDeclaredByOp) {
  loom_module_t* module = CreateFunctionModule();
  for (loom_bytecode_symbol_flags_t role : {
           LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION,
           LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY,
       }) {
    SCOPED_TRACE(role);
    auto bytes = WriteModule(module);
    size_t offset = FirstSymbolFlagsOffset(bytes);
    uint16_t flags = ReadU16LE(bytes, offset);
    WriteU16LE(&bytes, offset, flags | role);

    ExpectReadError(bytes, "ERR_BYTECODE_006");
  }
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsPredicatesFlagOnNonFunctionSymbol) {
  loom_module_t* module = CreateGlobalModule();
  auto bytes = WriteModule(module);
  size_t flags_offset = FirstSymbolFlagsOffset(bytes);
  uint16_t flags = ReadU16LE(bytes, flags_offset);
  WriteU16LE(&bytes, flags_offset,
             flags | LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES);

  ExpectReadError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsNonemptyPredicatesWithoutFlag) {
  loom_module_t* module = CreatePredicateFunctionModule();
  auto bytes = WriteModule(module);
  size_t flags_offset = FirstSymbolFlagsOffset(bytes);
  uint16_t flags = ReadU16LE(bytes, flags_offset);
  ASSERT_TRUE(iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES));
  WriteU16LE(&bytes, flags_offset,
             flags & ~LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES);

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");
  loom_module_free(module);
}

TEST_F(ReaderTest, RetainedFunctionSurvivesModuleRead) {
  loom_module_t* module = CreateRetainedFunctionModule();
  auto bytes = WriteModule(module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t metadata_result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);

  EXPECT_EQ(metadata_result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  ASSERT_EQ(module_metadata.symbol_count, 1u);
  const loom_bytecode_symbol_metadata_t& metadata_symbol =
      module_metadata.symbols[0];
  EXPECT_TRUE(iree_all_bits_set(metadata_symbol.flags,
                                LOOM_BYTECODE_SYMBOL_FLAG_RETAIN));

  loom_module_t* read_module = nullptr;
  loom_bytecode_read_result_t read_result =
      ReadModule(bytes, &read_module, &error_ids);
  EXPECT_EQ(read_result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  const loom_symbol_t& symbol = read_module->symbols.entries[0];
  EXPECT_TRUE(iree_all_bits_set(symbol.flags, LOOM_SYMBOL_FLAG_RETAIN));
  ASSERT_NE(symbol.defining_op, nullptr);
  EXPECT_EQ(loom_func_def_retain(symbol.defining_op), LOOM_FUNC_RETAIN_RETAIN);

  loom_module_free(read_module);
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

TEST_F(ReaderTest, EnumArraysPreserveStableValuesAndOrder) {
  loom_module_t* module = CreateEnumArrayModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(func_op, nullptr);
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry = loom_region_entry_block(body);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 2u);
  loom_op_t* op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_test_enum_array_attrs_isa(op));
  loom_enum_array_t required = loom_test_enum_array_attrs_required_values(op);
  ASSERT_EQ(required.count, 3u);
  EXPECT_EQ(required.values[0], LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_LOW);
  EXPECT_EQ(required.values[1],
            LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_HIGH);
  EXPECT_EQ(required.values[2], LOOM_TEST_ENUM_ARRAY_ATTRS_REQUIRED_VALUES_LOW);
  loom_enum_array_t optional = loom_test_enum_array_attrs_optional_values(op);
  ASSERT_EQ(optional.count, 3u);
  EXPECT_EQ(optional.values[0],
            LOOM_TEST_ENUM_ARRAY_ATTRS_OPTIONAL_VALUES_MIDDLE);
  EXPECT_EQ(optional.values[1], 42u);
  EXPECT_EQ(optional.values[2],
            LOOM_TEST_ENUM_ARRAY_ATTRS_OPTIONAL_VALUES_MIDDLE);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, SignedEnumSetsPreserveAssertionsAndPresence) {
  loom_module_t* module = CreateSignedEnumSetModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(func_op, nullptr);
  loom_block_t* entry = loom_region_entry_block(loom_test_func_body(func_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 2u);
  loom_op_t* op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_test_signed_enum_set_attrs_isa(op));

  loom_signed_enum_set_t required =
      loom_test_signed_enum_set_attrs_required_features(op);
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(
      required, LOOM_TEST_SIGNED_ENUM_SET_ATTRS_REQUIRED_FEATURES_LOW));
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(
      required, LOOM_TEST_SIGNED_ENUM_SET_ATTRS_REQUIRED_FEATURES_HIGH));
  EXPECT_TRUE(loom_signed_enum_set_contains_negative(
      required, LOOM_TEST_SIGNED_ENUM_SET_ATTRS_REQUIRED_FEATURES_MIDDLE));
  EXPECT_EQ(required.word_count, 4u);
  EXPECT_FALSE(loom_attr_is_absent(loom_op_attrs(op)[1]));
  loom_signed_enum_set_t optional =
      loom_test_signed_enum_set_attrs_optional_features(op);
  EXPECT_EQ(optional.word_count, 0u);
  EXPECT_EQ(optional.words, nullptr);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, SymbolArraysPreserveNamesPresenceAndOrder) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 3u);
  const loom_string_id_t function_name_id =
      loom_module_lookup_string(read_module, IREE_SV("symbol_arrays"));
  const uint16_t function_symbol_id =
      loom_module_find_symbol(read_module, function_name_id);
  ASSERT_NE(function_symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* func_op =
      read_module->symbols.entries[function_symbol_id].defining_op;
  ASSERT_NE(func_op, nullptr);
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_block_t* entry = loom_region_entry_block(loom_test_func_body(func_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 4u);

  const loom_string_id_t a_name_id =
      loom_module_lookup_string(read_module, IREE_SV("a"));
  const loom_string_id_t b_name_id =
      loom_module_lookup_string(read_module, IREE_SV("b"));
  const uint16_t a_symbol_id = loom_module_find_symbol(read_module, a_name_id);
  const uint16_t b_symbol_id = loom_module_find_symbol(read_module, b_name_id);
  ASSERT_NE(a_symbol_id, LOOM_SYMBOL_ID_INVALID);
  ASSERT_NE(b_symbol_id, LOOM_SYMBOL_ID_INVALID);

  loom_op_t* mixed_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_test_symbol_array_attrs_isa(mixed_op));
  loom_symbol_ref_array_t dependencies =
      loom_test_symbol_array_attrs_dependencies(mixed_op);
  ASSERT_EQ(dependencies.count, 3u);
  EXPECT_EQ(dependencies.values[0].symbol_id, b_symbol_id);
  EXPECT_EQ(dependencies.values[1].symbol_id, a_symbol_id);
  EXPECT_EQ(dependencies.values[2].symbol_id, b_symbol_id);
  loom_symbol_ref_array_t available =
      loom_test_symbol_array_attrs_available(mixed_op);
  ASSERT_EQ(available.count, 1u);
  EXPECT_EQ(available.values[0].symbol_id, a_symbol_id);

  loom_op_t* present_empty_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_test_symbol_array_attrs_isa(present_empty_op));
  EXPECT_FALSE(loom_attr_is_absent(loom_op_attrs(
      present_empty_op)[loom_test_symbol_array_attrs_available_ATTR_INDEX]));
  EXPECT_EQ(loom_test_symbol_array_attrs_available(present_empty_op).count, 0u);

  loom_op_t* absent_op = loom_block_op(entry, 2);
  ASSERT_TRUE(loom_test_symbol_array_attrs_isa(absent_op));
  EXPECT_TRUE(loom_attr_is_absent(loom_op_attrs(
      absent_op)[loom_test_symbol_array_attrs_available_ATTR_INDEX]));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, SymbolSetsPreserveCanonicalNamesAndPresence) {
  loom_module_t* module = CreateSymbolSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_SET);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);
  const loom_string_id_t function_name_id =
      loom_module_lookup_string(read_module, IREE_SV("symbol_sets"));
  const uint16_t function_symbol_id =
      loom_module_find_symbol(read_module, function_name_id);
  ASSERT_NE(function_symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* func_op =
      read_module->symbols.entries[function_symbol_id].defining_op;
  ASSERT_NE(func_op, nullptr);
  loom_block_t* entry = loom_region_entry_block(loom_test_func_body(func_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);

  const loom_string_id_t alpha_name_id =
      loom_module_lookup_string(read_module, IREE_SV("alpha"));
  const loom_string_id_t zeta_name_id =
      loom_module_lookup_string(read_module, IREE_SV("zeta"));
  const uint16_t alpha_symbol_id =
      loom_module_find_symbol(read_module, alpha_name_id);
  const uint16_t zeta_symbol_id =
      loom_module_find_symbol(read_module, zeta_name_id);
  ASSERT_NE(alpha_symbol_id, LOOM_SYMBOL_ID_INVALID);
  ASSERT_NE(zeta_symbol_id, LOOM_SYMBOL_ID_INVALID);

  loom_op_t* populated_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_test_symbol_set_attrs_isa(populated_op));
  loom_symbol_ref_array_t symbols =
      loom_test_symbol_set_attrs_symbols(populated_op);
  ASSERT_EQ(symbols.count, 2u);
  EXPECT_EQ(symbols.values[0].symbol_id, alpha_symbol_id);
  EXPECT_EQ(symbols.values[1].symbol_id, zeta_symbol_id);

  loom_op_t* empty_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_test_symbol_set_attrs_isa(empty_op));
  EXPECT_EQ(loom_test_symbol_set_attrs_symbols(empty_op).count, 0u);

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ParameterizedAttrsPreserveNamedSlotsAndPresence) {
  loom_module_t* module = CreateParameterizedAttrModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids, /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(func_op, nullptr);
  loom_block_t* entry = loom_region_entry_block(loom_test_func_body(func_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 8u);

  loom_attribute_t full =
      loom_test_parameterized_attr_options(loom_block_op(entry, 0));
  ASSERT_TRUE(loom_test_options_attr_isa(full));
  EXPECT_EQ(loom_test_options_attr_mode(full), LOOM_TEST_OPTIONS_MODE_FAST);
  ASSERT_TRUE(loom_test_options_attr_has_scopes(full));
  loom_enum_array_t scopes = loom_test_options_attr_scopes(full);
  ASSERT_EQ(scopes.count, 2u);
  EXPECT_EQ(scopes.values[0], LOOM_TEST_OPTIONS_SCOPES_SUBGROUP);
  EXPECT_EQ(scopes.values[1], 254u);
  ASSERT_TRUE(loom_test_options_attr_has_element_type(full));
  const loom_type_id_t element_type_id =
      loom_test_options_attr_element_type(full);
  ASSERT_LT(element_type_id, read_module->types.count);
  EXPECT_TRUE(loom_type_equal(read_module->types.entries[element_type_id],
                              loom_type_scalar(LOOM_SCALAR_TYPE_BF16)));
  ASSERT_TRUE(loom_test_options_attr_has_tile(full));
  loom_attribute_t tile = loom_test_options_attr_tile(full);
  ASSERT_TRUE(loom_test_tile_attr_isa(tile));
  EXPECT_EQ(loom_test_tile_attr_width(tile), 16);

  loom_attribute_t present_empty =
      loom_test_parameterized_attr_options(loom_block_op(entry, 1));
  ASSERT_TRUE(loom_test_options_attr_has_scopes(present_empty));
  EXPECT_EQ(loom_test_options_attr_scopes(present_empty).count, 0u);

  loom_attribute_t absent =
      loom_test_parameterized_attr_options(loom_block_op(entry, 2));
  EXPECT_FALSE(loom_test_options_attr_has_scopes(absent));

  loom_attribute_t compact =
      loom_test_compact_parameterized_attr_value(loom_block_op(entry, 3));
  ASSERT_TRUE(loom_test_compact_attr_isa(compact));
  EXPECT_EQ(loom_test_compact_attr_value(compact), 64);
  ASSERT_TRUE(loom_test_compact_attr_has_label(compact));
  loom_string_id_t label_id = loom_test_compact_attr_label(compact);
  ASSERT_LT(label_id, read_module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(read_module->strings.entries[label_id],
                                     IREE_SV("wave")));

  loom_op_t* array_op = loom_block_op(entry, 4);
  ASSERT_TRUE(loom_test_parameterized_attr_array_isa(array_op));
  loom_parameterized_attr_array_t values =
      loom_test_parameterized_attr_array_values(array_op);
  ASSERT_EQ(values.count, 3u);
  EXPECT_TRUE(loom_test_tile_attr_isa(values.values[0]));
  EXPECT_TRUE(loom_test_options_attr_isa(values.values[1]));
  EXPECT_TRUE(loom_test_tile_attr_isa(values.values[2]));
  EXPECT_TRUE(loom_attribute_equal(&values.values[0], &values.values[2]));
  loom_parameterized_attr_array_t exact_tiles =
      loom_test_parameterized_attr_array_tiles(array_op);
  ASSERT_EQ(exact_tiles.count, 1u);
  EXPECT_TRUE(loom_test_tile_attr_isa(exact_tiles.values[0]));

  loom_op_t* present_empty_array_op = loom_block_op(entry, 5);
  EXPECT_FALSE(loom_attr_is_absent(loom_op_attrs(present_empty_array_op)[1]));
  EXPECT_EQ(
      loom_test_parameterized_attr_array_tiles(present_empty_array_op).count,
      0u);
  loom_op_t* absent_array_op = loom_block_op(entry, 6);
  EXPECT_TRUE(loom_attr_is_absent(loom_op_attrs(absent_array_op)[1]));

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
  const loom_value_t& dim_value = *loom_module_value(read_module, dim_id);
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

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t metadata = {0};
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t index_result =
      ReadIndex(bytes, &metadata_arena, &metadata, &error_ids);
  ASSERT_EQ(index_result.error_count, 0u);
  ASSERT_TRUE(error_ids.empty());
  ASSERT_EQ(metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t& module_metadata = metadata.modules[0];
  ASSERT_EQ(module_metadata.sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_metadata.sources.values[0],
                                     IREE_SV("model.loom")));
  ASSERT_EQ(module_metadata.locations.count, 2u);
  ASSERT_NE(module_metadata.locations.entries, nullptr);
  for (iree_host_size_t i = 0; i < module_metadata.locations.count; ++i) {
    EXPECT_GT(module_metadata.locations.entries[i].entry_length, 0u);
  }

  loom_context_t read_context;
  InitializeBytecodeTestContext(&read_context);

  loom_module_t* read_module = nullptr;
  error_ids.clear();
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
  iree_arena_deinitialize(&metadata_arena);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsTaggedLocationTables) {
  loom_module_t* module = CreateLocatedModule();
  const uint8_t data[] = {0x01, 0x2A, 0xFF};
  loom_location_id_t tagged_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_CHECK_OK(loom_module_add_location(
      module,
      loom_location_tagged(LOOM_LOCATION_TAG_SANITIZER_SITE,
                           /*child=*/1, data, IREE_ARRAYSIZE(data)),
      &tagged_location_id));
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->locations.count, 3u);
  const loom_location_entry_t& tagged_location =
      read_module->locations.entries[tagged_location_id];
  EXPECT_EQ(tagged_location.kind, LOOM_LOCATION_TAGGED);
  EXPECT_EQ(tagged_location.tagged.tag, LOOM_LOCATION_TAG_SANITIZER_SITE);
  EXPECT_EQ(tagged_location.tagged.child, 1u);
  EXPECT_EQ(tagged_location.tagged.data_length, IREE_ARRAYSIZE(data));
  ASSERT_NE(tagged_location.tagged.data, nullptr);
  EXPECT_EQ(
      std::memcmp(tagged_location.tagged.data, data, IREE_ARRAYSIZE(data)), 0);

  loom_module_free(read_module);
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
  loom_op_t* source_func_op = module->symbols.entries[0].defining_op;
  loom_region_t* source_body = loom_test_func_body(source_func_op);
  loom_op_t* source_deflate_op = loom_region_entry_block(source_body)->first_op;
  loom_value_slice_t source_results =
      loom_test_deflate_results(source_deflate_op);
  ASSERT_EQ(source_results.count, 2u);
  EXPECT_TRUE(
      loom_module_value_has_type_uses(module, source_results.values[1]));
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
  EXPECT_TRUE(loom_module_value_has_type_uses(read_module, results.values[1]));

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

  loom_module_t* parameterized_type_module =
      CreateParameterizedTypeDeclModule();
  ExpectCanonicalBytecodeRoundTrip(parameterized_type_module);
  loom_module_free(parameterized_type_module);

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

  loom_module_t* empty_predicate_module = CreateEmptyPredicateFunctionModule();
  ExpectCanonicalBytecodeRoundTrip(empty_predicate_module);
  loom_module_free(empty_predicate_module);
}

TEST_F(ReaderTest, ReadsStructuralRegisterValueType) {
  loom_module_t* source_module = CreateRegisterDeclModule();
  auto bytes = WriteModule(source_module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);
  EXPECT_EQ(result.error_count, 0u) << ::testing::PrintToString(error_ids);
  EXPECT_TRUE(error_ids.empty()) << ::testing::PrintToString(error_ids);
  ASSERT_NE(read_module, nullptr);

  const loom_type_t* register_type = nullptr;
  for (iree_host_size_t i = 0; i < read_module->types.count; ++i) {
    if (loom_type_is_register(read_module->types.entries[i])) {
      register_type = &read_module->types.entries[i];
      break;
    }
  }
  ASSERT_NE(register_type, nullptr);
  EXPECT_EQ(loom_type_register_payload0(*register_type), 1u);
  EXPECT_EQ(loom_type_register_payload1(*register_type), (uint64_t)4 << 16);
  const loom_type_t* value_type = loom_type_register_value_type(*register_type);
  ASSERT_NE(value_type, nullptr);
  ASSERT_TRUE(loom_type_is_dialect(*value_type));
  EXPECT_TRUE(iree_string_view_equal(
      read_module->strings.entries[loom_type_dialect_name_id(*value_type)],
      IREE_SV("test.payload")));
  ASSERT_EQ(loom_type_dialect_param_count(*value_type), 1u);
  EXPECT_TRUE(loom_type_equal(
      loom_type_dialect_params(*value_type)[0],
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0)));

  loom_module_free(read_module);
  loom_module_free(source_module);
}

TEST_F(ReaderTest, ReadsDescriptorBackedParameterizedTypes) {
  loom_module_t* source_module = CreateParameterizedTypeDeclModule();
  auto bytes = WriteModule(source_module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  const loom_op_t* decl_op = read_module->symbols.entries[0].defining_op;
  ASSERT_NE(decl_op, nullptr);
  loom_value_slice_t arguments = loom_test_decl_args(decl_op);
  ASSERT_EQ(arguments.count, 6u);

  loom_type_t scope_type =
      loom_module_value_type(read_module, arguments.values[0]);
  ASSERT_TRUE(loom_test_scope_type_isa(scope_type));
  EXPECT_EQ(loom_test_scope_type_scope(scope_type),
            LOOM_TEST_SCOPE_TYPE_SCOPE_SUBGROUP);

  loom_type_t matrix_type =
      loom_module_value_type(read_module, arguments.values[1]);
  ASSERT_TRUE(loom_test_matrix_type_isa(matrix_type));
  EXPECT_EQ(loom_test_matrix_type_scope(matrix_type),
            LOOM_TEST_MATRIX_TYPE_SCOPE_WORKGROUP);
  EXPECT_EQ(loom_test_matrix_type_rows(matrix_type), 16);
  loom_type_id_t element_type_id =
      loom_test_matrix_type_element_type(matrix_type);
  ASSERT_LT(element_type_id, read_module->types.count);
  EXPECT_TRUE(loom_type_equal(read_module->types.entries[element_type_id],
                              loom_type_scalar(LOOM_SCALAR_TYPE_BF16)));

  loom_type_t packed_type =
      loom_module_value_type(read_module, arguments.values[2]);
  ASSERT_TRUE(loom_test_array_type_isa(packed_type));
  EXPECT_FALSE(loom_test_array_type_has_alignment(packed_type));
  loom_type_t aligned_type =
      loom_module_value_type(read_module, arguments.values[3]);
  ASSERT_TRUE(loom_test_array_type_isa(aligned_type));
  EXPECT_TRUE(loom_test_array_type_has_alignment(aligned_type));
  EXPECT_EQ(loom_test_array_type_alignment(aligned_type), 32);

  loom_type_t variants_type =
      loom_module_value_type(read_module, arguments.values[4]);
  ASSERT_TRUE(loom_test_variant_set_type_isa(variants_type));
  loom_parameterized_attr_array_t variants =
      loom_test_variant_set_type_values(variants_type);
  ASSERT_EQ(variants.count, 3u);
  EXPECT_TRUE(loom_test_tile_attr_isa(variants.values[0]));
  EXPECT_TRUE(loom_test_options_attr_isa(variants.values[1]));
  EXPECT_TRUE(loom_test_tile_attr_isa(variants.values[2]));
  EXPECT_TRUE(loom_attribute_equal(&variants.values[0], &variants.values[2]));
  EXPECT_TRUE(loom_test_variant_set_type_has_alternatives(variants_type));
  EXPECT_EQ(loom_test_variant_set_type_alternatives(variants_type).count, 0u);

  loom_type_t variants_absent_type =
      loom_module_value_type(read_module, arguments.values[5]);
  ASSERT_TRUE(loom_test_variant_set_type_isa(variants_absent_type));
  EXPECT_FALSE(
      loom_test_variant_set_type_has_alternatives(variants_absent_type));

  loom_module_free(read_module);
  loom_module_free(source_module);
}

TEST_F(ReaderTest, RejectsUnassignedTypeKind) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_TYPES);
  ASSERT_GT(ReadUVarint(bytes, &offset), 0u);
  bytes[offset] = 4;

  ExpectReadError(bytes, "ERR_BYTECODE_004");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownParameterizedTypeFamily) {
  loom_module_t* module = CreateParameterizedTypeDeclModule();
  auto bytes = WriteModule(module);
  ReplaceBytes(&bytes, "test.scope", "test.bogus");

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidRegisterValueTypePresence) {
  loom_module_t* module = CreateRegisterDeclModule();
  auto bytes = WriteModule(module);
  RegisterTypeOffsets offsets = ReadRegisterDeclTypeOffsets(bytes);
  bytes[offsets.has_value_type] = 2;

  ExpectReadError(bytes, "ERR_BYTECODE_011");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsForwardRegisterValueTypeReference) {
  loom_module_t* module = CreateRegisterDeclModule();
  auto bytes = WriteModule(module);
  RegisterTypeOffsets offsets = ReadRegisterDeclTypeOffsets(bytes);
  ASSERT_EQ(bytes[offsets.value_type], 2u);
  bytes[offsets.value_type] = 3;

  ExpectReadError(bytes, "ERR_BYTECODE_012");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsRegisterPayloadReservedBits) {
  loom_module_t* module =
      CreateRegisterDeclModule(((uint64_t)1 << 48) | ((uint64_t)4 << 16));
  auto bytes = WriteModule(module);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
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

TEST_F(ReaderTest, RejectsUnsupportedRegionSourceFlags) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t source_flags_offset = RootRegionSourceFlagsOffset(bytes);
  ASSERT_EQ(bytes[source_flags_offset], 0u);
  bytes[source_flags_offset] = 1u << 1;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSourceTriviaCommentCountBeyondFieldWidth) {
  loom_module_t* module = CreateFunctionModule();
  loom_op_t* func_op = module->symbols.entries[0].defining_op;
  loom_region_t* body =
      loom_func_like_body(loom_func_like_cast(module, func_op));
  const iree_string_view_t block_comments[] = {IREE_SV("x")};
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      module, loom_region_entry_block(body), block_comments,
      IREE_ARRAYSIZE(block_comments)));
  auto bytes = WriteModule(module);
  size_t source_trivia_offset = RootBlockSourceTriviaOffset(bytes);
  ASSERT_EQ(bytes[source_trivia_offset], 2u);
  ASSERT_EQ(bytes[source_trivia_offset + 1], 2u);
  ASSERT_EQ(bytes[source_trivia_offset + 2], ' ');
  bytes[source_trivia_offset] = 0x80;
  bytes[source_trivia_offset + 1] = 0x80;
  bytes[source_trivia_offset + 2] = 0x08;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

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

TEST_F(ReaderTest, RejectsEnumArrayForNonEnumArrayField) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.attr_kind] = LOOM_BYTECODE_ATTR_ENUM_ARRAY;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsEnumArrayNestedInGenericDictionary) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.nested_dict_first_value_kind] =
      LOOM_BYTECODE_ATTR_ENUM_ARRAY;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSignedEnumSetForNonSignedEnumSetField) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.attr_kind] = LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSymbolArrayForNonSymbolArrayField) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.attr_kind] = LOOM_BYTECODE_ATTR_SYMBOL_ARRAY;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSignedEnumSetWithExcessWordCount) {
  loom_module_t* module = CreateSignedEnumSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET);
  ASSERT_EQ(bytes[value_offsets.payload], 4u);
  bytes[value_offsets.payload] = 5;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_009");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSymbolArrayNestedInGenericDictionary) {
  loom_module_t* module = CreateAttributeFunctionModule();
  auto bytes = WriteModule(module);
  BodyOpAttrOffsets attr_offsets = FirstBodyOpAttrOffsets(bytes);
  bytes[attr_offsets.nested_dict_first_value_kind] =
      LOOM_BYTECODE_ATTR_SYMBOL_ARRAY;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsOversizedSymbolArray) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_ARRAY);
  ASSERT_EQ(bytes[value_offsets.payload], 3u);
  ASSERT_LT(value_offsets.payload + 2, bytes.size());
  bytes[value_offsets.payload] = 0x80;
  bytes[value_offsets.payload + 1] = 0x80;
  bytes[value_offsets.payload + 2] = 0x04;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_009");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSignedEnumSetWithContradictoryAssertion) {
  loom_module_t* module = CreateSignedEnumSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET);
  const size_t positive_word_offset = value_offsets.payload + 1;
  const size_t negative_word_offset =
      positive_word_offset + 4 * sizeof(uint64_t);
  WriteU64LE(&bytes, negative_word_offset,
             ReadU64LE(bytes, negative_word_offset) |
                 ReadU64LE(bytes, positive_word_offset));

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSignedEnumSetWithTrailingZeroPair) {
  loom_module_t* module = CreateSignedEnumSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET);
  const size_t positive_word_offset = value_offsets.payload + 1;
  ASSERT_NE(ReadU64LE(bytes, positive_word_offset + 3 * sizeof(uint64_t)), 0u);
  WriteU64LE(&bytes, positive_word_offset + 3 * sizeof(uint64_t), 0);

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSignedEnumSetWithUndeclaredStableValue) {
  loom_module_t* module = CreateSignedEnumSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET);
  const size_t positive_word_offset = value_offsets.payload + 1;
  WriteU64LE(&bytes, positive_word_offset,
             ReadU64LE(bytes, positive_word_offset) | (UINT64_C(1) << 2));

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsOutOfRangeSymbolArrayName) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_ARRAY);
  ASSERT_EQ(bytes[value_offsets.payload], 3u);
  ASSERT_LT(value_offsets.payload + 1, bytes.size());
  bytes[value_offsets.payload + 1] = 0x7F;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_010");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownSymbolArrayName) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_ARRAY);
  ASSERT_EQ(bytes[value_offsets.payload], 3u);
  ASSERT_LT(value_offsets.payload + 1, bytes.size());
  bytes[value_offsets.payload + 1] = 0;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsNoncanonicalSymbolSetOrder) {
  loom_module_t* module = CreateSymbolSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_SET);
  ASSERT_EQ(bytes[value_offsets.payload], 2u);
  ASSERT_LT(value_offsets.payload + 2, bytes.size());
  std::swap(bytes[value_offsets.payload + 1], bytes[value_offsets.payload + 2]);

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsDuplicateSymbolSetName) {
  loom_module_t* module = CreateSymbolSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_SET);
  ASSERT_EQ(bytes[value_offsets.payload], 2u);
  ASSERT_LT(value_offsets.payload + 2, bytes.size());
  bytes[value_offsets.payload + 2] = bytes[value_offsets.payload + 1];

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSymbolArrayForSymbolSetField) {
  loom_module_t* module = CreateSymbolSetModule();
  auto bytes = WriteModule(module);
  AttributeValueOffsets value_offsets =
      FirstBodyOpFirstAttributeValueOffsets(bytes);
  ASSERT_EQ(bytes[value_offsets.kind], LOOM_BYTECODE_ATTR_SYMBOL_SET);
  bytes[value_offsets.kind] = LOOM_BYTECODE_ATTR_SYMBOL_ARRAY;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_006");

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

TEST_F(ReaderTest, RejectsSymbolReferenceTargetOutsideSymbolTable) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);
  SymbolReferenceLayout layout = ReadSymbolReferenceLayout(bytes);
  ASSERT_EQ(layout.symbol_dependencies.size(), 3u);
  ASSERT_EQ(layout.symbol_dependencies[2].size(), 3u);
  const size_t target_offset = layout.symbol_dependencies[2][0];
  ASSERT_LT(target_offset, bytes.size());
  bytes[target_offset] = 3;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSymbolReferenceCountBeyondSectionPayload) {
  loom_module_t* module = CreateSymbolArrayModule();
  auto bytes = WriteModule(module);
  SymbolReferenceLayout layout = ReadSymbolReferenceLayout(bytes);
  ASSERT_LT(layout.total_dependency_count, bytes.size());
  ASSERT_EQ(bytes[layout.total_dependency_count], 3u);
  bytes[layout.total_dependency_count] = 4;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidContractDemandStringReference) {
  loom_module_t* module = CreateContractDemandModule();
  auto bytes = WriteModule(module);
  SymbolReferenceLayout layout = ReadSymbolReferenceLayout(bytes);
  ASSERT_EQ(layout.symbol_contract_demands.size(), 1u);
  ASSERT_EQ(layout.symbol_contract_demands[0].size(), 2u);
  const size_t contract_offset = layout.symbol_contract_demands[0][0];
  ASSERT_LT(contract_offset, bytes.size());
  bytes[contract_offset] = 0x7F;

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
  EXPECT_EQ(loom_module_encoding(module, encoding_id)->family.id,
            LOOM_ENCODING_FAMILY_ID_INVALID);
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
