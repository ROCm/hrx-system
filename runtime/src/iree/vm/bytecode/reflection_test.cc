// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/reflection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/execution_testdata.h"
#include "iree/vm/bytecode/verifier.h"

namespace {

constexpr uint16_t kRunCallableTypeOrdinal = 3;
constexpr uint16_t kRunExportOrdinal = 16;

class BytecodeReflectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_file_toc_t* files = iree_vm_bytecode_execution_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_execution_testdata_size(), 1u);
    const iree_const_byte_span_t contents =
        iree_make_const_byte_span(files[0].data, files[0].size);
    IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(contents, &plan_));
    image_.layout = plan_.layout;
  }

  iree_vm_bytecode_module_plan_t plan_ = {};
  iree_vm_bytecode_image_t image_ = {};
};

TEST_F(BytecodeReflectionTest, QueriesExportAndCallableDeclarations) {
  iree_vm_module_export_declaration_t export_declaration = {};
  iree_vm_bytecode_reflection_query_export(
      &image_.base_module, kRunExportOrdinal, &export_declaration);
  EXPECT_TRUE(
      iree_string_view_equal(export_declaration.export_name, IREE_SV("run")));
  EXPECT_EQ(export_declaration.callable_type_ordinal, kRunCallableTypeOrdinal);
  EXPECT_EQ(export_declaration.function_ordinal, 1u);
  EXPECT_EQ(export_declaration.metadata_count, 1u);

  iree_vm_module_callable_type_declaration_t callable_type = {};
  iree_vm_bytecode_reflection_query_callable_type(
      &image_.base_module, kRunCallableTypeOrdinal, &callable_type);
  ASSERT_EQ(callable_type.signature.arguments.count, 1u);
  ASSERT_EQ(callable_type.signature.results.count, 2u);
  EXPECT_EQ(callable_type.signature.arguments.value_count, 1u);
  EXPECT_EQ(callable_type.signature.arguments.ref_count, 0u);
  EXPECT_EQ(callable_type.signature.results.value_count, 1u);
  EXPECT_EQ(callable_type.signature.results.ref_count, 1u);
  EXPECT_EQ(callable_type.signature.arguments.data[0].kind,
            IREE_VM_SCALAR_TYPE_I32);
  EXPECT_EQ(callable_type.signature.results.data[1].kind,
            IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(callable_type.signature.results.data[1].type_ordinal, 0u);
}

TEST_F(BytecodeReflectionTest, QueriesStableAuthoredPresentation) {
  iree_vm_signature_field_t fields[3] = {};
  const iree_vm_module_presentation_query_t query = {
      {IREE_VM_MODULE_DECLARATION_KIND_EXPORT, kRunExportOrdinal},
      {fields, IREE_ARRAYSIZE(fields)},
      iree_byte_span_empty(),
  };
  iree_vm_module_presentation_t presentation = {};
  iree_vm_bytecode_reflection_query_presentation(&image_.base_module, &query,
                                                 &presentation);

  EXPECT_EQ(presentation.required_transient_storage_size, 0u);
  EXPECT_TRUE(iree_string_view_equal(
      presentation.documentation,
      IREE_SV("Adds the process bias and returns module read-only data.")));
  EXPECT_TRUE(
      iree_string_view_equal(presentation.authored_type,
                             IREE_SV("(i32) -> (i32, vm.ref<vm, buffer>)")));
  EXPECT_TRUE(iree_string_view_equal(fields[0].name, IREE_SV("value")));
  EXPECT_TRUE(iree_string_view_equal(fields[0].authored_type, IREE_SV("i32")));
  EXPECT_TRUE(iree_string_view_equal(fields[1].name, IREE_SV("sum")));
  EXPECT_TRUE(iree_string_view_equal(fields[2].name, IREE_SV("payload")));
  EXPECT_TRUE(iree_string_view_equal(fields[2].authored_type,
                                     IREE_SV("vm.ref<vm, buffer>")));
}

TEST_F(BytecodeReflectionTest, QueriesTypedMetadataByScope) {
  const iree_vm_module_metadata_query_t module_query = {
      {IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE, 0},
      0,
  };
  iree_vm_metadata_entry_t module_entry = {};
  iree_vm_bytecode_reflection_metadata_by_ordinal(&image_.base_module,
                                                  &module_query, &module_entry);
  EXPECT_TRUE(iree_string_view_equal(module_entry.key, IREE_SV("author")));
  EXPECT_EQ(module_entry.value.type, IREE_VM_METADATA_VALUE_TYPE_UTF8);
  EXPECT_TRUE(iree_string_view_equal(
      iree_make_string_view((const char*)module_entry.value.data.data,
                            module_entry.value.data.data_length),
      IREE_SV("loom")));

  const iree_vm_module_metadata_query_t export_query = {
      {IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT, kRunExportOrdinal},
      0,
  };
  iree_vm_metadata_entry_t export_entry = {};
  iree_vm_bytecode_reflection_metadata_by_ordinal(&image_.base_module,
                                                  &export_query, &export_entry);
  EXPECT_TRUE(iree_string_view_equal(export_entry.key, IREE_SV("purpose")));
  EXPECT_EQ(export_entry.value.type, IREE_VM_METADATA_VALUE_TYPE_UTF8);
  EXPECT_TRUE(iree_string_view_equal(
      iree_make_string_view((const char*)export_entry.value.data.data,
                            export_entry.value.data.data_length),
      IREE_SV("execution-test")));
}

}  // namespace
