// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/symbol_header_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/template/ops.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class SelectedFunctionHeaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_template_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEMPLATE, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseModule(iree_string_view_t source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("provider.loom"), &context_,
                                  &block_pool_, &options, &module));
    return module;
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(SelectedFunctionHeaderTest,
       DecodesProviderMetadataWithoutReadingBodyBytes) {
  loom_module_t* source_module = ParseModule(IREE_SV(R"(
template.decl @demo.dynamic(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>)

template.def<@demo.dynamic> priority(7) @identity(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)] {
  template.return %arg : tensor<[%m]xf32>
}
)"));
  std::vector<uint8_t> bytecode = WriteModule(source_module);
  loom_module_free(source_module);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool_, &metadata_arena);
  loom_bytecode_file_metadata_t file_metadata = {};
  const loom_bytecode_index_options_t index_options = {
      /*.diagnostic_sink=*/{AcceptDiagnostic, nullptr},
  };
  loom_bytecode_read_result_t index_result = {};
  IREE_ASSERT_OK(loom_bytecode_read_index(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("provider.loombc"), &context_, &block_pool_, &metadata_arena,
      &index_options, &index_result, &file_metadata));
  ASSERT_EQ(index_result.error_count, 0u);
  ASSERT_EQ(file_metadata.module_count, 1u);
  const loom_bytecode_module_metadata_t* module_metadata =
      &file_metadata.modules[0];
  const loom_bytecode_symbol_metadata_t* provider_metadata = nullptr;
  uint32_t family_ordinal = UINT32_MAX;
  uint32_t provider_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < module_metadata->symbol_count; ++i) {
    if (module_metadata->symbols[i].kind ==
        LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL) {
      family_ordinal = i;
    }
    if (module_metadata->symbols[i].kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF) {
      provider_metadata = &module_metadata->symbols[i];
      provider_ordinal = i;
      break;
    }
  }
  ASSERT_NE(family_ordinal, UINT32_MAX);
  ASSERT_NE(provider_metadata, nullptr);
  ASSERT_GT(provider_metadata->region_payload_count, 0u);
  const loom_bytecode_region_payload_metadata_t* provider_payloads =
      &module_metadata
           ->region_payloads[provider_metadata->first_region_payload_index];
  for (uint8_t i = 0; i < provider_metadata->region_payload_count; ++i) {
    ASSERT_GT(provider_payloads[i].length, 0u);
    std::fill_n(bytecode.data() + provider_payloads[i].absolute_offset,
                provider_payloads[i].length, UINT8_C(0xFF));
  }

  const loom_bytecode_symbol_header_reader_options_t reader_options = {
      /*.diagnostic_sink=*/{AcceptDiagnostic, nullptr},
  };
  loom_bytecode_symbol_header_reader_t* reader = nullptr;
  IREE_ASSERT_OK(loom_bytecode_symbol_header_reader_create(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("provider.loombc"), &context_, &block_pool_, module_metadata,
      &reader_options, iree_allocator_system(), &reader));
  loom_module_t* target_module =
      loom_bytecode_symbol_header_reader_module(reader);

  loom_bytecode_function_header_t header;
  IREE_ASSERT_OK(loom_bytecode_symbol_header_reader_read_function(
      reader, provider_ordinal, &header));
  ASSERT_EQ(header.region_payload_count,
            provider_metadata->region_payload_count);
  EXPECT_EQ(header.body_region_payload_ordinal_plus_one,
            provider_metadata->body_region_payload_ordinal_plus_one);
  EXPECT_EQ(header.kernel_workload_region_payload_ordinal_plus_one,
            provider_metadata->kernel_workload_region_payload_ordinal_plus_one);
  for (uint8_t i = 0; i < header.region_payload_count; ++i) {
    EXPECT_EQ(header.region_payloads[i].region_index,
              provider_payloads[i].region_index);
    EXPECT_EQ(header.region_payloads[i].offset, provider_payloads[i].offset);
    EXPECT_EQ(header.region_payloads[i].length, provider_payloads[i].length);
  }
  EXPECT_TRUE(iree_string_view_equal(header.name, IREE_SV("identity")));
  EXPECT_EQ(header.argument_count, 2u);
  EXPECT_EQ(header.result_count, 1u);

  const loom_symbol_ref_t family = loom_attr_as_symbol(
      header.attributes[header.func_like->template_family_attr_index]);
  ASSERT_TRUE(loom_symbol_ref_is_valid(family));
  const loom_symbol_t* family_symbol =
      &target_module->symbols.entries[family.symbol_id];
  EXPECT_TRUE(iree_string_view_equal(
      target_module->strings.entries[family_symbol->name_id],
      IREE_SV("demo.dynamic")));
  EXPECT_EQ(loom_attr_as_i64(
                header.attributes[header.func_like->priority_attr_index]),
            7);

  const loom_value_id_t* arguments =
      header.signature_values + header.workload_argument_count;
  const loom_value_id_t* results = arguments + header.argument_count;
  const loom_type_t argument_type =
      loom_module_value_type(target_module, arguments[1]);
  const loom_type_t result_type =
      loom_module_value_type(target_module, results[0]);
  ASSERT_TRUE(loom_type_is_tensor(argument_type));
  ASSERT_TRUE(loom_type_is_tensor(result_type));
  EXPECT_EQ(loom_type_dim_value_id_at(argument_type, 0), arguments[0]);
  EXPECT_EQ(loom_type_dim_value_id_at(result_type, 0), arguments[0]);

  const loom_attribute_t predicates =
      header.attributes[header.func_like->predicates_attr_index];
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].kind, LOOM_PREDICATE_MUL);
  EXPECT_EQ(predicates.predicate_list[0].arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ((loom_value_id_t)predicates.predicate_list[0].args[0],
            arguments[0]);

  // Header projection created a placeholder for the family reference. Exact
  // selected materialization must fill that same symbol instead of inventing
  // a duplicate identity.
  IREE_ASSERT_OK(loom_bytecode_symbol_header_reader_materialize_bodyless_symbol(
      reader, family_ordinal));
  ASSERT_NE(target_module->symbols.entries[family.symbol_id].defining_op,
            nullptr);
  EXPECT_EQ(target_module->symbols.entries[family.symbol_id].defining_op->kind,
            LOOM_OP_TEMPLATE_DECL);

  loom_bytecode_symbol_header_reader_free(reader);
  iree_arena_deinitialize(&metadata_arena);
}

}  // namespace
}  // namespace loom
