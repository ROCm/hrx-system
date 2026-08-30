// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C bytecode reader round-trip coverage over the checked-in text corpus.
//
// The text corpus is the durable source of representative Loom IR. This test
// uses it as bytecode reader seeds without checking in generated .loombc blobs:
// parse text, write bytecode, read bytecode through the C reader, and require a
// second write to produce identical bytes.

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/repr.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/test/corpus/text/golden_text_corpus.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

struct ScopedEnumPacket {
  // Canonical packet operation carrying the scoped identity.
  loom_op_t* op = nullptr;
  // Scoped-enum attribute storing the contract-local ordinal.
  loom_attribute_t* descriptor_attr = nullptr;
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* error_ids = static_cast<std::vector<std::string>*>(user_data);
  error_ids->push_back(loom_error_def_id(diagnostic->error));
  return loom_diagnostic_stderr_sink(nullptr, diagnostic);
}

// A deliberately reordered view of the descriptor used by the stable-key
// test below. The opaque set token and ordinal have no relationship to the
// generated core-test tables: only the stable contract and descriptor keys are
// shared across the two codec environments.
static const uint8_t kReorderedDescriptorSetToken = 0;
static constexpr uint32_t kReorderedAddI32Ordinal = 73;

static const loom_low_repr_descriptor_set_t* ReorderedLookupDescriptorSet(
    const loom_low_repr_environment_state_t* state, iree_string_view_t key) {
  (void)state;
  if (!iree_string_view_equal(key, IREE_SV("test.low.core"))) return nullptr;
  return reinterpret_cast<const loom_low_repr_descriptor_set_t*>(
      &kReorderedDescriptorSetToken);
}

static bool ReorderedResolveDescriptor(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set,
    iree_string_view_t key, loom_low_repr_descriptor_value_t* out_value) {
  (void)state;
  if (descriptor_set != reinterpret_cast<const loom_low_repr_descriptor_set_t*>(
                            &kReorderedDescriptorSetToken) ||
      !iree_string_view_equal(key, IREE_SV("test.add.i32"))) {
    return false;
  }
  *out_value = {
      /*.ordinal=*/kReorderedAddI32Ordinal,
      /*.effective_traits=*/LOOM_TRAIT_PURE,
  };
  return true;
}

static iree_string_view_t ReorderedDescriptorKey(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set, uint32_t ordinal) {
  (void)state;
  if (descriptor_set != reinterpret_cast<const loom_low_repr_descriptor_set_t*>(
                            &kReorderedDescriptorSetToken) ||
      ordinal != kReorderedAddI32Ordinal) {
    return iree_string_view_empty();
  }
  return IREE_SV("test.add.i32");
}

static const loom_low_repr_environment_vtable_t kReorderedEnvironmentVtable = {
    /*.lookup_descriptor_set=*/ReorderedLookupDescriptorSet,
    /*.resolve_descriptor=*/ReorderedResolveDescriptor,
    /*.descriptor_key=*/ReorderedDescriptorKey,
};

static const loom_low_repr_environment_t kReorderedEnvironment = {
    /*.vtable=*/&kReorderedEnvironmentVtable,
    // This codec has no state beyond its function table.
    /*.state=*/nullptr,
};

class ReaderCorpusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    // This is a full text-corpus integration test: every checked-in dialect can
    // appear in corpus entries and must round-trip through bytecode.
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_target_core_test_low_descriptor_registry_initialize(&low_registry_);
    loom_low_repr_environment_initialize(&low_registry_.registry,
                                         &low_repr_environment_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source, iree_string_view_t filename) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &options, &module));
    EXPECT_NE(module, nullptr)
        << "parse failed for " << std::string(filename.data, filename.size);
    return module;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_text_low_asm_environment_t low_asm_environment = {};
    loom_low_descriptor_text_asm_environment_initialize(&low_registry_.registry,
                                                        &low_asm_environment);
    const loom_text_print_options_t options = {
        /*.flags=*/LOOM_TEXT_PRINT_DEFAULT,
        /*.low_asm_environment=*/low_asm_environment,
    };
    IREE_EXPECT_OK(loom_text_print_module_to_builder_with_options(
        module, &builder, &options));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  ScopedEnumPacket FindScopedEnumPacket(loom_module_t* module) {
    loom_op_t* module_op = nullptr;
    loom_block_for_each_op(loom_module_block(module), module_op) {
      loom_func_like_t function = loom_func_like_cast(module, module_op);
      if (!loom_func_like_isa(function)) continue;
      loom_region_t* body = loom_func_like_body(function);
      if (!body || body->block_count == 0) continue;
      loom_op_t* body_op = nullptr;
      loom_block_for_each_op(loom_region_entry_block(body), body_op) {
        loom_attribute_t* attrs = loom_op_attrs(body_op);
        for (uint8_t i = 0; i < body_op->attribute_count; ++i) {
          if (attrs[i].kind == LOOM_ATTR_SCOPED_ENUM) {
            ScopedEnumPacket packet;
            packet.op = body_op;
            packet.descriptor_attr = &attrs[i];
            return packet;
          }
        }
      }
    }
    return {};
  }

  iree_status_t WriteModule(
      const loom_module_t* module,
      const loom_low_repr_environment_t& low_repr_environment,
      std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    loom_bytecode_write_options_t options = {};
    options.low_repr_environment = low_repr_environment;
    iree_status_t status =
        loom_bytecode_write_module(module, stream, &options, &block_pool_);

    if (iree_status_is_ok(status)) {
      iree_io_stream_pos_t length = iree_io_stream_length(stream);
      out_bytes->resize((size_t)length);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_read(stream, out_bytes->size(), out_bytes->data(),
                                   nullptr);
    }
    iree_io_stream_release(stream);
    return status;
  }

  loom_bytecode_read_result_t ReadModule(
      const std::vector<uint8_t>& bytes,
      const loom_low_repr_environment_t& low_repr_environment,
      loom_module_t** out_module, std::vector<std::string>* error_ids) {
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
        // The checked-in text corpus is a syntax/format corpus. Some entries
        // intentionally exercise constructs without being standalone semantic
        // programs, so this test isolates bytecode reader/writer canonicality.
        /*.verify_module=*/false,
        /*.verify_max_errors=*/20,
        /*.low_repr_environment=*/low_repr_environment,
    };
    loom_bytecode_read_result_t result = {0};
    IREE_EXPECT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("corpus.loombc"), &context_, &block_pool_, &options, &result,
        out_module, iree_allocator_system()));
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_;
  loom_low_repr_environment_t low_repr_environment_;
};

TEST_F(ReaderCorpusTest, CorpusIsNotEmpty) {
  EXPECT_GT(loom_test_corpus_text_size(), 0u);
}

TEST_F(ReaderCorpusTest, TextCorpusBytecodeRoundTripsCanonically) {
  const iree_file_toc_t* corpus = loom_test_corpus_text_create();
  size_t supported_count = 0;
  for (size_t i = 0; i < loom_test_corpus_text_size(); ++i) {
    const iree_file_toc_t& file = corpus[i];
    SCOPED_TRACE(file.name);
    iree_string_view_t filename = iree_make_cstring_view(file.name);
    iree_string_view_t source =
        iree_make_string_view(file.data, (iree_host_size_t)file.size);

    loom_module_t* module = Parse(source, filename);
    if (!module) continue;
    std::vector<uint8_t> first;
    iree_status_t write_status =
        WriteModule(module, low_repr_environment_, &first);
    if (!iree_status_is_ok(write_status)) {
      IREE_EXPECT_OK(write_status);
      loom_module_free(module);
      continue;
    }
    std::string source_text = Print(module);
    loom_module_free(module);
    ++supported_count;

    loom_module_t* read_module = nullptr;
    std::vector<std::string> error_ids;
    loom_bytecode_read_result_t result =
        ReadModule(first, low_repr_environment_, &read_module, &error_ids);
    EXPECT_EQ(result.error_count, 0u)
        << (error_ids.empty() ? "" : error_ids.front());
    EXPECT_TRUE(error_ids.empty())
        << (error_ids.empty() ? "" : error_ids.front());
    ASSERT_NE(read_module, nullptr);

    EXPECT_EQ(source_text, Print(read_module));

    std::vector<uint8_t> second;
    IREE_EXPECT_OK(WriteModule(read_module, low_repr_environment_, &second));
    loom_module_free(read_module);

    EXPECT_EQ(first, second);
  }
  EXPECT_GT(supported_count, 0u);
}

TEST_F(ReaderCorpusTest, DescriptorRowOrderIsNotSerialized) {
  static constexpr char kSource[] = R"(
test.target<low_core> @test_target

low.func.def target<test.low.core>(@test_target) @add(
    %lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) {
  %sum = low.op<test.add.i32>(%lhs, %rhs) :
      (reg<test.i32>, reg<test.i32>) -> reg<test.i32>
  low.return %sum : reg<test.i32>
}
)";

  loom_module_t* module =
      Parse(iree_make_cstring_view(kSource), IREE_SV("row_order.loom"));
  ASSERT_NE(module, nullptr);
  ScopedEnumPacket original_packet = FindScopedEnumPacket(module);
  ASSERT_NE(original_packet.op, nullptr);
  ASSERT_NE(original_packet.descriptor_attr, nullptr);
  const uint32_t original_ordinal =
      loom_attr_as_scoped_enum(*original_packet.descriptor_attr);
  EXPECT_EQ(original_packet.op->traits, LOOM_TRAIT_PURE);
  const std::string original_text = Print(module);
  std::vector<uint8_t> first;
  IREE_ASSERT_OK(WriteModule(module, low_repr_environment_, &first));
  loom_module_free(module);

  loom_module_t* reordered_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(first, kReorderedEnvironment, &reordered_module, &error_ids);
  ASSERT_EQ(result.error_count, 0u)
      << (error_ids.empty() ? "" : error_ids.front());
  ASSERT_TRUE(error_ids.empty())
      << (error_ids.empty() ? "" : error_ids.front());
  ASSERT_NE(reordered_module, nullptr);
  ScopedEnumPacket reordered_packet = FindScopedEnumPacket(reordered_module);
  ASSERT_NE(reordered_packet.op, nullptr);
  ASSERT_NE(reordered_packet.descriptor_attr, nullptr);
  EXPECT_EQ(loom_attr_as_scoped_enum(*reordered_packet.descriptor_attr),
            kReorderedAddI32Ordinal);
  EXPECT_EQ(reordered_packet.op->traits, LOOM_TRAIT_PURE);

  std::vector<uint8_t> second;
  IREE_ASSERT_OK(WriteModule(reordered_module, kReorderedEnvironment, &second));
  EXPECT_EQ(second, first);
  loom_module_free(reordered_module);

  loom_module_t* restored_module = nullptr;
  error_ids.clear();
  result =
      ReadModule(second, low_repr_environment_, &restored_module, &error_ids);
  ASSERT_EQ(result.error_count, 0u)
      << (error_ids.empty() ? "" : error_ids.front());
  ASSERT_TRUE(error_ids.empty())
      << (error_ids.empty() ? "" : error_ids.front());
  ASSERT_NE(restored_module, nullptr);
  ScopedEnumPacket restored_packet = FindScopedEnumPacket(restored_module);
  ASSERT_NE(restored_packet.op, nullptr);
  ASSERT_NE(restored_packet.descriptor_attr, nullptr);
  EXPECT_EQ(loom_attr_as_scoped_enum(*restored_packet.descriptor_attr),
            original_ordinal);
  EXPECT_EQ(restored_packet.op->traits, LOOM_TRAIT_PURE);
  EXPECT_EQ(Print(restored_module), original_text);
  loom_module_free(restored_module);
}

}  // namespace
}  // namespace loom
