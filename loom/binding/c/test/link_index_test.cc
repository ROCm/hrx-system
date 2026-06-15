// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link_index.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

SourcePtr CreateSource(loomc_source_format_t format, const char* identifier,
                       const void* contents, size_t contents_length) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, contents_length),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

SourcePtr CreateTextSource(const char* identifier, const char* source_text) {
  return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, identifier, source_text,
                      strlen(source_text));
}

BuilderPtr CreateBuilder(loomc_context_t* context) {
  loomc_link_index_builder_t* builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder);
  LOOMC_EXPECT_OK(status);
  return BuilderPtr(builder);
}

void FinishSucceeded(loomc_link_index_builder_t* builder,
                     LinkIndexPtr* out_link_index) {
  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_index_builder_finish(builder, &link_index, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(link_index, nullptr);
  *out_link_index = LinkIndexPtr(link_index);
}

std::vector<uint8_t> WriteBytecodeModule(const char* source_text) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  loom_context_t context = {};
  loom_context_initialize(allocator, &context);
  IREE_CHECK_OK(loom_op_registry_register_all_dialects(&context));
  IREE_CHECK_OK(loom_context_finalize(&context));

  loom_text_parse_options_t parse_options = {
      /*.diagnostic_sink=*/{},
      /*.max_errors=*/20,
      /*.low_asm_environment=*/{},
  };
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source_text),
                                IREE_SV("bytecode-input.loom"), &context,
                                &block_pool, &parse_options, &module));
  if (module == nullptr) {
    ADD_FAILURE() << "bytecode fixture source did not parse";
    loom_context_deinitialize(&context);
    iree_arena_block_pool_deinitialize(&block_pool);
    return {};
  }

  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, allocator, &stream));
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, &block_pool));

  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  std::vector<uint8_t> bytes(length);
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  iree_io_stream_release(stream);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
  return bytes;
}

TEST(LinkIndexTest, IndexesTextSourceAndPrivateSymbols) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("kernel.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  loomc_link_index_source_options_t options = {
      /*.provider_name=*/loomc_make_cstring_view("app"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), &options, nullptr));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  EXPECT_EQ(loomc_link_index_provider_count(link_index.get()), 1u);
  EXPECT_EQ(loomc_link_index_module_count(link_index.get()), 1u);
  EXPECT_EQ(loomc_link_index_symbol_count(link_index.get()), 2u);

  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 0, &provider));
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_TEXT);
  EXPECT_EQ(provider.role, LOOMC_LINK_PROVIDER_ROLE_INPUT);
  EXPECT_EQ(ToString(provider.name), "app");

  loomc_link_index_symbol_t entry = {};
  ASSERT_TRUE(loomc_link_index_lookup_global(
      link_index.get(), loomc_make_cstring_view("@entry"), &entry));
  EXPECT_EQ(entry.identity, LOOMC_LINK_SYMBOL_IDENTITY_GLOBAL);
  EXPECT_TRUE((entry.flags & LOOMC_LINK_SYMBOL_FLAG_EXPORT) != 0);

  loomc_link_index_module_t module = {};
  ASSERT_TRUE(loomc_link_index_module_at(link_index.get(), 0, &module));
  loomc_link_index_symbol_t helper = {};
  ASSERT_TRUE(loomc_link_index_lookup_private(
      link_index.get(), &module, loomc_make_cstring_view("@helper"), &helper));
  EXPECT_EQ(helper.identity, LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE);
}

TEST(LinkIndexTest, InputProviderPrecedesLibraryProvider) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr library = CreateTextSource("library.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr input = CreateTextSource("input.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");

  loomc_link_index_source_options_t library_options = {
      /*.provider_name=*/loomc_make_cstring_view("library"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), library.get(), &library_options, nullptr));
  loomc_link_index_source_options_t input_options = {
      /*.provider_name=*/loomc_make_cstring_view("input"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), input.get(), &input_options, nullptr));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  loomc_link_index_symbol_t selected = {};
  ASSERT_TRUE(loomc_link_index_lookup_global(
      link_index.get(), loomc_make_cstring_view("entry"), &selected));
  loomc_link_index_provider_t selected_provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(
      link_index.get(), selected.provider_ordinal, &selected_provider));
  EXPECT_EQ(ToString(selected_provider.name), "input");

  loomc_link_index_symbol_t duplicate = {};
  ASSERT_TRUE(loomc_link_index_next_global_duplicate(link_index.get(),
                                                     &selected, &duplicate));
  loomc_link_index_provider_t duplicate_provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(
      link_index.get(), duplicate.provider_ordinal, &duplicate_provider));
  EXPECT_EQ(ToString(duplicate_provider.name), "library");
}

TEST(LinkIndexTest, DeterministicReservedSlotsIgnoreFillOrder) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr first = CreateTextSource("first.loom", R"(
func.def public @first(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr second = CreateTextSource("second.loom", R"(
func.def public @second(%x: i32) -> (i32) {
  func.return %x : i32
}
)");

  loomc_link_index_source_slot_t first_slot = {};
  loomc_link_index_source_options_t first_options = {
      /*.provider_name=*/loomc_make_cstring_view("first"),
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_source_slot(
      builder.get(), &first_options, &first_slot));
  loomc_link_index_source_slot_t second_slot = {};
  loomc_link_index_source_options_t second_options = {
      /*.provider_name=*/loomc_make_cstring_view("second"),
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_source_slot(
      builder.get(), &second_options, &second_slot));

  LOOMC_ASSERT_OK(loomc_link_index_builder_fill_source_slot(
      builder.get(), second_slot, second.get()));
  LOOMC_ASSERT_OK(loomc_link_index_builder_fill_source_slot(
      builder.get(), first_slot, first.get()));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 0, &provider));
  EXPECT_EQ(ToString(provider.name), "first");
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 1, &provider));
  EXPECT_EQ(ToString(provider.name), "second");
}

TEST(LinkIndexTest, IndexesBytecodeSourceWithoutMaterializedModule) {
  std::vector<uint8_t> bytecode = WriteBytecodeModule(R"(
func.def public @from_bytecode(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "module.loombc",
                                  bytecode.data(), bytecode.size());
  loomc_link_index_source_options_t options = {
      /*.provider_name=*/loomc_make_cstring_view("bytecode"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), &options, nullptr));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);
  source.reset();
  bytecode.clear();

  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 0, &provider));
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_BYTECODE);
  loomc_link_index_symbol_t symbol = {};
  ASSERT_TRUE(loomc_link_index_lookup_global(
      link_index.get(), loomc_make_cstring_view("from_bytecode"), &symbol));
  EXPECT_EQ(symbol.kind, LOOMC_LINK_SYMBOL_KIND_FUNCTION_DEFINITION);
}

TEST(LinkIndexTest, ParseErrorsProduceFailedResultDiagnostics) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("broken.loom", R"(
func.def public @broken(%x: i32) -> (i32) {
  func.return %missing : i32
}
)");
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), nullptr, nullptr));

  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_index_builder_finish(builder.get(), &link_index, &result);
  LOOMC_ASSERT_OK(status);
  LinkIndexPtr link_index_ptr(link_index);
  ResultPtr result_ptr(result);

  EXPECT_EQ(link_index_ptr.get(), nullptr);
  ASSERT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GT(loomc_result_diagnostic_count(result_ptr.get()), 0u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "broken.loom");
}

TEST(LinkIndexTest, EmptyReservedSlotProducesFailedResult) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  loomc_link_index_source_slot_t slot = {};
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_source_slot(builder.get(),
                                                               nullptr, &slot));

  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_index_builder_finish(builder.get(), &link_index, &result);
  LOOMC_ASSERT_OK(status);
  LinkIndexPtr link_index_ptr(link_index);
  ResultPtr result_ptr(result);

  EXPECT_EQ(link_index_ptr.get(), nullptr);
  ASSERT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "LINK_INDEX/EMPTY_SLOT");
}

}  // namespace
