// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link_index.h"

#include <cstring>
#include <limits>
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
#include "loomc/link.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
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

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* linker = nullptr;
  loomc_status_t status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  LOOMC_EXPECT_OK(status);
  return LinkerPtr(linker);
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

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(result_ptr && loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
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

TEST(LinkIndexTest, RejectsInvalidBlockSizes) {
  ContextPtr context = CreateContext();
  loomc_link_index_builder_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_INDEX_BUILDER_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.block_size=*/1,
  };
  loomc_link_index_builder_t* builder =
      reinterpret_cast<loomc_link_index_builder_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_link_index_builder_create(context.get(), &options,
                                      loomc_allocator_system(), &builder));
  EXPECT_EQ(builder, nullptr);

  options.block_size = std::numeric_limits<loomc_host_size_t>::max();
  builder = reinterpret_cast<loomc_link_index_builder_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_OUT_OF_RANGE,
      loomc_link_index_builder_create(context.get(), &options,
                                      loomc_allocator_system(), &builder));
  EXPECT_EQ(builder, nullptr);
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
  loomc_link_index_provider_options_t options = {
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

TEST(LinkIndexTest, ExposesGenericTestOnlySymbolRole) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("checks.loom", R"(
check.case public @kernel_case {
  check.return
}
)");
  loomc_link_index_provider_options_t options = {
      /*.provider_name=*/loomc_make_cstring_view("checks"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), &options, nullptr));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  loomc_link_index_symbol_t symbol = {};
  ASSERT_TRUE(loomc_link_index_lookup_global(
      link_index.get(), loomc_make_cstring_view("@kernel_case"), &symbol));
  EXPECT_TRUE((symbol.flags & LOOMC_LINK_SYMBOL_FLAG_TEST_ONLY) != 0);
}

TEST(LinkIndexTest, ExposesRetainedSymbolRole) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("retained.loom", R"(
func.def retain @entry() {
  func.return
}
)");
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), /*options=*/nullptr, /*out_slot=*/nullptr));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  loomc_link_index_module_t module = {};
  ASSERT_TRUE(loomc_link_index_module_at(link_index.get(), 0, &module));
  loomc_link_index_symbol_t symbol = {};
  ASSERT_TRUE(loomc_link_index_lookup_private(
      link_index.get(), &module, loomc_make_cstring_view("@entry"), &symbol));
  EXPECT_TRUE((symbol.flags & LOOMC_LINK_SYMBOL_FLAG_RETAIN) != 0);
}

TEST(LinkIndexTest, CanonicalGlobalOrderPlacesInputBeforeLibrary) {
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

  loomc_link_index_provider_options_t library_options = {
      /*.provider_name=*/loomc_make_cstring_view("library"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), library.get(), &library_options, nullptr));
  loomc_link_index_provider_options_t input_options = {
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
  WorkspacePtr workspace = CreateWorkspace();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr first_source = CreateTextSource("first.loom", R"(
func.def public @first(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ModulePtr first =
      DeserializeModule(context.get(), workspace.get(), first_source.get());
  ASSERT_NE(first, nullptr);
  SourcePtr second = CreateTextSource("second.loom", R"(
func.def public @second(%x: i32) -> (i32) {
  func.return %x : i32
}
)");

  loomc_link_index_provider_slot_t first_slot = {};
  loomc_link_index_provider_options_t first_options = {
      /*.provider_name=*/loomc_make_cstring_view("first"),
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_provider_slot(
      builder.get(), &first_options, &first_slot));
  loomc_link_index_provider_slot_t second_slot = {};
  loomc_link_index_provider_options_t second_options = {
      /*.provider_name=*/loomc_make_cstring_view("second"),
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_provider_slot(
      builder.get(), &second_options, &second_slot));

  LOOMC_ASSERT_OK(loomc_link_index_builder_fill_source_slot(
      builder.get(), second_slot, second.get()));
  LOOMC_ASSERT_OK(loomc_link_index_builder_fill_module_slot(
      builder.get(), first_slot, first.get()));

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);

  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 0, &provider));
  EXPECT_EQ(ToString(provider.name), "first");
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_MATERIALIZED);
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 1, &provider));
  EXPECT_EQ(ToString(provider.name), "second");
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_TEXT);
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
  loomc_link_index_provider_options_t options = {
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

TEST(LinkIndexTest, RetainsMaterializedModuleThroughFutureLink) {
  ContextPtr context = CreateContext();
  WorkspacePtr provider_workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("materialized.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ModulePtr provider_module =
      DeserializeModule(context.get(), provider_workspace.get(), source.get());
  ASSERT_NE(provider_module, nullptr);

  BuilderPtr builder = CreateBuilder(context.get());
  loomc_link_index_provider_options_t provider_options = {
      /*.provider_name=*/loomc_make_cstring_view("materialized"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_module(
      builder.get(), provider_module.get(), &provider_options, nullptr));

  // The builder retains the module and its workspace until ownership transfers
  // to the frozen index.
  provider_module.reset();
  provider_workspace.reset();
  source.reset();

  LinkIndexPtr link_index;
  FinishSucceeded(builder.get(), &link_index);
  builder.reset();

  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(link_index.get(), 0, &provider));
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_MATERIALIZED);
  EXPECT_EQ(ToString(provider.name), "materialized");

  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@entry"),
  };
  loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      /*.structure_size=*/sizeof(link_options),
      /*.next=*/nullptr,
      /*.link_index=*/link_index.get(),
      /*.module_name=*/loomc_string_view_empty(),
      /*.mode=*/LOOMC_LINK_MODE_LINK,
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
  };
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr output_workspace = CreateWorkspace();
  loomc_module_t* linked_module = nullptr;
  loomc_result_t* link_result = nullptr;
  LOOMC_ASSERT_OK(loomc_link_module(linker.get(), output_workspace.get(),
                                    &link_options, &linked_module,
                                    &link_result));
  ModulePtr linked_module_ptr(linked_module);
  ResultPtr link_result_ptr(link_result);
  ASSERT_TRUE(loomc_result_succeeded(link_result_ptr.get()));
  ASSERT_NE(linked_module_ptr, nullptr);

  // A linked module owns its cloned IR and remains valid after the provider
  // index (and therefore the original materialized module) is released.
  link_index.reset();

  loomc_module_function_t functions[2] = {};
  loomc_host_size_t function_count = 0;
  loomc_result_t* query_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_query_functions(
      linked_module_ptr.get(), nullptr, loomc_allocator_system(),
      IREE_ARRAYSIZE(functions), functions, &function_count, &query_result));
  ResultPtr query_result_ptr(query_result);
  ASSERT_TRUE(loomc_result_succeeded(query_result_ptr.get()));
  ASSERT_EQ(function_count, IREE_ARRAYSIZE(functions));

  const loomc_module_function_t* entry = nullptr;
  const loomc_module_function_t* helper = nullptr;
  for (const loomc_module_function_t& function : functions) {
    if (ToString(function.symbol_name) == "entry") {
      entry = &function;
    } else if (ToString(function.symbol_name) == "helper") {
      helper = &function;
    }
  }
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC, 0u);
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(helper->flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC, 0u);
}

TEST(LinkIndexTest, RejectedModuleDoesNotConsumeProviderSlot) {
  ContextPtr index_context = CreateContext();
  ContextPtr module_context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("provider.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ModulePtr module =
      DeserializeModule(module_context.get(), workspace.get(), source.get());
  ASSERT_NE(module, nullptr);

  BuilderPtr builder = CreateBuilder(index_context.get());
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_index_builder_add_module(
                             builder.get(), module.get(), nullptr, nullptr));

  loomc_link_index_provider_slot_t slot = {};
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), source.get(), nullptr, &slot));
  EXPECT_EQ(slot.ordinal, 0u);
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
  loomc_link_index_provider_slot_t slot = {};
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_provider_slot(
      builder.get(), nullptr, &slot));

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
