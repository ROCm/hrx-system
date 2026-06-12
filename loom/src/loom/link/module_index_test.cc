// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/module_index.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  std::string result = iree_status_code_string(iree_status_code(status));
  if (iree_status_to_string(status, &allocator, &buffer, &buffer_length)) {
    result.assign(buffer, buffer_length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return result;
}

class ModuleIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables,
                    loom_check_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables,
                    loom_config_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables,
                    loom_test_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream,
                                             /*options=*/nullptr,
                                             &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_create(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(ModuleIndexTest, IndexesMaterializedSymbolsByIdentity) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("app"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), module, &options, /*out_provider_ordinal=*/nullptr));

  ASSERT_EQ(loom_link_module_index_provider_count(index.get()), 1u);
  ASSERT_EQ(loom_link_module_index_module_count(index.get()), 1u);
  ASSERT_EQ(loom_link_module_index_symbol_count(index.get()), 2u);

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index.get(), 0);
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->kind, LOOM_LINK_PROVIDER_MATERIALIZED);
  EXPECT_EQ(provider->role, LOOM_LINK_PROVIDER_ROLE_INPUT);
  EXPECT_EQ(StringViewToString(provider->name), "app");

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@entry"));
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->identity, LOOM_LINK_SYMBOL_IDENTITY_GLOBAL);
  EXPECT_TRUE(iree_all_bits_set(entry->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("@helper"));
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(helper->identity, LOOM_LINK_SYMBOL_IDENTITY_PRIVATE);
  EXPECT_EQ(
      loom_link_module_index_lookup_global(index.get(), IREE_SV("helper")),
      nullptr);
}

TEST_F(ModuleIndexTest, InputProviderPrecedesBytecodeLibraryProvider) {
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  std::vector<uint8_t> library_bytes = WriteModule(library);
  loom_module_t* input = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t library_options = {
      /*.provider_name=*/IREE_SV("kernel-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(library_bytes.data(), library_bytes.size()),
      IREE_SV("kernel-lib.loombc"), /*read_options=*/nullptr, &library_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t input_options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), input, &input_options, /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* selected =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  ASSERT_NE(selected, nullptr);
  const loom_link_module_index_provider_t* selected_provider =
      loom_link_module_index_symbol_provider(index.get(), selected);
  ASSERT_NE(selected_provider, nullptr);
  EXPECT_EQ(StringViewToString(selected_provider->name), "input");

  const loom_link_module_index_symbol_t* duplicate =
      loom_link_module_index_next_global_duplicate(index.get(), selected);
  ASSERT_NE(duplicate, nullptr);
  const loom_link_module_index_provider_t* duplicate_provider =
      loom_link_module_index_symbol_provider(index.get(), duplicate);
  ASSERT_NE(duplicate_provider, nullptr);
  EXPECT_EQ(StringViewToString(duplicate_provider->name), "kernel-lib");

  std::string diagnostic =
      StatusToStringAndFree(loom_link_module_index_duplicate_global_status(
          index.get(), selected, duplicate));
  EXPECT_THAT(diagnostic, ::testing::HasSubstr("@entry"));
  EXPECT_THAT(diagnostic, ::testing::HasSubstr("input"));
  EXPECT_THAT(diagnostic, ::testing::HasSubstr("kernel-lib"));
}

TEST_F(ModuleIndexTest, ImportedDeclarationsHaveGlobalIdentity) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.decl public import("math", "dot") @dot(%a: f32, %b: f32) -> (f32)
)"));
  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("imports"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), module, &options, /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@dot"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->identity, LOOM_LINK_SYMBOL_IDENTITY_GLOBAL);
  EXPECT_TRUE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_IMPORT));
  EXPECT_TRUE(
      iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION));
  EXPECT_FALSE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));
}

TEST_F(ModuleIndexTest, IndexesBytecodeProviderWithoutMaterializingModule) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @exported(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("bytecode-provider"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("kernels.loombc"), /*read_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index.get(), 0);
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->kind, LOOM_LINK_PROVIDER_BYTECODE);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  EXPECT_EQ(indexed_module->materialized_module, nullptr);

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@exported"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->kind, LOOM_SYMBOL_FUNC_DEF);
  EXPECT_TRUE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));
}

TEST_F(ModuleIndexTest, IndexesCheckSymbolsForStripPolicy) {
  loom_module_t* module = Parse(IREE_SV(R"(
check.case public @kernel_case {
  check.return
}

check.benchmark<@kernel_case>
check.benchmark<@kernel_case> @kernel_bench {}
)"));
  ASSERT_NE(module, nullptr);
  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("checks"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("checks.loombc"), /*read_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  EXPECT_EQ(loom_link_module_index_symbol_count(index.get()), 2u);
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  ASSERT_NE(check_case, nullptr);
  EXPECT_TRUE(
      iree_all_bits_set(check_case->flags, LOOM_LINK_SYMBOL_FLAG_CHECK_CASE));
  EXPECT_TRUE(
      iree_all_bits_set(check_case->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY));

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* benchmark =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("kernel_bench"));
  ASSERT_NE(benchmark, nullptr);
  EXPECT_TRUE(iree_all_bits_set(benchmark->flags,
                                LOOM_LINK_SYMBOL_FLAG_CHECK_BENCHMARK));
}

TEST_F(ModuleIndexTest, IndexesTextProviderThroughMaterializedColdPath) {
  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("text-provider"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_text(
      index.get(), IREE_SV(R"(
func.def public @from_text(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
      IREE_SV("from_text.loom"), /*parse_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index.get(), 0);
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->kind, LOOM_LINK_PROVIDER_TEXT);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  EXPECT_NE(indexed_module->materialized_module, nullptr);
  EXPECT_TRUE(indexed_module->owns_materialized_module);

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@from_text"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->provider_ordinal, provider->ordinal);
}

TEST_F(ModuleIndexTest, DuplicatePrivateNamesRemainProviderLocal) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t first_options = {
      /*.provider_name=*/IREE_SV("first"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), first, &first_options, /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t second_options = {
      /*.provider_name=*/IREE_SV("second"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), second, &second_options, /*out_provider_ordinal=*/nullptr));

  ASSERT_EQ(loom_link_module_index_module_count(index.get()), 2u);
  const loom_link_module_index_module_t* first_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* second_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(first_module, nullptr);
  ASSERT_NE(second_module, nullptr);

  const loom_link_module_index_symbol_t* first_helper =
      loom_link_module_index_lookup_private(index.get(), first_module,
                                            IREE_SV("@helper"));
  const loom_link_module_index_symbol_t* second_helper =
      loom_link_module_index_lookup_private(index.get(), second_module,
                                            IREE_SV("@helper"));
  ASSERT_NE(first_helper, nullptr);
  ASSERT_NE(second_helper, nullptr);
  EXPECT_NE(first_helper->provider_ordinal, second_helper->provider_ordinal);
  EXPECT_EQ(
      loom_link_module_index_lookup_global(index.get(), IREE_SV("helper")),
      nullptr);
}

}  // namespace
}  // namespace loom
