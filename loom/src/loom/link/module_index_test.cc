// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/module_index.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
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
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

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
    RegisterDialect(LOOM_DIALECT_PIPELINE, loom_pipeline_dialect_vtables,
                    loom_pipeline_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables,
                    loom_target_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables,
                    loom_template_dialect_op_semantics);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
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
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
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
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  IndexPtr CreateOverlay(const loom_link_module_index_t* base_index) {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate_overlay(
        base_index, &block_pool_, iree_allocator_system(), &index));
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
  EXPECT_EQ(StringViewToString(entry->defining_op_name), "func.def");
  EXPECT_TRUE(iree_all_bits_set(entry->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("@helper"));
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(helper->identity, LOOM_LINK_SYMBOL_IDENTITY_PRIVATE);
  EXPECT_EQ(StringViewToString(helper->defining_op_name), "func.def");
  EXPECT_EQ(
      loom_link_module_index_lookup_global(index.get(), IREE_SV("helper")),
      nullptr);
}

TEST_F(ModuleIndexTest, CanonicalGlobalOrderPlacesInputBeforeBytecodeLibrary) {
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
      IREE_SV("kernel-lib.loombc"), /*index_options=*/nullptr, &library_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t input_options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), input, &input_options, /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t second_library_options = {
      /*.provider_name=*/IREE_SV("kernel-lib-2"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(library_bytes.data(), library_bytes.size()),
      IREE_SV("kernel-lib-2.loombc"), /*index_options=*/nullptr,
      &second_library_options, /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* selected =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  ASSERT_NE(selected, nullptr);
  const loom_link_module_index_provider_t* selected_provider =
      loom_link_module_index_symbol_provider(index.get(), selected);
  ASSERT_NE(selected_provider, nullptr);
  EXPECT_EQ(StringViewToString(selected_provider->name), "input");
  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(index.get());
  ASSERT_EQ(input_exports.count, 1u);
  EXPECT_EQ(input_exports.values[0], selected->ordinal);

  const loom_link_module_index_symbol_t* duplicate =
      loom_link_module_index_next_global_duplicate(index.get(), selected);
  ASSERT_NE(duplicate, nullptr);
  const loom_link_module_index_provider_t* duplicate_provider =
      loom_link_module_index_symbol_provider(index.get(), duplicate);
  ASSERT_NE(duplicate_provider, nullptr);
  EXPECT_EQ(StringViewToString(duplicate_provider->name), "kernel-lib-2");

  const loom_link_module_index_symbol_t* second_duplicate =
      loom_link_module_index_next_global_duplicate(index.get(), duplicate);
  ASSERT_NE(second_duplicate, nullptr);
  const loom_link_module_index_provider_t* second_duplicate_provider =
      loom_link_module_index_symbol_provider(index.get(), second_duplicate);
  ASSERT_NE(second_duplicate_provider, nullptr);
  EXPECT_EQ(StringViewToString(second_duplicate_provider->name), "kernel-lib");
  EXPECT_EQ(loom_link_module_index_next_global_duplicate(index.get(),
                                                         second_duplicate),
            nullptr);
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

TEST_F(ModuleIndexTest, TargetDeclarationsHaveGlobalIdentity) {
  loom_module_t* module = Parse(IREE_SV(R"(
target.decl @gpu
)"));
  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("targets"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index.get(), module, &options, /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@gpu"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->identity, LOOM_LINK_SYMBOL_IDENTITY_GLOBAL);
  EXPECT_TRUE(
      iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION));
  EXPECT_FALSE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));
}

TEST_F(ModuleIndexTest, IndexesBytecodeTargetDeclarations) {
  loom_module_t* module = Parse(IREE_SV(R"(
target.decl @gpu
)"));
  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("targets"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("targets.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@gpu"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->identity, LOOM_LINK_SYMBOL_IDENTITY_GLOBAL);
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
      IREE_SV("kernels.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index.get(), 0);
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->kind, LOOM_LINK_PROVIDER_BYTECODE);
  EXPECT_EQ(provider->bytecode.contents.data, bytes.data());
  EXPECT_EQ(provider->bytecode.contents.data_length, bytes.size());
  EXPECT_EQ(StringViewToString(provider->bytecode.filename), "kernels.loombc");
  ASSERT_EQ(provider->bytecode.metadata.module_count, 1u);
  EXPECT_EQ(provider->bytecode.metadata.modules[0].symbol_count, 1u);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  EXPECT_EQ(indexed_module->materialized_module, nullptr);

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@exported"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->kind, LOOM_SYMBOL_FUNC_DEF);
  EXPECT_EQ(StringViewToString(symbol->defining_op_name), "func.def");
  EXPECT_TRUE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));
}

TEST_F(ModuleIndexTest, PreservesRetainedSymbolRoleAcrossProviderForms) {
  const iree_string_view_t source = IREE_SV(R"(
func.def retain @retained() {
  func.return
}

func.def @ordinary() {
  func.return
}
)");
  loom_module_t* module = Parse(source);
  std::vector<uint8_t> bytes = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    const loom_link_module_index_symbol_t* retained =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("retained"));
    const loom_link_module_index_symbol_t* ordinary =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("ordinary"));
    ASSERT_NE(retained, nullptr);
    ASSERT_NE(ordinary, nullptr);
    EXPECT_EQ(StringViewToString(retained->defining_op_name), "func.def");
    EXPECT_EQ(StringViewToString(ordinary->defining_op_name), "func.def");
    EXPECT_TRUE(
        iree_all_bits_set(retained->flags, LOOM_LINK_SYMBOL_FLAG_RETAIN));
    EXPECT_FALSE(
        iree_any_bit_set(ordinary->flags, LOOM_LINK_SYMBOL_FLAG_RETAIN));
  };

  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), module, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr bytecode_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      bytecode_index.get(),
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("input.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(bytecode_index.get());

  IndexPtr text_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_text(
      text_index.get(), source, IREE_SV("input.loom"),
      /*parse_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(text_index.get());
}

TEST_F(ModuleIndexTest, PreservesProductCarrierAcrossProviderForms) {
  const iree_string_view_t source = IREE_SV(R"(
pipeline.def<kernel> @kernel_pipeline() launch() {
  pipeline.return
}

pipeline.def<command> @command_pipeline() launch() {
  pipeline.return
}

pipeline.def @generic_pipeline() launch() {
  pipeline.return
}
)");
  loom_module_t* module = Parse(source);
  std::vector<uint8_t> bytes = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    const loom_link_module_index_symbol_t* kernel_pipeline =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("kernel_pipeline"));
    const loom_link_module_index_symbol_t* command_pipeline =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("command_pipeline"));
    const loom_link_module_index_symbol_t* generic_pipeline =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("generic_pipeline"));
    ASSERT_NE(kernel_pipeline, nullptr);
    ASSERT_NE(command_pipeline, nullptr);
    ASSERT_NE(generic_pipeline, nullptr);
    EXPECT_EQ(StringViewToString(kernel_pipeline->defining_op_name),
              "pipeline.def");
    EXPECT_EQ(kernel_pipeline->product_carrier, LOOM_PIPELINE_DEF_SCOPE_KERNEL);
    EXPECT_EQ(command_pipeline->product_carrier,
              LOOM_PIPELINE_DEF_SCOPE_COMMAND);
    EXPECT_EQ(generic_pipeline->product_carrier, 0u);
  };

  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), module, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr bytecode_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      bytecode_index.get(),
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("input.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(bytecode_index.get());

  IndexPtr text_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_text(
      text_index.get(), source, IREE_SV("input.loom"),
      /*parse_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(text_index.get());
}

TEST_F(ModuleIndexTest, IndexesExplicitBytecodeExportWithoutPublicVisibility) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def export("entry") @entry(%x: i32) -> (i32) {
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
      IREE_SV("functions.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("@entry"));
  ASSERT_NE(symbol, nullptr);
  EXPECT_EQ(symbol->identity, LOOM_LINK_SYMBOL_IDENTITY_GLOBAL);
  EXPECT_TRUE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT));
  EXPECT_FALSE(iree_all_bits_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC));
}

TEST_F(ModuleIndexTest, ProjectsMaterializedAndBytecodeReferenceMetadata) {
  const iree_string_view_t source = IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  %z = template.apply<@demo.contract>(%y) : (i32) -> (i32)
  func.return %z : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

template.decl @demo.contract(%x: i32) -> (i32)

template.def<@demo.contract> @provider(%x: i32) -> (i32) {
  template.return %x : i32
}
)");
  loom_module_t* module = Parse(source);
  std::vector<uint8_t> bytes = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    EXPECT_EQ(indexed_module->dependencies.root_count, 0u);
    ASSERT_EQ(indexed_module->dependencies.count, 3u);
    ASSERT_EQ(indexed_module->template_demands.count, 1u);

    const loom_link_module_index_symbol_t* entry =
        loom_link_module_index_lookup_global(index, IREE_SV("entry"));
    const loom_link_module_index_symbol_t* helper =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("helper"));
    const loom_link_module_index_symbol_t* provider =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("provider"));
    const loom_link_module_index_symbol_t* family_declaration =
        loom_link_module_index_lookup_global(index, IREE_SV("demo.contract"));
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(helper, nullptr);
    ASSERT_NE(provider, nullptr);
    ASSERT_NE(family_declaration, nullptr);
    auto has_dependency = [&](const loom_link_module_index_symbol_t* source,
                              const loom_link_module_index_symbol_t* target,
                              uint8_t expected_source_root,
                              loom_symbol_interface_flags_t
                                  expected_target_interfaces) {
      for (iree_host_size_t i = 0; i < source->dependencies.count; ++i) {
        const iree_host_size_t dependency_index =
            source->dependencies.first + i;
        if (indexed_module->dependencies.values[dependency_index] ==
            target->module_symbol_ordinal) {
          EXPECT_EQ(indexed_module->dependencies
                        .source_root_region_indices_plus_one[dependency_index],
                    expected_source_root);
          EXPECT_EQ(
              indexed_module->dependencies.target_interfaces[dependency_index],
              expected_target_interfaces);
          return true;
        }
      }
      return false;
    };
    ASSERT_EQ(entry->dependencies.count, 2u);
    EXPECT_TRUE(
        has_dependency(entry, helper, 1u, LOOM_SYMBOL_INTERFACE_CALLABLE));
    EXPECT_TRUE(has_dependency(entry, family_declaration, 1u,
                               LOOM_SYMBOL_INTERFACE_TEMPLATE_FAMILY));
    ASSERT_EQ(provider->dependencies.count, 1u);
    EXPECT_TRUE(has_dependency(provider, family_declaration, 0u,
                               LOOM_SYMBOL_INTERFACE_TEMPLATE_FAMILY));
    ASSERT_EQ(entry->template_demands.count, 1u);
    EXPECT_EQ(
        indexed_module->template_demands
            .source_root_region_indices_plus_one[entry->template_demands.first],
        1u);

    ASSERT_EQ(loom_link_module_index_template_family_count(index), 1u);
    const loom_link_template_family_ordinal_t family_ordinal =
        indexed_module->template_demands.values[entry->template_demands.first];
    const loom_link_module_index_template_family_t* family =
        loom_link_module_index_template_family_at(index, family_ordinal);
    ASSERT_NE(family, nullptr);
    EXPECT_EQ(StringViewToString(family->name), "demo.contract");
    EXPECT_EQ(provider->template_family_ordinal, family_ordinal);
    EXPECT_EQ(family_declaration->template_family_ordinal, family_ordinal);
    EXPECT_TRUE(iree_all_bits_set(family_declaration->flags,
                                  LOOM_LINK_SYMBOL_FLAG_DECLARATION));
    EXPECT_FALSE(iree_any_bit_set(family_declaration->flags,
                                  LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION));
    EXPECT_EQ(family->providers.first_symbol_ordinal, provider->ordinal);
    EXPECT_EQ(family->providers.last_symbol_ordinal, provider->ordinal);
    EXPECT_EQ(family->providers.count, 1u);
    EXPECT_EQ(provider->next.template_provider_ordinal,
              LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  };

  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("library"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), module, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr bytecode_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      bytecode_index.get(),
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("library.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(bytecode_index.get());

  IndexPtr text_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_text(
      text_index.get(), source, IREE_SV("library.loom"),
      /*parse_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(text_index.get());
}

TEST_F(ModuleIndexTest, ProjectsMultiRootReferenceOriginsAcrossProviders) {
  loom_module_t* module = Parse(IREE_SV(R"(
test.record @config_dependency
test.record @implementation_dependency

test.split_func @split_root() {
  test.symbol_array_attrs [@config_dependency] using []
  test.yield
} launch {
  test.symbol_array_attrs [@implementation_dependency] using []
  test.yield
}
)"));
  std::vector<uint8_t> bytes = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    const loom_link_module_index_symbol_t* split_root =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("split_root"));
    const loom_link_module_index_symbol_t* config_dependency =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("config_dependency"));
    const loom_link_module_index_symbol_t* implementation_dependency =
        loom_link_module_index_lookup_private(
            index, indexed_module, IREE_SV("implementation_dependency"));
    ASSERT_NE(split_root, nullptr);
    ASSERT_NE(config_dependency, nullptr);
    ASSERT_NE(implementation_dependency, nullptr);
    ASSERT_EQ(split_root->dependencies.count, 2u);

    bool found_config_dependency = false;
    bool found_implementation_dependency = false;
    for (uint32_t i = 0; i < split_root->dependencies.count; ++i) {
      const uint32_t occurrence_index = split_root->dependencies.first + i;
      const uint32_t target =
          indexed_module->dependencies.values[occurrence_index];
      const uint8_t origin =
          indexed_module->dependencies
              .source_root_region_indices_plus_one[occurrence_index];
      const loom_symbol_interface_flags_t target_interfaces =
          indexed_module->dependencies.target_interfaces[occurrence_index];
      if (target == config_dependency->module_symbol_ordinal) {
        EXPECT_EQ(origin, 1u);
        EXPECT_EQ(target_interfaces, LOOM_SYMBOL_INTERFACE_RECORD);
        found_config_dependency = true;
      } else if (target == implementation_dependency->module_symbol_ordinal) {
        EXPECT_EQ(origin, 2u);
        EXPECT_EQ(target_interfaces, LOOM_SYMBOL_INTERFACE_RECORD);
        found_implementation_dependency = true;
      }
    }
    EXPECT_TRUE(found_config_dependency);
    EXPECT_TRUE(found_implementation_dependency);
  };

  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("library"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), module, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr bytecode_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      bytecode_index.get(),
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("library.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));
  verify_index(bytecode_index.get());
}

TEST_F(ModuleIndexTest, IndexesTestOnlySymbolRole) {
  loom_module_t* module = Parse(IREE_SV(R"(
check.case public @kernel_case {
  check.return
}

check.benchmark<@kernel_case>
check.benchmark<@kernel_case> @kernel_bench {}
)"));
  ASSERT_NE(module, nullptr);
  loom_link_module_index_add_options_t options = {
      /*.provider_name=*/IREE_SV("checks"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), module, &options,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_symbol_t* materialized_case =
      loom_link_module_index_lookup_global(materialized_index.get(),
                                           IREE_SV("kernel_case"));
  ASSERT_NE(materialized_case, nullptr);
  EXPECT_TRUE(iree_all_bits_set(materialized_case->flags,
                                LOOM_LINK_SYMBOL_FLAG_TEST_ONLY));
  EXPECT_TRUE(iree_all_bits_set(materialized_case->flags,
                                LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION));

  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("checks.loombc"), /*index_options=*/nullptr, &options,
      /*out_provider_ordinal=*/nullptr));

  EXPECT_EQ(loom_link_module_index_symbol_count(index.get()), 2u);
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  ASSERT_NE(check_case, nullptr);
  EXPECT_TRUE(
      iree_all_bits_set(check_case->flags, LOOM_LINK_SYMBOL_FLAG_TEST_ONLY));
  EXPECT_TRUE(iree_all_bits_set(check_case->flags,
                                LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION));

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* benchmark =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("kernel_bench"));
  ASSERT_NE(benchmark, nullptr);
  EXPECT_TRUE(
      iree_all_bits_set(benchmark->flags, LOOM_LINK_SYMBOL_FLAG_TEST_ONLY));
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
  EXPECT_EQ(loom_link_module_index_symbol_provider(index.get(), symbol),
            provider);
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
  EXPECT_NE(loom_link_module_index_symbol_provider(index.get(), first_helper),
            loom_link_module_index_symbol_provider(index.get(), second_helper));
  EXPECT_EQ(
      loom_link_module_index_lookup_global(index.get(), IREE_SV("helper")),
      nullptr);
}

TEST_F(ModuleIndexTest, OverlayRequiresLibraryOnlyBase) {
  loom_module_t* input = Parse(IREE_SV(R"(
func.def public @entry() {
  func.return
}
)"));
  IndexPtr base_index = CreateIndex();
  const loom_link_module_index_add_options_t input_options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      base_index.get(), input, &input_options,
      /*out_provider_ordinal=*/nullptr));

  loom_link_module_index_t* overlay = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_module_index_allocate_overlay(
          base_index.get(), &block_pool_, iree_allocator_system(), &overlay));
  EXPECT_EQ(overlay, nullptr);
}

TEST_F(ModuleIndexTest, OverlayCannotUseAnotherOverlayAsBase) {
  IndexPtr base_index = CreateIndex();
  IndexPtr first_overlay = CreateOverlay(base_index.get());

  loom_link_module_index_t* second_overlay = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_link_module_index_allocate_overlay(
                            first_overlay.get(), &block_pool_,
                            iree_allocator_system(), &second_overlay));
  EXPECT_EQ(second_overlay, nullptr);
}

TEST_F(ModuleIndexTest, OverlayAppendsInputWithoutCopyingBaseRecords) {
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* input = Parse(IREE_SV(R"(
func.decl public @entry(%x: i32) -> (i32)
)"));

  IndexPtr base_index = CreateIndex();
  const loom_link_module_index_add_options_t library_options = {
      /*.provider_name=*/IREE_SV("library"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      base_index.get(), library, &library_options,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_provider_t* base_provider =
      loom_link_module_index_provider_at(base_index.get(), 0);
  const loom_link_module_index_module_t* base_module =
      loom_link_module_index_module_at(base_index.get(), 0);
  const loom_link_module_index_symbol_t* base_entry =
      loom_link_module_index_lookup_global(base_index.get(), IREE_SV("entry"));
  ASSERT_NE(base_provider, nullptr);
  ASSERT_NE(base_module, nullptr);
  ASSERT_NE(base_entry, nullptr);

  IndexPtr overlay = CreateOverlay(base_index.get());
  EXPECT_EQ(loom_link_module_index_provider_at(overlay.get(), 0),
            base_provider);
  EXPECT_EQ(loom_link_module_index_module_at(overlay.get(), 0), base_module);
  EXPECT_EQ(loom_link_module_index_symbol_at(overlay.get(), 0), base_entry);
  EXPECT_EQ(loom_link_module_index_input_provider_count(overlay.get()), 0u);

  const loom_link_module_index_add_options_t input_options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      overlay.get(), input, &input_options,
      /*out_provider_ordinal=*/nullptr));
  EXPECT_EQ(loom_link_module_index_provider_count(overlay.get()), 2u);
  EXPECT_EQ(loom_link_module_index_module_count(overlay.get()), 2u);
  EXPECT_EQ(loom_link_module_index_symbol_count(overlay.get()), 2u);
  EXPECT_EQ(loom_link_module_index_input_provider_count(overlay.get()), 1u);

  const loom_link_module_index_symbol_t* selected =
      loom_link_module_index_lookup_global(overlay.get(), IREE_SV("entry"));
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->ordinal, 1u);
  const loom_link_module_index_provider_t* selected_provider =
      loom_link_module_index_symbol_provider(overlay.get(), selected);
  ASSERT_NE(selected_provider, nullptr);
  EXPECT_EQ(selected_provider->role, LOOM_LINK_PROVIDER_ROLE_INPUT);

  const loom_link_module_index_symbol_t* first =
      loom_link_module_index_lookup_name(overlay.get(), IREE_SV("entry"));
  EXPECT_EQ(first, base_entry);
  EXPECT_EQ(loom_link_module_index_next_same_name(overlay.get(), first),
            selected);
  EXPECT_EQ(loom_link_module_index_next_same_name(overlay.get(), selected),
            nullptr);
  EXPECT_EQ(
      loom_link_module_index_next_global_duplicate(overlay.get(), selected),
      base_entry);
  EXPECT_EQ(
      loom_link_module_index_next_global_duplicate(overlay.get(), base_entry),
      nullptr);

  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(overlay.get());
  ASSERT_EQ(input_exports.count, 1u);
  EXPECT_EQ(input_exports.values[0], selected->ordinal);
  EXPECT_EQ(loom_link_module_index_provider_count(base_index.get()), 1u);
  EXPECT_EQ(loom_link_module_index_input_provider_count(base_index.get()), 0u);
  EXPECT_EQ(
      loom_link_module_index_lookup_global(base_index.get(), IREE_SV("entry")),
      base_entry);
}

TEST_F(ModuleIndexTest, OverlayExtendsTemplateProviderEnumeration) {
  loom_module_t* library = Parse(IREE_SV(R"(
template.decl @demo.contract(%x: i32) -> (i32)

template.def<@demo.contract> @library_provider(%x: i32) -> (i32) {
  template.return %x : i32
}

template.decl @demo.empty(%x: i32) -> (i32)
)"));
  loom_module_t* input = Parse(IREE_SV(R"(
template.decl @demo.contract(%x: i32) -> (i32)

template.def<@demo.contract> @input_provider(%x: i32) -> (i32) {
  template.return %x : i32
}

template.decl @demo.empty(%x: i32) -> (i32)

template.def<@demo.empty> @first_empty_provider(%x: i32) -> (i32) {
  template.return %x : i32
}
)"));

  IndexPtr base_index = CreateIndex();
  const loom_link_module_index_add_options_t library_options = {
      /*.provider_name=*/IREE_SV("library"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      base_index.get(), library, &library_options,
      /*out_provider_ordinal=*/nullptr));
  ASSERT_EQ(loom_link_module_index_template_family_count(base_index.get()), 2u);
  const loom_link_module_index_template_family_t* base_family =
      loom_link_module_index_template_family_at(base_index.get(), 0);
  ASSERT_NE(base_family, nullptr);
  ASSERT_EQ(base_family->providers.count, 1u);
  const loom_link_module_index_symbol_t* library_provider =
      loom_link_module_index_symbol_at(
          base_index.get(), base_family->providers.first_symbol_ordinal);
  ASSERT_NE(library_provider, nullptr);

  IndexPtr overlay = CreateOverlay(base_index.get());
  const loom_link_module_index_add_options_t input_options = {
      /*.provider_name=*/IREE_SV("input"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      overlay.get(), input, &input_options,
      /*out_provider_ordinal=*/nullptr));

  ASSERT_EQ(loom_link_module_index_template_family_count(overlay.get()), 2u);
  const loom_link_module_index_template_family_t* family =
      loom_link_module_index_template_family_at(overlay.get(), 0);
  ASSERT_NE(family, nullptr);
  EXPECT_NE(family, base_family);
  EXPECT_EQ(family->providers.count, 2u);
  EXPECT_EQ(family->providers.first_symbol_ordinal, library_provider->ordinal);

  const loom_link_module_index_module_t* input_module =
      loom_link_module_index_module_at(overlay.get(), 1);
  ASSERT_NE(input_module, nullptr);
  const loom_link_module_index_symbol_t* input_provider =
      loom_link_module_index_lookup_private(overlay.get(), input_module,
                                            IREE_SV("input_provider"));
  ASSERT_NE(input_provider, nullptr);
  EXPECT_EQ(family->providers.last_symbol_ordinal, input_provider->ordinal);
  EXPECT_EQ(loom_link_module_index_next_template_provider(overlay.get(),
                                                          library_provider),
            input_provider);
  EXPECT_EQ(loom_link_module_index_next_template_provider(overlay.get(),
                                                          input_provider),
            nullptr);

  const loom_link_module_index_template_family_t* base_empty_family =
      loom_link_module_index_template_family_at(base_index.get(), 1);
  const loom_link_module_index_template_family_t* empty_family =
      loom_link_module_index_template_family_at(overlay.get(), 1);
  ASSERT_NE(base_empty_family, nullptr);
  ASSERT_NE(empty_family, nullptr);
  EXPECT_EQ(base_empty_family->providers.count, 0u);
  EXPECT_EQ(empty_family->providers.count, 1u);
  const loom_link_module_index_symbol_t* first_empty_provider =
      loom_link_module_index_lookup_private(overlay.get(), input_module,
                                            IREE_SV("first_empty_provider"));
  ASSERT_NE(first_empty_provider, nullptr);
  EXPECT_EQ(empty_family->providers.first_symbol_ordinal,
            first_empty_provider->ordinal);
  EXPECT_EQ(empty_family->providers.last_symbol_ordinal,
            first_empty_provider->ordinal);
  EXPECT_EQ(loom_link_module_index_next_template_provider(overlay.get(),
                                                          first_empty_provider),
            nullptr);

  EXPECT_EQ(loom_link_module_index_template_family_at(base_index.get(), 0),
            base_family);
  EXPECT_EQ(base_family->providers.count, 1u);
  EXPECT_EQ(base_family->providers.last_symbol_ordinal,
            library_provider->ordinal);
  EXPECT_EQ(base_empty_family->providers.count, 0u);
}

TEST_F(ModuleIndexTest, ImmutableBaseSupportsConcurrentRequestOverlays) {
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public @library_entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* input = Parse(IREE_SV(R"(
func.decl public @library_entry(%x: i32) -> (i32)
)"));
  const std::vector<uint8_t> input_bytes = WriteModule(input);

  IndexPtr base_index = CreateIndex();
  const loom_link_module_index_add_options_t library_options = {
      /*.provider_name=*/IREE_SV("library"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      base_index.get(), library, &library_options,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_symbol_t* base_symbol =
      loom_link_module_index_lookup_global(base_index.get(),
                                           IREE_SV("library_entry"));
  ASSERT_NE(base_symbol, nullptr);

  std::atomic<bool> failed = false;
  std::vector<std::thread> threads;
  for (int thread_ordinal = 0; thread_ordinal < 8; ++thread_ordinal) {
    threads.emplace_back([&]() {
      iree_arena_block_pool_t thread_block_pool;
      iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                       &thread_block_pool);
      const loom_link_module_index_add_options_t input_options = {
          /*.provider_name=*/IREE_SV("requester"),
          /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
      };
      for (int iteration = 0; iteration < 64 && !failed.load(); ++iteration) {
        loom_link_module_index_t* overlay = nullptr;
        iree_status_t status = loom_link_module_index_allocate_overlay(
            base_index.get(), &thread_block_pool, iree_allocator_system(),
            &overlay);
        if (iree_status_is_ok(status)) {
          status = loom_link_module_index_add_bytecode(
              overlay,
              iree_make_const_byte_span(input_bytes.data(), input_bytes.size()),
              IREE_SV("requester.loombc"), /*index_options=*/nullptr,
              &input_options, /*out_provider_ordinal=*/nullptr);
        }
        if (!iree_status_is_ok(status)) {
          failed.store(true);
          iree_status_free(status);
        } else {
          const loom_link_module_index_symbol_t* selected =
              loom_link_module_index_lookup_global(overlay,
                                                   IREE_SV("library_entry"));
          if (selected == nullptr || selected == base_symbol ||
              loom_link_module_index_next_global_duplicate(overlay, selected) !=
                  base_symbol) {
            failed.store(true);
          }
        }
        loom_link_module_index_free(overlay);
      }
      iree_arena_block_pool_deinitialize(&thread_block_pool);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_FALSE(failed.load());
  EXPECT_EQ(loom_link_module_index_provider_count(base_index.get()), 1u);
  EXPECT_EQ(loom_link_module_index_symbol_count(base_index.get()), 1u);
}

}  // namespace
}  // namespace loom
