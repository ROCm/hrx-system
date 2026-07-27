// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_dce.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t
    kSymbolDCETestDescriptorSetProviders[] = {
        loom_test_low_core_descriptor_set,
};

class SymbolDCETest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    low_descriptor_registry_ = {};
    low_descriptor_registry_.descriptor_set_providers =
        kSymbolDCETestDescriptorSetProviders;
    low_descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kSymbolDCETestDescriptorSetProviders);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &options.low_asm_environment);
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("symbol_dce_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  std::string Print(const loom_module_t* module) {
    loom_text_low_asm_environment_t low_asm_environment = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &low_asm_environment);
    loom_text_print_options_t options = {
        /*.flags=*/LOOM_TEXT_PRINT_DEFAULT,
        /*.low_asm_environment=*/low_asm_environment,
    };
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder_with_options(
        module, &builder, &options));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  void RunSymbolDCE(loom_module_t* module, int64_t expected_symbols_eliminated,
                    int64_t expected_functions_eliminated) {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    const loom_pass_info_t* pass_info = loom_symbol_dce_pass_info();
    const loom_pass_statistic_layout_t* statistic_layout =
        pass_info->statistic_layout;
    std::vector<uint8_t> statistic_storage(statistic_layout->storage_size, 0);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.statistic_storage = statistic_storage.data();
    IREE_EXPECT_OK(loom_symbol_dce_run(&pass, module));
    const uint8_t* storage = statistic_storage.data();
    const int64_t symbols_eliminated = *reinterpret_cast<const int64_t*>(
        storage + statistic_layout->fields[0].offset);
    const int64_t functions_eliminated = *reinterpret_cast<const int64_t*>(
        storage + statistic_layout->fields[1].offset);
    EXPECT_EQ(symbols_eliminated, expected_symbols_eliminated);
    EXPECT_EQ(functions_eliminated, expected_functions_eliminated);
    iree_arena_deinitialize(&pass_arena);
  }

  iree_status_t WriteModule(const loom_module_t* module,
                            std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    iree_status_t status =
        loom_bytecode_write_module(module, stream, NULL, &block_pool_);

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

  loom_module_t* ReadModule(const std::vector<uint8_t>& bytes) {
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.verify_module=*/true,
        /*.verify_max_errors=*/20,
    };
    loom_bytecode_read_result_t result = {0};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("symbol_dce_test.loombc"), &context_, &block_pool_, &options,
        &result, &module, iree_allocator_system()));
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_NE(module, nullptr);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t low_descriptor_registry_;
};

TEST_F(SymbolDCETest, PrunedLowIslandRoundTripsThroughBytecode) {
  const char* source = R"(
test.target<low_core> @test_target

func.decl @add_i32(%lhs: i32, %rhs: i32) -> (i32)

func.def public @entry(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = low.invoke @live_add(%lhs, %rhs) : (i32, i32) -> (i32)
  func.return %sum : i32
}

func.def @dead_wrapper(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = low.invoke @dead_add(%lhs, %rhs) : (i32, i32) -> (i32)
  func.return %sum : i32
}

low.func.decl target(@test_target) @live_add(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>)

low.func.decl target(@test_target) @dead_add(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>)
)";

  ModulePtr module(Parse(iree_make_cstring_view(source)));
  ASSERT_NE(module.get(), nullptr);

  RunSymbolDCE(module.get(), 3, 3);
  std::string pruned_text = Print(module.get());
  EXPECT_NE(pruned_text.find("@live_add"), std::string::npos);
  EXPECT_EQ(pruned_text.find("low.abi"), std::string::npos);
  EXPECT_EQ(pruned_text.find("add_i32"), std::string::npos);
  EXPECT_EQ(pruned_text.find("dead_add"), std::string::npos);
  EXPECT_EQ(pruned_text.find("dead_wrapper"), std::string::npos);

  std::vector<uint8_t> bytes;
  IREE_ASSERT_OK(WriteModule(module.get(), &bytes));
  ModulePtr read_module(ReadModule(bytes));
  ASSERT_NE(read_module.get(), nullptr);
  EXPECT_EQ(pruned_text, Print(read_module.get()));
}

TEST_F(SymbolDCETest, FunctionExportRootsPrivateEntryAndClosure) {
  const char* source = R"(
test.target<low_core> @test_target

func.def target(@test_target) abi(object_function) export("entry") @entry() {
  func.call @helper() : ()
  func.return
}

func.def target(@test_target) abi(object_function) @helper() {
  func.return
}

func.def target(@test_target) abi(object_function) @dead() {
  func.return
}
)";

  ModulePtr module(Parse(iree_make_cstring_view(source)));
  ASSERT_NE(module.get(), nullptr);

  RunSymbolDCE(module.get(), 1, 1);
  std::string pruned_text = Print(module.get());
  EXPECT_NE(pruned_text.find("@entry"), std::string::npos);
  EXPECT_NE(pruned_text.find("@helper"), std::string::npos);
  EXPECT_EQ(pruned_text.find("@dead"), std::string::npos);
}

TEST_F(SymbolDCETest, ConfigSymbolsAreReachableThroughReads) {
  const char* source = R"(
config.decl @live_config : index
config.decl @dead_config : index

func.def public @entry() -> (index) {
  %value = config.get @live_config : index
  func.return %value : index
}
)";

  ModulePtr module(Parse(iree_make_cstring_view(source)));
  ASSERT_NE(module.get(), nullptr);

  RunSymbolDCE(module.get(), 1, 0);
  std::string pruned_text = Print(module.get());
  EXPECT_NE(pruned_text.find("config.decl @live_config"), std::string::npos);
  EXPECT_EQ(pruned_text.find("config.decl @dead_config"), std::string::npos);
  EXPECT_NE(pruned_text.find("config.get @live_config"), std::string::npos);
}

}  // namespace
}  // namespace loom
