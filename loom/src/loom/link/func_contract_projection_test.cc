// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/func_contract_projection.h"

#include <memory>
#include <string_view>
#include <vector>

#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/testdata/func_contract_projection_testdata.h"
#include "loom/ops/op_registry.h"
#include "loom/rewrite/remap.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

struct ProjectionDeleter {
  void operator()(loom_link_func_contract_projection_t* projection) const {
    loom_link_func_contract_projection_free(projection);
  }
};
using ProjectionPtr =
    std::unique_ptr<loom_link_func_contract_projection_t, ProjectionDeleter>;

class LinkFuncContractProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &comparison_arena_);

    const iree_file_toc_t* files =
        loom_link_func_contract_projection_testdata_create();
    ASSERT_EQ(loom_link_func_contract_projection_testdata_size(), 2u);
    for (iree_host_size_t i = 0;
         i < loom_link_func_contract_projection_testdata_size(); ++i) {
      const std::string_view name(files[i].name);
      if (name == "input.loom") {
        input_source_ = iree_make_string_view(files[i].data, files[i].size);
      } else if (name == "library.loom") {
        library_source_ = iree_make_string_view(files[i].data, files[i].size);
      }
    }
    ASSERT_FALSE(iree_string_view_is_empty(input_source_));
    ASSERT_FALSE(iree_string_view_is_empty(library_source_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    iree_arena_deinitialize(&comparison_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source, iree_string_view_t filename) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    IREE_CHECK_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                  &options, &module));
    IREE_ASSERT(module != nullptr);
    modules_.push_back(module);
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
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(),
                                      /*out_buffer_length=*/nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  IndexPtr CreateMixedIndex() {
    loom_link_module_index_t* raw_index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &raw_index));
    IndexPtr index(raw_index);

    const loom_link_module_index_add_options_t input_options = {
        /*.provider_name=*/IREE_SV("input"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    IREE_CHECK_OK(loom_link_module_index_add_text(
        index.get(), input_source_, IREE_SV("input.loom"),
        /*parse_options=*/nullptr, &input_options,
        /*out_provider_ordinal=*/nullptr));

    loom_module_t* library = Parse(library_source_, IREE_SV("library.loom"));
    library_bytes_ = WriteModule(library);
    const loom_link_module_index_add_options_t library_options = {
        /*.provider_name=*/IREE_SV("library"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
    };
    IREE_CHECK_OK(loom_link_module_index_add_bytecode(
        index.get(),
        iree_make_const_byte_span(library_bytes_.data(), library_bytes_.size()),
        IREE_SV("library.loombc"), /*index_options=*/nullptr, &library_options,
        /*out_provider_ordinal=*/nullptr));
    return index;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  iree_arena_allocator_t comparison_arena_;
  std::vector<loom_module_t*> modules_;
  std::vector<uint8_t> library_bytes_;
  iree_string_view_t input_source_ = iree_string_view_empty();
  iree_string_view_t library_source_ = iree_string_view_empty();
};

TEST_F(LinkFuncContractProjectionTest,
       ComparesTextDeclarationWithBytecodeDefinitionWithoutBodies) {
  IndexPtr index = CreateMixedIndex();
  const loom_link_module_index_symbol_t* declaration =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("identity"));
  ASSERT_NE(declaration, nullptr);
  const loom_link_module_index_symbol_t* definition =
      loom_link_module_index_next_global_duplicate(index.get(), declaration);
  ASSERT_NE(definition, nullptr);

  loom_link_func_contract_projection_t* raw_projection = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_allocate(
      index.get(), &block_pool_, iree_allocator_system(), &raw_projection));
  ProjectionPtr projection(raw_projection);
  EXPECT_EQ(loom_link_func_contract_projection_module(projection.get())
                ->symbols.count,
            0u);
  const loom_link_func_contract_t* source_contract = nullptr;
  const loom_link_func_contract_t* selected_contract = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(
      projection.get(), declaration, &source_contract));
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(
      projection.get(), definition, &selected_contract));
  ASSERT_EQ(source_contract->module, selected_contract->module);
  EXPECT_EQ(source_contract->module->symbols.count, 2u);

  loom_ir_remap_t remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(
      source_contract->module,
      loom_link_func_contract_projection_module(projection.get()),
      &comparison_arena_, /*options=*/nullptr, &remap));
  loom_link_func_contract_mismatch_t mismatch = {};
  IREE_ASSERT_OK(loom_link_func_contract_check(
      source_contract, selected_contract, &remap, &mismatch));
  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE);
}

TEST_F(LinkFuncContractProjectionTest, PreservesProjectedTypeDifferences) {
  IndexPtr index = CreateMixedIndex();
  const loom_link_module_index_symbol_t* declaration =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("identity"));
  const loom_link_module_index_symbol_t* wrong_type =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("wrong_type"));
  ASSERT_NE(declaration, nullptr);
  ASSERT_NE(wrong_type, nullptr);

  loom_link_func_contract_projection_t* raw_projection = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_allocate(
      index.get(), &block_pool_, iree_allocator_system(), &raw_projection));
  ProjectionPtr projection(raw_projection);
  const loom_link_func_contract_t* source_contract = nullptr;
  const loom_link_func_contract_t* selected_contract = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(
      projection.get(), declaration, &source_contract));
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(
      projection.get(), wrong_type, &selected_contract));

  loom_ir_remap_t remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(
      source_contract->module,
      loom_link_func_contract_projection_module(projection.get()),
      &comparison_arena_, /*options=*/nullptr, &remap));
  loom_link_func_contract_mismatch_t mismatch = {};
  IREE_ASSERT_OK(loom_link_func_contract_check(
      source_contract, selected_contract, &remap, &mismatch));
  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name, IREE_SV("arg")));
  EXPECT_EQ(mismatch.detail.type_ordinal, 0u);
}

TEST_F(LinkFuncContractProjectionTest, CachesProjectedContractViews) {
  IndexPtr index = CreateMixedIndex();
  const loom_link_module_index_symbol_t* declaration =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("identity"));
  ASSERT_NE(declaration, nullptr);

  loom_link_func_contract_projection_t* raw_projection = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_allocate(
      index.get(), &block_pool_, iree_allocator_system(), &raw_projection));
  ProjectionPtr projection(raw_projection);
  const loom_link_func_contract_t* first = nullptr;
  const loom_link_func_contract_t* second = nullptr;
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(projection.get(),
                                                         declaration, &first));
  IREE_ASSERT_OK(loom_link_func_contract_projection_load(projection.get(),
                                                         declaration, &second));
  EXPECT_EQ(first, second);
}

}  // namespace
}  // namespace loom
