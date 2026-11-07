// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/testing/source_generator.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/printer.h"
#include "loom/import/cxx/import.h"
#include "loom/ops/op_registry.h"

namespace loom::cxx_import::testing {
namespace {

constexpr std::array<loom_cxx_data_model_t, 3> kDataModels = {
    LOOM_CXX_DATA_MODEL_LP64,
    LOOM_CXX_DATA_MODEL_LLP64,
    LOOM_CXX_DATA_MODEL_ILP32,
};

class SourceGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    const std::string representative_source = GenerateSource({});
    for (size_t i = 0; i < kDataModels.size(); ++i) {
      loom_cxx_import_options_t options;
      loom_cxx_import_options_initialize(&options);
      options.data_model = kDataModels[i];
      IREE_ASSERT_OK(loom_cxx_preamble_create(
          iree_make_string_view(representative_source.data(),
                                representative_source.size()),
          IREE_SV("/test/generated.cxx"), &options, iree_allocator_system(),
          &preambles_[i]));
      ASSERT_NE(preambles_[i], nullptr);
    }
  }

  void TearDown() override {
    for (loom_cxx_preamble_t* preamble : preambles_) {
      loom_cxx_preamble_release(preamble);
    }
    loom_context_deinitialize(&context_);
  }

  iree_status_t ImportAndPrint(const std::string& source,
                               size_t data_model_index,
                               loom_cxx_fp_model_t fp_model,
                               std::string* out_text) {
    iree_arena_block_pool_t block_pool;
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool);
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    options.data_model = kDataModels[data_model_index];
    options.fp_model = fp_model;
    options.preamble = preambles_[data_model_index];
    loom_module_t* module = nullptr;
    iree_status_t status =
        loom_cxx_import(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("/test/generated.cxx"), &context_, &block_pool,
                        &options, iree_allocator_system(), &module);
    if (iree_status_is_ok(status) && !module) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "generated source was rejected");
    }
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    if (iree_status_is_ok(status)) {
      status = loom_text_print_module_to_builder(module, &builder,
                                                 LOOM_TEXT_PRINT_DEFAULT);
    }
    if (iree_status_is_ok(status)) {
      out_text->assign(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    }
    iree_string_builder_deinitialize(&builder);
    loom_module_free(module);
    iree_arena_block_pool_deinitialize(&block_pool);
    return status;
  }

  // Finalized dialect registry shared by imports in one test.
  loom_context_t context_ = {};
  // Immutable semantic prefixes for each supported source data model.
  std::array<loom_cxx_preamble_t*, kDataModels.size()> preambles_ = {};
};

TEST(SourceGenerationTest, IsDeterministicAndBounded) {
  for (size_t seed = 0; seed < 256; ++seed) {
    std::array<uint8_t, kMaximumMutationByteCount + 17> bytes;
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(seed * 37 + i * 31 + i * seed);
    }
    const std::string first = GenerateSource(bytes);
    const std::string second = GenerateSource(bytes);
    EXPECT_EQ(first, second);
    EXPECT_LE(first.size(), kMaximumGeneratedSourceByteCount);
    EXPECT_EQ(first, GenerateSource(std::span<const uint8_t>(bytes).first(
                         kMaximumMutationByteCount)));
  }
}

TEST(SourceGenerationTest, ContainsRepresentativeSourceFamilies) {
  const std::string source = GenerateSource({});
  EXPECT_NE(source.find("#include <loomcxx/numeric.h>"), std::string::npos);
  EXPECT_NE(source.find("#include <loomcxx/view.h>"), std::string::npos);
  EXPECT_NE(source.find("template <class T>"), std::string::npos);
  EXPECT_NE(source.find("loom::convert<"), std::string::npos);
  EXPECT_NE(source.find("loom::scalar::fmaf"), std::string::npos);
  EXPECT_NE(source.find("for (unsigned index"), std::string::npos);
  EXPECT_NE(source.find("[[loom::kernel"), std::string::npos);

  bool found_pointer_storage = false;
  bool found_buffer_view_storage = false;
  for (size_t seed = 0; seed < 256; ++seed) {
    std::array<uint8_t, kMaximumMutationByteCount> bytes;
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(seed * 37 + i * 19 + i * seed);
    }
    const std::string candidate = GenerateSource(bytes);
    found_pointer_storage |=
        candidate.find("input[index]") != std::string::npos;
    found_buffer_view_storage |=
        candidate.find("loom::buffer::view<") != std::string::npos;
  }
  EXPECT_TRUE(found_pointer_storage);
  EXPECT_TRUE(found_buffer_view_storage);
}

TEST_F(SourceGeneratorTest, ImportsAndPrintsDeterministically) {
  for (size_t seed = 0; seed < 64; ++seed) {
    std::array<uint8_t, kMaximumMutationByteCount> bytes;
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(seed * 37 + i * 19 + i * seed);
    }
    const std::string source = GenerateSource(bytes);
    SCOPED_TRACE(source);
    ASSERT_LE(source.size(), kMaximumGeneratedSourceByteCount);
    const size_t data_model_index = seed % kDataModels.size();
    const loom_cxx_fp_model_t fp_model =
        seed & 1 ? LOOM_CXX_FP_MODEL_FAST : LOOM_CXX_FP_MODEL_PRECISE;
    std::string first;
    std::string second;
    IREE_ASSERT_OK(ImportAndPrint(source, data_model_index, fp_model, &first));
    IREE_ASSERT_OK(ImportAndPrint(source, data_model_index, fp_model, &second));
    EXPECT_EQ(first, second);
  }
}

}  // namespace
}  // namespace loom::cxx_import::testing
