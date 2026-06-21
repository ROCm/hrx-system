// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/kernel_library.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_const_byte_span_t ByteSpan(const char* value) {
  return iree_make_const_byte_span(value, strlen(value));
}

TEST(KernelLibraryTest, CreatesFromSourceFiles) {
  static const id4_pipeline_kernel_source_file_t kSourceFiles[] = {
      {
          // Source identifier that determines the module path.
          /*.source_identifier=*/IREE_SV("qwen3_vl/rmsnorm.loom"),
          // In-memory Loom source text.
          /*.source_contents=*/ByteSpan("module @rmsnorm {}"),
      },
      {
          // Source identifier that determines the module path.
          /*.source_identifier=*/IREE_SV("qwen3_vl/linear_bf16_f32.loom"),
          // In-memory Loom source text.
          /*.source_contents=*/ByteSpan("module @linear {}"),
      },
  };

  id4_pipeline_kernel_library_t* library = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_library_create_from_source_files(
      IREE_ARRAYSIZE(kSourceFiles), kSourceFiles, iree_allocator_system(),
      &library));
  EXPECT_EQ(id4_pipeline_kernel_library_module_count(library), 2u);

  const id4_pipeline_kernel_module_t* module = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_library_lookup(
      library, IREE_SV("qwen3_vl/rmsnorm"), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_TRUE(iree_string_view_equal(module->source_identifier,
                                     IREE_SV("qwen3_vl/rmsnorm.loom")));
  ASSERT_EQ(module->source_contents.data_length,
            kSourceFiles[0].source_contents.data_length);
  EXPECT_EQ(
      memcmp(module->source_contents.data, kSourceFiles[0].source_contents.data,
             module->source_contents.data_length),
      0);

  id4_pipeline_kernel_library_release(library);
}

TEST(KernelLibraryTest, RejectsSourceFileWithoutLoomSuffix) {
  id4_pipeline_kernel_source_file_t source_file = {
      // Source identifier missing the required .loom suffix.
      /*.source_identifier=*/IREE_SV("qwen3_vl/rmsnorm"),
      // In-memory Loom source text.
      /*.source_contents=*/ByteSpan("module @rmsnorm {}"),
  };

  id4_pipeline_kernel_library_t* library = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_kernel_library_create_from_source_files(
          1, &source_file, iree_allocator_system(), &library));
  EXPECT_EQ(library, nullptr);
}

TEST(KernelLibraryTest, RejectsDuplicateModulePaths) {
  static const id4_pipeline_kernel_source_file_t kSourceFiles[] = {
      {
          // Source identifier that determines the module path.
          /*.source_identifier=*/IREE_SV("qwen3_vl/rmsnorm.loom"),
          // In-memory Loom source text.
          /*.source_contents=*/ByteSpan("module @rmsnorm_a {}"),
      },
      {
          // Source identifier that maps to the same module path.
          /*.source_identifier=*/IREE_SV("qwen3_vl/rmsnorm.loom"),
          // In-memory Loom source text.
          /*.source_contents=*/ByteSpan("module @rmsnorm_b {}"),
      },
  };

  id4_pipeline_kernel_library_t* library = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        id4_pipeline_kernel_library_create_from_source_files(
                            IREE_ARRAYSIZE(kSourceFiles), kSourceFiles,
                            iree_allocator_system(), &library));
  EXPECT_EQ(library, nullptr);
}

}  // namespace
