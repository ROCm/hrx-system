// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/inputs.h"

#include <cstring>
#include <fstream>
#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, input_policy, "", "Source caller of weight input capture.");

namespace {

class WeightInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The metadata order and tensor extents differ across sources with the
    // same key. The policy receives indexed metadata, never an inferred shape.
    WriteCheckpoint(first_.path(),
                    R"({"dtype":"F32","shape":[4],"data_offsets":[0,16]})", 16);
    WriteCheckpoint(second_.path(),
                    R"({"shape":[8],"dtype":"BF16","data_offsets":[0,16]})",
                    16);
    const loom_model_file_t paths[] = {
        {iree_make_cstring_view(first_.path().c_str()), {}},
        {iree_make_cstring_view(second_.path().c_str()),
         IREE_SV("safetensors")}};
    const loom_model_file_list_t checkpoints[] = {{1, &paths[0]},
                                                  {1, &paths[1]}};
    IREE_ASSERT_OK(loom_serve_parameter_sources_create(
        nullptr, &sources_, 2, checkpoints, allocator_, &catalog_));
    const uint32_t ordinals[] = {0, 1};
    const loom_serve_preparation_parameter_t parameter = {
        0, 2, checkpoints, IREE_SV("unused.loom"), ordinals};
    IREE_ASSERT_OK(loom_serve_weight_inputs_allocate(&parameter, catalog_,
                                                     allocator_, &inputs_));
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator_, &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(allocator_), &target_environment_)));
    IREE_ASSERT_OK(loom_serve_weight_inputs_module_create(
        inputs_, environment_, allocator_, &module_));
    const iree_string_view_t roots[] = {IREE_SV("capture"),
                                        IREE_SV("capture_buffer")};
    IREE_ASSERT_OK(loom_serve_program_allocate(
        target_environment_, environment_, &sources_,
        iree_make_cstring_view(FLAG_input_policy), IREE_ARRAYSIZE(roots), roots,
        {&module_, 1}, &compilation_, allocator_, &program_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program_), IREE_SV("model"),
        IREE_SV("capture"), &function_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program_), IREE_SV("model"),
        IREE_SV("capture_buffer"), &buffer_function_));
  }

  void TearDown() override {
    loom_serve_program_free(program_);
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
    loomc_target_environment_release(target_environment_);
    loom_serve_weight_inputs_free(inputs_);
    loom_serve_parameter_sources_release(catalog_);
    if (first_.Exists()) {
      EXPECT_TRUE(first_.Remove());
    }
    if (second_.Exists()) {
      EXPECT_TRUE(second_.Remove());
    }
  }

  void WriteCheckpoint(const std::string& path, const char* metadata,
                       size_t byte_length) {
    const std::string header = std::string("{\"same\":") + metadata + "}";
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (size_t i = 0; i < byte_length; ++i) {
      file.put(0);
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  iree_status_t Invoke(int32_t source, int64_t offset, int64_t length,
                       int64_t alignment, const char* key = "same") {
    iree_vm_buffer_t* key_buffer = nullptr;
    if (key) {
      IREE_RETURN_IF_ERROR(
          iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                               iree_make_const_byte_span(key, std::strlen(key)),
                               1, allocator_, &key_buffer));
    }
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(source),
        iree_vm_buffer_variant_from_ptr_move(&types_, &key_buffer),
        iree_vm_variant_from_i64(offset), iree_vm_variant_from_i64(length),
        iree_vm_variant_from_i64(alignment)};
    iree_vm_variant_t results[3] = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), function_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    if (iree_status_is_ok(status)) {
      iree_vm_buffer_t* metadata = nullptr;
      status = iree_vm_buffer_ptr_from_variant_borrowed(&types_, results[0],
                                                        &metadata);
      if (iree_status_is_ok(status)) {
        iree_const_byte_span_t bytes = iree_const_byte_span_empty();
        status = iree_vm_buffer_map_read(
            metadata, 0, iree_vm_buffer_length(metadata), &bytes);
        if (iree_status_is_ok(status)) {
          metadata_.assign(reinterpret_cast<const char*>(bytes.data),
                           bytes.data_length);
        }
      }
      if (iree_status_is_ok(status)) {
        status = iree_vm_i32_from_variant(results[1], &count_);
      }
      if (iree_status_is_ok(status)) {
        status = iree_vm_i64_from_variant(results[2], &offset_);
      }
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    return status;
  }

  iree_status_t InvokeBuffer(iree_vm_buffer_t* buffer, int64_t alignment) {
    iree_vm_buffer_retain(buffer);
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
        iree_vm_variant_from_i64(alignment)};
    iree_vm_variant_t results[2] = {};
    auto status = iree_vm_invoke(loom_serve_program_invocation(program_),
                                 buffer_function_,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results));
    if (iree_status_is_ok(status)) {
      status = iree_vm_i64_from_variant(results[0], &offset_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(results[1], &count_);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    return status;
  }

  // Host allocation for the actual VM and file indexes.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Explicit server-local namespace for checkpoint and source fixtures.
  const loom_serve_sources_t sources_ = {.base_directory = IREE_SV(".")};
  // Input checkpoints, removed after index/file ownership has retired.
  iree::testing::TempFilePath first_{"weight_input_first", ".safetensors"};
  // Independent source with the same tensor name and different metadata.
  iree::testing::TempFilePath second_{"weight_input_second", ".data"};
  // Owned indexes and captured ranges, outliving every borrowed VM buffer.
  loom_serve_weight_inputs_t* inputs_ = nullptr;
  // Shared immutable source catalog, independent of the selected input recipe.
  loom_serve_parameter_sources_t* catalog_ = nullptr;
  // Reference environment enclosing both native and source modules.
  iree_vm_environment_t* environment_ = nullptr;
  // Resolved canonical reference types used for arguments and results.
  iree_vm_ref_types_t types_ = {};
  // VM grammar used to compile the real source caller.
  loomc_target_environment_t* target_environment_ = nullptr;
  // Owned native capability linked by the source program.
  iree_vm_module_t* module_ = nullptr;
  // Owned source policy and its invocation storage.
  loom_serve_program_t* program_ = nullptr;
  // Borrowed export resolved once from the linked source program.
  iree_vm_function_t function_ = {};
  // Source wrapper capturing host bytes and querying visible checkpoint count.
  iree_vm_function_t buffer_function_ = {};
  // Copied compiler diagnostics retained independently of source ownership.
  loom_serve_compilation_report_t compilation_ = {};
  // Last successful source-visible metadata copy.
  std::string metadata_;
  // Last successful source-visible checkpoint count.
  int32_t count_ = 0;
  // Last successful source-visible staging offset.
  int64_t offset_ = 0;
};

TEST_F(WeightInputsTest, SeparateDomainsCaptureAlignedRangesAndRetainMetadata) {
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(0, 4, 4, 4));
  EXPECT_EQ(count_, 2);
  EXPECT_EQ(offset_, 0);
  EXPECT_NE(metadata_.find("\"dtype\":\"F32\""), std::string::npos);
  IREE_ASSERT_OK(Invoke(1, 8, 8, 16));
  EXPECT_EQ(offset_, 16);
  EXPECT_NE(metadata_.find("\"dtype\":\"BF16\""), std::string::npos);
  const auto recipe = loom_serve_weight_inputs_module_take(module_);
  ASSERT_EQ(recipe.count, 2u);
  EXPECT_EQ(recipe.byte_length, 24u);
  EXPECT_EQ(loom_serve_weight_inputs_staging_length(inputs_), 32u);
  EXPECT_EQ(loom_serve_weight_inputs_staging_alignment(inputs_), 16u);
  const auto* first = loom_serve_weight_inputs_at(inputs_, recipe.first);
  const auto* second = loom_serve_weight_inputs_at(inputs_, recipe.first + 1);
  EXPECT_EQ(first->source, 0u);
  EXPECT_EQ(first->span.parameter_offset, 4u);
  EXPECT_EQ(first->span.length, 4u);
  EXPECT_EQ(second->source, 1u);
  EXPECT_EQ(second->span.parameter_offset, 8u);
  EXPECT_EQ(second->span.buffer_offset, 16u);
  EXPECT_EQ(second->span.length, 8u);
  EXPECT_NE(first->entry, second->entry);
  EXPECT_TRUE(iree_string_view_equal(first->entry->key, second->entry->key));
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(1, 0, 2, 2));
  EXPECT_EQ(offset_, 0);
  const auto next = loom_serve_weight_inputs_module_take(module_);
  EXPECT_EQ(next.first, recipe.first + recipe.count);
  EXPECT_EQ(next.count, 1u);
  EXPECT_EQ(next.byte_length, 2u);
  EXPECT_EQ(loom_serve_weight_inputs_staging_length(inputs_), 32u);
}

TEST_F(WeightInputsTest, InvalidSourceKeysRangesAndAlignmentsRejectAtCapture) {
  const struct {
    // Requested independent source ordinal.
    int32_t source;
    // Source-relative byte offset.
    int64_t offset;
    // Requested nonempty byte count.
    int64_t length;
    // Requested power-of-two staging alignment.
    int64_t alignment;
    // Source key, including missing and null cases.
    const char* key;
    // Expected external-input diagnostic class.
    iree_status_code_t code;
  } cases[] = {
      {-1, 0, 4, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {2, 0, 4, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, -1, 4, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, 0, 0, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, 0, -1, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, 15, 2, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, INT64_MAX, 4, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, 0, INT64_MAX, 4, "same", IREE_STATUS_OUT_OF_RANGE},
      {0, 0, 4, 0, "same", IREE_STATUS_INVALID_ARGUMENT},
      {0, 0, 4, -1, "same", IREE_STATUS_INVALID_ARGUMENT},
      {0, 0, 4, 3, "same", IREE_STATUS_INVALID_ARGUMENT},
      {0, 0, 4, 4, "missing", IREE_STATUS_NOT_FOUND},
      {0, 0, 4, 4, nullptr, IREE_STATUS_INVALID_ARGUMENT},
  };
  for (const auto& test : cases) {
    loom_serve_weight_inputs_module_begin(module_);
    IREE_EXPECT_STATUS_IS(
        test.code, Invoke(test.source, test.offset, test.length, test.alignment,
                          test.key));
    EXPECT_EQ(loom_serve_weight_inputs_module_take(module_).count, 0u);
  }
}

TEST_F(WeightInputsTest, DirectReadsValidateSourceZeroWithoutCapturingStaging) {
  for (const uint64_t length : {0u, 8u, 16u}) {
    IREE_ASSERT_OK(loom_serve_weight_inputs_validate_direct(
        inputs_, IREE_SV("same"), length));
  }
  iree::Status status(
      loom_serve_weight_inputs_validate_direct(inputs_, IREE_SV("same"), 17));
  EXPECT_THAT(status,
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("weight 'same'"));
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("requires 17 bytes"));
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("entry has 16"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_weight_inputs_validate_direct(
                            inputs_, IREE_SV("same"), UINT64_MAX));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_weight_inputs_validate_direct(inputs_, IREE_SV("missing"), 8));
  EXPECT_EQ(loom_serve_weight_inputs_staging_length(inputs_), 0u);
}

TEST_F(WeightInputsTest, AlignedExtentCannotOverflowTheAddressableRecipe) {
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(0, 0, 4, 4));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Invoke(0, 0, 4, INT64_C(1) << 62));
  const auto recipe = loom_serve_weight_inputs_module_take(module_);
  EXPECT_EQ(recipe.count, 1u);
  EXPECT_EQ(recipe.byte_length, 4u);
}

TEST_F(WeightInputsTest, CapabilityIsUnavailableOutsidePolicyCapture) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Invoke(0, 0, 4, 4));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        InvokeBuffer(nullptr, 1));
}

TEST_F(WeightInputsTest, BufferSnapshotsSurviveMutationAndSourceVmTeardown) {
  const uint8_t first_bytes[] = {1, 3, 5, 7};
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(first_bytes, sizeof(first_bytes)), 1,
      allocator_, &buffer));
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(0, 0, 4, 4));
  IREE_ASSERT_OK(InvokeBuffer(buffer, 16));
  EXPECT_EQ(offset_, 16);
  EXPECT_EQ(count_, 2);
  const uint8_t second_bytes[] = {2, 4, 6, 8};
  std::memcpy(iree_vm_buffer_data(buffer), second_bytes, sizeof(second_bytes));
  IREE_ASSERT_OK(InvokeBuffer(buffer, 32));
  EXPECT_EQ(offset_, 32);
  EXPECT_EQ(count_, 2);
  std::memset(iree_vm_buffer_data(buffer), 0, sizeof(second_bytes));
  iree_vm_buffer_release(buffer);
  // The private provider must not become a policy-addressable checkpoint.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Invoke(2, 0, 4, 4));
  const auto recipe = loom_serve_weight_inputs_module_take(module_);
  ASSERT_EQ(recipe.count, 3u);
  EXPECT_EQ(recipe.byte_length, 36u);
  EXPECT_EQ(loom_serve_weight_inputs_staging_length(inputs_), 64u);
  EXPECT_EQ(loom_serve_weight_inputs_source_count(inputs_), 3u);
  loom_serve_program_free(program_);
  program_ = nullptr;
  iree_vm_module_release(module_);
  module_ = nullptr;
  iree_vm_environment_free(environment_);
  environment_ = nullptr;
  for (uint32_t i = 0; i < 2; ++i) {
    const auto* input =
        loom_serve_weight_inputs_at(inputs_, recipe.first + 1 + i);
    EXPECT_EQ(input->source, 2u);
    EXPECT_EQ(input->span.parameter_offset, 0u);
    EXPECT_EQ(input->span.buffer_offset, i ? 32u : 16u);
    EXPECT_EQ(input->span.length, 4u);
    auto* handle = input->entry->storage.file.handle;
    EXPECT_EQ(iree_io_file_handle_access(handle), IREE_IO_FILE_ACCESS_READ);
    ASSERT_EQ(iree_io_file_handle_type(handle),
              IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
    const auto bytes = iree_io_file_handle_value(handle).host_allocation;
    ASSERT_EQ(bytes.data_length, 4u);
    EXPECT_EQ(std::memcmp(bytes.data, i ? second_bytes : first_bytes, 4), 0);
  }
}

TEST_F(WeightInputsTest, InvalidBufferCapturesLeaveTheRecipeUnchanged) {
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(0, 0, 4, 4));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, InvokeBuffer(nullptr, 1));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(0, 1, allocator_, &buffer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, InvokeBuffer(buffer, 1));
  iree_vm_buffer_release(buffer);
  const uint8_t bytes[] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_const_byte_span(bytes, sizeof(bytes)), 1, allocator_, &buffer));
  for (int64_t alignment : {INT64_C(0), INT64_C(-1), INT64_C(3)}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          InvokeBuffer(buffer, alignment));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        InvokeBuffer(buffer, INT64_C(1) << 62));
  iree_vm_buffer_release(buffer);
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(bytes, sizeof(bytes)), 1, allocator_, &buffer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, InvokeBuffer(buffer, 1));
  iree_vm_buffer_release(buffer);
  EXPECT_EQ(loom_serve_weight_inputs_module_take(module_).count, 1u);
  EXPECT_EQ(loom_serve_weight_inputs_source_count(inputs_), 2u);
}

TEST_F(WeightInputsTest,
       RecipesRetainSelectedSourcesAndMapPolicyLocalOrdinals) {
  const uint32_t reversed[] = {1, 0};
  const loom_serve_preparation_parameter_t parameter = {
      0, 2, nullptr, IREE_SV("unused.loom"), reversed};
  loom_serve_weight_inputs_t* other = nullptr;
  IREE_ASSERT_OK(loom_serve_weight_inputs_allocate(&parameter, catalog_,
                                                   allocator_, &other));
  EXPECT_EQ(loom_serve_weight_inputs_provider(other, 0),
            loom_serve_weight_inputs_provider(inputs_, 1));
  EXPECT_EQ(loom_serve_weight_inputs_provider(other, 1),
            loom_serve_weight_inputs_provider(inputs_, 0));
  loom_serve_parameter_sources_release(catalog_);
  catalog_ = nullptr;
  loom_serve_weight_inputs_free(other);
  loom_serve_weight_inputs_module_begin(module_);
  IREE_ASSERT_OK(Invoke(1, 0, 16, 16));
  EXPECT_NE(metadata_.find("\"dtype\":\"BF16\""), std::string::npos);
  EXPECT_EQ(loom_serve_weight_inputs_module_take(module_).count, 1u);
}

}  // namespace
