// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/media/image_request.h"
#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/image/output.h"
#include "loom/serve/testing/service_application.h"

namespace {

constexpr loom_serve_media_image_limits_t kLimits = {
    .compressed_bytes = 2048,
    .width = 64,
    .height = 64,
    .pixels = 4096,
};

class ProjectorFixtureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto application_options =
        loom_serve_test_application_options_default();
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &application_options, host_allocator_, &application));
    auto* device = loom_serve_test_application_device_at(application, 0);
    const char* source_path = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source_path, nullptr);
    directory_ =
        std::filesystem::path(source_path).parent_path().generic_string();
    const loom_serve_finite_model_options_t options = {
        .sources =
            {
                .base_directory = iree_make_cstring_view(directory_.c_str()),
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = 1,
                .values = &provider_,
            },
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    // The application thread may join cold loading; shared workers never wait.
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(device, &options,
                                                    host_allocator_, &model_));
    EXPECT_EQ(loom_serve_finite_model_tokenizer(model_), nullptr);
    EXPECT_EQ(loom_serve_finite_model_batch_limit(model_), 64u);
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_finite_model_free(model_));
    ReleaseRequest();
    loom_serve_finite_result_deinitialize(&result_);
    loom_serve_finite_result_deinitialize(&retained_);
  }

  void ReleaseRequest() {
    iree_vm_buffer_release(prepared_.bytes);
    prepared_ = {};
    iree_vm_buffer_release(request_);
    request_ = nullptr;
  }

  void MakeRequest(uint32_t rows, size_t metadata_length,
                   const std::array<uint8_t, 3>& color) {
    ReleaseRequest();
    // Identical geometry and metadata can carry different actual pixels.
    std::array<float, 3> pixels;
    for (size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = static_cast<float>(color[i]) * (2.0f / 255.0f) - 1.0f;
    }
    iree_byte_span_t png;
    IREE_ASSERT_OK(loom_serve_image_encode_rgb_f32_png(
        1, 1, iree_make_const_byte_span(pixels.data(), sizeof(pixels)),
        host_allocator_, &png));
    std::string encoded(iree_base64_encoded_size(png.data_length), '\0');
    size_t encoded_length = 0;
    iree_status_t status = iree_base64_encode(
        iree_make_const_byte_span(png.data, png.data_length),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &encoded_length);
    iree_allocator_free(host_allocator_, png.data);
    IREE_ASSERT_OK(status);
    encoded.resize(encoded_length);
    const std::string url = "data:image/png;base64," + encoded;
    const iree_string_view_t url_view =
        iree_make_string_view(url.data(), url.size());
    loom_serve_media_image_info_t info;
    IREE_ASSERT_OK(loom_serve_media_image_inspect(url_view, &kLimits,
                                                  host_allocator_, &info));
    const std::vector<uint8_t> metadata(metadata_length, 0x7b);
    IREE_ASSERT_OK(loom_serve_media_image_request_create(
        url_view, &kLimits, &info,
        iree_make_const_byte_span(metadata.data(), metadata.size()), rows,
        host_allocator_, &request_));
    iree_const_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        request_, 0, iree_vm_buffer_length(request_), &bytes));
    EXPECT_EQ(
        iree_unaligned_load_le_u32(
            bytes.data + 4 * LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET),
        (64 + metadata_length + 15) & ~size_t{15});
  }

  void Prepare() {
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_request(model_, request_, &prepared_));
  }

  void Invoke(iree_host_size_t rows, loom_serve_finite_result_t* result) {
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_batch(model_, 1, &prepared_, &rows));
    // Batch ownership is independent even when its payload aliases the decoded
    // request. Dropping both caller references makes that lifetime observable.
    ReleaseRequest();
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_residency_try_acquire(
        loom_serve_finite_model_residency(model_), &admitted));
    ASSERT_TRUE(admitted);
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(loom_serve_finite_model_submit(model_, &frontier));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    loom_serve_finite_model_complete(model_);
    loom_serve_finite_model_take_result(model_, result);
  }

  void ExpectRows(const loom_serve_finite_result_t& result, uint32_t rows,
                  const std::array<uint8_t, 3>& color) {
    ASSERT_EQ(result.request_count, 1u);
    EXPECT_EQ(result.feedback.data_length, rows * 16u);
    EXPECT_EQ(iree_unaligned_load_le_u64(result.records.data), rows * 16u);
    EXPECT_EQ(loom_serve_finite_result_input_tokens(&result, 0), 0u);
    for (uint32_t row = 0; row < rows; ++row) {
      SCOPED_TRACE(row);
      const auto bytes = loom_serve_finite_result_row(&result, 0, row);
      ASSERT_EQ(bytes.data_length, 16u);
      EXPECT_EQ(bytes.data, result.feedback.data + row * 16u);
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data), color[0] + row);
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 4), color[1]);
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 8), color[2]);
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 12), row);
    }
  }

  // Allocator owning model, request, and detached result storage.
  iree_allocator_t host_allocator_ = iree_allocator_system();
  // Runfiles package path backing borrowed source locators.
  std::string directory_;
  // Direct numerical provider in the source catalog.
  iree_string_view_t provider_ = IREE_SV("project.loom");
  // Owned finite source model under test.
  loom_serve_finite_model_t* model_ = nullptr;
  // Owned decoded image request, retained through batch preparation.
  iree_vm_buffer_t* request_ = nullptr;
  // Owned source-prepared bytes retained through batch preparation.
  loom_serve_finite_prepared_t prepared_ = {};
  // Owned newest completed result.
  loom_serve_finite_result_t result_ = {};
  // Owned earlier result, independent of subsequent model invocations.
  loom_serve_finite_result_t retained_ = {};
};

TEST_F(ProjectorFixtureTest, ObservesPixelsAtVariableOffsetsAndExactRowCounts) {
  const std::array<uint8_t, 3> colors[] = {{255, 0, 17}, {3, 127, 231}};
  for (const auto& color : colors) {
    for (const auto& shape :
         {std::array<uint32_t, 2>{1, 0}, {5, 3}, {64, 33}}) {
      SCOPED_TRACE(shape[0]);
      SCOPED_TRACE(shape[1]);
      ASSERT_NO_FATAL_FAILURE(MakeRequest(shape[0], shape[1], color));
      ASSERT_NO_FATAL_FAILURE(Prepare());
      EXPECT_EQ(prepared_.units, shape[0]);
      ASSERT_NO_FATAL_FAILURE(Invoke(shape[0], &result_));
      ASSERT_NO_FATAL_FAILURE(ExpectRows(result_, shape[0], color));
      loom_serve_finite_result_deinitialize(&result_);
    }
  }
}

TEST_F(ProjectorFixtureTest, DetachedRowsSurviveAnotherImageAndModelRemoval) {
  const std::array<uint8_t, 3> first = {255, 0, 17};
  const std::array<uint8_t, 3> second = {3, 127, 231};
  ASSERT_NO_FATAL_FAILURE(MakeRequest(3, 3, first));
  ASSERT_NO_FATAL_FAILURE(Prepare());
  ASSERT_NO_FATAL_FAILURE(Invoke(3, &retained_));
  ASSERT_NO_FATAL_FAILURE(MakeRequest(2, 33, second));
  ASSERT_NO_FATAL_FAILURE(Prepare());
  ASSERT_NO_FATAL_FAILURE(Invoke(2, &result_));
  IREE_ASSERT_OK(loom_serve_finite_model_free(model_));
  model_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(ExpectRows(retained_, 3, first));
  ASSERT_NO_FATAL_FAILURE(ExpectRows(result_, 2, second));
}

TEST_F(ProjectorFixtureTest, RejectsUnsupportedRowsAndThenServesValidImage) {
  const std::array<uint8_t, 3> color = {255, 0, 17};
  for (const uint32_t rows : {0u, 65u}) {
    ASSERT_NO_FATAL_FAILURE(MakeRequest(rows, 0, color));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_finite_model_prepare_request(model_, request_, &prepared_));
    EXPECT_EQ(prepared_.bytes, nullptr);
  }
  ASSERT_NO_FATAL_FAILURE(MakeRequest(4, 3, color));
  ASSERT_NO_FATAL_FAILURE(Prepare());
  ASSERT_NO_FATAL_FAILURE(Invoke(4, &result_));
  ASSERT_NO_FATAL_FAILURE(ExpectRows(result_, 4, color));
}

TEST_F(ProjectorFixtureTest, RejectsMultipleImagesAndThenServesOne) {
  const std::array<uint8_t, 3> color = {3, 127, 231};
  ASSERT_NO_FATAL_FAILURE(MakeRequest(3, 3, color));
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const loom_serve_finite_prepared_t requests[] = {prepared_, prepared_};
  const iree_host_size_t rows[] = {3, 3};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_finite_model_prepare_batch(model_, 2, requests, rows));
  ASSERT_NO_FATAL_FAILURE(Invoke(3, &result_));
  ASSERT_NO_FATAL_FAILURE(ExpectRows(result_, 3, color));
}

TEST_F(ProjectorFixtureTest, EnforcesTheConsumersExpectedRowCount) {
  const std::array<uint8_t, 3> color = {255, 0, 17};
  ASSERT_NO_FATAL_FAILURE(MakeRequest(3, 0, color));
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const iree_host_size_t wrong_rows = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_finite_model_prepare_batch(
                            model_, 1, &prepared_, &wrong_rows));
  ASSERT_NO_FATAL_FAILURE(Invoke(3, &result_));
  ASSERT_NO_FATAL_FAILURE(ExpectRows(result_, 3, color));
}

}  // namespace
