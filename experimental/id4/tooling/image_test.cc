// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/image.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

static id4_pipeline_tensor_shape_t MakeRgbShape(uint64_t width,
                                                uint64_t height) {
  id4_pipeline_tensor_shape_t shape = {};
  shape.rank = 4;
  shape.dims[0] = width;
  shape.dims[1] = height;
  shape.dims[2] = 3;
  shape.dims[3] = 1;
  return shape;
}

TEST(ImageWriterTest, WritesWhcbF32RgbAsRowMajorBinaryPpm) {
  const float pixels[] = {
      1.0f, 0.0f, 0.0f,  // x0, y0: red
      0.0f, 1.0f, 0.0f,  // x0, y1: green
      0.0f, 0.0f, 1.0f,  // x1, y0: blue
      1.0f, 1.0f, 1.0f,  // x1, y1: white
  };

  iree::testing::TempFilePath output_path("id4_image_writer_test.ppm");
  id4_tooling_write_f32_rgb_ppm_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.path = output_path.path_view();
  options.shape = MakeRgbShape(2, 2);
  options.pixels = iree_make_const_byte_span(pixels, sizeof(pixels));
  options.normalization = ID4_TOOLING_IMAGE_NORMALIZATION_ZERO_TO_ONE;
  options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_write_f32_rgb_ppm(&options));

  const std::vector<uint8_t> contents = ReadBinaryFile(output_path.path());
  const std::vector<uint8_t> expected = {
      'P', '6', '\n', '2', ' ', '2', '\n', '2', '5', '5', '\n', 255,
      0,   0,   0,    0,   255, 0,   255,  0,   255, 255, 255,
  };
  EXPECT_EQ(contents, expected);
}

TEST(ImageWriterTest, NormalizesMinusOneToOneBeforeQuantization) {
  const float pixels[] = {-1.0f, 0.0f, 1.0f};

  iree::testing::TempFilePath output_path("id4_image_writer_normalized.ppm");
  id4_tooling_write_f32_rgb_ppm_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.path = output_path.path_view();
  options.shape = MakeRgbShape(1, 1);
  options.pixels = iree_make_const_byte_span(pixels, sizeof(pixels));
  options.normalization = ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
  options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_write_f32_rgb_ppm(&options));

  const std::vector<uint8_t> contents = ReadBinaryFile(output_path.path());
  const std::vector<uint8_t> expected = {
      'P', '6', '\n', '1', ' ', '1', '\n', '2', '5', '5', '\n', 0, 128, 255,
  };
  EXPECT_EQ(contents, expected);
}

TEST(ImageWriterTest, RejectsNonFinitePixels) {
  const float pixels[] = {0.0f, 0.0f, INFINITY};

  iree::testing::TempFilePath output_path("id4_image_writer_nonfinite.ppm");
  id4_tooling_write_f32_rgb_ppm_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.path = output_path.path_view();
  options.shape = MakeRgbShape(1, 1);
  options.pixels = iree_make_const_byte_span(pixels, sizeof(pixels));
  options.normalization = ID4_TOOLING_IMAGE_NORMALIZATION_ZERO_TO_ONE;
  options.host_allocator = iree_allocator_system();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_tooling_write_f32_rgb_ppm(&options));
}

}  // namespace
