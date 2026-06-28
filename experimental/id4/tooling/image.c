// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/image.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static iree_status_t id4_tooling_image_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_tooling_image_dup_cstring(
    iree_string_view_t value, iree_allocator_t host_allocator,
    char** out_string) {
  *out_string = NULL;
  if (value.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image path is too large");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.size);
  storage[value.size] = 0;
  *out_string = storage;
  return iree_ok_status();
}

static iree_status_t id4_tooling_image_validate_normalization(
    id4_tooling_image_normalization_t normalization) {
  switch (normalization) {
    case ID4_TOOLING_IMAGE_NORMALIZATION_ZERO_TO_ONE:
    case ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "image normalization %d is invalid",
                              (int)normalization);
  }
}

static iree_status_t id4_tooling_image_validate_f32_rgb_tensor(
    id4_pipeline_tensor_shape_t shape, iree_const_byte_span_t pixels,
    iree_host_size_t* out_element_count) {
  IREE_ASSERT_ARGUMENT(out_element_count);
  *out_element_count = 0;
  if (shape.rank != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor must be rank-4 WHCB");
  }
  if (shape.dims[0] == 0 || shape.dims[1] == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image width and height must be non-zero");
  }
  if (shape.dims[2] != 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor channel count must be 3");
  }
  if (shape.dims[3] != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor batch count must be 1");
  }
  iree_host_size_t pixel_count = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)shape.dims[0],
                                  (iree_host_size_t)shape.dims[1],
                                  &pixel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image pixel count overflowed");
  }
  iree_host_size_t element_count = 0;
  if (!iree_host_size_checked_mul(pixel_count, 3, &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image element count overflowed");
  }
  iree_host_size_t expected_byte_length = 0;
  if (!iree_host_size_checked_mul(element_count, sizeof(float),
                                  &expected_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image tensor byte length overflowed");
  }
  if (pixels.data_length != expected_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor byte length %" PRIhsz
                            " does not match expected %" PRIhsz,
                            pixels.data_length, expected_byte_length);
  }
  if (!pixels.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor bytes are required");
  }
  *out_element_count = element_count;
  return iree_ok_status();
}

static iree_status_t id4_tooling_image_validate_options(
    const id4_tooling_write_f32_rgb_ppm_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image write options are required");
  }
  IREE_RETURN_IF_ERROR(id4_tooling_image_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("image write")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "image write extension structures are not "
                            "supported");
  }
  if (iree_string_view_is_empty(options->path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image output path is required");
  }
  if (iree_allocator_is_null(options->host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image host allocator is required");
  }
  if (options->shape.dims[0] > UINT32_MAX ||
      options->shape.dims[1] > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image dimensions exceed PPM writer limits");
  }
  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_tooling_image_validate_f32_rgb_tensor(
      options->shape, options->pixels, &element_count));
  return id4_tooling_image_validate_normalization(options->normalization);
}

static float id4_tooling_image_normalize(
    float value, id4_tooling_image_normalization_t normalization) {
  switch (normalization) {
    case ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE:
      return (value + 1.0f) * 0.5f;
    case ID4_TOOLING_IMAGE_NORMALIZATION_ZERO_TO_ONE:
    default:
      return value;
  }
}

static uint8_t id4_tooling_image_quantize(float value) {
  if (value < 0.0f) value = 0.0f;
  if (value > 1.0f) value = 1.0f;
  return (uint8_t)(value * 255.0f + 0.5f);
}

iree_status_t id4_tooling_write_f32_rgb_ppm(
    const id4_tooling_write_f32_rgb_ppm_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_tooling_image_validate_options(options));

  char* path_string = NULL;
  IREE_RETURN_IF_ERROR(id4_tooling_image_dup_cstring(
      options->path, options->host_allocator, &path_string));
  FILE* file = fopen(path_string, "wb");
  const int open_errno = errno;
  iree_allocator_free(options->host_allocator, path_string);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(open_errno),
                            "failed to open image file (%d)", open_errno);
  }

  iree_status_t status = iree_ok_status();
  const uint32_t width = (uint32_t)options->shape.dims[0];
  const uint32_t height = (uint32_t)options->shape.dims[1];
  if (fprintf(file, "P6\n%u %u\n255\n", width, height) < 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS, "failed to write image header");
  }

  const float* pixels = (const float*)options->pixels.data;
  for (uint32_t y = 0; iree_status_is_ok(status) && y < height; ++y) {
    for (uint32_t x = 0; iree_status_is_ok(status) && x < width; ++x) {
      uint8_t rgb[3] = {0, 0, 0};
      for (uint32_t channel = 0; channel < 3; ++channel) {
        const uint64_t offset = (((uint64_t)x * height + y) * 3u) + channel;
        const float value = pixels[offset];
        if (!isfinite(value)) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "image pixel at x=%u y=%u channel=%u is not finite", x, y,
              channel);
          break;
        }
        rgb[channel] = id4_tooling_image_quantize(
            id4_tooling_image_normalize(value, options->normalization));
      }
      if (iree_status_is_ok(status) &&
          fwrite(rgb, 1, sizeof(rgb), file) != sizeof(rgb)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to write image pixel data");
      }
    }
  }
  if (fclose(file) != 0 && iree_status_is_ok(status)) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "failed to close image file (%d)", errno);
  }
  return status;
}

iree_status_t id4_tooling_validate_f32_rgb_image_contents(
    const id4_tooling_validate_f32_rgb_image_contents_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image validation options are required");
  }
  IREE_RETURN_IF_ERROR(id4_tooling_image_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("image validation")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "image validation extension structures are not "
                            "supported");
  }
  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_tooling_image_validate_f32_rgb_tensor(
      options->shape, options->pixels, &element_count));
  IREE_RETURN_IF_ERROR(
      id4_tooling_image_validate_normalization(options->normalization));

  const float* pixels = (const float*)options->pixels.data;
  bool saw_nonzero = false;
  uint8_t quantized_min = UINT8_MAX;
  uint8_t quantized_max = 0;
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    const float value = pixels[i];
    if (!isfinite(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "image element %" PRIhsz " is not finite", i);
    }
    if (value != 0.0f) saw_nonzero = true;
    const uint8_t quantized = id4_tooling_image_quantize(
        id4_tooling_image_normalize(value, options->normalization));
    if (quantized < quantized_min) quantized_min = quantized;
    if (quantized > quantized_max) quantized_max = quantized;
  }
  if (!saw_nonzero) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor contains only zero values");
  }
  if (quantized_min == quantized_max) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image tensor quantizes to a flat image");
  }
  return iree_ok_status();
}
