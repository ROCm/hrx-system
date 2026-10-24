// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "amdf/media/av1.h"
#include "amdf/media/gfx11.h"
#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/interop/gpu/media/consume.h"
#include "libamdf/cts/interop/gpu/media/encode_fixture.h"

namespace {

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

amdf_media_frame_t Frame(amdf_memory_t* memory, amdf_media_image_t* image,
                         uint64_t byte_offset,
                         const amdf_media_image_description_t& description) {
  amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = description.format,
      .width = description.width,
      .height = description.height,
      .storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE,
  };
  frame.storage.image.memory = memory;
  frame.storage.image.byte_offset = byte_offset;
  frame.storage.image.image = image;
  return frame;
}

struct ExpectedComponents {
  uint32_t y;
  uint32_t u;
  uint32_t v;
};

ExpectedComponents ExpectedPixel(uint32_t x, uint32_t y, bool p010) {
  uint32_t luma = 16 + x / 3 + y / 5;
  if (y < 32) {
    luma = 40;
  }
  if (x >= 32 && x < 64 && y >= 96 && y < 128) {
    luma = 224;
  }
  const uint32_t chroma_x = x / 2;
  const uint32_t chroma_y = y / 2;
  uint32_t u = 96 + (chroma_x * 2) / 8;
  uint32_t v = 160 - (chroma_y * 2) / 8;
  if (p010) {
    luma = luma * 4 + (x + y) % 4;
    u = u * 4 + chroma_x % 4;
    v = v * 4 + chroma_y % 4;
  }
  return {luma, u, v};
}

void CheckRoundTripPixels(const uint32_t* pixels, uint32_t width,
                          uint32_t height, bool p010, double* out_psnr) {
  double squared_error = 0.0;
  uint64_t sample_count = 0;
  double stripe_sum = 0.0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint64_t pixel = uint64_t(y) * width + x;
      uint32_t actual_y = pixels[pixel * 2] & 65535u;
      uint32_t actual_u = pixels[pixel * 2] >> 16;
      uint32_t actual_v = pixels[pixel * 2 + 1] & 65535u;
      ASSERT_EQ(pixels[pixel * 2 + 1] >> 16, 0u);
      if (p010) {
        ASSERT_EQ((actual_y | actual_u | actual_v) & 63u, 0u);
        actual_y >>= 6;
        actual_u >>= 6;
        actual_v >>= 6;
      }
      const auto expected = ExpectedPixel(x, y, p010);
      const double y_error = double(actual_y) - expected.y;
      squared_error += y_error * y_error;
      ++sample_count;
      if (y < 32) {
        stripe_sum += actual_y;
      }
      if ((x & 1u) == 0 && (y & 1u) == 0) {
        const double u_error = double(actual_u) - expected.u;
        const double v_error = double(actual_v) - expected.v;
        squared_error += u_error * u_error + v_error * v_error;
        sample_count += 2;
      }
    }
  }
  ASSERT_EQ(sample_count, uint64_t(width) * height * 3 / 2);
  const double stripe = stripe_sum / (width * 32) / (p010 ? 4.0 : 1.0);
  uint32_t detected_generation = 0;
  for (uint32_t generation = 1; generation < 12; ++generation) {
    if (std::abs(stripe - (40.0 + generation * 12.0)) <
        std::abs(stripe - (40.0 + detected_generation * 12.0))) {
      detected_generation = generation;
    }
  }
  EXPECT_EQ(detected_generation, 0u) << "stripe=" << stripe;
  if (squared_error == 0.0) {
    *out_psnr = INFINITY;
    return;
  }
  const double maximum = p010 ? 1023.0 : 255.0;
  *out_psnr =
      10.0 * std::log10(maximum * maximum / (squared_error / sample_count));
}

}  // namespace

void MediaEncodeInteropTest::DecodeKeyPicture(
    const EncodeExecutionCase& test_case, uint32_t width, uint32_t height,
    amdf_memory_t* compressed_memory, uint64_t compressed_byte_offset,
    uint64_t compressed_byte_length, uint64_t compressed_byte_capacity,
    const amdf_media_av1_encode_result_info_t* av1_result,
    uint64_t encode_point) {
  ASSERT_EQ(decode_session_, nullptr);
  ASSERT_EQ(decode_command_, nullptr);
  ASSERT_NE(encode_point, 0u);
  const auto* kernel =
      kernels::media_consume::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  const auto* profile = Pm4CommandProfile::Find(gpu_endpoint_info_);
  ASSERT_NE(profile, nullptr);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 64u);

  amdf_media_h264_configuration_t h264_configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
      .structure_size = sizeof(h264_configuration),
      .profile = AMDF_MEDIA_H264_PROFILE_MAIN,
      .level_idc = 30,
  };
  amdf_media_hevc_configuration_t hevc_configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
      .structure_size = sizeof(hevc_configuration),
      .profile = static_cast<amdf_media_hevc_profile_t>(
          test_case.sample_bit_depth == 10 ? AMDF_MEDIA_HEVC_PROFILE_MAIN10
                                           : AMDF_MEDIA_HEVC_PROFILE_MAIN),
      .sample_bit_depth = test_case.sample_bit_depth,
  };
  amdf_media_av1_configuration_t av1_configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
      .structure_size = sizeof(av1_configuration),
      .profile = 0,
      .sample_bit_depth = test_case.sample_bit_depth,
  };
  const void* codec_configuration =
      test_case.codec == AMDF_MEDIA_CODEC_H264
          ? static_cast<const void*>(&h264_configuration)
      : test_case.codec == AMDF_MEDIA_CODEC_HEVC
          ? static_cast<const void*>(&hevc_configuration)
          : static_cast<const void*>(&av1_configuration);
  const amdf_media_configuration_query_t configuration_query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(configuration_query),
      .next = codec_configuration,
      .queue_family_ordinal = decode_family_,
      .configuration =
          {
              .codec = test_case.codec,
              .direction = AMDF_MEDIA_DIRECTION_DECODE,
              .format = test_case.format,
              .coded_width = width,
              .coded_height = height,
              .reference_image_count =
                  test_case.codec == AMDF_MEDIA_CODEC_AV1 ? 9u : 2u,
          },
  };
  amdf_media_configuration_info_t configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(configuration),
  };
  ASSERT_EQ(media_->device_query_configuration(
                media_device_, &configuration_query, &configuration),
            AMDF_STATUS_OK);
  ASSERT_EQ(configuration.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
  ASSERT_EQ(configuration.configuration.coded_width, width);
  ASSERT_EQ(configuration.configuration.coded_height, height);
  ASSERT_NE(configuration.output.byte_length, 0u);
  ASSERT_NE(configuration.reference.byte_length, 0u);
  ASSERT_EQ(
      compressed_byte_offset % configuration.compressed_byte_offset_alignment,
      0u);
  const uint64_t decode_compressed_byte_length = AlignUp(
      compressed_byte_length, configuration.compressed_byte_length_alignment);
  ASSERT_LE(decode_compressed_byte_length, compressed_byte_capacity);

  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = codec_configuration,
      .configuration = configuration.configuration,
  };
  ASSERT_EQ(
      media_->session_create(decode_queue_, &session_create, &decode_session_),
      AMDF_STATUS_OK);

  const uint64_t feedback_offset =
      AlignUp(64, configuration.feedback_byte_offset_alignment);
  const uint64_t feedback_length =
      feedback_offset + configuration.feedback_byte_length + 64;
  const amdf_memory_device_access_t feedback_access = {
      media_device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
  const amdf_memory_create_info_t feedback_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(feedback_create),
      .memory_profile_ordinal = configuration.feedback_memory_profile_ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = feedback_length,
      .minimum_alignment = configuration.feedback_byte_offset_alignment,
      .accesses = &feedback_access,
  };
  amdf_memory_t* feedback_memory = nullptr;
  ASSERT_EQ(api_->memory_create(configuration.feedback_memory_scope,
                                &feedback_create, &feedback_memory, nullptr),
            AMDF_STATUS_OK);
  memories_.push_back(feedback_memory);
  const amdf_memory_map_info_t feedback_map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(feedback_map),
      .byte_length = feedback_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api_->memory_map(feedback_memory, &feedback_map,
                             &decode_feedback_mapping_),
            AMDF_STATUS_OK);
  amdf_host_mapping_info_t feedback_mapping = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(feedback_mapping),
  };
  ASSERT_EQ(api_->host_mapping_query_info(decode_feedback_mapping_,
                                          &feedback_mapping),
            AMDF_STATUS_OK);
  decode_feedback_ = static_cast<uint8_t*>(feedback_mapping.pointer);
  std::memset(decode_feedback_, 0xa5, feedback_length);
  const amdf_media_command_create_info_t command_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
      .structure_size = sizeof(command_create),
      .maximum_slice_count = 1,
      .maximum_reference_count =
          configuration.configuration.reference_image_count - 1,
      .maximum_activation_count = 2,
      .feedback = {feedback_memory, 0, feedback_offset,
                   configuration.feedback_byte_length},
  };
  ASSERT_EQ(media_->command_create(decode_session_, &command_create,
                                   &decode_command_),
            AMDF_STATUS_OK);

  const uint64_t output_offset =
      AlignUp(65536, configuration.output.byte_offset_alignment);
  const uint64_t output_heap_length =
      output_offset + configuration.output.byte_length + 65536;
  const uint64_t reference_offset =
      AlignUp(65536, configuration.reference.byte_offset_alignment);
  const uint64_t reference_heap_length =
      reference_offset + configuration.reference.byte_length + 65536;
  amdf_memory_t* output_memory = nullptr;
  amdf_memory_t* reference_memory = nullptr;
  uint64_t output_address = 0;
  uint64_t reference_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      CreateSharedMemory(configuration.image_heap_memory_scope,
                         configuration.image_heap_memory_profile_ordinal,
                         output_heap_length, &output_memory, &output_address));
  ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
      configuration.image_heap_memory_scope,
      configuration.image_heap_memory_profile_ordinal, reference_heap_length,
      &reference_memory, &reference_address));

  const uint32_t gpu_access_ordinal = 1;
  const amdf_media_image_create_info_t output_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      .structure_size = sizeof(output_create),
      .memory = output_memory,
      .requirements = configuration.output,
      .byte_offset = output_offset,
      .layout_access_count = 1,
      .layout_access_ordinals = &gpu_access_ordinal,
  };
  amdf_media_image_t* output_image = nullptr;
  ASSERT_EQ(media_->image_create(media_device_, &output_create, &output_image),
            AMDF_STATUS_OK);
  images_.push_back(output_image);
  const amdf_media_image_create_info_t reference_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      .structure_size = sizeof(reference_create),
      .memory = reference_memory,
      .requirements = configuration.reference,
      .byte_offset = reference_offset,
  };
  amdf_media_image_t* reference_image = nullptr;
  ASSERT_EQ(
      media_->image_create(media_device_, &reference_create, &reference_image),
      AMDF_STATUS_OK);
  images_.push_back(reference_image);
  const amdf_media_frame_t output =
      Frame(output_memory, output_image, output_offset,
            configuration.output.description);
  const amdf_media_frame_t reference =
      Frame(reference_memory, reference_image, reference_offset,
            configuration.reference.description);

  amdf_media_gfx11_image_layout_t layout = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT,
      .structure_size = sizeof(layout),
  };
  amdf_media_image_info_t image_info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
      .structure_size = sizeof(image_info),
      .next = &layout,
  };
  const amdf_media_image_query_t image_query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
      .structure_size = sizeof(image_query),
  };
  ASSERT_EQ(
      media_->image_query_info(output_image, 1, &image_query, &image_info),
      AMDF_STATUS_OK);
  ASSERT_EQ(layout.plane_count, 2u);
  for (const auto& plane : layout.planes) {
    ASSERT_EQ(plane.swizzle_mode, 26u);
  }

  amdf_media_h264_scaling_matrices_t h264_matrices;
  std::memset(&h264_matrices, 16, sizeof(h264_matrices));
  const amdf_media_h264_slice_t h264_slice = {
      0, static_cast<uint32_t>(compressed_byte_length)};
  amdf_media_h264_decode_picture_info_t h264_picture = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO,
      .structure_size = sizeof(h264_picture),
      .sequence =
          {
              .maximum_reference_frame_count = 1,
              .picture_order_count_type = 2,
              .direct_8x8_inference = 1,
          },
      .picture = {.deblocking_filter_control_present = 1},
      .scaling_matrices = &h264_matrices,
      .is_reference = 1,
      .is_intra = 1,
      .slice_count = 1,
      .slices = &h264_slice,
  };
  const amdf_media_hevc_slice_t hevc_slice = {
      0, static_cast<uint32_t>(compressed_byte_length)};
  amdf_media_hevc_decode_picture_info_t hevc_picture = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO,
      .structure_size = sizeof(hevc_picture),
      .sequence =
          {
              .maximum_decoded_picture_buffer_count_minus1 = 1,
              .log2_diff_max_min_luma_coding_block_size = 3,
              .log2_diff_max_min_luma_transform_block_size = 3,
              .maximum_transform_hierarchy_depth_inter = 4,
              .maximum_transform_hierarchy_depth_intra = 4,
          },
      .picture =
          {
              .cabac_initialization_present = 1,
              .coding_unit_qp_delta_enabled = 1,
              .slice_chroma_qp_offsets_present = 1,
              .loop_filter_across_slices_enabled = 1,
          },
      .is_random_access = 1,
      .is_idr = 1,
      .is_intra = 1,
      .slice_count = 1,
      .slices = &hevc_slice,
  };
  amdf_media_av1_tile_t av1_tile = {
      0, static_cast<uint32_t>(compressed_byte_length)};
  amdf_media_av1_decode_picture_info_t av1_picture = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO,
      .structure_size = sizeof(av1_picture),
      .sequence = {.cdef = 1, .order_hint_bit_count = 4},
      .frame_type = 0,
      .show_frame = 1,
      .primary_reference = 7,
      .interpolation_filter = 4,
      .transform_mode = 1,
      .superres_denominator = 8,
      .tile_column_count = 1,
      .tile_row_count = 1,
      .tiles = &av1_tile,
  };
  std::fill(std::begin(av1_picture.reference_map),
            std::end(av1_picture.reference_map), UINT8_MAX);
  std::fill(std::begin(av1_picture.active_reference_indices),
            std::end(av1_picture.active_reference_indices), UINT8_MAX);
  av1_picture.tile_column_starts[1] = (width + 63) / 64;
  av1_picture.tile_row_starts[1] = (height + 63) / 64;
  if (test_case.codec == AMDF_MEDIA_CODEC_AV1) {
    ASSERT_NE(av1_result, nullptr);
    av1_picture.loop_filter = av1_result->loop_filter;
    for (auto& unit_size :
         av1_picture.loop_filter.log2_restoration_unit_sizes) {
      unit_size = 5;
    }
    av1_picture.quantization = av1_result->quantization;
    av1_picture.cdef = av1_result->cdef;
    av1_picture.segmentation = av1_result->segmentation;
  }
  const void* codec_picture = test_case.codec == AMDF_MEDIA_CODEC_H264
                                  ? static_cast<const void*>(&h264_picture)
                              : test_case.codec == AMDF_MEDIA_CODEC_HEVC
                                  ? static_cast<const void*>(&hevc_picture)
                                  : static_cast<const void*>(&av1_picture);
  const amdf_media_image_activation_t activations[] = {
      {nullptr, output_image},
      {nullptr, reference_image},
  };
  const amdf_media_decode_picture_info_t decode = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
      .structure_size = sizeof(decode),
      .next = codec_picture,
      .compressed = {compressed_memory, 0, compressed_byte_offset,
                     decode_compressed_byte_length},
      .output = &output,
      .reference_output = &reference,
      .reference_slot_index = 0,
  };
  const amdf_media_command_prepare_info_t prepare = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      .structure_size = sizeof(prepare),
      .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
      .activation_count = 2,
      .activations = activations,
      .decode = &decode,
  };
  ASSERT_EQ(media_->command_prepare(decode_command_, &prepare), AMDF_STATUS_OK);

  const uint64_t pixel_count = uint64_t(width) * height;
  const uint64_t result_byte_length = (pixel_count * 2 + 4096) * 4;
  GpuMemory* code = nullptr;
  GpuMemory* controls = nullptr;
  GpuMemory* results = nullptr;
  GpuMemory* heap_copy = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      kernel->executable.byte_length, &code, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   3 * 4096, &controls, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      result_byte_length + 128, &results, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   output_heap_length + reference_heap_length, &heap_copy,
                   AMDF_MEMORY_FLAG_HOST_COHERENT));
  std::memcpy(code->host.pointer, kernel->executable.words,
              kernel->executable.byte_length);
  auto* control_bytes = static_cast<uint8_t*>(controls->host.pointer);
  std::memset(control_bytes, 0, controls->info.byte_length);
  std::memset(results->host.pointer, 0xa5, results->info.byte_length);
  std::memset(heap_copy->host.pointer, 0xa5, heap_copy->info.byte_length);
  const Pm4ComputeProgram program = {
      code->device_address + kernel->entry_byte_offset,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {64, 1, 1},
  };
  const uint64_t pointers[] = {
      output_address + layout.planes[0].byte_offset,
      results->device_address + 64,
  };
  const uint32_t constants[] = {
      0,
      0,
      0,
      static_cast<uint32_t>(layout.planes[1].byte_offset -
                            layout.planes[0].byte_offset),
      layout.planes[0].pipe_bank_xor,
      layout.planes[1].pipe_bank_xor,
      width,
      height,
      test_case.sample_bit_depth == 10 ? 1u : 0u,
  };
  std::memcpy(control_bytes, pointers, sizeof(pointers));
  std::memcpy(control_bytes + sizeof(pointers), constants, sizeof(constants));
  auto make_command = [&](uint32_t slot, auto emit) {
    Pm4CommandWriter writer(
        reinterpret_cast<uint32_t*>(control_bytes + slot * 4096 + 256),
        *profile);
    writer.SystemBarrier();
    emit(writer);
    writer.WaitDma();
    writer.SystemBarrier();
    writer.PadToEightWords();
    return amdf_gpu_kernel_command_t{
        .memory = controls->memory,
        .byte_offset = slot * 4096 + 256,
        .byte_length = writer.word_count() * sizeof(uint32_t),
    };
  };
  const auto initialize = make_command(0, [&](Pm4CommandWriter& writer) {
    writer.DmaFill32(output_address, 0xa5a5a5a5,
                     static_cast<uint32_t>(output_heap_length));
    writer.DmaFill32(reference_address, 0xa5a5a5a5,
                     static_cast<uint32_t>(reference_heap_length));
  });
  const auto consume = make_command(1, [&](Pm4CommandWriter& writer) {
    writer.BindCompute(program, controls->device_address,
                       family_.format_features);
    writer.Dispatch(program, static_cast<uint32_t>(pixel_count), 1, 1);
  });
  const auto capture = make_command(2, [&](Pm4CommandWriter& writer) {
    writer.DmaCopy(output_address, heap_copy->device_address,
                   static_cast<uint32_t>(output_heap_length));
    writer.DmaCopy(reference_address,
                   heap_copy->device_address + output_heap_length,
                   static_cast<uint32_t>(reference_heap_length));
  });
  const std::vector<uint8_t> immutable_consumer(
      control_bytes + consume.byte_offset,
      control_bytes + consume.byte_offset + consume.byte_length);
  ASSERT_NO_FATAL_FAILURE(code->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(controls->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(results->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(heap_copy->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(initialize));
  const uint64_t initialized_point = gpu_point_;

  const amdf_kernel_queue_dependency_wait_t decode_waits[] = {
      {encode_to_decode_, encode_point},
      {gpu_to_decode_, initialized_point},
  };
  const amdf_media_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command = decode_command_,
      .wait_count = 2,
      .waits = decode_waits,
  };
  uint64_t point = 0;
  const auto status =
      media_->kernel_queue_submit(decode_queue_, &submit, &point);
  if (point != 0) {
    decode_point_ = point;
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_NE(point, 0u);
  const amdf_kernel_queue_dependency_wait_t decoded = {decode_to_gpu_, point};
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(consume, &decoded));
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(capture));
  ASSERT_EQ(
      api_->kernel_queue_wait(gpu_queue_, gpu_point_, AMDF_TIMEOUT_INFINITE, 0),
      AMDF_STATUS_OK);
  amdf_kernel_queue_status_t progress = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(progress),
  };
  ASSERT_EQ(api_->kernel_queue_refresh_status(decode_queue_, &progress),
            AMDF_STATUS_OK);
  ASSERT_EQ(progress.terminal_status, AMDF_STATUS_OK);
  ASSERT_EQ(progress.retired_submission, decode_point_);
  amdf_media_command_result_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
      .structure_size = sizeof(result),
  };
  ASSERT_EQ(media_->command_query_result(decode_command_, &result),
            AMDF_STATUS_OK);
  ASSERT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_SUCCEEDED);
  ASSERT_EQ(result.operation, AMDF_MEDIA_COMMAND_OPERATION_PICTURE);
  ASSERT_EQ(result.submission, decode_point_);
  ASSERT_NO_FATAL_FAILURE(results->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  ASSERT_NO_FATAL_FAILURE(heap_copy->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  const auto* result_bytes = static_cast<const uint8_t*>(results->host.pointer);
  double psnr = 0.0;
  ASSERT_NO_FATAL_FAILURE(CheckRoundTripPixels(
      reinterpret_cast<const uint32_t*>(result_bytes + 64), width, height,
      test_case.sample_bit_depth == 10, &psnr));
  ASSERT_GE(psnr, 30.0);
  for (uint32_t guard = 0; guard < 64; ++guard) {
    ASSERT_EQ(result_bytes[guard], 0xa5u);
    ASSERT_EQ(result_bytes[64 + result_byte_length + guard], 0xa5u);
  }
  const auto* heap_bytes = static_cast<const uint8_t*>(heap_copy->host.pointer);
  for (uint64_t index = 0; index < output_offset; ++index) {
    ASSERT_EQ(heap_bytes[index], 0xa5u);
  }
  for (uint64_t index = output_offset + configuration.output.byte_length;
       index < output_heap_length; ++index) {
    ASSERT_EQ(heap_bytes[index], 0xa5u);
  }
  const uint64_t reference_copy_offset = output_heap_length;
  for (uint64_t index = 0; index < reference_offset; ++index) {
    ASSERT_EQ(heap_bytes[reference_copy_offset + index], 0xa5u);
  }
  for (uint64_t index = reference_offset + configuration.reference.byte_length;
       index < reference_heap_length; ++index) {
    ASSERT_EQ(heap_bytes[reference_copy_offset + index], 0xa5u);
  }
  for (uint64_t index = 0; index < feedback_length; ++index) {
    if (index < feedback_offset ||
        index >= feedback_offset + configuration.feedback_byte_length) {
      ASSERT_EQ(decode_feedback_[index], 0xa5u);
    }
  }
  ASSERT_EQ(std::memcmp(control_bytes + consume.byte_offset,
                        immutable_consumer.data(), immutable_consumer.size()),
            0);
  RecordProperty("round_trip_psnr_db", psnr);
  RecordProperty("round_trip_compressed_bytes", compressed_byte_length);
  RecordProperty("round_trip_device_only_payload", 1);
}
