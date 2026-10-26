// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "amdf/media/av1.h"
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

ExpectedComponents ExpectedPixel(uint32_t x, uint32_t y, bool p010,
                                 uint32_t generation) {
  uint32_t luma = 16 + x / 3 + y / 5 + generation * 4;
  if (y < 32) {
    luma = 40 + generation * 12;
  }
  if (x >= 32 + generation * 8 && x < 64 + generation * 8 && y >= 96 &&
      y < 128) {
    luma = 224;
  }
  const uint32_t chroma_x = x / 2;
  const uint32_t chroma_y = y / 2;
  uint32_t u = 96 + (chroma_x * 2) / 8 + generation;
  uint32_t v = 160 - (chroma_y * 2) / 8 - generation;
  if (p010) {
    luma = luma * 4 + (x + y + generation) % 4;
    u = u * 4 + (chroma_x + generation) % 4;
    v = v * 4 + (chroma_y + generation) % 4;
  }
  return {luma, u, v};
}

class Av1BitReader {
 public:
  Av1BitReader(const uint8_t* bytes, uint64_t byte_length, uint64_t byte_offset)
      : bytes_(bytes),
        bit_capacity_(byte_length * 8),
        bit_offset_(byte_offset * 8) {}

  uint32_t Read(uint32_t bit_count) {
    if (!valid_ || bit_count > 32 ||
        bit_count > bit_capacity_ - std::min(bit_offset_, bit_capacity_)) {
      valid_ = false;
      return 0;
    }
    uint32_t value = 0;
    for (uint32_t index = 0; index < bit_count; ++index) {
      value = (value << 1) |
              ((bytes_[bit_offset_ / 8] >> (7 - bit_offset_ % 8)) & 1u);
      ++bit_offset_;
    }
    return value;
  }

  int32_t ReadSigned(uint32_t bit_count) {
    const uint32_t value = Read(bit_count);
    return (value & (1u << (bit_count - 1))) != 0
               ? static_cast<int32_t>(value | (~uint32_t{0} << bit_count))
               : static_cast<int32_t>(value);
  }

  bool AlignWithZero() {
    while (valid_ && bit_offset_ % 8 != 0) {
      if (Read(1) != 0) {
        return false;
      }
    }
    return valid_;
  }

  bool valid() const { return valid_; }
  uint64_t byte_offset() const { return bit_offset_ / 8; }

 private:
  const uint8_t* bytes_;
  uint64_t bit_capacity_;
  uint64_t bit_offset_;
  bool valid_ = true;
};

// Linux VCN emits the complete sized frame OBU while the decode API takes the
// frame controls and tile payload separately. Recover exactly the shown,
// single-tile KEY/INTER envelope produced by this CTS configuration. INTER
// frames inherit their retained loop-filter deltas from the prior picture.
::testing::AssertionResult ParseAv1FrameObu(
    const uint8_t* bytes, uint64_t byte_length, uint32_t temporal_layer_count,
    uint32_t expected_temporal_layer_index,
    const amdf_media_av1_loop_filter_t* primary_filter,
    amdf_media_av1_decode_picture_info_t* picture,
    amdf_media_av1_tile_t* tile) {
  const bool scalable = temporal_layer_count > 1;
  const uint8_t expected_header = scalable ? 0x36 : 0x32;
  if (bytes == nullptr || byte_length < (scalable ? 4u : 3u) ||
      bytes[0] != expected_header) {
    return ::testing::AssertionFailure() << "expected one sized AV1 frame OBU";
  }
  uint64_t position = 1;
  if (scalable) {
    const uint8_t extension = bytes[position++];
    const uint32_t temporal_layer_index = extension >> 5;
    const uint32_t spatial_layer_index = (extension >> 3) & 3u;
    if (temporal_layer_index != expected_temporal_layer_index ||
        spatial_layer_index != 0 || (extension & 7u) != 0) {
      return ::testing::AssertionFailure()
             << "AV1 OBU extension has temporal_id=" << temporal_layer_index
             << " spatial_id=" << spatial_layer_index;
    }
  }
  uint64_t obu_byte_length = 0;
  uint32_t shift = 0;
  for (;;) {
    if (position >= byte_length || shift > 56) {
      return ::testing::AssertionFailure() << "invalid AV1 OBU size";
    }
    const uint8_t value = bytes[position++];
    obu_byte_length |= static_cast<uint64_t>(value & 127) << shift;
    if ((value & 128) == 0) {
      break;
    }
    shift += 7;
  }
  if (obu_byte_length != byte_length - position) {
    return ::testing::AssertionFailure()
           << "AV1 frame OBU size " << obu_byte_length
           << " does not cover the encoded region " << byte_length;
  }

  Av1BitReader reader(bytes, byte_length, position);
  const uint32_t show_existing_frame = reader.Read(1);
  picture->frame_type = reader.Read(2);
  picture->show_frame = reader.Read(1);
  const bool inter = picture->frame_type == 1;
  if (show_existing_frame != 0 ||
      (picture->frame_type != 0 && picture->frame_type != 1) ||
      picture->show_frame != 1 || inter != (primary_filter != nullptr)) {
    return ::testing::AssertionFailure()
           << "expected the admitted shown AV1 KEY/INTER sequence";
  }
  const uint32_t error_resilient = inter ? reader.Read(1) : 1u;
  picture->disable_cdf_update = reader.Read(1);
  const uint32_t frame_size_override = reader.Read(1);
  picture->order_hint = reader.Read(4);
  picture->primary_reference = inter ? reader.Read(3) : 7u;
  const uint32_t refresh_frame_flags = inter ? reader.Read(8) : 255u;
  const uint32_t short_reference_signaling = inter ? reader.Read(1) : 0u;
  if (inter) {
    for (auto& reference_index : picture->active_reference_indices) {
      reference_index = static_cast<uint8_t>(reader.Read(3));
    }
  }
  const uint32_t render_size_differs = reader.Read(1);
  if (inter) {
    picture->high_precision_motion_vectors = reader.Read(1);
    if (reader.Read(1) != 0) {
      picture->interpolation_filter = 4;
    } else {
      picture->interpolation_filter = reader.Read(2);
    }
    picture->switchable_motion_mode = reader.Read(1);
  }
  picture->disable_frame_end_cdf_update = reader.Read(1);
  const bool active_references_valid =
      !inter || std::all_of(std::begin(picture->active_reference_indices),
                            std::end(picture->active_reference_indices),
                            [](uint8_t index) { return index == 0; });
  const uint32_t expected_refresh_frame_flags =
      expected_temporal_layer_index == 0 ? 255u : 0u;
  if (error_resilient != (inter ? 0u : 1u) ||
      picture->disable_cdf_update != 0 || frame_size_override != 0 ||
      picture->primary_reference != (inter ? 0u : 7u) ||
      refresh_frame_flags != expected_refresh_frame_flags ||
      short_reference_signaling != 0 || !active_references_valid ||
      render_size_differs != 0 || picture->high_precision_motion_vectors != 0 ||
      picture->interpolation_filter != 4 ||
      picture->switchable_motion_mode != 0 ||
      picture->disable_frame_end_cdf_update != 0) {
    return ::testing::AssertionFailure()
           << "AV1 frame controls exceed the admitted encode envelope";
  }

  const uint32_t uniform_tiles = reader.Read(1);
  const uint32_t additional_tile_column = reader.Read(1);
  const uint32_t additional_tile_row = reader.Read(1);
  if (uniform_tiles != 1 || additional_tile_column != 0 ||
      additional_tile_row != 0) {
    return ::testing::AssertionFailure() << "expected one uniform AV1 tile";
  }

  auto read_delta = [&]() -> int32_t {
    return reader.Read(1) != 0 ? reader.ReadSigned(7) : 0;
  };
  auto& quantization = picture->quantization;
  quantization.base_index = reader.Read(8);
  quantization.y_dc_delta = read_delta();
  const int32_t chroma_dc_delta = read_delta();
  const int32_t chroma_ac_delta = read_delta();
  quantization.chroma_dc_deltas[0] = chroma_dc_delta;
  quantization.chroma_dc_deltas[1] = chroma_dc_delta;
  quantization.chroma_ac_deltas[0] = chroma_ac_delta;
  quantization.chroma_ac_deltas[1] = chroma_ac_delta;
  const uint32_t using_quantization_matrix = reader.Read(1);
  if (using_quantization_matrix != 0) {
    quantization.matrix_levels[0] = reader.Read(4);
    quantization.matrix_levels[1] = reader.Read(4);
    quantization.matrix_levels[2] = reader.Read(4);
  } else {
    std::fill(std::begin(quantization.matrix_levels),
              std::end(quantization.matrix_levels), UINT8_MAX);
  }
  if (reader.Read(1) != 0) {
    return ::testing::AssertionFailure()
           << "AV1 segmentation exceeds the admitted encode envelope";
  }

  if (quantization.base_index != 0) {
    quantization.delta_present = reader.Read(1);
    if (quantization.delta_present != 0) {
      quantization.delta_resolution = reader.Read(2);
    }
  }
  auto& filter = picture->loop_filter;
  if (primary_filter != nullptr) {
    std::copy(std::begin(primary_filter->reference_deltas),
              std::end(primary_filter->reference_deltas),
              filter.reference_deltas);
    std::copy(std::begin(primary_filter->mode_deltas),
              std::end(primary_filter->mode_deltas), filter.mode_deltas);
  } else {
    const int8_t reference_defaults[8] = {1, 0, 0, 0, -1, 0, -1, -1};
    std::copy(std::begin(reference_defaults), std::end(reference_defaults),
              filter.reference_deltas);
  }
  if (quantization.delta_present != 0) {
    filter.delta_present = reader.Read(1);
    if (filter.delta_present != 0) {
      filter.delta_resolution = reader.Read(2);
      filter.delta_multi = reader.Read(1);
    }
  }

  const bool coded_lossless = quantization.base_index == 0 &&
                              quantization.y_dc_delta == 0 &&
                              chroma_dc_delta == 0 && chroma_ac_delta == 0;
  if (!coded_lossless) {
    filter.levels[0] = static_cast<uint8_t>(reader.Read(6));
    filter.levels[1] = static_cast<uint8_t>(reader.Read(6));
    if (filter.levels[0] != 0 || filter.levels[1] != 0) {
      filter.levels[2] = static_cast<uint8_t>(reader.Read(6));
      filter.levels[3] = static_cast<uint8_t>(reader.Read(6));
    }
    filter.sharpness = reader.Read(3);
    filter.deltas_enabled = reader.Read(1);
    if (filter.deltas_enabled != 0) {
      filter.deltas_updated = reader.Read(1);
      if (filter.deltas_updated != 0) {
        for (auto& delta : filter.reference_deltas) {
          if (reader.Read(1) != 0) {
            delta = static_cast<int8_t>(reader.ReadSigned(7));
          }
        }
        for (auto& delta : filter.mode_deltas) {
          if (reader.Read(1) != 0) {
            delta = static_cast<int8_t>(reader.ReadSigned(7));
          }
        }
      }
    }

    auto& cdef = picture->cdef;
    cdef.damping_minus3 = reader.Read(2);
    cdef.bits = reader.Read(2);
    for (uint32_t index = 0; index < (1u << cdef.bits); ++index) {
      cdef.luma_primary[index] = static_cast<uint8_t>(reader.Read(4));
      cdef.luma_secondary[index] = static_cast<uint8_t>(reader.Read(2));
      cdef.chroma_primary[index] = static_cast<uint8_t>(reader.Read(4));
      cdef.chroma_secondary[index] = static_cast<uint8_t>(reader.Read(2));
    }
    picture->transform_mode = reader.Read(1) != 0 ? 2u : 1u;
  }
  if (inter) {
    picture->reference_select = reader.Read(1);
  }
  picture->reduced_transform_set = reader.Read(1);
  bool identity_global_motion = true;
  if (inter) {
    for (auto& motion : picture->global_motion) {
      identity_global_motion &= reader.Read(1) == 0;
      motion.parameters[2] = 1 << 16;
      motion.parameters[5] = 1 << 16;
    }
  }
  if (picture->reference_select != 0 || picture->reduced_transform_set != 0 ||
      !identity_global_motion || !reader.AlignWithZero()) {
    return ::testing::AssertionFailure()
           << "invalid AV1 frame tail or byte alignment";
  }
  if (!reader.valid() || reader.byte_offset() >= byte_length ||
      reader.byte_offset() > std::numeric_limits<uint32_t>::max() ||
      byte_length - reader.byte_offset() >
          std::numeric_limits<uint32_t>::max()) {
    return ::testing::AssertionFailure()
           << "AV1 frame OBU contains no bounded tile payload";
  }
  tile->byte_offset = static_cast<uint32_t>(reader.byte_offset());
  tile->byte_length = static_cast<uint32_t>(byte_length - tile->byte_offset);
  return ::testing::AssertionSuccess();
}

uint32_t UnsignedExpGolombBitCount(uint32_t value) {
  uint32_t magnitude = 0;
  for (uint32_t code_number = value + 1; code_number > 1; code_number >>= 1) {
    ++magnitude;
  }
  return magnitude * 2 + 1;
}

uint32_t H264FrameNumber(uint32_t generation, uint32_t temporal_layer_count) {
  return temporal_layer_count == 1 ? generation : (generation + 1) / 2;
}

int32_t H264PictureOrderCount(uint32_t generation,
                              uint32_t temporal_layer_count) {
  return static_cast<int32_t>(temporal_layer_count == 1 ? generation * 2
                                                        : generation);
}

void CheckRoundTripPixels(const uint32_t* pixels, uint32_t width,
                          uint32_t height, bool p010,
                          uint32_t expected_generation, double* out_psnr) {
  double squared_error = 0.0;
  uint64_t sample_count = 0;
  double stripe_sum = 0.0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint64_t pixel = uint64_t(y) * width + x;
      uint32_t actual_y = pixels[pixel * 2] & 65535u;
      uint32_t actual_u = pixels[pixel * 2] >> 16;
      uint32_t actual_v = pixels[pixel * 2 + 1] & 65535u;
      ASSERT_EQ(pixels[pixel * 2 + 1] >> 16, expected_generation);
      if (p010) {
        ASSERT_EQ((actual_y | actual_u | actual_v) & 63u, 0u);
        actual_y >>= 6;
        actual_u >>= 6;
        actual_v >>= 6;
      }
      const auto expected = ExpectedPixel(x, y, p010, expected_generation);
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
  EXPECT_EQ(detected_generation, expected_generation) << "stripe=" << stripe;
  if (squared_error == 0.0) {
    *out_psnr = INFINITY;
    return;
  }
  const double maximum = p010 ? 1023.0 : 255.0;
  *out_psnr =
      10.0 * std::log10(maximum * maximum / (squared_error / sample_count));
}

}  // namespace

void MediaEncodeInteropTest::DecodePictures(
    const EncodeExecutionCase& test_case, uint32_t width, uint32_t height,
    amdf_memory_t* compressed_memory,
    std::span<const EncodedPictureRegion> pictures, uint64_t encode_point,
    uint32_t temporal_layer_count) {
  ASSERT_EQ(decode_session_, nullptr);
  ASSERT_TRUE(decode_commands_.empty());
  ASSERT_FALSE(pictures.empty());
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
  std::vector<uint64_t> decode_compressed_byte_lengths;
  decode_compressed_byte_lengths.reserve(pictures.size());
  for (const auto& picture : pictures) {
    ASSERT_EQ(
        picture.byte_offset % configuration.compressed_byte_offset_alignment,
        0u);
    const uint64_t decode_byte_length = AlignUp(
        picture.byte_length, configuration.compressed_byte_length_alignment);
    ASSERT_LE(decode_byte_length, picture.byte_capacity);
    decode_compressed_byte_lengths.push_back(decode_byte_length);
  }

  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = codec_configuration,
      .configuration = configuration.configuration,
  };
  ASSERT_EQ(
      media_->session_create(decode_queue_, &session_create, &decode_session_),
      AMDF_STATUS_OK);

  uint64_t feedback_offset = 0;
  uint64_t feedback_stride = 0;
  uint64_t feedback_length = 0;
  amdf_memory_t* feedback_memory = nullptr;
  if (configuration.feedback_byte_length != 0) {
    ASSERT_NE(configuration.feedback_byte_offset_alignment, 0u);
    feedback_offset = AlignUp(64, configuration.feedback_byte_offset_alignment);
    feedback_stride = AlignUp(configuration.feedback_byte_length,
                              configuration.feedback_byte_offset_alignment);
    feedback_length = feedback_offset + pictures.size() * feedback_stride + 64;
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
  }
  for (uint32_t index = 0; index < pictures.size(); ++index) {
    const amdf_media_command_create_info_t command_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
        .structure_size = sizeof(command_create),
        .maximum_slice_count = 1,
        .maximum_reference_count =
            configuration.configuration.reference_image_count - 1,
        .maximum_activation_count = 2,
        .feedback = {feedback_memory, 0,
                     feedback_offset + index * feedback_stride,
                     configuration.feedback_byte_length},
    };
    amdf_media_command_t* command = nullptr;
    ASSERT_EQ(
        media_->command_create(decode_session_, &command_create, &command),
        AMDF_STATUS_OK);
    decode_commands_.push_back(command);
  }
  if (configuration.requires_begin_end != 0) {
    ASSERT_NO_FATAL_FAILURE(
        SubmitMediaControl(decode_queue_, decode_commands_[0],
                           AMDF_MEDIA_COMMAND_OPERATION_BEGIN, &decode_point_));
    decode_session_active_ = true;
  }

  const uint64_t output_offset =
      AlignUp(65536, configuration.output.byte_offset_alignment);
  const uint64_t output_heap_length =
      output_offset + configuration.output.byte_length + 65536;
  const uint64_t reference_offset =
      AlignUp(65536, configuration.reference.byte_offset_alignment);
  const uint32_t reference_count =
      configuration.configuration.reference_image_count;
  const uint64_t reference_heap_length =
      reference_offset + reference_count * configuration.reference.byte_length +
      65536;
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
  const amdf_media_frame_t output =
      Frame(output_memory, output_image, output_offset,
            configuration.output.description);
  std::vector<amdf_media_image_t*> reference_images(reference_count);
  std::vector<amdf_media_frame_t> references(reference_count);
  for (uint32_t index = 0; index < reference_count; ++index) {
    const uint64_t offset =
        reference_offset + index * configuration.reference.byte_length;
    const amdf_media_image_create_info_t reference_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
        .structure_size = sizeof(reference_create),
        .memory = reference_memory,
        .requirements = configuration.reference,
        .byte_offset = offset,
    };
    ASSERT_EQ(media_->image_create(media_device_, &reference_create,
                                   &reference_images[index]),
              AMDF_STATUS_OK);
    images_.push_back(reference_images[index]);
    references[index] = Frame(reference_memory, reference_images[index], offset,
                              configuration.reference.description);
  }

  MediaKernelImageLayout layout;
  ASSERT_NO_FATAL_FAILURE(QueryKernelImageLayout(output_image, 0, &layout));

  amdf_media_h264_scaling_matrices_t h264_matrices;
  std::memset(&h264_matrices, 16, sizeof(h264_matrices));
  std::vector<uint32_t> reference_generations(reference_count, UINT32_MAX);
  std::vector<amdf_media_av1_loop_filter_t> av1_reference_filters(
      reference_count);
  std::vector<bool> reference_active(reference_count, false);
  for (uint32_t index = 0; index < pictures.size(); ++index) {
    const auto& encoded = pictures[index];
    ASSERT_LT(encoded.temporal_layer_index, temporal_layer_count);
    const bool key = encoded.reference_generation == UINT32_MAX;
    ASSERT_EQ(key, encoded.generation == 0);
    const bool retained =
        temporal_layer_count == 1 || encoded.temporal_layer_index == 0;
    uint32_t reference_slot = UINT32_MAX;
    if (!key) {
      const auto reference_position =
          std::find(reference_generations.begin(), reference_generations.end(),
                    encoded.reference_generation);
      ASSERT_NE(reference_position, reference_generations.end())
          << "missing generation " << encoded.reference_generation;
      reference_slot = static_cast<uint32_t>(reference_position -
                                             reference_generations.begin());
      ASSERT_LT(reference_slot, 2u);
    }
    const uint32_t reconstruction_slot =
        key ? 0u : (reference_slot == 0 ? 1u : 0u);
    ASSERT_LT(reconstruction_slot, references.size());
    const amdf_media_h264_slice_t h264_slice = {
        0, static_cast<uint32_t>(encoded.byte_length)};
    amdf_media_h264_reference_t h264_reference = {};
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
        .frame_number =
            H264FrameNumber(encoded.generation, temporal_layer_count),
        .is_reference = retained ? 1u : 0u,
        .is_intra = key ? 1u : 0u,
        .top_field_order_count =
            H264PictureOrderCount(encoded.generation, temporal_layer_count),
        .bottom_field_order_count =
            H264PictureOrderCount(encoded.generation, temporal_layer_count),
        .slice_count = 1,
        .slices = &h264_slice,
    };
    if (!key) {
      h264_reference.binding = {&references[reference_slot], reference_slot};
      h264_reference.frame_number =
          H264FrameNumber(encoded.reference_generation, temporal_layer_count);
      h264_reference.top_field_order_count = H264PictureOrderCount(
          encoded.reference_generation, temporal_layer_count);
      h264_reference.bottom_field_order_count =
          h264_reference.top_field_order_count;
      h264_picture.reference_count = 1;
      h264_picture.references = &h264_reference;
    }

    const amdf_media_hevc_slice_t hevc_slice = {
        0, static_cast<uint32_t>(encoded.byte_length)};
    amdf_media_hevc_reference_t hevc_reference = {};
    const uint32_t reference_delta =
        key ? 0u : encoded.generation - encoded.reference_generation;
    // One negative reference uses ue(1), ue(0), ue(delta_poc_minus1), and a
    // used flag after short_term_ref_pic_set_sps_flag.
    const uint32_t hevc_reference_set_bit_count =
        key ? 0u : 5u + UnsignedExpGolombBitCount(reference_delta - 1);
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
        .picture_order_count = static_cast<int32_t>(encoded.generation),
        .is_random_access = key ? 1u : 0u,
        .is_idr = key ? 1u : 0u,
        .is_intra = key ? 1u : 0u,
        .short_term_reference_set_bit_count = hevc_reference_set_bit_count,
        .slice_count = 1,
        .slices = &hevc_slice,
    };
    if (!key) {
      hevc_reference.binding = {&references[reference_slot], reference_slot};
      hevc_reference.picture_order_count =
          static_cast<int32_t>(encoded.reference_generation);
      hevc_picture.reference_count = 1;
      hevc_picture.references = &hevc_reference;
      hevc_picture.short_term_before.count = 1;
      hevc_picture.short_term_before.indices[0] = 0;
    }
    amdf_media_av1_tile_t av1_tile = {
        0, static_cast<uint32_t>(encoded.byte_length)};
    amdf_media_reference_t av1_reference = {};
    amdf_media_av1_decode_picture_info_t av1_picture = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO,
        .structure_size = sizeof(av1_picture),
        .sequence = {.cdef = 1, .order_hint_bit_count = 4},
        .frame_type = key ? 0u : 1u,
        .show_frame = 1,
        .primary_reference = key ? 7u : 0u,
        .order_hint = encoded.generation,
        .interpolation_filter = 4,
        .transform_mode = key ? 1u : 2u,
        .superres_denominator = 8,
        .tile_column_count = 1,
        .tile_row_count = 1,
        .tiles = &av1_tile,
    };
    std::fill(std::begin(av1_picture.reference_map),
              std::end(av1_picture.reference_map), UINT8_MAX);
    std::fill(std::begin(av1_picture.active_reference_indices),
              std::end(av1_picture.active_reference_indices), UINT8_MAX);
    for (auto& motion : av1_picture.global_motion) {
      motion.parameters[2] = 1 << 16;
      motion.parameters[5] = 1 << 16;
    }
    av1_picture.tile_column_starts[1] = (width + 63) / 64;
    av1_picture.tile_row_starts[1] = (height + 63) / 64;
    if (test_case.codec == AMDF_MEDIA_CODEC_AV1) {
      if (encoded.kind == AMDF_MEDIA_ENCODED_REGION_AV1_FRAME_OBU) {
        const amdf_media_av1_loop_filter_t* primary_filter =
            key ? nullptr : &av1_reference_filters[reference_slot];
        ASSERT_TRUE(ParseAv1FrameObu(encoded.bytes, encoded.byte_length,
                                     temporal_layer_count,
                                     encoded.temporal_layer_index,
                                     primary_filter, &av1_picture, &av1_tile));
      } else {
        ASSERT_EQ(encoded.kind,
                  AMDF_MEDIA_ENCODED_REGION_AV1_TILE_GROUP_PAYLOAD);
        av1_picture.loop_filter = encoded.av1_result.loop_filter;
        av1_picture.quantization = encoded.av1_result.quantization;
        av1_picture.cdef = encoded.av1_result.cdef;
        av1_picture.segmentation = encoded.av1_result.segmentation;
        if (!key) {
          av1_picture.primary_reference = encoded.av1_result.primary_reference;
          std::copy(std::begin(encoded.av1_result.reference_indices),
                    std::end(encoded.av1_result.reference_indices),
                    av1_picture.active_reference_indices);
        }
      }
      ASSERT_EQ(av1_picture.frame_type, key ? 0u : 1u);
      ASSERT_EQ(av1_picture.order_hint, encoded.generation);
      ASSERT_EQ(av1_picture.primary_reference, key ? 7u : 0u);
      ASSERT_EQ(av1_picture.interpolation_filter, 4u);
      ASSERT_EQ(av1_picture.transform_mode, key ? 1u : 2u);
      if (!key) {
        ASSERT_TRUE(std::all_of(
            std::begin(av1_picture.active_reference_indices),
            std::end(av1_picture.active_reference_indices),
            [](uint8_t reference_index) { return reference_index == 0; }));
        av1_reference = {&references[reference_slot], reference_slot};
        av1_picture.reference_count = 1;
        av1_picture.references = &av1_reference;
        std::fill(std::begin(av1_picture.reference_map),
                  std::end(av1_picture.reference_map), uint8_t{0});
      }
      for (auto& unit_size :
           av1_picture.loop_filter.log2_restoration_unit_sizes) {
        unit_size = 5;
      }
    }
    const void* codec_picture = test_case.codec == AMDF_MEDIA_CODEC_H264
                                    ? static_cast<const void*>(&h264_picture)
                                : test_case.codec == AMDF_MEDIA_CODEC_HEVC
                                    ? static_cast<const void*>(&hevc_picture)
                                    : static_cast<const void*>(&av1_picture);
    amdf_media_image_activation_t activations[2] = {};
    uint32_t activation_count = 0;
    if (index == 0) {
      activations[activation_count++].after = output_image;
    }
    if (!reference_active[reconstruction_slot]) {
      activations[activation_count++].after =
          reference_images[reconstruction_slot];
      reference_active[reconstruction_slot] = true;
    }
    const amdf_media_decode_picture_info_t decode = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
        .structure_size = sizeof(decode),
        .next = codec_picture,
        .compressed = {compressed_memory, 0, encoded.byte_offset,
                       decode_compressed_byte_lengths[index]},
        .output = &output,
        .reference_output = &references[reconstruction_slot],
        .reference_slot_index = reconstruction_slot,
    };
    const amdf_media_command_prepare_info_t prepare = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
        .structure_size = sizeof(prepare),
        .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
        .activation_count = activation_count,
        .activations = activations,
        .decode = &decode,
    };
    const amdf_status_t prepare_status =
        media_->command_prepare(decode_commands_[index], &prepare);
    ASSERT_EQ(prepare_status, AMDF_STATUS_OK)
        << "picture=" << index
        << " domain=" << amdf_status_domain(prepare_status)
        << " code=" << amdf_status_code(prepare_status);
    reference_generations[reconstruction_slot] = encoded.generation;
    if (test_case.codec == AMDF_MEDIA_CODEC_AV1 && retained) {
      av1_reference_filters[reconstruction_slot] = av1_picture.loop_filter;
    }
  }

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
      output_address + layout.byte_offsets[0],
      results->device_address + 64,
  };
  const uint32_t final_generation = pictures.back().generation;
  const uint32_t constants[] = {
      0,
      0,
      final_generation,
      static_cast<uint32_t>(layout.byte_offsets[1] - layout.byte_offsets[0]),
      layout.pipe_bank_xors[0],
      layout.pipe_bank_xors[1],
      width,
      height,
      test_case.sample_bit_depth == 10 ? 1u : 0u,
      layout.linear,
      layout.row_pitches[0],
      layout.row_pitches[1],
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
  uint64_t point = 0;
  std::vector<uint64_t> decode_points;
  decode_points.reserve(decode_commands_.size());
  for (uint32_t index = 0; index < decode_commands_.size(); ++index) {
    const amdf_media_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command = decode_commands_[index],
        .wait_count = index == 0 ? (gpu_queue_ != nullptr ? 2u : 1u) : 0u,
        .waits = index == 0 ? decode_waits : nullptr,
    };
    point = 0;
    const auto status =
        media_->kernel_queue_submit(decode_queue_, &submit, &point);
    if (point != 0) {
      decode_point_ = point;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_NE(point, 0u);
    decode_points.push_back(point);
  }
  const amdf_kernel_queue_dependency_wait_t decoded = {decode_to_gpu_, point};
  if (gpu_queue_ == nullptr) {
    ASSERT_EQ(
        api_->kernel_queue_wait(decode_queue_, point, AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
  }
  ASSERT_NO_FATAL_FAILURE(
      SubmitGpu(consume, gpu_queue_ != nullptr ? &decoded : nullptr));
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(capture));
  if (gpu_queue_ != nullptr) {
    ASSERT_EQ(api_->kernel_queue_wait(gpu_queue_, gpu_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  amdf_kernel_queue_status_t progress = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(progress),
  };
  ASSERT_EQ(api_->kernel_queue_refresh_status(decode_queue_, &progress),
            AMDF_STATUS_OK);
  ASSERT_EQ(progress.terminal_status, AMDF_STATUS_OK);
  ASSERT_EQ(progress.retired_submission, decode_point_);
  for (uint32_t index = 0; index < decode_commands_.size(); ++index) {
    amdf_media_command_result_info_t result = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
        .structure_size = sizeof(result),
    };
    ASSERT_EQ(media_->command_query_result(decode_commands_[index], &result),
              AMDF_STATUS_OK);
    ASSERT_EQ(result.codec_result, configuration.feedback_byte_length != 0
                                       ? AMDF_MEDIA_CODEC_RESULT_SUCCEEDED
                                       : AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE);
    ASSERT_EQ(result.operation, AMDF_MEDIA_COMMAND_OPERATION_PICTURE);
    ASSERT_EQ(result.submission, decode_points[index]);
  }
  ASSERT_NO_FATAL_FAILURE(results->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  ASSERT_NO_FATAL_FAILURE(heap_copy->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  const auto* result_bytes = static_cast<const uint8_t*>(results->host.pointer);
  double psnr = 0.0;
  ASSERT_NO_FATAL_FAILURE(CheckRoundTripPixels(
      reinterpret_cast<const uint32_t*>(result_bytes + 64), width, height,
      test_case.sample_bit_depth == 10, final_generation, &psnr));
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
  for (uint64_t index = reference_offset +
                        reference_count * configuration.reference.byte_length;
       index < reference_heap_length; ++index) {
    ASSERT_EQ(heap_bytes[reference_copy_offset + index], 0xa5u);
  }
  for (uint64_t index = 0; index < feedback_length; ++index) {
    bool result_byte = false;
    for (uint32_t picture = 0; picture < pictures.size(); ++picture) {
      const uint64_t begin = feedback_offset + picture * feedback_stride;
      result_byte |=
          index >= begin && index < begin + configuration.feedback_byte_length;
    }
    if (!result_byte) {
      ASSERT_EQ(decode_feedback_[index], 0xa5u);
    }
  }
  ASSERT_EQ(std::memcmp(control_bytes + consume.byte_offset,
                        immutable_consumer.data(), immutable_consumer.size()),
            0);
  if (configuration.requires_begin_end != 0) {
    ASSERT_NO_FATAL_FAILURE(
        SubmitMediaControl(decode_queue_, decode_commands_[0],
                           AMDF_MEDIA_COMMAND_OPERATION_END, &decode_point_));
    decode_session_active_ = false;
  }
  uint64_t compressed_byte_length = 0;
  for (const auto& picture : pictures) {
    compressed_byte_length += picture.byte_length;
  }
  RecordProperty("round_trip_psnr_db", psnr);
  RecordProperty("round_trip_compressed_bytes", compressed_byte_length);
  RecordProperty("round_trip_picture_count", pictures.size());
  RecordProperty("round_trip_device_only_payload", 1);
}
