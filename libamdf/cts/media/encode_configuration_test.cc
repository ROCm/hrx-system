// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "amdf/gpu.h"
#include "amdf/media.h"
#include "amdf/media/av1.h"
#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

template <typename T>
void Initialize(T* value, amdf_structure_type_t type) {
  value->type = type;
  value->structure_size = sizeof(*value);
}

struct EncodeCase {
  // Codec selected by the configuration chain.
  amdf_media_codec_t codec;
  // Required native input and reconstruction format.
  amdf_media_pixel_format_t format;
  // Component precision used by codec-specific syntax.
  uint32_t sample_bit_depth;
};

class EncodeConfigurationTest : public ::testing::Test {
 protected:
  struct ConfigurationChain {
    // Direction-common rate-control and GOP configuration.
    amdf_media_encode_configuration_t encode = {};
    // H.264 profile configuration at the head of an H.264 chain.
    amdf_media_h264_configuration_t h264 = {};
    // H.264 encode tools following the common configuration.
    amdf_media_h264_encode_configuration_t h264_encode = {};
    // HEVC profile and precision at the head of an HEVC chain.
    amdf_media_hevc_configuration_t hevc = {};
    // HEVC encode geometry following the common configuration.
    amdf_media_hevc_encode_configuration_t hevc_encode = {};
    // AV1 profile and precision at the head of an AV1 chain.
    amdf_media_av1_configuration_t av1 = {};
    // AV1 encode tools following the common configuration.
    amdf_media_av1_encode_configuration_t av1_encode = {};

    const void* InitializeFor(const EncodeCase& test_case,
                              amdf_media_encode_rate_control_mode_t mode) {
      Initialize(&encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION);
      encode.intra_picture_period = 12;
      encode.inter_picture_period = 1;
      encode.rate_control.mode = mode;
      encode.rate_control.frame_rate_numerator = 30;
      encode.rate_control.frame_rate_denominator = 1;
      if (mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP) {
        encode.rate_control.control.cqp = {20, 20};
      } else {
        const uint32_t minimum_qp =
            test_case.codec == AMDF_MEDIA_CODEC_AV1 ? 8u : 1u;
        const uint32_t maximum_qp =
            test_case.codec == AMDF_MEDIA_CODEC_AV1 ? 255u : 51u;
        encode.rate_control.control.bit_rate = {
            minimum_qp,
            maximum_qp,
            0,
            1200000,
            mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR ? 1200000u
                                                            : 2400000u,
            1200000,
            600000,
        };
      }
      if (test_case.codec == AMDF_MEDIA_CODEC_H264) {
        Initialize(&h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION);
        Initialize(&h264_encode,
                   AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION);
        h264.next = &encode;
        h264.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
        h264.level_idc = 30;
        encode.next = &h264_encode;
        h264_encode.picture_order_count_type = 2;
        return &h264;
      }
      if (test_case.codec == AMDF_MEDIA_CODEC_HEVC) {
        Initialize(&hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION);
        Initialize(&hevc_encode,
                   AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_CONFIGURATION);
        hevc.next = &encode;
        hevc.profile = test_case.sample_bit_depth == 10
                           ? AMDF_MEDIA_HEVC_PROFILE_MAIN10
                           : AMDF_MEDIA_HEVC_PROFILE_MAIN;
        hevc.sample_bit_depth = test_case.sample_bit_depth;
        encode.next = &hevc_encode;
        hevc_encode.level_idc = 90;
        hevc_encode.tier = AMDF_MEDIA_HEVC_TIER_MAIN;
        hevc_encode.minimum_luma_coding_unit_log2 = 3;
        hevc_encode.maximum_luma_coding_unit_log2 = 6;
        hevc_encode.minimum_luma_transform_unit_log2 = 2;
        hevc_encode.maximum_luma_transform_unit_log2 = 5;
        hevc_encode.maximum_transform_hierarchy_depth_inter = 4;
        hevc_encode.maximum_transform_hierarchy_depth_intra = 4;
        return &hevc;
      }
      Initialize(&av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION);
      Initialize(&av1_encode,
                 AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION);
      av1.next = &encode;
      av1.profile = 0;
      av1.sample_bit_depth = test_case.sample_bit_depth;
      encode.next = &av1_encode;
      av1_encode.level = 0;
      av1_encode.tier = AMDF_MEDIA_AV1_TIER_MAIN;
      av1_encode.feature_flags = AMDF_MEDIA_AV1_ENCODE_FEATURE_ORDER_HINT |
                                 AMDF_MEDIA_AV1_ENCODE_FEATURE_CDEF;
      av1_encode.order_hint_bit_count = 4;
      return &av1;
    }
  };

  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    const void* extension = nullptr;
    auto status = api_->query_extension(
        AMDF_EXTENSION_MEDIA, AMDF_MEDIA_EXTENSION_VERSION_2,
        AMDF_MEDIA_EXTENSION_VERSION_LATEST, &extension);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "media encode extension is not enabled";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    media_ = static_cast<const amdf_media_api_t*>(extension);

    amdf_instance_t* instance = nullptr;
    ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance), AMDF_STATUS_OK);
    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> summaries(count);
    ASSERT_EQ(
        api_->endpoint_enumerate(instance, count, summaries.data(), &count),
        AMDF_STATUS_OK);
    amdf_endpoint_t* endpoint = nullptr;
    for (const auto& summary : summaries) {
      if (summary.engine_kind == AMDF_ENGINE_KIND_GPU &&
          GetCtsDeviceCache().IsGpuEndpointSelected(summary.id)) {
        ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                  AMDF_STATUS_OK);
        break;
      }
    }
    if (endpoint == nullptr) {
      GTEST_SKIP() << "no selected GPU endpoint";
    }
    const void* gpu_extension = nullptr;
    ASSERT_EQ(api_->query_extension(
                  AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                  AMDF_GPU_EXTENSION_VERSION_LATEST, &gpu_extension),
              AMDF_STATUS_OK);
    const auto* gpu = static_cast<const amdf_gpu_api_t*>(gpu_extension);
    gpu_info_.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
    gpu_info_.structure_size = sizeof(gpu_info_);
    ASSERT_EQ(gpu->endpoint_query_info(endpoint, &gpu_info_), AMDF_STATUS_OK);
    const amdf_media_device_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create),
    };
    status = media_->device_create(endpoint, &create, &device_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native media provider is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);

    amdf_device_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_DEVICE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->device_query_info(device_, &info), AMDF_STATUS_OK);
    encode_family_ = info.queue_family_count;
    for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
      ASSERT_EQ(
          media_->device_enumerate_codecs(device_, ordinal, 0, nullptr, &count),
          AMDF_STATUS_OK);
      std::vector<amdf_media_codec_info_t> codecs(count);
      ASSERT_EQ(media_->device_enumerate_codecs(device_, ordinal, count,
                                                codecs.data(), &count),
                AMDF_STATUS_OK);
      const auto supports = [&](amdf_media_codec_t codec) {
        return std::any_of(
            codecs.begin(), codecs.end(), [&](const auto& candidate) {
              return candidate.codec == codec &&
                     candidate.direction == AMDF_MEDIA_DIRECTION_ENCODE;
            });
      };
      EXPECT_FALSE(supports(AMDF_MEDIA_CODEC_JPEG));
      EXPECT_FALSE(supports(AMDF_MEDIA_CODEC_VP9));
      if (encode_family_ == info.queue_family_count &&
          (supports(AMDF_MEDIA_CODEC_H264) || supports(AMDF_MEDIA_CODEC_HEVC) ||
           supports(AMDF_MEDIA_CODEC_AV1))) {
        encode_family_ = ordinal;
        for (const auto& candidate : codecs) {
          if (candidate.direction == AMDF_MEDIA_DIRECTION_ENCODE) {
            encode_codecs_.push_back(candidate.codec);
          }
        }
      }
    }
    if (encode_family_ == info.queue_family_count) {
      GTEST_SKIP() << "no native encode family";
    }
  }

  void TearDown() override {
    if (queue_ != nullptr) {
      ASSERT_EQ(api_->kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    if (device_ != nullptr) {
      ASSERT_EQ(api_->device_destroy(device_), AMDF_STATUS_OK);
    }
  }

  amdf_media_configuration_info_t Query(
      const EncodeCase& test_case, amdf_media_encode_rate_control_mode_t mode,
      ConfigurationChain* chain, uint32_t coded_width = 320,
      uint32_t coded_height = 192) {
    const amdf_media_configuration_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
        .structure_size = sizeof(query),
        .next = chain->InitializeFor(test_case, mode),
        .queue_family_ordinal = encode_family_,
        .configuration =
            {
                .codec = test_case.codec,
                .direction = AMDF_MEDIA_DIRECTION_ENCODE,
                .format = test_case.format,
                .coded_width = coded_width,
                .coded_height = coded_height,
                .reference_image_count = 2,
            },
    };
    amdf_media_configuration_info_t result = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
        .structure_size = sizeof(result),
    };
    EXPECT_EQ(media_->device_query_configuration(device_, &query, &result),
              AMDF_STATUS_OK);
    return result;
  }

  bool Supports(amdf_media_codec_t codec) const {
    return std::find(encode_codecs_.begin(), encode_codecs_.end(), codec) !=
           encode_codecs_.end();
  }

  const amdf_api_t* api_ = nullptr;
  const amdf_media_api_t* media_ = nullptr;
  amdf_device_t* device_ = nullptr;
  amdf_kernel_queue_t* queue_ = nullptr;
  uint32_t encode_family_ = 0;
  // Encode codecs advertised by encode_family_.
  std::vector<amdf_media_codec_t> encode_codecs_;
  // Compiler-visible identity of the media endpoint's GPU.
  amdf_gpu_endpoint_info_t gpu_info_ = {};
};

TEST_F(EncodeConfigurationTest, ExactAdvertisedMatrixAndStorageContracts) {
  static constexpr std::array<EncodeCase, 5> kCases = {{
      {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
      {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
      {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
      {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
      {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
  }};
  static constexpr std::array<amdf_media_encode_rate_control_mode_t, 3>
      kRateControlModes = {{
          AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
          AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
          AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
      }};
  uint32_t tested_case_count = 0;
  for (const auto& test_case : kCases) {
    if (!Supports(test_case.codec)) {
      continue;
    }
    ++tested_case_count;
    amdf_media_configuration_info_t baseline = {};
    for (const auto mode : kRateControlModes) {
      ConfigurationChain chain;
      const auto result = Query(test_case, mode, &chain);
      ASSERT_EQ(result.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED)
          << "codec=" << test_case.codec
          << " depth=" << test_case.sample_bit_depth << " rate=" << mode;
      EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_NONE);
      EXPECT_EQ(result.configuration.codec, test_case.codec);
      EXPECT_EQ(result.configuration.direction, AMDF_MEDIA_DIRECTION_ENCODE);
      EXPECT_EQ(result.configuration.format, test_case.format);
      EXPECT_EQ(result.output.byte_length, 0u);
      EXPECT_EQ(result.input.description.format, test_case.format);
      EXPECT_EQ(result.input.description.width, 320u);
      EXPECT_EQ(result.input.description.height, 192u);
      EXPECT_EQ(result.input.description.array_layer_count, 1u);
      EXPECT_EQ(result.input.description.roles,
                AMDF_MEDIA_IMAGE_ROLE_ENCODE_INPUT);
      EXPECT_EQ(result.reference.description.format, test_case.format);
      EXPECT_EQ(result.reference.description.width, 320u);
      EXPECT_EQ(result.reference.description.height, 192u);
      EXPECT_EQ(result.reference.description.array_layer_count, 2u);
      EXPECT_EQ(result.reference.description.roles,
                AMDF_MEDIA_IMAGE_ROLE_ENCODE_RECONSTRUCTION);
      EXPECT_GT(result.input.byte_length, 0u);
      EXPECT_GE(result.reference.byte_length, 2 * result.input.byte_length);
      EXPECT_NE(result.reference.byte_offset_alignment, 0u);
      EXPECT_EQ(result.reference.byte_offset_alignment &
                    (result.reference.byte_offset_alignment - 1),
                0u);
      EXPECT_NE(result.image_heap_memory_scope, nullptr);
      EXPECT_NE(result.compressed_memory_scope, nullptr);
      EXPECT_NE(result.feedback_memory_scope, nullptr);
      EXPECT_NE(result.compressed_byte_offset_alignment, 0u);
      EXPECT_EQ(result.compressed_byte_offset_alignment &
                    (result.compressed_byte_offset_alignment - 1),
                0u);
      EXPECT_NE(result.compressed_byte_length_alignment, 0u);
      EXPECT_EQ(result.minimum_compressed_byte_length,
                test_case.sample_bit_depth == 10 ? 294912u : 147456u);
      EXPECT_GT(result.feedback_byte_length, 0u);
      EXPECT_NE(result.feedback_byte_offset_alignment, 0u);
#ifndef _WIN32
      const uint64_t pixel_bytes =
          test_case.sample_bit_depth == 10 ? 786432u : 393216u;
      uint64_t codec_state_bytes = gpu_info_.gfx_ip.major == 12 ? 2048u : 0u;
      if (test_case.codec == AMDF_MEDIA_CODEC_AV1) {
        codec_state_bytes += 995328u;
      }
      EXPECT_EQ(result.reference.byte_length, pixel_bytes + codec_state_bytes);
      EXPECT_EQ(result.feedback_byte_length, 40u);
#endif  // _WIN32
      if (mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP) {
        baseline = result;
      } else {
        EXPECT_EQ(result.input.byte_length, baseline.input.byte_length);
        EXPECT_EQ(result.reference.byte_length, baseline.reference.byte_length);
        EXPECT_EQ(result.minimum_compressed_byte_length,
                  baseline.minimum_compressed_byte_length);
        EXPECT_EQ(result.feedback_byte_length, baseline.feedback_byte_length);
      }
    }
  }
  EXPECT_GT(tested_case_count, 0u);
}

TEST_F(EncodeConfigurationTest, RejectsUnrepresentableAv1CodedGeometry) {
  if (!Supports(AMDF_MEDIA_CODEC_AV1)) {
    GTEST_SKIP() << "AV1 encode is not advertised";
  }
  const EncodeCase av1_case = {AMDF_MEDIA_CODEC_AV1,
                               AMDF_MEDIA_PIXEL_FORMAT_NV12, 8};
  ConfigurationChain chain;
#if defined(_WIN32)
  const auto result = Query(av1_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
                            &chain, 320, 180);
#else
  const auto result = Query(av1_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
                            &chain, 322, 192);
#endif  // _WIN32
  EXPECT_EQ(result.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);
}

TEST_F(EncodeConfigurationTest, UnsupportedTuplesHaveSpecificReasons) {
  if (!Supports(AMDF_MEDIA_CODEC_H264)) {
    GTEST_SKIP() << "H.264 encode is not advertised";
  }
  const EncodeCase h264_case = {AMDF_MEDIA_CODEC_H264,
                                AMDF_MEDIA_PIXEL_FORMAT_NV12, 8};
  ConfigurationChain chain;
  const void* codec_configuration =
      chain.InitializeFor(h264_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP);
  amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = codec_configuration,
      .queue_family_ordinal = encode_family_,
      .configuration =
          {
              .codec = AMDF_MEDIA_CODEC_H264,
              .direction = AMDF_MEDIA_DIRECTION_ENCODE,
              .format = AMDF_MEDIA_PIXEL_FORMAT_P010,
              .coded_width = 320,
              .coded_height = 192,
              .reference_image_count = 2,
          },
  };
  amdf_media_configuration_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(result),
  };
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT);

  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
  query.configuration.reference_image_count = 1;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE);

  query.configuration.reference_image_count = 2;
  query.configuration.coded_width = 321;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);

  query.configuration.coded_width = 320;
  chain.h264.profile = AMDF_MEDIA_H264_PROFILE_HIGH;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

  chain.h264.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
  chain.h264.level_idc = 14;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

  codec_configuration =
      chain.InitializeFor(h264_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR);
  query.next = codec_configuration;
  chain.encode.rate_control.frame_rate_numerator = 0;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

  if (Supports(AMDF_MEDIA_CODEC_HEVC)) {
    const EncodeCase hevc_case = {AMDF_MEDIA_CODEC_HEVC,
                                  AMDF_MEDIA_PIXEL_FORMAT_NV12, 8};
    query.next =
        chain.InitializeFor(hevc_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP);
    query.configuration.codec = AMDF_MEDIA_CODEC_HEVC;
    chain.hevc_encode.level_idc = 91;
    ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

    chain.hevc_encode.level_idc = 90;
    chain.hevc_encode.tier = AMDF_MEDIA_HEVC_TIER_HIGH;
    ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  }

  if (Supports(AMDF_MEDIA_CODEC_AV1)) {
    const EncodeCase av1_case = {AMDF_MEDIA_CODEC_AV1,
                                 AMDF_MEDIA_PIXEL_FORMAT_NV12, 8};
    query.next =
        chain.InitializeFor(av1_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP);
    query.configuration.codec = AMDF_MEDIA_CODEC_AV1;
    chain.av1_encode.level = 24;
    ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

    chain.av1_encode.level = 7;
    chain.av1_encode.tier = AMDF_MEDIA_AV1_TIER_HIGH;
    ASSERT_EQ(media_->device_query_configuration(device_, &query, &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(result.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  }
}

TEST_F(EncodeConfigurationTest, SessionMatchesProviderExecutionSupport) {
  if (!Supports(AMDF_MEDIA_CODEC_H264)) {
    GTEST_SKIP() << "H.264 encode is not advertised";
  }
  const EncodeCase test_case = {AMDF_MEDIA_CODEC_H264,
                                AMDF_MEDIA_PIXEL_FORMAT_NV12, 8};
  ConfigurationChain chain;
  const auto requirements =
      Query(test_case, AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP, &chain);
  ASSERT_EQ(requirements.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
  const amdf_media_kernel_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = encode_family_,
      .maximum_pending_submission_count = 1,
  };
  ASSERT_EQ(media_->kernel_queue_create(device_, &queue_create, &queue_),
            AMDF_STATUS_OK);
  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = &chain.h264,
      .configuration = requirements.configuration,
  };
  amdf_media_session_t* session = nullptr;
  ASSERT_EQ(media_->session_create(queue_, &session_create, &session),
            AMDF_STATUS_OK);
  ASSERT_NE(session, nullptr);
  ASSERT_EQ(media_->session_destroy(session), AMDF_STATUS_OK);
}

}  // namespace
