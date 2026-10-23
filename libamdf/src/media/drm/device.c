// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/device.h"

#include <sys/ioctl.h>
#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/media/drm/command.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/structure.h"

amdf_status_t amdf_media_drm_query(int descriptor, uint32_t kind,
                                   uint32_t parameter, void* data,
                                   uint32_t byte_length) {
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)data,
      .return_size = byte_length,
      .query = kind,
      .query_hw_ip = {.type = parameter},
  };
  return ioctl(descriptor, DRM_IOCTL_AMDGPU_INFO, &query) == 0
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

static void amdf_media_drm_device_query_info(const amdf_device_t* base,
                                             amdf_device_info_t* out_info) {
  out_info->id.words[0] =
      amdf_endpoint_get_cached_info(base->endpoint)->id.words[0];
  out_info->id.words[1] = (uintptr_t)base;
  out_info->reset_epoch = 1;
}

static amdf_status_t amdf_media_drm_device_query_reservation_profile(
    amdf_device_t* device, uint32_t ordinal,
    struct amdf_memory_reservation_profile_t* out_profile) {
  (void)device;
  (void)ordinal;
  (void)out_profile;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_media_drm_device_destroy(amdf_device_t* base) {
  amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)base;
  const amdf_status_t status = amdf_linux_file_close(&device->descriptor);
  const int result = pthread_mutex_destroy(&device->memory_mutex);
  return amdf_status_is_ok(status) && result != 0 ? amdf_linux_error(result)
                                                  : status;
}

static const amdf_device_vtable_t amdf_media_drm_device_operations = {
    .query_info = amdf_media_drm_device_query_info,
    .query_memory_reservation_profile =
        amdf_media_drm_device_query_reservation_profile,
    .query_memory_profile = amdf_media_drm_memory_profile,
    .memory_prepare = amdf_media_drm_memory_prepare,
    .memory_prepare_import = amdf_media_drm_memory_prepare_import,
    .destroy_native = amdf_media_drm_device_destroy,
};

// Implemented schema and native capability identity form one admission table.
// Discovery, enumeration, and exact configuration queries use the same rows.
static const struct {
  // Public codec schema implemented by this provider.
  amdf_media_codec_t codec;
  // Kernel engine carrying this codec's commands.
  uint32_t engine;
  // Index in the kernel's decode-capability array.
  uint32_t capability;
} amdf_media_drm_codecs[] = {
    {AMDF_MEDIA_CODEC_JPEG, AMDGPU_HW_IP_VCN_JPEG,
     AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_JPEG},
    {AMDF_MEDIA_CODEC_H264, AMDGPU_HW_IP_VCN_ENC,
     AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4_AVC},
    {AMDF_MEDIA_CODEC_HEVC, AMDGPU_HW_IP_VCN_ENC,
     AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_HEVC},
    {AMDF_MEDIA_CODEC_VP9, AMDGPU_HW_IP_VCN_ENC,
     AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VP9},
    {AMDF_MEDIA_CODEC_AV1, AMDGPU_HW_IP_VCN_ENC,
     AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_AV1},
};

enum {
  AMDF_MEDIA_DRM_CODEC_COUNT =
      sizeof(amdf_media_drm_codecs) / sizeof(amdf_media_drm_codecs[0]),
};

static amdf_status_t AMDF_CALL amdf_media_drm_enumerate_codecs(
    amdf_device_t* device, uint32_t ordinal, uint32_t capacity,
    amdf_media_codec_info_t* codecs, uint32_t* count) {
  const amdf_media_drm_device_t* native =
      (const amdf_media_drm_device_t*)device;
  uint32_t total = 0;
  amdf_media_codec_info_t available[AMDF_MEDIA_DRM_CODEC_COUNT];
  for (uint32_t i = 0; i < AMDF_MEDIA_DRM_CODEC_COUNT; ++i) {
    if (native->engines[ordinal].ip_type == amdf_media_drm_codecs[i].engine &&
        native->decode_limits.codec_info[amdf_media_drm_codecs[i].capability]
            .valid) {
      available[total++] = (amdf_media_codec_info_t){
          amdf_media_drm_codecs[i].codec, AMDF_MEDIA_DIRECTION_DECODE};
    }
  }
  *count = total;
  if (capacity != 0) {
    if (capacity < total) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
    for (uint32_t i = 0; i < total; ++i) {
      codecs[i] = available[i];
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_drm_configuration(
    amdf_media_drm_device_t* device, uint32_t family_ordinal,
    const amdf_media_configuration_t* configuration, const void* next,
    amdf_media_configuration_info_t* out_info) {
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
      .configuration = *configuration,
  };
  const uint32_t engine = device->engines[family_ordinal].ip_type;
  const struct drm_amdgpu_info_video_codec_info* limits = NULL;
  for (uint32_t i = 0; i < AMDF_MEDIA_DRM_CODEC_COUNT; ++i) {
    if (configuration->codec == amdf_media_drm_codecs[i].codec &&
        engine == amdf_media_drm_codecs[i].engine) {
      limits = &device->decode_limits
                    .codec_info[amdf_media_drm_codecs[i].capability];
      break;
    }
  }
  if (limits == NULL || !limits->valid ||
      configuration->direction != AMDF_MEDIA_DIRECTION_DECODE) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  const bool jpeg = configuration->codec == AMDF_MEDIA_CODEC_JPEG;
  const bool hevc = configuration->codec == AMDF_MEDIA_CODEC_HEVC;
  const bool vp9 = configuration->codec == AMDF_MEDIA_CODEC_VP9;
  const bool av1 = configuration->codec == AMDF_MEDIA_CODEC_AV1;
  amdf_media_pixel_format_t format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
  if (jpeg) {
    if (next != NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  } else if (av1) {
    const amdf_status_t status = amdf_structure_validate_input(
        next, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
        sizeof(amdf_media_av1_configuration_t));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_media_av1_configuration_t* syntax = next;
    if (syntax->profile != 0 ||
        (syntax->sample_bit_depth != 8 && syntax->sample_bit_depth != 10)) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    }
    format = syntax->sample_bit_depth == 10 ? AMDF_MEDIA_PIXEL_FORMAT_P010
                                            : AMDF_MEDIA_PIXEL_FORMAT_NV12;
  } else if (vp9) {
    const amdf_status_t status = amdf_structure_validate_input(
        next, AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION,
        sizeof(amdf_media_vp9_configuration_t));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_media_vp9_configuration_t* syntax = next;
    if (!((syntax->profile == AMDF_MEDIA_VP9_PROFILE_0 &&
           syntax->sample_bit_depth == 8) ||
          (syntax->profile == AMDF_MEDIA_VP9_PROFILE_2 &&
           syntax->sample_bit_depth == 10))) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    }
    format = syntax->sample_bit_depth == 10 ? AMDF_MEDIA_PIXEL_FORMAT_P010
                                            : AMDF_MEDIA_PIXEL_FORMAT_NV12;
  } else if (hevc) {
    const amdf_status_t status = amdf_structure_validate_input(
        next, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
        sizeof(amdf_media_hevc_configuration_t));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_media_hevc_configuration_t* syntax = next;
    if ((syntax->profile != AMDF_MEDIA_HEVC_PROFILE_MAIN &&
         syntax->profile != AMDF_MEDIA_HEVC_PROFILE_MAIN10) ||
        (syntax->sample_bit_depth != 8 && syntax->sample_bit_depth != 10) ||
        (syntax->profile == AMDF_MEDIA_HEVC_PROFILE_MAIN &&
         syntax->sample_bit_depth != 8)) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    }
    format = syntax->sample_bit_depth == 10 ? AMDF_MEDIA_PIXEL_FORMAT_P010
                                            : AMDF_MEDIA_PIXEL_FORMAT_NV12;
  } else {
    const amdf_status_t status = amdf_structure_validate_input(
        next, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
        sizeof(amdf_media_h264_configuration_t));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_media_h264_configuration_t* syntax = next;
    if (syntax->profile < AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE ||
        syntax->profile > AMDF_MEDIA_H264_PROFILE_HIGH ||
        syntax->level_idc > 52) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    }
  }
  if (info.reason == AMDF_MEDIA_CONFIGURATION_REASON_NONE) {
    if (configuration->format != format) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_FORMAT;
    } else if (jpeg ? configuration->reference_image_count != 0
                    : configuration->reference_image_count == 0 ||
                          configuration->reference_image_count >
                              (vp9 || av1 ? 9u
                               : hevc     ? 16u
                                          : 17u)) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE;
    } else if (configuration->coded_width < (vp9 || av1 ? 16u : 64u) ||
               configuration->coded_height < (vp9 || av1 ? 16u : 64u) ||
               ((configuration->coded_width | configuration->coded_height) &
                (jpeg || vp9 || av1 ? 1u
                 : hevc             ? 7u
                                    : 15u)) != 0 ||
               configuration->coded_width > limits->max_width ||
               configuration->coded_height > limits->max_height ||
               (!jpeg && (configuration->coded_width >
                              (hevc || vp9 || av1 ? 8192u : 4096u) ||
                          configuration->coded_height >
                              (hevc || vp9 || av1 ? 8192u : 4096u))) ||
               (uint64_t)configuration->coded_width *
                       configuration->coded_height >
                   limits->max_pixels_per_frame) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS;
    }
  }
  if (info.reason != AMDF_MEDIA_CONFIGURATION_REASON_NONE) {
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  amdf_media_image_description_t description = {
      .format = configuration->format,
      .width = configuration->coded_width,
      .height = configuration->coded_height,
      .array_layer_count = 1,
      .roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT,
  };
  amdf_status_t status = amdf_media_drm_image_requirements(
      &device->base.base, &description, &info.output);
  if (amdf_status_is_ok(status) && !jpeg) {
    description.roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE;
    if (vp9 || av1) {
      // VP9/AV1 reconstruction operates on 64-row superblocks in both
      // precisions. The returned storage shape stays distinct from exact coded
      // dimensions.
      description.height = (description.height + 63) & ~63u;
    }
    status = amdf_media_drm_image_requirements(&device->base.base, &description,
                                               &info.reference);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  info.support = AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED;
  info.requires_begin_end = !jpeg;
  info.image_heap_memory_scope =
      &device->base.base.provider_instance->system_memory_scope;
  info.compressed_memory_scope = info.image_heap_memory_scope;
  info.compressed_byte_offset_alignment = jpeg ? 128 : 256;
  info.compressed_byte_length_alignment = 128;
  *out_info = info;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL amdf_media_drm_query_configuration(
    amdf_device_t* device, const amdf_media_configuration_query_t* query,
    amdf_media_configuration_info_t* out_info) {
  return amdf_media_drm_configuration(
      (amdf_media_drm_device_t*)device, query->queue_family_ordinal,
      &query->configuration, query->next, out_info);
}

static const amdf_media_api_t amdf_media_drm_operations = {
    .structure_size = sizeof(amdf_media_api_t),
    .extension_version = AMDF_MEDIA_EXTENSION_VERSION_1,
    .device_enumerate_codecs = amdf_media_drm_enumerate_codecs,
    .device_query_configuration = amdf_media_drm_query_configuration,
    .device_query_image_requirements = amdf_media_drm_image_requirements,
    .kernel_queue_create = amdf_media_drm_queue_create,
    .image_create = amdf_media_drm_image_create,
    .image_query_info = amdf_media_drm_image_query_info,
    .image_destroy = amdf_media_drm_image_destroy,
    .session_create = amdf_media_drm_session_create,
    .session_query_info = amdf_media_drm_session_query_info,
    .command_create = amdf_media_drm_command_create,
    .command_prepare = amdf_media_drm_command_prepare,
    .kernel_queue_submit = amdf_media_drm_queue_submit,
    .command_query_result = amdf_media_drm_command_query_result,
    .command_destroy = amdf_media_drm_command_destroy,
    .session_destroy = amdf_media_drm_session_destroy,
};

static amdf_status_t amdf_media_drm_device_discover(
    amdf_media_drm_device_t* device, uint32_t* out_family_count) {
  struct drm_amdgpu_info_device info = {0};
  amdf_status_t status = amdf_media_drm_query(
      device->descriptor, AMDGPU_INFO_DEV_INFO, 0, &info, sizeof(info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 ||
      ((uint64_t)page_size & ((uint64_t)page_size - 1)) != 0 ||
      info.virtual_address_alignment > (uint64_t)page_size ||
      info.virtual_address_max >= (UINT64_C(1) << 48)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  device->page_size = (uint64_t)page_size;
  device->address_begin =
      (info.virtual_address_offset + device->page_size - 1) &
      ~(device->page_size - 1);
  if (device->address_begin == 0) {
    device->address_begin = device->page_size;
  }
  device->address_end =
      (info.virtual_address_max + 1) & ~(device->page_size - 1);
  if (device->address_begin >= device->address_end) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  struct drm_get_cap capability = {.capability = DRM_CAP_SYNCOBJ_TIMELINE};
  if (ioctl(device->descriptor, DRM_IOCTL_GET_CAP, &capability) != 0) {
    return amdf_linux_error(errno);
  }
  if (capability.value == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  status = amdf_media_drm_query(
      device->descriptor, AMDGPU_INFO_VIDEO_CAPS, AMDGPU_INFO_VIDEO_CAPS_DECODE,
      &device->decode_limits, sizeof(device->decode_limits));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t count = 0;
  const uint32_t selectors[] = {AMDGPU_HW_IP_VCN_JPEG, AMDGPU_HW_IP_VCN_ENC};
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < 2; ++i) {
    struct drm_amdgpu_info_hw_ip engine = {0};
    status = amdf_media_drm_query(device->descriptor, AMDGPU_INFO_HW_IP_INFO,
                                  selectors[i], &engine, sizeof(engine));
    if (!amdf_status_is_ok(status)) {
      break;
    }
    const uint32_t version = engine.ip_discovery_version;
    const uint32_t major = (version >> 16) & 255;
    const bool admitted =
        i == 0 ? major >= 2 && major <= 4 && version != 0x040003 : major == 4;
    bool codec_present = false;
    for (uint32_t codec = 0; codec < AMDF_MEDIA_DRM_CODEC_COUNT; ++codec) {
      codec_present |= amdf_media_drm_codecs[codec].engine == selectors[i] &&
                       device->decode_limits
                           .codec_info[amdf_media_drm_codecs[codec].capability]
                           .valid;
    }
    const uint32_t command_alignment = i == 0 ? 64 : 4;
    if (!admitted || !codec_present || engine.available_rings == 0 ||
        engine.ib_start_alignment > device->page_size ||
        engine.ib_size_alignment == 0 ||
        command_alignment % engine.ib_size_alignment != 0) {
      continue;
    }
    device->engines[count++] = (amdf_media_drm_engine_t){selectors[i], engine};
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_queue_family_info_t family = (amdf_queue_family_info_t){
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(family),
      .command_type = AMDF_QUEUE_COMMAND_TYPE_MEDIA,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .format_version = AMDF_MEDIA_EXTENSION_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_MEDIA,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .kernel_queue_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
  };
  for (uint32_t i = 0; i < count; ++i) {
    device->families[i] = family;
  }
  *out_family_count = count;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_provider_device_create(
    amdf_endpoint_t* endpoint,
    const amdf_media_device_create_info_t* create_info,
    amdf_device_t** out_device) {
  (void)create_info;
  if (amdf_endpoint_get_cached_info(endpoint)->engine_kind !=
      AMDF_ENGINE_KIND_GPU) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_allocator_t allocator = amdf_endpoint_host_allocator(endpoint);
  amdf_media_drm_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(allocator, sizeof(*device),
                  amdf_alignof(amdf_media_drm_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->base.base.host_allocator = allocator;
  device->base.operations = &amdf_media_drm_operations;
  device->descriptor = -1;
  const int initialized = pthread_mutex_init(&device->memory_mutex, NULL);
  if (initialized != 0) {
    amdf_free(allocator, device);
    return amdf_linux_error(initialized);
  }
  uint32_t family_count = 0;
  status = amdf_linux_endpoint_open_file(amdf_endpoint_get_platform(endpoint),
                                         &device->descriptor);
  if (amdf_status_is_ok(status)) {
    status = amdf_media_drm_device_discover(device, &family_count);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_device_initialize(
        &device->base.base, &amdf_media_drm_device_operations, endpoint,
        AMDF_EXTENSION_MEDIA, family_count, device->families);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup =
        amdf_media_drm_device_destroy(&device->base.base);
    amdf_free(allocator, device);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_device = &device->base.base;
  return AMDF_STATUS_OK;
}
