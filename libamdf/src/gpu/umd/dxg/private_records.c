// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This implementation was independently derived from publicly downloadable
// ROCm artifacts and open source, then qualified against real /dev/dxg driver
// responses. No private header or documentation is needed to reproduce it. The
// key inputs were:
//
// - librocdxg v1.2.2's shipped thunk-proxy archive and public headers:
//   https://github.com/ROCm/librocdxg/tree/v1.2.2/shared
// - WKMI's public interface and matching shipped Linux and Windows archives at
//   rocm-systems commit 6b7847ef03b76c004f57b8cf40cdef9621ee6f53:
//   https://github.com/ROCm/rocm-systems/tree/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/shared/amdgpu-windows-interop/wkmi
// - the public ROCr topology implementation and GPU family identifiers:
//   https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/libhsakmt/src/topology.c
// - the open Linux KFD policy and DXG queue use documented by:
//   https://github.com/ROCm/rocm-systems/pull/10573
// - the public PAL scratch limits for GFX9 through GFX12 at commit
//   c5e800072a32f68b6ccc4422936d96167c6e0728:
//   https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9Chip.h
//   https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/gfx12Chip.h

#include "libamdf/src/gpu/umd/dxg/private_records.h"

#include <stdbool.h>
#include <string.h>

enum {
  AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH = 0x960,
  AMDF_DXG_PRIVATE_INITIAL_CHAIN_BYTE_LENGTH = 0xBC0,
  AMDF_DXG_PRIVATE_BASE_ADAPTER_BYTE_LENGTH = 0x260,
  AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH = 0x3A00,
  AMDF_DXG_PRIVATE_CHAIN_CHECKSUM_OFFSET = 0x8AC,
};

static uint16_t amdf_dxg_private_load_u16(const void* data) {
  uint16_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static uint32_t amdf_dxg_private_load_u32(const void* data) {
  uint32_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static uint64_t amdf_dxg_private_load_u64(const void* data) {
  uint64_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static void amdf_dxg_private_store_u32(void* data, uint32_t value) {
  memcpy(data, &value, sizeof(value));
}

static void amdf_dxg_private_store_u64(void* data, uint64_t value) {
  memcpy(data, &value, sizeof(value));
}

static uint32_t amdf_dxg_private_popcount_u16(uint16_t value) {
  uint32_t count = 0;
  while (value != 0) {
    value &= (uint16_t)(value - 1);
    ++count;
  }
  return count;
}

static uint32_t amdf_dxg_private_chain_checksum(const uint8_t* header) {
  uint32_t checksum = 0;
  for (uint32_t i = 0;
       i < AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH / sizeof(uint32_t); ++i) {
    uint32_t word = amdf_dxg_private_load_u32(header + i * sizeof(uint32_t));
    if (i * sizeof(uint32_t) == AMDF_DXG_PRIVATE_CHAIN_CHECKSUM_OFFSET) {
      word = 0;
    }
    checksum += word ^ i;
  }
  return checksum;
}

static bool amdf_dxg_private_chain_header_is_valid(const uint8_t* header) {
  return amdf_dxg_private_load_u32(header +
                                   AMDF_DXG_PRIVATE_CHAIN_CHECKSUM_OFFSET) ==
         amdf_dxg_private_chain_checksum(header);
}

static amdf_dxg_private_result_t amdf_dxg_private_issue_query(
    const amdf_dxg_private_adapter_query_t* query, uint32_t byte_length,
    uint32_t* out_native_status) {
  memset(query->scratch, 0, byte_length);
  const uint32_t native_status = query->query(
      query->query_context, query->adapter_handle, query->scratch, byte_length);
  if (native_status != 0) {
    *out_native_status = native_status;
    return AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE;
  }
  return AMDF_DXG_PRIVATE_RESULT_SUCCESS;
}

typedef struct amdf_dxg_private_gfxip_t {
  int32_t major;
  int32_t minor;
  int32_t stepping;
} amdf_dxg_private_gfxip_t;

static bool amdf_dxg_private_decode_fallback_gfxip(
    uint32_t family, uint32_t revision, amdf_dxg_private_gfxip_t* out_gfxip) {
  amdf_dxg_private_gfxip_t gfxip = {0};
  switch (family) {
    case 0x8F:  // AMDGPU_FAMILY_NV
      gfxip.major = 10;
      if (revision <= 9) {
        gfxip.minor = 1;
        gfxip.stepping = 0;
      } else if (revision <= 19) {
        gfxip.minor = 1;
        gfxip.stepping = 1;
      } else if (revision <= 39) {
        gfxip.minor = 1;
        gfxip.stepping = 2;
      } else if (revision <= 49 || (revision >= 192 && revision <= 199)) {
        gfxip.minor = 3;
        gfxip.stepping = 0;
      } else if (revision <= 59 || (revision >= 200 && revision <= 207)) {
        gfxip.minor = 3;
        gfxip.stepping = 1;
      } else if (revision <= 69) {
        gfxip.minor = 3;
        gfxip.stepping = 2;
      } else if (revision <= 191) {
        gfxip.minor = 3;
        gfxip.stepping = 3;
      } else {
        return false;
      }
      break;
    case 0x90:  // AMDGPU_FAMILY_VGH
    case 0x92:  // AMDGPU_FAMILY_YC
      if (revision == 0 || revision == 255) {
        return false;
      }
      gfxip = (amdf_dxg_private_gfxip_t){10, 3, 1};
      break;
    case 0x95:  // AMDGPU_FAMILY_GC_10_3_6
      if (revision == 0 || revision == 255) {
        return false;
      }
      gfxip = (amdf_dxg_private_gfxip_t){10, 3, revision < 32 ? 1 : 6};
      break;
    case 0x97:  // AMDGPU_FAMILY_GC_10_3_7
      if (revision == 0 || revision == 255) {
        return false;
      }
      gfxip = (amdf_dxg_private_gfxip_t){10, 3, 2};
      break;
    case 0x91:  // AMDGPU_FAMILY_GC_11_0_0
      gfxip.major = 11;
      gfxip.minor = 0;
      if (revision <= 15) {
        gfxip.stepping = 0;
      } else if (revision <= 31) {
        gfxip.stepping = 2;
      } else if (revision <= 254) {
        gfxip.stepping = 1;
      } else {
        return false;
      }
      break;
    case 0x94:  // AMDGPU_FAMILY_GC_11_0_1
      if (revision == 0 || revision == 255) {
        return false;
      }
      gfxip = (amdf_dxg_private_gfxip_t){11, 0, 1};
      break;
    case 0x96:  // AMDGPU_FAMILY_GC_11_5_0
      gfxip.major = 11;
      gfxip.minor = 5;
      if ((revision >= 1 && revision <= 25) ||
          (revision >= 64 && revision <= 73) ||
          (revision >= 80 && revision <= 191)) {
        gfxip.stepping = 0;
      } else if (revision >= 192 && revision <= 199) {
        gfxip.stepping = 1;
      } else {
        return false;
      }
      break;
    case 0x98:  // AMDGPU_FAMILY_GC_12_0_0
      gfxip.major = 12;
      gfxip.minor = 0;
      if (revision <= 31 || (revision >= 64 && revision <= 79)) {
        gfxip.stepping = 0;
      } else if (revision <= 254) {
        gfxip.stepping = 1;
      } else {
        return false;
      }
      break;
    default:
      return false;
  }
  *out_gfxip = gfxip;
  return true;
}

static bool amdf_dxg_private_decode_gfxip(const uint8_t* extended_record,
                                          uint32_t family,
                                          amdf_dxg_private_gfxip_t* out_gfxip) {
  const uint32_t encoded_gfxip =
      amdf_dxg_private_load_u32(extended_record + 0xAF4);
  if (encoded_gfxip == 0) {
    return amdf_dxg_private_decode_fallback_gfxip(
        family, amdf_dxg_private_load_u32(extended_record + 0xA08), out_gfxip);
  }

  amdf_dxg_private_gfxip_t gfxip = {
      .major = (int32_t)(encoded_gfxip >> 16),
      .minor = (int32_t)((encoded_gfxip >> 8) & 0xFF),
      .stepping = (int32_t)(encoded_gfxip & 0xFF),
  };
  if (gfxip.major == 13 && gfxip.minor == 0 && gfxip.stepping == 1) {
    gfxip.minor = 1;
    gfxip.stepping = 0;
  } else if (gfxip.stepping == 3) {
    gfxip.stepping = 2;
  }
  if (gfxip.major < 9 || gfxip.major > 13) {
    return false;
  }
  *out_gfxip = gfxip;
  return true;
}

static int32_t amdf_dxg_private_find_scheduler_ordinal(
    const uint8_t* scheduler_ids, uint8_t scheduler_id) {
  for (uint32_t i = 0; i < AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT; ++i) {
    if (scheduler_ids[i] == scheduler_id) {
      return (int32_t)i;
    }
  }
  return -1;
}

static uint64_t amdf_dxg_private_hardware_scheduling_mask(
    const uint8_t* extended_record, const uint8_t* scheduler_ids) {
  if ((extended_record[0x2E03] & 1u) == 0) {
    return 0;
  }

  uint64_t represented_nodes = 0;
  static const uint8_t scheduler_ids_with_hws[] = {0, 5, 4, 7};
  for (size_t i = 0; i < sizeof(scheduler_ids_with_hws); ++i) {
    const int32_t ordinal = amdf_dxg_private_find_scheduler_ordinal(
        scheduler_ids, scheduler_ids_with_hws[i]);
    if (ordinal >= 0) {
      represented_nodes |= UINT64_C(1) << ordinal;
    }
  }
  return represented_nodes &
         amdf_dxg_private_load_u64(extended_record + 0x35C4);
}

static amdf_dxg_private_result_t amdf_dxg_private_decode_adapter(
    uint32_t family, uint32_t supports_wddm2, uint32_t is_discrete,
    const uint8_t* extended_record, amdf_dxg_private_adapter_info_t* out_info) {
  if (!supports_wddm2) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }

  amdf_dxg_private_gfxip_t gfxip;
  if (!amdf_dxg_private_decode_gfxip(extended_record, family, &gfxip)) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }

  uint64_t compute_unit_count = 0;
  uint64_t shader_engine_count =
      amdf_dxg_private_load_u32(extended_record + 0xC60);
  uint32_t xcc_count = 1;
  if (shader_engine_count == 0) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }
  if (gfxip.major == 9) {
    const uint32_t shader_array_count =
        (uint32_t)(amdf_dxg_private_load_u64(extended_record + 0xC64) >> 32);
    const uint32_t compute_units_per_shader_array =
        amdf_dxg_private_load_u32(extended_record + 0xC6C);
    if (shader_array_count == 0 || compute_units_per_shader_array == 0) {
      return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
    }
    compute_unit_count = (uint64_t)shader_array_count * shader_engine_count *
                         compute_units_per_shader_array;
    if (gfxip.minor == 4) {
      xcc_count = 8;
      compute_unit_count *= xcc_count;
      shader_engine_count *= xcc_count;
    }
  } else {
    const uint32_t shader_array_count =
        amdf_dxg_private_load_u32(extended_record + 0xC68);
    if ((extended_record[0xC46] & 1u) == 0 || shader_array_count == 0 ||
        shader_array_count > 2 ||
        shader_engine_count > AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT) {
      return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
    }
    for (uint32_t i = 0; i < shader_engine_count; ++i) {
      compute_unit_count += amdf_dxg_private_popcount_u16(
          amdf_dxg_private_load_u16(extended_record + 0xD04 + i * 4));
      if (shader_array_count == 2) {
        compute_unit_count += amdf_dxg_private_popcount_u16(
            amdf_dxg_private_load_u16(extended_record + 0xD06 + i * 4));
      }
    }
    compute_unit_count *= 2;
    if (gfxip.major > 12 && (extended_record[0xC47] & 2u) != 0) {
      compute_unit_count = 0;
      for (uint32_t i = 0; i < 8; ++i) {
        compute_unit_count += extended_record[0xCEC + i];
      }
      compute_unit_count *= 2;
    }
  }
  if (compute_unit_count == 0 || compute_unit_count > UINT32_MAX ||
      shader_engine_count > UINT32_MAX) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }

  amdf_dxg_private_adapter_info_t info;
  memset(&info, 0, sizeof(info));
  info.asic_revision = amdf_dxg_private_load_u32(extended_record + 0xA08);
  info.gfx_ip_major = gfxip.major;
  info.gfx_ip_minor = gfxip.minor;
  info.gfx_ip_stepping = gfxip.stepping;
  info.is_discrete = is_discrete;
  info.wavefront_size = amdf_dxg_private_load_u32(extended_record + 0xC5C);
  info.compute_unit_count = (uint32_t)compute_unit_count;
  info.maximum_wave_count_per_compute_unit = gfxip.major == 9 ? 40 : 32;
  info.maximum_scratch_wave_count_per_compute_unit = 32;
  info.local_data_share_byte_length = 64 * 1024;
  info.shader_engine_count = (uint32_t)shader_engine_count;
  info.xcc_count = xcc_count;
  info.supports_firmware_state_shadowing = (extended_record[0xC46] >> 5) & 1u;
  info.supports_platform_atomics =
      !is_discrete || ((extended_record[0x208] >> 5) & 1u);
  info.user_queue_byte_length =
      amdf_dxg_private_load_u32(extended_record + 0x30E8);
  const uint32_t native_queue_bits = extended_record[0x35C0];
  info.pm4_scheduler_id = (native_queue_bits & 0x14u) == 0x14u ? 6u : 5u;
  info.native_queue_flags =
      ((native_queue_bits & (1u << 2)) != 0 ? AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL
                                            : 0) |
      ((native_queue_bits & (1u << 5)) != 0 ? AMDF_DXG_PRIVATE_NATIVE_QUEUE_SDMA
                                            : 0);
  info.invisible_local_heap_byte_length =
      amdf_dxg_private_load_u64(extended_record + 0xA38);
  memcpy(info.scheduler_ids, extended_record + 0x89C,
         sizeof(info.scheduler_ids));
  info.hardware_scheduling_node_mask =
      amdf_dxg_private_hardware_scheduling_mask(extended_record,
                                                info.scheduler_ids);
  *out_info = info;
  return AMDF_DXG_PRIVATE_RESULT_SUCCESS;
}

static bool amdf_dxg_private_try_make_queue_profile(
    const amdf_dxg_private_adapter_info_t* info, uint8_t scheduler_id,
    bool require_hardware_scheduling,
    amdf_dxg_private_queue_profile_t* out_profile) {
  uint32_t node_ordinal = 0;
  while (node_ordinal < AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT &&
         info->scheduler_ids[node_ordinal] != scheduler_id) {
    ++node_ordinal;
  }
  if (node_ordinal == AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT) {
    return false;
  }
  const bool uses_hardware_scheduling =
      ((info->hardware_scheduling_node_mask >> node_ordinal) & 1u) != 0;
  if (require_hardware_scheduling && !uses_hardware_scheduling) {
    return false;
  }
  *out_profile = (amdf_dxg_private_queue_profile_t){
      .scheduler_id = scheduler_id,
      .node_ordinal = node_ordinal,
      .is_supported = true,
      .uses_hardware_scheduling = uses_hardware_scheduling,
  };
  return true;
}

void amdf_dxg_private_initialize_queue_profiles(
    const amdf_dxg_private_adapter_info_t* info,
    amdf_dxg_private_queue_profile_t* out_pm4_profile,
    amdf_dxg_private_queue_profile_t* out_sdma_profile) {
  memset(out_pm4_profile, 0, sizeof(*out_pm4_profile));
  memset(out_sdma_profile, 0, sizeof(*out_sdma_profile));
  amdf_dxg_private_try_make_queue_profile(info, info->pm4_scheduler_id,
                                          /*require_hardware_scheduling=*/true,
                                          out_pm4_profile);

  for (uint32_t i = 0; i < AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT; ++i) {
    const uint8_t scheduler_id = info->scheduler_ids[i];
    if ((scheduler_id == 4 || scheduler_id == 7) &&
        amdf_dxg_private_try_make_queue_profile(
            info, scheduler_id, /*require_hardware_scheduling=*/true,
            out_sdma_profile)) {
      return;
    }
  }
  // A nonzero user-queue size means the driver selected persistent user-mode
  // SDMA rings, which the immutable command-stream API does not represent.
  if (info->user_queue_byte_length != 0) {
    return;
  }
  for (uint32_t i = 0; i < AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT; ++i) {
    const uint8_t scheduler_id = info->scheduler_ids[i];
    if ((scheduler_id == 4 || scheduler_id == 7) &&
        amdf_dxg_private_try_make_queue_profile(
            info, scheduler_id, /*require_hardware_scheduling=*/false,
            out_sdma_profile)) {
      return;
    }
  }
}

amdf_dxg_private_result_t amdf_dxg_private_query_linked_adapter(
    const amdf_dxg_private_adapter_query_t* query,
    amdf_dxg_private_adapter_info_t* out_info, uint32_t* out_native_status) {
  *out_native_status = 0;
  uint8_t* scratch = (uint8_t*)query->scratch;
  amdf_dxg_private_result_t result = amdf_dxg_private_issue_query(
      query, AMDF_DXG_PRIVATE_INITIAL_CHAIN_BYTE_LENGTH, out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  if (!amdf_dxg_private_chain_header_is_valid(scratch)) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }

  const uint32_t physical_adapter_count =
      amdf_dxg_private_load_u32(scratch + 0x08);
  if (physical_adapter_count == 0 ||
      physical_adapter_count > AMDF_DXG_PRIVATE_MAX_PHYSICAL_ADAPTER_COUNT) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }
  const uint32_t selected_index = query->physical_adapter_index;
  if (selected_index >= physical_adapter_count) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }
  const uint32_t vendor_id =
      amdf_dxg_private_load_u16(scratch + 0x0C + selected_index * 2);
  if (vendor_id != 0x1002 && vendor_id != 0x1022) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }
  const uint32_t is_discrete =
      amdf_dxg_private_load_u32(scratch + 0x2C + selected_index * 4) == 0;

  const uint8_t* base_record =
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH;
  if (selected_index != 0) {
    const uint32_t base_query_byte_length =
        AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
        physical_adapter_count * AMDF_DXG_PRIVATE_BASE_ADAPTER_BYTE_LENGTH;
    result = amdf_dxg_private_issue_query(query, base_query_byte_length,
                                          out_native_status);
    if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
      return result;
    }
    if (!amdf_dxg_private_chain_header_is_valid(scratch)) {
      return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
    }
    base_record += selected_index * AMDF_DXG_PRIVATE_BASE_ADAPTER_BYTE_LENGTH;
  }
  const uint32_t family = amdf_dxg_private_load_u32(base_record + 0x30);
  const uint32_t supports_wddm2 = (base_record[0x1F4] >> 7) & 1u;

  const uint32_t extended_query_byte_length =
      AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      physical_adapter_count * AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH;
  result = amdf_dxg_private_issue_query(query, extended_query_byte_length,
                                        out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  if (!amdf_dxg_private_chain_header_is_valid(scratch)) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }
  const uint8_t* extended_record =
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      selected_index * AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH;
  return amdf_dxg_private_decode_adapter(family, supports_wddm2, is_discrete,
                                         extended_record, out_info);
}

amdf_dxg_private_result_t amdf_dxg_private_query_single_adapter(
    const amdf_dxg_private_adapter_query_t* query,
    amdf_dxg_private_adapter_info_t* out_info, uint32_t* out_native_status) {
  *out_native_status = 0;
  if (query->physical_adapter_index != 0) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }

  uint8_t* scratch = (uint8_t*)query->scratch;
  amdf_dxg_private_result_t result = amdf_dxg_private_issue_query(
      query, AMDF_DXG_PRIVATE_INITIAL_CHAIN_BYTE_LENGTH, out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  const uint32_t is_discrete = amdf_dxg_private_load_u32(scratch + 0x2C) == 0;
  const uint8_t* base_record =
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH;
  const uint32_t family = amdf_dxg_private_load_u32(base_record + 0x30);
  const uint32_t supports_wddm2 = (base_record[0x1F4] >> 7) & 1u;

  const uint32_t extended_query_byte_length =
      AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH;
  result = amdf_dxg_private_issue_query(query, extended_query_byte_length,
                                        out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  return amdf_dxg_private_decode_adapter(
      family, supports_wddm2, is_discrete,
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH, out_info);
}

void amdf_dxg_private_initialize_allocation_driver_record(void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH);
  bytes[0x08] = 0x80;
  amdf_dxg_private_store_u32(bytes + 0x3C,
                             AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
}

static void amdf_dxg_private_initialize_common_allocation_record(
    const amdf_dxg_private_allocation_info_t* info, void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
  amdf_dxg_private_store_u32(bytes + 0x00, 0x1D8);
  amdf_dxg_private_store_u64(bytes + 0x3C, UINT64_C(0x1D800000001));
  amdf_dxg_private_store_u64(bytes + 0x44, UINT64_C(0x8000000001));
  amdf_dxg_private_store_u32(bytes + 0x54, (uint32_t)info->byte_length);
  amdf_dxg_private_store_u32(bytes + 0x58, 0x1000);
  amdf_dxg_private_store_u64(bytes + 0xA8, info->byte_length);
  amdf_dxg_private_store_u32(bytes + 0xF0, 0xC4);
  amdf_dxg_private_store_u32(bytes + 0x104, 0x114);
}

void amdf_dxg_private_initialize_wsl_allocation_record(
    const amdf_dxg_private_allocation_info_t* info,
    const amdf_dxg_private_adapter_info_t* adapter_info, void* record) {
  amdf_dxg_private_initialize_common_allocation_record(info, record);
  uint8_t* bytes = (uint8_t*)record;

  if (info->domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL) {
    bytes[0x50] = 1;
    bytes[0x60] = 1;
    uint32_t segment_count = 1;
    if (adapter_info->invisible_local_heap_byte_length != 0) {
      bytes[0x50] |= 2;
      bytes[0x61] = 2;
      segment_count = 2;
    }
    if (!adapter_info->is_discrete) {
      bytes[0x50] |= 8;
      bytes[0x60 + segment_count] = 4;
      ++segment_count;
    }
    bytes[0x5C] = (uint8_t)segment_count;
    return;
  }

  amdf_dxg_private_store_u32(bytes + 0x50, 0x04000008);
  bytes[0x5C] = 1;
  bytes[0x60] = 4;
  if ((adapter_info->gfx_ip_major < 12 &&
       (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN) != 0) ||
      (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED) != 0) {
    amdf_dxg_private_store_u32(bytes + 0x1D0, 4);
  }
}

void amdf_dxg_private_initialize_wddm_allocation_record(
    const amdf_dxg_private_allocation_info_t* info,
    const amdf_dxg_private_adapter_info_t* adapter_info, void* record) {
  amdf_dxg_private_initialize_common_allocation_record(info, record);
  uint8_t* bytes = (uint8_t*)record;

  if (info->domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL) {
    if (adapter_info->invisible_local_heap_byte_length != 0) {
      bytes[0x50] = 7;
      bytes[0x5C] = 3;
      bytes[0x60] = 2;
      bytes[0x61] = 1;
      bytes[0x62] = 3;
    } else {
      bytes[0x50] = 5;
      bytes[0x5C] = 2;
      bytes[0x60] = 1;
      bytes[0x61] = 3;
    }
    return;
  }

  amdf_dxg_private_store_u32(bytes + 0x50, 0x04000008);
  bytes[0x5C] = 1;
  bytes[0x60] = 4;
  if ((info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE) != 0 &&
      (adapter_info->native_queue_flags & AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL) !=
          0) {
    bytes[0xA6] |= 1;
    bytes[0xC0] |= 0x10;
  }
  if ((adapter_info->gfx_ip_major < 12 &&
       (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN) != 0) ||
      (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED) != 0) {
    amdf_dxg_private_store_u32(bytes + 0x1D0, 4);
  }
  if (info->domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM) {
    amdf_dxg_private_store_u32(
        bytes + 0x50, amdf_dxg_private_load_u32(bytes + 0x50) | 0x20002000);
    bytes[0xA7] |= 1;
  }
}

bool amdf_dxg_private_decode_allocation_metadata(
    const void* record, size_t record_byte_length,
    amdf_dxg_private_allocation_metadata_t* out_metadata) {
  if (record_byte_length < 0x1D8) {
    return false;
  }
  const uint8_t* bytes = (const uint8_t*)record;
  if (amdf_dxg_private_load_u32(bytes + 0x40) <= 0xC4 ||
      amdf_dxg_private_load_u32(bytes + 0x104) <= 3) {
    return false;
  }
  const uint32_t compression_block_sizes = bytes[0x1B8];
  const amdf_dxg_private_allocation_metadata_t metadata = {
      .byte_length = amdf_dxg_private_load_u64(bytes + 0xA8),
      .swizzle_mode = amdf_dxg_private_load_u32(bytes + 0x10C),
      .tile_swizzle = amdf_dxg_private_load_u32(bytes + 0x140),
      .compression_mode = amdf_dxg_private_load_u32(bytes + 0x1D4),
      .maximum_compressed_block_size = (compression_block_sizes >> 4) & 3u,
      .maximum_uncompressed_block_size = (compression_block_sizes >> 2) & 3u,
  };
  *out_metadata = metadata;
  return true;
}

void amdf_dxg_private_initialize_context_record(
    uint32_t supports_firmware_state_shadowing, uint32_t scheduler_id,
    uint64_t debugger_data, void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH);
  amdf_dxg_private_store_u64(bytes + 0x00, UINT64_C(0x0010010000000040));
  amdf_dxg_private_store_u32(
      bytes + 0x38,
      (scheduler_id & 0x7F) | ((supports_firmware_state_shadowing != 0) << 21));
  amdf_dxg_private_store_u32(bytes + 0x40, 0x168);
  amdf_dxg_private_store_u64(bytes + 0x150, debugger_data);
}

void amdf_dxg_private_initialize_hardware_queue_record(
    uint32_t supports_firmware_state_shadowing,
    amdf_dxg_private_queue_priority_t priority,
    uint32_t context_save_restore_allocation_handle, void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH);
  amdf_dxg_private_store_u32(
      bytes + 0x00, AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH);
  amdf_dxg_private_store_u32(
      bytes + 0x10,
      ((priority == AMDF_DXG_PRIVATE_QUEUE_PRIORITY_HIGH ? 2u : 1u) << 12) |
          ((supports_firmware_state_shadowing != 0) << 17));
  amdf_dxg_private_store_u32(bytes + 0x30,
                             context_save_restore_allocation_handle);
}

void amdf_dxg_private_initialize_submission_record(
    uint32_t queue_handle, uint64_t command_buffer_address,
    uint32_t command_buffer_byte_length, uint32_t uses_hardware_queue,
    void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH);
  amdf_dxg_private_store_u64(bytes + 0x00, UINT64_C(0x100000006));
  amdf_dxg_private_store_u64(bytes + 0x08, UINT64_C(1) << 38);
  amdf_dxg_private_store_u32(bytes + 0x10, command_buffer_byte_length);
  bytes[0x14] = 4;
  bytes[0x1C] = 1;
  amdf_dxg_private_store_u64(bytes + 0x20, command_buffer_address);
  if (!uses_hardware_queue) {
    amdf_dxg_private_store_u32(bytes + 0x28, queue_handle);
  }
}
