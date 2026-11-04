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
// - WKMI's public interface and matching Linux archive at rocm-systems commit
//   6b7847ef03b76c004f57b8cf40cdef9621ee6f53:
//   https://github.com/ROCm/rocm-systems/tree/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/shared/amdgpu-windows-interop/wkmi
// - the open Linux KFD policy and DXG queue use documented by:
//   https://github.com/ROCm/rocm-systems/pull/10573

#include "libamdf/src/gpu/umd/dxg/rocdxg/private_records.h"

#include <limits.h>
#include <string.h>

enum {
  AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH = 0x960,
  AMDF_DXG_PRIVATE_DISCOVERY_BYTE_LENGTH = 0xBC0,
  AMDF_DXG_PRIVATE_DISCOVERY_ADAPTER_BYTE_LENGTH = 0x200,
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

static const amdf_dxg_private_gfxip_entry_t* amdf_dxg_private_find_gfxip_entry(
    const amdf_dxg_private_adapter_query_t* query, uint16_t device_id) {
  for (uint32_t i = 0; i < query->gfxip_entry_count; ++i) {
    if (query->gfxip_entries[i].device_id == device_id) {
      return &query->gfxip_entries[i];
    }
  }
  return NULL;
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

amdf_dxg_private_result_t amdf_dxg_private_query_adapter(
    const amdf_dxg_private_adapter_query_t* query,
    amdf_dxg_private_adapter_info_t* out_info, uint32_t* out_native_status) {
  if (out_native_status != NULL) {
    *out_native_status = 0;
  }
  if (query == NULL ||
      query->structure_size < sizeof(amdf_dxg_private_adapter_query_t) ||
      query->adapter_handle == 0 || query->gfxip_entry_count == 0 ||
      query->gfxip_entries == NULL || query->query == NULL ||
      query->scratch == NULL || out_info == NULL || out_native_status == NULL) {
    return AMDF_DXG_PRIVATE_RESULT_INVALID_ARGUMENT;
  }
  if (query->scratch_byte_length < AMDF_DXG_PRIVATE_QUERY_SCRATCH_BYTE_LENGTH) {
    return AMDF_DXG_PRIVATE_RESULT_BUFFER_TOO_SMALL;
  }

  uint8_t* scratch = (uint8_t*)query->scratch;
  amdf_dxg_private_result_t result = amdf_dxg_private_issue_query(
      query, AMDF_DXG_PRIVATE_DISCOVERY_BYTE_LENGTH, out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }

  const uint32_t expected_checksum = amdf_dxg_private_load_u32(
      scratch + AMDF_DXG_PRIVATE_CHAIN_CHECKSUM_OFFSET);
  if (expected_checksum != amdf_dxg_private_chain_checksum(scratch)) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }
  const uint32_t physical_adapter_count =
      amdf_dxg_private_load_u32(scratch + 0x08);
  if (physical_adapter_count == 0 ||
      physical_adapter_count > AMDF_DXG_PRIVATE_MAX_PHYSICAL_ADAPTER_COUNT) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }
  if (query->physical_adapter_index >= physical_adapter_count) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }
  const uint32_t selected_index = query->physical_adapter_index;
  const uint32_t vendor_id =
      amdf_dxg_private_load_u16(scratch + 0x0C + selected_index * 2);
  const uint32_t is_discrete =
      amdf_dxg_private_load_u32(scratch + 0x2C + selected_index * 4) == 0;

  const uint32_t discovery_query_byte_length =
      AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      physical_adapter_count * AMDF_DXG_PRIVATE_DISCOVERY_ADAPTER_BYTE_LENGTH;
  result = amdf_dxg_private_issue_query(query, discovery_query_byte_length,
                                        out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }

  const uint32_t base_query_byte_length =
      AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      physical_adapter_count * AMDF_DXG_PRIVATE_BASE_ADAPTER_BYTE_LENGTH;
  result = amdf_dxg_private_issue_query(query, base_query_byte_length,
                                        out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  const uint8_t* base_record =
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      selected_index * AMDF_DXG_PRIVATE_BASE_ADAPTER_BYTE_LENGTH;
  const uint16_t device_id = amdf_dxg_private_load_u16(base_record + 0x2C);
  const uint32_t asic_revision = amdf_dxg_private_load_u32(base_record + 0x30);
  const uint32_t supports_wddm2 = (base_record[0x1F4] >> 7) & 1u;
  const amdf_dxg_private_gfxip_entry_t* gfxip_entry =
      amdf_dxg_private_find_gfxip_entry(query, device_id);
  if (gfxip_entry == NULL) {
    return AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED;
  }

  const uint32_t extended_query_byte_length =
      AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      physical_adapter_count * AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH;
  result = amdf_dxg_private_issue_query(query, extended_query_byte_length,
                                        out_native_status);
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return result;
  }
  const uint8_t* extended_record =
      scratch + AMDF_DXG_PRIVATE_CHAIN_HEADER_BYTE_LENGTH +
      selected_index * AMDF_DXG_PRIVATE_EXTENDED_ADAPTER_BYTE_LENGTH;

  uint64_t compute_unit_count = 0;
  if (gfxip_entry->major == 9) {
    const uint32_t shader_array_count =
        (uint32_t)(amdf_dxg_private_load_u64(extended_record + 0xC64) >> 32);
    compute_unit_count = (uint64_t)shader_array_count *
                         amdf_dxg_private_load_u32(extended_record + 0xC60) *
                         amdf_dxg_private_load_u32(extended_record + 0xC6C);
  } else {
    if ((extended_record[0xC46] & 1u) == 0) {
      return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
    }
    const uint32_t mask_count = gfxip_entry->major == 10 ? 8 : 12;
    for (uint32_t i = 0; i < mask_count; ++i) {
      compute_unit_count += amdf_dxg_private_popcount_u16(
          amdf_dxg_private_load_u16(extended_record + 0xD04 + i * 2));
    }
    compute_unit_count *= 2;
  }
  if (compute_unit_count > UINT32_MAX) {
    return AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD;
  }

  amdf_dxg_private_adapter_info_t info;
  memset(&info, 0, sizeof(info));
  info.physical_adapter_count = physical_adapter_count;
  info.vendor_id = vendor_id;
  info.device_id = device_id;
  info.asic_revision = asic_revision;
  info.gfx_ip_major = gfxip_entry->major;
  info.gfx_ip_minor = gfxip_entry->minor;
  info.gfx_ip_stepping = gfxip_entry->stepping;
  info.is_discrete = is_discrete;
  info.supports_wddm2 = supports_wddm2;
  info.wavefront_size = amdf_dxg_private_load_u32(extended_record + 0xC5C);
  info.compute_unit_count = (uint32_t)compute_unit_count;
  if (gfxip_entry->major == 9) {
    info.maximum_wave_count_per_compute_unit = 40;
    info.maximum_scratch_wave_count_per_compute_unit = 64;
  } else if (gfxip_entry->major == 12) {
    info.maximum_wave_count_per_compute_unit = 32;
    info.maximum_scratch_wave_count_per_compute_unit = 256;
  } else if (gfxip_entry->major == 10 || gfxip_entry->major == 11) {
    info.maximum_wave_count_per_compute_unit = 32;
    info.maximum_scratch_wave_count_per_compute_unit = 128;
  } else {
    info.maximum_wave_count_per_compute_unit = 32;
    info.maximum_scratch_wave_count_per_compute_unit = 64;
  }
  info.local_data_share_byte_length = 64 * 1024;
  info.shader_engine_count = amdf_dxg_private_load_u32(extended_record + 0xC60);
  info.supports_firmware_state_shadowing = (extended_record[0xC46] >> 5) & 1u;
  info.supports_platform_atomics =
      !is_discrete || ((extended_record[0x208] >> 5) & 1u);
  info.user_queue_byte_length =
      amdf_dxg_private_load_u32(extended_record + 0x30E8);
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

void amdf_dxg_private_initialize_allocation_driver_record(void* record) {
  uint8_t* bytes = (uint8_t*)record;
  memset(bytes, 0, AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH);
  bytes[0x08] = 0x80;
  amdf_dxg_private_store_u32(bytes + 0x3C,
                             AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
}

void amdf_dxg_private_initialize_allocation_record(
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

  if (info->domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL) {
    bytes[0x50] = 1;
    bytes[0x60] = 1;
    uint32_t segment_count = 1;
    if (info->invisible_local_heap_byte_length != 0) {
      bytes[0x50] |= 2;
      bytes[0x61] = 2;
      segment_count = 2;
    }
    if (!info->is_discrete) {
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
  if ((info->gfx_ip_major < 12 &&
       (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN) != 0) ||
      (info->flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED) != 0) {
    amdf_dxg_private_store_u32(bytes + 0x1D0, 4);
  }
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
