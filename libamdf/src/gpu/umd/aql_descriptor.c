// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/aql_descriptor.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/atomics.h"

_Static_assert(sizeof(amdf_gpu_aql_descriptor_t) == 256,
               "AMD queue descriptor prefix size");
_Static_assert(offsetof(amdf_gpu_aql_descriptor_t, write_dispatch_id) == 56,
               "AMD queue reservation index offset");
_Static_assert(offsetof(amdf_gpu_aql_descriptor_t, read_dispatch_id) == 128,
               "AMD queue consumption index offset");
_Static_assert(offsetof(amdf_gpu_aql_descriptor_t, inactive_signal_address) ==
                   192,
               "AMD queue inactive signal offset");
_Static_assert(sizeof(amdf_gpu_aql_signal_t) == 64, "AMD signal block size");
_Static_assert(offsetof(amdf_gpu_aql_signal_t, start_timestamp) == 32,
               "AMD dispatch start timestamp offset");
_Static_assert(offsetof(amdf_gpu_aql_signal_t, end_timestamp) == 40,
               "AMD dispatch end timestamp offset");

bool amdf_gpu_aql_descriptor_profile_initialize(
    const amdf_gpu_endpoint_properties_t* properties, uint32_t group_base_hi,
    uint32_t private_base_hi, amdf_gpu_aql_descriptor_profile_t* out_profile) {
  const uint32_t xcc_count = properties->topology.xcc_count;
  const uint32_t shader_engine_count =
      properties->topology.shader_engine_count_per_xcc;
  if (xcc_count == 0 || shader_engine_count == 0 ||
      (properties->compute.wavefront_size != 32 &&
       properties->compute.wavefront_size != 64) ||
      properties->compute.compute_unit_count == 0 ||
      properties->compute.compute_unit_count % xcc_count != 0 ||
      properties->compute.maximum_wave_count_per_compute_unit == 0 ||
      properties->compute.maximum_scratch_wave_count_per_compute_unit == 0 ||
      properties->compute.local_data_share_byte_length == 0) {
    return false;
  }
  const uint32_t compute_units_per_xcc =
      properties->compute.compute_unit_count / xcc_count;
  const uint64_t scratch_slots_per_engine =
      (((uint64_t)compute_units_per_xcc + shader_engine_count - 1) /
       shader_engine_count) *
      properties->compute.maximum_scratch_wave_count_per_compute_unit;
  const uint64_t scratch_slots_per_xcc =
      scratch_slots_per_engine * shader_engine_count;
  if (scratch_slots_per_xcc > UINT32_MAX / xcc_count) {
    return false;
  }
  const bool rdna = properties->gfx_ip.major >= 11;
  const uint64_t active_scratch_waves =
      (uint64_t)properties->compute.compute_unit_count *
      properties->compute.maximum_scratch_wave_count_per_compute_unit /
      (rdna ? xcc_count : 1);
  uint64_t temporary_ring_wave_count =
      rdna ? scratch_slots_per_engine : scratch_slots_per_xcc;
  if (temporary_ring_wave_count > active_scratch_waves) {
    temporary_ring_wave_count = active_scratch_waves;
  }
  if (temporary_ring_wave_count > 0xfff ||
      (!rdna && temporary_ring_wave_count % shader_engine_count != 0)) {
    return false;
  }
  *out_profile = (amdf_gpu_aql_descriptor_profile_t){
      .xcc_count = xcc_count,
      .maximum_compute_unit_id = properties->compute.compute_unit_count - 1,
      .maximum_wave_id =
          properties->compute.maximum_wave_count_per_compute_unit - 1,
      .apertures =
          {
              .group_base_hi = group_base_hi,
              .private_base_hi = private_base_hi,
          },
      .scratch =
          {
              .slot_count_per_xcc = (uint32_t)scratch_slots_per_xcc,
              .temporary_ring_wave_count = (uint32_t)temporary_ring_wave_count,
              .wave_size_shift = rdna ? 8 : 10,
              .maximum_wave_byte_length =
                  properties->gfx_ip.major == 12 ? 67106816 : 8387584,
              // Unsigned 32-bit swizzled scratch with thread-ID
              // addition. RDNA's CP supplies INDEX_STRIDE for the
              // dispatched wave size.
              .resource_descriptor = {0,
                                      rdna ? (UINT32_C(1) << 30)
                                           : (UINT32_C(1) << 31),
                                      0, rdna ? 0x20814fac : 0x00ea4fac},
          },
      .inactive_signal_byte_offset = 256,
  };
  return true;
}

amdf_status_t amdf_gpu_aql_descriptor_initialize(
    const amdf_gpu_aql_descriptor_profile_t* profile, uint64_t ring_byte_length,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_aql_descriptor_t* out_descriptor) {
  amdf_gpu_aql_descriptor_t descriptor = {
      .queue_type =
          (create_info->required_capabilities &
           AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH) != 0
              ? 2u
          : create_info->producer_mode == AMDF_QUEUE_PRODUCER_MODE_SINGLE ? 1u
                                                                          : 0u,
      .features = 1,
      .packet_count = (uint32_t)(ring_byte_length / 64),
      .group_segment_aperture_base_hi = profile->apertures.group_base_hi,
      .private_segment_aperture_base_hi = profile->apertures.private_base_hi,
      .maximum_compute_unit_id = profile->maximum_compute_unit_id,
      .maximum_wave_id = profile->maximum_wave_id,
      .read_dispatch_id_byte_offset =
          offsetof(amdf_gpu_aql_descriptor_t, read_dispatch_id),
      .scratch_resource_descriptor = {profile->scratch.resource_descriptor[0],
                                      profile->scratch.resource_descriptor[1],
                                      profile->scratch.resource_descriptor[2],
                                      profile->scratch.resource_descriptor[3]},
      .queue_properties =
          (1u << 1) | (create_info->dispatch_timestamps ? (1u << 3) : 0),
  };
  const amdf_gpu_umd_queue_scratch_t* scratch = &create_info->scratch;
  if (scratch->byte_length != 0) {
    const uint64_t wave_byte_length =
        ((uint64_t)scratch->maximum_private_segment_byte_length * 64 + 1023) &
        ~UINT64_C(1023);
    const amdf_status_t status = amdf_gpu_aql_scratch_initialize(
        profile, scratch, wave_byte_length, &descriptor);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  *out_descriptor = descriptor;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_aql_scratch_query(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    const amdf_gpu_aql_descriptor_t* descriptor, const void* ring,
    amdf_gpu_aql_scratch_request_t* out_request) {
  const uint64_t inactive_value = amdf_atomic_uint64_load_acquire(
      (const amdf_atomic_uint64_t*)((const uint8_t*)descriptor +
                                    profile->inactive_signal_byte_offset +
                                    offsetof(amdf_gpu_aql_signal_t, value)));
  if (inactive_value == 0) {
    *out_request = (amdf_gpu_aql_scratch_request_t){0};
    return AMDF_STATUS_OK;
  }
  if ((inactive_value & ~UINT64_C(0x401)) != 0) {
    return inactive_value > UINT32_MAX
               ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
               : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE,
                                  (uint32_t)inactive_value);
  }

  // EOP handling may leave consumption before the requesting dispatch. Only
  // this paused service path reads packet bodies; publication uses stores.
  const uint64_t read_index = amdf_atomic_uint64_load_acquire(
      (const amdf_atomic_uint64_t*)&descriptor->read_dispatch_id);
  const uint64_t write_index = amdf_atomic_uint64_load_acquire(
      (const amdf_atomic_uint64_t*)&descriptor->write_dispatch_id);
  const uint64_t packet_count = descriptor->packet_count;
  uint64_t count = write_index - read_index;
  if (count > packet_count) {
    count = packet_count;
  }
  for (uint64_t i = 0; i < count; ++i) {
    const uint8_t* packet =
        (const uint8_t*)ring + ((read_index + i) & (packet_count - 1)) * 64;
    const uint32_t header =
        __atomic_load_n((const uint32_t*)packet, __ATOMIC_ACQUIRE);
    if ((header & 0xff) != 2) {
      continue;  // KERNEL_DISPATCH.
    }
    uint32_t private_byte_length;
    memcpy(&private_byte_length, packet + 24, sizeof(private_byte_length));
    if (private_byte_length == 0) {
      continue;
    }
    amdf_gpu_aql_scratch_request_t request;
    const amdf_status_t status = amdf_gpu_aql_scratch_requirements(
        profile, private_byte_length, (inactive_value & 0x400) != 0 ? 32 : 64,
        &request);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    request.dispatch_index = read_index + i;
    *out_request = request;
    return AMDF_STATUS_OK;
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
}

amdf_status_t amdf_gpu_aql_scratch_requirements(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    uint32_t private_segment_byte_length, uint32_t wavefront_size,
    amdf_gpu_aql_scratch_request_t* out_request) {
  const uint64_t alignment = UINT64_C(1) << profile->scratch.wave_size_shift;
  const uint64_t wave_byte_length =
      ((uint64_t)private_segment_byte_length * wavefront_size + alignment - 1) &
      ~(alignment - 1);
  const uint64_t xcc_byte_length =
      wave_byte_length * profile->scratch.slot_count_per_xcc;
  if (wave_byte_length == 0 ||
      wave_byte_length > profile->scratch.maximum_wave_byte_length ||
      xcc_byte_length > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  *out_request = (amdf_gpu_aql_scratch_request_t){
      .byte_length =
          (xcc_byte_length * profile->xcc_count + 4095) & ~UINT64_C(4095),
      .private_segment_byte_length = private_segment_byte_length,
      .wavefront_size = wavefront_size,
      .wave_count = profile->scratch.slot_count_per_xcc * profile->xcc_count,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_aql_scratch_initialize(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    const amdf_gpu_umd_queue_scratch_t* scratch, uint64_t wave_byte_length,
    amdf_gpu_aql_descriptor_t* descriptor) {
  const uint64_t wave_count =
      (uint64_t)profile->scratch.slot_count_per_xcc * profile->xcc_count;
  // Retained scratch uses physical slot addressing. A smaller pool requires
  // firmware's one-dispatch reclaim protocol, not just a lower occupancy cap.
  if (scratch->maximum_wave_count != wave_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t xcc_byte_length =
      wave_byte_length * profile->scratch.slot_count_per_xcc;
  if (wave_byte_length == 0 ||
      wave_byte_length > profile->scratch.maximum_wave_byte_length ||
      xcc_byte_length > UINT32_MAX ||
      scratch->byte_length < xcc_byte_length * profile->xcc_count ||
      scratch->device_address % 4096 != 0 ||
      scratch->device_address >= (UINT64_C(1) << 48) ||
      scratch->byte_length > (UINT64_C(1) << 48) - scratch->device_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  descriptor->compute_temporary_ring_size =
      profile->scratch.temporary_ring_wave_count |
      ((uint32_t)(wave_byte_length >> profile->scratch.wave_size_shift) << 12);
  descriptor->scratch_resource_descriptor[0] =
      (uint32_t)scratch->device_address;
  descriptor->scratch_resource_descriptor[1] =
      profile->scratch.resource_descriptor[1] |
      (uint32_t)(scratch->device_address >> 32);
  descriptor->scratch_resource_descriptor[2] = (uint32_t)xcc_byte_length;
  descriptor->scratch_resource_descriptor[3] =
      profile->scratch.resource_descriptor[3];
  descriptor->scratch_backing_address = scratch->device_address;
  descriptor->scratch_wave64_lane_byte_length =
      (uint32_t)(wave_byte_length / 64);
  return AMDF_STATUS_OK;
}
