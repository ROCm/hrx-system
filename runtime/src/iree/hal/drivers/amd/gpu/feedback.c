// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/feedback.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"
#include "iree/hal/device.h"
#include "iree/hal/drivers/amd/status.h"

// This is the existing device channel capacity, backed once and mapped twice.
#define IREE_HAL_AMD_GPU_FEEDBACK_CAPACITY (2u * 1024u * 1024u)

static iree_status_t iree_hal_amd_gpu_feedback_initialize_ring(
    const iree_hal_amd_fabric_memory_options_t* memory,
    uint32_t address_domain_ordinal, iree_hal_amd_gpu_feedback_t* feedback) {
  const iree_hal_amd_gpu_context_t* context = feedback->context;
  const amdf_api_t* api = context->api;
  amdf_memory_reservation_capabilities_t capabilities[2] = {
      {.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
       .structure_size = sizeof(capabilities[0])},
      {.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
       .structure_size = sizeof(capabilities[1])},
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->instance_query_host_memory_reservation_capabilities(
          context->fabric->instance, &capabilities[0]),
      "feedback host reservation capabilities"));
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->device_query_memory_reservation_capabilities(
          context->device, address_domain_ordinal, &capabilities[1]),
      "feedback device reservation capabilities"));
  uint64_t capacity = IREE_HAL_AMD_GPU_FEEDBACK_CAPACITY;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(capabilities); ++i) {
    capacity = iree_max(capacity, capabilities[i].reservation_granularity);
    capacity = iree_max(capacity, capabilities[i].mapping_granularity);
  }
  capacity = iree_math_round_up_to_pow2_u64(capacity);
  if (!capacity || capacity > UINT64_MAX / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU feedback ring geometry overflows");
  }
  const amdf_memory_access_t permissions =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  amdf_memory_device_access_t access = memory->accesses[0];
  access.requirements.access = permissions | capabilities[1].required_access;
  access.requirements.flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
  iree_hal_amd_fabric_memory_options_t ring_options = *memory;
  ring_options.access_count = 1;
  ring_options.accesses = &access;
  ring_options.host_access = 0;
  ring_options.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE |
                                 capabilities[0].required_source_flags |
                                 capabilities[1].required_source_flags;
  amdf_memory_access_capabilities_t access_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access_capabilities),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_scope_query_device_profile(
          memory->scope, memory->profile.ordinal, 1, &access,
          &ring_options.profile, &access_capabilities),
      "feedback ring memory profile"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      api, &ring_options, capacity, capacity, context->event_sink,
      &feedback->ring_memory));
  const amdf_memory_reservation_access_t accesses[2] = {
      {.address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
       .access = permissions,
       .mapping_granularity = capacity},
      {.device = context->device,
       .address_domain_ordinal = address_domain_ordinal,
       .access = access.requirements.access,
       .mapping_granularity = capacity},
  };
  const amdf_memory_reservation_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .byte_length = capacity * 2,
      .minimum_alignment = capacity,
      .access_count = IREE_ARRAYSIZE(accesses),
      .accesses = accesses,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_reservation_create(context->fabric->instance, &create_info,
                                     &feedback->reservation),
      "feedback circular reservation"));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < IREE_ARRAYSIZE(accesses) && iree_status_is_ok(status); ++i) {
    amdf_memory_reservation_access_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(info),
    };
    status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(api->memory_reservation_query_access_info(
                                          feedback->reservation, i, &info),
                                      "feedback reservation address");
    if (iree_status_is_ok(status) && info.mapping_access_ordinal == i) {
      const amdf_memory_mapping_update_t updates[2] = {
          {.byte_length = capacity,
           .memory = feedback->ring_memory.handle,
           .access = accesses[i].access},
          {.target_byte_offset = capacity,
           .byte_length = capacity,
           .memory = feedback->ring_memory.handle,
           .access = accesses[i].access},
      };
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          api->memory_reservation_update(feedback->reservation, i,
                                         IREE_ARRAYSIZE(updates), updates),
          "feedback circular aliases");
    }
    if (iree_status_is_ok(status)) {
      if (i == 0) {
        feedback->ring = (uint8_t*)(uintptr_t)info.address;
      } else {
        feedback->control->ring_base = info.address;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    feedback->control->ring_capacity = capacity;
    memset(feedback->ring, 0, capacity);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_feedback_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    const iree_hal_amd_fabric_memory_options_t* memory,
    uint32_t address_domain_ordinal, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_feedback_t* out_feedback) {
  *out_feedback = (iree_hal_amd_gpu_feedback_t){0};
  // The existing compiler signal ABI uses the same ID for mailbox and sendmsg.
  // Admit only native operands that this ABI represents exactly.
  if (!context->event ||
      context->event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
      context->event_info.format_version != 1 ||
      context->event_info.trigger_data !=
          context->event_info.interrupt_context) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "GPU feedback requires native shader notification "
                            "operands supported by the executable signal ABI");
  }
  out_feedback->context = context;
  out_feedback->device = device;
  out_feedback->device_id = iree_hal_device_id(device);
  iree_hal_amd_gpu_source_context_registry_initialize(host_allocator,
                                                      &out_feedback->sources);
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, memory, 128, 64, context->event_sink,
      &out_feedback->control_memory);
  uint64_t control_address = 0;
  if (iree_status_is_ok(status)) {
    if (out_feedback->control_memory.host_cacheability !=
        IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
      status =
          iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                           "GPU feedback requires coherent write-back control");
    } else {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          context->api->memory_query_address(
              out_feedback->control_memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
              &control_address),
          "feedback control address");
    }
  }
  if (iree_status_is_ok(status)) {
    memset(out_feedback->control_memory.contents.data, 0, 128);
    out_feedback->control = (void*)out_feedback->control_memory.contents.data;
    out_feedback->control->record_length = sizeof(*out_feedback->control);
    out_feedback->control->flags = 1;
    // Native USER signal: kind, value, mailbox, and event ID at ABI offsets.
    uint8_t* signal = out_feedback->control_memory.contents.data + 64;
    iree_unaligned_store_le_u64(signal, 1);
    iree_unaligned_store_le_u64(signal + 16,
                                context->event_info.mailbox_address);
    iree_unaligned_store_le_u32(signal + 24, context->event_info.trigger_data);
    status = iree_hal_amd_gpu_feedback_initialize_ring(
        memory, address_domain_ordinal, out_feedback);
  }
  if (iree_status_is_ok(status)) {
    out_feedback->config = (iree_hal_amd_gpu_feedback_config_t){
        .record_length = sizeof(out_feedback->config),
        .flags = 1,
        .channel_base = control_address,
        .notify_signal = control_address + 64,
    };
  } else {
    iree_hal_amd_gpu_feedback_deinitialize(out_feedback);
  }
  return status;
}

bool iree_hal_amd_gpu_feedback_deinitialize(
    iree_hal_amd_gpu_feedback_t* feedback) {
  if (!feedback->context) {
    return true;
  }
  iree_hal_amd_gpu_context_t* context = feedback->context;
  if (feedback->control && iree_status_is_ok(feedback->failure) &&
      feedback->control->read_tail != feedback->control->reservation_head) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "GPU feedback has an unpublished reservation after producer shutdown");
    iree_hal_amd_gpu_context_report_status(context, feedback->device, status);
    iree_status_free(status);
  }
  if (feedback->reservation) {
    amdf_status_t status =
        context->api->memory_reservation_destroy(feedback->reservation);
    feedback->reservation = NULL;
    if (!amdf_status_is_ok(status)) {
      context->cleanup_blocked = true;
      iree_hal_amd_gpu_context_report(context, status,
                                      "feedback alias removal");
      return false;
    }
  }
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &feedback->ring_memory);
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &feedback->control_memory);
  iree_hal_amd_gpu_source_context_registry_deinitialize(&feedback->sources);
  iree_status_free(feedback->failure);
  *feedback = (iree_hal_amd_gpu_feedback_t){0};
  return true;
}

static uint64_t iree_hal_amd_gpu_feedback_load(const uint64_t* value) {
  return (uint64_t)iree_atomic_load((const iree_atomic_int64_t*)value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_amd_gpu_feedback_publish(
    iree_hal_amd_gpu_feedback_t* feedback,
    const iree_hal_amd_gpu_feedback_packet_t* packet) {
  const iree_host_size_t length = packet->record_length - packet->header_length;
  const uint8_t* payload = (const uint8_t*)packet + packet->header_length;
  // UBSAN includes authored assertions and the complete existing check family.
  if (packet->kind != 5 || length < 64 ||
      iree_unaligned_load_le_u32(payload) < 64 ||
      iree_unaligned_load_le_u32(payload) > length ||
      iree_unaligned_load_le_u32(payload + 4) != 0) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "unsupported GPU feedback packet kind %u or report layout",
        packet->kind);
  }
  const iree_hal_amd_gpu_source_context_t* source =
      (const void*)(uintptr_t)packet->source_context;
  // Device publication is the handoff from the executable-loading thread.
  IREE_TSAN_ACQUIRE((void*)source);
  const uint32_t check_kind = iree_unaligned_load_le_u32(payload + 8);
  const iree_hal_device_ubsan_report_t report = {
      .record_length = sizeof(report),
      .abi_version = IREE_HAL_DEVICE_UBSAN_REPORT_ABI_VERSION_0,
      .check_kind = check_kind <= IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION
                        ? check_kind
                        : IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN,
      .flags = iree_unaligned_load_le_u32(payload + 12),
      .site_id = iree_unaligned_load_le_u64(payload + 16),
      .operand0 = iree_unaligned_load_le_u64(payload + 24),
      .operand1 = iree_unaligned_load_le_u64(payload + 32),
      .workgroup_id = {packet->source_workgroup_id_x},
      .workitem_id = {packet->source_workitem_id_x},
      .source_dispatch_ptr = packet->source_dispatch_ptr,
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.sequence = packet->sequence;
  event.source.device = feedback->device;
  event.source.device_id = feedback->device_id;
  event.source.driver_id = IREE_SV("amd");
  event.source.executable_id =
      iree_hal_amd_gpu_source_context_executable_id(source);
  event.source.physical_device_ordinal = 0;
  iree_hal_device_event_site_t site;
  if (iree_hal_amd_gpu_source_context_resolve(source, report.site_id, &site)) {
    event.site = &site;
  }
  event.payload = iree_make_const_byte_span(&report, sizeof(report));
  event.implementation_payload =
      iree_make_const_byte_span(packet, packet->record_length);
  iree_hal_device_event_sink_publish(feedback->context->event_sink, &event);
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_feedback_drain_packets(
    iree_hal_amd_gpu_feedback_t* feedback, uint32_t limit) {
  if (!iree_status_is_ok(feedback->failure)) {
    return false;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_feedback_control_t* control = feedback->control;
  uint64_t position = iree_hal_amd_gpu_feedback_load(&control->read_tail);
  const uint64_t reserved =
      iree_hal_amd_gpu_feedback_load(&control->reservation_head);
  iree_status_t status = iree_ok_status();
  uint32_t count = 0;
  while (position != reserved && count < limit && iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_feedback_packet_t* packet =
        (const void*)(feedback->ring +
                      (position & (control->ring_capacity - 1)));
    if (iree_atomic_load((const iree_atomic_int32_t*)&packet->state,
                         iree_memory_order_acquire) != 1 ||
        packet->sequence != position) {
      break;
    }
    const uint32_t length = packet->record_length;
    if (packet->header_length < sizeof(*packet) ||
        length < packet->header_length || length > control->ring_capacity ||
        length > reserved - position || (length & 63u)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "invalid GPU feedback packet at position %" PRIu64, position);
    } else {
      status = iree_hal_amd_gpu_feedback_publish(feedback, packet);
      if (iree_status_is_ok(status)) {
        position += length;
        iree_atomic_store((iree_atomic_int64_t*)&control->read_tail,
                          (int64_t)position, iree_memory_order_release);
        ++count;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(feedback->context, feedback->device,
                                           status);
    feedback->failure = status;
  }
  const uint64_t dropped =
      iree_hal_amd_gpu_feedback_load(&control->dropped_packet_count);
  if (dropped != feedback->reported_drops) {
    iree_status_t dropped_status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "GPU feedback ring dropped %" PRIu64 " reports",
                         dropped - feedback->reported_drops);
    feedback->reported_drops = dropped;
    iree_hal_amd_gpu_context_report_status(feedback->context, feedback->device,
                                           dropped_status);
    iree_status_free(dropped_status);
  }
  IREE_TRACE_ZONE_END(z0);
  return count == limit;
}

bool iree_hal_amd_gpu_feedback_poll(void* user_data,
                                    amdf_status_t wait_status) {
  return amdf_status_is_ok(wait_status) &&
         iree_hal_amd_gpu_feedback_drain_packets(user_data, 64);
}

void iree_hal_amd_gpu_feedback_drain(iree_hal_amd_gpu_feedback_t* feedback) {
  iree_hal_amd_gpu_feedback_drain_packets(feedback, UINT32_MAX);
}
