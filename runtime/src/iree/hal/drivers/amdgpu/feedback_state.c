// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/feedback_state.h"

#include <stdio.h>

#include "iree/base/threading/affinity.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/source_context.h"
#include "iree/hal/drivers/amdgpu/system.h"

static const char* iree_hal_amdgpu_feedback_state_asan_access_kind_string(
    iree_hal_amdgpu_asan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ:
      return "read";
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE:
      return "write";
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_ATOMIC:
      return "atomic";
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static iree_hal_device_asan_access_kind_t
iree_hal_amdgpu_feedback_state_map_asan_access_kind(
    iree_hal_amdgpu_asan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ:
      return IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ;
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE:
      return IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE;
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_ATOMIC:
      return IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC;
    case IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_UNKNOWN:
    default:
      return IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN;
  }
}

static const char* iree_hal_amdgpu_feedback_state_tsan_check_kind_string(
    iree_hal_amdgpu_tsan_check_kind_t check_kind) {
  switch (check_kind) {
    case IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE:
      return "data_race";
    case IREE_HAL_AMDGPU_TSAN_CHECK_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_amdgpu_feedback_state_tsan_memory_space_string(
    iree_hal_amdgpu_tsan_memory_space_t memory_space) {
  switch (memory_space) {
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL:
      return "global";
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP:
      return "workgroup";
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE:
      return "private";
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_amdgpu_feedback_state_tsan_access_kind_string(
    iree_hal_amdgpu_tsan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ:
      return "read";
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE:
      return "write";
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE:
      return "read_write";
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_ATOMIC:
      return "atomic";
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static iree_hal_device_tsan_check_kind_t
iree_hal_amdgpu_feedback_state_map_tsan_check_kind(
    iree_hal_amdgpu_tsan_check_kind_t check_kind) {
  switch (check_kind) {
    case IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE:
      return IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE;
    case IREE_HAL_AMDGPU_TSAN_CHECK_KIND_UNKNOWN:
    default:
      return IREE_HAL_DEVICE_TSAN_CHECK_KIND_UNKNOWN;
  }
}

static iree_hal_device_tsan_memory_space_t
iree_hal_amdgpu_feedback_state_map_tsan_memory_space(
    iree_hal_amdgpu_tsan_memory_space_t memory_space) {
  switch (memory_space) {
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL:
      return IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_GLOBAL;
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP:
      return IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP;
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE:
      return IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_PRIVATE;
    case IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN:
    default:
      return IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_UNKNOWN;
  }
}

static iree_hal_device_tsan_access_kind_t
iree_hal_amdgpu_feedback_state_map_tsan_access_kind(
    iree_hal_amdgpu_tsan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ:
      return IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ;
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE:
      return IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE;
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE:
      return IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ_WRITE;
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_ATOMIC:
      return IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC;
    case IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN:
    default:
      return IREE_HAL_DEVICE_TSAN_ACCESS_KIND_UNKNOWN;
  }
}

static uint32_t iree_hal_amdgpu_feedback_state_physical_device_ordinal(
    iree_host_size_t physical_device_ordinal) {
  return physical_device_ordinal <= UINT32_MAX
             ? (uint32_t)physical_device_ordinal
             : UINT32_MAX;
}

static const iree_hal_amdgpu_source_context_t*
iree_hal_amdgpu_feedback_state_source_context(
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  return (const iree_hal_amdgpu_source_context_t*)(uintptr_t)
      packet->source_context;
}

static void iree_hal_amdgpu_feedback_state_publish_asan_event(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet,
    const iree_hal_amdgpu_asan_report_t* report) {
  const iree_hal_amdgpu_source_context_t* source_context =
      iree_hal_amdgpu_feedback_state_source_context(packet);

  iree_hal_device_asan_report_t asan_event;
  memset(&asan_event, 0, sizeof(asan_event));
  asan_event.record_length = sizeof(asan_event);
  asan_event.abi_version = IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0;
  asan_event.access_kind =
      iree_hal_amdgpu_feedback_state_map_asan_access_kind(report->access_kind);
  asan_event.flags = report->flags;
  asan_event.fault_address = report->fault_address;
  asan_event.access_length = report->access_size;
  asan_event.site_id = report->site_id;
  asan_event.shadow_address = report->shadow_address;
  asan_event.shadow_value = report->shadow_value;
  asan_event.workgroup_id[0] = packet->source_workgroup_id_x;
  asan_event.workitem_id[0] = packet->source_workitem_id_x;
  asan_event.source_dispatch_ptr = packet->source_dispatch_ptr;

  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.sequence = packet->sequence;
  event.source.device = state->device;
  event.source.device_id = state->device_id;
  event.source.driver_id = IREE_SV("amdgpu");
  event.source.executable_id =
      iree_hal_amdgpu_source_context_executable_id(source_context);
  event.source.physical_device_ordinal =
      iree_hal_amdgpu_feedback_state_physical_device_ordinal(
          physical_device_ordinal);
  iree_hal_device_event_site_t site;
  if (iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
          source_context, report->site_id, &site)) {
    event.site = &site;
  }
  event.payload = iree_make_const_byte_span(&asan_event, sizeof(asan_event));
  event.implementation_payload =
      iree_make_const_byte_span(packet, packet->record_length);
  iree_hal_device_event_sink_publish(state->event_sink, &event);
}

static iree_status_t iree_hal_amdgpu_feedback_state_handle_asan_packet(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  if (IREE_UNLIKELY(packet->record_length < packet->header_length)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU ASAN feedback packet on physical device %" PRIhsz
        " has invalid lengths: packet_record_length=%u, "
        "packet_header_length=%u",
        physical_device_ordinal, packet->record_length, packet->header_length);
  }
  const iree_host_size_t payload_length =
      packet->record_length - packet->header_length;
  if (IREE_UNLIKELY(payload_length < sizeof(iree_hal_amdgpu_asan_report_t))) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU ASAN feedback packet on physical device %" PRIhsz
        " is too small: packet_payload_length=%" PRIhsz
        ", asan_report_length=%" PRIhsz,
        physical_device_ordinal, payload_length,
        sizeof(iree_hal_amdgpu_asan_report_t));
  }

  const iree_hal_amdgpu_asan_report_t* report =
      (const iree_hal_amdgpu_asan_report_t*)((const uint8_t*)packet +
                                             packet->header_length);
  if (IREE_UNLIKELY(
          report->record_length < sizeof(iree_hal_amdgpu_asan_report_t) ||
          report->record_length > payload_length ||
          report->abi_version != IREE_HAL_AMDGPU_ASAN_REPORT_ABI_VERSION_0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU ASAN report on physical device %" PRIhsz
        " has unsupported ABI: record_length=%u, payload_length=%" PRIhsz
        ", abi_version=%u",
        physical_device_ordinal, report->record_length, payload_length,
        report->abi_version);
  }

  iree_hal_amdgpu_feedback_state_publish_asan_event(
      state, physical_device_ordinal, packet, report);
  if (state->asan_report_policy ==
      IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_REPORT_ONLY) {
    return iree_ok_status();
  }

  const uint64_t executable_id = iree_hal_amdgpu_source_context_executable_id(
      iree_hal_amdgpu_feedback_state_source_context(packet));
  return iree_make_status(
      IREE_STATUS_ABORTED,
      "AMDGPU ASAN %s access violation on physical device %" PRIhsz
      " site_id=0x%016" PRIx64 " fault_address=0x%016" PRIx64
      " access_size=%" PRIu64 " shadow_address=0x%016" PRIx64
      " shadow_value=0x%016" PRIx64
      " workgroup_x=%u workitem_x=%u executable_id=%" PRIu64
      " dispatch=0x%016" PRIx64,
      iree_hal_amdgpu_feedback_state_asan_access_kind_string(
          report->access_kind),
      physical_device_ordinal, report->site_id, report->fault_address,
      report->access_size, report->shadow_address, report->shadow_value,
      packet->source_workgroup_id_x, packet->source_workitem_id_x,
      executable_id, packet->source_dispatch_ptr);
}

static void iree_hal_amdgpu_feedback_state_publish_tsan_event(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet,
    const iree_hal_amdgpu_tsan_report_t* report) {
  const iree_hal_amdgpu_source_context_t* source_context =
      iree_hal_amdgpu_feedback_state_source_context(packet);

  iree_hal_device_tsan_report_t tsan_event;
  memset(&tsan_event, 0, sizeof(tsan_event));
  tsan_event.record_length = sizeof(tsan_event);
  tsan_event.abi_version = IREE_HAL_DEVICE_TSAN_REPORT_ABI_VERSION_0;
  tsan_event.check_kind =
      iree_hal_amdgpu_feedback_state_map_tsan_check_kind(report->check_kind);
  tsan_event.flags = report->flags;
  tsan_event.memory_space =
      iree_hal_amdgpu_feedback_state_map_tsan_memory_space(
          report->memory_space);
  tsan_event.current_access_kind =
      iree_hal_amdgpu_feedback_state_map_tsan_access_kind(
          report->current_access_kind);
  tsan_event.prior_access_kind =
      iree_hal_amdgpu_feedback_state_map_tsan_access_kind(
          report->prior_access_kind);
  tsan_event.access_length = report->access_size;
  tsan_event.current_site_id = report->current_site_id;
  tsan_event.prior_site_id = report->prior_site_id;
  tsan_event.memory_address = report->memory_address;
  tsan_event.shadow_address = report->shadow_address;
  tsan_event.shadow_value = report->shadow_value;
  tsan_event.current_workgroup_id[0] = report->current_workgroup_id[0];
  tsan_event.current_workgroup_id[1] = report->current_workgroup_id[1];
  tsan_event.current_workgroup_id[2] = report->current_workgroup_id[2];
  tsan_event.current_workitem_id[0] = report->current_workitem_id[0];
  tsan_event.current_workitem_id[1] = report->current_workitem_id[1];
  tsan_event.current_workitem_id[2] = report->current_workitem_id[2];
  tsan_event.prior_workgroup_id[0] = report->prior_workgroup_id[0];
  tsan_event.prior_workgroup_id[1] = report->prior_workgroup_id[1];
  tsan_event.prior_workgroup_id[2] = report->prior_workgroup_id[2];
  tsan_event.prior_workitem_id[0] = report->prior_workitem_id[0];
  tsan_event.prior_workitem_id[1] = report->prior_workitem_id[1];
  tsan_event.prior_workitem_id[2] = report->prior_workitem_id[2];
  tsan_event.source_dispatch_ptr = packet->source_dispatch_ptr;

  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.sequence = packet->sequence;
  event.source.device = state->device;
  event.source.device_id = state->device_id;
  event.source.driver_id = IREE_SV("amdgpu");
  event.source.executable_id =
      iree_hal_amdgpu_source_context_executable_id(source_context);
  event.source.physical_device_ordinal =
      iree_hal_amdgpu_feedback_state_physical_device_ordinal(
          physical_device_ordinal);
  iree_hal_device_event_site_t site;
  if (iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
          source_context, report->current_site_id, &site)) {
    event.site = &site;
  }
  event.payload = iree_make_const_byte_span(&tsan_event, sizeof(tsan_event));
  event.implementation_payload =
      iree_make_const_byte_span(packet, packet->record_length);
  iree_hal_device_event_sink_publish(state->event_sink, &event);
}

static iree_status_t iree_hal_amdgpu_feedback_state_handle_tsan_packet(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  if (IREE_UNLIKELY(packet->record_length < packet->header_length)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU TSAN feedback packet on physical device %" PRIhsz
        " has invalid lengths: packet_record_length=%u, "
        "packet_header_length=%u",
        physical_device_ordinal, packet->record_length, packet->header_length);
  }
  const iree_host_size_t payload_length =
      packet->record_length - packet->header_length;
  if (IREE_UNLIKELY(payload_length < sizeof(iree_hal_amdgpu_tsan_report_t))) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU TSAN feedback packet on physical device %" PRIhsz
        " is too small: packet_payload_length=%" PRIhsz
        ", tsan_report_length=%" PRIhsz,
        physical_device_ordinal, payload_length,
        sizeof(iree_hal_amdgpu_tsan_report_t));
  }

  const iree_hal_amdgpu_tsan_report_t* report =
      (const iree_hal_amdgpu_tsan_report_t*)((const uint8_t*)packet +
                                             packet->header_length);
  if (IREE_UNLIKELY(
          report->record_length < sizeof(iree_hal_amdgpu_tsan_report_t) ||
          report->record_length > payload_length ||
          report->abi_version != IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "AMDGPU TSAN report on physical device %" PRIhsz
        " has unsupported ABI: record_length=%u, payload_length=%" PRIhsz
        ", abi_version=%u",
        physical_device_ordinal, report->record_length, payload_length,
        report->abi_version);
  }

  iree_hal_amdgpu_feedback_state_publish_tsan_event(
      state, physical_device_ordinal, packet, report);
  if (state->tsan_report_policy ==
      IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_REPORT_ONLY) {
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_ABORTED,
      "AMDGPU TSAN %s violation on physical device %" PRIhsz
      " site_id=0x%016" PRIx64 " prior_site_id=0x%016" PRIx64
      " memory=%s address=0x%016" PRIx64 " current_access=%s prior_access=%s",
      iree_hal_amdgpu_feedback_state_tsan_check_kind_string(report->check_kind),
      physical_device_ordinal, report->current_site_id, report->prior_site_id,
      iree_hal_amdgpu_feedback_state_tsan_memory_space_string(
          report->memory_space),
      report->memory_address,
      iree_hal_amdgpu_feedback_state_tsan_access_kind_string(
          report->current_access_kind),
      iree_hal_amdgpu_feedback_state_tsan_access_kind_string(
          report->prior_access_kind));
}

static iree_status_t iree_hal_amdgpu_feedback_state_handle_packet(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  switch (packet->kind) {
    case IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_ASAN:
      return iree_hal_amdgpu_feedback_state_handle_asan_packet(
          state, physical_device_ordinal, packet);
    case IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN:
      return iree_hal_amdgpu_feedback_state_handle_tsan_packet(
          state, physical_device_ordinal, packet);
    default: {
      const uint64_t executable_id =
          iree_hal_amdgpu_source_context_executable_id(
              iree_hal_amdgpu_feedback_state_source_context(packet));
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unhandled AMDGPU feedback packet kind %u on physical device %" PRIhsz
          " workgroup_x=%u workitem_x=%u executable_id=%" PRIu64
          " dispatch=0x%016" PRIx64,
          (uint32_t)packet->kind, physical_device_ordinal,
          packet->source_workgroup_id_x, packet->source_workitem_id_x,
          executable_id, packet->source_dispatch_ptr);
    }
  }
}

static void iree_hal_amdgpu_feedback_state_report_error(
    iree_hal_amdgpu_feedback_state_t* state, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (state->error_handler) {
    state->error_handler(state->error_handler_user_data, status);
  } else {
    iree_status_free(status);
  }
}

static iree_status_t iree_hal_amdgpu_feedback_state_drain_packet(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data) {
  iree_hal_amdgpu_feedback_device_state_t* device_state =
      (iree_hal_amdgpu_feedback_device_state_t*)user_data;
  iree_status_t status = iree_hal_amdgpu_feedback_state_handle_packet(
      device_state->parent, device_state->physical_device_ordinal, packet);
  iree_hal_amdgpu_feedback_state_report_error(device_state->parent, status);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_feedback_state_drain_device_locked(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  iree_host_size_t packet_count = 0;
  return iree_hal_amdgpu_feedback_channel_drain(
      &device_state->channel, IREE_HOST_SIZE_MAX,
      iree_hal_amdgpu_feedback_state_drain_packet, device_state, &packet_count);
}

void iree_hal_amdgpu_feedback_state_drain_physical_device(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal) {
  if (!iree_hal_amdgpu_feedback_state_is_enabled(state)) return;
  if (IREE_UNLIKELY(physical_device_ordinal >= state->device_state_count)) {
    iree_status_t status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "AMDGPU feedback physical device ordinal %" PRIhsz
                         " exceeds device count %" PRIhsz,
                         physical_device_ordinal, state->device_state_count);
    iree_hal_amdgpu_feedback_state_report_error(state, status);
    return;
  }

  iree_hal_amdgpu_feedback_device_state_t* device_state =
      &state->device_states[physical_device_ordinal];
  iree_slim_mutex_lock(&device_state->drain_mutex);
  iree_status_t status =
      iree_hal_amdgpu_feedback_state_drain_device_locked(device_state);
  iree_slim_mutex_unlock(&device_state->drain_mutex);
  iree_hal_amdgpu_feedback_state_report_error(state, status);
}

static void iree_hal_amdgpu_feedback_state_request_device_stop(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  if (device_state->stop_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(device_state->parent->libhsa),
                                    device_state->stop_signal, 1);
  }
}

static int iree_hal_amdgpu_feedback_state_service_thread_main(void* entry_arg) {
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_feedback_state_service_thread_start");
    IREE_TRACE_ZONE_END(z0);
  }

  iree_hal_amdgpu_feedback_device_state_t* device_state =
      (iree_hal_amdgpu_feedback_device_state_t*)entry_arg;
  iree_hal_amdgpu_feedback_state_t* state = device_state->parent;
  const iree_hal_amdgpu_libhsa_t* libhsa = state->libhsa;

  enum {
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_NOTIFY_SIGNAL = 0,
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_STOP_SIGNAL = 1,
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT = 2,
  };

  bool keep_running = true;
  while (keep_running) {
    hsa_signal_t signals[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
        device_state->channel.notify_signal,
        device_state->stop_signal,
    };
    hsa_signal_condition_t
        conditions[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
        0,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(libhsa), IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT,
        signals, conditions, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z0, "iree_hal_amdgpu_feedback_state_service_thread_pump");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_index);

      if (signal_index == IREE_HAL_AMDGPU_FEEDBACK_WAIT_NOTIFY_SIGNAL) {
        (void)iree_hsa_signal_exchange_scacquire(
            IREE_LIBHSA(libhsa), device_state->channel.notify_signal, 0);
        iree_hal_amdgpu_feedback_state_drain_physical_device(
            state, device_state->physical_device_ordinal);
      }

      if (signal_index == IREE_HAL_AMDGPU_FEEDBACK_WAIT_STOP_SIGNAL) {
        keep_running = false;
      } else if (IREE_UNLIKELY(signal_index >=
                               IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT)) {
        iree_status_t status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "hsa_amd_signal_wait_any returned invalid signal index %u while "
            "waiting for AMDGPU feedback packets",
            signal_index);
        iree_hal_amdgpu_feedback_state_report_error(state, status);
        keep_running = false;
      }

      IREE_TRACE_ZONE_END(z0);
    }
  }

  iree_hal_amdgpu_feedback_state_drain_physical_device(
      state, device_state->physical_device_ordinal);

  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_feedback_state_service_thread_exit");
    IREE_TRACE_ZONE_END(z0);
  }
  return 0;
}

static iree_status_t iree_hal_amdgpu_feedback_state_initialize_device(
    iree_hal_amdgpu_feedback_state_t* state, iree_hal_amdgpu_system_t* system,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_feedback_device_state_t* out_device_state) {
  memset(out_device_state, 0, sizeof(*out_device_state));

  if (IREE_UNLIKELY(!physical_device)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires initialized physical devices");
  }

  hsa_amd_memory_pool_t control_memory_pool =
      physical_device->host_memory_pools.fine_pool;
  if (IREE_UNLIKELY(!control_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires a host fine-grained control memory pool");
  }

  hsa_amd_memory_pool_t ring_memory_pool =
      physical_device->coarse_block_pools.large.memory_pool;
  if (IREE_UNLIKELY(!ring_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires a coarse-grained device-local ring memory "
        "pool");
  }

  out_device_state->parent = state;
  out_device_state->physical_device_ordinal = physical_device->device_ordinal;
  iree_slim_mutex_initialize(&out_device_state->drain_mutex);

  const iree_hal_amdgpu_feedback_channel_params_t channel_params = {
      .libhsa = state->libhsa,
      .device_agent = physical_device->device_agent,
      .control_memory_pool = control_memory_pool,
      .ring_memory_pool = ring_memory_pool,
      .topology = &system->topology,
      .minimum_capacity = 0,
  };

  iree_status_t status = iree_hal_amdgpu_feedback_channel_initialize(
      &channel_params, &out_device_state->channel);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(state->libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_device_state->stop_signal);
  }
  if (!iree_status_is_ok(status)) {
    if (out_device_state->stop_signal.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
          state->libhsa, out_device_state->stop_signal));
    }
    iree_hal_amdgpu_feedback_channel_deinitialize(&out_device_state->channel);
    iree_slim_mutex_deinitialize(&out_device_state->drain_mutex);
    memset(out_device_state, 0, sizeof(*out_device_state));
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_feedback_state_start_device_thread(
    iree_hal_amdgpu_feedback_device_state_t* device_state,
    iree_hal_amdgpu_physical_device_t* physical_device) {
  iree_hal_amdgpu_feedback_state_t* state = device_state->parent;
  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  char thread_name[32] = {0};
  snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "amdgpu-d%u-feedback",
           (unsigned)physical_device->device_ordinal);
  thread_params.name = iree_make_cstring_view(thread_name);
  iree_thread_affinity_set_group_any(physical_device->host_numa_node,
                                     &thread_params.initial_affinity);
  return iree_thread_create(iree_hal_amdgpu_feedback_state_service_thread_main,
                            device_state, thread_params, state->host_allocator,
                            &device_state->service_thread);
}

iree_status_t iree_hal_amdgpu_feedback_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_hal_device_t* device, iree_string_view_t device_id,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amdgpu_feedback_error_handler_fn_t error_handler,
    void* error_handler_user_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->feedback.enabled && !options->asan.enabled &&
      !options->tsan.enabled) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(physical_device_count == 0 || !physical_devices)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires at least one initialized physical device");
  }
  if (IREE_UNLIKELY(!error_handler)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU feedback requires a service-thread error handler");
  }
  if (IREE_UNLIKELY(!iree_hal_device_event_sink_is_valid(event_sink))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback requires a device event sink");
  }

  out_state->libhsa = &system->libhsa;
  out_state->host_allocator = host_allocator;
  out_state->device = device;
  out_state->device_id = device_id;
  out_state->event_sink = event_sink;
  out_state->asan_report_policy = options->asan.report_policy;
  out_state->tsan_report_policy = options->tsan.report_policy;
  out_state->error_handler = error_handler;
  out_state->error_handler_user_data = error_handler_user_data;

  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, physical_device_count,
      sizeof(out_state->device_states[0]), (void**)&out_state->device_states);
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amdgpu_feedback_state_initialize_device(
        out_state, system, physical_devices[i], &out_state->device_states[i]);
    if (iree_status_is_ok(status)) {
      ++out_state->device_state_count;
    }
  }
  if (iree_status_is_ok(status)) {
    out_state->is_enabled = true;
    for (iree_host_size_t i = 0;
         i < out_state->device_state_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amdgpu_feedback_state_start_device_thread(
          &out_state->device_states[i], physical_devices[i]);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_feedback_state_deinitialize(out_state);
  }
  return status;
}

void iree_hal_amdgpu_feedback_state_deinitialize(
    iree_hal_amdgpu_feedback_state_t* state) {
  if (!state || !state->libhsa) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < state->device_state_count; ++i) {
    iree_hal_amdgpu_feedback_state_request_device_stop(
        &state->device_states[i]);
  }
  for (iree_host_size_t i = 0; i < state->device_state_count; ++i) {
    iree_hal_amdgpu_feedback_device_state_t* device_state =
        &state->device_states[i];
    iree_thread_release(device_state->service_thread);
    device_state->service_thread = NULL;
    if (device_state->stop_signal.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
          state->libhsa, device_state->stop_signal));
      device_state->stop_signal = iree_hsa_signal_null();
    }
    iree_hal_amdgpu_feedback_channel_deinitialize(&device_state->channel);
    iree_slim_mutex_deinitialize(&device_state->drain_mutex);
  }
  iree_allocator_free(state->host_allocator, state->device_states);
  memset(state, 0, sizeof(*state));

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_feedback_state_is_enabled(
    const iree_hal_amdgpu_feedback_state_t* state) {
  return state && state->is_enabled;
}

iree_status_t iree_hal_amdgpu_feedback_state_populate_config(
    const iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_feedback_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_config);
  memset(out_config, 0, sizeof(*out_config));

  if (IREE_UNLIKELY(!iree_hal_amdgpu_feedback_state_is_enabled(state))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU feedback state is not enabled");
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= state->device_state_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback physical device ordinal %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            physical_device_ordinal, state->device_state_count);
  }

  *out_config = state->device_states[physical_device_ordinal].channel.config;
  return iree_ok_status();
}
