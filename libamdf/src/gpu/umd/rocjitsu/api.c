// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/device.h"
#include "libamdf/src/gpu/umd/rocjitsu/event.h"
#include "libamdf/src/gpu/umd/rocjitsu/memory.h"
#include "libamdf/src/gpu/umd/rocjitsu/user_queue.h"

static amdf_status_t amdf_rocjitsu_instance_prepare(
    amdf_gpu_umd_instance_t** state, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator) {
  (void)state;
  (void)native_lifetime;
  (void)host_allocator;
  // Configured sources own VM connections; this UMD has no additional state.
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_rocjitsu_instance_destroy(
    amdf_gpu_umd_instance_t* instance) {
  (void)instance;
  return AMDF_STATUS_OK;
}

// These services have no constructible objects or advertised capabilities in
// this implementation. Every dispatch entry remains a callable typed operation.

static amdf_status_t amdf_rocjitsu_query_queue_clock_info(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_info_t* out_info) {
  (void)device;
  (void)command_type;
  (void)out_info;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_query_clock_sample(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_sample_t* out_sample) {
  (void)device;
  (void)command_type;
  (void)out_sample;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  (void)device;
  (void)profile;
  (void)import_info;
  (void)external_memory;
  (void)memory_state;
  (void)out_result;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)memory;
  (void)export_info;
  (void)out_value;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static uintptr_t amdf_rocjitsu_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory) {
  (void)memory;
  return 0;
}

static amdf_status_t amdf_rocjitsu_query_memory_reservation_capabilities(
    amdf_gpu_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  (void)device;
  (void)address_domain_ordinal;
  (void)out_capabilities;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_umd_memory_reservation_t** out_reservation) {
  (void)host_allocator;
  (void)device_count;
  (void)devices;
  (void)byte_length;
  (void)minimum_alignment;
  (void)common_address;
  (void)out_reservation;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static void amdf_rocjitsu_memory_reservation_query_access(
    const amdf_gpu_umd_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access) {
  (void)reservation;
  (void)access_ordinal;
  (void)out_address;
  (void)out_mapping_access_ordinal;
  (void)out_mapping_granularity;
  (void)out_required_access;
}

static amdf_status_t amdf_rocjitsu_memory_reservation_update(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started) {
  (void)reservation;
  (void)access_ordinal;
  (void)update_count;
  (void)updates;
  (void)mutation_started;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_reservation_update_on_queue(
    amdf_gpu_umd_memory_reservation_t* reservation,
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    uint64_t submission, bool* mutation_started, bool* accepted) {
  (void)reservation;
  (void)queue;
  (void)access_ordinal;
  (void)update_count;
  (void)updates;
  (void)submission;
  (void)mutation_started;
  (void)accepted;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_reservation_detach(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal) {
  (void)reservation;
  (void)access_ordinal;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_memory_reservation_destroy(
    amdf_gpu_umd_memory_reservation_t* reservation) {
  (void)reservation;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static uint32_t amdf_rocjitsu_kernel_queue_mapping_capacity(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return 0;
}

static amdf_native_event_types_t
amdf_rocjitsu_kernel_queue_query_notification_types(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return 0;
}

static amdf_status_t amdf_rocjitsu_kernel_queue_request_notification(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  (void)queue;
  (void)native_submission;
  (void)event;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_gpu_umd_kernel_queue_t** out_queue) {
  (void)device;
  (void)command_type;
  (void)out_queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t submission) {
  (void)queue;
  (void)command_buffer_address;
  (void)command_buffer_byte_length;
  (void)submission;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static uint64_t amdf_rocjitsu_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return 0;
}

static amdf_status_t amdf_rocjitsu_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_kernel_queue_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  (void)queue;
  (void)native_submission;
  (void)deadline;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_rocjitsu_kernel_queue_inactivate(
    amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

const amdf_gpu_umd_api_t amdf_rocjitsu_api = {
#define AMDF_GPU_UMD_PLATFORM_OPERATION(result, name, parameters) \
  .name = amdf_rocjitsu_##name,
#define AMDF_GPU_UMD_DXG_SHARED_OPERATION AMDF_GPU_UMD_PLATFORM_OPERATION
#include "libamdf/src/gpu/umd/api_operations.inc"
#undef AMDF_GPU_UMD_DXG_SHARED_OPERATION
#undef AMDF_GPU_UMD_PLATFORM_OPERATION
};
