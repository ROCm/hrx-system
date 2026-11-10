// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file intentionally has no include guard. The owning API uses the two
// operation macros to generate its table, provider declarations, and complete
// initializers from one signature list.

// Prepares instance-owned native state.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, instance_prepare,
                                  (amdf_gpu_umd_instance_t * *state,
                                   amdf_native_lifetime_t native_lifetime,
                                   amdf_allocator_t host_allocator))
// Releases instance-owned native state.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, instance_destroy,
                                  (amdf_gpu_umd_instance_t * instance))

// Creates immutable endpoint metadata.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, create_endpoint_profile,
                                (void* endpoint_context,
                                 amdf_allocator_t host_allocator,
                                 amdf_gpu_endpoint_profile_t** out_profile))
// Samples endpoint-local physical storage.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, sample_memory_scope,
    (const void* endpoint_context, amdf_memory_scope_flags_t flags,
     amdf_memory_scope_observation_t* out_observation))

// Creates one native execution device.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, device_create,
                                (amdf_gpu_umd_instance_t * instance,
                                 void* endpoint_context,
                                 amdf_allocator_t host_allocator,
                                 amdf_native_lifetime_t native_lifetime,
                                 amdf_gpu_umd_device_t** out_device,
                                 amdf_gpu_umd_device_result_t* out_result))
// Releases one native execution device.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, device_destroy,
                                (amdf_gpu_umd_device_t * device))
// Copies one native memory profile.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, device_query_memory_profile,
                                (amdf_gpu_umd_device_t * device,
                                 uint32_t memory_profile_ordinal,
                                 amdf_memory_native_profile_t* out_profile))
// Reads one native queue clock description.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, query_queue_clock_info,
                                (amdf_gpu_umd_device_t * device,
                                 amdf_queue_command_type_t command_type,
                                 amdf_observation_clock_info_t* out_info))
// Samples one native queue and host clock pair.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, query_clock_sample,
                                (amdf_gpu_umd_device_t * device,
                                 amdf_queue_command_type_t command_type,
                                 amdf_observation_clock_sample_t* out_sample))

// Prepares ordinary native memory.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_prepare,
    (amdf_gpu_umd_device_t * device, uint32_t peer_count,
     amdf_gpu_umd_device_t* const* peer_devices,
     const amdf_memory_native_profile_t* profile,
     const amdf_memory_native_create_info_t* create_info,
     amdf_gpu_umd_memory_t** memory_state,
     amdf_gpu_umd_memory_result_t* out_result))
// Imports externally owned native memory.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_prepare_import,
    (amdf_gpu_umd_device_t * device,
     const amdf_memory_native_profile_t* profile,
     const amdf_memory_native_import_info_t* import_info,
     const amdf_external_memory_t* external_memory,
     amdf_gpu_umd_memory_t** memory_state,
     amdf_gpu_umd_memory_result_t* out_result))
// Exports one native memory payload.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, memory_export,
                                (amdf_gpu_umd_memory_t * memory,
                                 const amdf_memory_export_info_t* export_info,
                                 amdf_external_memory_t* out_value))
// Borrows the persistent host-mapping source.
AMDF_GPU_UMD_PLATFORM_OPERATION(uintptr_t, memory_host_mapping_source,
                                (amdf_gpu_umd_memory_t * memory))
// Releases complete or partial native memory.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, memory_destroy,
                                (amdf_gpu_umd_memory_t * memory))
// Reports backing retained by partial native memory.
AMDF_GPU_UMD_PLATFORM_OPERATION(uint64_t, memory_remaining_backing_byte_length,
                                (const amdf_gpu_umd_memory_t* memory))
// Abandons metadata after terminal native cleanup failure.
AMDF_GPU_UMD_PLATFORM_OPERATION(void, memory_abandon,
                                (amdf_gpu_umd_memory_t * memory))
// Creates one native host view.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_map,
    (amdf_gpu_umd_memory_t * memory,
     const amdf_host_mapping_capabilities_t* capabilities,
     const amdf_memory_map_info_t* map_info,
     amdf_gpu_umd_host_mapping_t** out_mapping,
     amdf_gpu_umd_host_mapping_result_t* out_result))
// Performs explicit host cache control.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, host_mapping_cache_control,
                                (amdf_gpu_umd_host_mapping_t * mapping,
                                 amdf_host_cache_operation_t operation,
                                 uint64_t memory_byte_offset,
                                 uint64_t byte_length))
// Releases one lightweight native host view.
AMDF_GPU_UMD_PLATFORM_OPERATION(void, host_mapping_destroy,
                                (amdf_gpu_umd_host_mapping_t * mapping))

// Tests whether two devices share one native page-table domain.
AMDF_GPU_UMD_PLATFORM_OPERATION(bool, devices_share_mapping_domain,
                                (const amdf_gpu_umd_device_t* a,
                                 const amdf_gpu_umd_device_t* b))
// Copies native virtual-memory reservation limits.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, query_memory_reservation_capabilities,
    (amdf_gpu_umd_device_t * device, uint32_t address_domain_ordinal,
     amdf_memory_reservation_capabilities_t* out_capabilities))
// Creates one native virtual-memory reservation.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_reservation_create,
    (amdf_allocator_t host_allocator, uint32_t device_count,
     amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
     uint64_t minimum_alignment, uint64_t common_address,
     amdf_gpu_umd_memory_reservation_t** out_reservation))
// Copies one reservation access description.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    void, memory_reservation_query_access,
    (const amdf_gpu_umd_memory_reservation_t* reservation,
     uint32_t access_ordinal, uint64_t* out_address,
     uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
     amdf_memory_access_t* out_required_access))
// Applies host-published native mappings.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, memory_reservation_update,
                                (amdf_gpu_umd_memory_reservation_t *
                                     reservation,
                                 uint32_t access_ordinal, uint32_t update_count,
                                 const amdf_gpu_umd_mapping_update_t* updates,
                                 bool* mutation_started))
// Applies queue-ordered native mappings.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_reservation_update_on_queue,
    (amdf_gpu_umd_memory_reservation_t * reservation,
     amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
     uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
     uint64_t submission, bool* mutation_started, bool* accepted))
// Detaches every mapping in one native group.
AMDF_GPU_UMD_PLATFORM_OPERATION(
    amdf_status_t, memory_reservation_detach,
    (amdf_gpu_umd_memory_reservation_t * reservation, uint32_t access_ordinal))
// Releases one native virtual-memory reservation.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, memory_reservation_destroy,
                                (amdf_gpu_umd_memory_reservation_t *
                                 reservation))

// Creates one kernel-mediated queue.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, kernel_queue_create,
                                (amdf_gpu_umd_device_t * device,
                                 amdf_queue_command_type_t command_type,
                                 amdf_gpu_umd_kernel_queue_t** out_queue))
// Returns queue-ordered mapping capacity.
AMDF_GPU_UMD_PLATFORM_OPERATION(uint32_t, kernel_queue_mapping_capacity,
                                (const amdf_gpu_umd_kernel_queue_t* queue))
// Returns supported native notification types.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_native_event_types_t,
                                kernel_queue_query_notification_types,
                                (const amdf_gpu_umd_kernel_queue_t* queue))
// Requests one native progress notification.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t,
                                kernel_queue_request_notification,
                                (amdf_gpu_umd_kernel_queue_t * queue,
                                 uint64_t native_submission,
                                 const amdf_native_event_t* event))
// Submits one native command range.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, kernel_queue_submit,
                                (amdf_gpu_umd_kernel_queue_t * queue,
                                 uint64_t command_buffer_address,
                                 uint64_t command_buffer_byte_length,
                                 uint64_t submission))
// Samples one native progress fence.
AMDF_GPU_UMD_PLATFORM_OPERATION(uint64_t, kernel_queue_query_progress,
                                (const amdf_gpu_umd_kernel_queue_t* queue))
// Reads cached terminal queue status.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t,
                                kernel_queue_query_terminal_status,
                                (const amdf_gpu_umd_kernel_queue_t* queue))
// Waits for one accepted native submission.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, kernel_queue_wait,
                                (amdf_gpu_umd_kernel_queue_t * queue,
                                 uint64_t native_submission,
                                 const amdf_wait_deadline_t* deadline))
// Releases one idle native queue.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, kernel_queue_destroy,
                                (amdf_gpu_umd_kernel_queue_t * queue))
// Stops native queue access without consuming metadata.
AMDF_GPU_UMD_PLATFORM_OPERATION(amdf_status_t, kernel_queue_inactivate,
                                (amdf_gpu_umd_kernel_queue_t * queue))

// Creates one directly published queue.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(
    amdf_status_t, user_queue_create,
    (amdf_gpu_umd_device_t * device,
     const amdf_gpu_umd_user_queue_create_info_t* create_info,
     amdf_gpu_umd_user_queue_t** out_queue,
     amdf_gpu_umd_user_queue_result_t* out_result))
// Maps a queue into one producer domain.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(
    amdf_status_t, user_queue_map,
    (amdf_gpu_umd_user_queue_t * queue, amdf_gpu_umd_device_t* producer_device,
     amdf_gpu_umd_user_queue_mapping_t** out_mapping,
     amdf_gpu_umd_user_queue_mapping_result_t* out_result))
// Releases one producer-local queue mapping.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_mapping_destroy,
                                  (amdf_gpu_umd_user_queue_mapping_t * mapping))
// Flushes host stores through one queue mapping.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_mapping_flush,
                                  (amdf_gpu_umd_user_queue_mapping_t * mapping))
// Samples one native queue's status.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_query_status,
                                  (amdf_gpu_umd_user_queue_t * queue,
                                   amdf_user_queue_status_t* out_status))
// Samples one firmware scratch request.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_query_scratch,
                                  (amdf_gpu_umd_user_queue_t * queue,
                                   amdf_gpu_aql_scratch_request_t* out_request))
// Installs queue scratch and resumes firmware.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_resume_scratch,
                                  (amdf_gpu_umd_user_queue_t * queue,
                                   const amdf_gpu_umd_queue_scratch_t* scratch))
// Detaches idle queue scratch.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_trim_scratch,
                                  (amdf_gpu_umd_user_queue_t * queue))
// Selects native dispatch timestamp writes.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t,
                                  user_queue_set_dispatch_timestamps,
                                  (amdf_gpu_umd_user_queue_t * queue,
                                   bool enabled))
// Waits for one directly published queue index.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_wait_consumed,
                                  (amdf_gpu_umd_user_queue_t * queue,
                                   uint64_t published_index,
                                   const amdf_wait_deadline_t* deadline))
// Releases one idle directly published queue.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_destroy,
                                  (amdf_gpu_umd_user_queue_t * queue))
// Stops direct queue access without consuming metadata.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, user_queue_inactivate,
                                  (amdf_gpu_umd_user_queue_t * queue))

// Creates one native producer event.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, event_create,
                                  (amdf_gpu_umd_device_t * device,
                                   amdf_gpu_umd_event_t** out_event,
                                   amdf_gpu_umd_event_result_t* out_result))
// Signals one native event.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, event_signal,
                                  (amdf_gpu_umd_event_t * event))
// Waits for one native event.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, event_wait,
                                  (amdf_gpu_umd_event_t * event,
                                   uint64_t timeout_nanoseconds))
// Releases one native event.
AMDF_GPU_UMD_DXG_SHARED_OPERATION(amdf_status_t, event_destroy,
                                  (amdf_gpu_umd_event_t * event))
