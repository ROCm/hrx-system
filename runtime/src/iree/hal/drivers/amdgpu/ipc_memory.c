// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_memory.h"

#include <stddef.h>
#include <stdint.h>

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/allocator_ipc_memory.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/buffer.h"

static_assert(sizeof(iree_hal_amdgpu_ipc_memory_token_t) ==
                  sizeof(hsa_amd_ipc_memory_t),
              "AMDGPU IPC memory token must match the ROCr token size");

typedef struct iree_hal_amdgpu_imported_ipc_memory_release_data_t {
  // Unowned libhsa handle used to detach the imported ROCr mapping.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Base pointer returned by the single matching ROCr attach.
  void* mapped_ptr;

  // Host allocator used to release this thunk after buffer destruction.
  iree_allocator_t host_allocator;

  // Optional caller callback invoked after the mapping has been detached.
  iree_hal_buffer_release_callback_t caller_release_callback;

  // Immutable agents to which the process-wide mapping was attached.
  uint32_t mapping_agent_count;

  // Exact agents admitted by the exporting allocation's memory pool.
  hsa_agent_t mapping_agents[];
} iree_hal_amdgpu_imported_ipc_memory_release_data_t;

static void iree_hal_amdgpu_ipc_memory_release_import(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_amdgpu_imported_ipc_memory_release_data_t* data =
      (iree_hal_amdgpu_imported_ipc_memory_release_data_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_ipc_memory_detach_raw(data->libhsa, data->mapped_ptr));
  if (data->caller_release_callback.fn) {
    data->caller_release_callback.fn(data->caller_release_callback.user_data,
                                     buffer);
  }
  iree_allocator_free(data->host_allocator, data);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_ipc_memory_import_impl(
    iree_hal_allocator_t* allocator,
    const iree_hal_amdgpu_ipc_memory_allocator_context_t* allocator_context,
    const iree_hal_buffer_params_t* params,
    const iree_hal_amdgpu_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size, hsa_amd_memory_pool_t exporting_memory_pool,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  if (IREE_UNLIKELY(descriptor->allocation_size == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC memory import requires a non-zero allocation size");
  }
  if (IREE_UNLIKELY(view_size == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC memory import requires a non-zero buffer view size");
  }
  if (IREE_UNLIKELY(descriptor->allocation_size > (uint64_t)SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC memory allocation size exceeds the HSA size range");
  }
  if (IREE_UNLIKELY(descriptor->allocation_size >
                        (uint64_t)IREE_DEVICE_SIZE_MAX ||
                    descriptor->byte_offset > (uint64_t)IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC memory descriptor exceeds the HAL device size range");
  }
  iree_device_size_t view_end = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
                        (iree_device_size_t)descriptor->byte_offset, view_size,
                        &view_end) ||
                    view_end > descriptor->allocation_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC memory buffer view exceeds the shared allocation range");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_allocator_validate_ipc_memory_import_params(allocator,
                                                                  params));

  iree_hal_amdgpu_access_agent_list_t mapping_agents;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_access_agent_list_resolve_ipc_memory_agents(
          allocator_context->libhsa, allocator_context->topology,
          params->queue_family_affinity, exporting_memory_pool,
          &mapping_agents));

  iree_host_size_t mapping_agent_storage_size = 0;
  iree_host_size_t release_data_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(mapping_agents.count,
                                      sizeof(mapping_agents.values[0]),
                                      &mapping_agent_storage_size) ||
          !iree_host_size_checked_add(
              sizeof(iree_hal_amdgpu_imported_ipc_memory_release_data_t),
              mapping_agent_storage_size, &release_data_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU IPC mapping agent storage size overflow");
  }
  iree_hal_amdgpu_imported_ipc_memory_release_data_t* release_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator_context->host_allocator,
                                             release_data_size,
                                             (void**)&release_data));
  memset(release_data, 0, release_data_size);
  release_data->libhsa = allocator_context->libhsa;
  release_data->host_allocator = allocator_context->host_allocator;
  release_data->caller_release_callback = release_callback;
  release_data->mapping_agent_count = mapping_agents.count;
  memcpy(release_data->mapping_agents, mapping_agents.values,
         mapping_agent_storage_size);

  void* mapped_ptr = NULL;
  // Attach once for the importing GPUs and their accessible peers. The stored
  // list lets aliases validate membership without trying to expand ROCr's
  // immutable IPC mapping later.
  hsa_amd_ipc_memory_t ipc_memory;
  memcpy(&ipc_memory, &descriptor->token, sizeof(ipc_memory));
  iree_status_t status = iree_hsa_amd_ipc_memory_attach(
      IREE_LIBHSA(allocator_context->libhsa), &ipc_memory,
      (size_t)descriptor->allocation_size, mapping_agents.count,
      mapping_agents.values, &mapped_ptr);
  const bool mapping_attached = iree_status_is_ok(status);
  if (iree_status_is_ok(status) && IREE_UNLIKELY(!mapped_ptr)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "ROCr IPC memory attach succeeded without returning a mapping");
  }

  uintptr_t view_ptr = 0;
  if (iree_status_is_ok(status)) {
    const uintptr_t mapped_value = (uintptr_t)mapped_ptr;
    if (IREE_UNLIKELY(descriptor->byte_offset >
                      (uint64_t)(UINTPTR_MAX - mapped_value))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU IPC memory buffer view overflows the host pointer range");
    } else {
      view_ptr = mapped_value + (uintptr_t)descriptor->byte_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    release_data->mapped_ptr = mapped_ptr;
    const iree_hal_buffer_release_callback_t ipc_release_callback = {
        .fn = iree_hal_amdgpu_ipc_memory_release_import,
        .user_data = release_data,
    };
    iree_hal_external_buffer_t attached_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = view_size,
    };
    attached_buffer.handle.device_allocation.ptr = (uint64_t)view_ptr;
    status = iree_hal_amdgpu_allocator_wrap_attached_ipc_memory(
        allocator, params, &attached_buffer, /*owns_attachment=*/true,
        ipc_release_callback, out_buffer);
  }

  if (iree_status_is_ok(status)) {
    return status;  // The imported buffer now owns |release_data| and mapping.
  }
  if (mapping_attached) {
    status = iree_status_join(
        status, iree_hsa_amd_ipc_memory_detach(
                    IREE_LIBHSA(allocator_context->libhsa), mapped_ptr));
  }
  iree_allocator_free(allocator_context->host_allocator, release_data);
  return status;
}

iree_status_t iree_hal_amdgpu_ipc_memory_import(
    iree_hal_allocator_t* allocator, iree_hal_allocator_t* exporting_allocator,
    iree_hal_queue_family_affinity_t exporting_queue_family_affinity,
    iree_hal_buffer_params_t params,
    const iree_hal_amdgpu_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  if (IREE_UNLIKELY(!exporting_allocator || !descriptor || !out_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exporting_allocator, descriptor, and out_buffer "
                            "must be non-NULL");
  }
  *out_buffer = NULL;
  iree_hal_amdgpu_ipc_memory_allocator_context_t allocator_context;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_allocator_query_ipc_memory_context(
      allocator, &allocator_context));
  hsa_amd_memory_pool_t exporting_memory_pool;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_allocator_resolve_ipc_memory_export_pool(
      exporting_allocator, exporting_queue_family_affinity,
      &exporting_memory_pool));
  iree_hal_buffer_params_canonicalize(&params);
  return iree_hal_amdgpu_ipc_memory_import_impl(
      allocator, &allocator_context, &params, descriptor, view_size,
      exporting_memory_pool, release_callback, out_buffer);
}

iree_status_t iree_hal_amdgpu_ipc_memory_import_alias(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* attached_buffer, iree_hal_buffer_t** out_buffer) {
  if (IREE_UNLIKELY(!attached_buffer || !out_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attached_buffer and out_buffer must be non-NULL");
  }
  *out_buffer = NULL;
  iree_hal_amdgpu_ipc_memory_allocator_context_t allocator_context;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_allocator_query_ipc_memory_context(
      allocator, &allocator_context));
  iree_hal_buffer_params_canonicalize(&params);

  iree_hal_buffer_release_callback_t attachment_release_callback;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_allocator_query_ipc_memory_attachment(
          attached_buffer, &attachment_release_callback))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC alias requires a live attached IPC buffer");
  }
  if (IREE_UNLIKELY(attachment_release_callback.fn !=
                        iree_hal_amdgpu_ipc_memory_release_import ||
                    !attachment_release_callback.user_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC alias requires live attachment metadata");
  }
  iree_hal_amdgpu_imported_ipc_memory_release_data_t* release_data =
      (iree_hal_amdgpu_imported_ipc_memory_release_data_t*)
          attachment_release_callback.user_data;
  iree_hal_amdgpu_access_agent_list_t alias_agents;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_access_agent_list_resolve_queue_family_agents(
          allocator_context.topology, params.queue_family_affinity,
          &alias_agents));
  for (uint32_t i = 0; i < alias_agents.count; ++i) {
    bool is_mapped = false;
    for (uint32_t j = 0; j < release_data->mapping_agent_count; ++j) {
      if (alias_agents.values[i].handle ==
          release_data->mapping_agents[j].handle) {
        is_mapped = true;
        break;
      }
    }
    if (IREE_UNLIKELY(!is_mapped)) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "AMDGPU IPC mapping is not accessible to an alias device");
    }
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(attached_buffer);
  void* allocation_ptr =
      iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!allocation_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC alias requires an attached AMDGPU buffer");
  }
  const iree_device_size_t byte_offset =
      iree_hal_buffer_byte_offset(attached_buffer);
  const iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(attached_buffer);
  const uintptr_t allocation_value = (uintptr_t)allocation_ptr;
  if (IREE_UNLIKELY(byte_length == 0 ||
                    byte_offset >
                        (iree_device_size_t)(UINTPTR_MAX - allocation_value))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU IPC alias view is invalid");
  }

  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = byte_length,
  };
  external_buffer.handle.device_allocation.ptr =
      (uint64_t)(allocation_value + (uintptr_t)byte_offset);
  return iree_hal_amdgpu_allocator_wrap_attached_ipc_memory(
      allocator, &params, &external_buffer, /*owns_attachment=*/false,
      iree_hal_buffer_release_callback_null(), out_buffer);
}

static iree_status_t iree_hal_amdgpu_ipc_memory_export_impl(
    iree_hal_allocator_t* allocator,
    const iree_hal_amdgpu_ipc_memory_allocator_context_t* allocator_context,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_ipc_memory_descriptor_t* out_descriptor) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_buffer_memory_type(buffer);
  if (IREE_UNLIKELY(
          !iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU buffer memory type is not supported for IPC export");
  }
  if (IREE_UNLIKELY(iree_any_bit_set(
          memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                           IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                           IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                           IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU IPC export requires cached device-coarse memory");
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                       IREE_HAL_BUFFER_USAGE_SHARING_EXPORT))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU buffer was not allocated for external sharing");
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  void* allocation_ptr =
      iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!allocation_ptr)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU buffer has no HSA allocation pointer to export through IPC");
  }

  const iree_device_size_t allocation_size =
      iree_hal_buffer_allocation_size(allocated_buffer);
  const iree_device_size_t byte_offset = iree_hal_buffer_byte_offset(buffer);
  const iree_device_size_t byte_length = iree_hal_buffer_byte_length(buffer);
  if (IREE_UNLIKELY(allocation_size == 0 || byte_length == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC export requires non-zero allocation and buffer view sizes");
  }
  if (IREE_UNLIKELY(allocation_size > (iree_device_size_t)SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC export allocation size exceeds the HSA size range");
  }
  iree_device_size_t view_end = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_add(byte_offset, byte_length, &view_end) ||
          view_end > allocation_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC export buffer view exceeds the HAL allocation range");
  }

  const uintptr_t allocation_value = (uintptr_t)allocation_ptr;
  if (IREE_UNLIKELY(byte_offset >
                    (iree_device_size_t)(UINTPTR_MAX - allocation_value))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC export buffer view overflows the host pointer range");
  }
  const uintptr_t view_value = allocation_value + (uintptr_t)byte_offset;
  iree_hal_amdgpu_ipc_memory_export_range_t export_range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_allocator_query_ipc_memory_export_range(
      allocator, allocated_buffer, (uint64_t)view_value, byte_length,
      &export_range));

  const uint32_t pool_class = export_range.global_flags &
                              IREE_HAL_AMDGPU_ATOMIC_MEMORY_POOL_CLASS_FLAGS;
  if (IREE_UNLIKELY(pool_class !=
                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU IPC export requires a device-coarse ROCr allocation");
  }
  if (IREE_UNLIKELY(export_range.allocation_size >
                    (iree_device_size_t)SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU IPC export ROCr allocation exceeds the HSA size range");
  }

  const uintptr_t agent_base = (uintptr_t)export_range.agent_base;
  const uint64_t ipc_byte_offset = (uint64_t)(view_value - agent_base);
  hsa_amd_ipc_memory_t ipc_memory;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_ipc_memory_create(
      IREE_LIBHSA(allocator_context->libhsa), export_range.agent_base,
      (size_t)export_range.allocation_size, &ipc_memory));
  memcpy(&out_descriptor->token, &ipc_memory, sizeof(ipc_memory));
  out_descriptor->allocation_size = (uint64_t)export_range.allocation_size;
  out_descriptor->byte_offset = ipc_byte_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_ipc_memory_export(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_ipc_memory_descriptor_t* out_descriptor) {
  if (IREE_UNLIKELY(!buffer || !out_descriptor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer and out_descriptor must be non-NULL");
  }
  memset(out_descriptor, 0, sizeof(*out_descriptor));
  iree_hal_amdgpu_ipc_memory_allocator_context_t allocator_context;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_allocator_query_ipc_memory_context(
      allocator, &allocator_context));
  return iree_hal_amdgpu_ipc_memory_export_impl(allocator, &allocator_context,
                                                buffer, out_descriptor);
}
