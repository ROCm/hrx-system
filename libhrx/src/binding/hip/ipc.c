// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/ipc.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/internal.h"

#ifdef __linux__
#include <unistd.h>
#endif

#if defined(IREE_HIP_HAS_AMDGPU_IPC)
#include "iree/hal/drivers/amdgpu/ipc_memory.h"
#endif

typedef struct iree_hip_ipc_memory_wire_t {
  // Opaque process-independent ROCr memory token.
  uint8_t token[IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE];
  // Requested ROCr shareable/addressable extent required by attach, in bytes.
  uint64_t allocation_size;
  // Byte offset from the attached ROCr allocation base.
  uint64_t byte_offset;
  // Process that created the exported allocation.
  int32_t creator_process_id;
  // HIP-visible ordinal of the exporting device.
  int32_t device_ordinal;
  // Stock-reserved tail ignored by import and zeroed by export.
  uint8_t reserved[8];
} iree_hip_ipc_memory_wire_t;

static_assert(sizeof(size_t) == 8 && sizeof(void*) == 8,
              "HIP IPC memory wire format requires a 64-bit target");
static_assert(sizeof(iree_hip_ipc_memory_wire_t) == sizeof(hipIpcMemHandle_t),
              "HIP IPC memory wire format must occupy 64 bytes");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, token) == 0,
              "HIP IPC memory token must begin at byte 0");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, allocation_size) == 32,
              "HIP IPC memory size must begin at byte 32");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, byte_offset) == 40,
              "HIP IPC memory offset must begin at byte 40");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, creator_process_id) == 48,
              "HIP IPC memory PID must begin at byte 48");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, device_ordinal) == 52,
              "HIP IPC memory device ordinal must begin at byte 52");
static_assert(offsetof(iree_hip_ipc_memory_wire_t, reserved) == 56,
              "HIP IPC memory reserved tail must begin at byte 56");

static int32_t iree_hip_ipc_process_id(void) {
#ifdef __linux__
  return (int32_t)getpid();
#else
  return -1;
#endif
}

iree_status_t iree_hip_ipc_memory_handle_encode(
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_host_size_t device_ordinal, int32_t creator_process_id,
    hipIpcMemHandle_t* out_handle) {
  if (IREE_UNLIKELY(!out_handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must be non-NULL");
  }
  memset(out_handle, 0, sizeof(*out_handle));
  if (IREE_UNLIKELY(!descriptor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor must be non-NULL");
  }
  if (IREE_UNLIKELY(descriptor->allocation_size == 0 ||
                    descriptor->allocation_size > (uint64_t)SIZE_MAX ||
                    descriptor->byte_offset >= descriptor->allocation_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP IPC memory allocation view");
  }
  if (IREE_UNLIKELY(device_ordinal > INT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exporting device ordinal exceeds HIP range");
  }

  iree_hip_ipc_memory_wire_t wire = {0};
  memcpy(wire.token, descriptor->token, sizeof(wire.token));
  wire.allocation_size = descriptor->allocation_size;
  wire.byte_offset = descriptor->byte_offset;
  wire.creator_process_id = creator_process_id;
  wire.device_ordinal = (int32_t)device_ordinal;
  memcpy(out_handle->reserved, &wire, sizeof(wire));
  return iree_ok_status();
}

iree_status_t iree_hip_ipc_memory_handle_decode(
    hipIpcMemHandle_t handle, int32_t current_process_id,
    iree_host_size_t visible_device_count,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor,
    iree_device_size_t* out_view_size) {
  if (out_descriptor) memset(out_descriptor, 0, sizeof(*out_descriptor));
  if (out_view_size) *out_view_size = 0;
  if (IREE_UNLIKELY(!out_descriptor || !out_view_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "out_descriptor and out_view_size must be non-NULL");
  }

  iree_hip_ipc_memory_wire_t wire;
  memcpy(&wire, handle.reserved, sizeof(wire));
  if (IREE_UNLIKELY(wire.allocation_size == 0 ||
                    wire.allocation_size > (uint64_t)SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP IPC memory size");
  }
  // Match stock HIP's validation order: a handle from this process reports an
  // invalid context without interpreting its remaining placement metadata.
  if (IREE_UNLIKELY(wire.creator_process_id == current_process_id)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "HIP IPC memory handle came from this process");
  }
  if (IREE_UNLIKELY(wire.device_ordinal < 0 ||
                    (iree_host_size_t)wire.device_ordinal >=
                        visible_device_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP IPC exporting device is not visible");
  }
  if (IREE_UNLIKELY(wire.byte_offset >= wire.allocation_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP IPC memory offset");
  }

  memcpy(out_descriptor->token, wire.token, sizeof(wire.token));
  out_descriptor->allocation_size = wire.allocation_size;
  out_descriptor->byte_offset = wire.byte_offset;
  out_descriptor->exporter_process_id = wire.creator_process_id;
  out_descriptor->exporter_device_ordinal = (uint64_t)wire.device_ordinal;
  *out_view_size =
      (iree_device_size_t)(wire.allocation_size - wire.byte_offset);
  return iree_ok_status();
}

hipError_t iree_hip_ipc_memory_export_status_to_result(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;

  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  if (code == IREE_STATUS_UNIMPLEMENTED) return hipErrorNotSupported;
  if (code == IREE_STATUS_RESOURCE_EXHAUSTED) return hipErrorOutOfMemory;
  return hipErrorInvalidValue;
}

#if defined(IREE_HIP_HAS_AMDGPU_IPC)
static_assert(IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE ==
                  IREE_HAL_AMDGPU_IPC_MEMORY_TOKEN_SIZE,
              "streaming and AMDGPU IPC memory tokens must have equal size");

static iree_status_t iree_hip_ipc_memory_export_descriptor(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor) {
  iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_ipc_memory_export(
      context->device_allocator, buffer, &descriptor));
  memcpy(out_descriptor->token, descriptor.token.data,
         sizeof(out_descriptor->token));
  out_descriptor->allocation_size = descriptor.allocation_size;
  out_descriptor->byte_offset = descriptor.byte_offset;
  return iree_ok_status();
}

static iree_status_t iree_hip_ipc_memory_attach(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_streaming_device_t* exporting_device =
      iree_hal_streaming_device_entry(
          (iree_host_size_t)descriptor->exporter_device_ordinal);
  if (IREE_UNLIKELY(!exporting_device || !exporting_device->hal_device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP IPC exporting device is unavailable");
  }
  iree_hal_amdgpu_ipc_memory_descriptor_t amdgpu_descriptor = {
      .allocation_size = descriptor->allocation_size,
      .byte_offset = descriptor->byte_offset,
  };
  memcpy(amdgpu_descriptor.token.data, descriptor->token,
         sizeof(descriptor->token));
  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  iree_status_t status = iree_hal_amdgpu_ipc_memory_import(
      context->device_allocator,
      iree_hal_device_allocator(exporting_device->hal_device),
      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, params, &amdgpu_descriptor, view_size,
      iree_hal_buffer_release_callback_null(), out_buffer);
  // HIP exposes backend attachment failures as an invalid device pointer.
  // Preserve allocation failures originating inside the typed AMDGPU import.
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) != IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  return status;
}

static iree_status_t iree_hip_ipc_memory_import_alias(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* attached_buffer,
    iree_hal_buffer_t** out_alias_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  return iree_hal_amdgpu_ipc_memory_import_alias(
      context->device_allocator, params, attached_buffer, out_alias_buffer);
}
#endif  // IREE_HIP_HAS_AMDGPU_IPC

iree_status_t iree_hip_ipc_memory_export(iree_hal_streaming_context_t* context,
                                         void* device_ptr,
                                         hipIpcMemHandle_t* out_handle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_handle);
  memset(out_handle, 0, sizeof(*out_handle));
#if !defined(IREE_HIP_HAS_AMDGPU_IPC)
  (void)device_ptr;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HIP IPC memory requires the AMDGPU HAL driver");
#else
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (IREE_UNLIKELY(!device_registry)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }
  iree_hal_streaming_ipc_memory_descriptor_t descriptor = {0};
  iree_host_size_t device_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_ipc_memory_export(
      device_registry, context, (uint64_t)(uintptr_t)device_ptr,
      iree_hip_ipc_memory_export_descriptor, &descriptor, &device_ordinal));
  return iree_hip_ipc_memory_handle_encode(
      &descriptor, device_ordinal, iree_hip_ipc_process_id(), out_handle);
#endif
}

iree_status_t iree_hip_ipc_memory_import(iree_hal_streaming_context_t* context,
                                         hipIpcMemHandle_t handle,
                                         void** out_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = NULL;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (IREE_UNLIKELY(!device_registry)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_hal_streaming_ipc_memory_descriptor_t descriptor = {0};
  iree_device_size_t view_size = 0;
  IREE_RETURN_IF_ERROR(iree_hip_ipc_memory_handle_decode(
      handle, iree_hip_ipc_process_id(), device_registry->device_count,
      &descriptor, &view_size));

#if !defined(IREE_HIP_HAS_AMDGPU_IPC)
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HIP IPC memory requires the AMDGPU HAL driver");
#else
  return iree_hal_streaming_ipc_memory_import(
      &device_registry->ipc_memory_registry, context, &descriptor, view_size,
      iree_hip_ipc_memory_attach, iree_hip_ipc_memory_import_alias,
      out_device_ptr);
#endif
}

iree_status_t iree_hip_ipc_memory_close(iree_hal_streaming_context_t* context,
                                        void* device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (IREE_UNLIKELY(!device_registry)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }
#if defined(IREE_HIP_HAS_AMDGPU_IPC)
  return iree_hal_streaming_ipc_memory_close(
      &device_registry->ipc_memory_registry, context, device_ptr);
#else
  (void)device_ptr;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HIP IPC memory requires the AMDGPU HAL driver");
#endif
}
