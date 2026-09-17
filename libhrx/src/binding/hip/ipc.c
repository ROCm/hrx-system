// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/ipc.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/internal.h"

#ifdef __linux__
#include <unistd.h>
#endif

#if defined(IREE_HIP_HAS_AMDGPU_IPC)
#include "iree/hal/drivers/amdgpu/ipc_event.h"
#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"
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

enum {
  IREE_HIP_IPC_EVENT_TYPE_NATIVE = 1,
};

typedef struct iree_hip_ipc_event_wire_t {
  // Stock HIP IPC event implementation type.
  uint32_t type;
  // Process that exported this handle.
  int32_t creator_process_id;
  // Opaque process-independent ROCr signal token.
  uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  // Stock-reserved tail ignored by import and zeroed by export.
  uint8_t reserved[24];
} iree_hip_ipc_event_wire_t;

static_assert(sizeof(iree_hip_ipc_event_wire_t) == sizeof(hipIpcEventHandle_t),
              "HIP IPC event wire format must occupy 64 bytes");
static_assert(offsetof(iree_hip_ipc_event_wire_t, type) == 0,
              "HIP IPC event type must begin at byte 0");
static_assert(offsetof(iree_hip_ipc_event_wire_t, creator_process_id) == 4,
              "HIP IPC event PID must begin at byte 4");
static_assert(offsetof(iree_hip_ipc_event_wire_t, token) == 8,
              "HIP IPC event token must begin at byte 8");
static_assert(offsetof(iree_hip_ipc_event_wire_t, reserved) == 40,
              "HIP IPC event reserved tail must begin at byte 40");

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

bool iree_hip_ipc_event_supported(void) {
#if defined(IREE_HIP_HAS_AMDGPU_IPC)
  return true;
#else
  return false;
#endif
}

iree_status_t iree_hip_ipc_event_handle_encode(
    const uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE],
    int32_t creator_process_id, hipIpcEventHandle_t* out_handle) {
  if (IREE_UNLIKELY(!out_handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must be non-NULL");
  }
  memset(out_handle, 0, sizeof(*out_handle));
  if (IREE_UNLIKELY(!token || creator_process_id <= 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid native HIP IPC event identity");
  }

  iree_hip_ipc_event_wire_t wire = {0};
  wire.type = IREE_HIP_IPC_EVENT_TYPE_NATIVE;
  wire.creator_process_id = creator_process_id;
  // ROCr defines this as an opaque hsa_amd_ipc_signal_t payload. Copy every
  // byte verbatim; no byte pattern is reserved by the public ROCr contract.
  memcpy(wire.token, token, sizeof(wire.token));
  memcpy(out_handle->reserved, &wire, sizeof(wire));
  return iree_ok_status();
}

iree_status_t iree_hip_ipc_event_handle_decode(
    hipIpcEventHandle_t handle, int32_t current_process_id,
    uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]) {
  if (out_token) {
    memset(out_token, 0, IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE);
  }
  if (IREE_UNLIKELY(!out_token)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_token must be non-NULL");
  }

  iree_hip_ipc_event_wire_t wire;
  memcpy(&wire, handle.reserved, sizeof(wire));
  if (IREE_UNLIKELY(wire.type != IREE_HIP_IPC_EVENT_TYPE_NATIVE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HIP IPC event handle type %" PRIu32,
                            wire.type);
  }
  if (IREE_UNLIKELY(wire.creator_process_id <= 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP IPC event creator process");
  }
  if (IREE_UNLIKELY(wire.creator_process_id == current_process_id)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "HIP IPC event handle came from this process");
  }

  // ROCr signal tokens are opaque. In particular, importing must not infer
  // validity from their contents before asking ROCr to attach the signal.
  memcpy(out_token, wire.token, sizeof(wire.token));
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

hipError_t iree_hip_ipc_event_operation_status_to_result(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;

  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);

  switch (code) {
    case IREE_STATUS_ABORTED:
      return hipErrorContextIsDestroyed;
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

hipError_t iree_hip_ipc_event_export_status_to_result(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  return code == IREE_STATUS_RESOURCE_EXHAUSTED ? hipErrorOutOfMemory
                                                : hipErrorInvalidValue;
}

hipError_t iree_hip_ipc_event_open_status_to_result(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  switch (code) {
    case IREE_STATUS_ABORTED:
      return hipErrorContextIsDestroyed;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    default:
      return hipErrorInvalidValue;
  }
}

#if defined(IREE_HIP_HAS_AMDGPU_IPC)
static_assert(IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE ==
                  IREE_HAL_AMDGPU_IPC_EVENT_TOKEN_SIZE,
              "streaming and AMDGPU IPC event tokens must have equal size");

typedef struct iree_hip_ipc_event_adapter_t {
  // Common event adapter interface; must be the first field.
  iree_hal_streaming_ipc_event_t base;
  // Host allocator used for this adapter and its record state.
  iree_allocator_t host_allocator;
  // Event creator's device, borrowed from the common event's retained context.
  iree_hal_device_t* device;
  // Serializes creation and publication of the first native carrier.
  iree_slim_mutex_t mutex;
  // Native carrier, lazy for a source and present for an imported event.
  iree_hal_amdgpu_ipc_event_t* native_event;
} iree_hip_ipc_event_adapter_t;

typedef struct iree_hip_ipc_event_record_state_t {
  // Adapter whose first carrier publication this operation may complete.
  iree_hip_ipc_event_adapter_t* adapter;
  // Carrier prepared for this record.
  iree_hal_amdgpu_ipc_event_t* native_event;
  // Backend record operation consumed by commit or abort.
  iree_hal_amdgpu_ipc_event_record_t* native_record;
  // True when this operation holds |adapter->mutex| and owns the unpublished
  // carrier reference.
  bool publishes_native_event;
} iree_hip_ipc_event_record_state_t;

static iree_hip_ipc_event_adapter_t* iree_hip_ipc_event_adapter_cast(
    iree_hal_streaming_ipc_event_t* base) {
  return (iree_hip_ipc_event_adapter_t*)base;
}

static void iree_hip_ipc_event_adapter_destroy(
    iree_hal_streaming_ipc_event_t* base) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  const iree_allocator_t host_allocator = adapter->host_allocator;
  iree_hal_amdgpu_ipc_event_release(adapter->native_event);
  iree_slim_mutex_deinitialize(&adapter->mutex);
  iree_allocator_free(host_allocator, adapter);
}

static iree_status_t iree_hip_ipc_event_adapter_begin_record(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_semaphore_t* recorded_semaphore, uint64_t recorded_value,
    iree_hal_streaming_ipc_event_record_state_t* out_record_state) {
  *out_record_state = NULL;
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);

  iree_hip_ipc_event_record_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(adapter->host_allocator,
                                             sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->adapter = adapter;

  // A first record keeps the adapter lock through queue admission. An export
  // or another first record therefore cannot publish a second carrier while
  // this record's speculative carrier may still be rejected.
  iree_slim_mutex_lock(&adapter->mutex);
  state->native_event = adapter->native_event;
  if (state->native_event) {
    iree_slim_mutex_unlock(&adapter->mutex);
  } else {
    iree_status_t status =
        iree_hal_amdgpu_ipc_event_create(adapter->device, &state->native_event);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&adapter->mutex);
      iree_allocator_free(adapter->host_allocator, state);
      return status;
    }
    state->publishes_native_event = true;
  }

  iree_status_t status = iree_hal_amdgpu_ipc_event_record_prepare(
      state->native_event, recorded_semaphore, recorded_value,
      &state->native_record);
  if (!iree_status_is_ok(status)) {
    if (state->publishes_native_event) {
      iree_hal_amdgpu_ipc_event_release(state->native_event);
      iree_slim_mutex_unlock(&adapter->mutex);
    }
    iree_allocator_free(adapter->host_allocator, state);
    return status;
  }

  *out_record_state = state;
  return iree_ok_status();
}

static void iree_hip_ipc_event_adapter_commit_record(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_record_state_t record_state) {
  (void)base;
  iree_hip_ipc_event_record_state_t* state =
      (iree_hip_ipc_event_record_state_t*)record_state;
  if (!state) return;

  if (state->publishes_native_event) {
    IREE_ASSERT(!state->adapter->native_event);
    state->adapter->native_event = state->native_event;
  }
  iree_hal_amdgpu_ipc_event_record_commit(state->native_record);
  if (state->publishes_native_event) {
    iree_slim_mutex_unlock(&state->adapter->mutex);
  }
  iree_allocator_free(state->adapter->host_allocator, state);
}

static void iree_hip_ipc_event_adapter_abort_record(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_record_state_t record_state) {
  (void)base;
  iree_hip_ipc_event_record_state_t* state =
      (iree_hip_ipc_event_record_state_t*)record_state;
  if (!state) return;

  iree_hal_amdgpu_ipc_event_record_abort(state->native_record);
  if (state->publishes_native_event) {
    // Keep creation serialized until the rejected carrier has been destroyed;
    // there is never a second speculative first carrier in flight.
    iree_hal_amdgpu_ipc_event_release(state->native_event);
    iree_slim_mutex_unlock(&state->adapter->mutex);
  }
  iree_allocator_free(state->adapter->host_allocator, state);
}

static iree_status_t iree_hip_ipc_event_adapter_query(
    iree_hal_streaming_ipc_event_t* base, int* out_status) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  *out_status = 0;

  iree_slim_mutex_lock(&adapter->mutex);
  iree_hal_amdgpu_ipc_event_t* native_event = adapter->native_event;
  iree_slim_mutex_unlock(&adapter->mutex);
  if (!native_event) return iree_ok_status();

  bool reached = false;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_ipc_event_query(native_event, &reached));
  *out_status = reached ? 0 : 1;
  return iree_ok_status();
}

static iree_status_t iree_hip_ipc_event_adapter_synchronize(
    iree_hal_streaming_ipc_event_t* base) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  iree_slim_mutex_lock(&adapter->mutex);
  iree_hal_amdgpu_ipc_event_t* native_event = adapter->native_event;
  iree_slim_mutex_unlock(&adapter->mutex);
  return native_event ? iree_hal_amdgpu_ipc_event_wait(native_event)
                      : iree_ok_status();
}

static iree_status_t iree_hip_ipc_event_adapter_reserve_wait(
    iree_hal_streaming_ipc_event_t* base, iree_hal_device_t* destination_device,
    iree_hal_semaphore_t** out_semaphore, uint64_t* out_value,
    iree_hal_streaming_ipc_event_wait_state_t* out_wait_state) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  iree_slim_mutex_lock(&adapter->mutex);
  iree_hal_amdgpu_ipc_event_t* native_event = adapter->native_event;
  iree_slim_mutex_unlock(&adapter->mutex);
  if (native_event) {
    // Reject an event abandoned by monitor shutdown before allocating a wait
    // point or mutating the destination queue. Pending and reached events
    // continue through the transactional wait path unchanged; a concurrent
    // transition to abandoned is rechecked when the wait is armed.
    bool reached = false;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_ipc_event_query(native_event, &reached));
  }

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {0};
  iree_hal_amdgpu_ipc_event_wait_t* native_wait = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_ipc_event_wait_reserve(
      destination_device, &wait_point, &native_wait));
  *out_semaphore = wait_point.semaphore;
  *out_value = wait_point.value;
  *out_wait_state = native_wait;
  return iree_ok_status();
}

static bool iree_hip_ipc_event_adapter_arm_wait(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  iree_slim_mutex_lock(&adapter->mutex);
  iree_hal_amdgpu_ipc_event_t* native_event = adapter->native_event;
  iree_slim_mutex_unlock(&adapter->mutex);
  return native_event &&
         iree_hal_amdgpu_ipc_event_wait_arm(
             native_event, (iree_hal_amdgpu_ipc_event_wait_t*)wait_state);
}

static iree_status_t iree_hip_ipc_event_adapter_commit_wait(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  (void)base;
  return iree_hal_amdgpu_ipc_event_wait_commit(
      (iree_hal_amdgpu_ipc_event_wait_t*)wait_state);
}

static void iree_hip_ipc_event_adapter_abort_wait(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  (void)base;
  iree_hal_amdgpu_ipc_event_wait_abort(
      (iree_hal_amdgpu_ipc_event_wait_t*)wait_state);
}

static iree_status_t iree_hip_ipc_event_adapter_export_token(
    iree_hal_streaming_ipc_event_t* base,
    uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]) {
  iree_hip_ipc_event_adapter_t* adapter = iree_hip_ipc_event_adapter_cast(base);
  memset(out_token, 0, IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE);

  iree_slim_mutex_lock(&adapter->mutex);
  iree_hal_amdgpu_ipc_event_t* native_event = adapter->native_event;
  bool publishes_native_event = false;
  iree_status_t status = iree_ok_status();
  if (!native_event) {
    status = iree_hal_amdgpu_ipc_event_create(adapter->device, &native_event);
    publishes_native_event = iree_status_is_ok(status);
  }

  iree_hal_amdgpu_ipc_event_token_t token = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_ipc_event_export(native_event, &token);
  }
  if (publishes_native_event) {
    if (iree_status_is_ok(status)) {
      adapter->native_event = native_event;
    } else {
      iree_hal_amdgpu_ipc_event_release(native_event);
    }
  }
  iree_slim_mutex_unlock(&adapter->mutex);

  if (iree_status_is_ok(status)) {
    memcpy(out_token, token.data, sizeof(token.data));
  }
  return status;
}

static const iree_hal_streaming_ipc_event_ops_t iree_hip_ipc_event_adapter_ops =
    {
        .destroy = iree_hip_ipc_event_adapter_destroy,
        .begin_record = iree_hip_ipc_event_adapter_begin_record,
        .commit_record = iree_hip_ipc_event_adapter_commit_record,
        .abort_record = iree_hip_ipc_event_adapter_abort_record,
        .query = iree_hip_ipc_event_adapter_query,
        .synchronize = iree_hip_ipc_event_adapter_synchronize,
        .reserve_wait = iree_hip_ipc_event_adapter_reserve_wait,
        .arm_wait = iree_hip_ipc_event_adapter_arm_wait,
        .commit_wait = iree_hip_ipc_event_adapter_commit_wait,
        .abort_wait = iree_hip_ipc_event_adapter_abort_wait,
        .export_token = iree_hip_ipc_event_adapter_export_token,
};

static iree_status_t iree_hip_ipc_event_adapter_attach(
    iree_hal_streaming_event_t* event,
    iree_hal_amdgpu_ipc_event_t* native_event) {
  iree_hip_ipc_event_adapter_t* adapter = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      event->host_allocator, sizeof(*adapter), (void**)&adapter));
  memset(adapter, 0, sizeof(*adapter));
  adapter->base.ops = &iree_hip_ipc_event_adapter_ops;
  adapter->host_allocator = event->host_allocator;
  adapter->device = event->context->device;
  iree_slim_mutex_initialize(&adapter->mutex);
  adapter->native_event = native_event;

  iree_status_t status =
      iree_hal_streaming_event_attach_ipc_event(event, &adapter->base);
  if (!iree_status_is_ok(status)) {
    // Ownership of an imported carrier transfers only with the adapter.
    adapter->native_event = NULL;
    iree_hip_ipc_event_adapter_destroy(&adapter->base);
  }
  return status;
}

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

iree_status_t iree_hip_ipc_event_attach_source(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
#if !defined(IREE_HIP_HAS_AMDGPU_IPC)
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HIP IPC events require the AMDGPU HAL driver");
#else
  return iree_hip_ipc_event_adapter_attach(event, /*native_event=*/NULL);
#endif
}

iree_status_t iree_hip_ipc_event_export(iree_hal_streaming_event_t* event,
                                        hipIpcEventHandle_t* out_handle) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(out_handle);
  memset(out_handle, 0, sizeof(*out_handle));

  uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_streaming_event_export_ipc_token(event, token));
  return iree_hip_ipc_event_handle_encode(token, iree_hip_ipc_process_id(),
                                          out_handle);
}

iree_status_t iree_hip_ipc_event_import(
    iree_hal_streaming_context_t* context, hipIpcEventHandle_t handle,
    iree_hal_streaming_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(context);
  if (IREE_UNLIKELY(!out_event)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_event must be non-NULL");
  }
  *out_event = NULL;

  uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE] = {0};
  IREE_RETURN_IF_ERROR(iree_hip_ipc_event_handle_decode(
      handle, iree_hip_ipc_process_id(), token));

#if !defined(IREE_HIP_HAS_AMDGPU_IPC)
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HIP IPC events require the AMDGPU HAL driver");
#else
  iree_hal_streaming_event_t* event = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context,
      IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING |
          IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS,
      context->host_allocator, &event);

  iree_hal_amdgpu_ipc_event_t* native_event = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_ipc_event_token_t native_token = {0};
    memcpy(native_token.data, token, sizeof(token));
    status = iree_hal_amdgpu_ipc_event_import(context->device, native_token,
                                              &native_event);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_ipc_event_adapter_attach(event, native_event);
    if (iree_status_is_ok(status)) native_event = NULL;
  }

  iree_hal_amdgpu_ipc_event_release(native_event);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_event_release(event);
    return status;
  }
  *out_event = event;
  return iree_ok_status();
#endif
}

void iree_hip_ipc_event_shutdown(void) {
#if defined(IREE_HIP_HAS_AMDGPU_IPC)
  // Keep this as the only binding dependency on the backend monitor API.
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
#endif
}
