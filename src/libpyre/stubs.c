// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Implementations for dispatch, host_call, and execution_barrier APIs.
// These were originally stubs; now fully implemented.

#include "pyre_internal.h"

#include <string.h>

// Helper: begin command buffer recording on a stream (same as stream.c).
static pyre_status_t pyre_dispatch_begin_cb(pyre_stream_t stream) {
  if (stream->pending_cb) return pyre_ok_status();
  iree_status_t status = iree_hal_command_buffer_create(
      stream->device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER |
          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, 0, &stream->pending_cb);
  if (!iree_status_is_ok(status)) return pyre_status_from_iree(status);
  status = iree_hal_command_buffer_begin(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(stream->pending_cb);
    stream->pending_cb = NULL;
    return pyre_status_from_iree(status);
  }
  return pyre_ok_status();
}

// Convert pyre_buffer_ref_t array to iree_hal_buffer_ref_t array.
static void pyre_convert_bindings(const pyre_buffer_ref_t* bindings,
                                  size_t count,
                                  iree_hal_buffer_ref_t* out) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = iree_hal_make_buffer_ref(
        bindings[i].buffer ? bindings[i].buffer->hal_buffer : NULL,
        (iree_device_size_t)bindings[i].offset,
        (iree_device_size_t)bindings[i].length);
  }
}

#define PYRE_MAX_DISPATCH_BINDINGS 64
#define PYRE_MAX_QUEUE_SEMAPHORES 16

static iree_hal_semaphore_list_t pyre_build_iree_semaphore_list(
    const pyre_semaphore_list_t* list,
    iree_hal_semaphore_t** hal_sems, uint64_t* vals) {
  iree_hal_semaphore_list_t result = {0};
  if (!list || list->count == 0) return result;
  for (size_t i = 0; i < list->count; ++i) {
    hal_sems[i] = list->semaphores[i]->hal_semaphore;
    vals[i] = list->values[i];
  }
  result.count = (iree_host_size_t)list->count;
  result.semaphores = hal_sems;
  result.payload_values = vals;
  return result;
}

pyre_status_t pyre_queue_dispatch(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_executable_t executable, uint32_t export_ordinal,
    const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  if (!device || !config) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  // executable is type-punned from pyre_module_t.
  pyre_module_t module = (pyre_module_t)executable;
  if (!module || !module->executable) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "invalid executable");
  }

  // Create one-shot command buffer.
  iree_hal_command_buffer_t* cb = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH |
          IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      (iree_hal_queue_affinity_t)affinity, 0, &cb);
  if (!iree_status_is_ok(status)) return pyre_status_from_iree(status);

  status = iree_hal_command_buffer_begin(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  // Build binding refs.
  iree_hal_buffer_ref_t hal_bindings[PYRE_MAX_DISPATCH_BINDINGS];
  size_t actual_binding_count =
      binding_count < PYRE_MAX_DISPATCH_BINDINGS
          ? binding_count
          : PYRE_MAX_DISPATCH_BINDINGS;
  if (bindings && actual_binding_count > 0) {
    pyre_convert_bindings(bindings, actual_binding_count, hal_bindings);
  }

  iree_hal_buffer_ref_list_t binding_list = {
      .count = (iree_host_size_t)actual_binding_count,
      .values = hal_bindings,
  };

  iree_const_byte_span_t constants_span = {
      .data = (const uint8_t*)constants,
      .data_length = (iree_host_size_t)constants_size,
  };

  iree_hal_dispatch_config_t dispatch_config = {
      .workgroup_count = {config->workgroup_count[0],
                          config->workgroup_count[1],
                          config->workgroup_count[2]},
      .workgroup_size = {config->workgroup_size[0],
                         config->workgroup_size[1],
                         config->workgroup_size[2]},
  };

  status = iree_hal_command_buffer_dispatch(
      cb, module->executable, export_ordinal, dispatch_config,
      constants_span, binding_list, (iree_hal_dispatch_flags_t)flags);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  status = iree_hal_command_buffer_end(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  // Build semaphore lists.
  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_build_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_build_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_buffer_binding_table_t bt = {0};
  status = iree_hal_device_queue_execute(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, cb, bt, 0);
  iree_hal_command_buffer_release(cb);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_stream_dispatch(
    pyre_stream_t stream, pyre_executable_t executable,
    uint32_t export_ordinal, const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  if (!stream || !config) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  pyre_module_t module = (pyre_module_t)executable;
  if (!module || !module->executable) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "invalid executable");
  }

  pyre_status_t status = pyre_dispatch_begin_cb(stream);
  if (!pyre_status_is_ok(status)) return status;

  // Build binding refs.
  iree_hal_buffer_ref_t hal_bindings[PYRE_MAX_DISPATCH_BINDINGS];
  size_t actual_binding_count =
      binding_count < PYRE_MAX_DISPATCH_BINDINGS
          ? binding_count
          : PYRE_MAX_DISPATCH_BINDINGS;
  if (bindings && actual_binding_count > 0) {
    pyre_convert_bindings(bindings, actual_binding_count, hal_bindings);
  }

  iree_hal_buffer_ref_list_t binding_list = {
      .count = (iree_host_size_t)actual_binding_count,
      .values = hal_bindings,
  };

  iree_const_byte_span_t constants_span = {
      .data = (const uint8_t*)constants,
      .data_length = (iree_host_size_t)constants_size,
  };

  iree_hal_dispatch_config_t dispatch_config = {
      .workgroup_count = {config->workgroup_count[0],
                          config->workgroup_count[1],
                          config->workgroup_count[2]},
      .workgroup_size = {config->workgroup_size[0],
                         config->workgroup_size[1],
                         config->workgroup_size[2]},
  };

  iree_status_t iree_status = iree_hal_command_buffer_dispatch(
      stream->pending_cb, module->executable, export_ordinal,
      dispatch_config, constants_span, binding_list,
      (iree_hal_dispatch_flags_t)flags);
  if (!iree_status_is_ok(iree_status)) {
    return pyre_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return pyre_ok_status();
}

// Wrapper to adapt pyre_host_call_fn_t to iree_hal_host_call_fn_t.
typedef struct pyre_host_call_wrapper_t {
  pyre_host_call_fn_t fn;
  void* user_data;
} pyre_host_call_wrapper_t;

static iree_status_t pyre_host_call_thunk(
    void* wrapper_ptr, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  pyre_host_call_wrapper_t* wrapper =
      (pyre_host_call_wrapper_t*)wrapper_ptr;
  pyre_status_t s = wrapper->fn(wrapper->user_data);
  iree_allocator_free(iree_allocator_system(), wrapper);
  return (iree_status_t)(uintptr_t)s;
}

pyre_status_t pyre_queue_host_call(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_host_call_fn_t callback, void* user_data) {
  if (!device || !callback) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  pyre_host_call_wrapper_t* wrapper = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*wrapper), (void**)&wrapper);
  if (!iree_status_is_ok(alloc_status)) {
    return pyre_status_from_iree(alloc_status);
  }
  wrapper->fn = callback;
  wrapper->user_data = user_data;

  // Build semaphore lists.
  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_build_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_build_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_host_call_t call =
      iree_hal_make_host_call(pyre_host_call_thunk, wrapper);
  uint64_t args[4] = {0, 0, 0, 0};

  iree_status_t status = iree_hal_device_queue_host_call(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, call, args, 0);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), wrapper);
  }
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_stream_execution_barrier(pyre_stream_t stream) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }

  // Flush pending work and insert a barrier (wait on current timepoint
  // before allowing subsequent work).
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

  // The timeline semaphore model inherently serializes work within a stream.
  // After flush, subsequent work will wait on the latest timepoint.
  return pyre_ok_status();
}
