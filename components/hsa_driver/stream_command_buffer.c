// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hsa_driver/stream_command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "hsa_driver/hsa_buffer.h"
#include "hsa_driver/native_executable.h"
#include "hsa_driver/per_device_information.h"
#include "hsa_driver/status_util.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_hsa_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;


  // Per-device information for the target device.
  iree_hal_hsa_per_device_info_t* device_info;

  // Arena used for all allocations; references the block pool.
  iree_arena_allocator_t arena;

  // Maintains a reference to all resources used within the command buffer.
  iree_hal_resource_set_t* resource_set;

  // Device allocator for transient allocations.
  iree_hal_allocator_t* device_allocator;

  // Tracing context for this command buffer.
  iree_hal_stream_tracing_context_t* tracing_context;
} iree_hal_hsa_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable;

static iree_hal_hsa_stream_command_buffer_t*
iree_hal_hsa_stream_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_stream_command_buffer_vtable);
  return (iree_hal_hsa_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_hsa_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_hsa_per_device_info_t* device_info,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_info);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hsa_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device_info = device_info;
  command_buffer->device_allocator = device_allocator;
  command_buffer->tracing_context = tracing_context;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_hsa_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(command_buffer,
                              &iree_hal_hsa_stream_command_buffer_vtable);
}

static void iree_hal_hsa_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // Nothing to do - commands are executed immediately.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // No-op for now.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // No-op for now.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }

  // Each HSA dispatch already waits synchronously for its completion signal
  // before returning, so all prior work is finished by the time a barrier
  // is reached.  Nothing extra to do here — ordering and memory visibility
  // are already guaranteed.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // We could mark the memory as invalidated so that if managed HSA does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Use hsa_amd_memory_fill for 32-bit patterns.
  if (pattern_length == 4) {
    uint32_t pattern32 = *(const uint32_t*)pattern;
    size_t count = target_ref.length / sizeof(uint32_t);
    return IREE_HSA_CALL_TO_STATUS(
        hsa_amd_memory_fill(target_ptr, pattern32, count),
        "hsa_amd_memory_fill");
  }

  // For other pattern sizes, we need to do a manual fill.
  // This is a simple synchronous implementation.
  uint8_t* target_bytes = (uint8_t*)target_ptr;
  for (iree_device_size_t i = 0; i < target_ref.length; i += pattern_length) {
    memcpy(target_bytes + i, pattern,
           iree_min(pattern_length, target_ref.length - i));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Lock the completion signal mutex to serialize with concurrent dispatches
  // and copies on other threads. Without this, a dispatch on a worker thread
  // could submit an AQL packet that reads from target_ptr before our write
  // is visible to the GPU.
  iree_slim_mutex_lock(&command_buffer->device_info->completion_signal_mutex);

  memcpy(target_ptr, (const uint8_t*)source_buffer + source_offset,
         target_ref.length);

  // Ensure the CPU write is globally visible before any subsequent GPU work.
  __atomic_thread_fence(__ATOMIC_SEQ_CST);

  iree_slim_mutex_unlock(&command_buffer->device_info->completion_signal_mutex);

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  iree_hal_buffer_t* buffers[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  void* source_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  const void* source_ptr = (const uint8_t*)source_device_buffer + source_offset;

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Use async copy with completion signal.
  // Lock mutex to protect completion signal from concurrent access.
  iree_slim_mutex_lock(&command_buffer->device_info->completion_signal_mutex);
  
  hsa_signal_t completion_signal =
      command_buffer->device_info->completion_signal;
  hsa_signal_store_screlease(completion_signal, 1);

  iree_status_t status = IREE_HSA_CALL_TO_STATUS(
      hsa_amd_memory_async_copy(target_ptr, command_buffer->device_info->agent,
                                source_ptr, command_buffer->device_info->agent,
                                source_ref.length, 0, NULL, completion_signal),
      "hsa_amd_memory_async_copy");

  if (iree_status_is_ok(status)) {
    // Wait for copy to complete (synchronous for simplicity).
    hsa_signal_wait_scacquire(
        completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
  }

  iree_slim_mutex_unlock(&command_buffer->device_info->completion_signal_mutex);
  return status;
}

static iree_status_t iree_hal_hsa_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref,
    iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

// Debug flags for HSA dispatch - flip these to enable debug logging.
// Set to 1 to enable, 0 to disable. Override via compiler flags:
//   -DIREE_HSA_DEBUG_DISPATCH=1
#ifndef IREE_HSA_DEBUG_DISPATCH
#define IREE_HSA_DEBUG_DISPATCH 0  // General dispatch info (kernel name, grid, etc.)
#endif
#ifndef IREE_HSA_DEBUG_HIDDEN_ARGS
#define IREE_HSA_DEBUG_HIDDEN_ARGS 0  // Hidden/implicit argument filling
#endif

// Special debug: dump full kernarg hex for CUSTOM_DIRECT_ARGUMENTS kernels
#ifndef IREE_HSA_DEBUG_KERNARG_HEX
#define IREE_HSA_DEBUG_KERNARG_HEX 0  // Hex dump of kernarg for native kernels
#endif

// VALIDATION: Check if device pointers in kernarg are within our pool range
#ifndef IREE_HSA_VALIDATE_KERNARG_POINTERS
#define IREE_HSA_VALIDATE_KERNARG_POINTERS 0  // Disabled - too many false positives
#endif


static iree_status_t iree_hal_hsa_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Early debug logging moved to after we get kernel_params for filtering

  // TODO: we can support CUSTOM_DIRECT_ARGUMENTS quite easily here.
  // Static indirect arguments and parameters are also easy (as we can
  // map/capture them right now, even if slow as it's a host operation).
  // Dynamic indirect arguments and parameters require patching or some other
  // magic that may require recompiling dispatches.

  // For now custom direct arguments is a no-op. If everything comes in
  // in the constant buffer we just treat it normally.
  // TODO: We avoid all of the normal passing stuff, and just pass in the
  //       arguments directly.
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect parameters are not supported in HSA streams");
  }

  // If any of the workgroup counts are zero, we can skip execution.
  if (config.workgroup_count[0] == 0 || config.workgroup_count[1] == 0 ||
      config.workgroup_count[2] == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Get kernel params.
  const iree_hal_hsa_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_lookup_kernel_params(
              executable, export_ordinal, command_buffer->base.queue_affinity,
              &kernel_params));

#if IREE_HSA_DEBUG_DISPATCH
  // Log dispatch info when debugging.
  // For CUSTOM_DIRECT_ARGUMENTS, always log since those are native HIP kernels.
  bool use_direct_copy_check = iree_all_bits_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  bool hsa_debug_should_log = use_direct_copy_check;  // Always log native HIP kernels
  fprintf(stderr, "[HSA_DISPATCH]   export=%u flags=0x%lx constants_len=%zu bindings=%zu\n",
          export_ordinal, (unsigned long)flags, constants.data_length, bindings.count);
  fprintf(stderr, "[HSA_DISPATCH]   workgroup=(%u,%u,%u) grid=(%u,%u,%u) local_mem=%u\n",
          config.workgroup_size[0], config.workgroup_size[1], config.workgroup_size[2],
          config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2],
          config.dynamic_workgroup_local_memory);
  fprintf(stderr, "[HSA_DISPATCH]   kernarg_segment_size=%u explicit_kernarg_size=%u\n",
          kernel_params->kernarg_segment_size, kernel_params->explicit_kernarg_size);
  fprintf(stderr, "[HSA_DISPATCH]   group_segment_size=%u private_segment_size=%u\n",
          kernel_params->group_segment_size, kernel_params->private_segment_size);
  fprintf(stderr, "[HSA_DISPATCH]   block_dims=(%u,%u,%u) parameter_count=%u kernel_object=0x%lx\n",
          kernel_params->block_dims[0], kernel_params->block_dims[1], kernel_params->block_dims[2],
          kernel_params->parameter_count, (unsigned long)kernel_params->kernel_object);
  if (kernel_params->parameters && kernel_params->parameter_count > 0) {
    for (uint32_t i = 0; i < kernel_params->parameter_count && i < 8; ++i) {
      fprintf(stderr, "[HSA_DISPATCH]   param[%u]: offset=%u size=%u type=%u\n",
              i, kernel_params->parameters[i].offset, 
              kernel_params->parameters[i].size,
              (unsigned)kernel_params->parameters[i].type);
    }
  }
#endif

  IREE_TRACE({
    if (kernel_params->debug_info.function_name.size > 0) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0,
                                  kernel_params->debug_info.function_name.data,
                                  kernel_params->debug_info.function_name.size);
    }
  });

  // Track resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (bindings.values[i].buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &bindings.values[i].buffer));
    }
  }

  // Allocate kernarg memory.
  // For CUSTOM_DIRECT_ARGUMENTS, the constants buffer contains the actual
  // kernel arguments. We need to allocate kernarg memory based on the
  // constants size if kernarg_segment_size is 0 or smaller than constants.
  iree_host_size_t kernarg_size = kernel_params->kernarg_segment_size;
  bool use_direct_copy = iree_all_bits_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  
  // Calculate the minimum kernarg size needed for implicit args (256 bytes).
  // The implicit args start at explicit_kernarg_size aligned to 8 bytes.
  uint32_t explicit_size = kernel_params->explicit_kernarg_size;
  if (explicit_size == 0) {
    // Estimate from constants or bindings if not provided.
    explicit_size = (uint32_t)(bindings.count * sizeof(void*) + constants.data_length);
  }
  uint32_t implicit_offset = (explicit_size + 7) & ~7u;
  
  // Kernels with kernarg_segment_size=0 are typically precompiled library kernels
  // (e.g., hipBLASLt/rocBLAS GEMM) that don't use COV5 implicit args.
  // For these, allocate only the explicit arguments size.
  bool uses_implicit_args = (kernel_params->kernarg_segment_size > 0);
  iree_host_size_t min_size_for_implicit = uses_implicit_args ? (implicit_offset + 256) : explicit_size;
  
  // For CUSTOM_DIRECT_ARGUMENTS, ensure we allocate enough space for the
  // constants buffer. Native HIP/CUDA kernels may have incorrect metadata.
  if (use_direct_copy && constants.data_length > 0) {
    // The constants contain packed kernel arguments including device pointers.
    if (uses_implicit_args) {
      // We need at least constants.data_length + 256 bytes for implicit args.
      iree_host_size_t required_size = constants.data_length + 256;
      if (kernarg_size < required_size) {
        kernarg_size = required_size;
      }
    } else {
      // No implicit args - just allocate what we need for the constants.
      if (kernarg_size < constants.data_length) {
        kernarg_size = constants.data_length;
      }
    }
  }
  
  // Ensure minimum size for implicit args (if kernel uses them).
  if (kernarg_size < min_size_for_implicit) {
    kernarg_size = min_size_for_implicit;
  }
  void* kernarg_address = NULL;
  if (kernarg_size > 0 &&
      command_buffer->device_info->kernarg_memory_pool_valid) {
    iree_status_t status = IREE_HSA_CALL_TO_STATUS(
        hsa_amd_memory_pool_allocate(
            command_buffer->device_info->kernarg_memory_pool,
            kernarg_size, 0, &kernarg_address),
        "hsa_amd_memory_pool_allocate (kernarg)");
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Set up kernarg using parameter offsets from kernel metadata.
  if (kernarg_address) {
    // Zero out the kernarg buffer first.
    memset(kernarg_address, 0, kernarg_size);

    // Calculate workgroup dimensions and counts.
    uint32_t wg_size_x = config.workgroup_size[0]
                             ? config.workgroup_size[0]
                             : kernel_params->block_dims[0];
    uint32_t wg_size_y = config.workgroup_size[1]
                             ? config.workgroup_size[1]
                             : kernel_params->block_dims[1];
    uint32_t wg_size_z = config.workgroup_size[2]
                             ? config.workgroup_size[2]
                             : kernel_params->block_dims[2];
    if (wg_size_x == 0) wg_size_x = 1;
    if (wg_size_y == 0) wg_size_y = 1;
    if (wg_size_z == 0) wg_size_z = 1;

    // If we have parameter info, use the designated offsets.
    // EXCEPT for CUSTOM_DIRECT_ARGUMENTS - in that case, the constants buffer
    // already contains the complete packed arguments (including device pointers),
    // so we should NOT remap bindings.
    bool use_direct_copy = iree_all_bits_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
    if (kernel_params->parameters && kernel_params->parameter_count > 0 && !use_direct_copy) {
      iree_host_size_t binding_idx = 0;

      for (uint32_t i = 0; i < kernel_params->parameter_count; ++i) {
        const iree_hal_executable_export_parameter_t* param =
            &kernel_params->parameters[i];
        uint8_t* dest = (uint8_t*)kernarg_address + param->offset;

        if (param->type == IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING) {
          // This is a buffer pointer parameter.
          if (binding_idx < bindings.count) {
            void* buffer_ptr = NULL;
            if (bindings.values[binding_idx].buffer) {
              void* device_buffer = iree_hal_hsa_buffer_device_pointer(
                  iree_hal_buffer_allocated_buffer(
                      bindings.values[binding_idx].buffer));
              buffer_ptr =
                  (uint8_t*)device_buffer +
                  iree_hal_buffer_byte_offset(bindings.values[binding_idx].buffer) +
                  bindings.values[binding_idx].offset;
            }
            memcpy(dest, &buffer_ptr, sizeof(buffer_ptr));
            ++binding_idx;
          } else {
            // No binding available - write NULL pointer.
            void* null_ptr = NULL;
            memcpy(dest, &null_ptr, sizeof(null_ptr));
          }
        } else {
          // This is a constant parameter - copy from constants buffer.
          // Constants are packed at their ABI offsets (param->offset) in the
          // constants buffer, matching how they appear in the kernel argument
          // buffer.
          if (param->offset + param->size <= constants.data_length) {
            memcpy(dest, constants.data + param->offset, param->size);
          }
        }
      }
    } else if (use_direct_copy) {
      // CUSTOM_DIRECT_ARGUMENTS: copy the raw constants buffer directly.
      // The buffer already contains the complete packed kernel arguments.
      // 
      // For native HIP/CUDA kernels, the streaming layer's parameter metadata
      // may not accurately reflect the kernel's actual argument requirements.
      // The HSA kernel's explicit_kernarg_size is authoritative.
      //
      // If constants.data_length > explicit_kernarg_size:
      //   Copy only explicit_kernarg_size (avoid copying garbage).
      // If constants.data_length < explicit_kernarg_size:
      //   Copy what we have (kernel may access uninitialized memory, but
      //   this matches what the caller provided - they know the kernel's needs).
      // If constants.data_length == explicit_kernarg_size:
      //   Perfect match.
      if (constants.data_length > 0) {
        iree_host_size_t copy_size = constants.data_length;
        if (kernel_params->explicit_kernarg_size > 0 && 
            kernel_params->explicit_kernarg_size < copy_size) {
          copy_size = kernel_params->explicit_kernarg_size;
        }
        memcpy(kernarg_address, constants.data, copy_size);

        // Overlay binding pointers on top of the constants in kernarg.
        // When CUSTOM_DIRECT_ARGUMENTS is used with bindings, the bindings
        // represent device buffer pointers that were originally embedded as
        // raw values in the constants (e.g., for native HIP kernels whose
        // metadata classifies pointer args as by_value). The streaming layer
        // resolved these to proper HAL buffer bindings and zeroed the pointer
        // positions in the constants. Now we write the real device pointers
        // back at their kernarg offsets (carried in buffer_slot).
        for (iree_host_size_t i = 0; i < bindings.count; ++i) {
          int32_t kernarg_offset = bindings.values[i].buffer_slot;
          if (kernarg_offset < 0 ||
              (iree_host_size_t)(kernarg_offset + sizeof(void*)) > kernarg_size) {
            continue;  // Invalid or out-of-range offset - skip.
          }
          void* buffer_ptr = NULL;
          if (bindings.values[i].buffer) {
            void* device_buffer = iree_hal_hsa_buffer_device_pointer(
                iree_hal_buffer_allocated_buffer(
                    bindings.values[i].buffer));
            buffer_ptr =
                (uint8_t*)device_buffer +
                iree_hal_buffer_byte_offset(bindings.values[i].buffer) +
                bindings.values[i].offset;
          }
          memcpy((uint8_t*)kernarg_address + kernarg_offset, &buffer_ptr,
                 sizeof(buffer_ptr));
        }
        
        // Ensure kernarg write is visible to GPU before dispatch.
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        
#if IREE_HSA_VALIDATE_KERNARG_POINTERS
        // Validate that pointers in kernarg look reasonable.
        // GEMM kernels have pointers at offsets 32-56 (A, B, C, D).
        // Check that they're not NULL and look like device memory addresses.
        if (copy_size >= 64) {
          void** ptrs = (void**)((uint8_t*)kernarg_address + 32);
          for (int ptrIdx = 0; ptrIdx < 4; ++ptrIdx) {
            void* ptr = ptrs[ptrIdx];
            // Check for suspicious pointer values (< 4KB likely invalid, > 0x8000000000000000 likely invalid)
            uintptr_t addr = (uintptr_t)ptr;
            if (addr != 0 && (addr < 0x1000 || addr > 0x8000000000000000ULL)) {
              fprintf(stderr, "[HSA_VALIDATE] WARNING: Suspicious pointer at kernarg+%d: %p\n",
                      32 + ptrIdx * 8, ptr);
            }
          }
        }
#endif
#if IREE_HSA_DEBUG_DISPATCH
        {
          static int kernarg_dump_count = 0;
          ++kernarg_dump_count;
          fprintf(stderr, "[KERNARG #%d] copied %zu bytes to %p (kernel explicit_size=%u kernarg_seg=%u)\n", 
                  kernarg_dump_count, copy_size, kernarg_address,
                  kernel_params->explicit_kernarg_size, kernel_params->kernarg_segment_size);
          fprintf(stderr, "[KERNARG #%d]   allocated kernarg_size=%zu uses_implicit=%d\n",
                  kernarg_dump_count, kernarg_size, uses_implicit_args ? 1 : 0);
        }
#endif
#if IREE_HSA_DEBUG_KERNARG_HEX
        // Always dump kernarg hex for CUSTOM_DIRECT_ARGUMENTS to compare M=224 vs M=256
        fprintf(stderr, "[KERNARG_HEX] export=%u size=%zu\n", export_ordinal, copy_size);
        {
          const uint8_t* bytes = (const uint8_t*)constants.data;
          for (size_t i = 0; i < copy_size; i += 16) {
            fprintf(stderr, "[KERNARG_HEX] %04zx:", i);
            for (size_t j = 0; j < 16 && (i+j) < copy_size; ++j) {
              fprintf(stderr, " %02x", bytes[i+j]);
            }
            fprintf(stderr, "\n");
          }
        }
#endif
#if IREE_HSA_DEBUG_DISPATCH
        if (hsa_debug_should_log) {
          fprintf(stderr, "[HSA_DISPATCH]   DIRECT_COPY: copied %zu bytes (of %zu) to kernarg at %p\n",
                  copy_size, constants.data_length, kernarg_address);
          // Dump all pointer-sized values up to 16
          size_t num_ptrs = constants.data_length / sizeof(void*);
          if (num_ptrs > 16) num_ptrs = 16;
          fprintf(stderr, "[HSA_DISPATCH]   kernarg ptrs:");
          for (size_t i = 0; i < num_ptrs; ++i) {
            void* ptr_val = ((void**)kernarg_address)[i];
            fprintf(stderr, " %p", ptr_val);
          }
          fprintf(stderr, "\n");
          // For kernels where constants < explicit_kernarg_size, the hidden args region
          // is zero-filled. This may cause crashes if the kernel expects non-zero values.
          if (kernel_params->explicit_kernarg_size > copy_size) {
            iree_host_size_t hidden_size = kernel_params->explicit_kernarg_size - copy_size;
            // Only warn if the difference is significant (> 64 bytes indicates hidden args)
            if (hidden_size > 64) {
              fprintf(stderr, "[HSA_DISPATCH]   WARNING: Kernel requires HIP hidden args!\n");
              fprintf(stderr, "[HSA_DISPATCH]   constants=%zu bytes, kernel expects=%u bytes, hidden=%zu bytes\n",
                      copy_size, kernel_params->explicit_kernarg_size, hidden_size);
              fprintf(stderr, "[HSA_DISPATCH]   IREE streaming cannot provide HIP-specific hidden args.\n");
              fprintf(stderr, "[HSA_DISPATCH]   Proceeding with zero-filled hidden args - may cause GPU faults.\n");
            }
          }
        }
#endif
      }
    } else {
      // Fallback: pack buffers then constants sequentially.
      uint8_t* kernarg_ptr = (uint8_t*)kernarg_address;

      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        void* buffer_ptr = NULL;
        if (bindings.values[i].buffer) {
          void* device_buffer = iree_hal_hsa_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(bindings.values[i].buffer));
          buffer_ptr = (uint8_t*)device_buffer +
                       iree_hal_buffer_byte_offset(bindings.values[i].buffer) +
                       bindings.values[i].offset;
        }
        memcpy(kernarg_ptr, &buffer_ptr, sizeof(buffer_ptr));
        kernarg_ptr += sizeof(buffer_ptr);
      }

      if (constants.data_length > 0) {
        memcpy(kernarg_ptr, constants.data, constants.data_length);
      }
    }

    // Fill in COV5+ implicit arguments at the end of the kernarg buffer.
    // These are required for HIP/OpenCL kernels to compute blockIdx, gridDim,
    // etc.
    // The implicit args structure is:
    //   uint32_t BlockCountX, BlockCountY, BlockCountZ  (offset 0, 4, 8)
    //   uint16_t GroupSizeX, GroupSizeY, GroupSizeZ     (offset 12, 14, 16)
    //   uint8_t  Unused0[46]                            (offset 18)
    //   uint16_t GridDims                               (offset 64)
    //   uint8_t  Unused1[54]                            (offset 66)
    //   uint32_t DynamicLdsSize                         (offset 120)
    //   uint8_t  Unused2[132]                           (offset 124)
    // Total: 256 bytes (aligned to pointer size)
    //
    // The implicit args start at an aligned offset after the explicit args.
    // We use the kernel's explicit_kernarg_size to find this offset.
    uint32_t explicit_size = kernel_params->explicit_kernarg_size;
    if (explicit_size == 0) {
      // Estimate explicit size from parameters if not provided.
      if (kernel_params->parameters && kernel_params->parameter_count > 0) {
        for (uint32_t i = 0; i < kernel_params->parameter_count; ++i) {
          uint32_t end =
              kernel_params->parameters[i].offset + kernel_params->parameters[i].size;
          if (end > explicit_size) explicit_size = end;
        }
      } else {
        explicit_size = (uint32_t)(bindings.count * sizeof(void*) + constants.data_length);
      }
    }
    // For CUSTOM_DIRECT_ARGUMENTS, the constants contain the actual kernel args.
    // However, the kernel's explicit_kernarg_size is authoritative - it tells us
    // where implicit arguments should start. We should NOT use constants.data_length
    // to compute implicit_offset when we have valid kernel metadata, because
    // constants.data_length might be larger than actual kernel arguments.
    // (We already handle copying only min(constants.data_length, explicit_kernarg_size)
    // in the DIRECT_COPY path above.)

    // Align to 8 bytes for implicit args.
    uint32_t implicit_offset = (explicit_size + 7) & ~7u;
#if IREE_HSA_DEBUG_DISPATCH
    if (hsa_debug_should_log) {
      fprintf(stderr, "[HSA_DISPATCH]   explicit_size=%u implicit_offset=%u kernarg_size=%zu\n",
              explicit_size, implicit_offset, kernarg_size);
    }
#endif
    // Fill native HIP kernel hidden args at metadata-specified offsets.
    // These are hidden args that are part of the explicit kernarg buffer.
    const iree_hal_hip_hidden_args_t* ha = &kernel_params->hidden_args;
    bool has_embedded_hidden = (ha->block_count_x != UINT32_MAX ||
                                ha->group_size_x != UINT32_MAX ||
                                ha->grid_dims != UINT32_MAX);
#if IREE_HSA_DEBUG_HIDDEN_ARGS
    fprintf(stderr, "[HSA_HIDDEN_ARGS] Embedded hidden check: block_x=%u group_x=%u grid_dims=%u => has=%d\n",
            ha->block_count_x, ha->group_size_x, ha->grid_dims, has_embedded_hidden);
#endif
    if (has_embedded_hidden) {
      uint8_t* ka = (uint8_t*)kernarg_address;
      // BlockCountX/Y/Z (uint32_t)
      if (ha->block_count_x != UINT32_MAX) {
        uint32_t val = config.workgroup_count[0];
        memcpy(ka + ha->block_count_x, &val, sizeof(uint32_t));
      }
      if (ha->block_count_y != UINT32_MAX) {
        uint32_t val = config.workgroup_count[1];
        memcpy(ka + ha->block_count_y, &val, sizeof(uint32_t));
      }
      if (ha->block_count_z != UINT32_MAX) {
        uint32_t val = config.workgroup_count[2];
        memcpy(ka + ha->block_count_z, &val, sizeof(uint32_t));
      }
      // GroupSizeX/Y/Z (uint16_t)
      if (ha->group_size_x != UINT32_MAX) {
        uint16_t val = (uint16_t)wg_size_x;
        memcpy(ka + ha->group_size_x, &val, sizeof(uint16_t));
      }
      if (ha->group_size_y != UINT32_MAX) {
        uint16_t val = (uint16_t)wg_size_y;
        memcpy(ka + ha->group_size_y, &val, sizeof(uint16_t));
      }
      if (ha->group_size_z != UINT32_MAX) {
        uint16_t val = (uint16_t)wg_size_z;
        memcpy(ka + ha->group_size_z, &val, sizeof(uint16_t));
      }
      // RemainderX/Y/Z (uint16_t) - usually 0 for uniform grids
      if (ha->remainder_x != UINT32_MAX) {
        uint16_t val = 0;
        memcpy(ka + ha->remainder_x, &val, sizeof(uint16_t));
      }
      if (ha->remainder_y != UINT32_MAX) {
        uint16_t val = 0;
        memcpy(ka + ha->remainder_y, &val, sizeof(uint16_t));
      }
      if (ha->remainder_z != UINT32_MAX) {
        uint16_t val = 0;
        memcpy(ka + ha->remainder_z, &val, sizeof(uint16_t));
      }
      // GridDims (uint16_t)
      if (ha->grid_dims != UINT32_MAX) {
        uint16_t val = (config.workgroup_count[2] * wg_size_z > 1)
                           ? 3
                           : 1 + (config.workgroup_count[1] * wg_size_y != 1);
        memcpy(ka + ha->grid_dims, &val, sizeof(uint16_t));
      }
      // GlobalOffsetX/Y/Z (uint64_t) - always 0 for HIP
      if (ha->global_offset_x != UINT32_MAX) {
        uint64_t val = 0;
        memcpy(ka + ha->global_offset_x, &val, sizeof(uint64_t));
      }
      if (ha->global_offset_y != UINT32_MAX) {
        uint64_t val = 0;
        memcpy(ka + ha->global_offset_y, &val, sizeof(uint64_t));
      }
      if (ha->global_offset_z != UINT32_MAX) {
        uint64_t val = 0;
        memcpy(ka + ha->global_offset_z, &val, sizeof(uint64_t));
      }
    }

    // Use the actual allocated kernarg_size, not the kernel's reported size,
    // since we may have enlarged it for CUSTOM_DIRECT_ARGUMENTS.
    //
    // Skip implicit args filling for kernels that report kernarg_segment_size=0
    // OR for CUSTOM_DIRECT_ARGUMENTS kernels (precompiled library kernels like
    // hipBLASLt/rocBLAS GEMM that don't use the COV5 implicit args format).
    bool should_fill_implicit = (kernel_params->kernarg_segment_size > 0) &&
                                !use_direct_copy;
#if IREE_HSA_DEBUG_HIDDEN_ARGS
    fprintf(stderr, "[HSA_HIDDEN_ARGS] explicit_size=%u implicit_offset=%u kernarg_size=%zu need=%u should_fill=%d\n",
            explicit_size, implicit_offset, kernarg_size, implicit_offset + 256, should_fill_implicit);
#endif
    if (should_fill_implicit && implicit_offset + 256 <= kernarg_size) {
#if IREE_HSA_DEBUG_HIDDEN_ARGS
      fprintf(stderr, "[HSA_HIDDEN_ARGS] Filling implicit args at offset %u\n", implicit_offset);
#endif
      uint8_t* implicit_args = (uint8_t*)kernarg_address + implicit_offset;

      // BlockCountX/Y/Z (uint32_t at offsets 0, 4, 8)
      uint32_t block_count_x = config.workgroup_count[0];
      uint32_t block_count_y = config.workgroup_count[1];
      uint32_t block_count_z = config.workgroup_count[2];
      memcpy(implicit_args + 0, &block_count_x, sizeof(uint32_t));
      memcpy(implicit_args + 4, &block_count_y, sizeof(uint32_t));
      memcpy(implicit_args + 8, &block_count_z, sizeof(uint32_t));

      // GroupSizeX/Y/Z (uint16_t at offsets 12, 14, 16)
      uint16_t group_size_x = (uint16_t)wg_size_x;
      uint16_t group_size_y = (uint16_t)wg_size_y;
      uint16_t group_size_z = (uint16_t)wg_size_z;
      memcpy(implicit_args + 12, &group_size_x, sizeof(uint16_t));
      memcpy(implicit_args + 14, &group_size_y, sizeof(uint16_t));
      memcpy(implicit_args + 16, &group_size_z, sizeof(uint16_t));

      // RemainderX/Y/Z (uint16_t at offsets 18, 20, 22)
      // These are for partial workgroups at grid edges (usually 0 for full grids)
      uint16_t remainder_x = 0;
      uint16_t remainder_y = 0;
      uint16_t remainder_z = 0;
      memcpy(implicit_args + 18, &remainder_x, sizeof(uint16_t));
      memcpy(implicit_args + 20, &remainder_y, sizeof(uint16_t));
      memcpy(implicit_args + 22, &remainder_z, sizeof(uint16_t));

      // GlobalOffsetX/Y/Z (uint64_t at offsets 40, 48, 56)
      // These are for OpenCL global_id offsets (always 0 for HIP)
      uint64_t global_offset_x = 0;
      uint64_t global_offset_y = 0;
      uint64_t global_offset_z = 0;
      memcpy(implicit_args + 40, &global_offset_x, sizeof(uint64_t));
      memcpy(implicit_args + 48, &global_offset_y, sizeof(uint64_t));
      memcpy(implicit_args + 56, &global_offset_z, sizeof(uint64_t));

      // GridDims (uint16_t at offset 64)
      uint16_t grid_dims =
          (config.workgroup_count[2] * wg_size_z > 1)
              ? 3
              : 1 + (config.workgroup_count[1] * wg_size_y != 1);
      memcpy(implicit_args + 64, &grid_dims, sizeof(uint16_t));

      // DynamicLdsSize (uint32_t at offset 120)
      uint32_t dynamic_lds = config.dynamic_workgroup_local_memory;
      memcpy(implicit_args + 120, &dynamic_lds, sizeof(uint32_t));
    }
  }

  // Create and submit AQL dispatch packet.
  hsa_queue_t* queue = command_buffer->device_info->queue;

  // AGGRESSIVE_SYNC: Submit a barrier packet before each kernel dispatch.
  // This ensures all previous memory operations are complete before the kernel
  // reads any data. The barrier uses system-scope fences for maximum coherence.
#ifndef IREE_HSA_USE_BARRIER_PACKETS
#define IREE_HSA_USE_BARRIER_PACKETS 0  // Disabled - doesn't fix 160-byte kernel corruption
#endif
#if IREE_HSA_USE_BARRIER_PACKETS
  {
    // Reserve a slot for the barrier packet.
    uint64_t barrier_index =
        hsa_queue_add_write_index_relaxed(queue, 1);
    
    // Wait for queue space.
    while (barrier_index -
               hsa_queue_load_read_index_relaxed(
                   queue) >=
           queue->size) {
      // Busy wait for space.
    }
    
    // Get barrier packet address.
    hsa_barrier_and_packet_t* barrier =
        (hsa_barrier_and_packet_t*)queue->base_address +
        (barrier_index & (queue->size - 1));
    
    // Initialize barrier packet.
    memset(barrier, 0, sizeof(*barrier));
    // No dependent signals - we just want a memory fence.
    // The barrier will complete immediately but enforce memory ordering.
    barrier->completion_signal.handle = 0;  // No signal to wait on.
    
    // Set header with system-scope fences.
    uint16_t barrier_header =
        HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE |
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE |
        1 << HSA_PACKET_HEADER_BARRIER;  // Block until barrier completes.
    __atomic_store_n(&barrier->header, barrier_header, __ATOMIC_RELEASE);
    
    // Ring doorbell for the barrier packet.
    hsa_signal_store_screlease(
        queue->doorbell_signal, barrier_index);
  }
#endif

  // Get write index for the queue.
  uint64_t write_index =
      hsa_queue_add_write_index_relaxed(queue, 1);

  // Wait for queue space.
  while (write_index -
             hsa_queue_load_read_index_relaxed(
                 queue) >=
         queue->size) {
    // Busy wait for space.
  }

  // Get packet address.
  hsa_kernel_dispatch_packet_t* packet =
      (hsa_kernel_dispatch_packet_t*)queue->base_address +
      (write_index & (queue->size - 1));

  // Initialize packet fields (except header/setup which will be written
  // atomically last).
  memset(packet, 0, sizeof(*packet));
  packet->workgroup_size_x = config.workgroup_size[0]
                                 ? config.workgroup_size[0]
                                 : kernel_params->block_dims[0];
  packet->workgroup_size_y = config.workgroup_size[1]
                                 ? config.workgroup_size[1]
                                 : kernel_params->block_dims[1];
  packet->workgroup_size_z = config.workgroup_size[2]
                                 ? config.workgroup_size[2]
                                 : kernel_params->block_dims[2];
  packet->grid_size_x = config.workgroup_count[0] * packet->workgroup_size_x;
  packet->grid_size_y = config.workgroup_count[1] * packet->workgroup_size_y;
  packet->grid_size_z = config.workgroup_count[2] * packet->workgroup_size_z;
  // group_segment_size = static (from kernel metadata) + dynamic (from launch config)
  packet->group_segment_size = kernel_params->group_segment_size + 
                                config.dynamic_workgroup_local_memory;
  packet->private_segment_size = kernel_params->private_segment_size;
  packet->kernel_object = kernel_params->kernel_object;
  packet->kernarg_address = kernarg_address;
#if IREE_HSA_DEBUG_DISPATCH
  if (hsa_debug_should_log) {
    fprintf(stderr, "[HSA_DISPATCH] AQL packet: grid=(%u,%u,%u) wg=(%u,%u,%u) grp_seg=%u priv_seg=%u\n",
            packet->grid_size_x, packet->grid_size_y, packet->grid_size_z,
            packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
            packet->group_segment_size, packet->private_segment_size);
    fprintf(stderr, "[HSA_DISPATCH] AQL packet: kernel_object=0x%lx kernarg_addr=%p\n",
            (unsigned long)packet->kernel_object, packet->kernarg_address);
  }
#endif

  // Set up completion signal.
  // Lock mutex to protect completion signal from concurrent access.
  iree_slim_mutex_lock(&command_buffer->device_info->completion_signal_mutex);
  
  hsa_signal_t completion_signal =
      command_buffer->device_info->completion_signal;
  hsa_signal_store_screlease(completion_signal, 1);
  packet->completion_signal = completion_signal;

  // Ensure all writes to kernarg memory are visible before submitting the packet.
  // This is critical for correctness - without this barrier, the GPU might read
  // stale or uninitialized data from the kernarg buffer.
  __atomic_thread_fence(__ATOMIC_SEQ_CST);

  // Set header and setup atomically as a single 32-bit word.
  // The header and setup fields are the first 32 bits of the packet:
  // - bits[15:0]: header
  // - bits[31:16]: setup
  // This must be written atomically to prevent the GPU from seeing a partial
  // packet with valid header but invalid setup.
  uint16_t header =
      HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE |
      HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
      HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  uint16_t setup = 3;  // 3 dimensions
  uint32_t header_setup = ((uint32_t)setup << 16) | header;
  __atomic_store_n((uint32_t*)packet, header_setup, __ATOMIC_RELEASE);

  // Ring doorbell.
  hsa_signal_store_screlease(queue->doorbell_signal,
                                                          write_index);

  // Wait for completion (synchronous for simplicity).
  // Use a 10 second timeout to avoid hanging forever on kernel failures.
  hsa_signal_value_t wait_result =
      hsa_signal_wait_scacquire(
          completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
          10ULL * 1000 * 1000 * 1000,  // 10 seconds in nanoseconds
          HSA_WAIT_STATE_BLOCKED);
  
  // Full memory barrier to ensure GPU writes are visible to CPU and subsequent
  // kernel dispatches. The signal wait provides acquire semantics for the
  // completion signal, but we need an explicit fence for device memory coherence.
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  
  iree_slim_mutex_unlock(&command_buffer->device_info->completion_signal_mutex);
  
  if (wait_result != 0) {
    fprintf(stderr, "[HSA_DISPATCH] Kernel timed out! signal_value=%ld grid=(%u,%u,%u) block=(%u,%u,%u) kernarg=%p\n",
            (long)wait_result,
            config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2],
            config.workgroup_size[0] ? config.workgroup_size[0] : kernel_params->block_dims[0],
            config.workgroup_size[1] ? config.workgroup_size[1] : kernel_params->block_dims[1],
            config.workgroup_size[2] ? config.workgroup_size[2] : kernel_params->block_dims[2],
            kernarg_address);
    // NOTE: Native HIP/CUDA kernels may require hidden kernel arguments that
    // the HSA backend cannot provide. Use IREE_HAL_DRIVER=hip for native
    // PyTorch kernels, which uses hipModuleLaunchKernel to provide hidden args.
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "HSA kernel dispatch timed out (native kernels may "
                            "require HIP backend - set IREE_HAL_DRIVER=hip)");
  }

  // Free kernarg memory.
  // Enable IREE_HSA_LEAK_KERNARG=1 to test if kernarg reuse causes issues.
#ifndef IREE_HSA_LEAK_KERNARG
#define IREE_HSA_LEAK_KERNARG 0
#endif
#if !IREE_HSA_LEAK_KERNARG
  if (kernarg_address) {
    IREE_HSA_IGNORE_ERROR(
                          hsa_amd_memory_pool_free(kernarg_address));
  }
#else
  (void)kernarg_address;  // Intentionally leak for debugging
#endif

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable = {
        .destroy = iree_hal_hsa_stream_command_buffer_destroy,
        .begin = iree_hal_hsa_stream_command_buffer_begin,
        .end = iree_hal_hsa_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_hsa_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hsa_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hsa_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_hsa_stream_command_buffer_signal_event,
        .reset_event = iree_hal_hsa_stream_command_buffer_reset_event,
        .wait_events = iree_hal_hsa_stream_command_buffer_wait_events,
        .advise_buffer = iree_hal_hsa_stream_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_hsa_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hsa_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hsa_stream_command_buffer_copy_buffer,
        .collective = iree_hal_hsa_stream_command_buffer_collective,
        .dispatch = iree_hal_hsa_stream_command_buffer_dispatch,
};
