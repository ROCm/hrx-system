// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "streaming/internal.h"

#include <inttypes.h>
#include <stdio.h>

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

static void
iree_hal_streaming_stream_destroy(iree_hal_streaming_stream_t *stream);

iree_status_t
iree_hal_streaming_stream_create(iree_hal_streaming_context_t *context,
                                 iree_hal_streaming_stream_flags_t flags,
                                 int priority, iree_allocator_t host_allocator,
                                 iree_hal_streaming_stream_t **out_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t *stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void **)&stream));
  iree_atomic_ref_count_init(&stream->ref_count);
  stream->context = context;
  stream->flags = flags;
  stream->priority = priority;
  stream->command_buffer = NULL;
  stream->timeline_semaphore = NULL;
  stream->pending_value = 0;
  stream->submitted_value = 0;
  stream->completed_value = 0;
  stream->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stream->recorded_events = NULL;
  stream->event_count = 0;
  stream->event_capacity = 0;

  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  stream->capture_graph = NULL;
  stream->capture_id = 0;
  stream->capture_dependencies = NULL;
  stream->capture_dependency_count = 0;
  stream->capture_dependency_capacity = 0;

  stream->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&stream->mutex);

  // Create timeline semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(
      context->device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ULL,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &stream->timeline_semaphore);

  // Register stream with context.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_register_stream(context, stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_hal_streaming_stream_destroy(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void
iree_hal_streaming_stream_destroy(iree_hal_streaming_stream_t *stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture and clear context pointer to prevent re-entry during unregister.
  iree_hal_streaming_context_t *context = stream->context;
  stream->context = NULL;

  // Synchronize stream before cleanup to ensure all operations complete.
  // This is important to avoid leaking resources from pending operations.
  iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));

  // Unregister from context before cleanup.
  // Note: We already cleared stream->context, so if unregister tries to
  // release and that triggers another destroy, it will be a no-op.
  if (context) {
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

  // Clean up recorded events.
  if (stream->recorded_events) {
    for (iree_host_size_t i = 0; i < stream->event_count; ++i) {
      if (stream->recorded_events[i]) {
        iree_hal_streaming_event_release(stream->recorded_events[i]);
      }
    }
    iree_allocator_free(stream->host_allocator, stream->recorded_events);
  }

  // Release command buffer.
  iree_hal_command_buffer_release(stream->command_buffer);

  // Release timeline semaphore.
  iree_hal_semaphore_release(stream->timeline_semaphore);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&stream->mutex);

  // Free stream memory.
  const iree_allocator_t host_allocator = stream->host_allocator;
  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t *stream) {
  if (stream) {
    iree_atomic_ref_count_inc(&stream->ref_count);
  }
}

void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t *stream) {
  if (stream && iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    iree_hal_streaming_stream_destroy(stream);
  }
}

iree_status_t
iree_hal_streaming_stream_begin(iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);

  iree_status_t status = iree_ok_status();

  // Create command buffer if not already created.
  // Note that we set UNRETAINED as we ensure the resources we have to track are
  // retained at the graph exec level and CUDA/HIP don't make any statements
  // about resource lifetime.
  if (!stream->command_buffer) {
    status = iree_hal_command_buffer_create(
        stream->context->device,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
        IREE_HAL_COMMAND_CATEGORY_ANY, stream->queue_affinity,
        /*binding_capacity=*/0, &stream->command_buffer);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Begin recording.
  status = iree_hal_command_buffer_begin(stream->command_buffer);

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_stream_flush(iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);

  iree_status_t status = iree_ok_status();
  if (stream->command_buffer) {
    // End recording and submit command buffer.
    status = iree_hal_command_buffer_end(stream->command_buffer);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    // Wait for the previous submission (pending_value before increment).
    // This chains each flush after the one before it, so that operations
    // split across multiple command buffers (e.g. by an intervening
    // hipMemcpy) still execute in order.
    uint64_t wait_value = stream->pending_value;
    stream->pending_value++;

    // Submit to device queue with timeline semaphore.
    // Wait for the previous submission to complete before executing.
    iree_hal_queue_affinity_t queue_affinity = stream->queue_affinity;
    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_value > 0
                     ? 1
                     : 0, // Only wait if there was a previous submission.
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &stream->pending_value,
    };

    status = iree_hal_device_queue_execute(
        stream->context->device, queue_affinity, wait_semaphores,
        signal_semaphores, stream->command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);

    if (!iree_status_is_ok(status)) {
      // Error will propagate via iree_status_t return.
    }

    // Track the submitted value for wait_submitted.
    if (iree_status_is_ok(status)) {
      stream->submitted_value = stream->pending_value;
    }

    // Release command buffer (we're done with it).
    iree_hal_command_buffer_release(stream->command_buffer);
    stream->command_buffer = NULL;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_stream_query(iree_hal_streaming_stream_t *stream,
                                int *status) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(status);

  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value));

  if (current_value >= stream->pending_value) {
    *status = 0; // Complete
    stream->completed_value = current_value;
  } else {
    *status = 1; // Not complete
  }

  return iree_ok_status();
}

iree_status_t
iree_hal_streaming_stream_synchronize(iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  int status = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_stream_query(stream, &status));
  (void)status;

  // Wait for timeline semaphore to reach pending value.
  if (stream->pending_value > stream->completed_value) {
    // fprintf(stderr, "[STREAM] sync: waiting for semaphore pending=%"PRIu64"
    // completed=%"PRIu64"\n",
    //         stream->pending_value, stream->completed_value);
    iree_status_t wait_status = iree_hal_semaphore_wait(
        stream->timeline_semaphore, stream->pending_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    if (!iree_status_is_ok(wait_status)) {
      IREE_TRACE_ZONE_END(z0);
      return wait_status;
    }
    // fprintf(stderr, "[STREAM] sync: wait OK\n");
    stream->completed_value = stream->pending_value;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t
iree_hal_streaming_stream_wait_submitted(iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for already-submitted work to complete WITHOUT flushing.
  // This is safe to call from other threads as it doesn't modify stream state.
  // We wait for submitted_value (the last value that was actually submitted).
  if (stream->submitted_value > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_wait(
                stream->timeline_semaphore, stream->submitted_value,
                iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t
iree_hal_streaming_stream_wait_event(iree_hal_streaming_stream_t *stream,
                                     iree_hal_streaming_event_t *event) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Event wait during graph capture is not yet implemented.
    // TODO(#graph-capture): Add wait node to graph.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "event wait during graph capture not yet implemented");
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Get the current stream pending value to signal after waiting for the event.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;

  // Create a queue barrier that waits for the event and signals the stream.
  // This ensures the stream continues only after the event is signaled.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 1,
      .semaphores = &event->semaphore,
      .payload_values = &event->signal_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_barrier(
              stream->context->device, stream->queue_affinity, wait_semaphores,
              signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE));

  // Update completed value to track this barrier.
  stream->completed_value = signal_value;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t *context,
    const iree_hal_streaming_parameter_info_t *parameters,
    const void *parameter_buffer_ptr, void *out_constants,
    iree_hal_buffer_ref_list_t *out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (parameters->buffer_size == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parameter_buffer_ptr);
  IREE_ASSERT_ARGUMENT(out_bindings);

  const uint8_t *parameter_buffer = (const uint8_t *)parameter_buffer_ptr;

  // Copy constant data spans.
  // Each copy represents one or more constants laid out contiguously and
  // copied in order. Constants are placed at their ABI offsets (dst_offset)
  // matching how they'll appear in the kernel argument buffer.
  uint8_t *constants = (uint8_t *)out_constants;
  const iree_hal_streaming_parameter_op_t *op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    memcpy(constants + copy_op.dst_offset,
           parameter_buffer + copy_op.src_offset, copy_op.size);
  }

  // Resolve bindings, if any.
  // For native kernels with NULL or external device pointers, we can't use
  // IREE's binding mechanism. In that case, the caller should use
  // CUSTOM_DIRECT_ARGUMENTS to pass the raw parameter buffer directly.
  iree_hal_buffer_ref_t *bindings =
      (iree_hal_buffer_ref_t *)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void *device_ptr = *(void **)(parameter_buffer + resolve_op.src_offset);
    // TODO(benvanik): possibly calculate proper range here? We could easily
    // (at only the cost of a cache miss) get the total buffer size and then
    // subtract the offset to get the remaining size.

    // Handle NULL device pointers - some kernels pass NULL for optional
    // buffers. Return NOT_FOUND to signal that this kernel needs raw argument
    // passing.
    if (!device_ptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "binding %u has NULL device pointer", i);
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] = iree_hal_make_buffer_ref(
        stream_ref.buffer->buffer, stream_ref.offset, IREE_HAL_WHOLE_BUFFER);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_unpack_parameter_list(
    iree_hal_streaming_context_t *context,
    const iree_hal_streaming_parameter_info_t *parameters,
    void **parameter_list, void *out_constants,
    iree_hal_buffer_ref_list_t *out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (parameters->buffer_size == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parameter_list);
  IREE_ASSERT_ARGUMENT(out_bindings);

  // When parameters are provided as an array of pointers, each element in the
  // array points to the actual parameter value. We need to dereference each
  // pointer and handle buffer translation.

  // Copy constant data spans.
  // For each copy operation, we read from the parameter list at the source
  // ordinal and copy to the ABI offset in the constants buffer.
  uint8_t *constants = (uint8_t *)out_constants;
  const iree_hal_streaming_parameter_op_t *op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    // In pointer array mode, src_ordinal is an index into the parameter_list
    // array. Each parameter_list[index] is a pointer to the actual value.
    // We need to dereference it to get the value.
    void *param_ptr = parameter_list[copy_op.src_ordinal];
    memcpy(constants + copy_op.dst_offset, param_ptr, copy_op.size);
  }

  // Resolve bindings, if any.
  // For bindings, each parameter in the list is a pointer to a device pointer.
  // For native kernels with NULL or external device pointers, we can't use
  // IREE's binding mechanism. In that case, the caller should use
  // CUSTOM_DIRECT_ARGUMENTS to pass the raw parameter buffer directly.
  iree_hal_buffer_ref_t *bindings =
      (iree_hal_buffer_ref_t *)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    // In pointer array mode, src_offset is an index into the parameter_list.
    void *param_ptr = parameter_list[resolve_op.src_ordinal];
    // The parameter points to a device pointer (void*)
    void *device_ptr = *(void **)param_ptr;
    // TODO(benvanik): possibly calculate proper range here? We could easily
    // (at only the cost of a cache miss) get the total buffer size and then
    // subtract the offset to get the remaining size.

    // Handle NULL device pointers - some kernels pass NULL for optional
    // buffers. Return NOT_FOUND to signal that this kernel needs raw argument
    // passing.
    if (!device_ptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "binding %u has NULL device pointer", i);
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] = iree_hal_make_buffer_ref(
        stream_ref.buffer->buffer, stream_ref.offset, IREE_HAL_WHOLE_BUFFER);
  }

  return iree_ok_status();
}

// Debug flag - set to 1 to enable verbose kernel launch debugging
#ifndef IREE_STREAMING_DEBUG_KERNEL_LAUNCH
#define IREE_STREAMING_DEBUG_KERNEL_LAUNCH 0
#endif

// Filter to only log kernels matching this substring (NULL = log all)
static const char *IREE_STREAMING_DEBUG_KERNEL_FILTER = NULL;

// Helper to dump buffer contents
static void debug_dump_buffer(const char *label, const void *buf, size_t len) {
  fprintf(stderr, "[LAUNCH] %s (%zu bytes):\n", label, len);
  const uint8_t *p = (const uint8_t *)buf;
  for (size_t i = 0; i < len && i < 256; i += 16) {
    fprintf(stderr, "  %04zx: ", i);
    for (size_t j = 0; j < 16 && i + j < len; ++j) {
      fprintf(stderr, "%02x", p[i + j]);
      if (j == 7)
        fprintf(stderr, " ");
    }
    fprintf(stderr, "\n");
  }
}

iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t *symbol,
    const iree_hal_streaming_dispatch_params_t *params,
    iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

#if IREE_STREAMING_DEBUG_KERNEL_LAUNCH
  bool debug_should_log =
      !IREE_STREAMING_DEBUG_KERNEL_FILTER ||
      (symbol->name.data &&
       strstr(symbol->name.data, IREE_STREAMING_DEBUG_KERNEL_FILTER));
  if (debug_should_log) {
    fprintf(stderr, "[LAUNCH] Kernel: %.*s\n",
            (int)(symbol->name.size > 120 ? 120 : symbol->name.size),
            symbol->name.data ? symbol->name.data : "(null)");
    fprintf(stderr, "[LAUNCH]   grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u\n",
            params->grid_dim[0], params->grid_dim[1], params->grid_dim[2],
            params->block_dim[0], params->block_dim[1], params->block_dim[2],
            (unsigned)params->shared_memory_bytes);
    fprintf(stderr, "[LAUNCH]   flags=0x%x buffer=%p\n", params->flags,
            params->buffer);
    fprintf(stderr,
            "[LAUNCH]   symbol: copy_count=%u binding_count=%u "
            "constant_bytes=%u buffer_size=%u\n",
            symbol->parameters.copy_count, symbol->parameters.binding_count,
            (unsigned)symbol->parameters.constant_bytes,
            (unsigned)symbol->parameters.buffer_size);
  }
#else
  (void)0; // Suppress unused variable warning
#endif

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Check if cooperative launch is requested.
  if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE) {
    // TODO: Add HAL dispatch flag for cooperative kernel support and pass
    // through to the backend. For now, return unimplemented.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cooperative kernel launch not yet implemented in HAL layer");
  }

  // Verify parameter buffer.
  // TODO(benvanik): pass size when we have it so we can check it.
  if (!params->buffer && symbol->parameters.buffer_size > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct kernel launch missing expected parameters");
  }

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add kernel node to the graph instead of recording to command buffer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_kernel_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, symbol, params, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Ensure command buffer is recording.
  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Stack allocate arrays based on cached sizes.
  // Zero-initialize constants to prevent uninitialized padding bytes from
  // being misinterpreted as device pointers by the overlay scan.
  void *constants = symbol->parameters.constant_bytes
                        ? iree_alloca(symbol->parameters.constant_bytes)
                        : NULL;
  if (constants)
    memset(constants, 0, symbol->parameters.constant_bytes);
  iree_hal_buffer_ref_list_t binding_list = {
      .count = symbol->parameters.binding_count,
      .values = symbol->parameters.binding_count
                    ? iree_alloca(symbol->parameters.binding_count *
                                  sizeof(iree_hal_buffer_ref_t))
                    : NULL,
  };

  // Check if this is a "native" kernel without IREE parameter metadata.
  // Native kernels have no bindings and no copy operations.
  bool is_native_kernel = (symbol->parameters.binding_count == 0 &&
                           symbol->parameters.copy_count == 0);

  size_t constants_size = symbol->parameters.constant_bytes;
  // Track if we need to use raw argument passing (e.g., for external pointers).
  bool use_raw_arguments = false;

  // Check if this is a pre-packed buffer (HIP_LAUNCH_PARAM_BUFFER format).
  // Pre-packed buffers are already in the kernel's native ABI format and should
  // be passed directly without any unpacking or translation.
  bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;

  if (is_pre_packed && params->buffer) {
    // Pre-packed buffer: pass the raw buffer directly to the kernel.
    // This is used for kernels launched via HIP_LAUNCH_PARAM_BUFFER_POINTER
    // (e.g., hipBLASLt GEMM kernels) where the buffer is already packed.
    constants = params->buffer;
    if (params->buffer_size != 0) {
      constants_size = params->buffer_size;
    }
    binding_list.count = 0; // No IREE bindings, using raw pointers.
    use_raw_arguments = true;
#if IREE_STREAMING_DEBUG_KERNEL_LAUNCH
    if (debug_should_log) {
      fprintf(stderr, "[LAUNCH]   PATH: is_pre_packed (buffer=%p size=%zu)\n",
              params->buffer, params->buffer_size);
    }
#endif
  } else if (is_native_kernel && params->buffer) {
    // Native kernel with pre-packed buffer: pass raw arguments directly.
    // For native kernels, params->buffer contains the pre-packed kernel
    // arguments that should be passed as-is to the GPU.
    constants = params->buffer;
    if (params->buffer_size > 0) {
      constants_size = params->buffer_size;
    }
    use_raw_arguments = true;
  } else if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) {
    // Unpack parameters from array of pointers (void**).
    iree_status_t unpack_status = iree_hal_streaming_unpack_parameter_list(
        stream->context, &symbol->parameters, (void **)params->buffer,
        constants, &binding_list);
    if (!iree_status_is_ok(unpack_status)) {
      // If unpack fails due to NULL or external device pointers, fall back
      // to raw argument passing. This handles native kernels with optional
      // parameters or external allocations.
      if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(unpack_status);

        // Convert void** to packed buffer format.
        // Pack all parameters (copies + resolves) into a single constants
        // buffer. For native kernels, we use the parameter metadata to know the
        // layout.
        const iree_hal_streaming_parameter_info_t *params_info =
            &symbol->parameters;
        void **args_array = (void **)params->buffer;

        // Build packed buffer from args array using the ops.
        // Stack-allocate the buffer (using constant_bytes as size).
        constants = iree_alloca(params_info->constant_bytes);
        memset(constants, 0, params_info->constant_bytes);
        constants_size = params_info->constant_bytes;

        // Process all copy operations (constants/scalars).
        const iree_hal_streaming_parameter_op_t *op = &params_info->ops[0];
        for (uint32_t i = 0; i < params_info->copy_count; ++i, ++op) {
          const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
          // Dereference the arg pointer to get the value.
          void *param_ptr = args_array[copy_op.src_ordinal];
          memcpy((uint8_t *)constants + copy_op.dst_offset, param_ptr,
                 copy_op.size);
        }

        // Process all resolve operations (bindings/pointers) - copy the raw
        // pointer. For native kernels, we pass the device pointer value
        // directly.
        for (uint32_t i = 0; i < params_info->binding_count; ++i, ++op) {
          const iree_hal_streaming_parameter_resolve_op_t resolve_op =
              op->resolve;
          void *param_ptr = args_array[resolve_op.src_ordinal];
          // The parameter points to a device pointer (void*)
          void *device_ptr = *(void **)param_ptr;
          // Copy the raw device pointer value into the constants buffer at the
          // correct kernel ABI offset. dst_offset is the kernel's argument
          // offset (from the code object metadata), NOT src_offset (which is a
          // running total used for source buffer layout in non-args-array
          // mode).
          memcpy((uint8_t *)constants + resolve_op.dst_offset, &device_ptr,
                 sizeof(void *));
        }

        binding_list.count = 0; // No IREE bindings, using raw pointers.
        use_raw_arguments = true;
      } else {
        IREE_TRACE_ZONE_END(z0);
        return unpack_status;
      }
    }
  } else {
    // Unpack parameters from packed buffer.
    iree_status_t unpack_status = iree_hal_streaming_unpack_parameters(
        stream->context, &symbol->parameters, params->buffer, constants,
        &binding_list);
    if (!iree_status_is_ok(unpack_status)) {
      // If unpack fails due to NULL or external device pointers, fall back
      // to raw argument passing. This handles native kernels with optional
      // parameters or external allocations (e.g., hipBLASLt workspace).
      if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(unpack_status);
        // Use the raw packed buffer directly.
        constants = params->buffer;
        constants_size = params->buffer_size;
        binding_list.count = 0; // No IREE bindings, using raw pointers.
        use_raw_arguments = true;
      } else {
        IREE_TRACE_ZONE_END(z0);
        return unpack_status;
      }
    }
  }

  // Create IREE dispatch config.
  const iree_hal_dispatch_config_t config = {
      .workgroup_size =
          {
              params->block_dim[0],
              params->block_dim[1],
              params->block_dim[2],
          },
      .workgroup_count =
          {
              params->grid_dim[0],
              params->grid_dim[1],
              params->grid_dim[2],
          },
      .dynamic_workgroup_local_memory = params->shared_memory_bytes,
  };

  // Ensure constants_size is 4-byte aligned as required by HAL.
  constants_size = (constants_size + 3) & ~(size_t)3;

  // --- Resolve pointer-valued constants to overlay bindings ---
  // The constants buffer may contain device pointers passed as raw values
  // (e.g., embedded in struct-typed copy parameters like CatArrInputTensor).
  // For the remote HAL, these are synthetic addresses (0xDEAD...) that the
  // remote device cannot use. We scan the constants for known buffer table
  // entries and convert them to HAL buffer bindings. The dispatch handler
  // overlays the resolved device pointers on top of the constants in the
  // kernarg buffer.
  //
  // When the kernel also has regular bindings (binding_count > 0), we must
  // convert those to overlay format too: write their device pointer values
  // into the constants at their ABI offsets so the scan can find them.
  if (constants && constants_size >= sizeof(void *)) {
    // If we have regular bindings, convert them to overlay format by writing
    // the device pointer values into the constants buffer at their ABI
    // offsets. This allows the overlay scan below to find them alongside
    // any pointers embedded in copy data.
    if (binding_list.count > 0 && !use_raw_arguments && !is_pre_packed) {
      bool is_args_array =
          (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
      const iree_hal_streaming_parameter_op_t *op =
          &symbol->parameters.ops[symbol->parameters.copy_count];
      for (uint32_t i = 0; i < symbol->parameters.binding_count; ++i, ++op) {
        const iree_hal_streaming_parameter_resolve_op_t resolve_op =
            op->resolve;
        void *device_ptr;
        if (is_args_array) {
          void **args_array = (void **)params->buffer;
          device_ptr = *(void **)args_array[resolve_op.src_ordinal];
        } else {
          const uint8_t *parameter_buffer = (const uint8_t *)params->buffer;
          device_ptr = *(void **)(parameter_buffer + resolve_op.src_offset);
        }
        if (device_ptr &&
            resolve_op.dst_offset + sizeof(void *) <= constants_size) {
          memcpy((uint8_t *)constants + resolve_op.dst_offset, &device_ptr,
                 sizeof(void *));
        }
      }
      binding_list.count = 0;
    }

    iree_host_size_t max_overlay = constants_size / sizeof(void *);
    if (max_overlay > 64)
      max_overlay = 64;
    iree_hal_buffer_ref_t *overlay_refs = (iree_hal_buffer_ref_t *)iree_alloca(
        max_overlay * sizeof(iree_hal_buffer_ref_t));
    iree_host_size_t overlay_count = 0;

    // Make a mutable copy of constants so we can zero out resolved pointers.
    void *mutable_constants = iree_alloca(constants_size);
    memcpy(mutable_constants, constants, constants_size);
    constants = mutable_constants;

    // Scan all 8-byte aligned positions in the constants buffer.
    // Device pointers may be embedded in struct-typed parameters (e.g.,
    // std::array<char*, 3> appears as a single 24-byte copy op), so we
    // cannot rely on individual copy op sizes to find pointers.
    for (iree_host_size_t off = 0;
         off + sizeof(void *) <= constants_size && overlay_count < max_overlay;
         off += sizeof(void *)) {
      void *ptr_val;
      memcpy(&ptr_val, (uint8_t *)constants + off, sizeof(void *));
      if (!ptr_val)
        continue;

      iree_hal_streaming_buffer_ref_t sref;
      iree_status_t s = iree_hal_streaming_memory_lookup(
          stream->context, (iree_hal_streaming_deviceptr_t)ptr_val, &sref);
      if (iree_status_is_ok(s)) {
        overlay_refs[overlay_count] = iree_hal_make_buffer_ref(
            sref.buffer->buffer, sref.offset, IREE_HAL_WHOLE_BUFFER);
        overlay_refs[overlay_count].buffer_slot = (int32_t)off;
        ++overlay_count;
        memset((uint8_t *)constants + off, 0, sizeof(void *));
      } else {
        iree_status_ignore(s);
      }
    }

    if (overlay_count > 0) {
      binding_list.count = overlay_count;
      binding_list.values = overlay_refs;
    }
  }

// After the overlay scan, all bindings are in overlay format (buffer_slot
// encodes the constant byte offset). Always use CUSTOM_DIRECT_ARGUMENTS
// so the server overlays resolved device pointers into the constants.
#define IREE_HAL_DISPATCH_FLAG_PRE_PACKED_BUFFER (1ull << 16)
  iree_hal_dispatch_flags_t flags =
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  if (is_pre_packed) {
    flags |= IREE_HAL_DISPATCH_FLAG_PRE_PACKED_BUFFER;
  }

#if IREE_STREAMING_DEBUG_KERNEL_LAUNCH
  if (debug_should_log) {
    fprintf(stderr,
            "[LAUNCH]   DISPATCH: constants_size=%zu bindings=%zu flags=0x%lx "
            "use_raw=%d\n",
            constants_size, binding_list.count, (unsigned long)flags,
            use_raw_arguments);
    // Dump first few pointer-sized values from constants
    if (constants && constants_size >= sizeof(void *)) {
      size_t num_ptrs = constants_size / sizeof(void *);
      if (num_ptrs > 16)
        num_ptrs = 16;
      fprintf(stderr, "[LAUNCH]   constants (first %zu ptrs):", num_ptrs);
      for (size_t i = 0; i < num_ptrs; ++i) {
        void *ptr_val = ((void **)constants)[i];
        fprintf(stderr, " %p", ptr_val);
      }
      fprintf(stderr, "\n");
    }
  }
#endif

  iree_status_t status = iree_hal_command_buffer_dispatch(
      stream->command_buffer, symbol->module->executable,
      symbol->export_ordinal, config,
      iree_make_const_byte_span(constants, constants_size), binding_list,
      flags);

#if IREE_STREAMING_DEBUG_KERNEL_LAUNCH
  if (debug_should_log) {
    if (!iree_status_is_ok(status)) {
      iree_allocator_t allocator = iree_allocator_system();
      char *status_str = NULL;
      iree_host_size_t status_len = 0;
      if (iree_status_to_string(status, &allocator, &status_str, &status_len)) {
        fprintf(stderr, "[LAUNCH]   RESULT: FAILED - %s\n", status_str);
        iree_allocator_free(allocator, status_str);
      } else {
        fprintf(stderr, "[LAUNCH]   RESULT: FAILED - (unknown error)\n");
      }
    } else {
      fprintf(stderr, "[LAUNCH]   RESULT: OK (dispatch recorded)\n");
    }
  }
#endif

  // Insert an execution barrier after each dispatch to enforce serial
  // ordering within the command buffer, emulating HIP stream semantics.
  // This allows batching multiple dispatches per CB submission while
  // maintaining correctness. Inter-CB ordering is handled by timeline
  // semaphore chaining in iree_hal_streaming_stream_flush.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_execution_barrier(
        stream->command_buffer,
        IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, NULL, 0, NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Host callback wrapper structure to adapt CUDA/HIP callbacks to HAL callbacks.
typedef struct iree_hal_streaming_host_callback_t {
  void (*fn)(void *user_data);
  void *user_data;
} iree_hal_streaming_host_callback_t;

// HAL host call function that invokes the CUDA/HIP style callback.
static iree_status_t
iree_hal_streaming_host_callback_thunk(void *user_data, const uint64_t args[4],
                                       iree_hal_host_call_context_t *context) {
  iree_hal_streaming_host_callback_t *callback =
      (iree_hal_streaming_host_callback_t *)user_data;
  callback->fn(callback->user_data);
  iree_allocator_free(iree_allocator_system(), callback);
  return iree_ok_status();
}

iree_status_t
iree_hal_streaming_launch_host_function(iree_hal_streaming_stream_t *stream,
                                        void (*fn)(void *), void *user_data) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add host call node to the graph instead of executing immediately.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_host_call_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, fn, user_data, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Flush any pending operations in the stream's command buffer.
  if (stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_flush(stream));
  }

  // Allocate a wrapper structure to hold the callback and user data.
  iree_hal_streaming_host_callback_t *callback = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*callback),
                                (void **)&callback));
  callback->fn = fn;
  callback->user_data = user_data;

  // Set up semaphores for the host call.
  // Wait for the current stream position.
  uint64_t wait_value = stream->pending_value;
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };

  // Signal the next value after the host call completes.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  // Create the host call with our wrapper function.
  iree_hal_host_call_t call =
      iree_hal_make_host_call(iree_hal_streaming_host_callback_thunk, callback);

  // Empty args array (not used by CUDA/HIP callbacks).
  uint64_t args[4] = {0, 0, 0, 0};

  // Enqueue the host call on the device queue.
  // Use blocking mode so that stream synchronization waits for the host
  // function to complete before returning.
  iree_status_t status = iree_hal_device_queue_host_call(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, call, args, /*flags=*/0);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), callback);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
