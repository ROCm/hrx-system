// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/legacy_launch_state.h"

#include <stdbool.h>
#include <string.h>

#include "common/tls.h"
#include "iree/base/threading/call_once.h"

typedef struct iree_hip_legacy_launch_stack_t {
  // Owned array of launch frames in push order.
  iree_hip_legacy_launch_frame_t* frames;
  // Number of live frames in |frames|.
  size_t frame_count;
  // Allocated frame capacity of |frames|.
  size_t frame_capacity;
  // True after this thread registered for dynamic-storage cleanup.
  bool cleanup_registered;
  // Allocation-free storage for the common single-frame launch sequence.
  iree_hip_legacy_launch_frame_t inline_frame;
} iree_hip_legacy_launch_stack_t;

static IREE_THREAD_LOCAL iree_hip_legacy_launch_stack_t
    iree_hip_legacy_launch_stack = {0};
static iree_once_flag iree_hip_legacy_launch_key_once = IREE_ONCE_FLAG_INIT;
static iree_hal_streaming_tls_key_t iree_hip_legacy_launch_key =
    IREE_HAL_STREAMING_TLS_KEY_INVALID;
static iree_status_code_t iree_hip_legacy_launch_key_status = IREE_STATUS_OK;

void iree_hip_legacy_launch_frame_deinitialize(
    iree_hip_legacy_launch_frame_t* frame) {
  if (!frame) return;
  iree_allocator_free(iree_allocator_system(), frame->argument_data);
  memset(frame, 0, sizeof(*frame));
}

static void iree_hip_legacy_launch_stack_destroy(void* value) {
  iree_hip_legacy_launch_stack_t* stack =
      (iree_hip_legacy_launch_stack_t*)value;
  for (size_t i = 0; i < stack->frame_count; ++i) {
    iree_hip_legacy_launch_frame_deinitialize(&stack->frames[i]);
  }
  if (stack->frames != &stack->inline_frame) {
    iree_allocator_free(iree_allocator_system(), stack->frames);
  }
  memset(stack, 0, sizeof(*stack));
}

static void iree_hip_legacy_launch_key_initialize(void) {
  iree_status_t status = iree_hal_streaming_tls_key_create(
      &iree_hip_legacy_launch_key, iree_hip_legacy_launch_stack_destroy);
  iree_hip_legacy_launch_key_status = iree_status_code(status);
  iree_status_ignore(status);
}

static hipError_t iree_hip_status_code_to_legacy_launch_result(
    iree_status_code_t status_code) {
  return status_code == IREE_STATUS_RESOURCE_EXHAUSTED ? hipErrorOutOfMemory
                                                       : hipErrorUnknown;
}

static hipError_t iree_hip_legacy_launch_stack_register_cleanup(
    iree_hip_legacy_launch_stack_t* stack) {
  if (stack->cleanup_registered) return hipSuccess;
  iree_call_once(&iree_hip_legacy_launch_key_once,
                 iree_hip_legacy_launch_key_initialize);
  if (IREE_UNLIKELY(iree_hip_legacy_launch_key_status != IREE_STATUS_OK)) {
    return iree_hip_status_code_to_legacy_launch_result(
        iree_hip_legacy_launch_key_status);
  }

  iree_status_t status =
      iree_hal_streaming_tls_set(iree_hip_legacy_launch_key, stack);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return iree_hip_status_code_to_legacy_launch_result(status_code);
  }
  stack->cleanup_registered = true;
  return hipSuccess;
}

static hipError_t iree_hip_legacy_launch_stack_unregister_cleanup(
    iree_hip_legacy_launch_stack_t* stack) {
  if (!stack->cleanup_registered) return hipSuccess;
  iree_status_t status =
      iree_hal_streaming_tls_set(iree_hip_legacy_launch_key, NULL);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return iree_hip_status_code_to_legacy_launch_result(status_code);
  }
  stack->cleanup_registered = false;
  return hipSuccess;
}

static bool iree_hip_legacy_launch_grow_capacity(size_t current_capacity,
                                                 size_t required_capacity,
                                                 size_t initial_capacity,
                                                 size_t* out_capacity) {
  size_t new_capacity = current_capacity ? current_capacity : initial_capacity;
  while (new_capacity < required_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = required_capacity;
      break;
    }
    new_capacity *= 2;
  }
  if (new_capacity < required_capacity) return false;
  *out_capacity = new_capacity;
  return true;
}

static hipError_t iree_hip_legacy_launch_stack_reserve(
    iree_hip_legacy_launch_stack_t* stack, size_t required_capacity) {
  if (!stack->frames) {
    stack->frames = &stack->inline_frame;
    stack->frame_capacity = 1;
  }
  if (required_capacity <= stack->frame_capacity) return hipSuccess;
  hipError_t result = iree_hip_legacy_launch_stack_register_cleanup(stack);
  if (result != hipSuccess) return result;
  size_t new_capacity = 0;
  if (!iree_hip_legacy_launch_grow_capacity(
          stack->frame_capacity, required_capacity, 4, &new_capacity)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_legacy_launch_frame_t* new_frames = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(iree_allocator_system(), new_capacity,
                                  sizeof(*new_frames), (void**)&new_frames);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return iree_hip_status_code_to_legacy_launch_result(status_code);
  }
  memcpy(new_frames, stack->frames, stack->frame_count * sizeof(*new_frames));
  if (stack->frames != &stack->inline_frame) {
    iree_allocator_free(iree_allocator_system(), stack->frames);
  }
  stack->frames = new_frames;
  stack->frame_capacity = new_capacity;
  return hipSuccess;
}

static hipError_t iree_hip_legacy_launch_frame_reserve(
    iree_hip_legacy_launch_stack_t* stack,
    iree_hip_legacy_launch_frame_t* frame, size_t required_capacity) {
  if (required_capacity <= frame->argument_capacity) return hipSuccess;
  hipError_t result = iree_hip_legacy_launch_stack_register_cleanup(stack);
  if (result != hipSuccess) return result;
  size_t new_capacity = 0;
  if (!iree_hip_legacy_launch_grow_capacity(
          frame->argument_capacity, required_capacity, 64, &new_capacity)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status = iree_allocator_realloc(
      iree_allocator_system(), new_capacity, (void**)&frame->argument_data);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return iree_hip_status_code_to_legacy_launch_result(status_code);
  }
  frame->argument_capacity = new_capacity;
  return hipSuccess;
}

hipError_t iree_hip_legacy_launch_state_push(dim3 grid_dimension,
                                             dim3 block_dimension,
                                             size_t shared_memory_bytes,
                                             hipStream_t stream) {
  iree_hip_legacy_launch_stack_t* stack = &iree_hip_legacy_launch_stack;

  size_t required_capacity = 0;
  if (!iree_host_size_checked_add(stack->frame_count, 1, &required_capacity)) {
    return hipErrorOutOfMemory;
  }
  hipError_t result =
      iree_hip_legacy_launch_stack_reserve(stack, required_capacity);
  if (result != hipSuccess) return result;

  iree_hip_legacy_launch_frame_t* frame = &stack->frames[stack->frame_count++];
  memset(frame, 0, sizeof(*frame));
  frame->grid_dimension = grid_dimension;
  frame->block_dimension = block_dimension;
  frame->shared_memory_bytes = shared_memory_bytes;
  frame->stream = stream;
  return hipSuccess;
}

hipError_t iree_hip_legacy_launch_state_setup_argument(const void* argument,
                                                       size_t size,
                                                       size_t offset) {
  if (size != 0 && !argument) return hipErrorInvalidValue;
  size_t required_capacity = 0;
  if (!iree_host_size_checked_add(offset, size, &required_capacity)) {
    return hipErrorInvalidValue;
  }

  iree_hip_legacy_launch_stack_t* stack = &iree_hip_legacy_launch_stack;
  if (stack->frame_count == 0) {
    return hipErrorMissingConfiguration;
  }

  iree_hip_legacy_launch_frame_t* frame =
      &stack->frames[stack->frame_count - 1];
  hipError_t result =
      iree_hip_legacy_launch_frame_reserve(stack, frame, required_capacity);
  if (result != hipSuccess) return result;

  // Offsets describe the native kernarg layout and may leave alignment gaps.
  // Preserve prior writes, initialize only a newly exposed gap, and copy the
  // argument directly into its final position.
  if (offset > frame->argument_length) {
    memset(frame->argument_data + frame->argument_length, 0,
           offset - frame->argument_length);
  }
  if (size != 0) {
    memcpy(frame->argument_data + offset, argument, size);
  }
  if (required_capacity > frame->argument_length) {
    frame->argument_length = required_capacity;
  }
  return hipSuccess;
}

hipError_t iree_hip_legacy_launch_state_pop(
    iree_hip_legacy_launch_frame_t* out_frame) {
  IREE_ASSERT_ARGUMENT(out_frame);
  memset(out_frame, 0, sizeof(*out_frame));

  iree_hip_legacy_launch_stack_t* stack = &iree_hip_legacy_launch_stack;
  if (stack->frame_count == 0) {
    return hipErrorMissingConfiguration;
  }

  if (stack->frame_count == 1) {
    hipError_t result = iree_hip_legacy_launch_stack_unregister_cleanup(stack);
    if (result != hipSuccess) return result;
  }

  iree_hip_legacy_launch_frame_t* frame = &stack->frames[--stack->frame_count];
  *out_frame = *frame;
  memset(frame, 0, sizeof(*frame));
  if (stack->frame_count == 0 && stack->frames != &stack->inline_frame) {
    iree_allocator_free(iree_allocator_system(), stack->frames);
    stack->frames = &stack->inline_frame;
    stack->frame_capacity = 1;
  }
  return hipSuccess;
}
