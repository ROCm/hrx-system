// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/queue.h"

#include "iree/module/hal/native_schema.h"

static_assert((uint32_t)IREE_HAL_MODULE_TYPE_BUFFER ==
                  (uint32_t)IREE_HAL_NATIVE_REF_BUFFER,
              "buffer ordinal must match the generated ABI");
static_assert((uint32_t)IREE_HAL_MODULE_TYPE_QUEUE ==
                  (uint32_t)IREE_HAL_NATIVE_REF_QUEUE,
              "queue ordinal must match the generated ABI");
static_assert((uint32_t)IREE_HAL_MODULE_TYPE_SEMAPHORE ==
                  (uint32_t)IREE_HAL_NATIVE_REF_SEMAPHORE,
              "semaphore ordinal must match the generated ABI");

// Linking reserves this much space per physical record. Every allocation is
// eight-byte aligned and consumes only the mandatory native descriptor storage.
// Optional-list wrapper records need no scratch; their views live on the stack.
#define IREE_HAL_VM_SCRATCH_SIZE(type) ((sizeof(type) + 7u) & ~(size_t)7u)
static const uint32_t kCopyScratchSizes[IREE_ARRAYSIZE(kIreeHalNativeLayouts)] =
    {
        [IREE_HAL_NATIVE_TIMEPOINT_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_semaphore_t*) + sizeof(uint64_t),
        [IREE_HAL_NATIVE_MEMORY_BARRIER_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_memory_barrier_t),
        [IREE_HAL_NATIVE_TRANSITION_OPERATION_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_memory_transition_recipe_info_t),
        [IREE_HAL_NATIVE_TRANSITION_RECIPE_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_memory_transition_recipe_t),
        [IREE_HAL_NATIVE_BUFFER_BARRIER_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_buffer_barrier_t),
        [IREE_HAL_NATIVE_BARRIER_LAYOUT] =
            IREE_HAL_VM_SCRATCH_SIZE(iree_hal_barrier_t),
};
#undef IREE_HAL_VM_SCRATCH_SIZE

// Scratch extents and alignment have already been established by linking.
static void* iree_hal_vm_take_scratch(uint8_t** cursor, iree_host_size_t count,
                                      iree_host_size_t element_size) {
  if (count == 0) {
    return NULL;
  }
  void* result = *cursor;
  *cursor += iree_host_align(count * element_size, 8);
  return result;
}

static iree_status_t iree_hal_vm_copy_timepoints(
    const iree_hal_module_types_t* types, const iree_vm_native_call_t* call,
    const uint32_t* range, uint8_t** scratch,
    iree_hal_semaphore_list_t* out_list) {
  const uint32_t count = range[0];
  const uint32_t* record = call->record + range[1];
  iree_hal_semaphore_t** semaphores =
      iree_hal_vm_take_scratch(scratch, count, sizeof(*semaphores));
  uint64_t* values = iree_hal_vm_take_scratch(scratch, count, sizeof(*values));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    iree_hal_vm_semaphore_t* semaphore = NULL;
    status = iree_hal_vm_semaphore_ptr_from_ref_borrowed(
        types, call->refs[record[IREE_HAL_NATIVE_TIMEPOINT_SEMAPHORE]],
        &semaphore);
    if (iree_status_is_ok(status) && !semaphore) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null HAL semaphore");
    }
    if (iree_status_is_ok(status)) {
      semaphores[i] = semaphore->semaphore;
      values[i] = iree_vm_native_call_load_value(
          call, record[IREE_HAL_NATIVE_TIMEPOINT_VALUE]);
    }
    record += IREE_HAL_NATIVE_TIMEPOINT_WORD_COUNT;
  }
  *out_list = (iree_hal_semaphore_list_t){count, semaphores, values};
  return status;
}

static const iree_hal_memory_transition_recipe_t* iree_hal_vm_copy_recipe(
    const iree_vm_native_call_t* call, const uint32_t* optional,
    uint8_t** scratch) {
  if (!optional[0]) {
    return NULL;
  }
  const uint32_t* record = call->record + optional[1];
  iree_hal_memory_transition_recipe_t* recipe =
      iree_hal_vm_take_scratch(scratch, 1, sizeof(*recipe));
  const uint32_t* range = record + IREE_HAL_NATIVE_TRANSITION_RECIPE_OPERATIONS;
  iree_hal_memory_transition_recipe_info_t* operations =
      iree_hal_vm_take_scratch(scratch, range[0], sizeof(*operations));
  *recipe = (iree_hal_memory_transition_recipe_t){
      .effects = {(uint32_t)iree_vm_native_call_load_value(
          call, record[IREE_HAL_NATIVE_TRANSITION_RECIPE_EFFECTS])},
      .operation_count = range[0],
      .operations = operations,
  };
  record = call->record + range[1];
  for (uint32_t i = 0; i < range[0]; ++i) {
    operations[i] = (iree_hal_memory_transition_recipe_info_t){
        .kind =
            (iree_hal_memory_transition_kind_t)iree_vm_native_call_load_value(
                call, record[IREE_HAL_NATIVE_TRANSITION_OPERATION_KIND]),
        .executor = (iree_hal_memory_transition_executor_t)
            iree_vm_native_call_load_value(
                call, record[IREE_HAL_NATIVE_TRANSITION_OPERATION_EXECUTOR]),
        .operation = (iree_hal_memory_transition_operation_t)
            iree_vm_native_call_load_value(
                call, record[IREE_HAL_NATIVE_TRANSITION_OPERATION_OPERATION]),
        .range_granularity = iree_vm_native_call_load_value(
            call,
            record[IREE_HAL_NATIVE_TRANSITION_OPERATION_RANGE_GRANULARITY]),
        .host =
            {
                .instruction = (iree_hal_host_cache_instruction_t)
                    iree_vm_native_call_load_value(
                        call,
                        record
                            [IREE_HAL_NATIVE_TRANSITION_OPERATION_HOST_INSTRUCTION]),
                .fence_before =
                    (iree_hal_host_cache_fence_t)iree_vm_native_call_load_value(
                        call,
                        record
                            [IREE_HAL_NATIVE_TRANSITION_OPERATION_HOST_FENCE_BEFORE]),
                .fence_after =
                    (iree_hal_host_cache_fence_t)iree_vm_native_call_load_value(
                        call,
                        record
                            [IREE_HAL_NATIVE_TRANSITION_OPERATION_HOST_FENCE_AFTER]),
            },
    };
    record += IREE_HAL_NATIVE_TRANSITION_OPERATION_WORD_COUNT;
  }
  return recipe;
}

static iree_status_t iree_hal_vm_copy_barriers(
    const iree_hal_module_types_t* types, const iree_vm_native_call_t* call,
    const uint32_t* optional, uint8_t** scratch,
    iree_hal_barrier_list_t* out_list) {
  if (!optional[0]) {
    return iree_ok_status();
  }
  const uint32_t* range = call->record + optional[1];
  const uint32_t count = range[0];
  const uint32_t* record = call->record + range[1];
  iree_hal_barrier_t* barriers =
      iree_hal_vm_take_scratch(scratch, count, sizeof(*barriers));
  *out_list = (iree_hal_barrier_list_t){count, barriers};
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    const uint32_t* memory = record + IREE_HAL_NATIVE_BARRIER_MEMORY;
    const uint32_t* buffers = record + IREE_HAL_NATIVE_BARRIER_BUFFERS;
    iree_hal_memory_barrier_t* memory_barriers =
        iree_hal_vm_take_scratch(scratch, memory[0], sizeof(*memory_barriers));
    iree_hal_buffer_barrier_t* buffer_barriers =
        iree_hal_vm_take_scratch(scratch, buffers[0], sizeof(*buffer_barriers));
    barriers[i] = (iree_hal_barrier_t){
        .source_stage_mask =
            (iree_hal_execution_stage_t)iree_vm_native_call_load_value(
                call, record[IREE_HAL_NATIVE_BARRIER_SOURCE_STAGE]),
        .target_stage_mask =
            (iree_hal_execution_stage_t)iree_vm_native_call_load_value(
                call, record[IREE_HAL_NATIVE_BARRIER_TARGET_STAGE]),
        .flags = iree_vm_native_call_load_value(
            call, record[IREE_HAL_NATIVE_BARRIER_FLAGS]),
        .effects = {(uint32_t)iree_vm_native_call_load_value(
            call, record[IREE_HAL_NATIVE_BARRIER_EFFECTS])},
        .memory_barrier_count = memory[0],
        .memory_barriers = memory_barriers,
        .buffer_barrier_count = buffers[0],
        .buffer_barriers = buffer_barriers,
    };
    const uint32_t* item = call->record + memory[1];
    for (uint32_t j = 0; j < memory[0]; ++j) {
      memory_barriers[j] = (iree_hal_memory_barrier_t){
          .source_scope =
              (iree_hal_access_scope_t)iree_vm_native_call_load_value(
                  call, item[IREE_HAL_NATIVE_MEMORY_BARRIER_SOURCE_SCOPE]),
          .target_scope =
              (iree_hal_access_scope_t)iree_vm_native_call_load_value(
                  call, item[IREE_HAL_NATIVE_MEMORY_BARRIER_TARGET_SCOPE]),
      };
      item += IREE_HAL_NATIVE_MEMORY_BARRIER_WORD_COUNT;
    }
    item = call->record + buffers[1];
    for (uint32_t j = 0; iree_status_is_ok(status) && j < buffers[0]; ++j) {
      iree_hal_buffer_t* buffer = NULL;
      status = iree_hal_buffer_ptr_from_ref_borrowed(
          types, call->refs[item[IREE_HAL_NATIVE_BUFFER_BARRIER_BUFFER_BUFFER]],
          &buffer);
      if (iree_status_is_ok(status) && !buffer) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "null HAL barrier buffer");
      }
      if (iree_status_is_ok(status)) {
        buffer_barriers[j] = (iree_hal_buffer_barrier_t){
            .source_scope =
                (iree_hal_access_scope_t)iree_vm_native_call_load_value(
                    call, item[IREE_HAL_NATIVE_BUFFER_BARRIER_SOURCE_SCOPE]),
            .target_scope =
                (iree_hal_access_scope_t)iree_vm_native_call_load_value(
                    call, item[IREE_HAL_NATIVE_BUFFER_BARRIER_TARGET_SCOPE]),
            .buffer_ref =
                {
                    .buffer = buffer,
                    .offset = iree_vm_native_call_load_value(
                        call,
                        item[IREE_HAL_NATIVE_BUFFER_BARRIER_BUFFER_OFFSET]),
                    .length = iree_vm_native_call_load_value(
                        call,
                        item[IREE_HAL_NATIVE_BUFFER_BARRIER_BUFFER_LENGTH]),
                },
            .recipe = iree_hal_vm_copy_recipe(
                call, item + IREE_HAL_NATIVE_BUFFER_BARRIER_RECIPE, scratch),
        };
      }
      item += IREE_HAL_NATIVE_BUFFER_BARRIER_WORD_COUNT;
    }
    record += IREE_HAL_NATIVE_BARRIER_WORD_COUNT;
  }
  return status;
}

static iree_status_t iree_hal_vm_queue_copy(const void* user_data,
                                            const iree_vm_native_call_t* call) {
  const iree_hal_module_types_t* types = user_data;
  const uint32_t* record = call->record;
  iree_hal_vm_queue_t* queue = NULL;
  iree_hal_buffer_t *source = NULL, *target = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vm_queue_ptr_from_ref_borrowed(
      types, call->refs[record[IREE_HAL_NATIVE_QUEUE_COPY_QUEUE]], &queue));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_ptr_from_ref_borrowed(
      types, call->refs[record[IREE_HAL_NATIVE_QUEUE_COPY_SOURCE_BUFFER]],
      &source));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_ptr_from_ref_borrowed(
      types, call->refs[record[IREE_HAL_NATIVE_QUEUE_COPY_TARGET_BUFFER]],
      &target));
  if (!queue || !source || !target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "null HAL copy handle");
  }
  uint8_t* scratch = call->scratch;
  iree_hal_semaphore_list_t waits, signals;
  IREE_RETURN_IF_ERROR(iree_hal_vm_copy_timepoints(
      types, call, record + IREE_HAL_NATIVE_QUEUE_COPY_WAITS, &scratch,
      &waits));
  IREE_RETURN_IF_ERROR(iree_hal_vm_copy_timepoints(
      types, call, record + IREE_HAL_NATIVE_QUEUE_COPY_SIGNALS, &scratch,
      &signals));
  iree_hal_barrier_list_t before = {0}, after = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vm_copy_barriers(
      types, call, record + IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE,
      &scratch, &before));
  IREE_RETURN_IF_ERROR(iree_hal_vm_copy_barriers(
      types, call, record + IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_AFTER, &scratch,
      &after));
  const iree_hal_queue_barriers_t barriers = {
      .before =
          record[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE] ? &before : NULL,
      .after =
          record[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_AFTER] ? &after : NULL,
  };
  return iree_hal_queue_copy(
      queue->queue, waits, signals, source,
      iree_vm_native_call_load_value(
          call, record[IREE_HAL_NATIVE_QUEUE_COPY_SOURCE_OFFSET]),
      target,
      iree_vm_native_call_load_value(
          call, record[IREE_HAL_NATIVE_QUEUE_COPY_TARGET_OFFSET]),
      iree_vm_native_call_load_value(call,
                                     record[IREE_HAL_NATIVE_QUEUE_COPY_LENGTH]),
      &barriers,
      iree_vm_native_call_load_value(call,
                                     record[IREE_HAL_NATIVE_QUEUE_COPY_FLAGS]));
}

static iree_status_t iree_hal_vm_queue_copy_verify(const uint32_t* record) {
  return iree_hal_vm_native_verify_optionals(record,
                                             IREE_HAL_NATIVE_QUEUE_COPY_ROOT);
}

iree_vm_native_export_t iree_hal_vm_queue_copy_export(
    const iree_hal_module_types_t* types) {
  return (iree_vm_native_export_t){
      .name = kIreeHalNativeMethodNames[IREE_HAL_NATIVE_QUEUE_COPY],
      .version = IREE_HAL_NATIVE_QUEUE_COPY_VERSION,
      .schema = &kIreeHalNativeSchemas[IREE_HAL_NATIVE_QUEUE_COPY],
      .scratch_sizes = kCopyScratchSizes,
      .verify = iree_hal_vm_queue_copy_verify,
      .entry = iree_hal_vm_queue_copy,
      .user_data = types,
  };
}
