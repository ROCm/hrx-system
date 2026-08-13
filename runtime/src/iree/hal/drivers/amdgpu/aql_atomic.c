// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_atomic.h"

#include "iree/hal/drivers/amdgpu/aql_buffer_ref.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"

//===----------------------------------------------------------------------===//
// Recording
//===----------------------------------------------------------------------===//

static iree_hsa_fence_scope_t iree_hal_amdgpu_aql_atomic_handoff_scope(
    iree_hal_execution_stage_t stage_mask, iree_hal_atomic_flags_t atomic_flags,
    iree_hal_atomic_flags_t ordering_flag) {
  if (!iree_any_bit_set(atomic_flags, ordering_flag)) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  return iree_any_bit_set(atomic_flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE) ||
                 iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)
             ? IREE_HSA_FENCE_SCOPE_SYSTEM
             : IREE_HSA_FENCE_SCOPE_AGENT;
}

static void iree_hal_amdgpu_aql_atomic_record_dependency(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t stage_mask, iree_hal_atomic_flags_t atomic_flags,
    iree_hal_atomic_flags_t ordering_flag) {
  if (stage_mask == 0) return;
  const iree_hsa_fence_scope_t handoff_scope =
      iree_hal_amdgpu_aql_atomic_handoff_scope(stage_mask, atomic_flags,
                                               ordering_flag);
  iree_hal_amdgpu_aql_program_builder_add_execution_dependency(
      builder, (uint8_t)handoff_scope, (uint8_t)handoff_scope);
}

iree_status_t iree_hal_amdgpu_aql_atomic_record_wait(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_wait_params_t params) {
  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, source_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_NONE);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_WAIT,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_atomic_wait_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_binding_sources=*/NULL));
  iree_hal_amdgpu_command_buffer_atomic_wait_command_t* atomic_wait =
      (iree_hal_amdgpu_command_buffer_atomic_wait_command_t*)header;
  atomic_wait->target = target;
  atomic_wait->value = params.value;
  atomic_wait->mask = params.mask;
  atomic_wait->atomic_flags = params.flags;
  atomic_wait->width = params.width;
  atomic_wait->condition = params.condition;

  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, target_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_ACQUIRE);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_atomic_record_store(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_store_params_t params) {
  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, source_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_RELEASE);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_STORE,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_atomic_store_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_binding_sources=*/NULL));
  iree_hal_amdgpu_command_buffer_atomic_store_command_t* atomic_store =
      (iree_hal_amdgpu_command_buffer_atomic_store_command_t*)header;
  atomic_store->target = target;
  atomic_store->value = params.value;
  atomic_store->atomic_flags = params.flags;
  atomic_store->width = params.width;

  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, target_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_NONE);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_atomic_record_rmw(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, source_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_RELEASE);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_RMW,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_atomic_rmw_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_binding_sources=*/NULL));
  iree_hal_amdgpu_command_buffer_atomic_rmw_command_t* atomic_rmw =
      (iree_hal_amdgpu_command_buffer_atomic_rmw_command_t*)header;
  atomic_rmw->target = target;
  atomic_rmw->operand = params.operand;
  atomic_rmw->atomic_flags = params.flags;
  atomic_rmw->width = params.width;
  atomic_rmw->operation = params.operation;

  iree_hal_amdgpu_aql_atomic_record_dependency(
      builder, target_stage_mask, params.flags, IREE_HAL_ATOMIC_FLAG_ACQUIRE);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Replay
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_aql_atomic_target_params_t {
  // Recorded static or dynamic target reference.
  const iree_hal_amdgpu_command_buffer_atomic_target_t* target;
  // Width of the atomic memory location.
  iree_hal_atomic_width_t width;
  // Ordering and coherence-domain flags.
  iree_hal_atomic_flags_t flags;
  // Buffer usage required by the operation.
  iree_hal_buffer_usage_t required_usage;
  // Buffer access required by the operation.
  iree_hal_memory_access_t required_access;
} iree_hal_amdgpu_aql_atomic_target_params_t;

static iree_status_t iree_hal_amdgpu_aql_atomic_resolve_target(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_aql_atomic_target_params_t params,
    uint8_t** out_target_pointer) {
  *out_target_pointer = NULL;
  const iree_device_size_t byte_count =
      iree_hal_atomic_width_byte_count(params.width);
  if (IREE_UNLIKELY(byte_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed AQL atomic width %u", params.width);
  }

  iree_hal_buffer_ref_t resolved_ref;
  uint8_t* target_pointer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_resolve_command_buffer_ref(
      command_buffer, binding_table, params.target->kind,
      params.target->ordinal, params.target->offset, byte_count,
      params.required_usage, params.required_access, &resolved_ref,
      &target_pointer));
  if (IREE_UNLIKELY(((uintptr_t)target_pointer % byte_count) != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AQL atomic target address is not naturally aligned "
        "(address=0x%" PRIxPTR ", alignment=%" PRIdsz ")",
        (uintptr_t)target_pointer, byte_count);
  }

  const iree_hal_amdgpu_atomic_memory_cell_flags_t required_cell =
      iree_hal_amdgpu_atomic_memory_required_cell(params.width, params.flags);
  const iree_hal_amdgpu_atomic_memory_cell_flags_t available_cells =
      iree_hal_amdgpu_buffer_atomic_memory_cells(
          iree_hal_buffer_allocated_buffer(resolved_ref.buffer));
  if (IREE_UNLIKELY(!iree_all_bits_set(available_cells, required_cell))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AQL atomic target memory does not support %u-bit %s-scope atomics",
        params.width,
        iree_any_bit_set(params.flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE)
            ? "system"
            : "device");
  }

  *out_target_pointer = target_pointer;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_atomic_emplace_command(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hsa_kernel_dispatch_packet_t* dispatch_packet, void* kernarg_ptr) {
  iree_hal_amdgpu_aql_atomic_target_params_t target_params = {0};
  switch (command->opcode) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_WAIT: {
      const iree_hal_amdgpu_command_buffer_atomic_wait_command_t* atomic_wait =
          (const iree_hal_amdgpu_command_buffer_atomic_wait_command_t*)command;
      target_params = (iree_hal_amdgpu_aql_atomic_target_params_t){
          .target = &atomic_wait->target,
          .width = atomic_wait->width,
          .flags = atomic_wait->atomic_flags,
          .required_usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ,
          .required_access = IREE_HAL_MEMORY_ACCESS_READ,
      };
      uint8_t* target_pointer = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_atomic_resolve_target(
          command_buffer, binding_table, target_params, &target_pointer));
      iree_hal_amdgpu_device_atomic_wait_emplace(
          kernels, dispatch_packet, target_pointer,
          (iree_hal_atomic_wait_params_t){
              .value = atomic_wait->value,
              .mask = atomic_wait->mask,
              .flags = atomic_wait->atomic_flags,
              .width = atomic_wait->width,
              .condition = atomic_wait->condition,
          },
          kernarg_ptr);
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_STORE: {
      const iree_hal_amdgpu_command_buffer_atomic_store_command_t*
          atomic_store =
              (const iree_hal_amdgpu_command_buffer_atomic_store_command_t*)
                  command;
      target_params = (iree_hal_amdgpu_aql_atomic_target_params_t){
          .target = &atomic_store->target,
          .width = atomic_store->width,
          .flags = atomic_store->atomic_flags,
          .required_usage = IREE_HAL_BUFFER_USAGE_STORAGE_WRITE,
          .required_access = IREE_HAL_MEMORY_ACCESS_WRITE,
      };
      uint8_t* target_pointer = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_atomic_resolve_target(
          command_buffer, binding_table, target_params, &target_pointer));
      iree_hal_amdgpu_device_atomic_store_emplace(
          kernels, dispatch_packet, target_pointer,
          (iree_hal_atomic_store_params_t){
              .value = atomic_store->value,
              .flags = atomic_store->atomic_flags,
              .width = atomic_store->width,
          },
          kernarg_ptr);
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_ATOMIC_RMW: {
      const iree_hal_amdgpu_command_buffer_atomic_rmw_command_t* atomic_rmw =
          (const iree_hal_amdgpu_command_buffer_atomic_rmw_command_t*)command;
      target_params = (iree_hal_amdgpu_aql_atomic_target_params_t){
          .target = &atomic_rmw->target,
          .width = atomic_rmw->width,
          .flags = atomic_rmw->atomic_flags,
          .required_usage = IREE_HAL_BUFFER_USAGE_STORAGE,
          .required_access =
              IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      };
      uint8_t* target_pointer = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_atomic_resolve_target(
          command_buffer, binding_table, target_params, &target_pointer));
      iree_hal_amdgpu_device_atomic_rmw_emplace(
          kernels, dispatch_packet, target_pointer,
          (iree_hal_atomic_rmw_params_t){
              .operand = atomic_rmw->operand,
              .flags = atomic_rmw->atomic_flags,
              .width = atomic_rmw->width,
              .operation = atomic_rmw->operation,
          },
          kernarg_ptr);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL opcode %u is not an atomic command",
                              command->opcode);
  }
}
