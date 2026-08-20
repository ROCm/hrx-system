// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Compile-time checks for HAL ISA wire declarations.
// clang-format off

#include "iree/base/alignment.h"
#include "iree/vm/bytecode/wire/hal/buffer.h"
#include "iree/vm/bytecode/wire/hal/command_buffer.h"
#include "iree/vm/bytecode/wire/hal/device.h"
#include "iree/vm/bytecode/wire/hal/queue.h"
#include "iree/vm/bytecode/wire/hal/semaphore.h"

static_assert(sizeof(iree_vm_isa_hal_buffer_allocate_record_t) == 12, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_hal_buffer_allocate_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, device_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, usage_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, access_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, memory_type_v8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, affinity_v8) == 7, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, min_alignment_v8) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, allocation_size_v8) == 9,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_allocate_record_t, zero_padding_u16) == 10,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_buffer_map_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_buffer_map_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, source_buffer_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, source_offset_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, source_length_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, access_u8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_map_record_t, zero_padding_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_buffer_unmap_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_buffer_unmap_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_buffer_unmap_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_unmap_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_unmap_record_t, mapped_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_buffer_unmap_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_create_record_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_create_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, device_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, mode_u32) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, categories_u32) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, affinity_v8) == 12, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, binding_capacity_v8) == 13,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_create_record_t, zero_padding_u8) == 14, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_finalize_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_cmd_finalize_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_finalize_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_finalize_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_finalize_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_finalize_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_execution_barrier_record_t) == 32, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_execution_barrier_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, page_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, opcode_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, source_stage_v8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, target_stage_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, zero_padding_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, flags_u32) == 8,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, memory_source_scope_base_u16) == 12,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, memory_target_scope_base_u16) == 14,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, memory_count_u16) == 16,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_source_scope_base_u16) == 18,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_target_scope_base_u16) == 20,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_ref_base_u16) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_slot_base_u16) == 24,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_offset_base_u16) == 26,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_length_base_u16) == 28,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_execution_barrier_record_t, buffer_count_u16) == 30,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_advise_buffer_record_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_advise_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, buffer_r8_nullable) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, buffer_slot_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, buffer_offset_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, buffer_length_v8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, zero_padding0_u8) == 7,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, flags_u32) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, arg0_v8) == 12, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, arg1_v8) == 13, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_advise_buffer_record_t, zero_padding1_u8) == 14,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_fill_buffer_record_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_fill_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, target_buffer_r8_nullable) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, target_slot_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, target_offset_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, target_length_v8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, pattern_v8) == 7, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, pattern_width_u8) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, zero_padding_u8) == 9,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_fill_buffer_record_t, flags_u32) == 12, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_update_buffer_record_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_update_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, source_vm_buffer_r8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, source_offset_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, target_buffer_r8_nullable) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, target_slot_v8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, target_offset_v8) == 7,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, target_length_v8) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, zero_padding_u8) == 9,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_update_buffer_record_t, flags_u32) == 12, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_copy_buffer_record_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_copy_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, source_buffer_r8_nullable) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, source_slot_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, source_offset_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, target_buffer_r8_nullable) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, target_slot_v8) == 7,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, target_offset_v8) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, length_v8) == 9, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, zero_padding_u8) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_copy_buffer_record_t, flags_u32) == 12, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_collective_record_t) == 20, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_collective_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, channel_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, op_u32) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, param_v8) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, send_buffer_r8_nullable) == 9,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, send_slot_v8) == 10, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, send_offset_v8) == 11,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, send_length_v8) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, recv_buffer_r8_nullable) == 13,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, recv_slot_v8) == 14, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, recv_offset_v8) == 15,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, recv_length_v8) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, element_count_v8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_collective_record_t, zero_padding_u8) == 18,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_dispatch_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_dispatch_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, function_table_r8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, function_ordinal_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, barrier_before_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, launch_base_u16) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, constant_base_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, constant_count_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, binding_buffer_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, binding_slot_base_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, binding_offset_base_u16) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, binding_length_base_u16) == 18,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, binding_count_u16) == 20,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, zero_padding_u8) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t) == 32, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, page_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, opcode_u8) == 1,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, command_buffer_r8) == 2,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, function_table_r8) == 3,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, function_ordinal_v8) == 4,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, barrier_before_u8) == 5,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, launch_base_u16) == 6,
              "wire offset");
static_assert(
    offsetof(
        iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t,
        workgroup_count_buffer_r8_nullable) == 8,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, workgroup_count_slot_v8) == 9,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, workgroup_count_offset_v8) == 10,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, zero_padding0_u8) == 11,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, constant_base_u16) == 12,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, constant_count_u16) == 14,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, binding_buffer_base_u16) == 16,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, binding_slot_base_u16) == 18,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, binding_offset_base_u16) == 20,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, binding_length_base_u16) == 22,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, binding_count_u16) == 24,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, zero_padding1_u8) == 26,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t, flags_u32) == 28,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_debug_group_begin_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_cmd_debug_group_begin_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, page_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, opcode_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, zero_padding0_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, label_string_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, zero_padding1_u8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_begin_record_t, color_u32) == 8,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_cmd_debug_group_end_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_cmd_debug_group_end_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_end_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_end_record_t, opcode_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_end_record_t, command_buffer_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_cmd_debug_group_end_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_device_group_count_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_device_group_count_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_device_group_count_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_count_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_count_record_t, dst_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_count_record_t, group_r8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_device_group_get_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_device_group_get_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, group_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, index_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_device_group_get_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_executable_load_record_t) == 16, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_hal_executable_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, device_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, affinity_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, zero_padding0_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, resolver_string_u16) == 6,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_executable_load_record_t, payload_vm_buffer_r8_nullable) == 8,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, zero_padding1_u8) == 9,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, name_table_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, selected_ordinal_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_executable_load_record_t, selected_ordinal_count_u16) == 14,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_alloca_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_alloca_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, device_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, affinity_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, pool_r8_nullable) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, wait_semaphore_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, wait_payload_base_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, wait_count_u16) == 10, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, signal_semaphore_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, signal_payload_base_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, signal_count_u16) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, usage_v8) == 18, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, access_v8) == 19, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, memory_type_v8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, memory_affinity_v8) == 21,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, min_alignment_v8) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, allocation_size_v8) == 23,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_alloca_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_dealloca_record_t) == 24, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_dealloca_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, wait_count_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, buffer_r8) == 16, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, zero_padding_u8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dealloca_record_t, flags_u32) == 20, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_fill_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_fill_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, signal_count_u16) == 14, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, target_buffer_r8) == 16, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, target_offset_v8) == 17, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, length_v8) == 18, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, pattern_v8) == 19, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, pattern_width_u8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, zero_padding_u8) == 21, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_fill_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_update_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_update_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, source_vm_buffer_r8) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, source_offset_v8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, target_buffer_r8) == 18,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, target_offset_v8) == 19,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, length_v8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, zero_padding_u8) == 21,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_update_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_copy_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_copy_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, signal_count_u16) == 14, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, source_buffer_r8) == 16, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, source_offset_v8) == 17, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, target_buffer_r8) == 18, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, target_offset_v8) == 19, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, length_v8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, zero_padding_u8) == 21, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_copy_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_read_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_read_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, signal_count_u16) == 14, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, source_file_r8) == 16, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, source_offset_v8) == 17, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, target_buffer_r8) == 18, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, target_offset_v8) == 19, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, length_v8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, zero_padding_u8) == 21, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_read_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_write_record_t) == 28, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_write_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, source_buffer_r8) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, source_offset_v8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, target_file_r8) == 18, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, target_offset_v8) == 19,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, length_v8) == 20, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, zero_padding_u8) == 21, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_write_record_t, flags_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_dispatch_record_t) == 36, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_dispatch_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, wait_count_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, function_table_r8) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, function_ordinal_v8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, launch_base_u16) == 18,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, constant_base_u16) == 20,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, constant_count_u16) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, binding_buffer_base_u16) == 24,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, binding_offset_base_u16) == 26,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, binding_length_base_u16) == 28,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, binding_count_u16) == 30,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_record_t, flags_u32) == 32, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t) == 40, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, page_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, opcode_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, device_r8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, affinity_v8) == 3,
              "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, wait_semaphore_base_u16) == 4,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, wait_payload_base_u16) == 6,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, wait_count_u16) == 8,
              "wire offset");
static_assert(
    offsetof(
        iree_vm_isa_hal_queue_dispatch_indirect_count_record_t,
        signal_semaphore_base_u16) == 10,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, signal_payload_base_u16) == 12,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, signal_count_u16) == 14,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, function_table_r8) == 16,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, function_ordinal_v8) == 17,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, launch_base_u16) == 18,
    "wire offset");
static_assert(
    offsetof(
        iree_vm_isa_hal_queue_dispatch_indirect_count_record_t,
        workgroup_count_buffer_r8) == 20,
    "wire offset");
static_assert(
    offsetof(
        iree_vm_isa_hal_queue_dispatch_indirect_count_record_t,
        workgroup_count_offset_v8) == 21,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, constant_base_u16) == 22,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, constant_count_u16) == 24,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, binding_buffer_base_u16) == 26,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, binding_offset_base_u16) == 28,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, binding_length_base_u16) == 30,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, binding_count_u16) == 32,
    "wire offset");
static_assert(
    offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, zero_padding_u16) == 34,
    "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_dispatch_indirect_count_record_t, flags_u32) == 36,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_execute_record_t) == 32, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_execute_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, command_buffer_r8) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, zero_padding_u8) == 17,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, binding_buffer_base_u16) == 18,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, binding_offset_base_u16) == 20,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, binding_length_base_u16) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, binding_count_u16) == 24,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, zero_padding_u16) == 26,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_execute_record_t, flags_u32) == 28, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_barrier_record_t) == 20, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_queue_barrier_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, affinity_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, wait_semaphore_base_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, wait_payload_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, wait_count_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, signal_semaphore_base_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, signal_payload_base_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, signal_count_u16) == 14,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_barrier_record_t, flags_u32) == 16, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_queue_flush_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_queue_flush_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_queue_flush_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_flush_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_flush_record_t, device_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_queue_flush_record_t, affinity_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_semaphore_create_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_hal_semaphore_create_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, dst_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, device_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, affinity_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, initial_value_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, zero_padding_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_create_record_t, flags_u32) == 8, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_semaphore_query_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_semaphore_query_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_semaphore_query_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_query_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_query_record_t, dst_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_query_record_t, semaphore_r8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_hal_semaphore_signal_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_hal_semaphore_signal_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, group_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, semaphore_r8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, payload_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_signal_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_hal_semaphore_await_record_t) == 12, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_hal_semaphore_await_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, page_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, opcode_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, dst_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, mode_u8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, timeout_kind_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, timeout_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, semaphore_base_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, payload_base_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_isa_hal_semaphore_await_record_t, count_u16) == 10, "wire offset");

// clang-format on
