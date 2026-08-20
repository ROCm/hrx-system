// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Compile-time checks for CORE ISA wire declarations.
// clang-format off

#include "iree/base/alignment.h"
#include "iree/vm/bytecode/wire/core/abi.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "iree/vm/bytecode/wire/core/value.h"

static_assert(sizeof(iree_vm_isa_value_abi_argument_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_value_abi_argument_load_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_value_abi_argument_load_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_value_abi_argument_load_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_value_abi_argument_load_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_value_abi_result_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_value_abi_result_store_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_value_abi_result_store_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_value_abi_result_store_record_t, src_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_value_abi_result_store_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_abi_argument_load_borrow_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_abi_argument_load_borrow_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_borrow_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_borrow_record_t, dst_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_borrow_record_t, slot_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_ref_abi_argument_load_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_abi_argument_load_move_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_move_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_move_record_t, dst_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_argument_load_move_record_t, slot_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_ref_abi_result_store_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_abi_result_store_move_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_abi_result_store_move_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_result_store_move_record_t, src_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_abi_result_store_move_record_t, slot_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_abi_argument_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_abi_argument_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_abi_argument_load_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_abi_argument_load_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_abi_argument_load_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_func_abi_result_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_abi_result_store_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_abi_result_store_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_abi_result_store_record_t, src_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_abi_result_store_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_allocate_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_allocate_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_allocate_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_allocate_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_allocate_record_t, length_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_allocate_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_length_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_length_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_length_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_length_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_length_record_t, buffer_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_length_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_subspan_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_subspan_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, buffer_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, offset_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, length_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_subspan_record_t, zero_padding_u8) == 5, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_load_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, buffer_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, base_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, index_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, scale_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, format_u8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_load_record_t, zero_padding_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_store_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_store_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, buffer_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, base_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, index_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, scale_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, src_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, format_u8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_store_record_t, zero_padding_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_atomic_reduce_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_buffer_atomic_reduce_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, buffer_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, offset_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, operand_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, selector0_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, selector1_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_reduce_record_t, zero_padding_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_atomic_rmw_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_atomic_rmw_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, old_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, buffer_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, offset_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, operand_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, selector0_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, selector1_u8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_rmw_record_t, zero_padding_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_atomic_cmpxchg_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_atomic_cmpxchg_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, old_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, buffer_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, offset_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, expected_v8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, replacement_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, selector0_u8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_atomic_cmpxchg_record_t, selector1_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_fill_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_buffer_fill_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, buffer_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, offset_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, length_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, pattern_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, pattern_width_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_fill_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_copy_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_buffer_copy_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, target_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, target_offset_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, source_r8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, source_offset_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, length_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_compare_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_buffer_compare_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, lhs_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, lhs_offset_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, rhs_r8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, rhs_offset_v8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, length_v8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_compare_record_t, zero_padding_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_copy_rodata_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_buffer_copy_rodata_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, target_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, target_offset_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, zero_padding0_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, rodata_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, length_v8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, zero_padding1_u8) == 7,
              "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_copy_rodata_record_t, source_offset_u32) == 8,
              "wire offset");

static_assert(sizeof(iree_vm_isa_buffer_rodata_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_buffer_rodata_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_buffer_rodata_load_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_rodata_load_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_buffer_rodata_load_record_t, rodata_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_zero_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_constant_zero_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_zero_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_zero_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_zero_record_t, zero_padding_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_s16_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_constant_s16_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_s16_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_s16_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_s16_record_t, immediate_i16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_i32_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_constant_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i32_record_t, zero_padding_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i32_record_t, bits_u32le) == 4, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_i64_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_constant_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i64_record_t, zero_padding_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i64_record_t, bits_low_u32le) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_i64_record_t, bits_high_u32le) == 8, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_pool_load_i32_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_constant_pool_load_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i32_record_t, pool_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_constant_pool_load_i64_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_constant_pool_load_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_constant_pool_load_i64_record_t, pool_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_control_block_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_control_block_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_block_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_block_record_t, zero_padding_u8) == 1, "wire offset");

static_assert(sizeof(iree_vm_isa_control_return_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_control_return_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_return_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_return_record_t, zero_padding_u8) == 1, "wire offset");

static_assert(sizeof(iree_vm_isa_control_yield_s32_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_control_yield_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_yield_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_yield_s32_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_yield_s32_record_t, target_rel32) == 4, "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_s16_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_control_branch_s16_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_s16_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_s16_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_s16_record_t, target_rel16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_s32_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_control_branch_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_s32_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_s32_record_t, target_rel32) == 4, "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_if_s16_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_control_branch_if_s16_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_if_s16_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_if_s16_record_t, condition_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_if_s16_record_t, target_rel16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_if_s32_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_control_branch_if_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_if_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_if_s32_record_t, condition_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_if_s32_record_t, zero_padding_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_if_s32_record_t, target_rel32) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_unless_s16_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_control_branch_unless_s16_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s16_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s16_record_t, condition_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s16_record_t, target_rel16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_control_branch_unless_s32_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_control_branch_unless_s32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s32_record_t, condition_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s32_record_t, zero_padding_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_branch_unless_s32_record_t, target_rel32) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_control_switch_record_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_control_switch_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_switch_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_switch_record_t, selector_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_control_switch_record_t, target_count_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_control_switch_record_t, target_base_u32) == 4, "wire offset");

static_assert(sizeof(iree_vm_isa_control_call_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_control_call_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_call_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_record_t, target_kind_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_record_t, target_ordinal_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_record_t, direct_ref_move_mask_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_control_call_indirect_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_control_call_indirect_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_call_indirect_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_indirect_record_t, target_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_indirect_record_t, callable_type_ordinal_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_indirect_record_t, direct_ref_move_mask_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_call_indirect_record_t, zero_padding_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_control_assert_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_control_assert_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_assert_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_assert_record_t, condition_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_control_assert_record_t, message_r8_nullable) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_control_assert_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_control_fail_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_control_fail_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_control_fail_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_control_fail_record_t, status_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_control_fail_record_t, message_r8_nullable) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_control_fail_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_integer_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_integer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_integer_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_record_t, selector_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_float_extend_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_float_extend_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_float_extend_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_extend_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_extend_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_extend_record_t, selector_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_float_truncate_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_float_truncate_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_float_truncate_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_truncate_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_truncate_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_truncate_record_t, selector_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_float_width_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_float_width_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_float_width_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_width_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_width_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_width_record_t, selector_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_integer_to_float_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_integer_to_float_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_integer_to_float_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_to_float_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_to_float_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_integer_to_float_record_t, selector_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_conversion_float_to_integer_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_conversion_float_to_integer_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_conversion_float_to_integer_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_to_integer_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_to_integer_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_conversion_float_to_integer_record_t, selector_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_add_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_add_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_add_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_add_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_add_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_add_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_add_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_sub_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_sub_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_sub_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_sub_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_sub_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_sub_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_sub_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_mul_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_mul_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_mul_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_mul_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_mul_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_mul_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_mul_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_div_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_div_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_div_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_div_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_div_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_div_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_div_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_rem_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_rem_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_rem_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_rem_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_rem_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_rem_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_rem_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_neg_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_neg_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_neg_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f32_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_neg_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_neg_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_neg_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_neg_f64_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_abs_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_abs_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_abs_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f32_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_abs_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_abs_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_abs_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_abs_f64_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_minmax_f32_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_minmax_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, selector_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f32_record_t, zero_padding_u8) == 5, "wire offset");

static_assert(sizeof(iree_vm_isa_float_minmax_f64_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_minmax_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, selector_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_minmax_f64_record_t, zero_padding_u8) == 5, "wire offset");

static_assert(sizeof(iree_vm_isa_float_compare_f32_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_compare_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, predicate_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f32_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_compare_f64_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_compare_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, predicate_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_compare_f64_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_classify_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_classify_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_classify_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f32_record_t, selector_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_classify_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_classify_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_classify_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_classify_f64_record_t, selector_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_clamp_f32_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_float_clamp_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, value_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, lower_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, upper_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, mode_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f32_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_float_clamp_f64_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_float_clamp_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, value_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, lower_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, upper_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, mode_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_float_clamp_f64_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_float_copysign_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_copysign_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_copysign_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_copysign_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_copysign_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_copysign_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_copysign_f64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_unary_f32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_math_unary_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_unary_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f32_record_t, selector_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_unary_f64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_math_unary_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_unary_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_unary_f64_record_t, selector_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_binary_f32_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_math_binary_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, selector_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f32_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_binary_f64_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_float_math_binary_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, selector_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_binary_f64_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_ternary_f32_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_float_math_ternary_f32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, a_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, b_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, c_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, selector_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f32_record_t, zero_padding_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_float_math_ternary_f64_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_float_math_ternary_f64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, a_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, b_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, c_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, selector_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_float_math_ternary_f64_record_t, zero_padding_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_null_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_null_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_null_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_null_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_null_record_t, zero_padding_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_func_compare_null_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_func_compare_null_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_compare_null_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_compare_null_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_compare_null_record_t, src_f8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_func_compare_null_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_copy_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_func_copy_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_copy_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_copy_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_copy_record_t, src_f8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_func_copy_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_func_address_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_address_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_address_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_address_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_address_record_t, target_kind_u8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_func_address_record_t, zero_padding_u8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_func_address_record_t, target_ordinal_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_func_address_record_t, callable_type_ordinal_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_import_resolved_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_import_resolved_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_import_resolved_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_import_resolved_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_import_resolved_record_t, import_ordinal_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_stack_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_stack_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_stack_load_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_stack_load_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_stack_load_record_t, local_ordinal_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_func_stack_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_func_stack_store_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_func_stack_store_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_func_stack_store_record_t, src_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_func_stack_store_record_t, local_ordinal_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_value_immutable_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_value_immutable_load_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_value_immutable_load_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_immutable_load_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_immutable_load_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_value_immutable_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_value_immutable_store_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_value_immutable_store_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_immutable_store_record_t, src_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_immutable_store_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_value_mutable_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_value_mutable_load_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_value_mutable_load_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_mutable_load_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_mutable_load_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_value_mutable_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_value_mutable_store_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_value_mutable_store_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_mutable_store_record_t, src_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_value_mutable_store_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_ref_immutable_load_borrow_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_ref_immutable_load_borrow_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_load_borrow_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_load_borrow_record_t, dst_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_load_borrow_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_ref_immutable_store_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_ref_immutable_store_move_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_store_move_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_store_move_record_t, src_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_immutable_store_move_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_ref_mutable_load_retain_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_ref_mutable_load_retain_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_load_retain_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_load_retain_record_t, dst_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_load_retain_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_ref_mutable_store_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_ref_mutable_store_move_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_store_move_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_store_move_record_t, src_r8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_ref_mutable_store_move_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_func_immutable_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_func_immutable_load_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_func_immutable_load_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_immutable_load_record_t, dst_f8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_immutable_load_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_func_immutable_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_func_immutable_store_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_func_immutable_store_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_immutable_store_record_t, src_f8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_immutable_store_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_func_mutable_load_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_func_mutable_load_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_func_mutable_load_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_mutable_load_record_t, dst_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_mutable_load_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_global_func_mutable_store_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_global_func_mutable_store_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_global_func_mutable_store_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_mutable_store_record_t, src_f8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_global_func_mutable_store_record_t, global_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_add_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_add_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_add_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_add_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_add_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_add_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_add_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_sub_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_sub_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_sub_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_sub_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_sub_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_sub_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_sub_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_mul_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_mul_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_mul_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_mul_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_mul_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_mul_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_mul_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_div_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_div_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_div_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_div_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_div_s64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_div_s64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_s64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_div_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_div_u32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_div_u32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_div_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_div_u64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_div_u64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_div_u64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rem_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rem_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rem_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rem_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rem_s64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rem_s64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_s64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rem_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rem_u32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rem_u32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rem_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rem_u64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rem_u64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rem_u64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_neg_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_neg_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_neg_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i32_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_neg_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_neg_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_neg_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_neg_i64_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_abs_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_abs_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_abs_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s32_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_abs_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_abs_s64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_abs_s64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_abs_s64_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_min_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_min_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_min_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_min_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_min_s64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_min_s64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_s64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_min_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_min_u32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_min_u32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_min_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_min_u64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_min_u64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_min_u64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_max_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_max_s32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_max_s32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_max_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_max_s64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_max_s64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_s64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_max_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_max_u32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_max_u32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_max_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_max_u64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_max_u64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_max_u64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_and_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_and_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_and_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_and_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_and_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_and_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_and_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_or_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_or_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_or_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_or_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_or_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_or_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_or_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_xor_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_xor_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_xor_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_xor_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_xor_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_xor_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_xor_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_left_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_left_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_left_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_left_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_left_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_right_s32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_right_s32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_right_s64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_right_s64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_s64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_right_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_right_u32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_shift_right_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_shift_right_u64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_shift_right_u64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rotate_left_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rotate_left_i32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rotate_left_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rotate_left_i64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_left_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rotate_right_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rotate_right_i32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i32_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_rotate_right_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_rotate_right_i64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_rotate_right_i64_record_t, rhs_v8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_count_leading_zeros_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_count_leading_zeros_i32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i32_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i32_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i32_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_count_leading_zeros_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_count_leading_zeros_i64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i64_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i64_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_leading_zeros_i64_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i32_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t, dst_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t, src_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_count_trailing_zeros_i64_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_popcount_i32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_popcount_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_popcount_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i32_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_popcount_i64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_popcount_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_popcount_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_popcount_i64_record_t, zero_padding_u8) == 3,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_compare_i32_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_compare_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, predicate_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i32_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_compare_i64_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_compare_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, lhs_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, rhs_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, predicate_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_compare_i64_record_t, zero_padding_u8) == 5,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_lea_i32_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_integer_lea_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, base_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, index_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, scale_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, zero_padding_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i32_record_t, offset_i16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_lea_i64_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_integer_lea_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, base_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, index_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, scale_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, zero_padding_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_lea_i64_record_t, offset_i16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u32_record_t, log2_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_ceildiv_pow2_u64_record_t, log2_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_integer_bitstream_pack_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_bitstream_pack_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, result_base_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, source_base_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, field_width_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, source_count_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, result_count_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, source_width_u8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_pack_record_t, result_width_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_bitstream_unpack_u_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_bitstream_unpack_u_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, result_base_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, source_base_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, field_width_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, source_count_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, result_count_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, source_width_u8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_u_record_t, result_width_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_integer_bitstream_unpack_s_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_integer_bitstream_unpack_s_record_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, opcode_u8) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, result_base_v8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, source_base_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, field_width_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, source_count_u8) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, result_count_u8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, source_width_u8) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_isa_integer_bitstream_unpack_s_record_t, result_width_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_ref_null_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_null_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_null_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_null_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_null_record_t, zero_padding_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_compare_null_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_ref_compare_null_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_compare_null_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_null_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_null_record_t, src_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_null_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_compare_eq_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_ref_compare_eq_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_compare_eq_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_eq_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_eq_record_t, lhs_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_compare_eq_record_t, rhs_r8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_retain_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_ref_retain_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_retain_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_retain_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_retain_record_t, src_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_retain_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_move_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_ref_move_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_move_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_move_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_move_record_t, src_r8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_move_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_discard_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_discard_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_discard_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_discard_record_t, src_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_discard_record_t, zero_padding_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_stack_load_retain_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_stack_load_retain_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_stack_load_retain_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_load_retain_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_load_retain_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_stack_load_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_stack_load_move_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_stack_load_move_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_load_move_record_t, dst_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_load_move_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_stack_store_retain_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_stack_store_retain_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_stack_store_retain_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_store_retain_record_t, src_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_store_retain_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_stack_store_move_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_stack_store_move_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_stack_store_move_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_store_move_record_t, src_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_store_move_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_ref_stack_discard_record_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_ref_stack_discard_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_ref_stack_discard_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_discard_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_ref_stack_discard_record_t, slot_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_load_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_load_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_load_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_record_t, base_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_record_t, format_u8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_record_t, zero_padding_u8) == 5, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_store_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_store_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, zero_padding_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, base_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, src_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, format_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_record_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_load_indexed_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_load_indexed_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, base_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, index_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, scale_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, format_u8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_load_indexed_record_t, zero_padding_u8) == 7,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_store_indexed_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_store_indexed_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, base_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, index_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, scale_u8) == 5, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, src_v8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_store_indexed_record_t, format_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_fill_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_fill_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, zero_padding_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, target_base_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, length_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, pattern_v8) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_fill_record_t, pattern_width_u8) == 7, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_copy_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_copy_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_copy_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_record_t, zero_padding_u8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_record_t, source_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_record_t, length_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_compare_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_compare_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_compare_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_compare_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_compare_record_t, lhs_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_compare_record_t, rhs_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_compare_record_t, length_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_copy_rodata_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_stack_copy_rodata_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, rodata_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, length_u16) == 6, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_rodata_record_t, source_offset_u32) == 8,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_copy_from_buffer_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_copy_from_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, target_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, buffer_r8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, source_offset_v8) == 5,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_from_buffer_record_t, length_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_copy_to_buffer_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_copy_to_buffer_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, buffer_r8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, target_offset_v8) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, zero_padding_u8) == 3,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, source_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_copy_to_buffer_record_t, length_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_isa_stack_const_s16_i32_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_const_s16_i32_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i32_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i32_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i32_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i32_record_t, count_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i32_record_t, immediate_i16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_const_s16_i64_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_const_s16_i64_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i64_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i64_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i64_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i64_record_t, count_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_const_s16_i64_record_t, immediate_i16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i32_u16_x2_record_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_pack_i32_u16_x2_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x2_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x2_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x2_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x2_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i32_u16_x4_record_t) == 12, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_pack_i32_u16_x4_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x4_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x4_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x4_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x4_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i32_u16_x8_record_t) == 20, "wire size");
static_assert(2 % iree_alignof(iree_vm_isa_stack_pack_i32_u16_x8_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x8_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x8_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x8_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i32_u16_x8_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i64_u32_x2_record_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_stack_pack_i64_u32_x2_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x2_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x2_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x2_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x2_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i64_u32_x4_record_t) == 20, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_stack_pack_i64_u32_x4_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x4_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x4_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x4_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x4_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_stack_pack_i64_u32_x8_record_t) == 36, "wire size");
static_assert(4 % iree_alignof(iree_vm_isa_stack_pack_i64_u32_x8_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x8_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x8_record_t, zero_padding_u8) == 1,
              "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x8_record_t, target_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_stack_pack_i64_u32_x8_record_t, immediates_le) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_isa_value_copy_record_t) == 4, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_value_copy_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_value_copy_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_value_copy_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_value_copy_record_t, src_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_value_copy_record_t, zero_padding_u8) == 3, "wire offset");

static_assert(sizeof(iree_vm_isa_value_select_record_t) == 8, "wire size");
static_assert(1 % iree_alignof(iree_vm_isa_value_select_record_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_isa_value_select_record_t, opcode_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_isa_value_select_record_t, dst_v8) == 1, "wire offset");
static_assert(offsetof(iree_vm_isa_value_select_record_t, condition_v8) == 2, "wire offset");
static_assert(offsetof(iree_vm_isa_value_select_record_t, true_v8) == 3, "wire offset");
static_assert(offsetof(iree_vm_isa_value_select_record_t, false_v8) == 4, "wire offset");
static_assert(offsetof(iree_vm_isa_value_select_record_t, zero_padding_u8) == 5, "wire offset");

// clang-format on
