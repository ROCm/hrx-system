// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core integer instruction handlers included inside the interpreter dispatch
// function. Keeping the labels in that function preserves direct-threaded
// dispatch and makes the shared register banks explicit at each handler.

  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ADD_I32, integer_add_i32) {
    const iree_vm_isa_integer_add_i32_record_t* record =
        (const iree_vm_isa_integer_add_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs + rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_add_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ADD_I64, integer_add_i64) {
    const iree_vm_isa_integer_add_i64_record_t* record =
        (const iree_vm_isa_integer_add_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs + rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_add_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_SUB_I32, integer_sub_i32) {
    const iree_vm_isa_integer_sub_i32_record_t* record =
        (const iree_vm_isa_integer_sub_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs - rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_sub_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_SUB_I64, integer_sub_i64) {
    const iree_vm_isa_integer_sub_i64_record_t* record =
        (const iree_vm_isa_integer_sub_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs - rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_sub_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MUL_I32, integer_mul_i32) {
    const iree_vm_isa_integer_mul_i32_record_t* record =
        (const iree_vm_isa_integer_mul_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs * rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_mul_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MUL_I64, integer_mul_i64) {
    const iree_vm_isa_integer_mul_i64_record_t* record =
        (const iree_vm_isa_integer_mul_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs * rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_mul_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_DIV_S32, integer_div_s32) {
    const iree_vm_isa_integer_div_s32_record_t* record =
        (const iree_vm_isa_integer_div_s32_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_div_s32(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_div_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_DIV_S64, integer_div_s64) {
    const iree_vm_isa_integer_div_s64_record_t* record =
        (const iree_vm_isa_integer_div_s64_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_div_s64(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_div_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_DIV_U32, integer_div_u32) {
    const iree_vm_isa_integer_div_u32_record_t* record =
        (const iree_vm_isa_integer_div_u32_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_div_u32(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_div_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_DIV_U64, integer_div_u64) {
    const iree_vm_isa_integer_div_u64_record_t* record =
        (const iree_vm_isa_integer_div_u64_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_div_u64(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_div_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_REM_S32, integer_rem_s32) {
    const iree_vm_isa_integer_rem_s32_record_t* record =
        (const iree_vm_isa_integer_rem_s32_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_rem_s32(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_rem_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_REM_S64, integer_rem_s64) {
    const iree_vm_isa_integer_rem_s64_record_t* record =
        (const iree_vm_isa_integer_rem_s64_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_rem_s64(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_rem_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_REM_U32, integer_rem_u32) {
    const iree_vm_isa_integer_rem_u32_record_t* record =
        (const iree_vm_isa_integer_rem_u32_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_rem_u32(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_rem_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_REM_U64, integer_rem_u64) {
    const iree_vm_isa_integer_rem_u64_record_t* record =
        (const iree_vm_isa_integer_rem_u64_record_t*)record_data;
    const iree_vm_bytecode_integer_division_failure_t failure =
        iree_vm_bytecode_execute_integer_rem_u64(record, values);
    if (failure != IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE) {
      status = iree_vm_bytecode_make_integer_division_status(failure);
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_rem_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_NEG_I32, integer_neg_i32) {
    const iree_vm_isa_integer_neg_i32_record_t* record =
        (const iree_vm_isa_integer_neg_i32_record_t*)record_data;
    const uint32_t source = (uint32_t)values[record->src_v8];
    values[record->dst_v8] = UINT32_C(0) - source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_neg_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_NEG_I64, integer_neg_i64) {
    const iree_vm_isa_integer_neg_i64_record_t* record =
        (const iree_vm_isa_integer_neg_i64_record_t*)record_data;
    const uint64_t source = values[record->src_v8];
    values[record->dst_v8] = UINT64_C(0) - source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_neg_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ABS_S32, integer_abs_s32) {
    const iree_vm_isa_integer_abs_s32_record_t* record =
        (const iree_vm_isa_integer_abs_s32_record_t*)record_data;
    const uint32_t source = (uint32_t)values[record->src_v8];
    values[record->dst_v8] =
        source & UINT32_C(0x80000000) ? UINT32_C(0) - source : source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_abs_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ABS_S64, integer_abs_s64) {
    const iree_vm_isa_integer_abs_s64_record_t* record =
        (const iree_vm_isa_integer_abs_s64_record_t*)record_data;
    const uint64_t source = values[record->src_v8];
    values[record->dst_v8] =
        source & UINT64_C(0x8000000000000000) ? UINT64_C(0) - source : source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_abs_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_S32, integer_min_s32) {
    const iree_vm_isa_integer_min_s32_record_t* record =
        (const iree_vm_isa_integer_min_s32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] =
        (lhs ^ UINT32_C(0x80000000)) < (rhs ^ UINT32_C(0x80000000)) ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_S64, integer_min_s64) {
    const iree_vm_isa_integer_min_s64_record_t* record =
        (const iree_vm_isa_integer_min_s64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = (lhs ^ UINT64_C(0x8000000000000000)) <
                                     (rhs ^ UINT64_C(0x8000000000000000))
                                 ? lhs
                                 : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_U32, integer_min_u32) {
    const iree_vm_isa_integer_min_u32_record_t* record =
        (const iree_vm_isa_integer_min_u32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs < rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_U64, integer_min_u64) {
    const iree_vm_isa_integer_min_u64_record_t* record =
        (const iree_vm_isa_integer_min_u64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs < rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_S32, integer_max_s32) {
    const iree_vm_isa_integer_max_s32_record_t* record =
        (const iree_vm_isa_integer_max_s32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] =
        (lhs ^ UINT32_C(0x80000000)) > (rhs ^ UINT32_C(0x80000000)) ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_S64, integer_max_s64) {
    const iree_vm_isa_integer_max_s64_record_t* record =
        (const iree_vm_isa_integer_max_s64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = (lhs ^ UINT64_C(0x8000000000000000)) >
                                     (rhs ^ UINT64_C(0x8000000000000000))
                                 ? lhs
                                 : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_U32, integer_max_u32) {
    const iree_vm_isa_integer_max_u32_record_t* record =
        (const iree_vm_isa_integer_max_u32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs > rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_U64, integer_max_u64) {
    const iree_vm_isa_integer_max_u64_record_t* record =
        (const iree_vm_isa_integer_max_u64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs > rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_AND_I32, integer_and_i32) {
    const iree_vm_isa_integer_and_i32_record_t* record =
        (const iree_vm_isa_integer_and_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs & rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_and_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_AND_I64, integer_and_i64) {
    const iree_vm_isa_integer_and_i64_record_t* record =
        (const iree_vm_isa_integer_and_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs & rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_and_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_OR_I32, integer_or_i32) {
    const iree_vm_isa_integer_or_i32_record_t* record =
        (const iree_vm_isa_integer_or_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs | rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_or_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_OR_I64, integer_or_i64) {
    const iree_vm_isa_integer_or_i64_record_t* record =
        (const iree_vm_isa_integer_or_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs | rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_or_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_XOR_I32, integer_xor_i32) {
    const iree_vm_isa_integer_xor_i32_record_t* record =
        (const iree_vm_isa_integer_xor_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs ^ rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_xor_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_XOR_I64, integer_xor_i64) {
    const iree_vm_isa_integer_xor_i64_record_t* record =
        (const iree_vm_isa_integer_xor_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs ^ rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_xor_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_COMPARE_I32, integer_compare_i32) {
    const iree_vm_isa_integer_compare_i32_record_t* record =
        (const iree_vm_isa_integer_compare_i32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_compare_i32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_compare_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_COMPARE_I64, integer_compare_i64) {
    const iree_vm_isa_integer_compare_i64_record_t* record =
        (const iree_vm_isa_integer_compare_i64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_compare_i64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_compare_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_LEA_I32, integer_lea_i32) {
    const iree_vm_isa_integer_lea_i32_record_t* record =
        (const iree_vm_isa_integer_lea_i32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_lea_i32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_lea_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_LEA_I64, integer_lea_i64) {
    const iree_vm_isa_integer_lea_i64_record_t* record =
        (const iree_vm_isa_integer_lea_i64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_lea_i64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_lea_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_CEILDIV_POW2_U32,
                                 integer_ceildiv_pow2_u32) {
    const iree_vm_isa_integer_ceildiv_pow2_u32_record_t* record =
        (const iree_vm_isa_integer_ceildiv_pow2_u32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_integer_ceildiv_pow2_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_CEILDIV_POW2_U64,
                                 integer_ceildiv_pow2_u64) {
    const iree_vm_isa_integer_ceildiv_pow2_u64_record_t* record =
        (const iree_vm_isa_integer_ceildiv_pow2_u64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_integer_ceildiv_pow2_u64_record_t);
  }
