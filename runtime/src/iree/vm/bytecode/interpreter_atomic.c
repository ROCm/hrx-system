// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_atomic.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/bytecode/interpreter_float.h"

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && defined(IREE_COMPILER_MSVC_COMPAT)
#include <intrin.h>
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && IREE_COMPILER_MSVC_COMPAT

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && \
    !defined(IREE_COMPILER_MSVC_COMPAT) &&  \
    (defined(IREE_COMPILER_CLANG) || defined(IREE_COMPILER_GCC))
typedef uint32_t iree_vm_bytecode_atomic_alias_u32_t
    __attribute__((__may_alias__));
typedef uint64_t iree_vm_bytecode_atomic_alias_u64_t
    __attribute__((__may_alias__));
#endif  // GNU-compatible raw atomics

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_COMPILER_MSVC_COMPAT)
static int iree_vm_bytecode_atomic_builtin_ordering(
    iree_vm_bytecode_buffer_atomic_ordering_t ordering) {
  switch (ordering) {
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED:
      return __ATOMIC_RELAXED;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE:
      return __ATOMIC_ACQUIRE;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELEASE:
      return __ATOMIC_RELEASE;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQ_REL:
      return __ATOMIC_ACQ_REL;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST:
      return __ATOMIC_SEQ_CST;
    default:
      IREE_ASSERT_UNREACHABLE("atomic ordering must be verified");
      return __ATOMIC_SEQ_CST;
  }
}
#endif  // synchronized non-MSVC-compatible compiler

static uint32_t iree_vm_bytecode_atomic_load_u32(uint8_t* address) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  uint32_t value = 0;
  memcpy(&value, address, sizeof(value));
  return value;
#elif defined(IREE_COMPILER_MSVC_COMPAT)
  return (uint32_t)_InterlockedCompareExchange((volatile long*)address, 0, 0);
#else
  return __atomic_load_n((iree_vm_bytecode_atomic_alias_u32_t*)address,
                         __ATOMIC_RELAXED);
#endif  // raw atomic implementation
}

static bool iree_vm_bytecode_atomic_compare_exchange_u32(
    uint8_t* address, uint32_t* expected_bits, uint32_t replacement_bits,
    iree_vm_bytecode_buffer_atomic_ordering_t success_ordering,
    iree_vm_bytecode_buffer_atomic_ordering_t failure_ordering) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  uint32_t observed_bits = 0;
  memcpy(&observed_bits, address, sizeof(observed_bits));
  if (observed_bits == *expected_bits) {
    memcpy(address, &replacement_bits, sizeof(replacement_bits));
    return true;
  }
  *expected_bits = observed_bits;
  return false;
#elif defined(IREE_COMPILER_MSVC_COMPAT)
  (void)success_ordering;
  (void)failure_ordering;
  const uint32_t observed_bits = (uint32_t)_InterlockedCompareExchange(
      (volatile long*)address, (long)replacement_bits, (long)*expected_bits);
  if (observed_bits == *expected_bits) return true;
  *expected_bits = observed_bits;
  return false;
#else
  return __atomic_compare_exchange_n(
      (iree_vm_bytecode_atomic_alias_u32_t*)address, expected_bits,
      replacement_bits, /*weak=*/false,
      iree_vm_bytecode_atomic_builtin_ordering(success_ordering),
      iree_vm_bytecode_atomic_builtin_ordering(failure_ordering));
#endif  // raw atomic implementation
}

static uint64_t iree_vm_bytecode_atomic_load_u64(uint8_t* address) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  uint64_t value = 0;
  memcpy(&value, address, sizeof(value));
  return value;
#elif defined(IREE_COMPILER_MSVC_COMPAT)
  return (uint64_t)_InterlockedCompareExchange64((volatile __int64*)address, 0,
                                                 0);
#else
  return __atomic_load_n((iree_vm_bytecode_atomic_alias_u64_t*)address,
                         __ATOMIC_RELAXED);
#endif  // raw atomic implementation
}

static bool iree_vm_bytecode_atomic_compare_exchange_u64(
    uint8_t* address, uint64_t* expected_bits, uint64_t replacement_bits,
    iree_vm_bytecode_buffer_atomic_ordering_t success_ordering,
    iree_vm_bytecode_buffer_atomic_ordering_t failure_ordering) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  uint64_t observed_bits = 0;
  memcpy(&observed_bits, address, sizeof(observed_bits));
  if (observed_bits == *expected_bits) {
    memcpy(address, &replacement_bits, sizeof(replacement_bits));
    return true;
  }
  *expected_bits = observed_bits;
  return false;
#elif defined(IREE_COMPILER_MSVC_COMPAT)
  (void)success_ordering;
  (void)failure_ordering;
  const uint64_t observed_bits = (uint64_t)_InterlockedCompareExchange64(
      (volatile __int64*)address, (__int64)replacement_bits,
      (__int64)*expected_bits);
  if (observed_bits == *expected_bits) return true;
  *expected_bits = observed_bits;
  return false;
#else
  return __atomic_compare_exchange_n(
      (iree_vm_bytecode_atomic_alias_u64_t*)address, expected_bits,
      replacement_bits, /*weak=*/false,
      iree_vm_bytecode_atomic_builtin_ordering(success_ordering),
      iree_vm_bytecode_atomic_builtin_ordering(failure_ordering));
#endif  // raw atomic implementation
}

static uint32_t iree_vm_bytecode_atomic_apply_kind_u32(
    uint32_t old_bits, uint32_t operand_bits,
    iree_vm_bytecode_buffer_atomic_kind_t kind) {
  switch (kind) {
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT:
      return operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER:
      return old_bits + operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_FLOAT:
      return iree_vm_bytecode_float_f32_to_bits(
          iree_vm_bytecode_float_f32_from_bits(old_bits) +
          iree_vm_bytecode_float_f32_from_bits(operand_bits));
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_SUBTRACT_INTEGER:
      return old_bits - operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_AND_INTEGER:
      return old_bits & operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_OR_INTEGER:
      return old_bits | operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_XOR_INTEGER:
      return old_bits ^ operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_SIGNED:
      return (old_bits ^ UINT32_C(0x80000000)) <
                     (operand_bits ^ UINT32_C(0x80000000))
                 ? old_bits
                 : operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_SIGNED:
      return (old_bits ^ UINT32_C(0x80000000)) >
                     (operand_bits ^ UINT32_C(0x80000000))
                 ? old_bits
                 : operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_UNSIGNED:
      return iree_min(old_bits, operand_bits);
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_UNSIGNED:
      return iree_max(old_bits, operand_bits);
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINNUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT:
      return iree_vm_bytecode_float_minmax_f32_bits(
          old_bits, operand_bits,
          kind - IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT);
    default:
      IREE_ASSERT_UNREACHABLE("atomic kind must be verified");
      return old_bits;
  }
}

static uint64_t iree_vm_bytecode_atomic_apply_kind_u64(
    uint64_t old_bits, uint64_t operand_bits,
    iree_vm_bytecode_buffer_atomic_kind_t kind) {
  switch (kind) {
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT:
      return operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER:
      return old_bits + operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_FLOAT:
      return iree_vm_bytecode_float_f64_to_bits(
          iree_vm_bytecode_float_f64_from_bits(old_bits) +
          iree_vm_bytecode_float_f64_from_bits(operand_bits));
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_SUBTRACT_INTEGER:
      return old_bits - operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_AND_INTEGER:
      return old_bits & operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_OR_INTEGER:
      return old_bits | operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_XOR_INTEGER:
      return old_bits ^ operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_SIGNED:
      return (old_bits ^ UINT64_C(0x8000000000000000)) <
                     (operand_bits ^ UINT64_C(0x8000000000000000))
                 ? old_bits
                 : operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_SIGNED:
      return (old_bits ^ UINT64_C(0x8000000000000000)) >
                     (operand_bits ^ UINT64_C(0x8000000000000000))
                 ? old_bits
                 : operand_bits;
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_UNSIGNED:
      return iree_min(old_bits, operand_bits);
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_UNSIGNED:
      return iree_max(old_bits, operand_bits);
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINNUM_FLOAT:
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT:
      return iree_vm_bytecode_float_minmax_f64_bits(
          old_bits, operand_bits,
          kind - IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT);
    default:
      IREE_ASSERT_UNREACHABLE("atomic kind must be verified");
      return old_bits;
  }
}

uint64_t iree_vm_bytecode_atomic_apply(
    uint8_t* address, uint64_t operand_bits,
    iree_vm_bytecode_buffer_atomic_kind_t kind,
    iree_vm_bytecode_buffer_atomic_carrier_t carrier,
    iree_vm_bytecode_buffer_atomic_ordering_t ordering) {
  switch (carrier) {
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32: {
      uint32_t expected_bits = iree_vm_bytecode_atomic_load_u32(address);
      while (!iree_vm_bytecode_atomic_compare_exchange_u32(
          address, &expected_bits,
          iree_vm_bytecode_atomic_apply_kind_u32(expected_bits,
                                                 (uint32_t)operand_bits, kind),
          ordering, IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED)) {
      }
      return expected_bits;
    }
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64: {
      uint64_t expected_bits = iree_vm_bytecode_atomic_load_u64(address);
      while (!iree_vm_bytecode_atomic_compare_exchange_u64(
          address, &expected_bits,
          iree_vm_bytecode_atomic_apply_kind_u64(expected_bits, operand_bits,
                                                 kind),
          ordering, IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED)) {
      }
      return expected_bits;
    }
    default:
      IREE_ASSERT_UNREACHABLE("atomic carrier must be verified");
      return 0;
  }
}

uint64_t iree_vm_bytecode_atomic_compare_exchange(
    uint8_t* address, uint64_t expected_bits, uint64_t replacement_bits,
    iree_vm_bytecode_buffer_atomic_carrier_t carrier,
    iree_vm_bytecode_buffer_atomic_ordering_t success_ordering,
    iree_vm_bytecode_buffer_atomic_ordering_t failure_ordering) {
  switch (carrier) {
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32: {
      uint32_t observed_bits = (uint32_t)expected_bits;
      iree_vm_bytecode_atomic_compare_exchange_u32(
          address, &observed_bits, (uint32_t)replacement_bits, success_ordering,
          failure_ordering);
      return observed_bits;
    }
    case IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64: {
      uint64_t observed_bits = expected_bits;
      iree_vm_bytecode_atomic_compare_exchange_u64(
          address, &observed_bits, replacement_bits, success_ordering,
          failure_ordering);
      return observed_bits;
    }
    default:
      IREE_ASSERT_UNREACHABLE("atomic carrier must be verified");
      return 0;
  }
}
