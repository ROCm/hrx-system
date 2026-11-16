// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SPAN_H_
#define IREE_VM_SPAN_H_

#include "iree/vm/ref.h"
#include "iree/vm/scalar.h"

// Maximum byte extent addressable within one borrowed VM storage region.
// The remaining address bits identify a region in the current function frame.
#define IREE_VM_SPAN_MAX_BYTE_LENGTH UINT64_C(0x0000FFFFFFFFFFFF)

// Reference coordinates use the VM's 128-bit reference representation. Native
// arrays still contain iree_vm_ref_t cells, whose size follows host bitness.
#define IREE_VM_SPAN_REF_STRIDE 16u

enum iree_vm_span_access_bits_e {
  IREE_VM_SPAN_ACCESS_NONE = 0u,
  IREE_VM_SPAN_ACCESS_READ = 1u << 0,
  IREE_VM_SPAN_ACCESS_WRITE = 1u << 1,
  IREE_VM_SPAN_ACCESS_READ_WRITE =
      IREE_VM_SPAN_ACCESS_READ | IREE_VM_SPAN_ACCESS_WRITE,
};
typedef uint32_t iree_vm_span_access_t;

// Typed storage borrowed by a synchronous invocation. The descriptor,
// array, and every object exposed as a ptr element must remain live until the
// invocation returns, even if another alias replaces an owning array cell.
// Element arrays are neither copied nor retained for transport. A scalar span
// has a scalar_type and a null ref_type; a reference span has scalar_type NONE
// and an exact ref_type. Ref cells retain their usual
// ownership: the host must reset any owning cells when retiring the array.
typedef struct iree_vm_typed_span_t {
  // Contiguous scalar bytes or naturally aligned iree_vm_ref_t cells. WRITE
  // permission promises mutable storage; native consumers honor that contract.
  const void* data;
  // Number of elements, including null cells in a reference array.
  uint64_t count;
  // Exact scalar element type, or NONE for reference cells.
  iree_vm_scalar_type_t scalar_type;
  // Shallow permissions on the array, independently of referenced objects.
  iree_vm_span_access_t access;
  // Exact reference element type, or null for scalar bytes.
  iree_vm_ref_type_t ref_type;
} iree_vm_typed_span_t;

#endif  // IREE_VM_SPAN_H_
