// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_TEST_TYPES_H_
#define IREE_VM_BYTECODE_MODULE_TEST_TYPES_H_

#include "iree/vm/ref.h"

namespace iree::vm::bytecode::testing {

// Ref-counted test object deliberately incompatible with vm.buffer.
struct RefStateTestObject {
  // Required offset-zero VM-visible ownership prefix.
  iree_vm_ref_object_t ref_object;
  // Incremented when the final owner is released.
  int* destruction_count;
};

// Test object descriptor used by ref-state execution and verification tests.
extern const iree_vm_ref_type_descriptor_t kRefStateTestObjectType;

// Test type family containing only |kRefStateTestObjectType|.
extern const iree_vm_ref_type_table_t kRefStateTestTypeTable;

// Increments the integer passed as |user_data| when storage is released.
void CountBufferRelease(void* user_data, iree_byte_span_t storage);

}  // namespace iree::vm::bytecode::testing

#endif  // IREE_VM_BYTECODE_MODULE_TEST_TYPES_H_
