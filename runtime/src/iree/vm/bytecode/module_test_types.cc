// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module_test_types.h"

namespace iree::vm::bytecode::testing {
namespace {

struct RefStateTestTypes {
  // Type deliberately incompatible with vm.buffer.
  iree_vm_ref_type_t object;
};

void DestroyRefStateTestObject(void* object) {
  auto* test_object = static_cast<RefStateTestObject*>(object);
  ++*test_object->destruction_count;
}

}  // namespace

const iree_vm_ref_type_descriptor_t kRefStateTestObjectType = {
    DestroyRefStateTestObject,
    &kRefStateTestTypeTable,
    IREE_SV("object"),
};
const RefStateTestTypes kRefStateTestTypes = {
    &kRefStateTestObjectType,
};
const iree_vm_ref_type_table_t kRefStateTestTypeTable = {
    sizeof(kRefStateTestTypeTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("zz_test"),
    {&kRefStateTestTypes, 1},
};

void CountBufferRelease(void* user_data, iree_byte_span_t storage) {
  (void)storage;
  ++*static_cast<int*>(user_data);
}

}  // namespace iree::vm::bytecode::testing
