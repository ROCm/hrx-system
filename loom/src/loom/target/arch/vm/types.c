// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/types.h"

#include "iree/vm/bytecode/wire/module.h"
#include "loom/ops/type_registry.h"

uint16_t loom_vm_type_scalar_kind(loom_type_t type) {
  static const uint16_t kScalarKinds[LOOM_SCALAR_TYPE_COUNT_] = {
      [LOOM_SCALAR_TYPE_I1] = IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
      [LOOM_SCALAR_TYPE_I8] = IREE_VM_BYTECODE_SIGNATURE_KIND_I8,
      [LOOM_SCALAR_TYPE_I16] = IREE_VM_BYTECODE_SIGNATURE_KIND_I16,
      [LOOM_SCALAR_TYPE_I32] = IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
      [LOOM_SCALAR_TYPE_I64] = IREE_VM_BYTECODE_SIGNATURE_KIND_I64,
      [LOOM_SCALAR_TYPE_F8E4M3] = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN,
      [LOOM_SCALAR_TYPE_F8E5M2] = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2,
      [LOOM_SCALAR_TYPE_F16] = IREE_VM_BYTECODE_SIGNATURE_KIND_F16,
      [LOOM_SCALAR_TYPE_BF16] = IREE_VM_BYTECODE_SIGNATURE_KIND_BF16,
      [LOOM_SCALAR_TYPE_F32] = IREE_VM_BYTECODE_SIGNATURE_KIND_F32,
      [LOOM_SCALAR_TYPE_F64] = IREE_VM_BYTECODE_SIGNATURE_KIND_F64,
  };
  return loom_type_is_scalar(type) ? kScalarKinds[loom_type_element_type(type)]
                                   : 0;
}

const loom_type_reference_key_t* loom_vm_type_reference_key(
    const loom_module_t* module, loom_type_t type) {
  if (loom_type_is_buffer(type)) {
    static const loom_type_reference_key_t kBuffer = {
        .namespace_name = IREE_SVL("vm"), .type_name = IREE_SVL("buffer")};
    return &kBuffer;
  }
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return NULL;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  const loom_type_descriptor_t* descriptor = loom_type_registry_lookup(
      module->context, loom_string_table_get(&module->strings, name_id));
  return descriptor ? descriptor->reference : NULL;
}
