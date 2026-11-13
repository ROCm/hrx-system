// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/types.h"

#include "loom/ops/type_registry.h"

bool loom_vm_type_reference_key(const loom_module_t* module, loom_type_t type,
                                loom_type_reference_key_t* out_key) {
  if (loom_type_is_reference(type)) {
    *out_key = (loom_type_reference_key_t){
        .namespace_name = loom_string_table_get(
            &module->strings, loom_type_reference_namespace_id(type)),
        .type_name = loom_string_table_get(&module->strings,
                                           loom_type_reference_name_id(type)),
    };
    return true;
  }
  if (loom_type_is_buffer(type)) {
    *out_key = (loom_type_reference_key_t){.namespace_name = IREE_SVL("vm"),
                                           .type_name = IREE_SVL("buffer")};
    return true;
  }
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  const loom_type_descriptor_t* descriptor = loom_type_registry_lookup(
      module->context, loom_string_table_get(&module->strings, name_id));
  if (!descriptor || !descriptor->reference) {
    return false;
  }
  *out_key = *descriptor->reference;
  return true;
}
