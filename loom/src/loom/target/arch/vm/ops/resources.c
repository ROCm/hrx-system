// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/vm/ops/ops.h"

static iree_status_t loom_vm_resource_verify_ordinal(
    const loom_module_t* module, const loom_op_t* op,
    uint16_t ordinal_attr_index, int64_t ordinal,
    iree_diagnostic_emitter_t emitter) {
  if (ordinal >= 0 && ordinal <= UINT16_MAX) return iree_ok_status();

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  IREE_ASSERT(vtable != NULL && vtable->attr_descriptors != NULL);
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_attr_descriptor_name(
              &vtable->attr_descriptors[ordinal_attr_index])),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    ordinal_attr_index)),
      loom_param_i64(ordinal),
      loom_param_string(IREE_SV("an unsigned 16-bit integer")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_vm_global_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_vm_resource_verify_ordinal(module, op,
                                         loom_vm_global_ordinal_ATTR_INDEX,
                                         loom_vm_global_ordinal(op), emitter);
}

iree_status_t loom_vm_rodata_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_vm_resource_verify_ordinal(module, op,
                                         loom_vm_rodata_ordinal_ATTR_INDEX,
                                         loom_vm_rodata_ordinal(op), emitter);
}
