// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/reference.h"

#include "loom/ops/type_registry.h"

loom_type_t loom_func_ref_resolve_signature(const loom_module_t* module,
                                            loom_type_t reference_type) {
  if (!loom_func_ref_type_isa(reference_type)) return loom_type_none();
  const loom_type_id_t signature_id =
      loom_func_ref_type_signature(reference_type);
  if (signature_id >= module->types.count) return loom_type_none();
  const loom_type_t signature_type = module->types.entries[signature_id];
  return loom_type_is_function(signature_type) ? signature_type
                                               : loom_type_none();
}
