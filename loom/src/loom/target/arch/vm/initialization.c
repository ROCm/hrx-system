// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/initialization.h"

#include "loom/target/arch/vm/lower/initialization.h"

static const loom_pass_info_t
    loom_vm_materialize_initializer_pass_info_storage = {
        .name = IREE_SVL("vm-materialize-initializer"),
        .description = IREE_SVL("Materialize the canonical VM initializer."),
        .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_vm_materialize_initializer_pass_info(void) {
  return &loom_vm_materialize_initializer_pass_info_storage;
}

iree_status_t loom_vm_materialize_initializer_run(loom_pass_t* pass,
                                                  loom_module_t* module) {
  loom_low_lower_prepare_module_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_vm_materialize_initializer(
      module, pass->diagnostic_emitter, pass->arena, &result));
  if (result.changed) loom_pass_mark_changed(pass);
  return iree_ok_status();
}
