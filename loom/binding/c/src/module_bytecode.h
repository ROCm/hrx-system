// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_BYTECODE_H_
#define LOOMC_MODULE_BYTECODE_H_

#include "loom/ir/module.h"
#include "loomc/module.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional module-symbol projection produced during bytecode serialization.
typedef struct loomc_module_symbol_projection_t {
  // Module-local symbol IDs to project.
  const loom_symbol_id_t* module_symbol_ids;

  // Bytecode SYMBOLS ordinals corresponding to |module_symbol_ids|.
  loom_symbol_id_t* bytecode_symbol_ordinals;

  // Number of entries in both arrays.
  iree_host_size_t count;
} loomc_module_symbol_projection_t;

// Serializes |internal_module| with the representation providers registered in
// |context| into an in-memory Loom bytecode source. When supplied, |projection|
// is populated by the same writer traversal that serializes the module.
LOOMC_API_PRIVATE loomc_status_t
loomc_module_serialize_internal_bytecode_to_source(
    const loomc_context_t* context, const loom_module_t* internal_module,
    loomc_string_view_t identifier,
    const loomc_module_symbol_projection_t* projection,
    loomc_allocator_t allocator, loomc_source_t** out_source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_BYTECODE_H_
