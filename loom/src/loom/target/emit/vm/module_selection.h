// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Artifact-local Core VM module selection.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_SELECTION_H_
#define LOOM_TARGET_EMIT_VM_MODULE_SELECTION_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_vm_module_emission_selection_flags_t;

enum loom_vm_module_emission_selection_flag_bits_e {
  // The selected callable closure has no global or rodata accesses. Module
  // state records are omitted from the image while constant-pool records are
  // retained because selected instructions may reference them by ordinal.
  LOOM_VM_MODULE_EMISSION_SELECTION_FLAG_STATELESS = 1u << 0,
};

// Optional artifact-local symbol and state selection.
//
// Symbol liveness is computed from explicit artifact roots over the finalized
// module. Only live Core VM function definitions and declarations are emitted;
// their wire ordinals are reassigned within the resulting image.
typedef struct loom_vm_module_emission_selection_t {
  // Reachable module symbols for this artifact.
  const loom_symbol_liveness_t* symbol_liveness;

  // Optional byte-per-symbol export set. When present, only selected function
  // symbols with a non-zero entry are public in the emitted image.
  const uint8_t* export_symbols;

  // Optional compiler function-version handles indexed by module symbol. When
  // present, export projection uses these identities instead of identities on
  // the selected functions themselves.
  const loom_function_version_t* const* export_function_versions;

  // Additional artifact-local selection guarantees.
  loom_vm_module_emission_selection_flags_t flags;
} loom_vm_module_emission_selection_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_SELECTION_H_
