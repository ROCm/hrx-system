// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed implementation of the parser-facing low asm environment.

#ifndef LOOM_CODEGEN_LOW_TEXT_ASM_H_
#define LOOM_CODEGEN_LOW_TEXT_ASM_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/format/text/low_asm.h"
#include "loom/format/text/printer.h"
#include "loom/target/low_asm_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |out_environment| to resolve asm packets through
// |descriptor_registry| and build canonical low dialect operations. The
// descriptor registry is borrowed and must outlive every parse using the
// returned environment.
void loom_low_descriptor_text_asm_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* out_environment);

typedef struct loom_low_descriptor_text_asm_environment_storage_t {
  // Registry used for descriptor-set, packet, and register-type lookup.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Target-owned diagnostic providers used for unknown asm mnemonics.
  loom_target_low_asm_diagnostic_provider_list_t diagnostic_provider_list;
} loom_low_descriptor_text_asm_environment_storage_t;

// Initializes |out_environment| with target-owned diagnostic providers. The
// descriptor registry and |out_storage| are borrowed and must outlive every
// parse using the returned environment.
void loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_low_asm_diagnostic_provider_list_t diagnostic_provider_list,
    loom_low_descriptor_text_asm_environment_storage_t* out_storage,
    loom_text_low_asm_environment_t* out_environment);

// Descriptor-backed text printer context for target-low types.
typedef struct loom_low_descriptor_text_print_context_t {
  // Single descriptor-set backing storage used by the descriptor-set
  // initializer.
  const loom_low_descriptor_set_t* descriptor_sets[1];
  // Registry view used by |options.low_asm_environment|.
  loom_low_descriptor_registry_t descriptor_registry;
  // Text printer options configured for descriptor-backed register types.
  loom_text_print_options_t options;
} loom_low_descriptor_text_print_context_t;

// Initializes |out_context| to print target-low register types through
// |descriptor_registry|. The registry is borrowed and must outlive the context.
void loom_low_descriptor_text_print_context_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_descriptor_text_print_context_t* out_context);

// Initializes |out_context| to print target-low register types from one
// descriptor set.
void loom_low_descriptor_text_print_context_initialize_for_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_descriptor_text_print_context_t* out_context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TEXT_ASM_H_
