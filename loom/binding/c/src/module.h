// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_STORAGE_H_
#define LOOMC_MODULE_STORAGE_H_

#include "config.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/function_version.h"
#include "loom/ir/module.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/target.h"
#include "loomc/workspace.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fully resolved source deserialization options.
typedef struct loomc_module_resolved_deserialize_options_t {
  // Input source format.
  loomc_source_format_t format;

  // Identifier used for diagnostics and module provenance.
  loomc_string_view_t identifier;
} loomc_module_resolved_deserialize_options_t;

// Fully resolved source serialization options.
typedef struct loomc_module_resolved_serialize_options_t {
  // Output source format.
  loomc_source_format_t format;

  // Identifier attached to returned source handles.
  loomc_string_view_t identifier;

  // Text presentation policy for textual output.
  loomc_module_text_presentation_t text_presentation;
} loomc_module_resolved_serialize_options_t;

// Decodes one resolved source format into internal module storage.
typedef loomc_status_t (*loomc_module_source_decoder_fn_t)(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loom_diagnostic_sink_t diagnostic_sink, loomc_allocator_t allocator,
    loomc_module_t* module, loom_module_t** out_internal_module);

// Encodes one projected internal module into an immutable source.
typedef loomc_status_t (*loomc_module_source_encoder_fn_t)(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

// Creates an empty public module handle with workspace-backed storage.
LOOMC_API_PRIVATE loomc_status_t loomc_module_create_empty(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_module_t** out_module);

// Returns the allocator owned by the public module handle.
LOOMC_API_PRIVATE loomc_allocator_t
loomc_module_allocator(const loomc_module_t* module);

// Returns the context retained by the public module handle.
LOOMC_API_PRIVATE loomc_context_t* loomc_module_context(
    const loomc_module_t* module);

// Returns the arena block pool backing the internal module.
LOOMC_API_PRIVATE iree_arena_block_pool_t* loomc_module_block_pool(
    loomc_module_t* module);

// Structural admission already established by the native producer.
typedef enum loomc_module_input_state_e {
  LOOMC_MODULE_INPUT_UNVERIFIED = 0,
  LOOMC_MODULE_INPUT_STRUCTURALLY_VERIFIED = 1,
} loomc_module_input_state_t;

// Transfers one internal module into an empty public module handle, retaining
// the producer's structural verification fact. Target-Low verification remains
// owned by the public module's target environment.
LOOMC_API_PRIVATE void loomc_module_set_loom_module(
    loomc_module_t* module, loom_module_t* internal_module,
    loomc_module_input_state_t input_state);

// Returns the internal module owned by a public module handle.
LOOMC_API_PRIVATE loom_module_t* loomc_module_loom_module(
    loomc_module_t* module);

// Returns the internal module owned by a public module handle.
LOOMC_API_PRIVATE const loom_module_t* loomc_module_const_loom_module(
    const loomc_module_t* module);

// Establishes structural and target-Low input invariants before compilation or
// direct emission. Successful verification against the module context is
// retained across trusted compiler transforms. Other target environments are
// checked without retaining their lifetime. Diagnostics mark |result| failed;
// status is reserved for infrastructure failures.
LOOMC_API_PRIVATE loomc_status_t
loomc_module_verify(loomc_module_t* module,
                    const loomc_target_environment_t* target_environment,
                    loomc_result_t* result);

// Invalidates input verification after an unsuccessful mutating operation.
LOOMC_API_PRIVATE void loomc_module_invalidate_verification(
    loomc_module_t* module);

// Validates source deserialization options without selecting a format.
LOOMC_API_PRIVATE loomc_status_t loomc_module_validate_deserialize_options(
    const loomc_module_deserialize_options_t* options);

// Validates source-to-module arguments and clears both outputs.
LOOMC_API_PRIVATE loomc_status_t loomc_module_validate_source_arguments(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, loomc_module_t** out_module,
    loomc_result_t** out_result);

// Deserializes a source through one statically selected format decoder.
LOOMC_API_PRIVATE loomc_status_t loomc_module_deserialize_explicit_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_source_decoder_fn_t decoder, loomc_allocator_t allocator,
    loomc_module_t** out_module, loomc_result_t** out_result);

// Validates source serialization options without selecting a format.
LOOMC_API_PRIVATE loomc_status_t loomc_module_validate_serialize_options(
    const loomc_module_serialize_options_t* options);

// Serializes a module through one statically selected format encoder.
LOOMC_API_PRIVATE loomc_status_t loomc_module_serialize_explicit_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_source_encoder_fn_t encoder, loomc_allocator_t allocator,
    loomc_source_t** out_source);

// Returns the owner carried by continuing compiler invocations on this module.
// Its products live until a failed mutation or module destruction.
LOOMC_API_PRIVATE loom_function_version_owner_t*
loomc_module_function_version_owner(loomc_module_t* module);

// Clears products after a failed mutation invalidates their construction.
LOOMC_API_PRIVATE void loomc_module_invalidate_compilation(
    loomc_module_t* module);

// Returns the resettable arena and result for completed-Low workgroup layouts.
// Only companion construction allocates this result. It is independent of
// durable function versions and becomes invalid before the next compilation.
LOOMC_API_PRIVATE iree_arena_allocator_t* loomc_module_workgroup_layout_arena(
    loomc_module_t* module);
LOOMC_API_PRIVATE loom_low_workgroup_layouts_t*
loomc_module_mutable_workgroup_layouts(loomc_module_t* module);
LOOMC_API_PRIVATE const loom_low_workgroup_layouts_t*
loomc_module_workgroup_layouts(const loomc_module_t* module);

// Discards only physical workgroup snapshots before compiler mutation or after
// failed native emission; live function versions and memory proofs survive.
LOOMC_API_PRIVATE void loomc_module_invalidate_workgroup_layouts(
    loomc_module_t* module);

// Captures applied invocation bindings into the module's compiler storage.
LOOMC_API_PRIVATE loom_tooling_config_binding_sink_t
loomc_module_config_binding_sink(loomc_module_t* module);

// Returns accumulated applied bindings, valid until a failed mutation or
// module destruction.
LOOMC_API_PRIVATE const loomc_config_binding_list_t*
loomc_module_config_bindings(const loomc_module_t* module);

// Returns function versions from the last successful compilation, or NULL.
LOOMC_API_PRIVATE const loom_function_version_list_t*
loomc_module_function_versions(const loomc_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_STORAGE_H_
