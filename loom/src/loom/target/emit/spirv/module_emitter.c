// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_emitter.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/module_contract.h"
#include "loom/target/emit/spirv/function_emitter.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/function_version.h"

typedef enum loom_spirv_emit_module_state_flag_bits_e {
  LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_NONE = 0u,
  LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED = 1u << 0,
} loom_spirv_emit_module_state_flag_bits_t;
typedef uint32_t loom_spirv_emit_module_state_flags_t;

typedef struct loom_spirv_emit_module_state_t {
  // Module containing the emitted low functions.
  loom_module_t* module;
  // Low descriptor registry used to resolve target-bound packets.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Structured diagnostic emitter for target-resolution failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Case/module scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Cached symbol facts shared by target resolution for every function.
  loom_symbol_fact_table_t symbol_facts;
  // Compiler function versions observed against this module symbol snapshot.
  loom_target_function_version_snapshot_t function_versions;
  // Optional caller-owned export projection buffer.
  loom_target_emit_export_projection_buffer_t* export_projection;
  // Number of export projection rows prepared during emission.
  iree_host_size_t export_projection_count;
  // Sectioned SPIR-V module builder shared by every emitted function.
  loom_spirv_module_builder_t builder;
  // SPIR-V type and constant emission cache shared by the module.
  loom_spirv_type_context_t type_context;
  // Module-level raw-BDA ABI layout shared by HAL kernel entries.
  loom_spirv_module_raw_bda_layout_t raw_bda_layout;
  // Shared Input variables for workgroup/local/global invocation builtins.
  uint32_t builtin_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
  // First function's module-level target contract.
  loom_spirv_module_contract_t contract;
  // Module-emission state flags.
  loom_spirv_emit_module_state_flags_t flags;
  // Number of functions emitted into the current module.
  iree_host_size_t function_count;
} loom_spirv_emit_module_state_t;

static bool loom_spirv_emit_selects_target(
    const loom_low_resolved_target_t* target) {
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  // Targetless Low assembly is selected by its representation contract below;
  // concrete targets participate only in their declared codegen format.
  return bundle == NULL || bundle->snapshot == NULL ||
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_SPIRV;
}

static iree_status_t loom_spirv_emit_validate_target(
    const loom_low_resolved_target_t* target) {
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V low function target did not resolve to a "
                            "descriptor set");
  }
  if (target->descriptor_set->stable_id !=
      SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    const iree_string_view_t descriptor_set_key =
        loom_low_descriptor_set_string(
            target->descriptor_set, target->descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low function selected descriptor set '%.*s'; "
        "expected 'spirv.logical.core'",
        (int)descriptor_set_key.size, descriptor_set_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_module_prepare_contract(
    loom_spirv_emit_module_state_t* state,
    const loom_low_resolved_target_t* target, iree_allocator_t allocator) {
  const loom_spirv_module_contract_t contract =
      loom_spirv_module_contract_from_target(target);
  if (!iree_any_bit_set(
          state->flags,
          LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED)) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_builder_initialize(
        loom_low_resolved_target_bundle(target), allocator, &state->builder));
    loom_spirv_type_context_initialize(&state->builder, state->scratch_arena,
                                       &state->type_context);
    state->contract = contract;
    state->flags |= LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED;
    return iree_ok_status();
  }
  if (!loom_spirv_module_contract_equal(&state->contract, &contract)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low module mixes target contract '%.*s' with "
        "target contract '%.*s'",
        (int)state->contract.target_name.size, state->contract.target_name.data,
        (int)contract.target_name.size, contract.target_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_low_function_into_module(
    loom_spirv_emit_module_state_t* state, loom_op_t* low_function_op,
    iree_allocator_t allocator) {
  if (!loom_low_function_def_isa(low_function_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V emission requires a low function "
                            "definition");
  }

  const loom_symbol_ref_t function_ref =
      loom_low_function_callee(low_function_op);
  const loom_target_function_version_t* function_version =
      loom_target_function_version_snapshot_at(&state->function_versions,
                                               function_ref.symbol_id);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      state->module, &state->symbol_facts, low_function_op,
      function_version ? function_version->function_target_facts : NULL,
      state->descriptor_registry, state->diagnostic_emitter, &target));
  if (!loom_spirv_emit_selects_target(&target)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_spirv_emit_validate_target(&target));
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_module_prepare_contract(state, &target, allocator));

  const loom_function_version_ordinal_t function_version_ordinal =
      loom_target_function_version_snapshot_ordinal_at(
          &state->function_versions, function_ref.symbol_id);
  if (state->export_projection != NULL &&
      function_version_ordinal != LOOM_FUNCTION_VERSION_ORDINAL_INVALID) {
    if (state->export_projection_count >= state->export_projection->capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "SPIR-V export projection capacity %" PRIhsz
                              " is too small for mapped entry points",
                              state->export_projection->capacity);
    }
    if (state->function_count >= UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "SPIR-V entry-point projection exceeds its "
                              "ordinal domain");
    }
  }

  loom_spirv_function_emission_context_t function_context = {
      .module = state->module,
      .scratch_arena = state->scratch_arena,
      .builder = &state->builder,
      .type_context = &state->type_context,
      .raw_bda_layout = &state->raw_bda_layout,
      .builtin_variable_ids = state->builtin_variable_ids,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_emit_low_function(&function_context,
                                                    low_function_op, &target));
  if (state->export_projection != NULL &&
      function_version_ordinal != LOOM_FUNCTION_VERSION_ORDINAL_INVALID) {
    state->export_projection->values[state->export_projection_count++] =
        (loom_target_emit_export_projection_t){
            .function_version_ordinal = function_version_ordinal,
            .ordinal = (uint32_t)state->function_count,
        };
  }
  ++state->function_count;
  return iree_ok_status();
}

static void loom_spirv_emit_module_state_initialize(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_emit_module_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  *out_state = (loom_spirv_emit_module_state_t){
      .module = module,
      .descriptor_registry = descriptor_registry,
      .diagnostic_emitter = diagnostic_emitter,
      .scratch_arena = scratch_arena,
  };
  loom_symbol_fact_table_initialize(&out_state->symbol_facts, scratch_arena);
}

static void loom_spirv_emit_module_state_deinitialize(
    loom_spirv_emit_module_state_t* state) {
  if (iree_any_bit_set(state->flags,
                       LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED)) {
    loom_spirv_module_builder_deinitialize(&state->builder);
  }
}

static iree_status_t loom_spirv_emit_module_state_finalize(
    loom_spirv_emit_module_state_t* state,
    loom_spirv_module_binary_t* out_module) {
  if (state->function_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V low module has no low function definitions");
  }
  loom_spirv_module_abi_context_t context = {
      .module = state->module,
      .scratch_arena = state->scratch_arena,
      .builder = &state->builder,
      .type_context = &state->type_context,
      .raw_bda_layout = &state->raw_bda_layout,
  };
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_abi_emit_metadata(&context, &state->raw_bda_layout));
  return loom_spirv_module_builder_finalize(&state->builder, out_module);
}

static iree_status_t loom_spirv_emit_low_module_initialize(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_spirv_emit_low_module_options_t* options,
    loom_spirv_module_binary_t* out_module,
    loom_spirv_emit_module_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  loom_spirv_emit_module_state_initialize(module, descriptor_registry,
                                          diagnostic_emitter, scratch_arena,
                                          out_state);
  out_state->export_projection = options ? options->export_projection : NULL;
  return loom_target_function_version_snapshot_build(
      module, options ? options->function_versions : NULL, scratch_arena,
      &out_state->function_versions);
}

void loom_spirv_emit_low_module_options_initialize(
    loom_spirv_emit_low_module_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_spirv_emit_low_module_options_t){0};
}

static iree_status_t loom_spirv_emit_low_module_options_validate(
    const loom_spirv_emit_low_module_options_t* options) {
  if (options == NULL) return iree_ok_status();
  if (options->export_projection != NULL &&
      options->export_projection->capacity != 0 &&
      options->export_projection->values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V export projection capacity requires caller-owned row storage");
  }
  if (options->entry_count == 0) return iree_ok_status();
  if (options->entry_ops == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected SPIR-V low module entries require an entry op list");
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected SPIR-V low module entry list contains a null op");
    }
  }
  return iree_ok_status();
}

static bool loom_spirv_emit_low_module_options_selects_entry(
    const loom_spirv_emit_low_module_options_t* options,
    loom_op_t* low_function_op) {
  if (options == NULL || options->entry_count == 0) {
    return true;
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == low_function_op) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_spirv_emit_low_module(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_spirv_emit_low_module_options_t* options,
    loom_spirv_module_binary_t* out_module, iree_allocator_t allocator) {
  loom_spirv_emit_module_state_t state = {0};
  if (options != NULL && options->export_projection != NULL) {
    options->export_projection->count = 0;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_emit_low_module_options_validate(options));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_low_module_initialize(
      module, descriptor_registry, diagnostic_emitter, scratch_arena, options,
      out_module, &state));

  iree_status_t status = iree_ok_status();
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_low_function_def_isa(symbol->defining_op) ||
        !loom_spirv_emit_low_module_options_selects_entry(
            options, symbol->defining_op)) {
      continue;
    }
    status = loom_spirv_emit_low_function_into_module(
        &state, symbol->defining_op, allocator);
    if (!iree_status_is_ok(status)) break;
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_module_state_finalize(&state, out_module);
  }
  if (iree_status_is_ok(status) && state.export_projection != NULL) {
    state.export_projection->count = state.export_projection_count;
  }
  loom_spirv_emit_module_state_deinitialize(&state);
  if (!iree_status_is_ok(status)) {
    loom_spirv_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
