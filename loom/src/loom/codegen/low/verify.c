// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/function_version.h"
#include "loom/target/registers.h"
#include "loom/util/walk.h"

typedef struct loom_low_verify_state_t {
  // Module being verified and owning the transient local-value map.
  loom_module_t* module;
  // Descriptor registry used to resolve target-low operations.
  const loom_low_descriptor_registry_t* registry;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t emitter;
  // Target-owned semantic verification providers.
  loom_low_verify_provider_list_t provider_list;
  // Accumulated verification result returned to the caller.
  loom_low_verify_result_t* result;
  // Maximum number of errors emitted before aborting the walk.
  uint32_t max_errors;
  // Storage retained across the full module verification run.
  iree_arena_allocator_t arena;
  // Function-local storage reset after each verified function.
  iree_arena_allocator_t walk_arena;
  // Lazily materialized module symbol facts.
  loom_symbol_fact_table_t symbol_facts;
  // Version handles densely indexed by the observed module symbol table.
  loom_target_function_version_snapshot_t function_version_snapshot;
  // Per-provider state retained for the full module verification run.
  void** provider_module_states;
} loom_low_verify_state_t;

typedef struct loom_low_function_verify_state_t {
  // Module-level verification state shared by the function walk.
  loom_low_verify_state_t* state;
  // Resolved target selected by the function declaration.
  const loom_low_resolved_target_t* target;
  // Descriptor-backed register type resolver for the selected target.
  loom_low_register_type_resolver_t register_type_resolver;
  // Stable function name used in diagnostics.
  iree_string_view_t function_name;
  // Target-low function definition being verified.
  const loom_op_t* function_op;
  // Function body or NULL for declarations and malformed definitions.
  loom_region_t* body;
  // Dense local ordinal domain covering the complete function region tree.
  loom_local_value_domain_t value_domain;
  // Register-part masks indexed by function-local value ordinal.
  loom_low_register_part_mask_t* defined_masks;
  // Packed storage sizes accumulated during the existing verifier walk.
  loom_low_storage_layout_space_sizes_t storage_space_sizes;
  // Function-local state for each target-owned verification provider.
  void** provider_states;
} loom_low_function_verify_state_t;

struct loom_low_verify_context_t {
  // Function currently being verified.
  loom_low_function_verify_state_t* function_state;
  // Module-level state for the provider receiving this callback.
  void* provider_module_state;
};

struct loom_low_verify_module_context_t {
  // Module verification state shared by all functions in this run.
  loom_low_verify_state_t* state;
};

typedef struct loom_low_packet_field_t {
  iree_string_view_t field_name;
  loom_diagnostic_field_ref_t field_ref;
  loom_value_id_t value_id;
  loom_type_t type;
} loom_low_packet_field_t;

static bool loom_low_verify_should_stop(const loom_low_verify_state_t* state) {
  return state->max_errors != 0 &&
         state->result->error_count >= state->max_errors;
}

static iree_status_t loom_low_verify_emit(
    loom_low_verify_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count) {
  if (loom_low_verify_should_stop(state)) {
    return iree_ok_status();
  }
  if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_ERROR) {
    ++state->result->error_count;
  } else if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_WARNING) {
    ++state->result->warning_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  return iree_diagnostic_emit(state->emitter, &emission);
}

static iree_status_t loom_low_verify_counting_emitter(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_low_verify_state_t* state = (loom_low_verify_state_t*)user_data;
  return loom_low_verify_emit(
      state, emission->op, emission->error, emission->params,
      emission->param_count, emission->related_ops, emission->related_op_count);
}

const loom_module_t* loom_low_verify_context_module(
    const loom_low_verify_context_t* context) {
  return context->function_state->state->module;
}

const loom_module_t* loom_low_verify_module_context_module(
    const loom_low_verify_module_context_t* context) {
  return context->state->module;
}

iree_arena_allocator_t* loom_low_verify_module_context_arena(
    loom_low_verify_module_context_t* context) {
  return &context->state->arena;
}

bool loom_low_verify_module_context_should_stop(
    const loom_low_verify_module_context_t* context) {
  return loom_low_verify_should_stop(context->state);
}

const loom_op_t* loom_low_verify_context_function_op(
    const loom_low_verify_context_t* context) {
  return context->function_state->function_op;
}

const loom_region_t* loom_low_verify_context_function_body(
    const loom_low_verify_context_t* context) {
  return context->function_state->body;
}

const loom_low_resolved_target_t* loom_low_verify_context_target(
    const loom_low_verify_context_t* context) {
  return context->function_state->target;
}

iree_arena_allocator_t* loom_low_verify_context_arena(
    loom_low_verify_context_t* context) {
  return &context->function_state->state->walk_arena;
}

void* loom_low_verify_context_provider_module_state(
    const loom_low_verify_context_t* context) {
  return context->provider_module_state;
}

bool loom_low_verify_context_should_stop(
    const loom_low_verify_context_t* context) {
  return loom_low_verify_should_stop(context->function_state->state);
}

iree_status_t loom_low_verify_context_emit(
    loom_low_verify_context_t* context, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  return loom_low_verify_emit(context->function_state->state, op, error, params,
                              param_count, NULL, 0);
}

static iree_string_view_t loom_low_verify_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_verify_function_name(
    const loom_module_t* module, const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_verify_symbol_name(module,
                                       loom_low_func_def_callee(low_func_op));
  }
  if (loom_low_kernel_def_isa(low_func_op)) {
    return loom_low_verify_symbol_name(module,
                                       loom_low_kernel_def_callee(low_func_op));
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    return loom_low_verify_symbol_name(module,
                                       loom_low_func_decl_callee(low_func_op));
  }
  return IREE_SV("<unnamed>");
}

static loom_region_t* loom_low_verify_function_body(
    const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_func_def_body(low_func_op);
  }
  if (loom_low_kernel_def_isa(low_func_op)) {
    return loom_low_kernel_def_body(low_func_op);
  }
  return NULL;
}

static const loom_target_facts_t* loom_low_verify_function_target_facts(
    const loom_low_verify_state_t* state, const loom_op_t* low_func_op) {
  const loom_func_like_t function =
      loom_func_like_const_cast(state->module, low_func_op);
  if (!loom_func_like_isa(function)) {
    return NULL;
  }
  const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(function_ref) || function_ref.module_id != 0 ||
      function_ref.symbol_id >= state->function_version_snapshot.symbol_count) {
    return NULL;
  }
  return loom_target_function_version_target_facts(
      loom_target_function_version_snapshot_handle_at(
          &state->function_version_snapshot, function_ref.symbol_id));
}

static iree_string_view_t loom_low_verify_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_low_verify_descriptor_op_name(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->carrier) {
    case LOOM_LOW_DESCRIPTOR_CARRIER_PACKET:
      switch (descriptor->op_kind) {
        case LOOM_LOW_DESCRIPTOR_OP_KIND_OP:
          return IREE_SV("low.op");
        case LOOM_LOW_DESCRIPTOR_OP_KIND_CONST:
          return IREE_SV("low.const");
      }
      break;
    case LOOM_LOW_DESCRIPTOR_CARRIER_BRANCH:
      return IREE_SV("low.br");
    case LOOM_LOW_DESCRIPTOR_CARRIER_SWITCH:
      return IREE_SV("low.switch");
  }
  IREE_ASSERT_UNREACHABLE("invalid descriptor carrier");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_low_verify_emit_descriptor_op_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t actual_op_name, iree_string_view_t expected_op_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_string(actual_op_name),
      loom_param_string(expected_op_name),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_STRUCTURE_039,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}
static iree_status_t loom_low_verify_emit_missing_immediate(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_047,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_unexpected_immediate_attr(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t attr_name, uint16_t attrs_attr_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_048,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_immediate_kind_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    loom_attr_kind_t actual_kind, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_049,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_immediate_range_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    int64_t actual_value, iree_string_view_t expected_range) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_i64(actual_value),
      loom_param_string(expected_range),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_STRUCTURE_014,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_format_enum_constraint(
    loom_low_function_verify_state_t* function_state,
    iree_string_view_t enum_domain, iree_string_view_t* out_constraint) {
  const iree_host_size_t byte_count =
      IREE_SV("value from enum domain ''").size + enum_domain.size;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           byte_count + 1, (void**)&storage));
  int length =
      iree_snprintf(storage, byte_count + 1, "value from enum domain '%.*s'",
                    (int)enum_domain.size, enum_domain.data);
  if (length < 0 || (iree_host_size_t)length != byte_count) {
    IREE_ASSERT_UNREACHABLE("failed to format enum constraint");
    IREE_BUILTIN_UNREACHABLE();
  }
  *out_constraint = iree_make_string_view(storage, byte_count);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_emit_enum_domain_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    iree_string_view_t actual_value, iree_string_view_t enum_domain) {
  iree_string_view_t expected_constraint = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_format_enum_constraint(
      function_state, enum_domain, &expected_constraint));
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_string(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_STRUCTURE_027,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_constraint_type_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_packet_field_t* lhs_field,
    const loom_low_packet_field_t* rhs_field) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(lhs_field->field_name),
                                lhs_field->field_ref),
      loom_param_type(lhs_field->type),
      loom_param_with_field_ref(loom_param_string(rhs_field->field_name),
                                rhs_field->field_ref),
      loom_param_type(rhs_field->type),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TYPE_001,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_count_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_error_def_t* error, iree_string_view_t descriptor_key,
    uint16_t descriptor_attr_index, uint32_t actual_count,
    uint32_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_verify_emit(function_state->state, op, error, params,
                              IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_missing_features(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    uint32_t feature_word_index, uint64_t missing_bits) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    descriptor_attr_index)),
      loom_param_string(target->descriptor_set_key),
      loom_param_u32(feature_word_index),
      loom_param_u64(missing_bits),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_046,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static const loom_named_attr_t* loom_low_verify_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t entry_name =
        loom_low_verify_string_or_empty(module, entry->name_id);
    if (iree_string_view_equal(entry_name, name)) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_low_verify_descriptor_immediate_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_immediate_index,
    const loom_low_immediate_t** out_immediate,
    iree_string_view_t* out_immediate_name) {
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor_immediate_index >=
          descriptor_set->immediate_count - descriptor->immediate_start) {
    const uint64_t immediate_row =
        (uint64_t)descriptor->immediate_start + descriptor_immediate_index;
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor immediate row %" PRIu64
                            " is out of range",
                            immediate_row);
  }
  const uint32_t immediate_row =
      descriptor->immediate_start + descriptor_immediate_index;
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  *out_immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  *out_immediate = immediate;
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_has_immediate_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t attr_name,
    bool* out_has_immediate) {
  *out_has_immediate = false;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate = NULL;
    iree_string_view_t immediate_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_immediate_name(
        descriptor_set, descriptor, i, &immediate, &immediate_name));
    (void)immediate;
    if (iree_string_view_equal(attr_name, immediate_name)) {
      *out_has_immediate = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_format_signed_range(
    loom_low_function_verify_state_t* function_state, int64_t minimum,
    int64_t maximum, iree_string_view_t* out_range) {
  char scratch[96];
  int length = iree_snprintf(scratch, sizeof(scratch), "%" PRId64 "..%" PRId64,
                             minimum, maximum);
  if (length < 0 || (iree_host_size_t)length >= sizeof(scratch)) {
    IREE_ASSERT_UNREACHABLE("failed to format signed immediate range");
    IREE_BUILTIN_UNREACHABLE();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           (iree_host_size_t)length,
                                           (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_range = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_format_unsigned_range(
    loom_low_function_verify_state_t* function_state, uint64_t maximum,
    iree_string_view_t* out_range) {
  char scratch[96];
  int length = iree_snprintf(scratch, sizeof(scratch), "0..%" PRIu64, maximum);
  if (length < 0 || (iree_host_size_t)length >= sizeof(scratch)) {
    IREE_ASSERT_UNREACHABLE("failed to format unsigned immediate range");
    IREE_BUILTIN_UNREACHABLE();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           (iree_host_size_t)length,
                                           (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_range = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_format_i64(
    loom_low_function_verify_state_t* function_state, int64_t value,
    iree_string_view_t* out_value) {
  char scratch[32];
  int length = iree_snprintf(scratch, sizeof(scratch), "%" PRId64, value);
  if (length < 0 || (iree_host_size_t)length >= sizeof(scratch)) {
    IREE_ASSERT_UNREACHABLE("failed to format enum immediate value");
    IREE_BUILTIN_UNREACHABLE();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           (iree_host_size_t)length,
                                           (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_value = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_immediate_enum_domain(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate,
    iree_string_view_t* out_domain_name) {
  if (immediate->enum_domain_id >= descriptor_set->enum_domain_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low enum immediate references enum domain %" PRIu16
                            " but only %" PRIu32 " domains exist",
                            immediate->enum_domain_id,
                            descriptor_set->enum_domain_count);
  }
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[immediate->enum_domain_id];
  *out_domain_name = loom_low_descriptor_set_string(descriptor_set,
                                                    domain->name_string_offset);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_i64_immediate_range(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    const loom_low_immediate_t* immediate, int64_t value) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      if (value >= immediate->signed_min && value <= maximum) {
        return iree_ok_status();
      }
      iree_string_view_t expected_range = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_format_signed_range(
          function_state, immediate->signed_min, maximum, &expected_range));
      return loom_low_verify_emit_immediate_range_mismatch(
          function_state, op, immediate_name, attrs_attr_index, value,
          expected_range);
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED: {
      // I64 attributes carry the bits of full-width unsigned immediates.
      if ((uint64_t)value <= immediate->unsigned_max) {
        return iree_ok_status();
      }
      iree_string_view_t expected_range = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_format_unsigned_range(
          function_state, immediate->unsigned_max, &expected_range));
      return loom_low_verify_emit_immediate_range_mismatch(
          function_state, op, immediate_name, attrs_attr_index, value,
          expected_range);
    }
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL: {
      if (value >= 0 && (uint64_t)value <= immediate->unsigned_max) {
        return iree_ok_status();
      }
      iree_string_view_t expected_range = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_format_unsigned_range(
          function_state, immediate->unsigned_max, &expected_range));
      return loom_low_verify_emit_immediate_range_mismatch(
          function_state, op, immediate_name, attrs_attr_index, value,
          expected_range);
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low immediate '%.*s' has unsupported kind %u",
                              (int)immediate_name.size, immediate_name.data,
                              (unsigned)immediate->kind);
  }
}

static iree_string_view_t loom_low_verify_expected_immediate_kind(
    const loom_low_immediate_t* immediate) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
      return IREE_SV("i64 signed integer");
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
      return IREE_SV("i64 unsigned integer");
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      if (iree_all_bits_set(immediate->flags,
                            LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC)) {
        return IREE_SV("symbol reference or i64 ordinal");
      }
      return IREE_SV("i64 ordinal");
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      return IREE_SV("string enum token or i64 enum ordinal");
    default:
      return IREE_SV("known immediate kind");
  }
}

static bool loom_low_immediate_has_default(
    const loom_low_immediate_t* immediate) {
  return iree_any_bit_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE);
}

static iree_status_t loom_low_verify_descriptor_immediate_attr(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    const loom_low_immediate_t* immediate, const loom_named_attr_t* attr) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED: {
      if (attr->value.kind != LOOM_ATTR_I64) {
        return loom_low_verify_emit_immediate_kind_mismatch(
            function_state, op, descriptor_key, descriptor_attr_index,
            immediate_name, attrs_attr_index,
            (loom_attr_kind_t)attr->value.kind,
            loom_low_verify_expected_immediate_kind(immediate));
      }
      return loom_low_verify_i64_immediate_range(
          function_state, op, descriptor_key, descriptor_attr_index,
          immediate_name, attrs_attr_index, immediate, attr->value.i64);
    }
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL: {
      if (attr->value.kind == LOOM_ATTR_SYMBOL &&
          iree_all_bits_set(immediate->flags,
                            LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC)) {
        return iree_ok_status();
      }
      if (attr->value.kind != LOOM_ATTR_I64) {
        return loom_low_verify_emit_immediate_kind_mismatch(
            function_state, op, descriptor_key, descriptor_attr_index,
            immediate_name, attrs_attr_index,
            (loom_attr_kind_t)attr->value.kind,
            loom_low_verify_expected_immediate_kind(immediate));
      }
      return loom_low_verify_i64_immediate_range(
          function_state, op, descriptor_key, descriptor_attr_index,
          immediate_name, attrs_attr_index, immediate, attr->value.i64);
    }
    case LOOM_LOW_IMMEDIATE_KIND_ENUM: {
      const loom_low_descriptor_set_t* descriptor_set =
          function_state->target->descriptor_set;
      iree_string_view_t domain_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_immediate_enum_domain(
          descriptor_set, immediate, &domain_name));
      if (attr->value.kind == LOOM_ATTR_STRING) {
        iree_string_view_t actual_token = loom_low_verify_string_or_empty(
            function_state->state->module, attr->value.string_id);
        int64_t enum_value = 0;
        if (loom_low_descriptor_set_lookup_enum_value_by_token(
                descriptor_set, immediate->enum_domain_id, actual_token,
                &enum_value)) {
          return iree_ok_status();
        }
        return loom_low_verify_emit_enum_domain_mismatch(
            function_state, op, immediate_name, attrs_attr_index, actual_token,
            domain_name);
      }
      if (attr->value.kind != LOOM_ATTR_I64) {
        return loom_low_verify_emit_immediate_kind_mismatch(
            function_state, op, descriptor_key, descriptor_attr_index,
            immediate_name, attrs_attr_index,
            (loom_attr_kind_t)attr->value.kind,
            loom_low_verify_expected_immediate_kind(immediate));
      }
      iree_string_view_t enum_token = iree_string_view_empty();
      if (loom_low_descriptor_set_lookup_enum_token_by_value(
              descriptor_set, immediate->enum_domain_id, attr->value.i64,
              &enum_token)) {
        return iree_ok_status();
      }
      iree_string_view_t actual_value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_format_i64(
          function_state, attr->value.i64, &actual_value));
      return loom_low_verify_emit_enum_domain_mismatch(
          function_state, op, immediate_name, attrs_attr_index, actual_value,
          domain_name);
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low immediate '%.*s' has unsupported kind %u",
                              (int)immediate_name.size, immediate_name.data,
                              (unsigned)immediate->kind);
  }
}

static iree_status_t loom_low_verify_descriptor_immediates(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    const loom_low_descriptor_t* descriptor) {
  const loom_module_t* module = function_state->state->module;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  uint16_t attrs_attr_index = 0;
  if (!loom_low_packet_try_op_attrs(op, &attrs, &attrs_attr_index)) {
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate = NULL;
    iree_string_view_t immediate_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_immediate_name(
        descriptor_set, descriptor, i, &immediate, &immediate_name));
    const loom_named_attr_t* attr =
        loom_low_verify_find_named_attr(module, attrs, immediate_name);
    if (!attr) {
      if (!loom_low_immediate_has_default(immediate)) {
        IREE_RETURN_IF_ERROR(loom_low_verify_emit_missing_immediate(
            function_state, op, descriptor_key, descriptor_attr_index,
            immediate_name, attrs_attr_index));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_immediate_attr(
          function_state, op, descriptor_key, descriptor_attr_index,
          immediate_name, attrs_attr_index, immediate, attr));
    }
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }

  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    iree_string_view_t attr_name =
        loom_low_verify_string_or_empty(module, attr->name_id);
    bool has_immediate = false;
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_has_immediate_name(
        descriptor_set, descriptor, attr_name, &has_immediate));
    if (has_immediate) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_unexpected_immediate_attr(
        function_state, op, descriptor_key, descriptor_attr_index, attr_name,
        attrs_attr_index));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_verify_constraint_requires_matching_types(
    loom_low_constraint_kind_t kind) {
  return kind == LOOM_LOW_CONSTRAINT_KIND_TIED ||
         kind == LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE;
}

static iree_status_t loom_low_verify_descriptor_packet_operand_field(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    uint16_t packet_operand_index, loom_low_packet_field_t* out_field) {
  const loom_module_t* module = function_state->state->module;
  if (packet_operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low packet operand index %" PRIu16
                            " exceeds op operand count %" PRIu16,
                            packet_operand_index, op->operand_count);
  }
  const loom_value_id_t value_id =
      loom_op_const_operands(op)[packet_operand_index];
  out_field->field_ref = loom_diagnostic_field_ref(
      LOOM_DIAGNOSTIC_FIELD_OPERAND, packet_operand_index);
  out_field->value_id = value_id;
  out_field->type = loom_module_value_type(module, value_id);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_packet_field(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    loom_low_packet_field_t* out_field) {
  const loom_module_t* module = function_state->state->module;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  memset(out_field, 0, sizeof(*out_field));
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor operand index %" PRIu16
                            " exceeds descriptor operand count %" PRIu16,
                            descriptor_operand_index,
                            descriptor->operand_count);
  }

  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  if (operand_row >= descriptor_set->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor operand row %" PRIu32 " is out of range", operand_row);
  }
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set->operands[operand_row];
  out_field->field_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor_operand->field_name_string_offset);

  if (descriptor_operand_index < descriptor->result_count) {
    const uint16_t result_index = descriptor_operand->source_value_index;
    if (result_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low packet result index %" PRIu16
                              " exceeds op result count %" PRIu16,
                              result_index, op->result_count);
    }
    const loom_value_id_t value_id = loom_op_const_results(op)[result_index];
    out_field->field_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, result_index);
    out_field->value_id = value_id;
    out_field->type = loom_module_value_type(module, value_id);
    return iree_ok_status();
  }

  if (descriptor_operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor constraint references implicit "
                            "operand row %" PRIu16,
                            descriptor_operand_index);
  }

  return loom_low_verify_descriptor_packet_operand_field(
      function_state, op, descriptor_operand->source_value_index, out_field);
}

static iree_status_t loom_low_verify_format_expected_register_classes(
    loom_low_function_verify_state_t* function_state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* descriptor_operand,
    iree_string_view_t* out_expected_reg_classes) {
  *out_expected_reg_classes = iree_string_view_empty();
  iree_host_size_t byte_count = 0;
  uint32_t reg_class_count = 0;
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[alt->reg_class_id];
    iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset);
    if (reg_class_count > 0) {
      byte_count += 3;
    }
    byte_count += reg_class_name.size;
    ++reg_class_count;
  }
  if (reg_class_count == 0) {
    *out_expected_reg_classes = IREE_SV("<none>");
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           byte_count, (void**)&storage));
  char* cursor = storage;
  uint32_t appended_count = 0;
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[alt->reg_class_id];
    iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset);
    if (appended_count > 0) {
      memcpy(cursor, " | ", 3);
      cursor += 3;
    }
    memcpy(cursor, reg_class_name.data, reg_class_name.size);
    cursor += reg_class_name.size;
    ++appended_count;
  }
  *out_expected_reg_classes = iree_make_string_view(storage, byte_count);
  return iree_ok_status();
}

static bool loom_low_verify_operand_accepts_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* descriptor_operand,
    uint16_t descriptor_register_class_id) {
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (alt->reg_class_id == descriptor_register_class_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_verify_format_register_constraint(
    loom_low_function_verify_state_t* function_state,
    iree_string_view_t expected_reg_classes, uint32_t minimum_unit_count,
    uint32_t maximum_unit_count, iree_string_view_t* out_constraint) {
  const iree_host_size_t maximum_unit_count_digits = 10;
  const iree_host_size_t byte_capacity =
      IREE_SV("register class in [] with  through  unit(s)").size +
      expected_reg_classes.size + 2 * maximum_unit_count_digits;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &function_state->state->arena, byte_capacity + 1, (void**)&storage));
  int length = 0;
  if (minimum_unit_count == maximum_unit_count) {
    length = iree_snprintf(storage, byte_capacity + 1,
                           "register class in [%.*s] with %" PRIu32 " unit(s)",
                           (int)expected_reg_classes.size,
                           expected_reg_classes.data, maximum_unit_count);
  } else {
    length = iree_snprintf(
        storage, byte_capacity + 1,
        "register class in [%.*s] with %" PRIu32 " through %" PRIu32 " unit(s)",
        (int)expected_reg_classes.size, expected_reg_classes.data,
        minimum_unit_count, maximum_unit_count);
  }
  if (length < 0 || (iree_host_size_t)length > byte_capacity) {
    IREE_ASSERT_UNREACHABLE("failed to format register constraint");
    IREE_BUILTIN_UNREACHABLE();
  }
  *out_constraint = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_emit_register_type_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    loom_diagnostic_field_ref_t field_ref, iree_string_view_t field_name,
    loom_type_t actual_type, iree_string_view_t expected_reg_classes,
    uint32_t minimum_unit_count, uint32_t maximum_unit_count) {
  iree_string_view_t expected_constraint = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_format_register_constraint(
      function_state, expected_reg_classes, minimum_unit_count,
      maximum_unit_count, &expected_constraint));
  const loom_error_def_t* error = field_ref.kind == LOOM_DIAGNOSTIC_FIELD_RESULT
                                      ? LOOM_ERR_TYPE_004
                                      : LOOM_ERR_TYPE_003;
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_low_verify_emit(function_state->state, op, error, params,
                              IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_undefined_register_part(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t op_name, loom_diagnostic_field_ref_t field_ref,
    iree_string_view_t field_name, loom_low_register_part_mask_t required_mask,
    loom_low_register_part_mask_t defined_mask) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_string(op_name),
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_u32(required_mask),
      loom_param_u32(defined_mask),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_DOMINANCE_011,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static loom_low_register_part_mask_t loom_low_verify_value_defined_mask(
    const loom_low_function_verify_state_t* function_state,
    loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(&function_state->value_domain, value_id);
  return function_state->defined_masks[value_ordinal];
}

static void loom_low_verify_set_value_defined_mask(
    loom_low_function_verify_state_t* function_state, loom_value_id_t value_id,
    loom_low_register_part_mask_t mask) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(&function_state->value_domain, value_id);
  function_state->defined_masks[value_ordinal] = mask;
}

static loom_low_register_part_mask_t
loom_low_verify_register_full_mask_for_type(
    loom_low_function_verify_state_t* function_state, loom_type_t type) {
  if (!loom_low_type_is_register(type)) {
    return 0;
  }
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = NULL;
  bool found_descriptor_register_class =
      loom_low_register_type_resolver_try_resolve(
          &function_state->register_type_resolver, type,
          &descriptor_register_class_id, &descriptor_register_class);
  if (found_descriptor_register_class) {
    return descriptor_register_class->full_register_part_mask;
  }
  return 0;
}

static loom_low_register_part_mask_t
loom_low_verify_descriptor_operand_part_mask(
    loom_low_function_verify_state_t* function_state,
    const loom_low_operand_t* descriptor_operand, loom_type_t actual_type) {
  if (descriptor_operand->register_part_id == LOOM_LOW_REGISTER_PART_NONE) {
    return loom_low_verify_register_full_mask_for_type(function_state,
                                                       actual_type);
  }
  if (!loom_low_type_is_register(actual_type)) {
    return 0;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  IREE_ASSERT(descriptor_operand->register_part_id <
                  descriptor_set->register_part_count,
              "verified low descriptor operand register part");
  const loom_low_register_part_t* register_part =
      &descriptor_set->register_parts[descriptor_operand->register_part_id];
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_descriptor_register_class =
      loom_low_register_type_resolver_try_resolve(
          &function_state->register_type_resolver, actual_type,
          &descriptor_register_class_id, NULL);
  IREE_ASSERT(found_descriptor_register_class &&
                  descriptor_register_class_id == register_part->reg_class_id,
              "verified low descriptor operand register-part class");
  return register_part->mask;
}

static iree_status_t loom_low_verify_emit_resource_register_class_missing(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    loom_type_t actual_type) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_string(IREE_SV("resource")),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("low.resource")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_resource_import_kind_ATTR_INDEX)),
      loom_param_with_field_ref(
          loom_param_type(actual_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_string(target->descriptor_set_key),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_042,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_resource_unit_count_exceeded(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    loom_type_t actual_type, uint32_t unit_count, uint32_t allocatable_count) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_string(IREE_SV("resource")),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("low.resource")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_resource_import_kind_ATTR_INDEX)),
      loom_param_with_field_ref(
          loom_param_type(actual_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_u32(unit_count),
      loom_param_string(target->descriptor_set_key),
      loom_param_u32(allocatable_count),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_043,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_resource(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op) {
  const loom_module_t* module = function_state->state->module;
  const loom_type_t actual_type =
      loom_module_value_type(module, loom_low_resource_result(op));
  const loom_low_reg_class_t* descriptor_register_class = NULL;
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_descriptor_register_class =
      loom_low_register_type_resolver_try_resolve(
          &function_state->register_type_resolver, actual_type,
          &descriptor_register_class_id, &descriptor_register_class);
  if (!found_descriptor_register_class) {
    return loom_low_verify_emit_resource_register_class_missing(
        function_state, op, actual_type);
  }

  const uint32_t unit_count = loom_low_register_type_unit_count(actual_type);
  if (descriptor_register_class->allocatable_count != 0 &&
      unit_count > descriptor_register_class->allocatable_count) {
    return loom_low_verify_emit_resource_unit_count_exceeded(
        function_state, op, actual_type, unit_count,
        descriptor_register_class->allocatable_count);
  }

  return iree_ok_status();
}

static iree_string_view_t loom_low_verify_value_name_or_placeholder(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t placeholder) {
  if (value_id >= module->values.count) {
    return placeholder;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (value->name_id == LOOM_STRING_ID_INVALID ||
      value->name_id >= module->strings.count) {
    return placeholder;
  }
  return module->strings.entries[value->name_id];
}

static iree_status_t loom_low_verify_emit_unresolved_register_class(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t value_kind, iree_string_view_t value_name,
    loom_type_t actual_type) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_string(value_kind),
      loom_param_string(value_name),
      loom_param_type(actual_type),
      loom_param_string(target->descriptor_set_key),
  };
  return loom_low_verify_emit(function_state->state, op, LOOM_ERR_TARGET_042,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_register_value_class(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t value_kind, iree_string_view_t placeholder,
    loom_value_id_t value_id) {
  const loom_module_t* module = function_state->state->module;
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_low_type_is_register(type)) {
    return iree_ok_status();
  }
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_descriptor_register_class =
      loom_low_register_type_resolver_try_resolve(
          &function_state->register_type_resolver, type,
          &descriptor_register_class_id, NULL);
  if (found_descriptor_register_class) {
    return iree_ok_status();
  }
  return loom_low_verify_emit_unresolved_register_class(
      function_state, op, value_kind,
      loom_low_verify_value_name_or_placeholder(module, value_id, placeholder),
      type);
}

static iree_status_t loom_low_verify_function_register_values(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    loom_region_t* body) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_register_value_class(
        function_state, op, IREE_SV("result"), IREE_SV("result"), results[i]));
  }
  if (loom_low_func_decl_isa(op)) {
    const loom_value_id_t* args = loom_op_const_operands(op);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_verify_register_value_class(
          function_state, op, IREE_SV("argument"), IREE_SV("argument"),
          args[i]));
    }
    return iree_ok_status();
  }
  if (!body) {
    return iree_ok_status();
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_verify_register_value_class(
          function_state, op, IREE_SV("argument"), IREE_SV("argument"),
          loom_block_arg_id(block, i)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_workgroup_storage_limit(
    loom_low_function_verify_state_t* function_state) {
  const loom_target_bundle_t* bundle =
      loom_low_resolved_target_bundle(function_state->target);
  if (bundle == NULL) return iree_ok_status();
  const uint64_t limit = bundle->snapshot->max_workgroup_storage_bytes;
  if (limit == 0 || function_state->body == NULL ||
      loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  if (function_state->storage_space_sizes.workgroup_bytes <= limit) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_string(bundle->name),
      loom_param_u64(function_state->storage_space_sizes.workgroup_bytes),
      loom_param_u64(limit),
  };
  return loom_low_verify_emit(function_state->state,
                              function_state->function_op, LOOM_ERR_TARGET_051,
                              params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_descriptor_register_field(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    bool is_result, uint16_t field_index) {
  const loom_module_t* module = function_state->state->module;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  if (operand_row >= descriptor_set->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor operand row %" PRIu32 " is out of range", operand_row);
  }
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set->operands[operand_row];

  const loom_value_id_t value_id =
      is_result ? loom_op_const_results(op)[field_index]
                : loom_op_const_operands(op)[field_index];
  const loom_type_t actual_type = loom_module_value_type(module, value_id);

  bool accepted = false;
  if (loom_low_type_is_register(actual_type) &&
      loom_low_operand_accepts_unit_count(
          descriptor_operand, loom_low_register_type_unit_count(actual_type))) {
    uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
    bool found_descriptor_register_class =
        loom_low_register_type_resolver_try_resolve(
            &function_state->register_type_resolver, actual_type,
            &descriptor_register_class_id, NULL);
    accepted =
        found_descriptor_register_class &&
        loom_low_verify_operand_accepts_register_class(
            descriptor_set, descriptor_operand, descriptor_register_class_id);
  }
  if (accepted) {
    return iree_ok_status();
  }

  iree_string_view_t expected_reg_classes = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_format_expected_register_classes(
      function_state, descriptor_set, descriptor_operand,
      &expected_reg_classes));
  iree_string_view_t field_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor_operand->field_name_string_offset);
  const loom_diagnostic_field_ref_t field_ref = loom_diagnostic_field_ref(
      is_result ? LOOM_DIAGNOSTIC_FIELD_RESULT : LOOM_DIAGNOSTIC_FIELD_OPERAND,
      field_index);
  return loom_low_verify_emit_register_type_mismatch(
      function_state, op, field_ref, field_name, actual_type,
      expected_reg_classes,
      iree_any_bit_set(descriptor_operand->flags,
                       LOOM_LOW_OPERAND_FLAG_VARIABLE_UNIT_COUNT)
          ? 1
          : descriptor_operand->unit_count,
      descriptor_operand->unit_count);
}

static iree_status_t loom_low_verify_descriptor_registers(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  const bool has_variadic_operands =
      loom_low_descriptor_has_variadic_operands(descriptor);
  const uint16_t fixed_descriptor_operand_end =
      descriptor->operand_count - (has_variadic_operands ? 1 : 0);
  uint16_t operand_field_index = 0;
  for (uint16_t i = 0; i < fixed_descriptor_operand_end; ++i) {
    const bool is_result = i < descriptor->result_count;
    uint16_t field_index = i;
    if (!is_result) {
      const loom_low_descriptor_set_t* descriptor_set =
          function_state->target->descriptor_set;
      const uint32_t operand_row = descriptor->operand_start + i;
      if (operand_row >= descriptor_set->operand_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low descriptor operand row %" PRIu32
                                " is out of range",
                                operand_row);
      }
      const loom_low_operand_t* descriptor_operand =
          &descriptor_set->operands[operand_row];
      if (descriptor_operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
        continue;
      }
      field_index = operand_field_index++;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_register_field(
        function_state, op, descriptor, i, is_result, field_index));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  if (has_variadic_operands) {
    const uint16_t descriptor_operand_index = fixed_descriptor_operand_end;
    for (uint16_t field_index = descriptor->minimum_packet_operand_count;
         field_index < op->operand_count; ++field_index) {
      IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_register_field(
          function_state, op, descriptor, descriptor_operand_index,
          /*is_result=*/false, field_index));
      if (loom_low_verify_should_stop(function_state->state)) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_constraints(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_row = descriptor->constraint_start + i;
    if (constraint_row >= descriptor_set->constraint_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor constraint row %" PRIu32
                              " is out of range",
                              constraint_row);
    }
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_row];
    if (!loom_low_verify_constraint_requires_matching_types(constraint->kind)) {
      continue;
    }
    if (constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor binary constraint kind %u has no "
                              "rhs operand",
                              (unsigned)constraint->kind);
    }

    loom_low_packet_field_t lhs_field;
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_field(
        function_state, op, descriptor, constraint->lhs_operand_index,
        &lhs_field));
    loom_low_packet_field_t rhs_field;
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_field(
        function_state, op, descriptor, constraint->rhs_operand_index,
        &rhs_field));
    if (loom_type_equal(lhs_field.type, rhs_field.type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_constraint_type_mismatch(
        function_state, op, &lhs_field, &rhs_field));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_verify_descriptor_tied_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index,
    uint16_t* out_operand_index) {
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED ||
        constraint->lhs_operand_index != result_index ||
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
      continue;
    }
    *out_operand_index = constraint->rhs_operand_index;
    return true;
  }
  return false;
}

static bool loom_low_verify_descriptor_declares_tied_result(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_tied_result_t tied) {
  if (descriptor == NULL || tied.result_index >= descriptor->result_count) {
    return false;
  }
  uint16_t tied_operand_index = LOOM_LOW_ID_NONE;
  if (!loom_low_verify_descriptor_tied_operand_index(
          descriptor_set, descriptor, tied.result_index, &tied_operand_index)) {
    return false;
  }
  const loom_low_operand_t* tied_operand =
      &descriptor_set->operands[descriptor->operand_start + tied_operand_index];
  return tied_operand->source_value_index == tied.operand_index;
}

static iree_status_t loom_low_verify_descriptor_register_parts(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const bool has_variadic_operands =
      loom_low_descriptor_has_variadic_operands(descriptor);
  const uint16_t fixed_descriptor_operand_end =
      descriptor->operand_count - (has_variadic_operands ? 1 : 0);
  for (uint16_t i = descriptor->result_count; i < fixed_descriptor_operand_end;
       ++i) {
    const uint32_t operand_row = descriptor->operand_start + i;
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set->operands[operand_row];
    if (descriptor_operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
      continue;
    }

    loom_low_packet_field_t field = {
        .field_name = loom_low_descriptor_set_string(
            descriptor_set, descriptor_operand->field_name_string_offset),
    };
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_operand_field(
        function_state, op, descriptor_operand->source_value_index, &field));
    const loom_low_register_part_mask_t required_mask =
        loom_low_verify_descriptor_operand_part_mask(
            function_state, descriptor_operand, field.type);
    if (required_mask == 0) {
      continue;
    }
    const loom_low_register_part_mask_t defined_mask =
        loom_low_verify_value_defined_mask(function_state, field.value_id);
    if ((defined_mask & required_mask) == required_mask) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_undefined_register_part(
        function_state, op, descriptor_key, field.field_ref, field.field_name,
        required_mask, defined_mask));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }

  if (has_variadic_operands) {
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  fixed_descriptor_operand_end];
    const iree_string_view_t field_name = loom_low_descriptor_set_string(
        descriptor_set, descriptor_operand->field_name_string_offset);
    for (uint16_t packet_operand_index =
             descriptor->minimum_packet_operand_count;
         packet_operand_index < op->operand_count; ++packet_operand_index) {
      loom_low_packet_field_t field = {.field_name = field_name};
      IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_operand_field(
          function_state, op, packet_operand_index, &field));
      const loom_low_register_part_mask_t required_mask =
          loom_low_verify_descriptor_operand_part_mask(
              function_state, descriptor_operand, field.type);
      if (required_mask == 0) {
        continue;
      }
      const loom_low_register_part_mask_t defined_mask =
          loom_low_verify_value_defined_mask(function_state, field.value_id);
      if ((defined_mask & required_mask) == required_mask) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_verify_emit_undefined_register_part(
          function_state, op, descriptor_key, field.field_ref, field.field_name,
          required_mask, defined_mask));
      if (loom_low_verify_should_stop(function_state->state)) {
        return iree_ok_status();
      }
    }
  }

  for (uint16_t i = 0; i < descriptor->result_count; ++i) {
    loom_low_packet_field_t field;
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_field(
        function_state, op, descriptor, i, &field));
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    loom_low_register_part_mask_t result_mask =
        loom_low_verify_descriptor_operand_part_mask(
            function_state, descriptor_operand, field.type);

    uint16_t tied_operand_index = LOOM_LOW_ID_NONE;
    if (loom_low_verify_descriptor_tied_operand_index(
            descriptor_set, descriptor, i, &tied_operand_index)) {
      loom_low_packet_field_t tied_field;
      IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_packet_field(
          function_state, op, descriptor, tied_operand_index, &tied_field));
      result_mask |= loom_low_verify_value_defined_mask(function_state,
                                                        tied_field.value_id);
    }
    loom_low_verify_set_value_defined_mask(function_state, field.value_id,
                                           result_mask);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_define_full_register_results(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op) {
  const loom_module_t* module = function_state->state->module;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t value_id = results[i];
    const loom_type_t type = loom_module_value_type(module, value_id);
    const loom_low_register_part_mask_t result_mask =
        loom_low_verify_register_full_mask_for_type(function_state, type);
    if (result_mask != 0) {
      loom_low_verify_set_value_defined_mask(function_state, value_id,
                                             result_mask);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_structural_register_parts(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op) {
  const loom_module_t* module = function_state->state->module;
  const iree_string_view_t op_name = loom_op_name(module, op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t value_id = operands[i];
    const loom_type_t type = loom_module_value_type(module, value_id);
    const loom_low_register_part_mask_t required_mask =
        loom_low_verify_register_full_mask_for_type(function_state, type);
    if (required_mask == 0) {
      continue;
    }
    const loom_low_register_part_mask_t defined_mask =
        loom_low_verify_value_defined_mask(function_state, value_id);
    if ((defined_mask & required_mask) == required_mask) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_undefined_register_part(
        function_state, op, op_name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i),
        IREE_SV("operand"), required_mask, defined_mask));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }

  return loom_low_verify_define_full_register_results(function_state, op);
}

static bool loom_low_verify_is_source_preserving_structural_op(
    const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op);
}

static iree_status_t loom_low_verify_reference_source_preserving_ops(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op) {
  if (!loom_low_verify_is_source_preserving_structural_op(op)) {
    return iree_ok_status();
  }
  const loom_module_t* module = function_state->state->module;
  const iree_string_view_t op_name = loom_op_name(module, op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t actual_type = loom_module_value_type(module, operands[i]);
    if (!loom_low_register_type_resolver_has_class_flags(
            &function_state->register_type_resolver, actual_type,
            LOOM_LOW_REG_CLASS_FLAG_REFERENCE)) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(function_state->function_name),
        loom_param_string(op_name),
        loom_param_with_field_ref(
            loom_param_type(actual_type),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i)),
    };
    IREE_RETURN_IF_ERROR(loom_low_verify_emit(function_state->state, op,
                                              LOOM_ERR_TARGET_071, params,
                                              IREE_ARRAYSIZE(params), NULL, 0));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_type_changing_tied_results(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t op_name, const loom_low_descriptor_packet_t* packet) {
  if (op->tied_result_count == 0) return iree_ok_status();

  const loom_module_t* module = function_state->state->module;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    if (!tied.has_type_change) {
      continue;
    }
    if (tied.result_index >= op->result_count ||
        tied.operand_index >= op->operand_count) {
      continue;
    }
    if (packet != NULL && loom_low_verify_descriptor_declares_tied_result(
                              descriptor_set, packet->descriptor, tied)) {
      continue;
    }

    const loom_type_t result_type =
        loom_module_value_type(module, results[tied.result_index]);
    const loom_type_t operand_type =
        loom_module_value_type(module, operands[tied.operand_index]);
    loom_diagnostic_param_t params[] = {
        loom_param_string(function_state->function_name),
        loom_param_string(op_name),
        loom_param_with_field_ref(
            loom_param_u32(tied.result_index),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT,
                                      tied.result_index)),
        loom_param_type(result_type),
        loom_param_with_field_ref(
            loom_param_u32(tied.operand_index),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                      tied.operand_index)),
        loom_param_type(operand_type),
    };
    IREE_RETURN_IF_ERROR(loom_low_verify_emit(
        function_state->state, op, LOOM_ERR_TARGET_057, params,
        IREE_ARRAYSIZE(params), /*related_ops=*/NULL,
        /*related_op_count=*/0));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_features(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t descriptor_key, uint16_t descriptor_attr_index,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_resolved_target_t* target = function_state->target;
  const loom_low_descriptor_set_t* descriptor_set = target->descriptor_set;
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t feature_mask_row = descriptor->feature_mask_word_start + i;
    const uint64_t required_bits =
        descriptor_set->feature_mask_words[feature_mask_row];
    const uint64_t available_bits = i == 0 ? target->feature_bits : 0;
    const uint64_t missing_bits = required_bits & ~available_bits;
    if (missing_bits == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_missing_features(
        function_state, op, descriptor_key, descriptor_attr_index, i,
        missing_bits));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_packet(
    loom_low_function_verify_state_t* function_state,
    const loom_low_descriptor_packet_t* packet) {
  const loom_op_t* op = packet->op;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(descriptor_set, packet);
  const uint16_t descriptor_attr_index =
      loom_low_descriptor_packet_attribute_index(packet);
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const iree_string_view_t actual_op_name =
      loom_op_name(function_state->state->module, op);
  const iree_string_view_t expected_op_name =
      loom_low_verify_descriptor_op_name(descriptor);
  if (!iree_string_view_equal(actual_op_name, expected_op_name)) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_descriptor_op_mismatch(
        function_state, op, descriptor_key, descriptor_attr_index,
        actual_op_name, expected_op_name));
    return loom_low_verify_define_full_register_results(function_state, op);
  }
  if (descriptor->result_count > descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor '%.*s' has %" PRIu16
                            " results but only %" PRIu16 " operand rows",
                            (int)descriptor_key.size, descriptor_key.data,
                            descriptor->result_count,
                            descriptor->operand_count);
  }

  const uint32_t expected_result_count = descriptor->result_count;
  const uint32_t minimum_operand_count =
      descriptor->minimum_packet_operand_count;
  const bool has_variadic_operands =
      loom_low_descriptor_has_variadic_operands(descriptor);
  const uint32_t packet_start_error_count =
      function_state->state->result->error_count;
  if (op->result_count != expected_result_count) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_count_mismatch(
        function_state, op, LOOM_ERR_STRUCTURE_002, descriptor_key,
        descriptor_attr_index, op->result_count, expected_result_count));
  }
  const bool operand_count_matches =
      has_variadic_operands ? op->operand_count >= minimum_operand_count
                            : op->operand_count == minimum_operand_count;
  if (!operand_count_matches) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_count_mismatch(
        function_state, op, LOOM_ERR_STRUCTURE_001, descriptor_key,
        descriptor_attr_index, op->operand_count, minimum_operand_count));
  }
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_immediates(
      function_state, op, descriptor_key, descriptor_attr_index, descriptor));
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  if (op->result_count == expected_result_count && operand_count_matches) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_descriptor_registers(function_state, op, descriptor));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_verify_descriptor_constraints(function_state, op, descriptor));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
    if (function_state->state->result->error_count ==
        packet_start_error_count) {
      IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_register_parts(
          function_state, op, descriptor_key, descriptor));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_verify_define_full_register_results(function_state, op));
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_define_full_register_results(function_state, op));
  }
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  return loom_low_verify_descriptor_features(function_state, op, descriptor_key,
                                             descriptor_attr_index, descriptor);
}

static iree_status_t loom_low_verify_control_descriptor(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  const uint16_t descriptor_attr_index =
      loom_low_descriptor_attribute_index_for_op(op);
  const iree_string_view_t actual_op_name =
      loom_op_name(function_state->state->module, op);
  const iree_string_view_t expected_op_name =
      loom_low_verify_descriptor_op_name(descriptor);
  if (!iree_string_view_equal(actual_op_name, expected_op_name)) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_descriptor_op_mismatch(
        function_state, op, descriptor_key, descriptor_attr_index,
        actual_op_name, expected_op_name));
    return loom_low_verify_structural_register_parts(function_state, op);
  }

  if (descriptor->carrier == LOOM_LOW_DESCRIPTOR_CARRIER_BRANCH) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_structural_register_parts(function_state, op));
  } else {
    IREE_ASSERT_EQ(descriptor->carrier, LOOM_LOW_DESCRIPTOR_CARRIER_SWITCH);
    IREE_RETURN_IF_ERROR(
        loom_low_verify_descriptor_registers(function_state, op, descriptor));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_verify_descriptor_constraints(function_state, op, descriptor));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_register_parts(
        function_state, op, descriptor_key, descriptor));
  }
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  return loom_low_verify_descriptor_features(function_state, op, descriptor_key,
                                             descriptor_attr_index, descriptor);
}

static iree_status_t loom_low_verify_begin_module_providers(
    loom_low_verify_state_t* state) {
  loom_low_verify_provider_list_t provider_list = state->provider_list;
  if (loom_low_verify_provider_list_is_empty(provider_list)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->arena, provider_list.count,
                                sizeof(*state->provider_module_states),
                                (void**)&state->provider_module_states));
  memset(state->provider_module_states, 0,
         provider_list.count * sizeof(*state->provider_module_states));
  loom_low_verify_module_context_t context = {
      .state = state,
  };
  for (iree_host_size_t i = 0; i < provider_list.count; ++i) {
    const loom_low_verify_provider_t* provider = provider_list.values[i];
    if (provider->begin_module != NULL) {
      IREE_RETURN_IF_ERROR(provider->begin_module(
          provider, &context, &state->provider_module_states[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_end_module_providers(
    loom_low_verify_state_t* state) {
  loom_low_verify_provider_list_t provider_list = state->provider_list;
  if (loom_low_verify_provider_list_is_empty(provider_list) ||
      state->provider_module_states == NULL ||
      loom_low_verify_should_stop(state)) {
    return iree_ok_status();
  }
  loom_low_verify_module_context_t context = {
      .state = state,
  };
  for (iree_host_size_t i = 0; i < provider_list.count; ++i) {
    const loom_low_verify_provider_t* provider = provider_list.values[i];
    if (provider->end_module != NULL) {
      IREE_RETURN_IF_ERROR(provider->end_module(
          provider, &context, state->provider_module_states[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_begin_function_providers(
    loom_low_function_verify_state_t* function_state) {
  loom_low_verify_provider_list_t provider_list =
      function_state->state->provider_list;
  if (loom_low_verify_provider_list_is_empty(provider_list)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &function_state->state->walk_arena, provider_list.count,
      sizeof(*function_state->provider_states),
      (void**)&function_state->provider_states));
  memset(function_state->provider_states, 0,
         provider_list.count * sizeof(*function_state->provider_states));
  loom_low_verify_context_t context = {
      .function_state = function_state,
  };
  for (iree_host_size_t i = 0; i < provider_list.count; ++i) {
    const loom_low_verify_provider_t* provider = provider_list.values[i];
    if (provider->begin_function != NULL) {
      context.provider_module_state =
          function_state->state->provider_module_states[i];
      IREE_RETURN_IF_ERROR(provider->begin_function(
          provider, &context, &function_state->provider_states[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_run_op_providers(
    loom_low_function_verify_state_t* function_state,
    const loom_low_descriptor_packet_t* packet) {
  loom_low_verify_provider_list_t provider_list =
      function_state->state->provider_list;
  if (loom_low_verify_provider_list_is_empty(provider_list) ||
      loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  loom_low_verify_context_t context = {
      .function_state = function_state,
  };
  for (iree_host_size_t i = 0; i < provider_list.count; ++i) {
    const loom_low_verify_provider_t* provider = provider_list.values[i];
    if (provider->verify_op != NULL) {
      context.provider_module_state =
          function_state->state->provider_module_states[i];
      IREE_RETURN_IF_ERROR(provider->verify_op(
          provider, &context, function_state->provider_states[i], packet));
    }
    if (loom_low_verify_should_stop(function_state->state)) {
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_end_function_providers(
    loom_low_function_verify_state_t* function_state) {
  loom_low_verify_provider_list_t provider_list =
      function_state->state->provider_list;
  if (loom_low_verify_provider_list_is_empty(provider_list)) {
    return iree_ok_status();
  }
  loom_low_verify_context_t context = {
      .function_state = function_state,
  };
  for (iree_host_size_t i = 0; i < provider_list.count; ++i) {
    const loom_low_verify_provider_t* provider = provider_list.values[i];
    if (provider->end_function != NULL) {
      context.provider_module_state =
          function_state->state->provider_module_states[i];
      IREE_RETURN_IF_ERROR(provider->end_function(
          provider, &context, function_state->provider_states[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_walk_op(void* user_data, loom_op_t* op,
                                             const loom_walk_context_t* context,
                                             loom_walk_result_t* out_result) {
  (void)context;
  loom_low_function_verify_state_t* function_state =
      (loom_low_function_verify_state_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (loom_low_verify_should_stop(function_state->state)) {
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }
  if (loom_low_storage_reserve_isa(op) && op->parent_block != NULL &&
      op->parent_block->parent_region == function_state->body) {
    IREE_RETURN_IF_ERROR(loom_low_storage_layout_accumulate_reservation(
        function_state->state->module, op,
        &function_state->storage_space_sizes));
  }

  const uint32_t descriptor_ordinal = loom_low_descriptor_ordinal_for_op(op);
  const loom_low_descriptor_t* control_descriptor = NULL;
  if (descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    if (descriptor_ordinal >=
        function_state->target->descriptor_set->descriptor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Low descriptor ordinal %" PRIu32
                              " is outside the active representation contract",
                              descriptor_ordinal);
    }
    if (loom_low_br_isa(op) || loom_low_switch_isa(op)) {
      control_descriptor = &function_state->target->descriptor_set
                                ->descriptors[descriptor_ordinal];
    }
  }

  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(function_state->target->descriptor_set,
                                        op, &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    const loom_module_t* module = function_state->state->module;
    IREE_RETURN_IF_ERROR(loom_low_verify_type_changing_tied_results(
        function_state, op, loom_op_name(module, op), /*packet=*/NULL));
    if (loom_low_verify_should_stop(function_state->state)) {
      *out_result = LOOM_WALK_ABORT;
      return iree_ok_status();
    }
    if (loom_low_resource_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_low_verify_resource(function_state, op));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_verify_reference_source_preserving_ops(function_state, op));
    if (control_descriptor != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_verify_control_descriptor(
          function_state, op, control_descriptor));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_verify_structural_register_parts(function_state, op));
    }
    if (loom_traits_are_compile_time_only(op->traits)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_verify_run_op_providers(function_state, &packet));
    if (loom_low_verify_should_stop(function_state->state)) {
      *out_result = LOOM_WALK_ABORT;
    }
    return iree_ok_status();
  }
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(
          function_state->target->descriptor_set, &packet);
  IREE_RETURN_IF_ERROR(loom_low_verify_type_changing_tied_results(
      function_state, op, descriptor_key, &packet));
  if (loom_low_verify_should_stop(function_state->state)) {
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_packet(function_state, &packet));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_run_op_providers(function_state, &packet));
  if (loom_low_verify_should_stop(function_state->state)) {
    *out_result = LOOM_WALK_ABORT;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_initialize_block_arg_masks(
    loom_low_function_verify_state_t* function_state, loom_region_t* region) {
  const loom_module_t* module = function_state->state->module;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      const loom_value_id_t value_id = block->arg_ids[i];
      const loom_type_t type = loom_module_value_type(module, value_id);
      const loom_low_register_part_mask_t mask =
          loom_low_verify_register_full_mask_for_type(function_state, type);
      if (mask != 0) {
        loom_low_verify_set_value_defined_mask(function_state, value_id, mask);
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] != NULL) {
          IREE_RETURN_IF_ERROR(loom_low_verify_initialize_block_arg_masks(
              function_state, regions[i]));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_function(loom_low_verify_state_t* state,
                                              const loom_op_t* low_func_op) {
  iree_diagnostic_emitter_t counting_emitter = {
      .fn = loom_low_verify_counting_emitter,
      .user_data = state,
  };
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      state->module, &state->symbol_facts, low_func_op,
      loom_low_verify_function_target_facts(state, low_func_op),
      state->registry, counting_emitter, &target));
  loom_region_t* body = loom_low_verify_function_body(low_func_op);
  if (target.descriptor_set == NULL || loom_low_verify_should_stop(state)) {
    return iree_ok_status();
  }

  loom_low_function_verify_state_t function_state = {
      .state = state,
      .target = &target,
      .function_op = low_func_op,
      .body = body,
      .function_name =
          loom_low_verify_function_name(state->module, low_func_op),
  };
  function_state.register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(target.descriptor_set);
  IREE_RETURN_IF_ERROR(loom_low_verify_function_register_values(
      &function_state, low_func_op, body));
  if (loom_low_verify_should_stop(state)) {
    return iree_ok_status();
  }
  if (body == NULL) {
    return iree_ok_status();
  }

  iree_status_t status = loom_local_value_domain_acquire_for_region_tree(
      state->module, body, &state->walk_arena, &function_state.value_domain);
  if (iree_status_is_ok(status) &&
      function_state.value_domain.value_count > 0) {
    status = iree_arena_allocate_array(&state->walk_arena,
                                       function_state.value_domain.value_count,
                                       sizeof(*function_state.defined_masks),
                                       (void**)&function_state.defined_masks);
    if (iree_status_is_ok(status)) {
      memset(function_state.defined_masks, 0,
             function_state.value_domain.value_count *
                 sizeof(*function_state.defined_masks));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_begin_function_providers(&function_state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_initialize_block_arg_masks(&function_state, body);
  }
  if (iree_status_is_ok(status)) {
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    status = loom_walk_region(
        state->module, body, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_low_verify_walk_op, &function_state},
        &state->walk_arena, &walk_result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_workgroup_storage_limit(&function_state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_end_function_providers(&function_state);
  }
  loom_local_value_domain_release(&function_state.value_domain);
  return status;
}

iree_status_t loom_low_verify_module(loom_module_t* module,
                                     const loom_low_verify_options_t* options,
                                     loom_low_verify_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_low_verify_result_t){0};
  loom_low_verify_state_t state = {
      .module = module,
      .registry = options->descriptor_registry,
      .emitter = options->emitter,
      .provider_list = options->provider_list,
      .result = out_result,
      .max_errors = options->max_errors,
  };
  iree_arena_initialize(module->arena.block_pool, &state.arena);
  iree_arena_initialize(module->arena.block_pool, &state.walk_arena);
  loom_symbol_fact_table_initialize(&state.symbol_facts, &state.arena);

  iree_status_t status = loom_target_function_version_snapshot_build(
      module, options->function_versions, &state.arena,
      &state.function_version_snapshot);
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_begin_module_providers(&state);
  }
  if (iree_status_is_ok(status) && module->body &&
      module->body->block_count > 0) {
    loom_block_t* entry_block = loom_region_entry_block(module->body);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(entry_block, op) {
      if (!loom_low_func_def_isa(op) && !loom_low_kernel_def_isa(op) &&
          !loom_low_func_decl_isa(op)) {
        continue;
      }
      status = loom_low_verify_function(&state, op);
      iree_arena_reset(&state.walk_arena);
      if (!iree_status_is_ok(status) || loom_low_verify_should_stop(&state)) {
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_verify_end_module_providers(&state);
  }

  iree_arena_deinitialize(&state.walk_arena);
  iree_arena_deinitialize(&state.arena);
  return status;
}
