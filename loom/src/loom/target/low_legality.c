// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_legality.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/view_regions.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/target/registers.h"
#include "loom/util/walk.h"

typedef uint8_t loom_target_low_legality_t;

enum loom_target_low_legality_e {
  LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED = 0,
  LOOM_TARGET_LOW_LEGALITY_CORE = 1,
  LOOM_TARGET_LOW_LEGALITY_PROVIDER = 2,
  LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY = 3,
  LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA = 4,
  LOOM_TARGET_LOW_LEGALITY_COMPILE_TIME_QUERY = 5,
};

typedef struct loom_target_low_legality_target_state_record_t {
  // Target-owned static key identifying this function-local state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the legality arena.
  void* data;
} loom_target_low_legality_target_state_record_t;

struct loom_target_low_legality_context_t {
  // Source module being checked.
  const loom_module_t* module;
  // Source function being checked.
  loom_func_like_t function;
  // Caller-owned verification options.
  const loom_target_low_legality_options_t* options;
  // Descriptor set selected by options.target_facts.
  const loom_low_descriptor_set_t* descriptor_set;
  // Function-local target state records populated by legality providers.
  loom_target_low_legality_target_state_record_t* target_state_records;
  // Number of populated target_state_records entries.
  iree_host_size_t target_state_record_count;
  // Number of allocated target_state_records entries.
  iree_host_size_t target_state_record_capacity;
  // Result object receiving counters and selected descriptor set.
  loom_target_low_legality_result_t* result;
  // Scratch arena for the IR walker.
  iree_arena_allocator_t arena;
};

static const loom_target_bundle_t* loom_target_low_legality_options_bundle(
    const loom_target_low_legality_options_t* options) {
  return loom_target_facts_bundle(options->target_facts);
}

static iree_string_view_t loom_target_low_legality_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_target_low_legality_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

iree_string_view_t loom_target_low_legality_function_name(
    const loom_target_low_legality_context_t* context) {
  if (!loom_func_like_isa(context->function)) {
    return IREE_SV("<module>");
  }
  return loom_target_low_legality_symbol_name(
      context->module, loom_func_like_callee(context->function));
}

static iree_string_view_t loom_target_low_legality_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->export_plan->name,
                                           IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->config->name,
                                           IREE_SV("<empty>"));
}

static bool loom_target_low_legality_should_stop(
    const loom_target_low_legality_context_t* context) {
  return context->options->max_errors != 0 &&
         context->result->error_count >= context->options->max_errors;
}

static iree_status_t loom_target_low_legality_emit(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_ERROR) {
    if (loom_target_low_legality_should_stop(context)) {
      return iree_ok_status();
    }
    ++context->result->error_count;
  } else if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_REMARK) {
    ++context->result->remark_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(context->options->emitter, &emission);
}

iree_status_t loom_target_low_legality_emit_error_ref(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_error_ref_t error_ref, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_error_def_t* error =
      loom_error_catalog_lookup_ref(context->options->error_catalog, error_ref);
  IREE_ASSERT(error != NULL);
  return loom_target_low_legality_emit(context, op, error, params, param_count);
}

#define LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT 5

static void loom_target_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle =
      loom_target_low_legality_options_bundle(context->options);
  params[0] = loom_param_string(loom_target_low_legality_target_key(bundle));
  params[1] = loom_param_string(loom_target_low_legality_export_name(bundle));
  params[2] = loom_param_string(loom_target_low_legality_config_key(bundle));
  params[3] =
      loom_param_string(loom_target_low_legality_function_name(context));
  params[4] = loom_param_string(loom_op_name(context->module, op));
}

static iree_status_t loom_target_low_legality_emit_target_context_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* extra_params,
    iree_host_size_t extra_param_count) {
  IREE_ASSERT_LE(extra_param_count, 4);
  loom_diagnostic_param_t
      params[LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 4];
  loom_target_low_legality_make_context_params(context, op, params);
  for (iree_host_size_t i = 0; i < extra_param_count; ++i) {
    params[LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + i] = extra_params[i];
  }
  return loom_target_low_legality_emit(
      context, op, error, params,
      LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + extra_param_count);
}

static iree_status_t loom_target_low_legality_emit_no_target_contract(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  return loom_target_low_legality_emit_target_context_error(
      context, op, LOOM_ERR_TARGET_001, /*extra_params=*/NULL,
      /*extra_param_count=*/0);
}

static iree_status_t loom_target_low_legality_emit_unresolved_query(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  return loom_target_low_legality_emit_target_context_error(
      context, op, LOOM_ERR_TARGET_070, /*extra_params=*/NULL,
      /*extra_param_count=*/0);
}

static iree_status_t loom_target_low_legality_emit_type_constraint(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type, iree_string_view_t type_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(type),
      loom_param_string(type_constraint),
  };
  return loom_target_low_legality_emit_target_context_error(
      context, op, LOOM_ERR_TARGET_031, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_low_legality_emit_op_constraint(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t op_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(op_constraint),
  };
  return loom_target_low_legality_emit_target_context_error(
      context, op, LOOM_ERR_TARGET_032, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_low_legality_reject_error_ref(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_target_contract_rejection_t* rejection) {
  const loom_error_def_t* error = loom_error_catalog_lookup_ref(
      context->options->error_catalog, rejection->error_ref);
  IREE_ASSERT(error != NULL);
  return loom_target_low_legality_emit(context, op, error, rejection->params,
                                       rejection->param_count);
}

iree_status_t loom_target_low_legality_record_memory_access(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t operation_kind,
    iree_string_view_t packet_key, iree_string_view_t address_form,
    iree_string_view_t dynamic_term_kind, iree_string_view_t fallback_reason,
    iree_string_view_t decision, int64_t static_offset_bytes,
    uint32_t element_bytes, uint32_t vector_lanes,
    uint32_t dynamic_stride_bytes, uint32_t vector_lane_stride_bytes) {
  const loom_target_bundle_t* bundle =
      loom_target_low_legality_options_bundle(context->options);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_target_low_legality_target_key(bundle)),
      loom_param_string(loom_target_low_legality_export_name(bundle)),
      loom_param_string(loom_target_low_legality_config_key(bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(memory_space),
      loom_param_string(operation_kind),
      loom_param_string(packet_key),
      loom_param_string(address_form),
      loom_param_string(dynamic_term_kind),
      loom_param_string(fallback_reason),
      loom_param_string(decision),
      loom_param_i64(static_offset_bytes),
      loom_param_u32(element_bytes),
      loom_param_u32(vector_lanes),
      loom_param_u32(dynamic_stride_bytes),
      loom_param_u32(vector_lane_stride_bytes),
  };
  return loom_target_low_legality_emit(context, op, LOOM_ERR_BACKEND_017,
                                       params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_target_low_legality_record_memory_cache_policy(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t operation_kind,
    iree_string_view_t cache_scope, iree_string_view_t cache_temporal,
    iree_string_view_t decision_key, iree_string_view_t decision,
    iree_string_view_t encoding_key, bool scope_attr_present,
    int64_t scope_attr, bool th_attr_present, int64_t th_attr,
    bool nt_attr_present, int64_t nt_attr) {
  const loom_target_bundle_t* bundle =
      loom_target_low_legality_options_bundle(context->options);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_target_low_legality_target_key(bundle)),
      loom_param_string(loom_target_low_legality_export_name(bundle)),
      loom_param_string(loom_target_low_legality_config_key(bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(memory_space),
      loom_param_string(operation_kind),
      loom_param_string(cache_scope),
      loom_param_string(cache_temporal),
      loom_param_string(decision_key),
      loom_param_string(decision),
      loom_param_string(encoding_key),
      loom_param_bool(scope_attr_present),
      loom_param_i64(scope_attr),
      loom_param_bool(th_attr_present),
      loom_param_i64(th_attr),
      loom_param_bool(nt_attr_present),
      loom_param_i64(nt_attr),
  };
  return loom_target_low_legality_emit(context, op, LOOM_ERR_BACKEND_040,
                                       params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_target_low_legality_record_memory_prefetch(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t prefetch_intent,
    iree_string_view_t prefetch_locality, iree_string_view_t decision_key,
    iree_string_view_t decision, iree_string_view_t packet_key,
    int64_t immediate_offset, uint32_t scalar_byte_offset,
    iree_string_view_t dynamic_index_kind, uint32_t count) {
  const loom_target_bundle_t* bundle =
      loom_target_low_legality_options_bundle(context->options);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_target_low_legality_target_key(bundle)),
      loom_param_string(loom_target_low_legality_export_name(bundle)),
      loom_param_string(loom_target_low_legality_config_key(bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(memory_space),
      loom_param_string(prefetch_intent),
      loom_param_string(prefetch_locality),
      loom_param_string(decision_key),
      loom_param_string(decision),
      loom_param_string(packet_key),
      loom_param_i64(immediate_offset),
      loom_param_u32(scalar_byte_offset),
      loom_param_string(dynamic_index_kind),
      loom_param_u32(count),
  };
  return loom_target_low_legality_emit(context, op, LOOM_ERR_BACKEND_041,
                                       params, IREE_ARRAYSIZE(params));
}

const loom_module_t* loom_target_low_legality_module(
    const loom_target_low_legality_context_t* context) {
  return context->module;
}

loom_func_like_t loom_target_low_legality_function(
    const loom_target_low_legality_context_t* context) {
  return context->function;
}

const loom_target_bundle_t* loom_target_low_legality_bundle(
    const loom_target_low_legality_context_t* context) {
  return loom_target_low_legality_options_bundle(context->options);
}

const loom_target_facts_t* loom_target_low_legality_target_facts(
    const loom_target_low_legality_context_t* context) {
  return context->options->target_facts;
}

const loom_low_descriptor_set_t* loom_target_low_legality_descriptor_set(
    const loom_target_low_legality_context_t* context) {
  return context->descriptor_set;
}

const loom_value_fact_table_t* loom_target_low_legality_fact_table(
    const loom_target_low_legality_context_t* context) {
  return context->options->view_regions->expression_context->fact_table;
}

const loom_local_value_domain_t* loom_target_low_legality_value_domain(
    const loom_target_low_legality_context_t* context) {
  return context->options->view_regions->value_domain;
}

const loom_view_region_table_t* loom_target_low_legality_view_regions(
    const loom_target_low_legality_context_t* context) {
  return context->options->view_regions;
}

iree_arena_allocator_t* loom_target_low_legality_scratch_arena(
    loom_target_low_legality_context_t* context) {
  return &context->arena;
}

iree_status_t loom_target_low_legality_get_or_allocate_target_state(
    loom_target_low_legality_context_t* context, const void* key,
    iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT(key != NULL);
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  for (iree_host_size_t i = 0; i < context->target_state_record_count; ++i) {
    loom_target_low_legality_target_state_record_t* record =
        &context->target_state_records[i];
    if (record->key != key) continue;
    IREE_ASSERT_EQ(record->data_length, data_length);
    *out_data = record->data;
    return iree_ok_status();
  }

  if (context->target_state_record_count ==
      context->target_state_record_capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(context->target_state_record_count, 1,
                                    &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &context->arena, context->target_state_record_count, minimum_capacity,
        sizeof(*context->target_state_records),
        &context->target_state_record_capacity,
        (void**)&context->target_state_records));
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, 1, data_length, &data));
  memset(data, 0, data_length);
  const iree_host_size_t record_index = context->target_state_record_count++;
  loom_target_low_legality_target_state_record_t* record =
      &context->target_state_records[record_index];
  *record = (loom_target_low_legality_target_state_record_t){
      .key = key,
      .data_length = data_length,
      .data = data,
  };
  *out_data = data;
  return iree_ok_status();
}

loom_target_low_legality_diagnostic_flags_t
loom_target_low_legality_diagnostic_flags(
    const loom_target_low_legality_context_t* context) {
  return context->options->diagnostic_flags;
}

static bool loom_target_low_legality_codegen_format_is_low(
    loom_target_codegen_format_t codegen_format) {
  switch (codegen_format) {
    case LOOM_TARGET_CODEGEN_FORMAT_LLVMIR:
    case LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE:
    case LOOM_TARGET_CODEGEN_FORMAT_SPIRV:
    case LOOM_TARGET_CODEGEN_FORMAT_VM:
    case LOOM_TARGET_CODEGEN_FORMAT_WASM:
      return true;
    default:
      return false;
  }
}

static bool loom_target_low_legality_abi_is_low(
    loom_target_abi_kind_t abi_kind) {
  switch (abi_kind) {
    // Module-internal functions intentionally have no artifact ABI.
    case LOOM_TARGET_ABI_UNKNOWN:
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
    case LOOM_TARGET_ABI_HAL_KERNEL:
    case LOOM_TARGET_ABI_SHADER_ENTRY_POINT:
    case LOOM_TARGET_ABI_VM_FUNCTION:
    case LOOM_TARGET_ABI_WASM_FUNCTION:
    case LOOM_TARGET_ABI_COMMAND_PROGRAM:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_low_legality_validate_options(
    const loom_target_low_legality_options_t* options,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (options->view_regions == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality requires function analysis");
  }
  const loom_target_bundle_t* bundle =
      loom_target_low_legality_options_bundle(options);
  if (!loom_target_low_legality_codegen_format_is_low(
          bundle->snapshot->codegen_format)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' codegen format is not target-low",
        (int)bundle->name.size, bundle->name.data);
  }
  if (!loom_target_low_legality_abi_is_low(bundle->export_plan->abi_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' ABI is not accepted by target-low legality",
        (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->snapshot->default_pointer_bitwidth == 0 ||
      bundle->snapshot->index_bitwidth == 0 ||
      bundle->snapshot->offset_bitwidth == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' pointer, index, and offset bit widths must be "
        "non-zero",
        (int)bundle->name.size, bundle->name.data);
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       ~LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality diagnostics use unknown "
                            "flag bits 0x%08x",
                            (unsigned)options->diagnostic_flags);
  }
  return loom_target_low_descriptor_set_select_for_source_lowering(
      options->descriptor_registry, bundle, out_descriptor_set);
}

static iree_status_t loom_target_low_legality_verify_scalar_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type) {
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      if (context->options->type_supported.fn != NULL &&
          context->options->type_supported.fn(
              context->options->type_supported.user_data, context->module,
              type)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("scalar.fp8_decode_or_contract"));
    case LOOM_SCALAR_TYPE_COUNT_:
      break;
  }
  return loom_target_low_legality_emit_type_constraint(context, op, type,
                                                       IREE_SV("scalar.known"));
}

static const loom_type_descriptor_t*
loom_target_low_legality_resolve_dialect_type(const loom_module_t* module,
                                              loom_type_t type) {
  if (!loom_type_is_dialect(type)) {
    return NULL;
  }
  loom_string_id_t name_id = loom_type_dialect_name_id(type);
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return NULL;
  }
  iree_string_view_t name = module->strings.entries[name_id];
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_lookup(module->context, name);
  if (descriptor == NULL ||
      descriptor->param_count != loom_type_dialect_param_count(type)) {
    return NULL;
  }
  return descriptor;
}

static bool loom_target_low_legality_op_accepts_type_contract(
    const loom_module_t* module, const loom_op_t* op,
    const loom_type_descriptor_t* descriptor) {
  loom_op_semantics_t op_semantics = loom_op_semantics(module, op);
  return loom_contract_family_set_has_any(
      op_semantics.contract_families, descriptor->semantics.contract_families);
}

static iree_status_t loom_target_low_legality_verify_registered_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type, bool* out_handled) {
  *out_handled = false;
  const loom_type_descriptor_t* descriptor =
      loom_target_low_legality_resolve_dialect_type(context->module, type);
  if (descriptor == NULL ||
      descriptor->semantics.semantic == LOOM_TYPE_SEMANTIC_ORDINARY) {
    return iree_ok_status();
  }
  *out_handled = true;
  if (descriptor->semantics.contract_families != 0 &&
      loom_target_low_legality_op_accepts_type_contract(context->module, op,
                                                        descriptor)) {
    return iree_ok_status();
  }
  switch (descriptor->semantics.semantic) {
    case LOOM_TYPE_SEMANTIC_CONTROL_TOKEN:
    case LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE:
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("type.matching_contract_family"));
    case LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE:
      if (context->options->type_supported.fn != NULL &&
          context->options->type_supported.fn(
              context->options->type_supported.user_data, context->module,
              type)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("type.target_low_mapping"));
    default:
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("type.target_low_mapping"));
  }
}

static iree_status_t loom_target_low_legality_verify_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type) {
  if (loom_type_is_scalar(type)) {
    return loom_target_low_legality_verify_scalar_type(context, op, type);
  }
  if (loom_type_is_buffer(type) || loom_type_is_view(type) ||
      loom_low_type_is_register(type) || loom_type_is_encoding(type)) {
    return iree_ok_status();
  }
  bool registered_type_handled = false;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_registered_type(
      context, op, type, &registered_type_handled));
  if (registered_type_handled) {
    return iree_ok_status();
  }
  if (context->options->type_supported.fn != NULL &&
      context->options->type_supported.fn(
          context->options->type_supported.user_data, context->module, type)) {
    return iree_ok_status();
  }
  if (loom_type_is_vector(type)) {
    if (!loom_type_is_all_static(type)) {
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("vector.static"));
    }
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count > UINT32_MAX) {
      return loom_target_low_legality_emit_type_constraint(
          context, op, type, IREE_SV("vector.lane_count_u32"));
    }
    return loom_target_low_legality_verify_scalar_type(
        context, op, loom_type_scalar(loom_type_element_type(type)));
  }
  return loom_target_low_legality_emit_type_constraint(
      context, op, type, IREE_SV("type.target_low_mapping"));
}

static iree_status_t loom_target_low_legality_verify_value(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t value_id) {
  const loom_type_t type = loom_module_value_type(context->module, value_id);
  return loom_target_low_legality_verify_type(context, op, type);
}

static iree_status_t loom_target_low_legality_verify_op_value_types(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_low_legality_verify_value(context, op, operands[i]));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_low_legality_verify_value(context, op, results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_try_provider_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  const uint8_t dialect_id = loom_op_dialect_id(op->kind);
  for (iree_host_size_t i = 0; i < context->options->provider_list.count; ++i) {
    const loom_target_low_legality_provider_t* provider =
        context->options->provider_list.values[i];
    if (!loom_target_low_legality_builtin_dialect_bits_contain(
            provider->builtin_dialect_bits, dialect_id)) {
      continue;
    }
    bool handled = false;
    IREE_RETURN_IF_ERROR(
        provider->try_verify_op(provider, context, op, &handled));
    if (handled) {
      *out_handled = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_reject_contract_query(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    return loom_target_low_legality_reject_error_ref(context, op,
                                                     result->rejection);
  }
  return loom_target_low_legality_emit_no_target_contract(context, op);
}

static iree_status_t
loom_target_low_legality_contract_query_get_or_allocate_target_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data) {
  return loom_target_low_legality_get_or_allocate_target_state(
      (loom_target_low_legality_context_t*)user_data, key, data_length,
      out_data);
}

static iree_status_t loom_target_low_legality_try_contract_query_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (loom_target_contract_query_callback_is_empty(
          context->options->contract_query)) {
    return iree_ok_status();
  }

  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .function = context->function,
      .target_facts = context->options->target_facts,
      .descriptor_set = context->descriptor_set,
      .fact_table = loom_target_low_legality_fact_table(context),
      .value_domain = loom_target_low_legality_value_domain(context),
      .view_regions = loom_target_low_legality_view_regions(context),
      .arena = &context->arena,
      .target_state_allocator =
          {
              .fn =
                  loom_target_low_legality_contract_query_get_or_allocate_target_state,
              .user_data = context,
          },
  };
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(context->options->contract_query.fn(
      context->options->contract_query.user_data, &environment, op, &result));
  switch (result.outcome) {
    case LOOM_TARGET_CONTRACT_QUERY_UNHANDLED:
      return iree_ok_status();
    case LOOM_TARGET_CONTRACT_QUERY_LEGAL:
      *out_handled = true;
      return iree_ok_status();
    case LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED: {
      bool provider_handled = false;
      IREE_RETURN_IF_ERROR(loom_target_low_legality_try_provider_op(
          context, op, &provider_handled));
      if (provider_handled) {
        *out_handled = true;
        return iree_ok_status();
      }
      *out_handled = true;
      return loom_target_low_legality_reject_contract_query(context, op,
                                                            &result);
    }
    case LOOM_TARGET_CONTRACT_QUERY_INVALID_IR:
      *out_handled = true;
      return loom_target_low_legality_reject_contract_query(context, op,
                                                            &result);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "target contract query returned unknown outcome "
                              "%d",
                              (int)result.outcome);
  }
}

static iree_status_t loom_target_low_legality_reject_source_only_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const bool allow_source_scf =
      iree_any_bit_set(context->options->structural_legality_flags,
                       LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF);
  switch (op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      if (allow_source_scf) {
        return iree_ok_status();
      }
      return loom_target_low_legality_emit_op_constraint(
          context, op, IREE_SV("source_structure.lower_to_cfg"));
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      if (allow_source_scf) {
        return iree_ok_status();
      }
      return loom_target_low_legality_emit_op_constraint(
          context, op, IREE_SV("source_structure.parent_lowering"));
    default:
      return loom_target_low_legality_emit_op_constraint(
          context, op, IREE_SV("source_phase.lower_before_target_low"));
  }
}

static iree_string_view_t loom_target_low_legality_contract_family_constraint(
    loom_contract_family_set_t contract_families) {
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_VECTOR_CONTRACTION)) {
    return IREE_SV("target_contract.vector_contraction.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_VECTOR_TABLE_LOOKUP)) {
    return IREE_SV("target_contract.vector_table_lookup.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_REGISTER_PERMUTATION)) {
    return IREE_SV("target_contract.register_permutation.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_VECTOR_COORDINATE)) {
    return IREE_SV("target_contract.vector_coordinate.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_MEMORY_ATOMIC)) {
    return IREE_SV("target_contract.memory_atomic.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_KERNEL_ASYNC)) {
    return IREE_SV("target_contract.kernel_async.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_KERNEL_SYNCHRONIZATION)) {
    return IREE_SV("target_contract.kernel_synchronization.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_TENSOR_MEMORY)) {
    return IREE_SV("target_contract.tensor_memory.provider_required");
  }
  if (loom_contract_family_set_has_any(contract_families,
                                       LOOM_CONTRACT_SANITIZER_RACE)) {
    return IREE_SV("target_contract.sanitizer_race.provider_required");
  }
  return IREE_SV("target_contract.provider_required");
}

static iree_status_t loom_target_low_legality_verify_op_class(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, op);
  if (loom_traits_are_fact_identity(traits)) {
    return iree_ok_status();
  }
  if (loom_traits_are_value_alias(traits)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_CFG_SWITCH:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_CALL_INDIRECT:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
      return iree_ok_status();
    default:
      break;
  }
  loom_op_semantics_t semantics = loom_op_semantics(context->module, op);
  loom_target_low_legality_t legality = LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED;
  if (semantics.contract_families != 0) {
    legality = LOOM_TARGET_LOW_LEGALITY_PROVIDER;
  } else {
    switch (semantics.phase) {
      case LOOM_OP_PHASE_EXECUTABLE:
        legality = LOOM_TARGET_LOW_LEGALITY_CORE;
        break;
      case LOOM_OP_PHASE_SOURCE_STRUCTURE:
        legality = LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY;
        break;
      case LOOM_OP_PHASE_MODULE_METADATA:
        legality = LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA;
        break;
      case LOOM_OP_PHASE_COMPILE_TIME_QUERY:
        legality = LOOM_TARGET_LOW_LEGALITY_COMPILE_TIME_QUERY;
        break;
      case LOOM_OP_PHASE_UNSPECIFIED:
      default:
        legality = LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED;
        break;
    }
  }
  switch (legality) {
    case LOOM_TARGET_LOW_LEGALITY_CORE:
      return iree_ok_status();
    case LOOM_TARGET_LOW_LEGALITY_PROVIDER:
      return loom_target_low_legality_emit_op_constraint(
          context, op,
          loom_target_low_legality_contract_family_constraint(
              semantics.contract_families));
    case LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY:
      return loom_target_low_legality_reject_source_only_op(context, op);
    case LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA:
      return loom_target_low_legality_emit_op_constraint(
          context, op, IREE_SV("module_metadata.outside_executable_region"));
    case LOOM_TARGET_LOW_LEGALITY_COMPILE_TIME_QUERY:
      return loom_target_low_legality_emit_unresolved_query(context, op);
    case LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED:
      return loom_target_low_legality_emit_no_target_contract(context, op);
    default: {
      iree_string_view_t op_name = loom_op_name(context->module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "op '%.*s' has unknown target-low legality class %u",
          (int)op_name.size, op_name.data, (unsigned)legality);
    }
  }
}

static iree_status_t loom_target_low_legality_verify_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  // Target contracts and providers own their shape-specific legality. Let them
  // emit precise diagnostics before the generic executable type gate runs.
  bool contract_handled = false;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_try_contract_query_op(
      context, op, &contract_handled));
  if (contract_handled) {
    return iree_ok_status();
  }

  bool provider_handled = false;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_try_provider_op(context, op, &provider_handled));
  if (provider_handled) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_verify_op_value_types(context, op));

  return loom_target_low_legality_verify_op_class(context, op);
}

static bool loom_target_low_legality_skip_children_after_rejection(
    loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_low_legality_walk_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  loom_target_low_legality_context_t* context =
      (loom_target_low_legality_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_op(context, op));
  if (loom_target_low_legality_should_stop(context)) {
    *out_result = LOOM_WALK_ABORT;
  } else if (context->result->error_count != previous_error_count &&
             loom_target_low_legality_skip_children_after_rejection(op->kind)) {
    *out_result = LOOM_WALK_SKIP;
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_verify_function_signature(
    loom_target_low_legality_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(context->function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_value(
        context, context->function.op, argument_ids[i]));
  }
  const loom_value_id_t* result_ids =
      loom_op_const_results(context->function.op);
  for (uint16_t i = 0; i < context->function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_value(
        context, context->function.op, result_ids[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_target_low_verify_function_legality(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_low_legality_options_t* options,
    loom_target_low_legality_result_t* out_result) {
  *out_result = (loom_target_low_legality_result_t){0};
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_validate_options(options, &descriptor_set));
  out_result->descriptor_set = descriptor_set;

  loom_target_low_legality_context_t context = {
      .module = module,
      .function = function,
      .options = options,
      .descriptor_set = descriptor_set,
      .result = out_result,
  };
  iree_arena_initialize(module->arena.block_pool, &context.arena);

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = loom_target_low_legality_verify_function_signature(&context);
  }
  loom_region_t* body = loom_func_like_body(function);
  if (iree_status_is_ok(status) && body) {
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    status = loom_walk_region(
        module, body, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_target_low_legality_walk_op, &context},
        &context.arena, &walk_result);
  }

  iree_arena_deinitialize(&context.arena);
  return status;
}
