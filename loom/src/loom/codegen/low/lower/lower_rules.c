// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_rules.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower_rule_descriptor.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/lower_rule_value.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"

struct loom_low_lower_rule_descriptor_map_t {
  // Rule set whose local descriptor refs are resolved by descriptors.
  const loom_low_lower_rule_set_t* rule_set;
  // Descriptor rows indexed by rule-set-local descriptor ref.
  const loom_low_descriptor_t* const* descriptors;
  // Number of entries in descriptors.
  uint16_t descriptor_count;
};

iree_string_view_t loom_low_lower_rule_set_string(
    const loom_low_lower_rule_set_t* rule_set,
    loom_bstring_table_offset_t string_offset) {
  return loom_bstring_view(
      loom_bstring_table_get(&rule_set->string_table, string_offset));
}

static const loom_low_lower_rule_span_t* loom_low_lower_rule_set_find_span(
    const loom_low_lower_rule_set_t* rule_set, loom_op_kind_t source_op_kind) {
  uint16_t low = 0;
  uint16_t high = rule_set->span_count;
  while (low < high) {
    uint16_t mid = low + (uint16_t)((high - low) / 2);
    const loom_low_lower_rule_span_t* span = &rule_set->spans[mid];
    if (span->source_op_kind == source_op_kind) {
      return span;
    }
    if (span->source_op_kind < source_op_kind) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_emit_no_mapping(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_string_view_t loom_low_lower_rule_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_rule_symbol_name(
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

static iree_string_view_t loom_low_lower_rule_function_name(
    const loom_low_lower_rule_match_context_t* match_context) {
  if (!loom_func_like_isa(match_context->function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_rule_symbol_name(
      match_context->module, loom_func_like_callee(match_context->function));
}

static iree_string_view_t loom_low_lower_rule_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_rule_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->export_plan->name,
                                      IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_rule_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

iree_status_t loom_low_lower_rule_resolve_descriptor_ref(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_ref, rule_set->descriptor_ref_count);
  if (match_context->descriptor_ref.fn != NULL) {
    return match_context->descriptor_ref.fn(
        match_context->descriptor_ref.user_data, match_context, rule_set,
        descriptor_ref, out_descriptor);
  }
  IREE_ASSERT(match_context->descriptor_set != NULL);
  IREE_ASSERT(rule_set->descriptor_refs != NULL);
  const iree_string_view_t key = loom_low_lower_rule_set_string(
      rule_set, rule_set->descriptor_refs[descriptor_ref].key_string_offset);
  const uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor(
      match_context->descriptor_set, key);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }
  *out_descriptor = loom_low_descriptor_set_descriptor_at(
      match_context->descriptor_set, descriptor_ordinal);
  IREE_ASSERT(*out_descriptor != NULL);
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  const uint16_t rule_start = span ? span->rule_start : 0;
  const uint16_t rule_count = span ? span->rule_count : 0;
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

static iree_status_t loom_low_lower_rule_match_map_value_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, &low_type));
  if (!loom_low_type_is_register(low_type)) {
    return iree_ok_status();
  }
  *out_mapped_value = (loom_low_lower_rule_mapped_value_t){
      .is_register = true,
      .descriptor_register_class_id = loom_low_register_type_class_id(low_type),
      .register_unit_count = loom_low_register_type_unit_count(low_type),
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_match_can_materialize_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id,
    bool* out_can_materialize) {
  (void)match_context;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const loom_low_lower_value_materializer_t* materializer =
      loom_low_lower_rule_value_materializer(rule_set, value_ref);
  return materializer->can_materialize(context, source_op, source_value_id,
                                       out_can_materialize);
}

static const loom_low_lower_rule_descriptor_map_t*
loom_low_lower_rule_descriptor_map_find(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set) {
  for (uint16_t i = 0; i < context->lowering.rule_descriptor_map_count; ++i) {
    const loom_low_lower_rule_descriptor_map_t* map =
        &context->lowering.rule_descriptor_maps[i];
    if (map->rule_set == rule_set) {
      return map;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_descriptor_maps_initialize(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set) {
  IREE_ASSERT(descriptor_set != NULL);
  if (context->lowering.rule_descriptor_map_set == descriptor_set)
    return iree_ok_status();

  context->lowering.rule_descriptor_map_set = descriptor_set;
  context->lowering.rule_descriptor_maps = NULL;
  context->lowering.rule_descriptor_map_count = 0;

  const loom_low_lower_rule_set_list_t rule_sets =
      context->contract_set->rule_sets;
  if (rule_sets.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, rule_sets.count,
      sizeof(*context->lowering.rule_descriptor_maps),
      (void**)&context->lowering.rule_descriptor_maps));
  context->lowering.rule_descriptor_map_count = rule_sets.count;

  for (uint16_t i = 0; i < rule_sets.count; ++i) {
    const loom_low_lower_rule_set_t* rule_set = rule_sets.values[i];
    loom_low_lower_rule_descriptor_map_t* map =
        &context->lowering.rule_descriptor_maps[i];
    *map = (loom_low_lower_rule_descriptor_map_t){
        .rule_set = rule_set,
        .descriptors = NULL,
        .descriptor_count = rule_set->descriptor_ref_count,
    };
    if (rule_set->descriptor_ref_count == 0) {
      continue;
    }
    IREE_ASSERT(rule_set->descriptor_refs != NULL);
    const loom_low_descriptor_t** descriptors = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->function_arena, rule_set->descriptor_ref_count,
        sizeof(*descriptors), (void**)&descriptors));
    map->descriptors = descriptors;
    for (uint16_t j = 0; j < rule_set->descriptor_ref_count; ++j) {
      descriptors[j] = NULL;
      const iree_string_view_t key = loom_low_lower_rule_set_string(
          rule_set, rule_set->descriptor_refs[j].key_string_offset);
      const uint32_t descriptor_ordinal =
          loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      descriptors[j] = loom_low_descriptor_set_descriptor_at(
          descriptor_set, descriptor_ordinal);
      IREE_ASSERT(descriptors[j] != NULL);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_match_descriptor_ref_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_maps_initialize(
      context, match_context->descriptor_set));
  const loom_low_lower_rule_descriptor_map_t* map = NULL;
  if (match_context->policy_rule_set_ordinal == 0) {
    map = loom_low_lower_rule_descriptor_map_find(context, rule_set);
  } else {
    const uint16_t rule_set_index =
        (uint16_t)(match_context->policy_rule_set_ordinal - 1u);
    IREE_ASSERT_LT(rule_set_index, context->lowering.rule_descriptor_map_count);
    map = &context->lowering.rule_descriptor_maps[rule_set_index];
  }
  IREE_ASSERT(map != NULL);
  IREE_ASSERT_EQ(map->rule_set, rule_set);
  IREE_ASSERT_LT(descriptor_ref, map->descriptor_count);
  *out_descriptor = map->descriptors[descriptor_ref];
  return iree_ok_status();
}

void loom_low_lower_rule_match_context_initialize_from_lowering(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t* view_regions,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    loom_low_lower_rule_match_context_t* out_match_context) {
  *out_match_context = (loom_low_lower_rule_match_context_t){
      .module = loom_low_lower_context_module(context),
      .function = loom_low_lower_context_source_function(context),
      .bundle = loom_low_lower_context_bundle(context),
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .feature_bits =
          loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      .map_value =
          {
              .fn = loom_low_lower_rule_match_map_value_from_lowering,
              .user_data = context,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_rule_match_can_materialize_from_lowering,
              .user_data = context,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .fact_table = loom_low_lower_context_fact_table(context),
      .view_regions = view_regions,
      .source_memory_state = source_memory_state,
      .symbolic_expr_context =
          loom_low_lower_context_symbolic_expr_context(context),
  };
}

static iree_status_t loom_low_lower_rule_set_match_view_regions(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_view_region_table_t** out_view_regions) {
  *out_view_regions = NULL;
  if (rule_set->source_memory_count == 0) {
    return iree_ok_status();
  }
  return loom_low_lower_context_view_regions(context, out_view_regions);
}

static iree_status_t loom_low_lower_rule_set_select_range_from_lowering(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_match_flags_t match_flags, uint16_t rule_start,
    uint16_t rule_count, loom_low_source_memory_access_plan_t* access_plan,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_match_view_regions(
      context, rule_set, &view_regions));
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(source_op, access_plan,
                                                     &source_memory_state);
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, view_regions, &source_memory_state, &match_context);
  match_context.flags |= match_flags;
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      &match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

static iree_status_t loom_low_lower_rule_set_select_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_match_flags_t match_flags, uint16_t rule_start,
    uint16_t rule_count, loom_low_lower_rule_selection_t* out_selection) {
  if (rule_set->source_memory_count != 0) {
    loom_low_source_memory_access_plan_t access_plan;
    return loom_low_lower_rule_set_select_range_from_lowering(
        context, rule_set, source_op, match_flags, rule_start, rule_count,
        &access_plan, out_selection);
  }
  return loom_low_lower_rule_set_select_range_from_lowering(
      context, rule_set, source_op, match_flags, rule_start, rule_count,
      /*access_plan=*/NULL, out_selection);
}

void loom_low_lower_rule_materialize_diagnostic_params(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_diagnostic_t* diagnostic,
    loom_diagnostic_param_t* out_params) {
  uint8_t param_index = 0;
  if (iree_any_bit_set(
          diagnostic->flags,
          LOOM_LOW_LOWER_DIAGNOSTIC_FLAG_IMPLICIT_TARGET_CONTEXT)) {
    out_params[0] = loom_param_string(
        loom_low_lower_rule_target_key(match_context->bundle));
    out_params[1] = loom_param_string(
        loom_low_lower_rule_export_name(match_context->bundle));
    out_params[2] = loom_param_string(
        loom_low_lower_rule_config_key(match_context->bundle));
    out_params[3] =
        loom_param_string(loom_low_lower_rule_function_name(match_context));
    out_params[4] =
        loom_param_string(loom_op_name(match_context->module, source_op));
    param_index = LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT;
  }
  const uint8_t stored_param_count = diagnostic->param_count - param_index;
  for (uint8_t stored_param_index = 0; stored_param_index < stored_param_count;
       ++stored_param_index, ++param_index) {
    const uint16_t param_ref_index =
        (uint16_t)(diagnostic->param_start + stored_param_index);
    const loom_low_lower_diagnostic_param_ref_t param_ordinal =
        rule_set->diagnostic_param_refs[param_ref_index];
    const loom_low_lower_diagnostic_param_t* row =
        &rule_set->diagnostic_params[param_ordinal];
    switch (row->kind) {
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_target_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_export_name(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_config_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME:
        out_params[param_index] =
            loom_param_string(loom_low_lower_rule_function_name(match_context));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME:
        out_params[param_index] =
            loom_param_string(loom_op_name(match_context->module, source_op));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL:
        out_params[param_index] =
            loom_param_string(loom_low_lower_rule_set_string(
                rule_set, row->value.string_value_offset));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE: {
        const loom_value_id_t value_id = loom_low_lower_rule_source_value(
            match_context->module, rule_set, source_op,
            row->value.value_ref_index);
        out_params[param_index] = loom_param_type(
            loom_module_value_type(match_context->module, value_id));
        break;
      }
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL:
        out_params[param_index] = loom_param_i64(row->value.i64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL:
        out_params[param_index] = loom_param_u32(row->value.u32_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL:
        out_params[param_index] = loom_param_u64(row->value.u64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL:
        out_params[param_index] = loom_param_bool(row->value.bool_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_MEMORY_MINIMUM_ALIGNMENT:
        out_params[param_index] =
            loom_param_u32(loom_low_lower_rule_source_memory_minimum_alignment(
                match_context, source_op));
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated diagnostic param kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
}

static iree_status_t loom_low_lower_rule_emit_diagnostic(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t diagnostic_index,
    loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  if (diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      diagnostic_index >= rule_set->diagnostic_count) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  const loom_low_lower_diagnostic_t* diagnostic =
      &rule_set->diagnostics[diagnostic_index];
  loom_diagnostic_param_t params[LOOM_LOW_LOWER_MAX_DIAGNOSTIC_PARAMS] = {0};
  IREE_ASSERT_LE(diagnostic->param_count, IREE_ARRAYSIZE(params));
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_match_view_regions(
      context, rule_set, &view_regions));
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, view_regions, source_memory_state, &match_context);
  loom_low_lower_rule_materialize_diagnostic_params(
      &match_context, rule_set, source_op, diagnostic, params);
  return loom_low_lower_emit_error_ref(context, source_op,
                                       diagnostic->error_ref, params,
                                       diagnostic->param_count);
}

iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  return loom_low_lower_rule_set_select_range(
      context, rule_set, source_op, /*match_flags=*/0,
      span ? span->rule_start : 0, span ? span->rule_count : 0, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_contract(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  return loom_low_lower_rule_set_select_range(
      context, rule_set, source_op,
      LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY, span ? span->rule_start : 0,
      span ? span->rule_count : 0, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_rule_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  return loom_low_lower_rule_set_select_range(context, rule_set, source_op,
                                              /*match_flags=*/0, rule_start,
                                              rule_count, out_selection);
}

const loom_low_lower_diagnostic_t* loom_low_lower_rule_set_selection_diagnostic(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t selection) {
  if (selection.rule != NULL ||
      selection.diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      selection.diagnostic_index >= rule_set->diagnostic_count) {
    return NULL;
  }
  return &rule_set->diagnostics[selection.diagnostic_index];
}

loom_low_lower_descriptor_ref_t loom_low_lower_rule_first_descriptor_ref(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_descriptor_ref_t descriptor_ref =
        rule_set->emits[emit_index].descriptor_ref;
    if (descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
      return descriptor_ref;
    }
  }
  return LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE;
}

iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t selection,
    loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  IREE_ASSERT(selection.rule == NULL);
  if (!selection.has_source_op_span) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  return loom_low_lower_rule_emit_diagnostic(context, rule_set, source_op,
                                             selection.diagnostic_index,
                                             source_memory_state);
}

iree_status_t loom_low_lower_rule_set_resolve_emit_program(
    loom_low_lower_context_t* context, uint16_t rule_set_index,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t** out_resolved_emits) {
  *out_resolved_emits = NULL;
  if (rule->emit_count == 0) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, rule->emit_count, sizeof(*resolved_emits),
      (void**)&resolved_emits));
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, /*view_regions=*/NULL, /*source_memory_state=*/NULL,
      &match_context);
  match_context.policy_rule_set_ordinal = (uint16_t)(rule_set_index + 1u);
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    resolved_emits[i].emit = emit;
    IREE_ASSERT_NE(emit->descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
    const loom_low_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
        &match_context, rule_set, emit->descriptor_ref, &descriptor));
    IREE_ASSERT(descriptor != NULL,
                "generated target-low rule references a missing descriptor");
    resolved_emits[i].descriptor.descriptor = descriptor;
  }
  *out_resolved_emits = resolved_emits;
  return iree_ok_status();
}
