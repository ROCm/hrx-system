// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/operand_forms.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

#define LOOM_LOW_SELECT_OPERAND_FORMS_STATISTICS(V, statistics_type)      \
  V(statistics_type, forms_selected, "forms-selected",                    \
    "Number of low packets rewritten to descriptor operand forms.")       \
  V(statistics_type, forms_rejected, "forms-rejected",                    \
    "Number of matched descriptor operand forms rejected.")               \
  V(statistics_type, operands_rematerialized, "operands-rematerialized",  \
    "Number of replacement operands rematerialized near selected forms.") \
  V(statistics_type, packets_folded, "packets-folded",                    \
    "Number of descriptor-backed low packets folded away.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_select_operand_forms_statistics,
                            loom_low_select_operand_forms_statistics_t,
                            LOOM_LOW_SELECT_OPERAND_FORMS_STATISTICS)

typedef struct loom_low_select_operand_forms_pass_state_t {
  // True when the pass should emit operand-form selection diagnostics.
  bool emit_operand_form_diagnostics;
  // True once the diagnostics option has been parsed.
  bool has_diagnostics_option;
} loom_low_select_operand_forms_pass_state_t;

typedef struct loom_low_select_operand_forms_parse_context_t {
  // Mutable pass state being populated.
  loom_low_select_operand_forms_pass_state_t* state;
} loom_low_select_operand_forms_parse_context_t;

static const loom_pass_option_def_t kLowSelectOperandFormsOptions[] = {
    {IREE_SVL("diagnostics"),
     IREE_SVL("Diagnostic feedback to emit: none or operand-forms.")},
};

static const loom_pass_info_t loom_low_select_operand_forms_pass_info_storage =
    {
        .name = IREE_SVL("low-select-operand-forms"),
        .description = IREE_SVL(
            "Rewrite low packets to descriptor-selected operand forms."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kLowSelectOperandFormsOptions,
        .option_count = IREE_ARRAYSIZE(kLowSelectOperandFormsOptions),
        .statistic_layout = &loom_low_select_operand_forms_statistics_layout,
};

const loom_pass_info_t* loom_low_select_operand_forms_pass_info(void) {
  return &loom_low_select_operand_forms_pass_info_storage;
}

static iree_status_t loom_low_select_operand_forms_parse_diagnostics(
    iree_string_view_t text,
    loom_low_select_operand_forms_parse_context_t* context) {
  if (context->state->has_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'diagnostics' for pass 'low-select-operand-forms'");
  }
  text = iree_string_view_trim(text);
  if (iree_string_view_equal(text, IREE_SV("none"))) {
    context->state->emit_operand_form_diagnostics = false;
  } else if (iree_string_view_equal(text, IREE_SV("operand-forms"))) {
    context->state->emit_operand_form_diagnostics = true;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-select-operand-forms' option 'diagnostics' expected "
        "'none' or 'operand-forms', got '%.*s'",
        (int)text.size, text.data);
  }
  context->state->has_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_select_operand_forms_parse_context_t* context =
      (loom_low_select_operand_forms_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_low_select_operand_forms_parse_diagnostics(value, context);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'low-select-operand-forms'",
      (int)name.size, name.data);
}

iree_status_t loom_low_select_operand_forms_create(loom_pass_t* pass,
                                                   iree_string_view_t options) {
  loom_low_select_operand_forms_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));

  loom_low_select_operand_forms_parse_context_t context = {
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("diagnostics"))) {
        iree_string_view_t value =
            option->schema->enum_values[option->enum_value_index].value;
        IREE_RETURN_IF_ERROR(
            loom_low_select_operand_forms_parse_diagnostics(value, &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'low-select-operand-forms'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_low_select_operand_forms_parse_option,
            .user_data = &context,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

typedef struct loom_low_select_operand_forms_state_t {
  // Pass invocation used for scratch arena allocation.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_low_select_operand_forms_statistics_t* statistics;
  // Module being rewritten.
  loom_module_t* module;
  // Function op that owns the current rewrite walk.
  const loom_op_t* function_op;
  // Resolved target for the low function.
  const loom_low_resolved_target_t* target;
  // Scoped function facts used to match descriptor operand-form predicates.
  loom_value_fact_table_t* value_facts;
  // True when the pass should emit operand-form selection diagnostics.
  bool emit_operand_form_diagnostics;
  // True once this pass has rewritten at least one packet.
  bool changed;
} loom_low_select_operand_forms_state_t;

static bool loom_low_select_operand_form_matches(
    const loom_value_fact_table_t* value_facts, loom_value_id_t value_id,
    const loom_low_operand_form_match_t* match, int64_t* out_matched_value) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(value_facts, value_id);
  switch (match->match_kind) {
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64: {
      loom_value_facts_t element = loom_value_facts_unknown();
      if (!loom_value_facts_query_all_equal_element(&value_facts->context,
                                                    facts, &element)) {
        return false;
      }
      int64_t value = 0;
      if (!loom_value_facts_as_exact_i64(element, &value) ||
          value != match->match_i64) {
        return false;
      }
      *out_matched_value = value;
      return true;
    }
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_EXACT_I64: {
      loom_value_facts_t element = loom_value_facts_unknown();
      if (!loom_value_facts_query_all_equal_element(&value_facts->context,
                                                    facts, &element)) {
        return false;
      }
      return loom_value_facts_as_exact_i64(element, out_matched_value);
    }
    default:
      return false;
  }
}

static bool loom_low_packet_has_operand_forms(
    const loom_low_resolved_descriptor_packet_t* packet) {
  return packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_OP &&
         packet->descriptor != NULL &&
         packet->descriptor->operand_form_count != 0;
}

static bool loom_low_semantic_tag_has_token(iree_string_view_t semantic_tag,
                                            iree_string_view_t token) {
  if (semantic_tag.size == 0) {
    return false;
  }
  iree_host_size_t start = 0;
  while (start <= semantic_tag.size) {
    iree_host_size_t end = iree_string_view_find_char(semantic_tag, '.', start);
    if (end == IREE_STRING_VIEW_NPOS) {
      end = semantic_tag.size;
    }
    if (iree_string_view_equal(
            iree_make_string_view(semantic_tag.data + start, end - start),
            token)) {
      return true;
    }
    if (end == semantic_tag.size) {
      break;
    }
    start = end + 1;
  }
  return false;
}

static bool loom_low_descriptor_is_select(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      descriptor_set, descriptor->semantic_tag_string_offset);
  return loom_low_semantic_tag_has_token(semantic_tag, IREE_SV("select"));
}

static bool loom_low_operand_field_is(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t field_name) {
  return iree_string_view_equal(
      loom_low_descriptor_set_string(descriptor_set,
                                     operand->field_name_string_offset),
      field_name);
}

static bool loom_low_descriptor_packet_operand_indices_for_select(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t* out_true_operand_index,
    uint16_t* out_false_operand_index) {
  *out_true_operand_index = UINT16_MAX;
  *out_false_operand_index = UINT16_MAX;
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(operand->role) ||
        iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    if (loom_low_operand_field_is(descriptor_set, operand,
                                  IREE_SV("true_value"))) {
      *out_true_operand_index = packet_operand_index;
    } else if (loom_low_operand_field_is(descriptor_set, operand,
                                         IREE_SV("false_value"))) {
      *out_false_operand_index = packet_operand_index;
    }
    ++packet_operand_index;
  }
  return *out_true_operand_index != UINT16_MAX &&
         *out_false_operand_index != UINT16_MAX;
}

static bool loom_low_values_have_same_type(loom_module_t* module,
                                           loom_value_id_t lhs,
                                           loom_value_id_t rhs) {
  return loom_type_equal(loom_module_value_type(module, lhs),
                         loom_module_value_type(module, rhs));
}

static iree_status_t loom_low_select_operand_forms_try_fold_select_packet(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* op, const loom_low_descriptor_t* descriptor, bool* out_folded) {
  *out_folded = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  if (op->result_count != 1 || descriptor->result_count != 1 ||
      descriptor->effect_count != 0 ||
      !iree_all_bits_set(descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE) ||
      !loom_low_descriptor_is_select(descriptor_set, descriptor)) {
    return iree_ok_status();
  }

  uint16_t true_operand_index = UINT16_MAX;
  uint16_t false_operand_index = UINT16_MAX;
  if (!loom_low_descriptor_packet_operand_indices_for_select(
          descriptor_set, descriptor, &true_operand_index,
          &false_operand_index) ||
      true_operand_index >= op->operand_count ||
      false_operand_index >= op->operand_count) {
    return iree_ok_status();
  }

  const loom_value_id_t true_value = loom_op_operands(op)[true_operand_index];
  const loom_value_id_t false_value = loom_op_operands(op)[false_operand_index];
  if (true_value != false_value ||
      !loom_low_values_have_same_type(state->module, loom_op_results(op)[0],
                                      true_value)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &true_value, 1));
  state->changed = true;
  *out_folded = true;
  loom_pass_mark_changed(state->pass);
  ++state->statistics->packets_folded;
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_function_has_candidate(
    loom_module_t* module, loom_func_like_t function,
    const loom_low_resolved_target_t* target, bool* out_has_candidate) {
  *out_has_candidate = false;
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = body->blocks[block_index];
    for (loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        continue;
      }
      loom_low_resolved_descriptor_packet_t packet = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_resolve_descriptor_packet(module, target, op, &packet));
      if (loom_low_packet_has_operand_forms(&packet) ||
          (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_OP &&
           packet.descriptor != NULL &&
           loom_low_descriptor_is_select(target->descriptor_set,
                                         packet.descriptor))) {
        *out_has_candidate = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static uint16_t loom_low_descriptor_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor_operand_index;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (loom_low_operand_role_is_packet_operand(operand->role) &&
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      ++packet_operand_index;
    }
  }
  return packet_operand_index;
}

typedef struct loom_low_select_operand_form_destructive_info_t {
  // True when the replacement descriptor has a destructive operand.
  bool has_destructive_operand;
  // Replacement descriptor operand index named by the destructive constraint.
  uint16_t tied_descriptor_operand_index;
  // Source packet operand index that supplies the replacement operand.
  uint16_t tied_source_packet_operand_index;
  // SSA value carried by the tied replacement operand.
  loom_value_id_t tied_value_id;
} loom_low_select_operand_form_destructive_info_t;

static void loom_low_select_operand_form_reset_destructive_info(
    loom_low_select_operand_form_destructive_info_t* out_info) {
  *out_info = (loom_low_select_operand_form_destructive_info_t){
      .tied_descriptor_operand_index = LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE,
      .tied_source_packet_operand_index = LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE,
      .tied_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static iree_string_view_t loom_low_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->key_string_offset);
}

static iree_string_view_t loom_low_descriptor_operand_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  if (descriptor_operand_index == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
      descriptor_operand_index >= descriptor->operand_count) {
    return IREE_SV("<none>");
  }
  const loom_low_operand_t* operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  return loom_low_descriptor_set_string(descriptor_set,
                                        operand->field_name_string_offset);
}

static iree_string_view_t loom_low_descriptor_immediate_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t immediate_index) {
  if (immediate_index == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
      immediate_index >= descriptor->immediate_count) {
    return IREE_SV("<none>");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set
           ->immediates[descriptor->immediate_start + immediate_index];
  return loom_low_descriptor_set_string(descriptor_set,
                                        immediate->field_name_string_offset);
}

static uint16_t loom_low_select_operand_form_diagnostic_source_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_form_t* form) {
  IREE_ASSERT(form->match_count > 0);
  uint16_t match_index = 0;
  if (form->immediate_match_index != LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    IREE_ASSERT(form->immediate_match_index < form->match_count);
    match_index = form->immediate_match_index;
  }
  const loom_low_operand_form_match_t* match =
      &descriptor_set->operand_form_matches[form->match_start + match_index];
  return match->source_operand_index;
}

static uint16_t
loom_low_select_operand_form_diagnostic_source_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_form_t* form) {
  IREE_ASSERT(form->match_count > 0);
  uint16_t match_index = 0;
  if (form->immediate_match_index != LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    IREE_ASSERT(form->immediate_match_index < form->match_count);
    match_index = form->immediate_match_index;
  }
  const loom_low_operand_form_match_t* match =
      &descriptor_set->operand_form_matches[form->match_start + match_index];
  return match->source_packet_operand_index;
}

static loom_diagnostic_param_t loom_low_select_operand_form_operand_param(
    iree_string_view_t value, uint16_t packet_operand_index) {
  loom_diagnostic_param_t param = loom_param_string(value);
  if (packet_operand_index != LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    param = loom_param_with_field_ref(
        param, loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                         packet_operand_index));
  }
  return param;
}

static iree_status_t loom_low_select_operand_form_emit_decision(
    loom_low_select_operand_forms_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* source_descriptor,
    const loom_low_descriptor_t* replacement_descriptor,
    const loom_low_operand_form_t* form, iree_string_view_t decision,
    iree_string_view_t reason, int64_t immediate_value,
    const loom_low_select_operand_form_destructive_info_t* destructive_info) {
  if (!state->emit_operand_form_diagnostics) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  const uint16_t source_operand_index =
      loom_low_select_operand_form_diagnostic_source_operand_index(
          descriptor_set, form);
  const uint16_t source_packet_operand_index =
      loom_low_select_operand_form_diagnostic_source_packet_operand_index(
          descriptor_set, form);
  const iree_string_view_t source_operand_name =
      loom_low_descriptor_operand_name(descriptor_set, source_descriptor,
                                       source_operand_index);
  iree_string_view_t tied_operand_name = IREE_SV("<none>");
  iree_string_view_t tied_value_name = IREE_SV("<none>");
  uint16_t tied_source_packet_operand_index =
      LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE;
  if (destructive_info && destructive_info->has_destructive_operand) {
    tied_operand_name = loom_low_descriptor_operand_name(
        descriptor_set, replacement_descriptor,
        destructive_info->tied_descriptor_operand_index);
    tied_source_packet_operand_index =
        destructive_info->tied_source_packet_operand_index;
    if (destructive_info->tied_value_id != LOOM_VALUE_ID_INVALID) {
      tied_value_name = loom_low_diagnostic_value_name(
          state->module, destructive_info->tied_value_id);
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(state->target)),
      loom_param_string(loom_low_diagnostic_export_name(state->target)),
      loom_param_string(loom_low_diagnostic_config_key(state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_op_name(state->module, op)),
      loom_param_string(
          loom_low_descriptor_key(descriptor_set, source_descriptor)),
      loom_param_string(
          loom_low_descriptor_key(descriptor_set, replacement_descriptor)),
      loom_param_string(decision),
      loom_param_string(reason),
      loom_low_select_operand_form_operand_param(source_operand_name,
                                                 source_packet_operand_index),
      loom_param_string(loom_low_descriptor_immediate_name(
          descriptor_set, replacement_descriptor,
          form->replacement_immediate_index)),
      loom_param_i64(immediate_value),
      loom_low_select_operand_form_operand_param(
          tied_operand_name, tied_source_packet_operand_index),
      loom_param_string(tied_value_name),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_BACKEND_042,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_low_descriptor_build_tied_results(
    iree_arena_allocator_t* arena,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  *out_tied_result_count = tied_result_count;
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tied_result_count, sizeof(*tied_results), (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index = loom_low_descriptor_packet_operand_index(
            descriptor_set, descriptor, constraint->rhs_operand_index),
        .has_type_change = false,
    };
  }
  *out_tied_results = tied_results;
  return iree_ok_status();
}

static bool loom_low_select_operand_form_operand_has_single_use(
    const loom_module_t* module, const loom_value_id_t* operands,
    iree_host_size_t operand_count, uint16_t operand_index) {
  IREE_ASSERT(operand_index < operand_count);
  const loom_value_id_t value_id = operands[operand_index];
  IREE_ASSERT(value_id < module->values.count);
  return loom_value_has_single_use(loom_module_value(module, value_id));
}

static bool loom_low_select_operand_form_can_rewrite_destructive(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* replacement_descriptor,
    const loom_low_operand_form_t* form, const loom_value_id_t* operands,
    iree_host_size_t operand_count,
    loom_low_select_operand_form_destructive_info_t* out_info) {
  loom_low_select_operand_form_reset_destructive_info(out_info);
  for (uint16_t i = 0; i < replacement_descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[replacement_descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    const uint16_t tied_packet_operand_index =
        loom_low_descriptor_packet_operand_index(descriptor_set,
                                                 replacement_descriptor,
                                                 constraint->rhs_operand_index);
    IREE_ASSERT(tied_packet_operand_index < operand_count);
    const uint16_t source_packet_operand_index =
        descriptor_set->operand_form_operand_indices[form->operand_map_start +
                                                     tied_packet_operand_index];
    if (!out_info->has_destructive_operand) {
      *out_info = (loom_low_select_operand_form_destructive_info_t){
          .has_destructive_operand = true,
          .tied_descriptor_operand_index = constraint->rhs_operand_index,
          .tied_source_packet_operand_index = source_packet_operand_index,
          .tied_value_id = operands[tied_packet_operand_index],
      };
    }
    if (!loom_low_select_operand_form_operand_has_single_use(
            module, operands, operand_count, tied_packet_operand_index)) {
      *out_info = (loom_low_select_operand_form_destructive_info_t){
          .has_destructive_operand = true,
          .tied_descriptor_operand_index = constraint->rhs_operand_index,
          .tied_source_packet_operand_index = source_packet_operand_index,
          .tied_value_id = operands[tied_packet_operand_index],
      };
      return false;
    }
  }
  return true;
}

static bool loom_low_descriptor_result_has_unary_constraint(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index,
    loom_low_constraint_kind_t kind) {
  if (result_index >= descriptor->result_count) return false;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == kind &&
        constraint->lhs_operand_index == result_index &&
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
      return true;
    }
  }
  return false;
}

static bool loom_low_descriptor_result_can_rematerialize(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  if (descriptor == NULL) return false;
  if (!iree_all_bits_set(descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE)) {
    return false;
  }
  if (!loom_low_descriptor_result_has_unary_constraint(
          descriptor_set, descriptor, result_index,
          LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE)) {
    return false;
  }

  const loom_trait_flags_t traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  return iree_all_bits_set(traits, LOOM_TRAIT_PURE) &&
         !loom_traits_may_read(traits) && !loom_traits_has_side_effects(traits);
}

static bool loom_low_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

// Rematerialization should shorten a local live range, not duplicate a value
// whose original producer must remain live for a later or cross-block use.
static bool loom_low_select_operand_form_rematerialization_shortens_live_range(
    const loom_value_t* value, const loom_op_t* before_op) {
  if (before_op->parent_block == NULL) return false;
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op == before_op) continue;
    if (user_op == NULL || user_op->parent_block != before_op->parent_block ||
        user_op->block_ordinal >= before_op->block_ordinal) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_select_operand_form_rematerialize_operand(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* before_op, loom_value_id_t* inout_value_id) {
  IREE_ASSERT(inout_value_id);
  IREE_ASSERT(*inout_value_id < state->module->values.count);

  const loom_value_t* value = loom_module_value(state->module, *inout_value_id);
  if (loom_value_is_block_arg(value) || loom_value_is_consumed(value)) {
    return iree_ok_status();
  }
  if (!loom_low_select_operand_form_rematerialization_shortens_live_range(
          value, before_op)) {
    return iree_ok_status();
  }
  const uint16_t result_index = loom_value_def_index(value);
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || defining_op == before_op ||
      defining_op == before_op->prev_op) {
    return iree_ok_status();
  }
  if (defining_op->operand_count != 0 || defining_op->result_count != 1 ||
      defining_op->region_count != 0 || defining_op->successor_count != 0 ||
      defining_op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t producer_packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, defining_op, &producer_packet));
  if (!loom_low_packet_kind_may_rematerialize(producer_packet.kind) ||
      !loom_low_descriptor_result_can_rematerialize(
          state->target->descriptor_set, producer_packet.descriptor,
          result_index)) {
    return iree_ok_status();
  }

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(state->module, state->module, state->pass->arena,
                               &(loom_ir_remap_options_t){
                                   .allow_unmapped_values = true,
                               },
                               &remap));

  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, before_op);
  loom_op_t* cloned_op = NULL;
  iree_status_t status =
      loom_ir_clone_op(&rewriter->builder, defining_op, &remap, &cloned_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  IREE_ASSERT(result_index < cloned_op->result_count);
  const loom_value_id_t cloned_value_id =
      loom_op_results(cloned_op)[result_index];
  IREE_RETURN_IF_ERROR(
      loom_rewriter_clear_value_name(rewriter, cloned_value_id));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      rewriter, *inout_value_id, cloned_value_id, IREE_SV("remat")));
  *inout_value_id = cloned_value_id;
  ++state->statistics->operands_rematerialized;
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_rematerialize_operands(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* before_op, loom_value_id_t* operands,
    iree_host_size_t operand_count) {
  loom_value_id_t* original_operands = NULL;
  if (operand_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, operand_count, sizeof(*original_operands),
        (void**)&original_operands));
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    const loom_value_id_t original_operand = operands[i];
    original_operands[i] = original_operand;
    bool reused_operand = false;
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (original_operands[j] == original_operand) {
        operands[i] = operands[j];
        reused_operand = true;
        break;
      }
    }
    if (reused_operand) continue;
    IREE_RETURN_IF_ERROR(loom_low_select_operand_form_rematerialize_operand(
        state, rewriter, before_op, &operands[i]));
  }
  return iree_ok_status();
}

static bool loom_low_enum_domain_contains_i64(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_enum_domain_t* domain, int64_t value) {
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const loom_low_enum_value_t* enum_value =
        &descriptor_set->enum_values[domain->value_start + i];
    if (enum_value->value == value) {
      return true;
    }
  }
  return false;
}

static bool loom_low_immediate_accepts_i64(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate, int64_t value) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t signed_max = immediate->unsigned_max > (uint64_t)INT64_MAX
                                     ? INT64_MAX
                                     : (int64_t)immediate->unsigned_max;
      return value >= immediate->signed_min && value <= signed_max;
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return value >= 0 && (uint64_t)value <= immediate->unsigned_max;
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      if (immediate->enum_domain_id >= descriptor_set->enum_domain_count) {
        return false;
      }
      return loom_low_enum_domain_contains_i64(
          descriptor_set,
          &descriptor_set->enum_domains[immediate->enum_domain_id], value);
    default:
      return false;
  }
}

static bool loom_low_immediate_has_default(
    const loom_low_immediate_t* immediate) {
  return iree_any_bit_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE);
}

static const loom_named_attr_t* loom_low_select_operand_form_find_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) {
      return &attrs.entries[i];
    }
  }
  return NULL;
}

static iree_status_t loom_low_select_operand_form_read_i64_immediate(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t immediate_index,
    loom_named_attr_slice_t attrs, bool* out_has_value, int64_t* out_value) {
  *out_has_value = false;
  *out_value = 0;
  if (immediate_index == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return iree_ok_status();
  }

  IREE_ASSERT(immediate_index < descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set
           ->immediates[descriptor->immediate_start + immediate_index];
  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, immediate_name, &immediate_name_id));
  const loom_named_attr_t* attr =
      loom_low_select_operand_form_find_attr(attrs, immediate_name_id);
  if (attr) {
    if (attr->value.kind != LOOM_ATTR_I64) {
      return iree_ok_status();
    }
    *out_value = loom_attr_as_i64(attr->value);
    *out_has_value = true;
    return iree_ok_status();
  }
  if (loom_low_immediate_has_default(immediate)) {
    *out_value = immediate->default_value;
    *out_has_value = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_resolve_immediate_value(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* source_descriptor,
    const loom_low_descriptor_t* replacement_descriptor,
    const loom_low_operand_form_t* form, int64_t matched_value,
    loom_named_attr_slice_t source_attrs, int64_t* out_immediate_value,
    bool* out_can_rewrite, iree_string_view_t* out_reject_reason) {
  *out_immediate_value = 0;
  *out_can_rewrite = true;
  *out_reject_reason = IREE_SV("selected");

  int64_t replacement_value = 0;
  switch (form->immediate_action) {
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE:
      return iree_ok_status();
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_SET_MATCHED_I64:
      replacement_value = matched_value;
      break;
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_ADD_MATCHED_I64: {
      bool has_source_value = false;
      int64_t source_value = 0;
      IREE_RETURN_IF_ERROR(loom_low_select_operand_form_read_i64_immediate(
          module, descriptor_set, source_descriptor,
          form->source_immediate_index, source_attrs, &has_source_value,
          &source_value));
      if (!has_source_value) {
        *out_can_rewrite = false;
        *out_reject_reason = IREE_SV("missing_source_immediate");
        return iree_ok_status();
      }
      if (!loom_checked_add_i64(source_value, matched_value,
                                &replacement_value)) {
        *out_can_rewrite = false;
        *out_reject_reason = IREE_SV("immediate_overflow");
        return iree_ok_status();
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown low operand-form immediate action %u",
                              (unsigned)form->immediate_action);
  }

  IREE_ASSERT(form->replacement_immediate_index <
              replacement_descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[replacement_descriptor->immediate_start +
                                  form->replacement_immediate_index];
  *out_immediate_value = replacement_value;
  if (!loom_low_immediate_accepts_i64(descriptor_set, immediate,
                                      replacement_value)) {
    *out_can_rewrite = false;
    *out_reject_reason = IREE_SV("immediate_out_of_range");
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_build_attrs(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* replacement_descriptor,
    const loom_low_operand_form_t* form, int64_t immediate_value,
    loom_named_attr_slice_t source_attrs, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = source_attrs;
  if (form->immediate_action == LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE) {
    return iree_ok_status();
  }

  IREE_ASSERT(form->replacement_immediate_index <
              replacement_descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[replacement_descriptor->immediate_start +
                                  form->replacement_immediate_index];
  IREE_ASSERT(loom_low_immediate_accepts_i64(descriptor_set, immediate,
                                             immediate_value));
  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, immediate_name, &immediate_name_id));
  const bool remove_default = loom_low_immediate_has_default(immediate) &&
                              immediate_value == immediate->default_value;
  loom_named_attr_update_t update =
      remove_default ? loom_named_attr_remove(immediate_name_id)
                     : loom_named_attr_replace(immediate_name_id,
                                               loom_attr_i64(immediate_value));
  loom_attribute_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_module_replace_canonical_attr_dict(
      module, source_attrs, loom_make_named_attr_update_slice(&update, 1),
      &attr));
  *out_attrs = loom_attr_as_dict(attr);
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_rewrite_packet(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* op, const loom_low_descriptor_t* source_descriptor,
    const loom_low_operand_form_t* form, int64_t matched_value,
    bool* out_rewritten) {
  *out_rewritten = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  const loom_low_descriptor_t* replacement_descriptor =
      &descriptor_set->descriptors[form->replacement_descriptor_ordinal];

  iree_string_view_t replacement_key = loom_low_descriptor_set_string(
      descriptor_set, replacement_descriptor->key_string_offset);
  loom_string_id_t replacement_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      &rewriter->builder, replacement_key, &replacement_key_id));

  loom_value_id_t* operands = NULL;
  if (form->operand_map_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->pass->arena, form->operand_map_count,
                                  sizeof(*operands), (void**)&operands));
    for (uint16_t i = 0; i < form->operand_map_count; ++i) {
      const uint16_t source_packet_operand_index =
          descriptor_set
              ->operand_form_operand_indices[form->operand_map_start + i];
      operands[i] = loom_op_operands(op)[source_packet_operand_index];
    }
  }

  int64_t immediate_value = 0;
  bool can_rewrite = false;
  iree_string_view_t reject_reason = IREE_SV("selected");
  IREE_RETURN_IF_ERROR(loom_low_select_operand_form_resolve_immediate_value(
      state->module, descriptor_set, source_descriptor, replacement_descriptor,
      form, matched_value, loom_low_op_attrs(op), &immediate_value,
      &can_rewrite, &reject_reason));
  if (!can_rewrite) {
    loom_low_select_operand_form_destructive_info_t no_destructive_info;
    loom_low_select_operand_form_reset_destructive_info(&no_destructive_info);
    ++state->statistics->forms_rejected;
    IREE_RETURN_IF_ERROR(loom_low_select_operand_form_emit_decision(
        state, op, source_descriptor, replacement_descriptor, form,
        IREE_SV("rejected"), reject_reason, immediate_value,
        &no_destructive_info));
    return iree_ok_status();
  }

  loom_low_select_operand_form_destructive_info_t destructive_info;
  if (!loom_low_select_operand_form_can_rewrite_destructive(
          state->module, descriptor_set, replacement_descriptor, form, operands,
          form->operand_map_count, &destructive_info)) {
    ++state->statistics->forms_rejected;
    IREE_RETURN_IF_ERROR(loom_low_select_operand_form_emit_decision(
        state, op, source_descriptor, replacement_descriptor, form,
        IREE_SV("rejected"), IREE_SV("destructive_operand_not_single_use"),
        immediate_value, &destructive_info));
    return iree_ok_status();
  }

  loom_type_t* result_types = NULL;
  if (op->result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, op->result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* results = loom_op_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_types[i] = loom_module_value_type(state->module, results[i]);
    }
  }

  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_build_tied_results(
      state->pass->arena, descriptor_set, replacement_descriptor, &tied_results,
      &tied_result_count));

  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  IREE_RETURN_IF_ERROR(loom_low_select_operand_form_rematerialize_operands(
      state, rewriter, op, operands, form->operand_map_count));

  loom_named_attr_slice_t replacement_attrs = {0};
  IREE_RETURN_IF_ERROR(loom_low_select_operand_form_build_attrs(
      state->module, descriptor_set, replacement_descriptor, form,
      immediate_value, loom_low_op_attrs(op), &replacement_attrs));
  IREE_RETURN_IF_ERROR(loom_low_select_operand_form_emit_decision(
      state, op, source_descriptor, replacement_descriptor, form,
      IREE_SV("selected"), IREE_SV("selected"), immediate_value,
      &destructive_info));

  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* replacement_op = NULL;
  iree_status_t status = loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, replacement_descriptor,
      replacement_key_id, operands, form->operand_map_count, replacement_attrs,
      result_types, op->result_count, tied_results, tied_result_count,
      op->location, &replacement_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  const loom_value_id_t* replacements = loom_op_results(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, replacement_op->result_count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, replacement_op->result_count));
  state->changed = true;
  *out_rewritten = true;
  loom_pass_mark_changed(state->pass);
  ++state->statistics->forms_selected;
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_try_rewrite_packet(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (!loom_low_packet_has_operand_forms(&packet)) {
    if (packet.kind != LOOM_LOW_DESCRIPTOR_PACKET_OP ||
        packet.descriptor == NULL) {
      return iree_ok_status();
    }
    bool folded = false;
    return loom_low_select_operand_forms_try_fold_select_packet(
        state, rewriter, op, packet.descriptor, &folded);
  }

  bool folded = false;
  IREE_RETURN_IF_ERROR(loom_low_select_operand_forms_try_fold_select_packet(
      state, rewriter, op, packet.descriptor, &folded));
  if (folded) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  for (uint16_t i = 0; i < packet.descriptor->operand_form_count; ++i) {
    const loom_low_operand_form_t* form =
        &descriptor_set
             ->operand_forms[packet.descriptor->operand_form_start + i];
    int64_t matched_value = 0;
    bool matched = true;
    for (uint16_t match_index = 0; match_index < form->match_count;
         ++match_index) {
      const loom_low_operand_form_match_t* match =
          &descriptor_set
               ->operand_form_matches[form->match_start + match_index];
      const loom_value_id_t value_id =
          loom_op_operands(op)[match->source_packet_operand_index];
      int64_t candidate_value = 0;
      if (!loom_low_select_operand_form_matches(state->value_facts, value_id,
                                                match, &candidate_value)) {
        matched = false;
        break;
      }
      if (match_index == form->immediate_match_index) {
        matched_value = candidate_value;
      }
    }
    if (matched) {
      bool rewrote = false;
      IREE_RETURN_IF_ERROR(loom_low_select_operand_form_rewrite_packet(
          state, rewriter, op, packet.descriptor, form, matched_value,
          &rewrote));
      if (rewrote) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_function(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_func_op = function.op;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_function_target(module, low_func_op, descriptor_registry,
                                       target_selection, emitter, &target));
  if (!target.descriptor_set) {
    return iree_ok_status();
  }

  bool has_candidate = false;
  IREE_RETURN_IF_ERROR(loom_low_select_operand_forms_function_has_candidate(
      module, function, &target, &has_candidate));
  if (!has_candidate) {
    return iree_ok_status();
  }

  loom_value_fact_table_t* value_facts = NULL;
  if (target.descriptor_set->operand_form_count != 0) {
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            function, &target.bundle_storage.bundle),
        &value_facts));
  }

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  const loom_low_select_operand_forms_pass_state_t* pass_state =
      (const loom_low_select_operand_forms_pass_state_t*)pass->state;
  loom_low_select_operand_forms_state_t state = {
      .pass = pass,
      .statistics = loom_low_select_operand_forms_statistics(pass),
      .module = module,
      .function_op = function.op,
      .target = &target,
      .value_facts = value_facts,
      .emit_operand_form_diagnostics =
          pass_state && pass_state->emit_operand_form_diagnostics,
  };

  loom_region_t* body = loom_func_like_body(function);
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = body->blocks[block_index];
    for (loom_op_t* op = block->first_op; iree_status_is_ok(status) && op;) {
      loom_op_t* next_op = op->next_op;
      if (!iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        status = loom_low_select_operand_forms_try_rewrite_packet(
            &state, &rewriter, op);
      }
      op = next_op;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  if (state.changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}

iree_status_t loom_low_select_operand_forms_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_isa(function) || !loom_func_like_body(function)) {
    return iree_ok_status();
  }
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_target_selection_t target_selection =
      loom_target_pass_capability_target_selection(target_capability);
  return loom_low_select_operand_forms_function(
      pass, module, function, descriptor_registry, target_selection,
      pass->diagnostic_emitter);
}
