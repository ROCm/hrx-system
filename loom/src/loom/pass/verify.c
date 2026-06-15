// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/verify.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/invocation.h"
#include "loom/pass/predicate.h"

typedef struct loom_pass_verify_state_t {
  // Module containing the pass pipeline IR.
  const loom_module_t* module;
  // Caller-supplied verification options.
  const loom_pass_verify_options_t* options;
  // Scratch arena used by descriptor option decoding.
  iree_arena_allocator_t* scratch_arena;
} loom_pass_verify_state_t;

typedef struct loom_pass_verify_call_frame_t {
  // Pipeline currently being verified.
  const loom_op_t* pipeline_op;
  // Caller frame, or NULL for the root pipeline.
  const struct loom_pass_verify_call_frame_t* parent;
} loom_pass_verify_call_frame_t;

static const char* loom_pass_anchor_name(uint8_t anchor) {
  switch (anchor) {
    case LOOM_PASS_ANCHOR_MODULE:
      return "module";
    case LOOM_PASS_ANCHOR_FUNC:
      return "func";
    default:
      return "unknown";
  }
}

static bool loom_pass_anchor_is_valid(uint8_t anchor) {
  return anchor < LOOM_PASS_ANCHOR_COUNT_;
}

static loom_pass_kind_t loom_pass_kind_from_anchor(uint8_t anchor) {
  switch (anchor) {
    case LOOM_PASS_ANCHOR_MODULE:
      return LOOM_PASS_MODULE;
    case LOOM_PASS_ANCHOR_FUNC:
      return LOOM_PASS_FUNCTION;
    default:
      return LOOM_PASS_COUNT_;
  }
}

static iree_status_t loom_pass_string_from_id(const loom_module_t* module,
                                              loom_string_id_t string_id,
                                              const char* label,
                                              iree_string_view_t* out_string) {
  if (!module || !out_string || string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %s string id", label);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_string_view_t loom_pass_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!module || !loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<invalid>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<invalid>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_pass_pipeline_name(
    const loom_module_t* module, const loom_op_t* pipeline_op) {
  if (!pipeline_op || !loom_pass_pipeline_isa(pipeline_op)) {
    return IREE_SV("<not-a-pipeline>");
  }
  return loom_pass_symbol_name_from_ref(module,
                                        loom_pass_pipeline_symbol(pipeline_op));
}

static iree_status_t loom_pass_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t attr_name, const loom_attribute_t** out_attr,
    bool* out_found) {
  *out_attr = NULL;
  *out_found = false;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_pass_string_from_id(
        module, attrs.entries[i].name_id, "pass.where attr name", &name));
    if (!iree_string_view_equal(name, attr_name)) {
      continue;
    }
    if (*out_found) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass.where duplicate predicate attr '%.*s'",
                              (int)attr_name.size, attr_name.data);
    }
    *out_attr = &attrs.entries[i].value;
    *out_found = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_where_attrs_are(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t first_name, iree_string_view_t second_name) {
  bool seen_first_attr = false;
  bool seen_second_attr = false;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_pass_string_from_id(
        module, attrs.entries[i].name_id, "pass.where attr name", &name));
    if (iree_string_view_equal(name, first_name)) {
      if (seen_first_attr) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pass.where duplicate predicate attr '%.*s'",
                                (int)name.size, name.data);
      }
      seen_first_attr = true;
      continue;
    }
    if (!iree_string_view_is_empty(second_name) &&
        iree_string_view_equal(name, second_name)) {
      if (seen_second_attr) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pass.where duplicate predicate attr '%.*s'",
                                (int)name.size, name.data);
      }
      seen_second_attr = true;
      continue;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where predicate does not accept attr '%.*s'",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_required_string_where_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t attr_name, const loom_attribute_t** out_attr) {
  bool found = false;
  IREE_RETURN_IF_ERROR(
      loom_pass_find_named_attr(module, attrs, attr_name, out_attr, &found));
  if (!found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where predicate requires string attr '%.*s'",
                            (int)attr_name.size, attr_name.data);
  }
  if ((*out_attr)->kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where predicate attr '%.*s' must be string",
                            (int)attr_name.size, attr_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_pipeline_symbol(
    const loom_module_t* module, const loom_op_t* pipeline_op) {
  loom_symbol_ref_t ref = loom_pass_pipeline_symbol(pipeline_op);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.pipeline has invalid symbol reference");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->defining_op != pipeline_op) {
    iree_string_view_t symbol_name =
        loom_pass_symbol_name_from_ref(module, ref);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.pipeline @%.*s is not linked as its symbol definition",
        (int)symbol_name.size, symbol_name.data);
  }
  return iree_ok_status();
}

static bool loom_pass_verify_stack_contains(
    const loom_pass_verify_call_frame_t* frame, const loom_op_t* pipeline_op) {
  for (const loom_pass_verify_call_frame_t* it = frame; it; it = it->parent) {
    if (it->pipeline_op == pipeline_op) return true;
  }
  return false;
}

static iree_status_t loom_pass_verify_region(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    loom_pass_kind_t current_kind, const loom_region_t* region,
    const loom_pass_verify_call_frame_t* call_stack);

static iree_status_t loom_pass_verify_pipeline(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    const loom_pass_verify_call_frame_t* call_stack) {
  if (!pipeline_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected pass.pipeline op");
  }
  if (!loom_pass_pipeline_isa(pipeline_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected pass.pipeline op");
  }
  iree_string_view_t pipeline_name =
      loom_pass_pipeline_name(state->module, pipeline_op);
  if (loom_pass_verify_stack_contains(call_stack, pipeline_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass pipeline call cycle reaches @%.*s",
                            (int)pipeline_name.size, pipeline_name.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_pass_verify_pipeline_symbol(state->module, pipeline_op));

  uint8_t anchor = loom_pass_pipeline_anchor(pipeline_op);
  if (!loom_pass_anchor_is_valid(anchor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.pipeline @%.*s has invalid anchor %u",
                            (int)pipeline_name.size, pipeline_name.data,
                            (unsigned)anchor);
  }

  const loom_pass_verify_call_frame_t frame = {
      .pipeline_op = pipeline_op,
      .parent = call_stack,
  };
  iree_status_t status = loom_pass_verify_region(
      state, pipeline_op, loom_pass_kind_from_anchor(anchor),
      loom_pass_pipeline_body(pipeline_op), &frame);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status,
                                  "while verifying pass.pipeline @%.*s<%s>",
                                  (int)pipeline_name.size, pipeline_name.data,
                                  loom_pass_anchor_name(anchor));
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_resolve_pipeline_ref(
    loom_pass_verify_state_t* state, const loom_op_t* source_op,
    loom_symbol_ref_t ref, const loom_op_t** out_pipeline_op) {
  *out_pipeline_op = NULL;
  iree_string_view_t callee_name =
      loom_pass_symbol_name_from_ref(state->module, ref);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.call in op '%.*s' has invalid callee @%.*s",
                            (int)loom_op_name(state->module, source_op).size,
                            loom_op_name(state->module, source_op).data,
                            (int)callee_name.size, callee_name.data);
  }
  const loom_symbol_t* symbol = &state->module->symbols.entries[ref.symbol_id];
  if (!symbol->defining_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.call references unresolved pass pipeline @%.*s",
        (int)callee_name.size, callee_name.data);
  }
  if (!loom_pass_pipeline_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.call callee @%.*s resolves to '%.*s', not pass.pipeline",
        (int)callee_name.size, callee_name.data,
        (int)loom_op_name(state->module, symbol->defining_op).size,
        loom_op_name(state->module, symbol->defining_op).data);
  }
  *out_pipeline_op = symbol->defining_op;
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_requirements(
    const loom_pass_verify_state_t* state,
    const loom_pass_invocation_t* invocation) {
  const loom_pass_descriptor_t* descriptor = invocation->descriptor;
  for (uint16_t i = 0; i < descriptor->requirement_count; ++i) {
    const loom_pass_requirement_def_t* requirement =
        &descriptor->requirement_defs[i];
    const loom_pass_environment_capability_t* capability =
        loom_pass_environment_lookup(&state->options->environment,
                                     requirement->capability_type);
    bool satisfied = loom_pass_environment_capability_satisfies_requirement(
        capability, requirement->key);
    if (satisfied) {
      continue;
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass '%.*s' requirement '%.*s' from capability '%.*s' is not "
        "satisfied by the pass environment: %.*s",
        (int)descriptor->key.size, descriptor->key.data,
        (int)requirement->key.size, requirement->key.data,
        (int)requirement->capability_type->name.size,
        requirement->capability_type->name.data,
        (int)requirement->description.size, requirement->description.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_run(loom_pass_verify_state_t* state,
                                          const loom_op_t* pipeline_op,
                                          const loom_op_t* op,
                                          loom_pass_kind_t current_kind) {
  loom_pass_invocation_t invocation = {0};
  iree_status_t status = loom_pass_invocation_resolve_run_op(
      state->module, state->options->registry, op, current_kind,
      state->scratch_arena, &invocation);
  if (iree_status_is_ok(status)) {
    status = loom_pass_verify_requirements(state, &invocation);
  }
  if (!iree_status_is_ok(status)) {
    iree_string_view_t key = iree_string_view_empty();
    if (loom_pass_run_key(op) < state->module->strings.count) {
      key = state->module->strings.entries[loom_pass_run_key(op)];
    }
    iree_string_view_t pipeline_name =
        loom_pass_pipeline_name(state->module, pipeline_op);
    return iree_status_annotate_f(
        status,
        "while resolving pass.run<%.*s> in pass.pipeline @%.*s at %s anchor",
        (int)key.size, key.data, (int)pipeline_name.size, pipeline_name.data,
        current_kind == LOOM_PASS_MODULE ? "module" : "func");
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_for(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    const loom_op_t* op, loom_pass_kind_t current_kind,
    const loom_pass_verify_call_frame_t* call_stack) {
  uint8_t anchor = loom_pass_for_anchor(op);
  if (!loom_pass_anchor_is_valid(anchor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.for has invalid anchor %u", (unsigned)anchor);
  }
  loom_pass_kind_t body_kind = loom_pass_kind_from_anchor(anchor);
  if (body_kind == current_kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "nested same-anchor pass.for<%s> is not supported",
                            loom_pass_anchor_name(anchor));
  }
  if (current_kind != LOOM_PASS_MODULE || body_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.for<%s> cannot be entered from %s anchor",
        loom_pass_anchor_name(anchor),
        current_kind == LOOM_PASS_MODULE ? "module" : "func");
  }
  return loom_pass_verify_region(state, pipeline_op, body_kind,
                                 loom_pass_for_body(op), call_stack);
}

static iree_status_t loom_pass_verify_where(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    const loom_op_t* op, loom_pass_kind_t current_kind,
    const loom_pass_verify_call_frame_t* call_stack) {
  iree_string_view_t predicate = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_pass_string_from_id(state->module, loom_pass_where_predicate(op),
                               "pass.where predicate", &predicate));
  if (iree_string_view_is_empty(predicate)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where predicate must not be empty");
  }
  loom_named_attr_slice_t attrs = loom_pass_where_attrs(op);
  if (iree_string_view_equal(predicate, IREE_SV("name"))) {
    IREE_RETURN_IF_ERROR(loom_pass_verify_where_attrs_are(
        state->module, attrs, IREE_SV("value"), iree_string_view_empty()));
    const loom_attribute_t* value_attr = NULL;
    IREE_RETURN_IF_ERROR(loom_pass_verify_required_string_where_attr(
        state->module, attrs, IREE_SV("value"), &value_attr));
    (void)value_attr;
  } else if (iree_string_view_equal(predicate, IREE_SV("attr"))) {
    if (current_kind != LOOM_PASS_FUNCTION) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass.where attr predicate requires func anchor");
    }
    IREE_RETURN_IF_ERROR(loom_pass_verify_where_attrs_are(
        state->module, attrs, IREE_SV("name"), IREE_SV("value")));
    const loom_attribute_t* name_attr = NULL;
    IREE_RETURN_IF_ERROR(loom_pass_verify_required_string_where_attr(
        state->module, attrs, IREE_SV("name"), &name_attr));
    (void)name_attr;
  } else if (iree_string_view_equal(predicate, IREE_SV("trait"))) {
    if (current_kind != LOOM_PASS_FUNCTION) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass.where trait predicate requires func anchor");
    }
    IREE_RETURN_IF_ERROR(loom_pass_verify_where_attrs_are(
        state->module, attrs, IREE_SV("name"), iree_string_view_empty()));
    const loom_attribute_t* name_attr = NULL;
    IREE_RETURN_IF_ERROR(loom_pass_verify_required_string_where_attr(
        state->module, attrs, IREE_SV("name"), &name_attr));
    iree_string_view_t trait_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_pass_string_from_id(state->module, name_attr->string_id,
                                 "pass.where trait name", &trait_name));
    loom_trait_flags_t trait_flag = 0;
    if (!loom_pass_predicate_lookup_trait(trait_name, &trait_flag)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown pass.where trait '%.*s'",
                              (int)trait_name.size, trait_name.data);
    }
  } else if (state->options->predicate_provider.verify) {
    IREE_RETURN_IF_ERROR(state->options->predicate_provider.verify(
        state->options->predicate_provider.user_data,
        &(loom_pass_predicate_verify_context_t){
            .pipeline_module = state->module,
            .where_op = op,
            .anchor_kind = current_kind,
            .predicate = predicate,
        }));
  } else if (!state->options->predicate_provider.evaluate) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.where predicate '%.*s'",
                            (int)predicate.size, predicate.data);
  }
  return loom_pass_verify_region(state, pipeline_op, current_kind,
                                 loom_pass_where_body(op), call_stack);
}

static iree_status_t loom_pass_verify_repeat_bounds(const loom_op_t* op) {
  uint8_t mode = loom_pass_repeat_mode(op);
  if (mode >= LOOM_PASS_REPEAT_MODE_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.repeat has invalid mode %u", (unsigned)mode);
  }
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  bool has_count =
      !loom_attr_is_absent(attrs[loom_pass_repeat_count_ATTR_INDEX]);
  bool has_max_iterations =
      !loom_attr_is_absent(attrs[loom_pass_repeat_max_iterations_ATTR_INDEX]);
  if (mode == LOOM_PASS_REPEAT_MODE_FIXED) {
    if (!has_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass.repeat<fixed> requires count");
    }
    if (has_max_iterations) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass.repeat<fixed> must not specify max_iterations");
    }
    if (loom_pass_repeat_count(op) <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass.repeat<fixed> count must be positive");
    }
    return iree_ok_status();
  }
  if (has_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.repeat<until_converged> must not specify count");
  }
  if (!has_max_iterations) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.repeat<until_converged> requires max_iterations");
  }
  if (loom_pass_repeat_max_iterations(op) <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.repeat<until_converged> max_iterations must be positive");
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_repeat(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    const loom_op_t* op, loom_pass_kind_t current_kind,
    const loom_pass_verify_call_frame_t* call_stack) {
  IREE_RETURN_IF_ERROR(loom_pass_verify_repeat_bounds(op));
  return loom_pass_verify_region(state, pipeline_op, current_kind,
                                 loom_pass_repeat_body(op), call_stack);
}

static iree_status_t loom_pass_verify_call(
    loom_pass_verify_state_t* state, const loom_op_t* op,
    loom_pass_kind_t current_kind,
    const loom_pass_verify_call_frame_t* call_stack) {
  const loom_op_t* callee_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_verify_resolve_pipeline_ref(
      state, op, loom_pass_call_callee(op), &callee_op));
  uint8_t callee_anchor = loom_pass_pipeline_anchor(callee_op);
  if (!loom_pass_anchor_is_valid(callee_anchor)) {
    iree_string_view_t callee_name =
        loom_pass_pipeline_name(state->module, callee_op);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.call callee @%.*s has invalid anchor %u",
                            (int)callee_name.size, callee_name.data,
                            (unsigned)callee_anchor);
  }
  loom_pass_kind_t callee_kind = loom_pass_kind_from_anchor(callee_anchor);
  if (callee_kind != current_kind) {
    iree_string_view_t callee_name =
        loom_pass_pipeline_name(state->module, callee_op);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.call callee @%.*s anchor %s does not match current %s anchor",
        (int)callee_name.size, callee_name.data,
        loom_pass_anchor_name(callee_anchor),
        current_kind == LOOM_PASS_MODULE ? "module" : "func");
  }
  return loom_pass_verify_pipeline(state, callee_op, call_stack);
}

static iree_status_t loom_pass_verify_message(loom_pass_verify_state_t* state,
                                              loom_string_id_t message_id,
                                              iree_string_view_t op_name) {
  iree_string_view_t message = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_string_from_id(state->module, message_id,
                                                "pass message", &message));
  if (iree_string_view_is_empty(message)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s message must not be empty", (int)op_name.size,
                            op_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_region_shape(
    const loom_region_t* region) {
  if (!region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass pipeline region is required");
  }
  if (region->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass pipeline regions must contain exactly one block, got %u",
        (unsigned)region->block_count);
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(region);
  if (entry_block->arg_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass pipeline regions must not declare block arguments");
  }
  if (entry_block->op_count == 0 || !entry_block->last_op ||
      !loom_pass_yield_isa(entry_block->last_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass pipeline regions must terminate with pass.yield");
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_region(
    loom_pass_verify_state_t* state, const loom_op_t* pipeline_op,
    loom_pass_kind_t current_kind, const loom_region_t* region,
    const loom_pass_verify_call_frame_t* call_stack) {
  IREE_RETURN_IF_ERROR(loom_pass_verify_region_shape(region));
  const loom_block_t* entry_block = loom_region_const_entry_block(region);
  const loom_op_t* last_op = entry_block->last_op;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (op == last_op) break;
    switch (op->kind) {
      case LOOM_OP_PASS_FOR: {
        IREE_RETURN_IF_ERROR(loom_pass_verify_for(state, pipeline_op, op,
                                                  current_kind, call_stack));
        break;
      }
      case LOOM_OP_PASS_WHERE: {
        IREE_RETURN_IF_ERROR(loom_pass_verify_where(state, pipeline_op, op,
                                                    current_kind, call_stack));
        break;
      }
      case LOOM_OP_PASS_REPEAT: {
        IREE_RETURN_IF_ERROR(loom_pass_verify_repeat(state, pipeline_op, op,
                                                     current_kind, call_stack));
        break;
      }
      case LOOM_OP_PASS_CALL: {
        IREE_RETURN_IF_ERROR(
            loom_pass_verify_call(state, op, current_kind, call_stack));
        break;
      }
      case LOOM_OP_PASS_RUN: {
        IREE_RETURN_IF_ERROR(
            loom_pass_verify_run(state, pipeline_op, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_FAIL: {
        IREE_RETURN_IF_ERROR(loom_pass_verify_message(
            state, loom_pass_fail_message(op), IREE_SV("pass.fail")));
        break;
      }
      case LOOM_OP_PASS_HALT: {
        IREE_RETURN_IF_ERROR(loom_pass_verify_message(
            state, loom_pass_halt_message(op), IREE_SV("pass.halt")));
        break;
      }
      case LOOM_OP_PASS_YIELD:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass.yield must be the final op in a pass pipeline region");
      default: {
        iree_string_view_t pipeline_name =
            loom_pass_pipeline_name(state->module, pipeline_op);
        iree_string_view_t op_name = loom_op_name(state->module, op);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "op '%.*s' is not allowed in pass.pipeline @%.*s at %s anchor",
            (int)op_name.size, op_name.data, (int)pipeline_name.size,
            pipeline_name.data,
            current_kind == LOOM_PASS_MODULE ? "module" : "func");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_verify_state_initialize(
    const loom_module_t* module, const loom_pass_verify_options_t* options,
    iree_arena_allocator_t* scratch_arena,
    loom_pass_verify_state_t* out_state) {
  IREE_RETURN_IF_ERROR(loom_pass_environment_verify(&options->environment));
  *out_state = (loom_pass_verify_state_t){
      .module = module,
      .options = options,
      .scratch_arena = scratch_arena,
  };
  return iree_ok_status();
}

iree_status_t loom_pass_verify_pipeline_op(
    const loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_verify_options_t* options,
    iree_arena_allocator_t* scratch_arena) {
  loom_pass_verify_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_pass_verify_state_initialize(
      module, options, scratch_arena, &state));
  return loom_pass_verify_pipeline(&state, pipeline_op, NULL);
}

iree_status_t loom_pass_verify_module(const loom_module_t* module,
                                      const loom_pass_verify_options_t* options,
                                      iree_arena_allocator_t* scratch_arena) {
  loom_pass_verify_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_pass_verify_state_initialize(
      module, options, scratch_arena, &state));
  if (!module->body || module->body->block_count == 0) {
    return iree_ok_status();
  }
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    if (loom_pass_pipeline_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pass_verify_pipeline(&state, op, NULL));
    }
  }
  return iree_ok_status();
}
