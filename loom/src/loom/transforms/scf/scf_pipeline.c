// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_pipeline.h"

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/scf/scf_pipeline_plan.h"
#include "loom/transforms/scf/scf_unroll.h"
#include "loom/util/fact_extensions.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

#define LOOM_SCF_PIPELINE_STATISTICS(V, statistics_type)             \
  V(statistics_type, loops_pipelined, "loops-pipelined",             \
    "Number of explicit scf.for read-ahead schedules materialized.") \
  V(statistics_type, policies_cleared, "policies-cleared",           \
    "Number of depth-one policies removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_scf_pipeline_statistics,
                            loom_scf_pipeline_statistics_t,
                            LOOM_SCF_PIPELINE_STATISTICS)

static const loom_pass_info_t loom_scf_pipeline_pass_info_storage = {
    .name = IREE_SVL("pipeline-scf-for"),
    .description =
        IREE_SVL("Materialize explicit scf.for read-ahead schedules."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_scf_pipeline_statistics_layout,
};

const loom_pass_info_t* loom_scf_pipeline_pass_info(void) {
  return &loom_scf_pipeline_pass_info_storage;
}

typedef struct loom_scf_pipeline_context_t {
  // Retained space classification, owned for the complete function rewrite.
  loom_scf_memory_t* spaces;
  // Selected original accesses for the serial path of the current loop.
  loom_scf_body_t* body;
  // Current invocation and its diagnostics/statistics/report sinks.
  loom_pass_t* pass;
  // Source and destination module.
  loom_module_t* module;
  // Mutation interface, with module-owned output IR.
  loom_rewriter_t* rewriter;
  // Scratch storage reset after reconstructing each source loop.
  iree_arena_allocator_t* arena;
} loom_scf_pipeline_context_t;

typedef uint8_t loom_scf_pipeline_loop_flags_t;
enum loom_scf_pipeline_loop_flag_bits_e {
  // This loop or an ancestor requests nonserial reconstruction.
  LOOM_SCF_PIPELINE_LOOP_PREPARE_DESCENDANTS = 1u << 0,
  // The original analysis proves an ordinary global load in this subtree.
  LOOM_SCF_PIPELINE_LOOP_GLOBAL_READ = 1u << 1,
  // This unroll-only descendant has a selected finite full-linear plan.
  LOOM_SCF_PIPELINE_LOOP_PREPARE = 1u << 2,
  // Exact bounds preserve participants across the main/drain split.
  LOOM_SCF_PIPELINE_LOOP_STATIC_BOUNDS = 1u << 3,
  // Cloned startup indices require explicit branch-local domain refinements.
  LOOM_SCF_PIPELINE_LOOP_REFINE_STARTUP_DOMAIN = 1u << 4,
};

typedef struct loom_scf_pipeline_loop_t {
  // Annotated source loop, processed after its annotated children.
  loom_op_t* source;
  // Enclosing selected loop, or UINT32_MAX at the function root.
  uint32_t parent;
  // Next selected loop in source-preserving postorder, or UINT32_MAX.
  uint32_t next;
  // Structural depth supplied by the owning traversal.
  uint16_t nesting_depth;
  // Retained preparation and participation properties.
  loom_scf_pipeline_loop_flags_t flags;
  // Source effects that must remain in the ordered consumer.
  loom_scf_body_effect_flags_t consumer_effects;
  // Original scalar facts, selected by the source loop's pipeline policy.
  union {
    // Full linear domain for an unroll-only preparation candidate.
    loom_scf_unroll_full_plan_t unroll;
    // Scalar evidence for a loop carrying an explicit pipeline depth.
    struct {
      // Source policy facts independent of rewritten value identities.
      loom_value_facts_t depth;
      // Source induction step facts.
      loom_value_facts_t step;
      // Source lower-bound facts for the overflow-safe guard.
      loom_value_facts_t lower_bound;
      // Largest source value representable by the selected address carrier.
      int64_t maximum_value;
    } pipeline;
  } plan;
} loom_scf_pipeline_loop_t;

typedef struct loom_scf_pipeline_loop_list_t {
  // Module supplying the memory-access interface during policy discovery.
  loom_module_t* module;
  // Most recently entered selected loop, or UINT32_MAX.
  uint32_t active;
  // First selected loop in source-preserving postorder.
  uint32_t first;
  // Last completed selected loop in postorder.
  uint32_t last;
  // Number of source pipeline policies, excluding preparation candidates.
  iree_host_size_t pipeline_count;
  // Access records captured by the existing policy discovery traversal.
  loom_scf_memory_t* spaces;
  // Invocation arena owning the collected source handles.
  iree_arena_allocator_t* arena;
  // Annotated source loops in preorder, with retained postorder links.
  loom_scf_pipeline_loop_t* values;
  // Number of annotated loops.
  iree_host_size_t count;
  // Allocated source handle slots.
  iree_host_size_t capacity;
} loom_scf_pipeline_loop_list_t;

static void loom_scf_pipeline_finish_scopes(loom_scf_pipeline_loop_list_t* list,
                                            uint16_t depth) {
  while (list->active != UINT32_MAX &&
         list->values[list->active].nesting_depth >= depth) {
    const uint32_t completed = list->active;
    list->active = list->values[completed].parent;
    if (list->last == UINT32_MAX) {
      list->first = completed;
    } else {
      list->values[list->last].next = completed;
    }
    list->last = completed;
  }
}

static iree_status_t loom_scf_pipeline_collect_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  loom_scf_pipeline_loop_list_t* list = user_data;
  // Every operation closes completed scopes, including siblings with no policy.
  loom_scf_pipeline_finish_scopes(list, context->depth);
  if (!loom_scf_for_isa(op)) {
    if (list->active != UINT32_MAX) {
      if (!loom_scf_if_isa(op) && !loom_scf_yield_isa(op)) {
        list->values[list->active].consumer_effects |=
            loom_scf_body_operation_effects(list->module, op) &
            (LOOM_SCF_BODY_EFFECT_WRITE | LOOM_SCF_BODY_EFFECT_ORDERED |
             LOOM_SCF_BODY_EFFECT_CONVERGENT |
             LOOM_SCF_BODY_EFFECT_SOURCE_ORDER);
      }
      if (loom_memory_access_isa(loom_memory_access_cast(list->module, op))) {
        return loom_scf_memory_insert(list->spaces, op,
                                      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
                                      list->active);
      }
    }
    return iree_ok_status();
  }
  const bool pipeline = loom_scf_for_pipeline_depth_is_present(op);
  const bool unroll = loom_scf_for_unroll_factor_is_present(op) ||
                      loom_scf_for_has_unroll_policy(op) ||
                      loom_scf_for_has_unroll_schedule(op);
  if (!pipeline && !(list->active != UINT32_MAX && unroll)) {
    return iree_ok_status();
  }
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  // Every selected loop owns a distinct induction value, so its ordinal fits
  // the module's uint32 value identity domain.
  const uint32_t index = (uint32_t)list->count++;
  list->values[index] = (loom_scf_pipeline_loop_t){
      .source = op,
      .parent = list->active,
      .next = UINT32_MAX,
      .nesting_depth = context->depth,
  };
  list->active = index;
  list->pipeline_count += pipeline;
  return iree_ok_status();
}

// Loop reconstruction preserves source value semantics but replaces SSA IDs.
// Retain the scalar facts needed by every policy before the first rewrite;
// recomputing the entire function for each loop would make this pass quadratic
// even when clearing depth-one policies. Body plans still follow postorder so
// parents see their children's reconstructed bodies.
static iree_status_t loom_scf_pipeline_resolve_facts(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_scf_pipeline_loop_list_t* loops) {
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts));
  for (iree_host_size_t i = 0; i < loops->spaces->capacity; ++i) {
    loom_scf_memory_entry_t* entry = &loops->spaces->entries[i];
    if (!entry->op) {
      continue;
    }
    loom_memory_access_t access = loom_memory_access_cast(module, entry->op);
    loom_value_fact_view_reference_t reference = {0};
    if (loom_value_facts_query_view_reference(
            &facts->context,
            loom_value_fact_table_lookup(facts,
                                         loom_memory_access_view(access)),
            &reference)) {
      entry->space = reference.memory_space;
    }
    if (entry->space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
        loom_memory_access_operation_kind(access) ==
            LOOM_MEMORY_ACCESS_OPERATION_LOAD) {
      loops->values[entry->scope].flags |= LOOM_SCF_PIPELINE_LOOP_GLOBAL_READ;
    }
  }
  for (iree_host_size_t i = loops->first; i != UINT32_MAX;
       i = loops->values[i].next) {
    const loom_scf_pipeline_loop_t* loop = &loops->values[i];
    if (loop->parent != UINT32_MAX) {
      loom_scf_pipeline_loop_t* parent = &loops->values[loop->parent];
      parent->consumer_effects |= loop->consumer_effects;
      parent->flags |= loop->flags & LOOM_SCF_PIPELINE_LOOP_GLOBAL_READ;
    }
  }
  for (iree_host_size_t i = 0; i < loops->count; ++i) {
    loom_scf_pipeline_loop_t* loop = &loops->values[i];
    const bool enclosing =
        loop->parent != UINT32_MAX &&
        iree_any_bit_set(loops->values[loop->parent].flags,
                         LOOM_SCF_PIPELINE_LOOP_PREPARE_DESCENDANTS);
    if (enclosing) {
      loop->flags |= LOOM_SCF_PIPELINE_LOOP_PREPARE_DESCENDANTS;
    }
    if (!loom_scf_for_pipeline_depth_is_present(loop->source)) {
      if (enclosing &&
          iree_any_bit_set(loop->flags, LOOM_SCF_PIPELINE_LOOP_GLOBAL_READ) &&
          loop->consumer_effects &&
          loom_scf_unroll_plan_full(pass, module, facts, loop->source,
                                    &loop->plan.unroll)) {
        loop->flags |= LOOM_SCF_PIPELINE_LOOP_PREPARE;
      }
      continue;
    }
    loop->plan.pipeline.depth = loom_value_fact_table_lookup(
        facts, loom_scf_for_pipeline_depth(loop->source));
    int64_t depth = 0;
    if (loom_value_facts_as_exact_i64(loop->plan.pipeline.depth, &depth) &&
        depth > 1) {
      loop->flags |= LOOM_SCF_PIPELINE_LOOP_PREPARE_DESCENDANTS;
    }
    loop->plan.pipeline.step =
        loom_value_fact_table_lookup(facts, loom_scf_for_step(loop->source));
    loop->plan.pipeline.lower_bound = loom_value_fact_table_lookup(
        facts, loom_scf_for_lower_bound(loop->source));
    const loom_value_facts_t upper_bound = loom_value_fact_table_lookup(
        facts, loom_scf_for_upper_bound(loop->source));
    if (loom_value_facts_is_exact(loop->plan.pipeline.lower_bound)) {
      if (loom_value_facts_is_exact(upper_bound)) {
        loop->flags |= LOOM_SCF_PIPELINE_LOOP_STATIC_BOUNDS;
      }
    } else if (depth > 1) {
      int64_t step = 0;
      int64_t last_startup_offset = 0;
      int64_t last_startup = 0;
      if (!loom_value_facts_as_exact_i64(loop->plan.pipeline.step, &step) ||
          step <= 0 ||
          !iree_checked_mul_i64(depth - 2, step, &last_startup_offset) ||
          !iree_checked_add_i64(loop->plan.pipeline.lower_bound.range_hi,
                                last_startup_offset, &last_startup) ||
          last_startup >= upper_bound.range_lo) {
        loop->flags |= LOOM_SCF_PIPELINE_LOOP_REFINE_STARTUP_DOMAIN;
      }
    }
    const loom_scalar_type_t scalar_type = loom_type_element_type(
        loom_module_value_type(module, loom_scf_for_lower_bound(loop->source)));
    const int32_t bitwidth =
        loom_index_target_carrier_bitwidth(&facts->context, scalar_type);
    loop->plan.pipeline.maximum_value = INT64_MAX;
    if (bitwidth > 0 && bitwidth < 64) {
      const uint64_t maximum = (UINT64_C(1) << bitwidth) - 1;
      loop->plan.pipeline.maximum_value = scalar_type == LOOM_SCALAR_TYPE_OFFSET
                                              ? (int64_t)maximum
                                              : (int64_t)(maximum >> 1);
    }
    // Only scalar ranges and exactness are consumed. Table-local extension
    // identities do not cross the invalidation boundary.
    loop->plan.pipeline.depth.extension_id = 0;
    loop->plan.pipeline.step.extension_id = 0;
    loop->plan.pipeline.lower_bound.extension_id = 0;
  }
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_reject(loom_pass_t* pass,
                                              const loom_op_t* op,
                                              int64_t depth,
                                              iree_string_view_t constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("pipeline")),
      loom_param_i64(depth),
      loom_param_string(constraint),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_scf_pipeline_report(
    loom_scf_pipeline_context_t* context, iree_host_size_t loop_ordinal,
    uint32_t depth, const loom_scf_pipeline_plan_t* plan) {
  if (!loom_pass_report_is_enabled(context->pass)) {
    return iree_ok_status();
  }
  loom_pass_report_detail_field_t fields[] = {
      loom_pass_report_detail_uint64_field(IREE_SV("loop"), loop_ordinal),
      loom_pass_report_detail_string_field(
          IREE_SV("outcome"),
          depth == 1 ? IREE_SV("serial") : IREE_SV("pipelined")),
      loom_pass_report_detail_string_field(IREE_SV("schedule"),
                                           IREE_SV("read_ahead")),
      loom_pass_report_detail_uint64_field(IREE_SV("depth"), depth),
      loom_pass_report_detail_uint64_field(IREE_SV("queue_records"), depth - 1),
      loom_pass_report_detail_uint64_field(IREE_SV("values_per_record"),
                                           plan->queue_value_count),
      loom_pass_report_detail_uint64_field(IREE_SV("read_count"),
                                           plan->read_count),
  };
  IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
      context->pass, IREE_SV("scf-pipeline"), fields, IREE_ARRAYSIZE(fields)));
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    for (loom_scf_pipeline_stage_flags_t stage =
             LOOM_SCF_PIPELINE_STAGE_PRODUCER;
         stage <= LOOM_SCF_PIPELINE_STAGE_CONSUMER; stage <<= 1) {
      if (!iree_any_bit_set(plan->stages[i], stage)) {
        continue;
      }
      const bool producer = stage == LOOM_SCF_PIPELINE_STAGE_PRODUCER;
      loom_pass_report_detail_field_t stage_fields[] = {
          loom_pass_report_detail_uint64_field(IREE_SV("loop"), loop_ordinal),
          loom_pass_report_detail_uint64_field(IREE_SV("position"), i),
          loom_pass_report_detail_string_field(
              IREE_SV("op"),
              loom_op_name(context->module, plan->body.operations[i].op)),
          loom_pass_report_detail_string_field(
              IREE_SV("stage"),
              producer ? IREE_SV("producer") : IREE_SV("consumer")),
          loom_pass_report_detail_uint64_field(IREE_SV("iteration_lookahead"),
                                               producer ? depth - 1 : 0),
      };
      IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
          context->pass, IREE_SV("scf-pipeline-stage"), stage_fields,
          IREE_ARRAYSIZE(stage_fields)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_retain(
    loom_scf_pipeline_context_t* context, uint32_t depth,
    const loom_scf_pipeline_plan_t* plan) {
  loom_target_function_version_t* version =
      loom_target_function_version_cast(context->pass->function_version);
  loom_function_version_owner_t* owner =
      loom_target_pass_capability_function_version_owner(
          loom_target_pass_capability_from_pass(context->pass));
  if (!version || !owner) {
    return iree_ok_status();
  }

  loom_source_loop_pipeline_t* observation = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(owner->arena, sizeof(*observation),
                                           (void**)&observation));
  loom_source_loop_pipeline_operation_t* operations = NULL;
  const uint32_t operation_count =
      plan->body.count + plan->rematerialized_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      owner->arena, operation_count, sizeof(*operations), (void**)&operations));
  uint32_t operation_index = 0;
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    for (loom_scf_pipeline_stage_flags_t stage =
             LOOM_SCF_PIPELINE_STAGE_PRODUCER;
         stage <= LOOM_SCF_PIPELINE_STAGE_CONSUMER; stage <<= 1) {
      if (!iree_any_bit_set(plan->stages[i], stage)) {
        continue;
      }
      operations[operation_index++] = (loom_source_loop_pipeline_operation_t){
          .op_name = loom_op_name(context->module, plan->body.operations[i].op),
          .iteration_lookahead =
              stage == LOOM_SCF_PIPELINE_STAGE_PRODUCER ? depth - 1 : 0,
      };
    }
  }
  loom_source_loop_pipeline_list_t* list = &version->loop_pipelines;
  *observation = (loom_source_loop_pipeline_t){
      .loop_ordinal = list->tail ? list->tail->loop_ordinal + 1 : 0,
      .depth = depth,
      .values_per_record = plan->queue_value_count,
      .read_count = plan->read_count,
      .operations = operations,
      .operation_count = operation_count,
  };
  if (list->tail) {
    list->tail->next = observation;
  } else {
    list->head = observation;
  }
  list->tail = observation;
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_initialize_remap(
    loom_scf_pipeline_context_t* context, loom_ir_remap_t* out_remap) {
  return loom_ir_remap_initialize(
      context->module, context->module, context->arena,
      &(loom_ir_remap_options_t){.allow_unmapped_values = true}, out_remap);
}

// Returns true when rebuilding from the initial operands would discard the
// source loop's result type scheme. Ordinary invariant result types take the
// allocation-free builder path.
static bool loom_scf_pipeline_requires_result_scheme(
    const loom_scf_pipeline_context_t* context, const loom_op_t* source) {
  const loom_value_slice_t initial = loom_scf_for_iter_args(source);
  const loom_value_id_t* results = loom_op_const_results(source);
  for (uint16_t i = 0; i < source->result_count; ++i) {
    if (!loom_type_equal(
            loom_module_value_type(context->module, results[i]),
            loom_module_value_type(context->module, initial.values[i]))) {
      return true;
    }
  }
  return false;
}

// Reserves a rebuilt result tuple and projects the source result scheme onto
// its prefix. Appended pipeline records have iteration-invariant types by the
// planning contract and retain their types directly.
static iree_status_t loom_scf_pipeline_reserve_result_scheme(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    const loom_value_id_t* appended_values, uint16_t appended_count,
    loom_type_t** out_result_types) {
  const uint16_t result_count = source->result_count + appended_count;
  loom_type_t* result_types = NULL;
  loom_value_id_t* reserved_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, result_count,
                                                 sizeof(*result_types),
                                                 (void**)&result_types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, result_count,
                                                 sizeof(*reserved_results),
                                                 (void**)&reserved_results));

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_initialize_remap(context, &remap));
  IREE_RETURN_IF_ERROR(loom_builder_reserve_results(
      &context->rewriter->builder, result_count, reserved_results));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&remap, loom_op_const_results(source),
                               reserved_results, source->result_count));
  for (uint16_t i = 0; i < source->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap,
        loom_module_value_type(context->module,
                               loom_op_const_results(source)[i]),
        &result_types[i]));
  }
  for (uint16_t i = 0; i < appended_count; ++i) {
    result_types[source->result_count + i] =
        loom_module_value_type(context->module, appended_values[i]);
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

// Rebuilt loops preserve unroll intent and original carried slots at the front
// of the tuple. The pipeline operand is consumed exactly once by this pass.
static iree_status_t loom_scf_pipeline_build_loop(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    loom_value_id_t lower, const loom_value_id_t* iter_args,
    uint16_t iter_arg_count, loom_op_t** out_loop) {
  loom_scf_for_build_flags_t flags = 0;
  loom_value_id_t factor = LOOM_VALUE_ID_INVALID;
  uint8_t policy = 0;
  uint8_t schedule = 0;
  if (loom_scf_for_unroll_factor_is_present(source)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    factor = loom_scf_for_unroll_factor(source);
  }
  if (loom_scf_for_has_unroll_policy(source)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    policy = loom_scf_for_unroll_policy(source);
  }
  if (loom_scf_for_has_unroll_schedule(source)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    schedule = loom_scf_for_unroll_schedule(source);
  }
  loom_type_t* result_types = NULL;
  if (loom_scf_pipeline_requires_result_scheme(context, source)) {
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_reserve_result_scheme(
        context, source, iter_args + source->result_count,
        iter_arg_count - source->result_count, &result_types));
  }
  return loom_scf_for_build(
      &context->rewriter->builder, flags, lower,
      loom_scf_for_upper_bound(source), loom_scf_for_step(source), iter_args,
      iter_arg_count, result_types, loom_op_tied_results(source),
      source->tied_result_count, LOOM_VALUE_ID_INVALID, factor, policy,
      schedule, source->location, out_loop);
}

static iree_status_t loom_scf_pipeline_emit_serial(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    loom_op_t** out_loop) {
  loom_value_slice_t initial = loom_scf_for_iter_args(source);
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_build_loop(
      context, source, loom_scf_for_lower_bound(source), initial.values,
      initial.count, out_loop));
  const loom_block_t* source_block =
      loom_region_entry_block(loom_scf_for_body(source));
  loom_region_t* target_body = loom_scf_for_body(*out_loop);
  const loom_block_t* target_block = loom_region_entry_block(target_body);
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_initialize_remap(context, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_block->arg_ids,
                                                target_block->arg_ids,
                                                source_block->arg_count));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, *out_loop, target_body);
  remap.op_projection.entries = context->body->accesses.operations;
  remap.op_projection.count = context->body->accesses.count;
  remap.op_projection.cursor = 0;
  iree_status_t status = loom_ir_clone_block_ops(&context->rewriter->builder,
                                                 source_block, &remap, NULL);
  if (iree_status_is_ok(status)) {
    status = loom_scf_memory_project(context->spaces, &remap);
  }
  loom_builder_restore(&context->rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_scf_pipeline_emit_stage(
    loom_scf_pipeline_context_t* context, const loom_scf_pipeline_plan_t* plan,
    loom_scf_pipeline_stage_flags_t stage, loom_ir_remap_t* remap) {
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    if (!iree_any_bit_set(plan->stages[i], stage)) {
      continue;
    }
    remap->op_projection.entries = plan->body.accesses.units[i].count
                                       ? plan->body.accesses.operations +
                                             plan->body.accesses.units[i].begin
                                       : NULL;
    remap->op_projection.count = plan->body.accesses.units[i].count;
    remap->op_projection.cursor = 0;
    loom_op_t* clone = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_clone_op(&context->rewriter->builder,
                                          plan->body.operations[i].op, remap,
                                          &clone));
    IREE_RETURN_IF_ERROR(loom_scf_memory_project(context->spaces, remap));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_emit_producer(
    loom_scf_pipeline_context_t* context, const loom_scf_pipeline_plan_t* plan,
    const loom_block_t* source_block, loom_value_id_t induction_value,
    loom_ir_remap_t* remap, loom_value_id_t* out_record) {
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(remap, source_block->arg_ids[0],
                                               induction_value));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_stage(
      context, plan, LOOM_SCF_PIPELINE_STAGE_PRODUCER, remap));
  for (uint32_t i = 0; i < plan->queue_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        remap, plan->queue_values[i], &out_record[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_emit_consumer(
    loom_scf_pipeline_context_t* context, const loom_scf_pipeline_plan_t* plan,
    const loom_block_t* source_block, const loom_value_id_t* record,
    const loom_value_id_t* carried_values, loom_ir_remap_t* remap,
    loom_value_id_t* out_carried_values) {
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(remap, source_block->arg_ids + 1, carried_values,
                               source_block->arg_count - 1));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      remap, plan->queue_values, record, plan->queue_value_count));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_stage(
      context, plan, LOOM_SCF_PIPELINE_STAGE_CONSUMER, remap));
  const loom_value_slice_t yielded =
      loom_scf_yield_values(source_block->last_op);
  for (uint16_t i = 0; i < yielded.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, yielded.values[i],
                                                     &out_carried_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_constant(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    int64_t integer, loom_type_t type, loom_value_id_t* out_value) {
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(&context->rewriter->builder,
                                                 loom_attr_i64(integer), type,
                                                 source->location, &constant));
  *out_value = loom_index_constant_result(constant);
  return iree_ok_status();
}

// Clamp before adding so even the untaken pipeline path performs no signed
// overflow. The unclamped comparison rules out that clamped path, while the
// future-index comparison proves every startup load is in the source domain.
static iree_status_t loom_scf_pipeline_build_guard(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    int64_t advance, int64_t maximum_value, loom_value_facts_t lower_facts,
    loom_value_id_t* out_main_lower, loom_value_id_t* out_condition) {
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_value_id_t lower = loom_scf_for_lower_bound(source);
  const loom_type_t type = loom_module_value_type(context->module, lower);
  const bool is_offset =
      loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
  const bool lower_fits = lower_facts.range_hi <= maximum_value - advance;
  int64_t exact_lower = 0;
  loom_value_id_t fits_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t maximum_lower = LOOM_VALUE_ID_INVALID;
  loom_value_id_t advance_value = LOOM_VALUE_ID_INVALID;
  if (lower_fits && loom_value_facts_as_exact_i64(lower_facts, &exact_lower)) {
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_constant(
        context, source, exact_lower + advance, type, out_main_lower));
  } else {
    loom_value_id_t bounded_lower = lower;
    if (!lower_fits) {
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_constant(
          context, source, maximum_value - advance, type, &maximum_lower));
      loom_op_t* fits = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_cmp_build(builder,
                               is_offset ? LOOM_INDEX_CMP_PREDICATE_ULE
                                         : LOOM_INDEX_CMP_PREDICATE_SLE,
                               lower, maximum_lower, source->location, &fits));
      fits_value = loom_index_cmp_result(fits);
      loom_op_t* clamp = NULL;
      if (is_offset) {
        IREE_RETURN_IF_ERROR(loom_scf_select_build(builder, fits_value, lower,
                                                   maximum_lower, type,
                                                   source->location, &clamp));
        bounded_lower = loom_scf_select_result(clamp);
      } else {
        IREE_RETURN_IF_ERROR(loom_index_min_build(builder, lower, maximum_lower,
                                                  source->location, &clamp));
        bounded_lower = loom_index_min_result(clamp);
      }
    }
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_constant(context, source, advance,
                                                    type, &advance_value));
    loom_op_t* main_lower = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(builder, bounded_lower,
                                              advance_value, type,
                                              source->location, &main_lower));
    *out_main_lower = loom_index_add_result(main_lower);
  }
  loom_op_t* nonempty = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder,
      is_offset ? LOOM_INDEX_CMP_PREDICATE_ULT : LOOM_INDEX_CMP_PREDICATE_SLT,
      *out_main_lower, loom_scf_for_upper_bound(source), source->location,
      &nonempty));
  if (lower_fits) {
    *out_condition = loom_index_cmp_result(nonempty);
    return iree_ok_status();
  }
  loom_op_t* condition = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, fits_value, loom_index_cmp_result(nonempty),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), source->location, &condition));
  *out_condition = loom_scalar_andi_result(condition);
  return iree_ok_status();
}

// Retains the long-path startup domain when the source lower bound cannot
// carry it through ordinary scalar facts. Keep this uncommon dynamic-bound
// construction out of the exact-lower startup emission path.
IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_scf_pipeline_refine_startup_index(loom_scf_pipeline_context_t* context,
                                       const loom_op_t* source,
                                       loom_value_id_t index,
                                       loom_type_t index_type,
                                       loom_value_id_t* out_index) {
  const loom_predicate_t in_domain = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                   LOOM_PRED_ARG_NONE},
      .args = {index, loom_scf_for_upper_bound(source), 0},
  };
  loom_op_t* bounded_index = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(
      &context->rewriter->builder, &index, 1, &in_domain, 1, &index_type, 1,
      source->location, &bounded_index));
  *out_index = loom_index_assume_results(bounded_index).values[0];
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_emit_long_path(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    const loom_scf_pipeline_plan_t* plan, uint32_t depth, int64_t step,
    uint16_t state_count, bool refine_startup_domain,
    loom_value_id_t main_lower) {
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_block_t* source_block =
      loom_region_entry_block(loom_scf_for_body(source));
  const uint16_t carried_count = source->result_count;
  const uint32_t record_count = depth - 1;
  const uint32_t record_width = plan->queue_value_count;
  const loom_type_t index_type =
      loom_module_value_type(context->module, loom_scf_for_lower_bound(source));
  loom_value_id_t* initial_state = NULL;
  loom_value_id_t* next_state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, state_count,
                                                 sizeof(*initial_state),
                                                 (void**)&initial_state));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, state_count, sizeof(*next_state), (void**)&next_state));
  const loom_value_slice_t initial = loom_scf_for_iter_args(source);
  if (carried_count) {
    memcpy(initial_state, initial.values,
           carried_count * sizeof(*initial_state));
  }
  loom_ir_remap_t producer_remap = {0};
  loom_ir_remap_t consumer_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_initialize_remap(context, &producer_remap));
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_initialize_remap(context, &consumer_remap));
  for (uint32_t i = 0; i < record_count; ++i) {
    loom_value_id_t index = loom_scf_for_lower_bound(source);
    if (i != 0) {
      loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_constant(
          context, source, (int64_t)i * step, index_type, &offset));
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, index, offset, index_type, source->location, &add));
      index = loom_index_add_result(add);
    }
    if (refine_startup_domain) {
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_refine_startup_index(
          context, source, index, index_type, &index));
    }
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_producer(
        context, plan, source_block, index, &producer_remap,
        initial_state + carried_count + i * record_width));
  }

  loom_op_t* main_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_build_loop(
      context, source, main_lower, initial_state, state_count, &main_loop));
  const loom_block_t* main_block =
      loom_region_entry_block(loom_scf_for_body(main_loop));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      builder, main_loop, loom_scf_for_body(main_loop));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_producer(
      context, plan, source_block, main_block->arg_ids[0], &producer_remap,
      next_state + carried_count + (record_count - 1) * record_width));
  // The SSA queue preserves the stage relationship while independent producer
  // copies share a target scheduling region after unrolling.
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_consumer(
      context, plan, source_block, main_block->arg_ids + 1 + carried_count,
      main_block->arg_ids + 1, &consumer_remap, next_state));
  if (record_count > 1 && record_width != 0) {
    memcpy(next_state + carried_count,
           main_block->arg_ids + 1 + carried_count + record_width,
           (record_count - 1) * record_width * sizeof(*next_state));
  }
  loom_op_t* yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, next_state, state_count,
                                            source->location, &yield));
  loom_builder_restore(builder, saved_ip);

  // Keep the queue results immutable while replacing the carried recurrence
  // after each drained record. In particular, rotations resolve every input
  // before the next iteration installs any new block-argument mapping.
  const loom_value_id_t* results = loom_op_const_results(main_loop);
  const loom_value_id_t* carried_values = results;
  for (uint32_t i = 0; i < record_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_consumer(
        context, plan, source_block, results + carried_count + i * record_width,
        carried_values, &consumer_remap, next_state));
    carried_values = next_state;
  }
  return loom_scf_yield_build(builder, carried_values, carried_count,
                              source->location, &yield);
}

static iree_status_t loom_scf_pipeline_reconstruct(
    loom_scf_pipeline_context_t* context, loom_op_t* source,
    const loom_scf_pipeline_plan_t* plan, uint32_t depth, int64_t step,
    uint16_t state_count, bool refine_startup_domain, int64_t maximum_value,
    loom_value_facts_t lower_facts) {
  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, source);
  const loom_value_id_t checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* replacement = NULL;
  if (depth == 1) {
    IREE_RETURN_IF_ERROR(
        loom_scf_pipeline_emit_serial(context, source, &replacement));
  } else {
    loom_value_id_t main_lower = LOOM_VALUE_ID_INVALID;
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_build_guard(
        context, source, (int64_t)(depth - 1) * step, maximum_value,
        lower_facts, &main_lower, &condition));
    loom_type_t* result_types = NULL;
    if (source->result_count > 0 &&
        loom_scf_pipeline_requires_result_scheme(context, source)) {
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_reserve_result_scheme(
          context, source, NULL, 0, &result_types));
    } else if (source->result_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          context->arena, source->result_count, sizeof(*result_types),
          (void**)&result_types));
      const loom_value_id_t* source_results = loom_op_const_results(source);
      for (uint16_t i = 0; i < source->result_count; ++i) {
        result_types[i] =
            loom_module_value_type(context->module, source_results[i]);
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                          condition, result_types, source->result_count, NULL,
                          0, source->location, &replacement));
    loom_builder_ip_t saved_ip = loom_builder_enter_region(
        builder, replacement, loom_scf_if_then_region(replacement));
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_long_path(
        context, source, plan, depth, step, state_count, refine_startup_domain,
        main_lower));
    loom_builder_restore(builder, saved_ip);
    saved_ip = loom_builder_enter_region(builder, replacement,
                                         loom_scf_if_else_region(replacement));
    loom_op_t* serial = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_pipeline_emit_serial(context, source, &serial));
    loom_op_t* yield = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, loom_op_const_results(serial),
                             serial->result_count, source->location, &yield));
    loom_builder_restore(builder, saved_ip);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, source, loom_op_const_results(replacement),
      source->result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_retain(context, depth, plan));
  return loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, source, loom_op_const_results(replacement),
      source->result_count);
}

static iree_status_t loom_scf_pipeline_process_loop(
    loom_scf_pipeline_context_t* context, const loom_scf_pipeline_loop_t* loop,
    iree_host_size_t loop_ordinal) {
  loom_op_t* source = loop->source;
  int64_t depth = 0;
  if (!loom_value_facts_as_exact_i64(loop->plan.pipeline.depth, &depth)) {
    return loom_scf_pipeline_reject(context->pass, source, 0,
                                    IREE_SV("a compile-time exact depth"));
  }
  if (depth < 1 || depth > UINT16_MAX) {
    return loom_scf_pipeline_reject(context->pass, source, depth,
                                    IREE_SV("a depth in [1, 65535]"));
  }
  loom_scf_pipeline_plan_t plan = {0};
  int64_t step = 0;
  uint64_t state_count = source->result_count;
  if (depth > 1) {
    // The guarded scf.if has no operand slots for the source loop's storage
    // ties. Ordinary SSA recurrences retain their value flow through yields;
    // consuming ownership contracts require an explicit capture boundary.
    if (source->tied_result_count != 0) {
      return loom_scf_pipeline_reject(
          context->pass, source, depth,
          IREE_SV("loop results without consuming operand storage ties"));
    }
    if (!loom_value_facts_as_exact_i64(loop->plan.pipeline.step, &step) ||
        step <= 0) {
      return loom_scf_pipeline_reject(context->pass, source, depth,
                                      IREE_SV("a positive exact static step"));
    }
    if ((uint64_t)(depth - 1) >
        (uint64_t)loop->plan.pipeline.maximum_value / (uint64_t)step) {
      return loom_scf_pipeline_reject(
          context->pass, source, depth,
          IREE_SV("(depth - 1) * step representable in the address carrier"));
    }
    loom_scf_pipeline_rejection_t rejection = {0};
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_plan_build(
        context->module, loom_region_entry_block(loom_scf_for_body(source)),
        iree_any_bit_set(loop->flags, LOOM_SCF_PIPELINE_LOOP_STATIC_BOUNDS),
        context->spaces, context->arena, &plan, &rejection));
    if (rejection.op) {
      return loom_scf_pipeline_reject(context->pass, rejection.op, depth,
                                      rejection.constraint);
    }
    state_count += (uint64_t)(depth - 1) * plan.queue_value_count;
    // SCF stores operand and block-argument counts in uint16. Reserve the
    // bounds and optional unroll factor in addition to the carried tuple.
    if (state_count > UINT16_MAX - 4) {
      return loom_scf_pipeline_reject(
          context->pass, source, depth,
          IREE_SV("pipeline state and bounds fitting the SCF operand limit"));
    }
  }
  if (depth == 1) {
    const loom_op_t* unstructured_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_body_build(
        context->module, loom_region_entry_block(loom_scf_for_body(source)),
        context->spaces, LOOM_SCF_BODY_MODE_PROJECT_MEMORY, context->arena,
        &plan.body, &unstructured_op));
  }
  context->body = &plan.body;
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_report(context, loop_ordinal, (uint32_t)depth, &plan));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_reconstruct(
      context, source, &plan, (uint32_t)depth, step, (uint16_t)state_count,
      iree_any_bit_set(loop->flags,
                       LOOM_SCF_PIPELINE_LOOP_REFINE_STARTUP_DOMAIN),
      loop->plan.pipeline.maximum_value, loop->plan.pipeline.lower_bound));
  loom_scf_pipeline_statistics_t* statistics =
      loom_scf_pipeline_statistics(context->pass);
  if (depth == 1) {
    ++statistics->policies_cleared;
  } else {
    ++statistics->loops_pipelined;
  }
  loom_pass_mark_changed(context->pass);
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_prepare_loop(
    loom_scf_pipeline_context_t* context,
    const loom_scf_pipeline_loop_t* loop) {
  loom_scf_body_t body = {0};
  const loom_op_t* unstructured = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_body_build(
      context->module, loom_region_entry_block(loom_scf_for_body(loop->source)),
      context->spaces, LOOM_SCF_BODY_MODE_PROJECT_MEMORY, context->arena, &body,
      &unstructured));
  // Generic clone correspondence accepts repeated source records in emission
  // order, so the unroller needs no private access-space callback or analysis.
  iree_host_size_t count = 0;
  if (!iree_host_size_checked_mul(body.accesses.count, loop->plan.unroll.count,
                                  &count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "unroll operation correspondence size overflow");
  }
  loom_ir_remap_op_projection_t* entries = NULL;
  if (count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, count, sizeof(*entries), (void**)&entries));
    for (uint32_t i = 0; i < loop->plan.unroll.count; ++i) {
      memcpy(entries + i * body.accesses.count, body.accesses.operations,
             body.accesses.count * sizeof(*entries));
    }
  }
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, context->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
          .op_projection = {.entries = entries, .count = count},
      },
      &remap));
  IREE_RETURN_IF_ERROR(loom_scf_unroll_emit_full(
      context->pass, context->module, context->rewriter, loop->source,
      &loop->plan.unroll, &remap));
  IREE_RETURN_IF_ERROR(loom_scf_memory_project(context->spaces, &remap));
  loom_pass_mark_changed(context->pass);
  return iree_ok_status();
}

iree_status_t loom_scf_pipeline_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_scf_memory_t spaces = {.arena = pass->arena};
  loom_scf_pipeline_loop_list_t loops = {
      .arena = pass->arena,
      .spaces = &spaces,
      .module = module,
      .active = UINT32_MAX,
      .first = UINT32_MAX,
      .last = UINT32_MAX,
  };
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_pipeline_collect_loop,
                             .user_data = &loops},
      pass->arena, &result));
  loom_scf_pipeline_finish_scopes(&loops, 0);
  if (loops.pipeline_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_resolve_facts(pass, module, function, &loops));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(pass->arena->block_pool, &scratch_arena);
  loom_scf_pipeline_context_t context = {
      .spaces = &spaces,
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .arena = &scratch_arena,
  };
  iree_status_t status = iree_ok_status();
  iree_host_size_t pipeline_ordinal = 0;
  for (iree_host_size_t i = loops.first;
       i != UINT32_MAX && iree_status_is_ok(status) &&
       !loom_pass_has_error_diagnostics(pass);
       i = loops.values[i].next) {
    iree_arena_reset(&scratch_arena);
    const loom_scf_pipeline_loop_t* loop = &loops.values[i];
    if (loom_scf_for_pipeline_depth_is_present(loop->source)) {
      status =
          loom_scf_pipeline_process_loop(&context, loop, pipeline_ordinal++);
    } else if (iree_any_bit_set(loop->flags, LOOM_SCF_PIPELINE_LOOP_PREPARE)) {
      status = loom_scf_pipeline_prepare_loop(&context, loop);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
