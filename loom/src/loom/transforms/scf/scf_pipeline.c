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
  // Current invocation and its diagnostics/statistics/report sinks.
  loom_pass_t* pass;
  // Source and destination module.
  loom_module_t* module;
  // Mutation interface, with module-owned output IR.
  loom_rewriter_t* rewriter;
  // Scratch storage reset after reconstructing each source loop.
  iree_arena_allocator_t* arena;
} loom_scf_pipeline_context_t;

typedef struct loom_scf_pipeline_loop_t {
  // Annotated source loop, processed after its annotated children.
  loom_op_t* source;
  // Source policy facts retained independently of rewritten value IDs.
  loom_value_facts_t depth;
  // Source induction step facts.
  loom_value_facts_t step;
  // Source lower-bound facts used to construct the overflow-safe guard.
  loom_value_facts_t lower_bound;
  // Exact bounds give every participant the same main/drain iteration split.
  bool has_static_bounds;
  // Largest source-domain value representable by the selected address carrier.
  int64_t maximum_value;
} loom_scf_pipeline_loop_t;

typedef struct loom_scf_pipeline_loop_list_t {
  // Invocation arena owning the collected source handles.
  iree_arena_allocator_t* arena;
  // Annotated source loops, with children before parents.
  loom_scf_pipeline_loop_t* values;
  // Number of annotated loops.
  iree_host_size_t count;
  // Allocated source handle slots.
  iree_host_size_t capacity;
} loom_scf_pipeline_loop_list_t;

static iree_status_t loom_scf_pipeline_collect_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_for_isa(op) || !loom_scf_for_pipeline_depth_is_present(op)) {
    return iree_ok_status();
  }
  loom_scf_pipeline_loop_list_t* list = user_data;
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = (loom_scf_pipeline_loop_t){.source = op};
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
  for (iree_host_size_t i = 0; i < loops->count; ++i) {
    loom_scf_pipeline_loop_t* loop = &loops->values[i];
    loop->depth = loom_value_fact_table_lookup(
        facts, loom_scf_for_pipeline_depth(loop->source));
    loop->step =
        loom_value_fact_table_lookup(facts, loom_scf_for_step(loop->source));
    loop->lower_bound = loom_value_fact_table_lookup(
        facts, loom_scf_for_lower_bound(loop->source));
    loop->has_static_bounds =
        loom_value_facts_is_exact(loop->lower_bound) &&
        loom_value_facts_is_exact(loom_value_fact_table_lookup(
            facts, loom_scf_for_upper_bound(loop->source)));
    const loom_scalar_type_t scalar_type = loom_type_element_type(
        loom_module_value_type(module, loom_scf_for_lower_bound(loop->source)));
    const int32_t bitwidth =
        loom_index_target_carrier_bitwidth(&facts->context, scalar_type);
    loop->maximum_value = INT64_MAX;
    if (bitwidth > 0 && bitwidth < 64) {
      const uint64_t maximum = (UINT64_C(1) << bitwidth) - 1;
      loop->maximum_value = scalar_type == LOOM_SCALAR_TYPE_OFFSET
                                ? (int64_t)maximum
                                : (int64_t)(maximum >> 1);
    }
    // Only scalar ranges and exactness are consumed. Table-local extension
    // identities do not cross the invalidation boundary.
    loop->depth.extension_id = 0;
    loop->step.extension_id = 0;
    loop->lower_bound.extension_id = 0;
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
    const bool producer = plan->stages[i] == LOOM_SCF_PIPELINE_STAGE_PRODUCER;
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
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(owner->arena, plan->body.count,
                                                 sizeof(*operations),
                                                 (void**)&operations));
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    operations[i] = (loom_source_loop_pipeline_operation_t){
        .op_name = loom_op_name(context->module, plan->body.operations[i].op),
        .iteration_lookahead =
            plan->stages[i] == LOOM_SCF_PIPELINE_STAGE_PRODUCER ? depth - 1 : 0,
    };
  }
  loom_source_loop_pipeline_list_t* list = &version->loop_pipelines;
  *observation = (loom_source_loop_pipeline_t){
      .loop_ordinal = list->tail ? list->tail->loop_ordinal + 1 : 0,
      .depth = depth,
      .values_per_record = plan->queue_value_count,
      .read_count = plan->read_count,
      .operations = operations,
      .operation_count = plan->body.count,
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
  if (!loom_attr_is_absent(
          loom_op_const_attrs(source)[loom_scf_for_unroll_policy_ATTR_INDEX])) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    policy = loom_scf_for_unroll_policy(source);
  }
  if (!loom_attr_is_absent(loom_op_const_attrs(
          source)[loom_scf_for_unroll_schedule_ATTR_INDEX])) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    schedule = loom_scf_for_unroll_schedule(source);
  }
  return loom_scf_for_build(
      &context->rewriter->builder, flags, lower,
      loom_scf_for_upper_bound(source), loom_scf_for_step(source), iter_args,
      iter_arg_count, loom_op_tied_results(source), source->tied_result_count,
      LOOM_VALUE_ID_INVALID, factor, policy, schedule, source->location,
      out_loop);
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
  iree_status_t status = loom_ir_clone_block_ops(&context->rewriter->builder,
                                                 source_block, &remap, NULL);
  loom_builder_restore(&context->rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_scf_pipeline_emit_stage(
    loom_scf_pipeline_context_t* context, const loom_scf_pipeline_plan_t* plan,
    loom_scf_pipeline_stage_t stage, loom_ir_remap_t* remap) {
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    if (plan->stages[i] != stage) {
      continue;
    }
    loom_op_t* clone = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_clone_op(&context->rewriter->builder,
                                          plan->body.operations[i].op, remap,
                                          &clone));
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

static iree_status_t loom_scf_pipeline_emit_long_path(
    loom_scf_pipeline_context_t* context, const loom_op_t* source,
    const loom_scf_pipeline_plan_t* plan, uint32_t depth, int64_t step,
    uint16_t state_count, loom_value_id_t main_lower) {
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
    uint16_t state_count, int64_t maximum_value,
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
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, source->result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* source_results = loom_op_const_results(source);
    for (uint16_t i = 0; i < source->result_count; ++i) {
      result_types[i] =
          loom_module_value_type(context->module, source_results[i]);
    }
    IREE_RETURN_IF_ERROR(
        loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                          condition, result_types, source->result_count, NULL,
                          0, source->location, &replacement));
    loom_builder_ip_t saved_ip = loom_builder_enter_region(
        builder, replacement, loom_scf_if_then_region(replacement));
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_emit_long_path(
        context, source, plan, depth, step, state_count, main_lower));
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
  if (!loom_value_facts_as_exact_i64(loop->depth, &depth)) {
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
    if (!loom_value_facts_as_exact_i64(loop->step, &step) || step <= 0) {
      return loom_scf_pipeline_reject(context->pass, source, depth,
                                      IREE_SV("a positive exact static step"));
    }
    if ((uint64_t)(depth - 1) >
        (uint64_t)loop->maximum_value / (uint64_t)step) {
      return loom_scf_pipeline_reject(
          context->pass, source, depth,
          IREE_SV("(depth - 1) * step representable in the address carrier"));
    }
    loom_scf_pipeline_rejection_t rejection = {0};
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_plan_build(
        context->module, loom_region_entry_block(loom_scf_for_body(source)),
        loop->has_static_bounds, context->arena, &plan, &rejection));
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
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_report(context, loop_ordinal, (uint32_t)depth, &plan));
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_reconstruct(
      context, source, &plan, (uint32_t)depth, step, (uint16_t)state_count,
      loop->maximum_value, loop->lower_bound));
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

iree_status_t loom_scf_pipeline_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_scf_pipeline_loop_list_t loops = {.arena = pass->arena};
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_POST_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_pipeline_collect_loop,
                             .user_data = &loops},
      pass->arena, &result));
  if (loops.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_scf_pipeline_resolve_facts(pass, module, function, &loops));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(pass->arena->block_pool, &scratch_arena);
  loom_scf_pipeline_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .arena = &scratch_arena,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < loops.count && iree_status_is_ok(status) &&
                               !loom_pass_has_error_diagnostics(pass);
       ++i) {
    iree_arena_reset(&scratch_arena);
    status = loom_scf_pipeline_process_loop(&context, &loops.values[i], i);
  }
  iree_arena_deinitialize(&scratch_arena);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
