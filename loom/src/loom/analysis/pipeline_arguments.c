// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_arguments.h"

#include <stdlib.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"

static iree_status_t loom_pipeline_arguments_reject(
    const loom_op_t* call, iree_string_view_t requirement,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_diagnostic_param_t parameters[] = {loom_param_string(requirement)};
  return iree_diagnostic_emit(diagnostic_emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = call,
                                  .error = LOOM_ERR_LOWERING_066,
                                  .params = parameters,
                                  .param_count = IREE_ARRAYSIZE(parameters),
                              });
}

const loom_pipeline_channel_binding_t* loom_pipeline_arguments_lookup_channel(
    const loom_pipeline_channel_binding_t* bindings,
    iree_host_size_t binding_count, loom_value_id_t value_id) {
  iree_host_size_t begin = 0, end = binding_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_channel_binding_t* binding = &bindings[middle];
    if (binding->value_id == value_id) {
      return binding;
    }
    if (binding->value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

static int loom_pipeline_arguments_compare_channel(const void* lhs,
                                                   const void* rhs) {
  const loom_value_id_t a =
      ((const loom_pipeline_channel_binding_t*)lhs)->value_id;
  const loom_value_id_t b =
      ((const loom_pipeline_channel_binding_t*)rhs)->value_id;
  return (a > b) - (a < b);
}

static int loom_pipeline_arguments_compare_pool(const void* lhs,
                                                const void* rhs) {
  const loom_value_id_t a =
      ((const loom_pipeline_resource_pool_binding_t*)lhs)->value_id;
  const loom_value_id_t b =
      ((const loom_pipeline_resource_pool_binding_t*)rhs)->value_id;
  return (a > b) - (a < b);
}

iree_status_t loom_pipeline_arguments_bind(
    loom_module_t* module, loom_op_t* call, loom_func_like_t callee,
    const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* caller_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count, loom_value_fact_table_t* callee_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_arguments_t* out_arguments, bool* out_valid) {
  *out_arguments = (loom_pipeline_arguments_t){0};
  *out_valid = false;
  const loom_value_slice_t actuals =
      loom_call_like_operands(loom_call_like_cast(module, call));
  uint16_t formal_count = 0;
  const loom_value_id_t* formals =
      loom_func_like_arg_ids(callee, &formal_count);
  if (formal_count == 0) {
    *out_valid = true;
    return iree_ok_status();
  }
  loom_value_fact_table_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*inputs), (void**)&inputs));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(inputs, arena, 0));
  inputs->context.resolve_type_domain =
      caller_facts->context.resolve_type_domain;
  loom_pipeline_channel_binding_t* channels = NULL;
  loom_pipeline_resource_pool_binding_t* pools = NULL;
  iree_host_size_t channel_capacity = 0, pool_capacity = 0;
  const loom_op_t** literal_captures = NULL;
  for (uint16_t i = 0; i < formal_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, formals[i]);
    loom_value_facts_t facts;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        inputs, caller_facts, module, type,
        loom_value_fact_table_lookup(caller_facts, actuals.values[i]), &facts));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(inputs, formals[i], facts));
    if (!loom_value_facts_can_materialize_constant(facts, type) &&
        !loom_type_may_reference_values(type)) {
      const loom_value_t* actual = loom_module_value(module, actuals.values[i]);
      const loom_op_t* literal =
          loom_value_is_block_arg(actual) ? NULL : loom_value_def_op(actual);
      if (literal &&
          iree_all_bits_set(literal->traits, LOOM_TRAIT_CONSTANT_LIKE) &&
          !loom_type_may_reference_values(
              loom_module_value_type(module, actuals.values[i]))) {
        if (!literal_captures) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              arena, formal_count, sizeof(*literal_captures),
              (void**)&literal_captures));
          memset(literal_captures, 0, formal_count * sizeof(*literal_captures));
        }
        literal_captures[i] = literal;
      }
    }
    if (loom_channel_type_isa(type)) {
      loom_pipeline_channel_binding_t binding = {
          .value_id = formals[i],
          .resources = resources,
          .channel = loom_pipeline_resources_lookup_channel(resources,
                                                            actuals.values[i]),
      };
      if (!binding.channel) {
        const loom_pipeline_channel_binding_t* parent =
            loom_pipeline_arguments_lookup_channel(incoming, incoming_count,
                                                   actuals.values[i]);
        if (parent) {
          binding.resources = parent->resources;
          binding.channel = parent->channel;
        }
      }
      if (!binding.channel) {
        return loom_pipeline_arguments_reject(
            call, IREE_SV("an incoming channel construction binding"),
            diagnostic_emitter);
      }
      if (out_arguments->channel_count == channel_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, out_arguments->channel_count,
            out_arguments->channel_count + 1, sizeof(*channels),
            &channel_capacity, (void**)&channels));
      }
      channels[out_arguments->channel_count++] = binding;
    } else if (loom_type_is_pool(type)) {
      const loom_pipeline_resource_pool_binding_t* parent =
          loom_pipeline_resources_lookup_pool(resources, actuals.values[i]);
      if (!parent) {
        continue;
      }
      if (out_arguments->pool_count == pool_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, out_arguments->pool_count, out_arguments->pool_count + 1,
            sizeof(*pools), &pool_capacity, (void**)&pools));
      }
      pools[out_arguments->pool_count++] =
          (loom_pipeline_resource_pool_binding_t){
              .value_id = formals[i], .pool_index = parent->pool_index};
    }
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_values(
      callee_facts,
      (loom_value_fact_table_view_t){inputs, formals, formal_count}, module));
  if (out_arguments->channel_count > 1) {
    qsort(channels, out_arguments->channel_count, sizeof(*channels),
          loom_pipeline_arguments_compare_channel);
  }
  if (out_arguments->pool_count > 1) {
    qsort(pools, out_arguments->pool_count, sizeof(*pools),
          loom_pipeline_arguments_compare_pool);
  }
  out_arguments->channels = channels;
  out_arguments->literal_captures = literal_captures;
  out_arguments->pools = pools;
  *out_valid = true;
  return iree_ok_status();
}
