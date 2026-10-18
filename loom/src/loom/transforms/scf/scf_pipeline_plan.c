// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_pipeline_plan.h"

#include "loom/ir/local_value_domain.h"
#include "loom/ir/types.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"

static bool loom_scf_pipeline_reference_depends_on_carried(
    const loom_local_value_domain_t* domain, const bool* depends_on_carried,
    const loom_scf_body_reference_t* reference) {
  if (reference->allow_identity_mapping) {
    return false;
  }
  return depends_on_carried[loom_local_value_domain_ordinal(
      domain, reference->value_id)];
}

static bool loom_scf_pipeline_operation_depends_on_carried(
    const loom_scf_body_t* body, const loom_scf_body_operation_t* operation,
    const loom_local_value_domain_t* domain, const bool* depends_on_carried) {
  for (iree_host_size_t i = 0; i < operation->reference_count; ++i) {
    if (loom_scf_pipeline_reference_depends_on_carried(
            domain, depends_on_carried,
            &body->references[operation->reference_begin + i])) {
      return true;
    }
  }
  return false;
}

static void loom_scf_pipeline_publish_result_dependence(
    const loom_local_value_domain_t* domain, const loom_op_t* op, bool depends,
    bool* depends_on_carried) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    depends_on_carried[loom_local_value_domain_ordinal(domain, results[i])] =
        depends;
  }
}

static bool loom_scf_pipeline_value_is_scoped_reference(
    const loom_module_t* module, const loom_block_t* block,
    const loom_op_t* owner, loom_value_id_t value_id,
    bool* out_allow_identity_mapping) {
  *out_allow_identity_mapping = false;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value) == block;
  }
  const loom_op_t* definition = loom_value_def_op(value);
  if (!definition || definition == owner) {
    return false;
  }
  if (!definition->parent_block) {
    *out_allow_identity_mapping = true;
    return true;
  }
  return definition->parent_block == block;
}

static iree_status_t loom_scf_pipeline_guarded_append_reference(
    iree_arena_allocator_t* arena,
    loom_scf_pipeline_guarded_partition_t* partition,
    iree_host_size_t* reference_count, iree_host_size_t* reference_capacity,
    loom_scf_body_reference_t reference) {
  if (*reference_count == *reference_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, *reference_count, *reference_count + 1,
        sizeof(*partition->references), reference_capacity,
        (void**)&partition->references));
  }
  partition->references[(*reference_count)++] = reference;
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_guarded_append_operation_references(
    iree_arena_allocator_t* arena,
    loom_scf_pipeline_guarded_partition_t* partition,
    iree_host_size_t* reference_count, iree_host_size_t* reference_capacity,
    const loom_scf_body_t* body, const loom_scf_body_operation_t* operation) {
  for (iree_host_size_t i = 0; i < operation->reference_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_reference(
        arena, partition, reference_count, reference_capacity,
        body->references[operation->reference_begin + i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_pipeline_guarded_append_scoped_value(
    const loom_module_t* module, const loom_block_t* block,
    const loom_op_t* owner, loom_value_id_t value_id,
    iree_arena_allocator_t* arena,
    loom_scf_pipeline_guarded_partition_t* partition,
    iree_host_size_t* reference_count, iree_host_size_t* reference_capacity) {
  bool allow_identity_mapping = false;
  if (!loom_scf_pipeline_value_is_scoped_reference(
          module, block, owner, value_id, &allow_identity_mapping)) {
    return iree_ok_status();
  }
  return loom_scf_pipeline_guarded_append_reference(
      arena, partition, reference_count, reference_capacity,
      (loom_scf_body_reference_t){
          .value_id = value_id,
          .allow_identity_mapping = allow_identity_mapping,
      });
}

static bool loom_scf_pipeline_guarded_consumer_depends_on_carried(
    const loom_module_t* module, const loom_block_t* block, const loom_op_t* op,
    const loom_scf_body_t* producer_body,
    const loom_scf_pipeline_stage_flags_t* producer_branch_stages,
    const loom_scf_body_t* other_body, const loom_local_value_domain_t* domain,
    const bool* depends_on_carried) {
  const loom_value_id_t condition = loom_scf_if_condition(op);
  bool allow_identity_mapping = false;
  if (loom_scf_pipeline_value_is_scoped_reference(module, block, op, condition,
                                                  &allow_identity_mapping) &&
      !allow_identity_mapping &&
      depends_on_carried[loom_local_value_domain_ordinal(domain, condition)]) {
    return true;
  }
  for (uint32_t i = 0; i < producer_body->count; ++i) {
    if (producer_branch_stages[i] == LOOM_SCF_PIPELINE_STAGE_CONSUMER &&
        loom_scf_pipeline_operation_depends_on_carried(
            producer_body, &producer_body->operations[i], domain,
            depends_on_carried)) {
      return true;
    }
  }
  if (loom_scf_pipeline_operation_depends_on_carried(
          producer_body, &producer_body->terminator, domain,
          depends_on_carried)) {
    return true;
  }
  for (uint32_t i = 0; i < other_body->count; ++i) {
    if (loom_scf_pipeline_operation_depends_on_carried(
            other_body, &other_body->operations[i], domain,
            depends_on_carried)) {
      return true;
    }
  }
  if (loom_scf_pipeline_operation_depends_on_carried(
          other_body, &other_body->terminator, domain, depends_on_carried)) {
    return true;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module, results[i], &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      if (provider != results[i] &&
          depends_on_carried[loom_local_value_domain_ordinal(domain,
                                                             provider)]) {
        return true;
      }
    }
  }
  return false;
}

static void loom_scf_pipeline_clear_branch_owners(
    const loom_local_value_domain_t* domain, const loom_scf_body_t* body,
    uint32_t* branch_owners) {
  for (uint32_t i = 0; i < body->count; ++i) {
    const loom_op_t* op = body->operations[i].op;
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t j = 0; j < op->result_count; ++j) {
      branch_owners[loom_local_value_domain_ordinal(domain, results[j])] =
          UINT32_MAX;
    }
  }
}

// Builds a retained branch cut only when consumer work depends on the outer
// carried tuple. Producer-only conditionals keep the cheaper atomic schedule.
static iree_status_t loom_scf_pipeline_try_build_guarded_partition(
    loom_module_t* module, const loom_block_t* block, const loom_op_t* op,
    const loom_scf_memory_t* spaces, const loom_local_value_domain_t* domain,
    bool has_static_bounds, bool* depends_on_carried, uint32_t* branch_owners,
    iree_arena_allocator_t* arena,
    loom_scf_pipeline_guarded_partition_t** out_partition) {
  *out_partition = NULL;
  if (!loom_scf_if_isa(op) || !loom_scf_if_else_region(op)) {
    return iree_ok_status();
  }

  loom_scf_body_t branches[LOOM_SCF_PIPELINE_BRANCH_COUNT] = {0};
  const loom_region_t* regions[LOOM_SCF_PIPELINE_BRANCH_COUNT] = {
      loom_scf_if_then_region(op),
      loom_scf_if_else_region(op),
  };
  for (uint8_t i = 0; i < LOOM_SCF_PIPELINE_BRANCH_COUNT; ++i) {
    const loom_op_t* unstructured_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_body_build(
        module, loom_region_const_entry_block(regions[i]), block, spaces,
        LOOM_SCF_BODY_MODE_SCHEDULE, arena, &branches[i], &unstructured_op));
    if (unstructured_op) {
      return iree_ok_status();
    }
  }

  uint32_t branch_read_counts[LOOM_SCF_PIPELINE_BRANCH_COUNT] = {0};
  for (uint8_t branch = 0; branch < LOOM_SCF_PIPELINE_BRANCH_COUNT; ++branch) {
    for (uint32_t i = 0; i < branches[branch].count; ++i) {
      branch_read_counts[branch] += branches[branch].operations[i].load_count;
    }
  }
  if ((branch_read_counts[0] == 0) == (branch_read_counts[1] == 0)) {
    return iree_ok_status();
  }
  const uint8_t producer_branch = branch_read_counts[0] ? 0 : 1;
  const uint8_t other_branch = producer_branch ^ 1;
  loom_scf_body_t* producer_body = &branches[producer_branch];
  loom_scf_body_t* other_body = &branches[other_branch];
  for (uint32_t i = 0; i < producer_body->count; ++i) {
    const loom_scf_body_effect_flags_t effects =
        producer_body->operations[i].effects;
    if (effects == LOOM_SCF_BODY_EFFECT_CONVERGENT && has_static_bounds) {
      continue;
    }
    if (effects != 0 && effects != LOOM_SCF_BODY_EFFECT_READ) {
      return iree_ok_status();
    }
  }
  for (uint32_t i = 0; i < other_body->count; ++i) {
    const loom_scf_body_effect_flags_t effects =
        other_body->operations[i].effects;
    if (effects != 0 &&
        (effects != LOOM_SCF_BODY_EFFECT_CONVERGENT || !has_static_bounds)) {
      return iree_ok_status();
    }
  }

  loom_scf_pipeline_stage_flags_t* branch_stages = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, producer_body->count,
                                                 sizeof(*branch_stages),
                                                 (void**)&branch_stages));
  for (uint32_t i = 0; i < producer_body->count; ++i) {
    branch_stages[i] =
        producer_body->operations[i].effects == LOOM_SCF_BODY_EFFECT_READ
            ? LOOM_SCF_PIPELINE_STAGE_PRODUCER
            : LOOM_SCF_PIPELINE_STAGE_CONSUMER;
    const loom_value_id_t* results =
        loom_op_const_results(producer_body->operations[i].op);
    for (uint16_t j = 0; j < producer_body->operations[i].op->result_count;
         ++j) {
      branch_owners[loom_local_value_domain_ordinal(domain, results[j])] = i;
    }
  }

  for (uint32_t reverse = producer_body->count; reverse > 0; --reverse) {
    const uint32_t i = reverse - 1;
    if (branch_stages[i] != LOOM_SCF_PIPELINE_STAGE_PRODUCER) {
      continue;
    }
    const loom_scf_body_operation_t* operation = &producer_body->operations[i];
    for (iree_host_size_t j = 0; j < operation->reference_count; ++j) {
      const loom_scf_body_reference_t* reference =
          &producer_body->references[operation->reference_begin + j];
      if (reference->allow_identity_mapping) {
        continue;
      }
      const uint32_t producer = branch_owners[loom_local_value_domain_ordinal(
          domain, reference->value_id)];
      if (producer == UINT32_MAX) {
        continue;
      }
      if (producer_body->operations[producer].effects != 0 &&
          branch_stages[producer] != LOOM_SCF_PIPELINE_STAGE_PRODUCER) {
        loom_scf_pipeline_clear_branch_owners(domain, producer_body,
                                              branch_owners);
        return iree_ok_status();
      }
      branch_stages[producer] = LOOM_SCF_PIPELINE_STAGE_PRODUCER;
    }
  }
  loom_scf_pipeline_clear_branch_owners(domain, producer_body, branch_owners);

  for (uint8_t branch = 0; branch < LOOM_SCF_PIPELINE_BRANCH_COUNT; ++branch) {
    loom_scf_body_t* branch_body = &branches[branch];
    for (uint32_t i = 0; i < branch_body->count; ++i) {
      const bool depends = loom_scf_pipeline_operation_depends_on_carried(
          branch_body, &branch_body->operations[i], domain, depends_on_carried);
      loom_scf_pipeline_publish_result_dependence(
          domain, branch_body->operations[i].op, depends, depends_on_carried);
    }
  }
  if (!loom_scf_pipeline_guarded_consumer_depends_on_carried(
          module, block, op, producer_body, branch_stages, other_body, domain,
          depends_on_carried)) {
    return iree_ok_status();
  }

  loom_scf_pipeline_guarded_partition_t* partition = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*partition), (void**)&partition));
  *partition = (loom_scf_pipeline_guarded_partition_t){
      .producer_branch_stages = branch_stages,
      .producer_branch = producer_branch,
  };
  memcpy(partition->branches, branches, sizeof(branches));

  iree_host_size_t reference_count = 0;
  iree_host_size_t reference_capacity = 0;
  partition->producer = (loom_scf_body_operation_t){
      .op = op,
      .reference_begin = reference_count,
      .effects = LOOM_SCF_BODY_EFFECT_READ,
  };
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_scoped_value(
      module, block, op, loom_scf_if_condition(op), arena, partition,
      &reference_count, &reference_capacity));
  for (uint32_t i = 0; i < producer_body->count; ++i) {
    if (branch_stages[i] != LOOM_SCF_PIPELINE_STAGE_PRODUCER) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_operation_references(
        arena, partition, &reference_count, &reference_capacity, producer_body,
        &producer_body->operations[i]));
    partition->producer.load_count += producer_body->operations[i].load_count;
  }
  partition->producer.reference_count =
      reference_count - partition->producer.reference_begin;

  partition->consumer = (loom_scf_body_operation_t){
      .op = op,
      .reference_begin = reference_count,
  };
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_scoped_value(
      module, block, op, loom_scf_if_condition(op), arena, partition,
      &reference_count, &reference_capacity));
  for (uint32_t i = 0; i < producer_body->count; ++i) {
    if (branch_stages[i] != LOOM_SCF_PIPELINE_STAGE_CONSUMER) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_operation_references(
        arena, partition, &reference_count, &reference_capacity, producer_body,
        &producer_body->operations[i]));
    partition->consumer.effects |= producer_body->operations[i].effects;
  }
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_operation_references(
      arena, partition, &reference_count, &reference_capacity, producer_body,
      &producer_body->terminator));
  for (uint32_t i = 0; i < other_body->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_operation_references(
        arena, partition, &reference_count, &reference_capacity, other_body,
        &other_body->operations[i]));
    partition->consumer.effects |= other_body->operations[i].effects;
  }
  IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_operation_references(
      arena, partition, &reference_count, &reference_capacity, other_body,
      &other_body->terminator));
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module, results[i], &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_guarded_append_scoped_value(
          module, block, op, provider, arena, partition, &reference_count,
          &reference_capacity));
    }
  }
  partition->consumer.reference_count =
      reference_count - partition->consumer.reference_begin;
  *out_partition = partition;
  return iree_ok_status();
}

static const loom_scf_body_operation_t* loom_scf_pipeline_stage_operation(
    const loom_scf_pipeline_plan_t* plan, uint32_t operation,
    loom_scf_pipeline_stage_flags_t stage,
    const loom_scf_body_reference_t** out_references) {
  const loom_scf_pipeline_guarded_partition_t* partition =
      plan->guarded_partitions ? plan->guarded_partitions[operation] : NULL;
  if (partition) {
    *out_references = partition->references;
    return stage == LOOM_SCF_PIPELINE_STAGE_PRODUCER ? &partition->producer
                                                     : &partition->consumer;
  }
  *out_references = plan->body.references;
  return &plan->body.operations[operation];
}

static loom_scf_pipeline_stage_flags_t loom_scf_pipeline_definition_stages(
    const loom_scf_pipeline_plan_t* plan, const uint32_t* definition_owners,
    const loom_scf_pipeline_stage_flags_t* definition_fixed_stages,
    loom_value_ordinal_t ordinal) {
  const uint32_t operation = definition_owners[ordinal];
  if (operation == UINT32_MAX) {
    return 0;
  }
  const loom_scf_pipeline_stage_flags_t fixed_stage =
      definition_fixed_stages ? definition_fixed_stages[ordinal] : 0;
  return fixed_stage ? fixed_stage : plan->stages[operation];
}

static bool loom_scf_pipeline_has_guard_placeholder(loom_type_t type) {
  if (!loom_type_is_scalar(type) && !loom_type_is_vector(type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  return loom_scalar_type_is_float(element_type) ||
         loom_scalar_type_is_integer(element_type) ||
         element_type == LOOM_SCALAR_TYPE_INDEX ||
         element_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_scf_pipeline_is_guarded_candidate(
    const loom_scf_body_operation_t* operation) {
  const loom_scf_body_effect_flags_t candidate_effects =
      LOOM_SCF_BODY_EFFECT_READ | LOOM_SCF_BODY_EFFECT_CONVERGENT;
  return (operation->effects == LOOM_SCF_BODY_EFFECT_READ ||
          operation->effects == candidate_effects) &&
         loom_scf_if_isa(operation->op);
}

static bool loom_scf_pipeline_has_guarded_candidate(
    const loom_scf_pipeline_plan_t* plan) {
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    if (loom_scf_pipeline_is_guarded_candidate(&plan->body.operations[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_scf_pipeline_plan_partition(
    loom_module_t* module, const loom_block_t* block,
    const loom_local_value_domain_t* domain, const loom_scf_memory_t* spaces,
    bool has_static_bounds, bool has_guarded_candidate,
    iree_arena_allocator_t* arena, loom_scf_pipeline_plan_t* plan,
    loom_scf_pipeline_rejection_t* rejection) {
  bool ordered_memory = false;
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    const loom_scf_body_operation_t* operation = &plan->body.operations[i];
    if (!iree_any_bit_set(
            operation->effects,
            LOOM_SCF_BODY_EFFECT_WRITE | LOOM_SCF_BODY_EFFECT_ORDERED)) {
      continue;
    }
    ordered_memory = true;
    if (plan->body.accesses.units[i].effects !=
        LOOM_SCF_BODY_MEMORY_WORKGROUP) {
      *rejection = (loom_scf_pipeline_rejection_t){
          .op = operation->op,
          .constraint = IREE_SV("workgroup-only stores and barriers in the "
                                "ordered consumer"),
      };
      return iree_ok_status();
    }
    if (!has_static_bounds) {
      *rejection = (loom_scf_pipeline_rejection_t){
          .op = operation->op,
          .constraint = IREE_SV("compile-time exact loop bounds to preserve "
                                "ordered consumer participation"),
      };
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->body.count, sizeof(*plan->stages), (void**)&plan->stages));
  uint32_t* branch_owners = NULL;
  bool* depends_on_carried = NULL;
  if (has_guarded_candidate) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, plan->body.count, sizeof(*plan->guarded_partitions),
        (void**)&plan->guarded_partitions));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain->value_count,
                                                   sizeof(*branch_owners),
                                                   (void**)&branch_owners));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, domain->value_count, sizeof(*depends_on_carried),
        (void**)&depends_on_carried));
    memset(plan->guarded_partitions, 0,
           plan->body.count * sizeof(*plan->guarded_partitions));
    memset(branch_owners, 0xFF, domain->value_count * sizeof(*branch_owners));
    memset(depends_on_carried, 0,
           domain->value_count * sizeof(*depends_on_carried));
    for (uint16_t i = 1; i < block->arg_count; ++i) {
      depends_on_carried[loom_local_value_domain_ordinal(
          domain, block->arg_ids[i])] = true;
    }
    for (uint32_t i = 0; i < plan->body.count; ++i) {
      const loom_scf_body_operation_t* operation = &plan->body.operations[i];
      const bool depends = loom_scf_pipeline_operation_depends_on_carried(
          &plan->body, operation, domain, depends_on_carried);
      loom_scf_pipeline_publish_result_dependence(domain, operation->op,
                                                  depends, depends_on_carried);
    }
    for (uint32_t i = 0; i < plan->body.count; ++i) {
      const loom_scf_body_operation_t* operation = &plan->body.operations[i];
      if (!loom_scf_pipeline_is_guarded_candidate(operation)) {
        continue;
      }
      // Workgroup reads observe the current ordered consumer iteration. Keep
      // their entire conditional intact instead of constructing a read-ahead
      // cut.
      if (ordered_memory && plan->body.accesses.units[i].effects ==
                                LOOM_SCF_BODY_MEMORY_WORKGROUP) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_scf_pipeline_try_build_guarded_partition(
          module, block, operation->op, spaces, domain, has_static_bounds,
          depends_on_carried, branch_owners, arena,
          &plan->guarded_partitions[i]));
      plan->guarded_partition_count += plan->guarded_partitions[i] != NULL;
    }
  }

  for (uint32_t i = 0; i < plan->body.count; ++i) {
    const loom_scf_body_operation_t* operation = &plan->body.operations[i];
    loom_scf_pipeline_guarded_partition_t* guarded_partition =
        plan->guarded_partitions ? plan->guarded_partitions[i] : NULL;
    plan->stages[i] = LOOM_SCF_PIPELINE_STAGE_CONSUMER;
    if (guarded_partition) {
      if (ordered_memory && plan->body.accesses.units[i].effects !=
                                LOOM_SCF_BODY_MEMORY_READ_AHEAD_LOAD) {
        *rejection = (loom_scf_pipeline_rejection_t){
            operation->op,
            IREE_SV("separate read-ahead load and workgroup consumer units")};
        return iree_ok_status();
      }
      plan->stages[i] =
          LOOM_SCF_PIPELINE_STAGE_PRODUCER | LOOM_SCF_PIPELINE_STAGE_CONSUMER;
      plan->read_count += guarded_partition->producer.load_count;
      continue;
    }
    if (ordered_memory && plan->body.accesses.units[i].effects) {
      if (plan->body.accesses.units[i].effects ==
          LOOM_SCF_BODY_MEMORY_WORKGROUP) {
        continue;
      }
      if (plan->body.accesses.units[i].effects ==
              LOOM_SCF_BODY_MEMORY_READ_AHEAD_LOAD &&
          operation->effects == LOOM_SCF_BODY_EFFECT_READ) {
        plan->stages[i] = LOOM_SCF_PIPELINE_STAGE_PRODUCER;
        plan->read_count += operation->load_count;
        continue;
      }
      *rejection = (loom_scf_pipeline_rejection_t){
          operation->op,
          IREE_SV("separate read-ahead load and workgroup consumer units")};
      return iree_ok_status();
    }
    if (operation->effects == 0) {
      continue;
    }
    if (iree_any_bit_set(operation->effects, LOOM_SCF_BODY_EFFECT_CONVERGENT) &&
        !has_static_bounds) {
      *rejection = (loom_scf_pipeline_rejection_t){
          .op = operation->op,
          .constraint = IREE_SV("compile-time exact loop bounds to preserve "
                                "convergent consumer participation"),
      };
      return iree_ok_status();
    }
    if (operation->effects == LOOM_SCF_BODY_EFFECT_CONVERGENT) {
      continue;
    }
    if (operation->effects ==
        (LOOM_SCF_BODY_EFFECT_READ | LOOM_SCF_BODY_EFFECT_CONVERGENT)) {
      *rejection = (loom_scf_pipeline_rejection_t){
          .op = operation->op,
          .constraint = IREE_SV("convergent operations separated from "
                                "read-ahead loads"),
      };
      return iree_ok_status();
    }
    if (operation->effects != LOOM_SCF_BODY_EFFECT_READ) {
      *rejection = (loom_scf_pipeline_rejection_t){
          .op = operation->op,
          .constraint =
              IREE_SV("ordinary loads and memory-pure consumers without "
                      "writes, ordered effects or async groups"),
      };
      return iree_ok_status();
    }
    plan->stages[i] = LOOM_SCF_PIPELINE_STAGE_PRODUCER;
    plan->read_count += operation->load_count;
  }
  if (plan->read_count == 0) {
    *rejection = (loom_scf_pipeline_rejection_t){
        .op = block->first_op->parent_op,
        .constraint = IREE_SV("at least one ordinary load to pipeline"),
    };
    return iree_ok_status();
  }

  uint32_t* definition_owners = NULL;
  loom_scf_pipeline_stage_flags_t* definition_fixed_stages = NULL;
  bool* queued_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain->value_count,
                                                 sizeof(*definition_owners),
                                                 (void**)&definition_owners));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain->value_count,
                                                 sizeof(*queued_values),
                                                 (void**)&queued_values));
  if (plan->guarded_partition_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, domain->value_count, sizeof(*definition_fixed_stages),
        (void**)&definition_fixed_stages));
    memset(definition_fixed_stages, 0,
           domain->value_count * sizeof(*definition_fixed_stages));
  }
  memset(definition_owners, 0xFF,
         domain->value_count * sizeof(*definition_owners));
  memset(queued_values, 0, domain->value_count * sizeof(*queued_values));
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    const loom_op_t* op = plan->body.operations[i].op;
    const loom_scf_pipeline_guarded_partition_t* guarded_partition =
        plan->guarded_partitions ? plan->guarded_partitions[i] : NULL;
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t j = 0; j < op->result_count; ++j) {
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_ordinal(domain, results[j]);
      definition_owners[ordinal] = i;
      if (guarded_partition) {
        definition_fixed_stages[ordinal] = LOOM_SCF_PIPELINE_STAGE_CONSUMER;
      }
    }
    if (!guarded_partition) {
      continue;
    }
    for (uint8_t branch = 0; branch < LOOM_SCF_PIPELINE_BRANCH_COUNT;
         ++branch) {
      const loom_scf_body_t* branch_body = &guarded_partition->branches[branch];
      for (uint32_t j = 0; j < branch_body->count; ++j) {
        const loom_op_t* branch_op = branch_body->operations[j].op;
        const loom_scf_pipeline_stage_flags_t fixed_stage =
            branch == guarded_partition->producer_branch
                ? guarded_partition->producer_branch_stages[j]
                : LOOM_SCF_PIPELINE_STAGE_CONSUMER;
        const loom_value_id_t* branch_results =
            loom_op_const_results(branch_op);
        for (uint16_t k = 0; k < branch_op->result_count; ++k) {
          const loom_value_ordinal_t ordinal =
              loom_local_value_domain_ordinal(domain, branch_results[k]);
          definition_owners[ordinal] = i;
          definition_fixed_stages[ordinal] = fixed_stage;
        }
      }
    }
  }

  // Verified SSA puts every captured outer dependency before its scheduling
  // unit. A reverse traversal therefore computes the complete producer cut.
  for (uint32_t reverse = plan->body.count; reverse > 0; --reverse) {
    const uint32_t i = reverse - 1;
    if (!iree_any_bit_set(plan->stages[i], LOOM_SCF_PIPELINE_STAGE_PRODUCER)) {
      continue;
    }
    const loom_scf_body_reference_t* references = NULL;
    const loom_scf_body_operation_t* operation =
        loom_scf_pipeline_stage_operation(
            plan, i, LOOM_SCF_PIPELINE_STAGE_PRODUCER, &references);
    for (iree_host_size_t j = 0; j < operation->reference_count; ++j) {
      const loom_scf_body_reference_t* reference =
          &references[operation->reference_begin + j];
      if (reference->allow_identity_mapping) {
        continue;
      }
      const loom_value_id_t value_id = reference->value_id;
      if (value_id == block->arg_ids[0]) {
        continue;
      }
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_ordinal(domain, value_id);
      const uint32_t owner = definition_owners[ordinal];
      const loom_scf_pipeline_stage_flags_t fixed_stage =
          definition_fixed_stages ? definition_fixed_stages[ordinal] : 0;
      if (owner == UINT32_MAX) {
        *rejection = (loom_scf_pipeline_rejection_t){
            .op = operation->op,
            .constraint = IREE_SV("read-ahead prerequisites independent of "
                                  "loop-carried state"),
        };
        return iree_ok_status();
      }
      if (!iree_any_bit_set(
              loom_scf_pipeline_definition_stages(
                  plan, definition_owners, definition_fixed_stages, ordinal),
              LOOM_SCF_PIPELINE_STAGE_PRODUCER) &&
          (fixed_stage != 0 || plan->body.operations[owner].effects != 0)) {
        *rejection = (loom_scf_pipeline_rejection_t){
            .op = plan->body.operations[owner].op,
            .constraint = IREE_SV("read-ahead prerequisites independent of "
                                  "ordered or convergent consumers"),
        };
        return iree_ok_status();
      }
      if (fixed_stage == 0) {
        plan->stages[owner] = LOOM_SCF_PIPELINE_STAGE_PRODUCER;
      }
    }
  }

  // Include the yield in the consumer: a loaded value may itself be the next
  // carried state without an intervening arithmetic operation.
  for (iree_host_size_t i = 0; i <= plan->body.count; ++i) {
    if (i < plan->body.count &&
        plan->stages[i] == LOOM_SCF_PIPELINE_STAGE_PRODUCER) {
      continue;
    }
    const loom_scf_body_reference_t* references = plan->body.references;
    const loom_scf_body_operation_t* operation = &plan->body.terminator;
    if (i < plan->body.count) {
      operation = loom_scf_pipeline_stage_operation(
          plan, (uint32_t)i, LOOM_SCF_PIPELINE_STAGE_CONSUMER, &references);
    }
    for (iree_host_size_t j = 0; j < operation->reference_count; ++j) {
      const loom_scf_body_reference_t* reference =
          &references[operation->reference_begin + j];
      if (reference->allow_identity_mapping) {
        continue;
      }
      const loom_value_id_t value_id = reference->value_id;
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_ordinal(domain, value_id);
      if (value_id == block->arg_ids[0] ||
          iree_any_bit_set(
              loom_scf_pipeline_definition_stages(
                  plan, definition_owners, definition_fixed_stages, ordinal),
              LOOM_SCF_PIPELINE_STAGE_PRODUCER)) {
        queued_values[ordinal] = true;
      }
    }
  }
  // Contract the cut without introducing any new carried inputs. Reconstruct
  // address increments from values already available in the consumer so related
  // indices retain their algebraic identity instead of becoming independent
  // block arguments. Source order makes earlier reconstructions available to
  // later ones; the materializer consumes the selected stages directly.
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    const loom_scf_body_operation_t* operation = &plan->body.operations[i];
    const loom_op_t* op = operation->op;
    if ((plan->guarded_partitions && plan->guarded_partitions[i]) ||
        plan->stages[i] != LOOM_SCF_PIPELINE_STAGE_PRODUCER ||
        (op->kind != LOOM_OP_INDEX_ADD && op->kind != LOOM_OP_INDEX_SUB)) {
      continue;
    }
    const loom_value_ordinal_t result =
        loom_local_value_domain_ordinal(domain, loom_op_const_results(op)[0]);
    if (!queued_values[result]) {
      continue;
    }
    bool available = true;
    for (iree_host_size_t j = 0; j < operation->reference_count; ++j) {
      const loom_scf_body_reference_t* reference =
          &plan->body.references[operation->reference_begin + j];
      if (reference->allow_identity_mapping) {
        continue;
      }
      const loom_value_ordinal_t operand =
          loom_local_value_domain_ordinal(domain, reference->value_id);
      if (!queued_values[operand] &&
          !iree_any_bit_set(
              loom_scf_pipeline_definition_stages(
                  plan, definition_owners, definition_fixed_stages, operand),
              LOOM_SCF_PIPELINE_STAGE_CONSUMER)) {
        available = false;
        break;
      }
    }
    if (available) {
      plan->stages[i] |= LOOM_SCF_PIPELINE_STAGE_CONSUMER;
      queued_values[result] = false;
      ++plan->rematerialized_count;
    }
  }
  for (loom_value_ordinal_t i = 0; i < domain->value_count; ++i) {
    if (queued_values[i]) {
      ++plan->queue_value_count;
    }
  }
  if (plan->queue_value_count == 0) {
    *rejection = (loom_scf_pipeline_rejection_t){
        .op = block->first_op->parent_op,
        .constraint = IREE_SV("a producer value used by the ordered consumer"),
    };
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->queue_value_count,
                                                 sizeof(*plan->queue_values),
                                                 (void**)&plan->queue_values));
  uint32_t next_queue_value = 0;
  for (loom_value_ordinal_t i = 0; i < domain->value_count; ++i) {
    if (!queued_values[i]) {
      continue;
    }
    const loom_value_id_t value_id = domain->value_ids[i];
    plan->queue_values[next_queue_value++] = value_id;
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module, value_id, &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      if (loom_local_value_domain_ordinal(domain, provider) <
          domain->definition_count) {
        *rejection = (loom_scf_pipeline_rejection_t){
            .op = block->first_op->parent_op,
            .constraint = IREE_SV("iteration-invariant types for values "
                                  "carried between pipeline stages"),
        };
        return iree_ok_status();
      }
    }
    const uint32_t owner = definition_owners[i];
    if (owner != UINT32_MAX && definition_fixed_stages &&
        definition_fixed_stages[i] == LOOM_SCF_PIPELINE_STAGE_PRODUCER &&
        plan->guarded_partitions && plan->guarded_partitions[owner]) {
      if (!loom_scf_pipeline_has_guard_placeholder(
              loom_module_value_type(module, value_id))) {
        *rejection = (loom_scf_pipeline_rejection_t){
            .op = plan->body.operations[owner].op,
            .constraint = IREE_SV("scalar or vector values carried from a "
                                  "guarded producer"),
        };
        return iree_ok_status();
      }
      ++plan->guarded_partitions[owner]->queue_value_count;
    }
  }
  for (uint32_t i = 0; i < plan->body.count; ++i) {
    loom_scf_pipeline_guarded_partition_t* partition =
        plan->guarded_partitions ? plan->guarded_partitions[i] : NULL;
    if (!partition || partition->queue_value_count == 0) {
      continue;
    }
    const uint32_t queue_value_count = partition->queue_value_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, queue_value_count, sizeof(*partition->queue_values),
        (void**)&partition->queue_values));
    partition->queue_value_count = 0;
  }
  for (uint32_t i = 0; i < plan->queue_value_count; ++i) {
    const loom_value_id_t value_id = plan->queue_values[i];
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_ordinal(domain, value_id);
    const uint32_t owner = definition_owners[ordinal];
    if (owner == UINT32_MAX || !definition_fixed_stages ||
        definition_fixed_stages[ordinal] != LOOM_SCF_PIPELINE_STAGE_PRODUCER) {
      continue;
    }
    loom_scf_pipeline_guarded_partition_t* partition =
        plan->guarded_partitions ? plan->guarded_partitions[owner] : NULL;
    if (partition) {
      partition->queue_values[partition->queue_value_count++] = value_id;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scf_pipeline_plan_build(
    loom_module_t* module, const loom_block_t* block, bool has_static_bounds,
    const loom_scf_memory_t* spaces, iree_arena_allocator_t* arena,
    loom_scf_pipeline_plan_t* out_plan,
    loom_scf_pipeline_rejection_t* out_rejection) {
  *out_plan = (loom_scf_pipeline_plan_t){0};
  *out_rejection = (loom_scf_pipeline_rejection_t){0};
  const loom_op_t* unstructured_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_body_build(module, block,
                                           /*capture_block=*/NULL, spaces,
                                           LOOM_SCF_BODY_MODE_SCHEDULE, arena,
                                           &out_plan->body, &unstructured_op));
  if (unstructured_op) {
    *out_rejection = (loom_scf_pipeline_rejection_t){
        .op = unstructured_op,
        .constraint =
            IREE_SV("body operations with only scf.if/scf.for regions "
                    "and no successors"),
    };
    return iree_ok_status();
  }
  const bool has_guarded_candidate =
      loom_scf_pipeline_has_guarded_candidate(out_plan);
  loom_local_value_domain_t domain = {0};
  IREE_RETURN_IF_ERROR(has_guarded_candidate
                           ? loom_local_value_domain_acquire_for_region_tree(
                                 module, block->parent_region, arena, &domain)
                           : loom_local_value_domain_acquire_for_region(
                                 module, block->parent_region, arena, &domain));
  iree_status_t status = loom_scf_pipeline_plan_partition(
      module, block, &domain, spaces, has_static_bounds, has_guarded_candidate,
      arena, out_plan, out_rejection);
  loom_local_value_domain_release(&domain);
  return status;
}
