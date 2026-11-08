// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/realization.h"

#include "loom/ir/symbol_map.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/special_values.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/cfg/cfg_simplify.h"
#include "loom/transforms/kernel/kernel_resources.h"

// Observe the source proof before physical selection or consuming rewrites.
// These rows describe transport eligibility, not a selected DMA implementation.
static iree_status_t loom_pipeline_realization_report(
    loom_pass_t* pass, const loom_module_t* module,
    const loom_pipeline_realization_t* realization) {
  if (!loom_pass_report_is_enabled(pass)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &realization->workers[i];
    const loom_pipeline_transport_t* transport = &worker->transport;
    const loom_pass_report_detail_field_t fields[] = {
        loom_pass_report_detail_uint64_field(IREE_SV("worker"), i),
        loom_pass_report_detail_string_field(
            IREE_SV("eligibility"),
            transport->cohort_count ? IREE_SV("channel-copy")
            : transport->count      ? IREE_SV("borrowed-movement")
                                    : IREE_SV("unproven")),
        loom_pass_report_detail_string_field(IREE_SV("execution_counts"),
                                             worker->block_execution_counts
                                                 ? IREE_SV("exact")
                                                 : IREE_SV("unknown")),
        loom_pass_report_detail_uint64_field(IREE_SV("batches"),
                                             transport->batch_count),
        loom_pass_report_detail_uint64_field(IREE_SV("streams"),
                                             transport->stream_count),
    };
    IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
        pass, IREE_SV("pipeline-transport"), fields, IREE_ARRAYSIZE(fields)));
    for (iree_host_size_t j = 0; j < transport->batch_count; ++j) {
      const loom_pipeline_transport_batch_t* batch = &transport->batches[j];
      for (iree_host_size_t k = 0; k < batch->record_count; ++k) {
        const loom_pipeline_transport_cohort_t* record = &batch->records[k];
        const loom_pass_report_detail_field_t record_fields[] = {
            loom_pass_report_detail_uint64_field(IREE_SV("worker"), i),
            loom_pass_report_detail_uint64_field(IREE_SV("batch"), j),
            loom_pass_report_detail_uint64_field(IREE_SV("record"), k),
            loom_pass_report_detail_string_field(
                IREE_SV("admission"),
                loom_module_value_name(
                    module, loom_op_const_results(record->admission->op)[0])),
            loom_pass_report_detail_string_field(
                IREE_SV("source"),
                loom_module_value_name(module,
                                       record->admission->channel->value_id)),
            loom_pass_report_detail_uint64_field(IREE_SV("batch_records"),
                                                 batch->record_count),
            loom_pass_report_detail_uint64_field(IREE_SV("admissions"),
                                                 record->admissions),
            loom_pass_report_detail_uint64_field(IREE_SV("transfers"),
                                                 record->transfers),
            loom_pass_report_detail_uint64_field(IREE_SV("replays"),
                                                 batch->replays),
            loom_pass_report_detail_bool_field(IREE_SV("final_release"),
                                               record->release != NULL),
            loom_pass_report_detail_uint64_field(IREE_SV("destinations"),
                                                 record->destination_count),
        };
        IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
            pass, IREE_SV("pipeline-transport-record"), record_fields,
            IREE_ARRAYSIZE(record_fields)));
        for (iree_host_size_t d = 0; d < record->destination_count; ++d) {
          const loom_pipeline_transport_destination_t* destination =
              &record->destinations[d];
          const loom_pass_report_detail_field_t destination_fields[] = {
              loom_pass_report_detail_uint64_field(IREE_SV("worker"), i),
              loom_pass_report_detail_uint64_field(IREE_SV("batch"), j),
              loom_pass_report_detail_uint64_field(IREE_SV("record"), k),
              loom_pass_report_detail_uint64_field(IREE_SV("destination"), d),
              loom_pass_report_detail_string_field(
                  IREE_SV("channel"),
                  loom_module_value_name(module,
                                         destination->channel->value_id)),
          };
          IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
              pass, IREE_SV("pipeline-transport-destination"),
              destination_fields, IREE_ARRAYSIZE(destination_fields)));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_realization_clone_workers(
    loom_pass_t* pass, loom_rewriter_t* rewriter,
    const loom_target_function_version_snapshot_t* versions,
    loom_pipeline_realization_t* realization) {
  loom_module_t* module = rewriter->module;
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  loom_function_version_owner_t* owner =
      loom_target_pass_capability_function_version_owner(capability);
  realization->version_owner = owner;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      pass->arena, realization->invocations.worker_count,
      sizeof(*realization->worker_versions),
      (void**)&realization->worker_versions));
  memset(realization->worker_versions, 0,
         realization->invocations.worker_count *
             sizeof(*realization->worker_versions));
  if (versions->target_context_capacity >=
      LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "pipeline realization exceeds target context capacity");
  }
  realization->entry_target_context =
      (loom_target_context_ordinal_t)versions->target_context_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      pass->arena, realization->invocations.worker_count,
      sizeof(*realization->instances), (void**)&realization->instances));
  loom_symbol_map_t names = {0};
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &names, pass->arena, module->symbols.entries[i].name_id,
        (loom_symbol_id_t)i));
  }
  uint32_t next_name = 0;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_pipeline_worker_t* worker = &realization->workers[i];
    const loom_target_function_version_t* source_version =
        loom_target_function_version_snapshot_at(
            versions, loom_func_call_callee(worker->source->call).symbol_id);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    do {
      char text[48];
      iree_snprintf(text, sizeof(text), "__pipeline_worker_%u", next_name++);
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          module, iree_make_cstring_view(text), &name));
    } while (loom_symbol_map_find(&names, name) != LOOM_SYMBOL_ID_INVALID);
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol));
    IREE_RETURN_IF_ERROR(
        loom_symbol_map_insert(&names, pass->arena, name, symbol));
    const loom_symbol_ref_t target = {0, symbol};
    loom_builder_set_block(&rewriter->builder, loom_module_block(module));
    loom_local_value_domain_restore(&worker->value_domain);
    iree_status_t status = loom_pipeline_worker_materialization_clone(
        rewriter, worker, target, &realization->instances[i]);
    loom_local_value_domain_release(&worker->value_domain);
    IREE_RETURN_IF_ERROR(status);
    const loom_func_like_t instance =
        realization->instances[i].channels.function;
    if (owner && source_version) {
      loom_target_function_version_t* version = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(owner->arena, sizeof(*version),
                                               (void**)&version));
      *version = *source_version;
      version->base.function = instance;
      version->memory_accesses = NULL;
      version->loop_execution = NULL;
      version->loop_pipelines = (loom_source_loop_pipeline_list_t){0};
      for (const loom_source_loop_pipeline_t* source_policy =
               source_version->loop_pipelines.head;
           source_policy; source_policy = source_policy->next) {
        loom_source_loop_pipeline_t* policy = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(owner->arena, sizeof(*policy),
                                                 (void**)&policy));
        *policy = *source_policy;
        policy->next = NULL;
        if (version->loop_pipelines.tail) {
          version->loop_pipelines.tail->next = policy;
        } else {
          version->loop_pipelines.head = policy;
        }
        version->loop_pipelines.tail = policy;
      }
      IREE_RETURN_IF_ERROR(
          loom_function_version_owner_append(owner, &version->base));
      realization->worker_versions[i] = version;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_realization_register_helper(
    loom_module_t* module, const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, loom_func_like_t helper,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_target_function_version_t* caller =
      realization->worker_versions[worker_index];
  if (caller == NULL) {
    return iree_ok_status();
  }
  loom_function_version_owner_t* owner = realization->version_owner;
  const loom_symbol_ref_t symbol = loom_func_like_callee(helper);
  const loom_func_symbol_facts_t function_facts = {
      .func_op = helper.op,
      .name = loom_string_table_get(
          &module->strings, module->symbols.entries[symbol.symbol_id].name_id),
  };
  bool valid = false;
  const loom_target_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_internal_facts(
      module, &function_facts,
      loom_target_facts_identity_name(caller->resolved_target.facts),
      caller->resolved_target.facts, diagnostic_emitter, owner->arena, &valid,
      &facts));
  loom_target_function_version_t* version = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(owner->arena, sizeof(*version), (void**)&version));
  *version = (loom_target_function_version_t){
      .base = {.type = &loom_target_function_version_type, .function = helper},
      .resolved_target = caller->resolved_target,
      .target_context_ordinal = caller->target_context_ordinal,
      .function_target_facts = facts,
  };
  return loom_function_version_owner_append(owner, &version->base);
}

static const loom_target_facts_t* loom_pipeline_realization_child_target(
    void* user_data, loom_func_like_t function) {
  const loom_target_function_version_snapshot_t* versions = user_data;
  const loom_target_function_version_t* version =
      loom_target_function_version_snapshot_at(
          versions, loom_func_like_callee(function).symbol_id);
  return version ? version->function_target_facts : NULL;
}

// Occurrence facts can make whole branches unreachable in a shared worker
// definition. Establish that program before channel, movement, and physical
// planning: assumptions inside a dead branch need not hold for this caller.
static iree_status_t loom_pipeline_realization_prepare_workers(
    loom_rewriter_t* rewriter, const loom_pipeline_realization_t* realization,
    const loom_target_function_version_snapshot_t* versions,
    loom_func_like_t** out_functions, bool** out_prepared) {
  loom_module_t* module = rewriter->module;
  loom_func_like_t* functions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, realization->invocations.worker_count,
      sizeof(*functions), (void**)&functions));
  bool* prepared = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, realization->invocations.worker_count, sizeof(*prepared),
      (void**)&prepared));
  memset(prepared, 0,
         realization->invocations.worker_count * sizeof(*prepared));
  loom_pass_value_fact_owner_t value_facts;
  loom_pass_value_fact_owner_initialize(rewriter->arena->block_pool,
                                        &value_facts);
  uint32_t next_name = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < realization->invocations.worker_count;
       ++i) {
    const loom_pipeline_invocation_worker_t* occurrence =
        &realization->invocations.workers[i];
    const loom_value_slice_t actuals =
        loom_func_call_operands(occurrence->strand->call);
    const loom_symbol_ref_t source =
        loom_func_call_callee(occurrence->strand->call);
    functions[i] = loom_func_like_cast(
        module, module->symbols.entries[source.symbol_id].defining_op);
    loom_region_t* body = loom_func_like_body(functions[i]);
    // Only a multi-block body can need branch pruning before physical
    // selection. Straight-line captures are materialized in the selected
    // consuming body.
    if (body->block_count == 1) {
      continue;
    }
    loom_block_t* entry = loom_region_entry_block(body);
    bool specialized = false;
    for (uint16_t j = 0; j < entry->arg_count; ++j) {
      const loom_value_id_t argument = entry->arg_ids[j];
      const loom_value_facts_t facts = loom_value_fact_table_lookup(
          occurrence->invocation->facts, actuals.values[j]);
      specialized |= loom_module_value_has_uses(module, argument) &&
                     loom_value_facts_can_materialize_constant(
                         facts, loom_module_value_type(module, argument));
    }
    if (!specialized) {
      continue;
    }
    // Specialized sources are ordinary callable definitions, even while
    // their consuming instances are being selected. Keep their symbol names
    // valid through erasure so snapshots can still project the module.
    char name[48];
    do {
      iree_snprintf(name, sizeof(name), "__pipeline_source_%u", next_name++);
    } while (loom_module_lookup_string(module, iree_make_cstring_view(name)) !=
             LOOM_STRING_ID_INVALID);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(module, iree_make_cstring_view(name),
                                       &name_id);
    loom_symbol_id_t symbol;
    if (iree_status_is_ok(status)) {
      status = loom_module_add_symbol(module, name_id, &symbol);
    }
    if (iree_status_is_ok(status)) {
      loom_builder_set_block(&rewriter->builder, loom_module_block(module));
      status = loom_callable_clone_definition(
          &rewriter->builder, functions[i], (loom_symbol_ref_t){0, symbol},
          NULL, rewriter->arena, &functions[i]);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    prepared[i] = true;
    entry = loom_region_entry_block(loom_func_like_body(functions[i]));
    loom_builder_set_before(&rewriter->builder, entry->first_op);
    for (uint16_t j = 0; iree_status_is_ok(status) && j < entry->arg_count;
         ++j) {
      const loom_value_id_t argument = entry->arg_ids[j];
      const loom_type_t type =
          loom_module_value_type(rewriter->module, argument);
      const loom_value_facts_t facts = loom_value_fact_table_lookup(
          occurrence->invocation->facts, actuals.values[j]);
      if (!loom_module_value_has_uses(rewriter->module, argument) ||
          !loom_value_facts_can_materialize_constant(facts, type)) {
        continue;
      }
      loom_value_id_t constant;
      status = loom_constant_build(&rewriter->builder, facts, type,
                                   functions[i].op->location, &constant);
      if (iree_status_is_ok(status)) {
        // Prune executable control while preserving the callable signature.
        // Argument types may depend on this formal; a body-local constant
        // cannot dominate those types when this source is cloned later.
        status = loom_rewriter_replace_all_uses_except(rewriter, argument,
                                                       constant, NULL);
      }
      specialized = true;
    }
    if (iree_status_is_ok(status) && specialized) {
      bool changed = false;
      const loom_target_function_version_t* version =
          loom_target_function_version_snapshot_at(versions, source.symbol_id);
      status = loom_cfg_simplify_function(
          rewriter->module, functions[i],
          version ? version->function_target_facts : NULL, rewriter->arena,
          &value_facts, &changed);
    }
  }
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  *out_functions = functions;
  *out_prepared = prepared;
  return status;
}

iree_status_t loom_pipeline_realize(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_pipeline_realization_inventory_t* inventory,
    const loom_pipeline_realization_callback_t* callback) {
  // This closed execution now owns an invocation ABI. Source compositions
  // retain view arguments until their execution owner admits that boundary.
  uint16_t normalized_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_kernel_resources_normalize(module, function, &normalized_count));
  if (normalized_count) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    loom_pass_mark_changed(pass);
  }
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts));
  loom_pipeline_realization_t realization = {.function = function,
                                             .facts = facts};
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
      module, function, facts, inventory->pools, inventory->pool_count,
      inventory->memories, inventory->memory_count, inventory->bindings,
      inventory->binding_count, pass->diagnostic_emitter, pass->arena,
      &realization.resources, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(capability),
      pass->arena, &versions));
  const loom_pipeline_invocation_environment_t environment = {
      .target_facts = loom_pipeline_realization_child_target,
      .user_data = &versions};
  IREE_RETURN_IF_ERROR(loom_pipeline_invocations_build(
      module, function, facts, &realization.resources, inventory->memories,
      inventory->memory_count, environment, pass->diagnostic_emitter,
      pass->arena, &realization.invocations, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  loom_func_like_t* functions = NULL;
  bool* prepared = NULL;
  iree_status_t status = loom_pipeline_realization_prepare_workers(
      &rewriter, &realization, &versions, &functions, &prepared);
  if (iree_status_is_ok(status)) {
    status = loom_pipeline_invocations_analyze_workers(
        module, &realization.invocations, functions, pass->diagnostic_emitter,
        pass->arena, &realization.workers, &valid);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_pipeline_realize_invocations(
        pass, module, &realization.invocations, realization.workers, &versions,
        callback);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < realization.invocations.worker_count;
       ++i) {
    if (prepared[i]) {
      status = loom_rewriter_erase(&rewriter, functions[i].op);
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_pipeline_realize_invocations(
    loom_pass_t* pass, loom_module_t* module,
    const loom_pipeline_invocations_t* invocations,
    loom_pipeline_worker_t* workers,
    const loom_target_function_version_snapshot_t* versions,
    const loom_pipeline_realization_callback_t* callback) {
  const loom_pipeline_invocation_t* root = invocations->entries[0];
  loom_pipeline_realization_t realization = {
      .function = root->function,
      .facts = root->facts,
      .resources = *root->resources,
      .invocations = *invocations,
      .workers = workers,
  };
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status =
      loom_pipeline_realization_report(pass, module, &realization);
  bool valid = true;
  if (iree_status_is_ok(status) && valid) {
    status =
        callback->select(callback->user_data, module, &realization, &valid);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_pipeline_realization_clone_workers(pass, &rewriter, versions,
                                                     &realization);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && valid &&
                               i < realization.invocations.worker_count;
       ++i) {
    loom_pipeline_worker_t* worker = &realization.workers[i];
    loom_channel_materialization_instance_t* instance =
        &realization.instances[i].channels;
    loom_local_value_domain_restore(&instance->value_domain);
    status = callback->worker(callback->user_data, &rewriter, &realization, i);
    loom_local_value_domain_release(&instance->value_domain);
    if (iree_status_is_ok(status)) {
      loom_block_t* entry =
          loom_region_entry_block(loom_func_like_body(instance->function));
      loom_builder_set_before(&rewriter.builder, entry->first_op);
      loom_ir_remap_t literals;
      if (worker->literal_captures) {
        status = loom_ir_remap_initialize(module, module, pass->arena, NULL,
                                          &literals);
      }
      uint16_t source_argument_count = 0;
      const loom_value_id_t* source_arguments =
          loom_func_like_arg_ids(worker->function, &source_argument_count);
      for (uint16_t j = 0;
           j < source_argument_count && iree_status_is_ok(status); ++j) {
        const loom_value_id_t argument = entry->arg_ids[j];
        const loom_type_t type = loom_module_value_type(module, argument);
        const loom_value_facts_t value_facts =
            loom_value_fact_table_lookup(&worker->facts, source_arguments[j]);
        if (loom_value_facts_can_materialize_constant(value_facts, type)) {
          loom_value_id_t constant;
          status =
              loom_constant_build(&rewriter.builder, value_facts, type,
                                  worker->function.op->location, &constant);
          if (iree_status_is_ok(status)) {
            status = loom_rewriter_replace_all_uses_with(&rewriter, argument,
                                                         constant);
          }
        } else if (worker->literal_captures && worker->literal_captures[j]) {
          loom_op_t* constant = NULL;
          status =
              loom_ir_clone_op(&rewriter.builder, worker->literal_captures[j],
                               &literals, &constant);
          if (iree_status_is_ok(status)) {
            status = loom_rewriter_replace_all_uses_with(
                &rewriter, argument, loom_op_results(constant)[0]);
          }
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_seed_function(&rewriter, instance->function);
      }
      loom_op_t* dead;
      while (iree_status_is_ok(status) &&
             (dead = loom_rewriter_pop(&rewriter))) {
        bool erased;
        status = loom_rewriter_erase_if_dead(&rewriter, dead, &erased);
      }
      for (uint16_t j = entry->arg_count; j > 0 && iree_status_is_ok(status);
           --j) {
        const loom_value_id_t argument = entry->arg_ids[j - 1];
        if (!loom_module_value_has_uses(module, argument)) {
          status = loom_block_remove_arg(module, entry, j - 1);
        }
      }
    }
  }
  if (iree_status_is_ok(status) && valid) {
    status = callback->entry(callback->user_data, &rewriter, &realization);
    loom_function_version_owner_prune_erased(realization.version_owner);
  }
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  loom_pass_mark_changed(pass);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
