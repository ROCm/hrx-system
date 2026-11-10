// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/prefill_group.h"

#include <string.h>

#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/pool.h"
#include "loom/serve/service/text/epoch_plan.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/variant.h"

static iree_vm_variant_t loom_serve_text_group_host(
    loom_serve_text_model_t* model, iree_vm_buffer_t* buffer) {
  return iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                                  buffer);
}

static iree_status_t loom_serve_text_group_invoke(
    loom_serve_text_model_t* model, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results) {
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                     function, arguments, results);
  iree_vm_variant_span_reset(arguments);
  return status;
}

bool loom_serve_text_model_two_bank(const loom_serve_text_model_t* model) {
  return model->two_bank;
}

bool loom_serve_text_model_can_pair(const loom_serve_text_model_t* model,
                                    iree_host_size_t first_shape,
                                    const loom_serve_text_span_t* first,
                                    iree_host_size_t second_shape,
                                    const loom_serve_text_span_t* second) {
  if (!model->two_bank || !model->execution_plan || !first || !second ||
      first->row_index >= model->row_count ||
      first_shape >= model->shape_count || second_shape >= model->shape_count) {
    return false;
  }
  const loom_serve_plan_group_query_t query = {
      .position = model->rows[first->row_index]->state->position,
      .shapes = {first_shape, second_shape},
      .tokens = {first->token_count, second->token_count}};
  return !model->plan_provider.select ||
         model->plan_provider.select(model->plan_provider.self,
                                     model->execution_plan, &query);
}

static const loom_serve_stage_t* loom_serve_text_group_stages(
    const loom_serve_text_model_t* model) {
  return model->prefill_group.stages ? model->prefill_group.stages
                                     : model->stage_bindings;
}

// Exports the source stage table for |control_state| and checks every leaf
// against |bindings| before the provider prepares an immutable plan.
static iree_status_t loom_serve_text_group_describe(
    loom_serve_text_model_t* model, iree_vm_buffer_t* control_state,
    const loom_serve_stage_t* bindings,
    loom_serve_execution_plan_t** out_plan) {
  *out_plan = NULL;
  iree_vm_variant_t argument = loom_serve_text_group_host(model, control_state);
  iree_vm_variant_t result = {0};
  iree_status_t status =
      loom_serve_text_group_invoke(model, model->prefill_group.describe,
                                   iree_vm_variant_span_from_ptr(&argument, 1),
                                   iree_vm_variant_span_from_ptr(&result, 1));
  iree_vm_buffer_t* descriptor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&model->source.vm_types,
                                                  &result, &descriptor);
  }
  iree_vm_variant_reset(&result);
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status) && !descriptor) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "null prefill group descriptor");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(descriptor, 0,
                                     iree_vm_buffer_length(descriptor), &bytes);
  }
  uint32_t* stages = NULL;
  if (iree_status_is_ok(status)) {
    if (bytes.data_length < 16 || iree_unaligned_load_le_u32(bytes.data) != 1 ||
        iree_unaligned_load_le_u32(bytes.data + 4) != model->shape_count ||
        !iree_unaligned_load_le_u32(bytes.data + 8) ||
        iree_unaligned_load_le_u32(bytes.data + 8) > 1024 ||
        (iree_unaligned_load_le_u32(bytes.data + 12) != 1 &&
         iree_unaligned_load_le_u32(bytes.data + 12) != 2) ||
        (model->draft_depth &&
         iree_unaligned_load_le_u32(bytes.data + 12) != 2) ||
        model->shape_count > 1024) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid text_prefill_group_v1 descriptor");
    } else {
      const uint32_t layers = iree_unaligned_load_le_u32(bytes.data + 8);
      const iree_host_size_t count = model->shape_count * layers * 3;
      if (bytes.data_length != 16 + count * 4) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "prefill group stage table extent mismatch");
      } else {
        status = iree_allocator_malloc_array(model->host_allocator, count,
                                             sizeof(*stages), (void**)&stages);
        for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status);
             ++i) {
          stages[i] = iree_unaligned_load_le_u32(bytes.data + 16 + i * 4);
          if (stages[i] != UINT32_MAX &&
              (stages[i] >= model->stage_count ||
               (bindings[stages[i]].binding_count != TEXT_BINDING_COUNT &&
                (iree_unaligned_load_le_u32(bytes.data + 12) != 2 ||
                 bindings[stages[i]].binding_count !=
                     TEXT_GROUP_DRAFT_ARITY)) ||
               bindings[stages[i]].dispatch ||
               !bindings[stages[i]].command_buffer)) {
            status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "group leaves require a declared "
                                      "six/twelve-resource command ABI");
          }
        }
        if (iree_status_is_ok(status)) {
          const loom_serve_plan_description_t description = {
              model->shape_count, layers, model->stage_count, stages};
          status = model->plan_provider.prepare(
              model->plan_provider.self, &description, model->host_allocator,
              out_plan);
        }
      }
    }
  }
  iree_allocator_free(model->host_allocator, stages);
  iree_vm_buffer_release(descriptor);
  if (iree_status_is_ok(status) && !*out_plan) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "provider returned no plan");
  }
  return status;
}

// Each bank workspace backs the largest reflected private requirement of any
// stage, as the native module sizes ordinary invocation scratch.
static iree_hal_pool_reservation_request_t loom_serve_text_group_workspace(
    const loom_serve_text_model_t* model, const loom_serve_stage_t* bindings) {
  iree_device_size_t length = 1;
  iree_device_size_t alignment = 256;
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    length = iree_max(length, bindings[i].transient.required_byte_length);
    alignment = iree_max(alignment, bindings[i].transient.minimum_alignment);
  }
  return iree_hal_make_pool_reservation_request(length, alignment);
}

// Bank B's private packets plus both persistent workspaces, checked as one
// working set. Their backing is charged by the storage pool, not again as an
// invocation scratch peak.
static iree_status_t loom_serve_text_group_check_working_set(
    const loom_serve_text_model_t* model, iree_hal_buffer_t* const* packets,
    iree_device_size_t workspace) {
  iree_device_size_t extents[6];
  for (iree_host_size_t i = 0; i < 4; ++i) {
    extents[i] = iree_hal_buffer_byte_length(packets[i]);
  }
  extents[4] = extents[5] = workspace;
  return loom_serve_residency_check_working_set(
      model->residency, IREE_ARRAYSIZE(extents), extents);
}

iree_status_t loom_serve_text_prefill_group_initialize(
    loom_serve_text_model_t* model) {
  if (!model->two_bank) {
    return iree_ok_status();
  }
  const iree_vm_signature_type_t buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = model->source.vm_types.buffer};
  const iree_vm_signature_type_t encode_args[] = {
      buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer};
  const iree_vm_signature_type_t publish_args[] = {buffer, buffer, buffer,
                                                   buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("text_prefill_group_plan_v1"),
       {&buffer, 1},
       {&buffer, 1},
       &model->prefill_group.describe},
      {IREE_SVL("text_encode_prefill_group_v1"),
       {encode_args, IREE_ARRAYSIZE(encode_args)},
       {NULL, 0},
       &model->prefill_group.encode},
      {IREE_SVL("text_publish_prefill_group_v1"),
       {publish_args, IREE_ARRAYSIZE(publish_args)},
       {NULL, 0},
       &model->prefill_group.publish},
  };
  IREE_RETURN_IF_ERROR(loom_serve_program_bind(
      model->control.program, IREE_ARRAYSIZE(bindings), bindings));
  IREE_RETURN_IF_ERROR(loom_serve_text_group_describe(
      model, model->source.control_state, model->stage_bindings,
      &model->execution_plan));

  iree_hal_buffer_t* original[] = {
      model->state.residual, model->state.epoch.buffers[1],
      model->state.epoch.buffers[4], model->state.epoch.buffers[5]};
  iree_hal_pool_reservation_request_t requests[6];
  for (iree_host_size_t i = 0; i < 4; ++i) {
    if (!original[i]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "group requires packed bank roots");
    }
    requests[i] = iree_hal_make_pool_reservation_request(
        iree_hal_buffer_byte_length(original[i]), 256);
  }
  model->prefill_group.workspace =
      loom_serve_text_group_workspace(model, model->stage_bindings);
  requests[4] = requests[5] = model->prefill_group.workspace;
  IREE_RETURN_IF_ERROR(loom_serve_text_group_check_working_set(
      model, original, model->prefill_group.workspace.allocation_size));
  iree_hal_buffer_t* extra[6] = {NULL};
  iree_status_t status = loom_serve_execution_alloca_from_pool(
      model->execution, loom_serve_pool_storage(model->pool),
      IREE_ARRAYSIZE(extra), requests, extra);
  for (iree_host_size_t i = 0; i < 6 && iree_status_is_ok(status); ++i) {
    status = loom_serve_retirement_track(&model->state.retirement, &extra[i],
                                         model->host_allocator);
  }
  const uint32_t slots[] = {TEXT_HOST_PLAN, TEXT_HOST_INPUT, TEXT_HOST_METADATA,
                            TEXT_HOST_TOKENS, TEXT_HOST_OUTPUTS};
  iree_host_size_t lengths[LOOM_SERVE_TEXT_BANK_HOST_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
    lengths[i] = model->control.host.bytes[slots[i]].data_length;
  }
  for (iree_host_size_t bank = 0; bank < 2 && iree_status_is_ok(status);
       ++bank) {
    iree_hal_buffer_t* roots[] = {
        bank ? extra[0] : original[0], bank ? extra[1] : original[1],
        bank ? extra[2] : original[2], bank ? extra[3] : original[3],
        extra[4 + bank]};
    status = loom_serve_text_bank_initialize(
        roots, lengths, model->host_allocator, &model->banks[bank]);
  }
  for (iree_host_size_t i = 0; i < 6; ++i) {
    iree_hal_buffer_release(extra[i]);
  }
  return status;
}

iree_status_t loom_serve_text_prefill_group_prepare_plan(
    loom_serve_text_model_t* model, iree_vm_buffer_t* control_state,
    loom_serve_execution_plan_t** out_plan) {
  *out_plan = NULL;
  if (!model->two_bank) {
    return iree_ok_status();
  }
  if (model->prefill_group.pending || model->prefill_group.active) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "group model must be drained before resizing");
  }
  return loom_serve_text_group_describe(
      model, control_state, loom_serve_text_group_stages(model), out_plan);
}

static void loom_serve_text_group_stages_free(iree_allocator_t allocator,
                                              iree_host_size_t count,
                                              loom_serve_stage_t* stages) {
  for (iree_host_size_t i = 0; stages && i < count; ++i) {
    iree_hal_command_buffer_release(stages[i].command_buffer);
  }
  iree_allocator_free(allocator, stages);
}

iree_status_t loom_serve_text_prefill_group_prepare_variant(
    loom_serve_text_model_t* model, const loom_serve_text_variant_t* variant,
    loom_serve_text_group_candidate_t* out_candidate) {
  memset(out_candidate, 0, sizeof(*out_candidate));
  if (!model->two_bank) {
    return iree_ok_status();
  }
  if (model->prefill_group.pending || model->prefill_group.active ||
      model->invocation.span_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "group model must be drained before variant selection");
  }
  const loom_serve_stage_t* bindings = model->stage_bindings;
  if (variant) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_variant_stages(variant, model, &bindings));
  }
  out_candidate->host_allocator = model->host_allocator;
  out_candidate->stage_count = model->stage_count;
  out_candidate->rebind = true;
  iree_status_t status = loom_serve_text_group_describe(
      model, model->source.control_state, bindings, &out_candidate->plan);
  if (iree_status_is_ok(status) && variant) {
    // Direct stages are never group leaves; their descriptors stay unborrowed.
    status = iree_allocator_malloc_array(
        model->host_allocator, model->stage_count,
        sizeof(*out_candidate->stages), (void**)&out_candidate->stages);
    for (iree_host_size_t i = 0;
         i < model->stage_count && iree_status_is_ok(status); ++i) {
      out_candidate->stages[i] = bindings[i];
      out_candidate->stages[i].dispatch = NULL;
      iree_hal_command_buffer_retain(bindings[i].command_buffer);
    }
  }
  out_candidate->workspace = loom_serve_text_group_workspace(model, bindings);
  const iree_hal_pool_reservation_request_t* current =
      &model->prefill_group.workspace;
  if (iree_status_is_ok(status) &&
      (out_candidate->workspace.allocation_size != current->allocation_size ||
       out_candidate->workspace.min_alignment != current->min_alignment)) {
    const loom_serve_text_bank_t* b = &model->banks[1];
    status = loom_serve_text_group_check_working_set(
        model, b->roots, out_candidate->workspace.allocation_size);
    const iree_hal_pool_reservation_request_t requests[] = {
        out_candidate->workspace, out_candidate->workspace};
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_alloca_from_pool(
          model->execution, loom_serve_pool_storage(model->pool),
          IREE_ARRAYSIZE(requests), requests, out_candidate->workspaces);
    }
    for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = loom_serve_retirement_track(&model->state.retirement,
                                           &out_candidate->workspaces[i],
                                           model->host_allocator);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_prefill_group_discard(out_candidate);
  }
  return status;
}

void loom_serve_text_prefill_group_commit(
    loom_serve_text_model_t* model,
    loom_serve_text_group_candidate_t* candidate) {
  if (candidate->plan) {
    loom_serve_execution_plan_release(model->execution_plan);
    model->execution_plan = candidate->plan;
    candidate->plan = NULL;
  }
  if (candidate->rebind) {
    loom_serve_text_group_stages_free(model->host_allocator, model->stage_count,
                                      model->prefill_group.stages);
    model->prefill_group.stages = candidate->stages;
    candidate->stages = NULL;
    candidate->rebind = false;
  }
  if (candidate->workspaces[0]) {
    // Accepted groups retain the previous workspaces until queue retirement.
    for (iree_host_size_t i = 0; i < 2; ++i) {
      iree_hal_buffer_t** root =
          &model->banks[i].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE];
      iree_hal_buffer_release(*root);
      *root = candidate->workspaces[i];
      candidate->workspaces[i] = NULL;
    }
    model->prefill_group.workspace = candidate->workspace;
  }
}

void loom_serve_text_prefill_group_discard(
    loom_serve_text_group_candidate_t* candidate) {
  loom_serve_execution_plan_release(candidate->plan);
  loom_serve_text_group_stages_free(candidate->host_allocator,
                                    candidate->stage_count, candidate->stages);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_hal_buffer_release(candidate->workspaces[i]);
  }
  memset(candidate, 0, sizeof(*candidate));
}

void loom_serve_text_prefill_group_retire(loom_serve_text_model_t* model) {
  loom_serve_text_group_stages_free(model->host_allocator, model->stage_count,
                                    model->prefill_group.stages);
  model->prefill_group.stages = NULL;
}

static iree_status_t loom_serve_text_group_emit(void* self, uint32_t bank_index,
                                                uint32_t stage_index) {
  loom_serve_text_model_t* model = self;
  if (bank_index > 1 || stage_index >= model->stage_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider emitted an unknown group leaf");
  }
  const loom_serve_stage_t* stage =
      &loom_serve_text_group_stages(model)[stage_index];
  const loom_serve_text_bank_t* bank = &model->banks[bank_index];
  iree_hal_buffer_t* roots[] = {bank->roots[LOOM_SERVE_TEXT_BANK_RESIDUAL],
                                bank->roots[LOOM_SERVE_TEXT_BANK_METADATA],
                                model->state.epoch.buffers[2],
                                model->state.row_arena,
                                bank->roots[LOOM_SERVE_TEXT_BANK_TOKENS],
                                bank->roots[LOOM_SERVE_TEXT_BANK_OUTPUTS],
                                model->state.carry,
                                model->state.draft.committed,
                                model->state.draft.results,
                                model->state.draft.cache,
                                model->state.draft.row_table,
                                model->state.draft.next_results};
  iree_hal_buffer_binding_t bindings[TEXT_GROUP_DRAFT_ARITY + 1];
  const bool scratch = stage->transient.required_byte_length != 0;
  if (stage->binding_count != TEXT_BINDING_COUNT &&
      stage->binding_count != TEXT_GROUP_DRAFT_ARITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider emitted an incompatible leaf");
  }
  // Carry, draft cache and draft page table are optional source state. Their
  // absence binds null; HAL submission validation at append rejects a leaf
  // that consumes one.
  const bool optional[IREE_ARRAYSIZE(roots)] = {
      [6] = true, [9] = true, [10] = true};
  for (iree_host_size_t i = 0; i < stage->binding_count; ++i) {
    if (!roots[i] && !optional[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "group leaf uses an unavailable resource");
    }
    iree_host_size_t slot =
        i + (scratch && i >= stage->transient.binding_index);
    bindings[slot] = (iree_hal_buffer_binding_t){
        roots[i], 0, roots[i] ? iree_hal_buffer_byte_length(roots[i]) : 0};
  }
  if (scratch) {
    bindings[stage->transient.binding_index] =
        (iree_hal_buffer_binding_t){bank->roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
                                    0, stage->transient.required_byte_length};
  }
  const loom_serve_group_node_t node = {
      .kind = LOOM_SERVE_GROUP_EXECUTE,
      .command_buffer = stage->command_buffer,
      .bindings = {stage->binding_count + scratch, bindings}};
  return loom_serve_execution_group_append(model->prefill_group.pending, &node);
}

// Appends the packet uploads, plan leaves and B's feedback, then submits. Every
// node is validated at append, so a rejection here accepts nothing.
static iree_status_t loom_serve_text_group_submit(
    loom_serve_text_model_t* model, iree_host_size_t first_shape,
    iree_host_size_t second_shape,
    loom_serve_group_submission_t* out_submission) {
  const loom_serve_text_bank_t* b = &model->banks[1];
  for (iree_host_size_t bank = 0; bank < 2; ++bank) {
    const loom_serve_text_bank_t* storage = &model->banks[bank];
    for (iree_host_size_t packet = 0; packet < 2; ++packet) {
      const uint32_t host_slot = packet ? LOOM_SERVE_TEXT_BANK_HOST_TOKENS
                                        : LOOM_SERVE_TEXT_BANK_HOST_METADATA;
      const uint32_t root_slot =
          packet ? LOOM_SERVE_TEXT_BANK_TOKENS : LOOM_SERVE_TEXT_BANK_METADATA;
      const loom_serve_group_node_t upload = {
          .kind = LOOM_SERVE_GROUP_TRANSFER,
          .transfer = {
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
              .upload = {.source = storage->bytes[host_slot].data,
                         .target_buffer = storage->roots[root_slot],
                         .length = storage->bytes[host_slot].data_length}}};
      IREE_RETURN_IF_ERROR(loom_serve_execution_group_append(
          model->prefill_group.pending, &upload));
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_plan_encode(
      model->execution_plan, first_shape, second_shape,
      loom_serve_text_group_emit, model));
  const loom_serve_group_node_t feedback = {
      .kind = LOOM_SERVE_GROUP_TRANSFER,
      .transfer = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {
              .source_buffer = b->roots[LOOM_SERVE_TEXT_BANK_OUTPUTS],
              .target = b->bytes[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS].data,
              .length =
                  b->bytes[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS].data_length}}};
  IREE_RETURN_IF_ERROR(loom_serve_execution_group_append(
      model->prefill_group.pending, &feedback));
  return loom_serve_execution_group_submit(model->prefill_group.pending,
                                           out_submission);
}

iree_status_t loom_serve_text_model_submit_epoch_pair(
    loom_serve_text_model_t* model, iree_host_size_t first_shape,
    const loom_serve_text_span_t* first, iree_host_size_t second_shape,
    const loom_serve_text_span_t* second,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (!model->two_bank || !model->execution_plan ||
      model->prefill_group.pending || model->prefill_group.active ||
      model->invocation.span_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "group model is unavailable or busy");
  }
  if (!first || !second || first->row_index >= model->row_count ||
      first->row_index != second->row_index ||
      first_shape >= model->shape_count || second_shape >= model->shape_count ||
      first->media || second->media || first->flags ||
      (second->flags & ~LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT) ||
      !first->token_count || !second->token_count || !first->token_ids ||
      !second->token_ids ||
      first->token_count > model->shapes[first_shape].token_capacity ||
      second->token_count > model->shapes[second_shape].token_capacity ||
      first->input_offset > IREE_HOST_SIZE_MAX - first->token_count ||
      second->input_offset != first->input_offset + first->token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid two-bank known-input pair");
  }
  if (!loom_serve_text_model_can_pair(model, first_shape, first, second_shape,
                                      second)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "provider declined the two-bank pair");
  }
  loom_serve_text_row_t* row = model->rows[first->row_index];
  const iree_host_size_t total = first->token_count + second->token_count;
  if (row->state->snapshot ||
      total > model->context_capacity - row->state->position ||
      (row->state->reserved_extent &&
       total > row->state->reserved_extent - row->state->position)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "group exceeds retained row capacity");
  }
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire(model->residency));
  iree_status_t status = loom_serve_execution_group_create(
      model->execution,
      loom_serve_execution_plan_node_capacity(model->execution_plan) + 5,
      model->host_allocator, &model->prefill_group.pending);
  if (!iree_status_is_ok(status)) {
    loom_serve_residency_release(model->residency);
    return status;
  }
  model->prefill_group.active = true;
  model->prefill_group.row_index = first->row_index;
  model->prefill_group.known_count = (uint32_t)total;
  model->prefill_group.selects = second->flags != 0;
  model->invocation.span_count = 1;
  model->invocation.row_indices[0] = first->row_index;
  const uint32_t extent = (uint32_t)total;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_grow(&model->state, 1, first, &extent));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_publish_state_bindings(model));
  const loom_serve_text_span_t* spans[] = {first, second};
  for (iree_host_size_t b = 0; b < 2; ++b) {
    loom_serve_text_bank_t* bank = &model->banks[b];
    loom_serve_text_epoch_plan_write(
        model->state.rows, bank->bytes[LOOM_SERVE_TEXT_BANK_PLAN],
        bank->bytes[LOOM_SERVE_TEXT_BANK_INPUT], 1, spans[b], NULL, NULL, 0, 0);
  }
  loom_serve_text_bank_t* a = &model->banks[0];
  loom_serve_text_bank_t* b = &model->banks[1];
  iree_vm_variant_t arguments[] = {
      loom_serve_text_group_host(model, model->source.control_state),
      loom_serve_text_group_host(model, a->host[LOOM_SERVE_TEXT_BANK_PLAN]),
      loom_serve_text_group_host(model, a->host[LOOM_SERVE_TEXT_BANK_INPUT]),
      loom_serve_text_group_host(model, b->host[LOOM_SERVE_TEXT_BANK_PLAN]),
      loom_serve_text_group_host(model, b->host[LOOM_SERVE_TEXT_BANK_INPUT]),
      loom_serve_text_group_host(model,
                                 a->host[LOOM_SERVE_TEXT_BANK_HOST_METADATA]),
      loom_serve_text_group_host(model,
                                 a->host[LOOM_SERVE_TEXT_BANK_HOST_TOKENS]),
      loom_serve_text_group_host(model,
                                 b->host[LOOM_SERVE_TEXT_BANK_HOST_METADATA]),
      loom_serve_text_group_host(model,
                                 b->host[LOOM_SERVE_TEXT_BANK_HOST_TOKENS])};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_group_invoke(model, model->prefill_group.encode,
                                   iree_vm_variant_span_from_array(arguments),
                                   iree_vm_variant_span_empty()));
  const uint8_t* plan_b = b->bytes[LOOM_SERVE_TEXT_BANK_PLAN].data;
  if (iree_unaligned_load_le_u32(plan_b + TEXT_PLAN_FIELD_POSITION * 4) !=
      row->state->position + first->token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source did not reserve contiguous group positions");
  }
  loom_serve_group_submission_t submission = {0};
  status = loom_serve_text_group_submit(model, first_shape, second_shape,
                                        &submission);
  *out_frontier = submission.frontier;
  if (!iree_status_is_ok(status) && !submission.accepted_count) {
    // No group node reached a queue: return to the drained boundary. Accepted
    // state growth only extends the row's reservation.
    *out_frontier = loom_serve_execution_frontier(model->execution);
    loom_serve_execution_group_free(model->prefill_group.pending);
    model->prefill_group.pending = NULL;
    model->prefill_group.active = false;
    model->invocation.span_count = 0;
    loom_serve_residency_release(model->residency);
  }
  return status;
}

iree_status_t loom_serve_text_prefill_group_complete(
    loom_serve_text_model_t* model, loom_serve_text_result_t* out_result) {
  loom_serve_text_bank_t* a = &model->banks[0];
  loom_serve_text_bank_t* b = &model->banks[1];
  iree_vm_variant_t arguments[] = {
      loom_serve_text_group_host(model, a->host[LOOM_SERVE_TEXT_BANK_PLAN]),
      loom_serve_text_group_host(model, b->host[LOOM_SERVE_TEXT_BANK_PLAN]),
      loom_serve_text_group_host(model,
                                 b->host[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS]),
      loom_serve_text_group_host(
          model, model->control.host.buffers[TEXT_HOST_PROGRESS])};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_group_invoke(model, model->prefill_group.publish,
                                   iree_vm_variant_span_from_array(arguments),
                                   iree_vm_variant_span_empty()));
  loom_serve_text_result_t result;
  uint32_t source_position = 0;
  loom_serve_text_epoch_plan_read_progress(
      model->control.host.bytes[TEXT_HOST_PROGRESS].data, &result,
      &source_position);
  if (result.known_count != model->prefill_group.known_count ||
      result.consumed_count != result.known_count ||
      result.output_count != (uint32_t)model->prefill_group.selects ||
      result.verification_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source published invalid group progress");
  }
  loom_serve_text_row_t* row = model->rows[model->prefill_group.row_index];
  row->state->position += result.consumed_count;
  row->state->source_position = source_position;
  loom_serve_text_state_row_trim(row->state);
  row->has_prediction = result.output_count != 0;
  if (row->has_prediction) {
    row->transfer.token = result.tokens[0];
    row->transfer.progress[0] = 1;
    row->transfer.progress[7] = result.tokens[0] == model->source.eos_token;
  }
  if (out_result) {
    *out_result = result;
  }
  model->invocation.span_count = 0;
  model->prefill_group.active = false;
  loom_serve_execution_group_free(model->prefill_group.pending);
  model->prefill_group.pending = NULL;
  loom_serve_residency_release(model->residency);
  return iree_ok_status();
}
