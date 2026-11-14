// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/prefill_group.h"

#include <string.h>

#include "iree/vm/sync.h"
#include "loom/runtime/device/pool.h"
#include "loom/serve/service/text/epoch_plan.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/variant.h"

static loom_runtime_pool_t* loom_serve_text_group_pool(
    const loom_serve_text_model_t* model) {
  return model->exchange_pool ? model->exchange_pool : model->pool;
}

static iree_hal_buffer_t* loom_serve_text_group_shared(
    const loom_serve_text_model_t* model) {
  return model->prefill_group.shared_storage
             ? model->prefill_group.shared_storage
             : model->banks[0].roots[LOOM_SERVE_TEXT_BANK_RESIDUAL];
}

static iree_vm_variant_t loom_serve_text_group_host(
    loom_serve_text_model_t* model, iree_vm_buffer_t* buffer) {
  return iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                                  buffer);
}

static iree_status_t loom_serve_text_group_invoke(
    loom_serve_text_model_t* model, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results) {
  iree_status_t status =
      iree_vm_invoke(loom_runtime_program_invocation(model->control.program),
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
  const loom_runtime_plan_group_query_t query = {
      .position = model->rows[first->row_index]->state->position,
      .shapes = {first_shape, second_shape},
      .tokens = {first->token_count, second->token_count}};
  return !model->plan_provider.select ||
         model->plan_provider.select(model->plan_provider.self,
                                     model->execution_plan, &query);
}

static const loom_runtime_stage_t* loom_serve_text_group_stages(
    const loom_serve_text_model_t* model) {
  return model->prefill_group.stages ? model->prefill_group.stages
                                     : model->stage_bindings;
}

static iree_status_t loom_serve_text_group_transport(
    loom_serve_text_model_t* model, const loom_runtime_stage_t* bindings,
    iree_const_byte_span_t bytes, iree_host_size_t offset,
    const uint32_t* stages, iree_host_size_t stage_entries, bool continuous) {
  if (offset > bytes.data_length || bytes.data_length - offset < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resident transport table is missing");
  }
  const uint32_t count = iree_unaligned_load_le_u32(bytes.data + offset);
  if (!count || count > (continuous ? 1 : model->stage_count) ||
      bytes.data_length - offset != 4 + (uint64_t)count * 24 ||
      !loom_runtime_pool_is_uncached(loom_serve_text_group_pool(model))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resident transport requires a valid table and "
                            "a shared pool with persistent host access");
  }
  loom_serve_text_resident_binding_t* records = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, count, sizeof(*records), (void**)&records));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const uint8_t* p = bytes.data + offset + 4 + i * 24;
    records[i] = (loom_serve_text_resident_binding_t){
        .producer = iree_unaligned_load_le_u32(p),
        .consumer = continuous ? UINT32_MAX : iree_unaligned_load_le_u32(p + 4),
        .request_offset = iree_unaligned_load_le_u64(p + 8),
        .response_offset = iree_unaligned_load_le_u64(p + 16),
        .generations = continuous ? iree_unaligned_load_le_u32(p + 4) : 1};
    const loom_serve_text_resident_binding_t* record = &records[i];
    if (continuous) {
      if (record->producer >= model->stage_count || !record->generations ||
          record->generations >= INT32_MAX || record->request_offset % 64 ||
          record->response_offset % 64 ||
          bindings[record->producer].binding_count != TEXT_BINDING_COUNT ||
          !bindings[record->producer].command_buffer ||
          !bindings[record->producer].execution ||
          bindings[record->producer].execution == model->execution) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid continuous resident lifetime");
      }
      for (iree_host_size_t j = 0; j < stage_entries; ++j) {
        if (stages[j] == record->producer) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "resident lifetime is also a plan leaf");
        }
        if (stages[j] != UINT32_MAX && bindings[stages[j]].execution &&
            bindings[stages[j]].execution != model->execution) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "continuous leaves require the primary lane");
        }
        if (stages[j] != UINT32_MAX && bindings[stages[j]].command_buffer &&
            !loom_runtime_pool_has_live_access(
                loom_serve_text_group_pool(model),
                iree_hal_command_buffer_queue_family(
                    bindings[stages[j]].command_buffer))) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "continuous leaves require live pool access");
        }
      }
      continue;
    }
    bool pair = false;
    for (iree_host_size_t j = 0; j < stage_entries; j += 12) {
      const uint32_t producers[] = {2, 5, 9};
      for (iree_host_size_t k = 0; k < IREE_ARRAYSIZE(producers); ++k) {
        pair |= stages[j + producers[k]] == record->producer &&
                stages[j + producers[k] + 1] == record->consumer;
      }
    }
    if (record->producer >= model->stage_count ||
        record->consumer >= model->stage_count || !pair ||
        record->request_offset % 64 || record->response_offset % 64 ||
        bindings[record->producer].execution ==
            bindings[record->consumer].execution) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid resident producer/consumer pair");
    }
    if (iree_status_is_ok(status)) {
      const loom_runtime_stage_t* consumer = &bindings[record->consumer];
      if (!consumer->command_buffer ||
          !loom_runtime_pool_has_live_access(
              loom_serve_text_group_pool(model),
              iree_hal_command_buffer_queue_family(consumer->command_buffer))) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "resident consumer requires uncached or host-coherent pool access");
      }
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (records[j].producer == record->producer) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "duplicate resident producer");
      }
    }
  }
  for (iree_host_size_t j = 0;
       !continuous && j < stage_entries && iree_status_is_ok(status); j += 12) {
    if (stages[j] == UINT32_MAX) {
      continue;
    }
    const uint32_t producers[] = {2, 5, 9};
    for (iree_host_size_t k = 0; k < IREE_ARRAYSIZE(producers); ++k) {
      if (stages[j + producers[k]] == UINT32_MAX) {
        continue;
      }
      bool found = false;
      for (iree_host_size_t i = 0; i < count; ++i) {
        found |= records[i].producer == stages[j + producers[k]];
      }
      if (!found) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "resident producer has no lifecycle");
      }
    }
  }
  if (iree_status_is_ok(status) && model->prefill_group.transport) {
    if (model->prefill_group.continuous != continuous ||
        model->prefill_group.transport_count != count ||
        memcmp(model->prefill_group.transport, records,
               count * sizeof(*records))) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "resident transport geometry cannot change");
    }
  } else if (iree_status_is_ok(status)) {
    model->prefill_group.transport = records;
    model->prefill_group.transport_count = count;
    model->prefill_group.continuous = continuous;
    records = NULL;
  }
  iree_allocator_free(model->host_allocator, records);
  return status;
}

// Exports the source stage table for |control_state| and checks every leaf
// against |bindings| before the provider prepares an immutable plan.
static iree_status_t loom_serve_text_group_describe(
    loom_serve_text_model_t* model, iree_vm_buffer_t* control_state,
    const loom_runtime_stage_t* bindings,
    loom_runtime_execution_plan_t** out_plan) {
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
    const uint32_t version =
        bytes.data_length >= 16 ? iree_unaligned_load_le_u32(bytes.data) : 0;
    const iree_host_size_t phases =
        version >= 3   ? LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT
        : version == 2 ? LOOM_RUNTIME_PLAN_FFN_PHASE_COUNT
                       : 3;
    if (bytes.data_length < 16 ||
        (version != 1 && version != 2 && version != 3 && version != 4) ||
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
      const iree_host_size_t count = model->shape_count * layers * phases;
      if ((version < 3 && bytes.data_length != 16 + count * 4) ||
          bytes.data_length < 16 + count * 4) {
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
                !(version == 4 && bindings[stages[i]].binding_count == 7) &&
                bindings[stages[i]].binding_count != TEXT_GROUP_PAIR_ARITY &&
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
        if (iree_status_is_ok(status) && version >= 3) {
          status = loom_serve_text_group_transport(model, bindings, bytes,
                                                   16 + count * 4, stages,
                                                   count, version == 4);
        } else if (iree_status_is_ok(status) &&
                   model->prefill_group.transport) {
          status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                    "resident plan cannot drop its transport");
        }
        if (iree_status_is_ok(status)) {
          const loom_runtime_plan_description_t description = {
              model->shape_count, layers, model->stage_count, stages, phases};
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
static iree_status_t loom_serve_text_group_workspace(
    const loom_serve_text_model_t* model, const loom_runtime_stage_t* bindings,
    iree_hal_pool_reservation_request_t* out_request) {
  iree_device_size_t length = 1;
  iree_device_size_t alignment = 256;
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    length = iree_max(length, bindings[i].transient.required_byte_length);
    alignment = iree_max(alignment, bindings[i].transient.minimum_alignment);
  }
  // Parallel lanes cannot alias command-private temporary storage.
  if (length > IREE_DEVICE_SIZE_MAX - (alignment - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "group workspace alignment overflows");
  }
  length = iree_device_align(length, alignment);
  const iree_host_size_t lane_count = iree_max(model->placement.count, 1);
  if (length > IREE_DEVICE_SIZE_MAX / lane_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "group lane workspaces overflow");
  }
  *out_request =
      iree_hal_make_pool_reservation_request(length * lane_count, alignment);
  return iree_ok_status();
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
  return loom_runtime_residency_check_working_set(
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
                                                   buffer, buffer};
  const loom_runtime_program_binding_t bindings[] = {
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
  IREE_RETURN_IF_ERROR(loom_runtime_program_bind(
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
  IREE_RETURN_IF_ERROR(loom_serve_text_group_workspace(
      model, model->stage_bindings, &model->prefill_group.workspace));
  requests[4] = requests[5] = model->prefill_group.workspace;
  IREE_RETURN_IF_ERROR(loom_serve_text_group_check_working_set(
      model, original, model->prefill_group.workspace.allocation_size));
  iree_hal_buffer_t* extra[6] = {NULL};
  iree_status_t status = loom_runtime_execution_alloca_from_pool(
      model->execution, loom_runtime_pool_storage(model->pool),
      IREE_ARRAYSIZE(extra), requests, extra);
  if (iree_status_is_ok(status) && model->exchange_pool &&
      model->prefill_group.continuous) {
    status = loom_runtime_execution_alloca_from_pool(
        model->execution, loom_runtime_pool_storage(model->exchange_pool), 1,
        &requests[0], &model->prefill_group.shared_storage);
    if (iree_status_is_ok(status)) {
      status = loom_runtime_retirement_track(
          &model->state.retirement, &model->prefill_group.shared_storage,
          model->host_allocator);
    }
  }
  for (iree_host_size_t i = 0; i < 6 && iree_status_is_ok(status); ++i) {
    status = loom_runtime_retirement_track(&model->state.retirement, &extra[i],
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
    loom_runtime_execution_plan_t** out_plan) {
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
                                              loom_runtime_stage_t* stages) {
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
  const loom_runtime_stage_t* bindings = model->stage_bindings;
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
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_group_workspace(model, bindings,
                                             &out_candidate->workspace);
  }
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
      status = loom_runtime_execution_alloca_from_pool(
          model->execution, loom_runtime_pool_storage(model->pool),
          IREE_ARRAYSIZE(requests), requests, out_candidate->workspaces);
    }
    for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = loom_runtime_retirement_track(&model->state.retirement,
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
    loom_runtime_execution_plan_release(model->execution_plan);
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
  loom_runtime_execution_plan_release(candidate->plan);
  loom_serve_text_group_stages_free(candidate->host_allocator,
                                    candidate->stage_count, candidate->stages);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_hal_buffer_release(candidate->workspaces[i]);
  }
  memset(candidate, 0, sizeof(*candidate));
}

void loom_serve_text_prefill_group_retire(loom_serve_text_model_t* model) {
  loom_serve_text_group_exchange_abort(model->prefill_group.exchange);
  loom_serve_text_group_exchange_free(model->prefill_group.exchange);
  model->prefill_group.exchange = NULL;
  iree_hal_buffer_release(model->prefill_group.shared_storage);
  model->prefill_group.shared_storage = NULL;
  loom_serve_text_group_stages_free(model->host_allocator, model->stage_count,
                                    model->prefill_group.stages);
  model->prefill_group.stages = NULL;
}

typedef struct loom_serve_text_group_encoder_t {
  loom_serve_text_model_t* model;
  uint32_t inputs;
  uint32_t last;
  const loom_serve_text_resident_binding_t** residents;
} loom_serve_text_group_encoder_t;

static iree_status_t loom_serve_text_group_emit(
    void* self, uint32_t bank_index, uint32_t stage_index,
    iree_host_size_t predecessor_count, const uint32_t* predecessors,
    uint32_t* out_node) {
  loom_serve_text_group_encoder_t* encoder = self;
  loom_serve_text_model_t* model = encoder->model;
  if (bank_index > 1 || stage_index >= model->stage_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider emitted an unknown group leaf");
  }
  const loom_runtime_stage_t* stage =
      &loom_serve_text_group_stages(model)[stage_index];
  // A pair command owns both banks and is recorded once, at bank A's visit.
  const bool pair = stage->binding_count == TEXT_GROUP_PAIR_ARITY;
  if (pair && bank_index) {
    if (encoder->last == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pair command requires bank A first");
    }
    *out_node = encoder->last;
    return iree_ok_status();
  }
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
                                model->state.draft.next_results,
                                loom_serve_text_group_shared(model)};
  if (model->prefill_group.continuous &&
      stage_index == model->prefill_group.transport[0].producer) {
    roots[0] = loom_serve_text_group_shared(model);
  }
  if (pair) {
    const loom_serve_text_bank_t* other = &model->banks[1];
    roots[6] = other->roots[LOOM_SERVE_TEXT_BANK_RESIDUAL];
    roots[7] = other->roots[LOOM_SERVE_TEXT_BANK_METADATA];
    roots[8] = model->state.epoch.buffers[2];
    roots[9] = model->state.row_arena;
    roots[10] = other->roots[LOOM_SERVE_TEXT_BANK_TOKENS];
    roots[11] = other->roots[LOOM_SERVE_TEXT_BANK_OUTPUTS];
  }
  if (model->prefill_group.continuous && stage->binding_count == 7) {
    roots[6] = loom_serve_text_group_shared(model);
  }
  iree_hal_buffer_binding_t bindings[TEXT_GROUP_DRAFT_ARITY + 2];
  const bool scratch = stage->transient.required_byte_length != 0;
  if (stage->binding_count != TEXT_BINDING_COUNT &&
      !(model->prefill_group.continuous && stage->binding_count == 7) &&
      stage->binding_count != TEXT_GROUP_DRAFT_ARITY && !pair) {
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
    const iree_device_size_t stride =
        model->prefill_group.workspace.allocation_size /
        iree_max(model->placement.count, 1);
    const uint32_t lane =
        model->placement.stages ? model->placement.stages[stage_index] : 0;
    bindings[stage->transient.binding_index] = (iree_hal_buffer_binding_t){
        bank->roots[LOOM_SERVE_TEXT_BANK_WORKSPACE], stride * lane,
        stage->transient.required_byte_length};
  }
  loom_runtime_group_node_t node = {
      .kind = LOOM_RUNTIME_GROUP_EXECUTE,
      .command_buffer = stage->command_buffer,
      .bindings = {stage->binding_count + scratch, bindings}};
  const loom_serve_text_resident_binding_t* resident = NULL;
  for (iree_host_size_t i = 0; !model->prefill_group.continuous &&
                               i < model->prefill_group.transport_count;
       ++i) {
    if (model->prefill_group.transport[i].producer == stage_index) {
      resident = &model->prefill_group.transport[i];
      break;
    }
  }
  if (resident) {
    IREE_RETURN_IF_ERROR(loom_serve_text_group_exchange_add(
        model->prefill_group.exchange, model->device, roots[0],
        resident->request_offset, resident->response_offset,
        &node.payload_semaphore));
    node.payload_value = 1;
  }
  if (!predecessor_count) {
    predecessors = &encoder->inputs;
    predecessor_count = 1;
  }
  loom_runtime_group_edge_t edges[LOOM_RUNTIME_GROUP_MAX_LANES];
  if (predecessor_count > IREE_ARRAYSIZE(edges)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many resident group predecessors");
  }
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    const loom_serve_text_resident_binding_t* producer =
        encoder->residents[predecessors[i]];
    if (producer && producer->consumer != stage_index) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "resident payload requires its declared consumer");
    }
    edges[i] = (loom_runtime_group_edge_t){
        predecessors[i], producer ? LOOM_RUNTIME_GROUP_EDGE_PAYLOAD
                                  : LOOM_RUNTIME_GROUP_EDGE_TERMINAL};
  }
  IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append_edges(
      model->prefill_group.pending, stage->execution, &node, predecessor_count,
      edges, out_node));
  encoder->last = *out_node;
  encoder->residents[*out_node] = resident;
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    const loom_serve_text_resident_binding_t* producer =
        encoder->residents[predecessors[i]];
    if (!producer) {
      continue;
    }
    const uint32_t ack_word = 1;
    const loom_runtime_group_node_t ack = {
        .kind = LOOM_RUNTIME_GROUP_TRANSFER,
        .transfer = {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                     .update = {.source_buffer = &ack_word,
                                .target_buffer = roots[0],
                                .target_offset = producer->request_offset + 64,
                                .length = 4}}};
    IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append_after(
        model->prefill_group.pending, stage->execution, &ack, 1, out_node,
        out_node));
    const uint32_t retirement[] = {*out_node, predecessors[i]};
    const loom_runtime_group_node_t barrier = {.kind =
                                                   LOOM_RUNTIME_GROUP_BARRIER};
    IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append_after(
        model->prefill_group.pending, stage->execution, &barrier,
        IREE_ARRAYSIZE(retirement), retirement, out_node));
  }
  return iree_ok_status();
}

// Appends the packet uploads, plan leaves and B's feedback, then submits. Every
// node is validated at append, so a rejection here accepts nothing.
static iree_status_t loom_serve_text_group_encode(
    loom_serve_text_model_t* model, iree_host_size_t first_shape,
    iree_host_size_t second_shape,
    const loom_serve_text_resident_binding_t** residents) {
  const loom_serve_text_bank_t* b = &model->banks[1];
  loom_serve_text_group_encoder_t encoder = {
      .model = model, .inputs = UINT32_MAX, .residents = residents};
  for (iree_host_size_t bank = 0; bank < 2; ++bank) {
    const loom_serve_text_bank_t* storage = &model->banks[bank];
    for (iree_host_size_t packet = 0; packet < 2; ++packet) {
      const uint32_t host_slot = packet ? LOOM_SERVE_TEXT_BANK_HOST_TOKENS
                                        : LOOM_SERVE_TEXT_BANK_HOST_METADATA;
      const uint32_t root_slot =
          packet ? LOOM_SERVE_TEXT_BANK_TOKENS : LOOM_SERVE_TEXT_BANK_METADATA;
      const loom_runtime_group_node_t upload = {
          .kind = LOOM_RUNTIME_GROUP_TRANSFER,
          .transfer = {
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
              .upload = {.source = storage->bytes[host_slot].data,
                         .target_buffer = storage->roots[root_slot],
                         .length = storage->bytes[host_slot].data_length}}};
      IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append_after(
          model->prefill_group.pending, NULL, &upload,
          encoder.inputs == UINT32_MAX ? 0 : 1, &encoder.inputs,
          &encoder.inputs));
    }
  }
  uint32_t lifetime = UINT32_MAX;
  if (model->prefill_group.continuous) {
    const loom_serve_text_resident_binding_t* resident =
        &model->prefill_group.transport[0];
    IREE_RETURN_IF_ERROR(loom_serve_text_group_exchange_add_resident(
        model->prefill_group.exchange, model->device,
        loom_serve_text_group_shared(model), resident->request_offset,
        resident->response_offset, resident->generations));
    IREE_RETURN_IF_ERROR(loom_serve_text_group_emit(
        &encoder, 0, resident->producer, 1, &encoder.inputs, &lifetime));
  }
  IREE_RETURN_IF_ERROR(loom_runtime_execution_plan_encode_dag(
      model->execution_plan, first_shape, second_shape,
      loom_serve_text_group_emit, &encoder));
  if (lifetime != UINT32_MAX) {
    const uint32_t terminals[] = {lifetime, encoder.last};
    const loom_runtime_group_node_t join = {.kind = LOOM_RUNTIME_GROUP_BARRIER};
    IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append_after(
        model->prefill_group.pending, NULL, &join, IREE_ARRAYSIZE(terminals),
        terminals, &encoder.last));
  }
  const loom_runtime_group_node_t feedback = {
      .kind = LOOM_RUNTIME_GROUP_TRANSFER,
      .transfer = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {
              .source_buffer = b->roots[LOOM_SERVE_TEXT_BANK_OUTPUTS],
              .target = b->bytes[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS].data,
              .length =
                  b->bytes[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS].data_length}}};
  IREE_RETURN_IF_ERROR(loom_runtime_execution_group_append(
      model->prefill_group.pending, &feedback));
  return iree_ok_status();
}

static iree_host_size_t loom_serve_text_group_capacity(
    const loom_serve_text_model_t* model) {
  return loom_runtime_execution_plan_node_capacity(model->execution_plan) + 5 +
         4 * model->prefill_group.transport_count;
}

static iree_status_t loom_serve_text_group_submit(
    loom_serve_text_model_t* model, iree_host_size_t first_shape,
    iree_host_size_t second_shape,
    loom_runtime_group_submission_t* out_submission) {
  const loom_serve_text_resident_binding_t** residents = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, loom_serve_text_group_capacity(model),
      sizeof(*residents), (void**)&residents));
  iree_status_t status = iree_ok_status();
  if (model->prefill_group.transport_count) {
    status = loom_serve_text_group_exchange_create(
        2 * model->prefill_group.transport_count, model->host_allocator,
        &model->prefill_group.exchange);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_group_encode(model, first_shape, second_shape,
                                          residents);
  }
  iree_allocator_free(model->host_allocator, residents);
  if (iree_status_is_ok(status) && model->prefill_group.exchange) {
    status = loom_serve_text_group_exchange_start(model->prefill_group.exchange,
                                                  INT64_C(90000000000));
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_execution_group_submit(model->prefill_group.pending,
                                                 out_submission);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_group_exchange_abort(model->prefill_group.exchange);
  }
  return status;
}

iree_status_t loom_serve_text_model_submit_epoch_pair(
    loom_serve_text_model_t* model, iree_host_size_t first_shape,
    const loom_serve_text_span_t* first, iree_host_size_t second_shape,
    const loom_serve_text_span_t* second,
    loom_runtime_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_runtime_execution_frontier_t){0};
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
  IREE_RETURN_IF_ERROR(loom_runtime_residency_acquire(model->residency));
  iree_status_t status = loom_runtime_execution_group_create(
      model->execution, loom_serve_text_group_capacity(model),
      model->host_allocator, &model->prefill_group.pending);
  if (!iree_status_is_ok(status)) {
    loom_runtime_residency_release(model->residency);
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
  loom_runtime_execution_plan_t* encoded_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_group_describe(
      model, model->source.control_state, loom_serve_text_group_stages(model),
      &encoded_plan));
  if (loom_runtime_execution_plan_node_capacity(encoded_plan) >
      loom_runtime_execution_plan_node_capacity(model->execution_plan)) {
    loom_runtime_execution_plan_release(encoded_plan);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "group encoder expanded the prepared plan");
  }
  if (loom_runtime_execution_plan_equal(model->execution_plan, encoded_plan)) {
    loom_runtime_execution_plan_release(encoded_plan);
  } else {
    loom_runtime_execution_plan_release(model->execution_plan);
    model->execution_plan = encoded_plan;
  }
  const uint8_t* plan_b = b->bytes[LOOM_SERVE_TEXT_BANK_PLAN].data;
  if (iree_unaligned_load_le_u32(plan_b + TEXT_PLAN_FIELD_POSITION * 4) !=
      row->state->position + first->token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source did not reserve contiguous group positions");
  }
  loom_runtime_group_submission_t submission = {0};
  status = loom_serve_text_group_submit(model, first_shape, second_shape,
                                        &submission);
  *out_frontier = submission.frontier;
  if (!iree_status_is_ok(status) && !submission.accepted_count) {
    // No group node reached a queue: return to the drained boundary. Accepted
    // state growth only extends the row's reservation.
    *out_frontier = loom_runtime_execution_frontier(model->execution);
    loom_runtime_execution_group_free(model->prefill_group.pending);
    model->prefill_group.pending = NULL;
    loom_serve_text_group_exchange_free(model->prefill_group.exchange);
    model->prefill_group.exchange = NULL;
    model->prefill_group.active = false;
    model->invocation.span_count = 0;
    loom_runtime_residency_release(model->residency);
  }
  return status;
}

iree_status_t loom_serve_text_prefill_group_complete(
    loom_serve_text_model_t* model, loom_serve_text_result_t* out_result) {
  IREE_RETURN_IF_ERROR(
      loom_serve_text_group_exchange_check(model->prefill_group.exchange));
  loom_serve_text_bank_t* a = &model->banks[0];
  loom_serve_text_bank_t* b = &model->banks[1];
  iree_vm_variant_t arguments[] = {
      loom_serve_text_group_host(model, model->source.control_state),
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
  loom_runtime_execution_group_free(model->prefill_group.pending);
  model->prefill_group.pending = NULL;
  loom_serve_text_group_exchange_free(model->prefill_group.exchange);
  model->prefill_group.exchange = NULL;
  loom_runtime_residency_release(model->residency);
  return iree_ok_status();
}
