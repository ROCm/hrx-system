// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 indirect dispatch requires device-populated "
                            "hidden launch geometry");
  }
  if (function->launch.requirements || function->kernarg->requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch requires native launch services 0x%x and argument services "
        "0x%x that this queue has not provisioned",
        function->launch.requirements, function->kernarg->requirements);
  }
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->metadata;
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 workgroup clusters require native queue "
                            "admission and cluster launch state");
  }
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  const bool native_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  if (native_arguments && indirect_arguments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native direct and indirect arguments conflict");
  }
  if (indirect_arguments) {
    if (bindings.count != 1 || constants.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "indirect arguments require one binding and "
                              "no constants");
    }
  } else if (native_arguments) {
    if (bindings.count ||
        constants.data_length < function->kernarg->native_byte_length ||
        constants.data_length > function->kernarg->byte_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native argument extent does not match the "
                              "function ABI");
    }
  } else if (bindings.count != function->kernarg->binding_count ||
             constants.data_length != function->kernarg->constant_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dense arguments do not match function reflection");
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct dispatch binding %" PRIhsz " is null", i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        bindings.values[i].buffer, bindings.values[i].offset,
        bindings.values[i].length));
  }
  if (indirect_arguments) {
    const iree_hal_buffer_ref_t reference = bindings.values[0];
    const iree_device_size_t length =
        reference.length == IREE_HAL_WHOLE_BUFFER
            ? iree_hal_buffer_byte_length(reference.buffer) - reference.offset
            : reference.length;
    if (length < function->kernarg->byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "indirect arguments do not cover the native ABI");
    }
  }
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {0};
  const bool default_size =
      !(config.workgroup_size[0] | config.workgroup_size[1] |
        config.workgroup_size[2]);
  const bool exact =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT);
  uint64_t flat_size = 1;
  bool partial = false;
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = default_size
                              ? (metadata->has_required_workgroup_size
                                     ? metadata->required_workgroup_size[i]
                                     : 1)
                              : config.workgroup_size[i];
    if (!size || size > 1024 ||
        (metadata->has_required_workgroup_size &&
         size != metadata->required_workgroup_size[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "workgroup dimension %d violates function or "
                              "PM4 geometry",
                              i);
    }
    flat_size *= size;
    dispatch.workgroup_size[i] = size;
    if (exact) {
      if (!config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "exact work-item dimensions must be nonzero");
      }
      const uint32_t remainder = config.workitem_count[i] % size;
      dispatch.workgroup_count[i] =
          config.workitem_count[i] / size + (remainder != 0);
      dispatch.partial_size[i] = remainder ? remainder : size;
      partial |= remainder != 0;
    } else {
      if (config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "work-item dimensions require the exact flag");
      }
      dispatch.workgroup_count[i] = config.workgroup_count[i];
    }
    if (metadata->max_workgroup_count[i] &&
        dispatch.workgroup_count[i] > metadata->max_workgroup_count[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "workgroup count exceeds function metadata");
    }
  }
  if (flat_size > 1024 || (metadata->max_flat_workgroup_size &&
                           flat_size > metadata->max_flat_workgroup_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup exceeds function or PM4 invocation "
                            "capacity");
  }
  if (partial && metadata->uniform_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function requires complete workgroups");
  }
  if (!partial) {
    memset(dispatch.partial_size, 0, sizeof(dispatch.partial_size));
  }
  const uint64_t group_bytes = (uint64_t)metadata->group_segment_fixed_size +
                               config.dynamic_workgroup_local_memory;
  if (group_bytes > maximum_group_segment_byte_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch LDS exceeds the native queue capacity");
  }
  dispatch.group_segment_byte_length = (uint32_t)group_bytes;
  *out_dispatch = dispatch;
  return iree_ok_status();
}

// Captures common ownership and allocates the operation payload in one block.
static iree_status_t iree_hal_amd_gpu_pm4_submission_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_size,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission, void** out_payload) {
  iree_host_size_t payload_offset = sizeof(iree_hal_amd_gpu_pm4_submission_t);
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul_add(
          payload_offset, waits.count,
          sizeof(iree_hal_amd_gpu_pm4_submission_wait_t), &payload_offset) ||
      !iree_host_size_checked_mul_add(
          payload_offset, signals.count,
          sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t), &payload_offset) ||
      !iree_host_size_checked_align(payload_offset, 8, &payload_offset) ||
      !iree_host_size_checked_add(payload_offset, payload_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "submission capture size overflows");
  }
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_capture_pool_acquire(pool, total_size, &storage));
  iree_hal_amd_gpu_pm4_submission_t* submission = (void*)storage.data;
  *submission = (iree_hal_amd_gpu_pm4_submission_t){
      .native_wait_count = waits.count ? UINT32_MAX : 0,
      .pool = pool,
      .allocation_size = storage.data_length,
      .capture_size = total_size,
      .wait_count = waits.count,
      .before = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->before : NULL),
      .after = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->after : NULL),
  };
  iree_atomic_ref_count_init(&submission->ref_count);
  uint8_t* cursor = (uint8_t*)(submission + 1);
  submission->waits = (iree_hal_amd_gpu_pm4_submission_wait_t*)cursor;
  cursor += waits.count * sizeof(*submission->waits);
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_pm4_submission_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  submission->signals.count = signals.count;
  // Values precede pointers so odd signal counts retain alignment on 32-bit
  // hosts. The payload begins at its independently aligned offset above.
  submission->signals.payload_values = (uint64_t*)cursor;
  cursor += signals.count * sizeof(uint64_t);
  submission->signals.semaphores = (iree_hal_semaphore_t**)cursor;
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    submission->signals.semaphores[i] = signals.semaphores[i];
    submission->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  *out_submission = submission;
  *out_payload = storage.data + payload_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  iree_host_size_t payload_size = 0;
  if (!iree_host_size_checked_mul_add(
          constants.data_length, bindings.count,
          sizeof(iree_hal_buffer_ref_t) + sizeof(uint64_t), &payload_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch capture size overflows");
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, waits, signals, payload_size, profile, barriers, &submission,
      &payload));
  submission->kind = function ? IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH
                              : IREE_HAL_AMD_GPU_PM4_SUBMISSION_BARRIER;
  submission->dispatch.executable = executable;
  submission->dispatch.function = function;
  submission->dispatch.geometry = dispatch;
  submission->dispatch.flags = flags;
  iree_hal_executable_retain(executable);
  iree_hal_buffer_ref_t* references = payload;
  submission->dispatch.bindings =
      (iree_hal_buffer_ref_list_t){bindings.count, references};
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    references[i] = bindings.values[i];
    iree_hal_buffer_retain(references[i].buffer);
  }
  uint8_t* cursor = (uint8_t*)payload +
                    bindings.count * (sizeof(*references) + sizeof(uint64_t));
  submission->dispatch.constants =
      iree_make_const_byte_span(cursor, constants.data_length);
  if (constants.data_length) {
    memcpy(cursor, constants.data, constants.data_length);
  }
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_amd_gpu_pm4_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_transfer_measure(count, operations, &layout));
  if (!layout.count) {
    return iree_hal_amd_gpu_pm4_submission_create(
        pool, waits, signals, NULL, NULL, (iree_hal_amd_gpu_pm4_dispatch_t){0},
        iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(), profile,
        barriers, 0, out_submission);
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, waits, signals, layout.byte_length, profile, barriers, &submission,
      &payload));
  submission->kind = IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER;
  iree_status_t status = iree_hal_amd_gpu_pm4_transfer_initialize(
      family, count, operations, layout, payload, &submission->transfer);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
}

iree_byte_span_t iree_hal_amd_gpu_pm4_submission_tail_storage(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  const iree_host_size_t aligned_size =
      iree_host_align(submission->capture_size, iree_max_align_t);
  if (aligned_size >= submission->allocation_size) {
    return iree_byte_span_empty();
  }
  return iree_make_byte_span((uint8_t*)submission + aligned_size,
                             submission->allocation_size - aligned_size);
}

void iree_hal_amd_gpu_pm4_submission_retain(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission) {
    iree_atomic_ref_count_inc(&submission->ref_count);
  }
}

void iree_hal_amd_gpu_pm4_submission_retire(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH) {
    for (iree_host_size_t i = 0; i < submission->dispatch.bindings.count; ++i) {
      iree_hal_buffer_release(submission->dispatch.bindings.values[i].buffer);
    }
    iree_hal_executable_release(submission->dispatch.executable);
  } else if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER) {
    iree_hal_amd_gpu_pm4_transfer_deinitialize(&submission->transfer);
  }
  submission->kind = IREE_HAL_AMD_GPU_PM4_SUBMISSION_BARRIER;
}

static void iree_hal_amd_gpu_pm4_submission_destroy(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_capture_pool_t* pool = submission->pool;
  iree_hal_amd_gpu_pm4_submission_retire(submission);
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    iree_hal_amd_gpu_pm4_signal_release(submission->waits[i].signal);
    iree_hal_semaphore_release(submission->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    iree_hal_semaphore_release(submission->signals.semaphores[i]);
  }
  iree_hal_amd_gpu_pm4_signal_release(submission->signal);
  iree_hal_amd_gpu_capture_pool_release(
      pool, iree_make_byte_span(submission, submission->allocation_size));
}

void iree_hal_amd_gpu_pm4_submission_release(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission && iree_atomic_ref_count_dec(&submission->ref_count) == 1) {
    iree_hal_amd_gpu_pm4_submission_destroy(submission);
  }
}

iree_status_t iree_hal_amd_gpu_pm4_submission_resolve_address(
    const iree_hal_queue_family_t* family, iree_hal_buffer_ref_t reference,
    iree_hal_memory_access_t access, uint64_t* out_address) {
  const iree_hal_memory_contract_t* contract =
      reference.buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id + 1;
  if (!contract || !family->memory.domain ||
      family->memory.domain != contract->domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer is outside the dispatch family domain");
  }
  const iree_hal_memory_scope_access_t* scope = &contract->scopes[scope_id];
  const iree_hal_buffer_interface_t interface =
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const uint16_t index = scope->bindings[interface];
  if (!iree_any_bit_set(scope->interfaces, UINT32_C(1) << interface) ||
      index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "GPU address access is not prepared for this "
                            "dispatch family");
  }
  if (access) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
        iree_hal_buffer_allowed_access(reference.buffer), access));
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage_any(
      scope->usage & iree_hal_buffer_allowed_usage(reference.buffer),
      IREE_HAL_BUFFER_USAGE_STORAGE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ));
  const uint64_t base =
      iree_hal_buffer_native_binding(reference.buffer,
                                     (iree_hal_buffer_native_binding_slot_t){
                                         .index = index, .type = interface})
          .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dispatch buffer has no committed GPU address");
  }
  if (!iree_checked_add_u64(base, reference.offset, out_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch buffer address overflows");
  }
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_submission_emplace_arguments(
    const iree_hal_amd_gpu_pm4_submission_t* submission,
    const uint64_t* binding_addresses, void* target) {
  const iree_hal_amd_gpu_pm4_dispatch_t* dispatch =
      &submission->dispatch.geometry;
  iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .dynamic_lds_size =
          dispatch->group_segment_byte_length -
          submission->dispatch.function->metadata->group_segment_fixed_size,
      .grid_dimensions = 1,
  };
  for (int i = 0; i < 3; ++i) {
    const uint32_t partial = dispatch->partial_size[i];
    const bool remainder = partial && partial != dispatch->workgroup_size[i];
    inputs.block_count[i] = dispatch->workgroup_count[i] - remainder;
    inputs.group_size[i] = (uint16_t)dispatch->workgroup_size[i];
    inputs.remainder[i] = remainder ? (uint16_t)partial : 0;
    if (dispatch->workgroup_count[i] != 1 || dispatch->workgroup_size[i] != 1) {
      inputs.grid_dimensions = (uint16_t)(i + 1);
    }
  }
  if (iree_any_bit_set(submission->dispatch.flags,
                       IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    iree_hal_amd_gpu_kernarg_layout_emplace_native(
        submission->dispatch.function->kernarg, submission->dispatch.constants,
        &inputs, target);
  } else {
    iree_hal_amd_gpu_kernarg_layout_emplace(
        submission->dispatch.function->kernarg,
        submission->dispatch.constants.data, binding_addresses, &inputs,
        target);
  }
}
