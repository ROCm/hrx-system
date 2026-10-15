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

iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_arena_block_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_pm4_submission_t);
  if (!iree_host_size_checked_mul_add(
          total_size, waits.count,
          sizeof(iree_hal_amd_gpu_pm4_submission_wait_t), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, signals.count,
          sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, bindings.count,
          sizeof(iree_hal_buffer_ref_t) + sizeof(uint64_t), &total_size) ||
      !iree_host_size_checked_add(total_size, constants.data_length,
                                  &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "submission capture size overflows");
  }
  iree_arena_block_t* block = NULL;
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  if (total_size <= pool->usable_block_size) {
    IREE_RETURN_IF_ERROR(
        iree_arena_block_pool_acquire(pool, &block, (void**)&submission));
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        pool->block_allocator, total_size, (void**)&submission));
  }
  *submission = (iree_hal_amd_gpu_pm4_submission_t){
      .pool = pool,
      .block = block,
      .executable = executable,
      .function = function,
      .wait_count = waits.count,
      .flags = flags,
      .dispatch = dispatch,
      .before = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->before : NULL),
      .after = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->after : NULL),
  };
  iree_atomic_ref_count_init(&submission->ref_count);
  iree_hal_executable_retain(executable);
  uint8_t* cursor = (uint8_t*)(submission + 1);
  submission->waits = (iree_hal_amd_gpu_pm4_submission_wait_t*)cursor;
  cursor += waits.count * sizeof(*submission->waits);
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_pm4_submission_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  submission->signals.count = signals.count;
  // Keep 64-bit arrays before pointer-only arrays so odd signal counts retain
  // their natural alignment on hosts with 32-bit pointers.
  submission->signals.payload_values = (uint64_t*)cursor;
  cursor += signals.count * sizeof(uint64_t);
  iree_hal_buffer_ref_t* references = (iree_hal_buffer_ref_t*)cursor;
  submission->bindings =
      (iree_hal_buffer_ref_list_t){bindings.count, references};
  cursor += bindings.count * (sizeof(*references) + sizeof(uint64_t));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    references[i] = bindings.values[i];
    iree_hal_buffer_retain(references[i].buffer);
  }
  submission->signals.semaphores = (iree_hal_semaphore_t**)cursor;
  cursor += signals.count * sizeof(iree_hal_semaphore_t*);
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    submission->signals.semaphores[i] = signals.semaphores[i];
    submission->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  submission->constants =
      iree_make_const_byte_span(cursor, constants.data_length);
  if (constants.data_length) {
    memcpy(cursor, constants.data, constants.data_length);
  }
  *out_submission = submission;
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_submission_retain(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission) {
    iree_atomic_ref_count_inc(&submission->ref_count);
  }
}

static void iree_hal_amd_gpu_pm4_submission_destroy(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_arena_block_pool_t* pool = submission->pool;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    iree_hal_amd_gpu_pm4_signal_release(submission->waits[i].signal);
    iree_hal_semaphore_release(submission->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    iree_hal_semaphore_release(submission->signals.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < submission->bindings.count; ++i) {
    iree_hal_buffer_release(submission->bindings.values[i].buffer);
  }
  iree_hal_amd_gpu_pm4_signal_release(submission->signal);
  iree_hal_executable_release(submission->executable);
  iree_arena_block_t* block = submission->block;
  if (block) {
    iree_arena_block_pool_release(pool, block, block);
  } else {
    iree_allocator_free(pool->block_allocator, submission);
  }
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
  const iree_hal_buffer_memory_view_t view =
      iree_hal_buffer_memory_view(reference.buffer);
  const iree_hal_memory_contract_t* contract = view.contract;
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
      iree_hal_buffer_memory_native_binding(
          &view, (iree_hal_buffer_native_binding_slot_t){.index = index,
                                                         .type = interface})
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
  const iree_hal_amd_gpu_pm4_dispatch_t* dispatch = &submission->dispatch;
  iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .dynamic_lds_size =
          dispatch->group_segment_byte_length -
          submission->function->metadata->group_segment_fixed_size,
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
  if (iree_any_bit_set(submission->flags,
                       IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    iree_hal_amd_gpu_kernarg_layout_emplace_native(
        submission->function->kernarg, submission->constants, &inputs, target);
  } else {
    iree_hal_amd_gpu_kernarg_layout_emplace(submission->function->kernarg,
                                            submission->constants.data,
                                            binding_addresses, &inputs, target);
  }
}
