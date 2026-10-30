// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/command_buffer.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

// Conditionally executes an expression based on whether command buffer
// validation was enabled in the build and the command buffer wants validation.
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
#define IF_VALIDATING(command_buffer, expr)                                  \
  if (((command_buffer)->mode & IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED) == \
      0) {                                                                   \
    expr;                                                                    \
  }
#define VALIDATION_STATE(command_buffer)                          \
  ((iree_hal_command_buffer_validation_state_t*)((command_buffer) \
                                                     ->validation_state))
#else
#define IF_VALIDATING(command_buffer, expr)
#define VALIDATION_STATE(command_buffer) \
  ((iree_hal_command_buffer_validation_state_t*)NULL)
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE

#define _VTABLE_DISPATCH(command_buffer, method_name) \
  IREE_HAL_VTABLE_DISPATCH(command_buffer, iree_hal_command_buffer, method_name)

static iree_atomic_int64_t iree_hal_command_buffer_next_profile_id =
    IREE_ATOMIC_VAR_INIT(1);

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t
iree_hal_command_buffer_mode_format(iree_hal_command_buffer_mode_t value,
                                    iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      {IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_SVL("ONE_SHOT")},
      {IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED, IREE_SVL("UNVALIDATED")},
      {IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED, IREE_SVL("UNRETAINED")},
      {IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
       IREE_SVL("RETAIN_PROFILE_METADATA")},
      {IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA,
       IREE_SVL("RETAIN_DISPATCH_METADATA")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

IREE_API_EXPORT iree_string_view_t iree_hal_command_category_format(
    iree_hal_command_category_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      // Combined:
      {IREE_HAL_COMMAND_CATEGORY_ANY, IREE_SVL("ANY")},
      // Separate:
      {IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_SVL("TRANSFER")},
      {IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_SVL("DISPATCH")},
      {IREE_HAL_COMMAND_CATEGORY_ATOMIC, IREE_SVL("ATOMIC")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(command_buffer);

IREE_API_EXPORT iree_host_size_t iree_hal_command_buffer_validation_state_size(
    iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity) {
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  return ((mode & IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED) == 0)
             ? sizeof(iree_hal_command_buffer_validation_state_t) +
                   binding_capacity *
                       sizeof(iree_hal_buffer_binding_requirements_t)
             : 0;
#else
  return 0;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
}

IREE_API_EXPORT void iree_hal_command_buffer_initialize(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, void* validation_state,
    const iree_hal_command_buffer_vtable_t* vtable,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(queue_family);
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  // If validation is compiled in and the command buffer requires validation
  // then check that state was provided.
  IREE_ASSERT(
      iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED) ||
      validation_state);
#else
  // If validation is not compiled in then force the disable bit. This helps
  // prevent issues with dynamic libraries that may be compiled with a different
  // setting, but we don't really support that kind of shady use anyway.
  mode &= ~IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
#endif  // !IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE

  iree_hal_resource_initialize(vtable, &command_buffer->resource);
  command_buffer->queue_family = queue_family;
  command_buffer->mode = mode;
  command_buffer->allowed_categories = command_categories;
  command_buffer->profile_id = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_command_buffer_next_profile_id, 1, iree_memory_order_relaxed);
  command_buffer->binding_capacity = binding_capacity;
  command_buffer->binding_count = 0;
  command_buffer->validation_state = validation_state;

  // Perform initialization validation after we allocate/initialize the concrete
  // implementation.
  IF_VALIDATING(command_buffer, {
    iree_hal_command_buffer_initialize_validation(
        command_buffer, VALIDATION_STATE(command_buffer));
  });
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_create(const iree_hal_queue_family_t* queue_family,
                               iree_hal_command_buffer_mode_t mode,
                               iree_hal_command_category_t command_categories,
                               iree_host_size_t binding_capacity,
                               iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  if (IREE_UNLIKELY(command_categories & ~IREE_HAL_COMMAND_CATEGORY_ANY)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown command categories: 0x%08" PRIx32,
                            command_categories);
  }

  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  iree_hal_queue_family_role_flags_t required_roles =
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_NONE;
  if (iree_any_bit_set(command_categories,
                       IREE_HAL_COMMAND_CATEGORY_TRANSFER)) {
    required_roles |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
  }
  if (iree_any_bit_set(command_categories,
                       IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
    required_roles |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
  }
  if (iree_any_bit_set(command_categories, IREE_HAL_COMMAND_CATEGORY_ATOMIC)) {
    required_roles |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(family_spec->role_flags, required_roles))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue family %u roles 0x%08" PRIx32
                            " do not cover command categories 0x%08" PRIx32,
                            iree_hal_queue_family_ordinal(queue_family),
                            family_spec->role_flags, command_categories);
  }

  iree_hal_device_t* device = iree_hal_queue_family_device(queue_family);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_command_buffer)(
          device, queue_family, mode, command_categories, binding_capacity,
          &command_buffer);
  if (iree_status_is_ok(status) && IREE_UNLIKELY(!command_buffer)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "device returned success without creating a command buffer");
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_command_buffer_mode_t
iree_hal_command_buffer_mode(const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->mode;
}

IREE_API_EXPORT iree_hal_command_category_t
iree_hal_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->allowed_categories;
}

IREE_API_EXPORT const iree_hal_queue_family_t*
iree_hal_command_buffer_queue_family(
    const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->queue_family;
}

IREE_API_EXPORT uint64_t iree_hal_command_buffer_profile_id(
    const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->profile_id;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_begin_validation(
                command_buffer, VALIDATION_STATE(command_buffer)));
  });
  iree_status_t status =
      _VTABLE_DISPATCH(command_buffer, begin)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_end_validation(
                command_buffer, VALIDATION_STATE(command_buffer)));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, end)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin_debug_group_validation(
        command_buffer, VALIDATION_STATE(command_buffer), label, label_color,
        location));
  });
  return _VTABLE_DISPATCH(command_buffer, begin_debug_group)(
      command_buffer, label, label_color, location);
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end_debug_group_validation(
        command_buffer, VALIDATION_STATE(command_buffer)));
  });
  return _VTABLE_DISPATCH(command_buffer, end_debug_group)(command_buffer);
}

IREE_API_EXPORT iree_status_t
iree_hal_barrier_validate(const iree_hal_barrier_t* barrier,
                          const iree_hal_queue_family_t* queue_family) {
  if (IREE_UNLIKELY(
          !barrier ||
          (barrier->memory_barrier_count && !barrier->memory_barriers) ||
          (barrier->buffer_barrier_count && !barrier->buffer_barriers))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "barrier descriptor storage is null");
  }
  if (IREE_UNLIKELY(!iree_hal_memory_effects_is_supported(barrier->effects))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "memory transition is not qualified");
  }
  const bool supports_host =
      iree_any_bit_set(queue_family->spec->flags,
                       IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_HOST_MEMORY_BARRIERS);
  const uint32_t queue_effects =
      IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
      IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
      IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM |
      IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM |
      (supports_host ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH |
                           IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE
                     : 0);
  if (IREE_UNLIKELY(barrier->effects.bits & ~queue_effects)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "barrier effects require an executor unavailable on this queue family");
  }
  const iree_hal_barrier_flags_t supported_flags =
      IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
      IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  if (IREE_UNLIKELY(barrier->flags & ~supported_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported barrier flags: 0x%016" PRIx64,
                            barrier->flags & ~supported_flags);
  }
  const iree_hal_execution_stage_t supported_stages =
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE | IREE_HAL_EXECUTION_STAGE_HOST |
      IREE_HAL_EXECUTION_STAGE_ATOMIC;
  if (IREE_UNLIKELY((barrier->source_stage_mask | barrier->target_stage_mask) &
                    ~supported_stages)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported barrier execution stage");
  }
  const iree_hal_access_scope_t supported_scopes =
      IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE | IREE_HAL_ACCESS_SCOPE_HOST_READ |
      IREE_HAL_ACCESS_SCOPE_HOST_WRITE | IREE_HAL_ACCESS_SCOPE_MEMORY_READ |
      IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE | IREE_HAL_ACCESS_SCOPE_ATOMIC_READ |
      IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE;
  iree_hal_access_scope_t scopes = 0;
  uint32_t recipe_effects = 0;
  for (iree_host_size_t i = 0; i < barrier->memory_barrier_count; ++i) {
    scopes |= barrier->memory_barriers[i].source_scope |
              barrier->memory_barriers[i].target_scope;
  }
  for (iree_host_size_t i = 0; i < barrier->buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t* buffer_barrier =
        &barrier->buffer_barriers[i];
    scopes |= buffer_barrier->source_scope | buffer_barrier->target_scope;
    if (!buffer_barrier->recipe) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_memory_transition_recipe_validate(buffer_barrier->recipe));
    for (uint32_t j = 0; j < buffer_barrier->recipe->operation_count; ++j) {
      const iree_hal_memory_transition_executor_t executor =
          buffer_barrier->recipe->operations[j].executor;
      if (IREE_UNLIKELY(
              executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE &&
              !(supports_host &&
                (executor == IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT ||
                 executor == IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API)))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer barrier recipe executor is unavailable "
                                "on this queue family");
      }
    }
    recipe_effects |= buffer_barrier->recipe->effects.bits;
  }
  if (IREE_UNLIKELY(scopes & ~supported_scopes)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported barrier access scope");
  }
  const uint32_t required_recipe_effects =
      barrier->effects.bits & IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK;
  if (IREE_UNLIKELY(recipe_effects != required_recipe_effects)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "buffer barrier recipes do not cover the ranged effects");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_barrier(iree_hal_command_buffer_t* command_buffer,
                                const iree_hal_barrier_t* barrier) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_barrier_validate(barrier, command_buffer->queue_family));
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_barrier_validation(
                command_buffer, VALIDATION_STATE(command_buffer), barrier));
  });

  // Global prepared actions and explicit minimum flags have identical native
  // semantics. Resolve those once while preserving ranged effects and their
  // copied buffer recipes for the native recorder.
  iree_hal_barrier_t resolved = *barrier;
  resolved.flags = iree_hal_barrier_resolve_flags(barrier);
  resolved.effects.bits &= IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK;
  iree_status_t status = iree_ok_status();
  if (resolved.source_stage_mask || resolved.target_stage_mask ||
      resolved.flags || resolved.memory_barrier_count ||
      resolved.buffer_barrier_count) {
    status =
        _VTABLE_DISPATCH(command_buffer, barrier)(command_buffer, &resolved);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_timestamp(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_timestamp_flags_t flags) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (IREE_UNLIKELY(flags != IREE_HAL_TIMESTAMP_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp flags: 0x%016" PRIx64,
                            flags);
  }
  const iree_hal_execution_stage_t supported_stages =
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE | IREE_HAL_EXECUTION_STAGE_ATOMIC;
  if (IREE_UNLIKELY((source_stage_mask | target_stage_mask) &
                    ~supported_stages)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp execution stage");
  }
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(command_buffer->queue_family);
  const iree_hal_queue_family_spec_flags_t required_capability =
      target_ref.buffer
          ? IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS
          : IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS;
  if (IREE_UNLIKELY(
          !iree_all_bits_set(family_spec->flags, required_capability))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue family does not support %s command timestamps",
        target_ref.buffer ? "direct" : "indirect");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_timestamp_validation(
                command_buffer, VALIDATION_STATE(command_buffer), target_ref));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, timestamp)(
      command_buffer, source_stage_mask, target_stage_mask, target_ref, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_atomic_wait_validation(
                command_buffer, VALIDATION_STATE(command_buffer),
                source_stage_mask, target_stage_mask, target_ref, params));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, atomic_wait)(
      command_buffer, source_stage_mask, target_stage_mask, target_ref, params);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_atomic_store(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_atomic_store_validation(
                command_buffer, VALIDATION_STATE(command_buffer),
                source_stage_mask, target_stage_mask, target_ref, params));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, atomic_store)(
      command_buffer, source_stage_mask, target_stage_mask, target_ref, params);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_atomic_rmw_validation(
                command_buffer, VALIDATION_STATE(command_buffer),
                source_stage_mask, target_stage_mask, target_ref, params));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, atomic_rmw)(
      command_buffer, source_stage_mask, target_stage_mask, target_ref, params);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_advise_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), buffer_ref,
                flags, arg0, arg1));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, advise_buffer)(
      command_buffer, buffer_ref, flags, arg0, arg1);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t target_ref,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (target_ref.length == 0) {
    // No-op fill. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_fill_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), target_ref,
                pattern, pattern_length, flags));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, fill_buffer)(
      command_buffer, target_ref, pattern, pattern_length, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_update_buffer(
    iree_hal_command_buffer_t* command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  if (target_ref.length == 0) {
    // No-op update. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_update_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), source_buffer,
                source_offset, target_ref, flags));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, update_buffer)(
      command_buffer, source_buffer, source_offset, target_ref, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref, iree_hal_copy_flags_t flags) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (target_ref.length == 0) {
    // No-op copy. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_copy_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), source_ref,
                target_ref, flags));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, copy_buffer)(
      command_buffer, source_ref, target_ref, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);

  const bool has_static_workgroup_count =
      !iree_hal_dispatch_uses_indirect_parameters(flags);
  if (has_static_workgroup_count &&
      (config.workgroup_count[0] | config.workgroup_count[1] |
       config.workgroup_count[2]) == 0) {
    // No-op dispatch. All implementations are expected to do this but we ensure
    // it happens here to avoid the overhead of going all the way down into the
    // device layer for something we know should have no (intentional)
    // side-effects. Note that this does mean that validation is skipped and
    // the executable/etc could be bogus but that's fine.
    return iree_ok_status();
  }

  if (IREE_UNLIKELY(iree_hal_executable_queue_family(executable) !=
                    command_buffer->queue_family)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable queue family %u does not match command buffer queue "
        "family %u",
        iree_hal_queue_family_ordinal(
            iree_hal_executable_queue_family(executable)),
        iree_hal_queue_family_ordinal(command_buffer->queue_family));
  }
  if (IREE_UNLIKELY(
          iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_COOPERATIVE) &&
          !iree_any_bit_set(
              iree_hal_queue_family_spec(command_buffer->queue_family)
                  ->supported_queue_features,
              IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cooperative dispatch requires a cooperative-capable queue family");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
#if IREE_HAL_VERBOSE_TRACING_ENABLE
  // TODO(benvanik): add a tracing.h helper that does the snprintf directly
  // into a tracy_malloc buffer so that we can avoid the memcpy. Today this can
  // take 4-5us which adds too much overhead when trying to get accurate timings
  // with tracing enabled. Because benchmarks shouldn't be run with asserts
  // enabled we only enable these when assertions are enabled. Ideally we'd
  // slice off a much larger allocation and then suballocate from that ourselves
  // so that we could avoid the tracy_malloc overheads per-dispatch.
  IREE_TRACE({
    if (has_static_workgroup_count) {
      char xyz_string[32];
      int xyz_string_length =
          iree_snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
                        config.workgroup_count[0], config.workgroup_count[1],
                        config.workgroup_count[2]);
      IREE_TRACE_ZONE_APPEND_TEXT(z0, xyz_string, xyz_string_length);
    } else {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(indirect)");
    }
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE

  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_dispatch_validation(
                command_buffer, VALIDATION_STATE(command_buffer), executable,
                function, config, constants, bindings, flags));
  });

  iree_status_t status = _VTABLE_DISPATCH(command_buffer, dispatch)(
      command_buffer, executable, function, config, constants, bindings, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Validation support
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_validate_submission(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_ASSERT_ARGUMENT(command_buffer);

  // Validate the command buffer has been recorded properly.
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_submission_validation(
        command_buffer, VALIDATION_STATE(command_buffer)));
  });

  // Only check binding tables when one is required and otherwise ignore any
  // bindings provided. Require at least as many bindings in the table as there
  // are used by the command buffer. This may be less than the total capacity
  // the command buffer was allocated with.
  if (command_buffer->binding_count == 0) {
    return iree_ok_status();
  } else if (binding_table.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer requires at least %u "
                            "bindings but no binding table was provided",
                            command_buffer->binding_count);
  } else if (binding_table.count < command_buffer->binding_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "indirect command buffer requires at least %u "
                            "bindings but only %" PRIhsz " were provided ",
                            command_buffer->binding_count, binding_table.count);
  } else if (!binding_table.bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding table storage is null");
  }

  // Validate the binding table against the commands consuming them.
  // This is O(binding_count) so something we only do if validation is
  // requested on the command buffer.
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_binding_table_validation(
        command_buffer, VALIDATION_STATE(command_buffer), binding_table));
  });

  return iree_ok_status();
}
