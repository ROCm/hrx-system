// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/hostcall_provider.h"

#include <stdio.h>

#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"

static int iree_hal_amdgpu_hostcall_provider_service_thread_main(
    void* entry_arg) {
  iree_hal_amdgpu_hostcall_provider_state_t* state =
      (iree_hal_amdgpu_hostcall_provider_state_t*)entry_arg;
  hsa_signal_value_t observed_value = 0;

  while (true) {
    observed_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(state->libhsa), state->notification_signal,
        HSA_SIGNAL_CONDITION_NE, observed_value, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    state->provider.service(state->provider_context);
    if (iree_atomic_load(&state->stop_requested, iree_memory_order_acquire)) {
      break;
    }
  }
  return 0;
}

iree_status_t iree_hal_amdgpu_hostcall_provider_state_create(
    const iree_hal_hostcall_provider_t* provider,
    iree_hal_device_t* logical_device, const iree_hal_amdgpu_libhsa_t* libhsa,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t shared_memory_pool,
    iree_host_size_t host_numa_node,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_hostcall_provider_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(device_info);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  iree_hal_hostcall_provider_requirements_t requirements = {0};
  IREE_RETURN_IF_ERROR(provider->query_requirements(
      provider->user_data, device_info, &requirements));
  if (IREE_UNLIKELY(
          requirements.allocation_size == 0 ||
          requirements.allocation_alignment == 0 ||
          !iree_host_size_is_power_of_two(requirements.allocation_alignment) ||
          requirements.reserved != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU hostcall provider returned invalid allocation requirements: "
        "size=%" PRIhsz ", alignment=%" PRIhsz,
        requirements.allocation_size, requirements.allocation_alignment);
  }
  if (IREE_UNLIKELY(requirements.notification_type !=
                    IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU cannot provide hostcall notification type %u",
        requirements.notification_type);
  }

  iree_hal_amdgpu_hostcall_provider_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->libhsa = libhsa;
  state->host_allocator = host_allocator;
  state->provider = *provider;
  state->shared_memory_size = requirements.allocation_size;
  iree_atomic_store(&state->stop_requested, 0, iree_memory_order_relaxed);

  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), shared_memory_pool, requirements.allocation_size,
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &state->shared_memory);
  if (iree_status_is_ok(status) &&
      ((uintptr_t)state->shared_memory &
       (requirements.allocation_alignment - 1)) != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HSA shared allocation does not meet hostcall provider alignment "
        "requirement: pointer=%p, alignment=%" PRIhsz,
        state->shared_memory, requirements.allocation_alignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(libhsa), /*num_agents=*/1, &device_agent,
        /*flags=*/NULL, state->shared_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &state->notification_signal);
  }
  if (iree_status_is_ok(status)) {
    state->device_address = (uint64_t)(uintptr_t)state->shared_memory;
    const iree_hal_hostcall_error_callback_t error_callback = {
        .fn = iree_hal_amdgpu_logical_device_error_handler,
        .user_data = logical_device,
    };
    const iree_hal_hostcall_notification_t notification = {
        .type = IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL,
        .token = state->notification_signal.handle,
    };
    status = provider->initialize(
        provider->user_data, device_info,
        iree_make_byte_span(state->shared_memory, state->shared_memory_size),
        state->device_address, notification, error_callback,
        &state->provider_context);
  }
  if (iree_status_is_ok(status) && !state->provider_context) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU hostcall provider initialization returned no context");
  }
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    char thread_name[32] = {0};
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "amdgpu-d%u-hostcall",
             (unsigned)device_info->physical_device_ordinal);
    thread_params.name = iree_make_cstring_view(thread_name);
    iree_thread_affinity_set_group_any(host_numa_node,
                                       &thread_params.initial_affinity);
    status = iree_thread_create(
        iree_hal_amdgpu_hostcall_provider_service_thread_main, state,
        thread_params, host_allocator, &state->service_thread);
  }

  if (iree_status_is_ok(status)) {
    *out_state = state;
  } else {
    iree_hal_amdgpu_hostcall_provider_state_destroy(state);
  }
  return status;
}

void iree_hal_amdgpu_hostcall_provider_state_destroy(
    iree_hal_amdgpu_hostcall_provider_state_t* state) {
  if (!state) return;

  if (state->service_thread) {
    iree_atomic_store(&state->stop_requested, 1, iree_memory_order_release);
    iree_hsa_signal_add_screlease(IREE_LIBHSA(state->libhsa),
                                  state->notification_signal, 1);
    iree_thread_release(state->service_thread);
  }
  if (state->provider_context) {
    state->provider.deinitialize(state->provider_context);
  }
  if (state->notification_signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_signal_destroy_raw(state->libhsa, state->notification_signal));
  }
  if (state->shared_memory) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(state->libhsa, state->shared_memory));
  }
  iree_allocator_free(state->host_allocator, state);
}
