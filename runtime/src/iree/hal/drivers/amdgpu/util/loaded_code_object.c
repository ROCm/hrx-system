// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/loaded_code_object.h"

#include <string.h>

typedef struct iree_hal_amdgpu_loaded_code_object_find_state_t {
  // Borrowed HSA API table used for loader extension queries.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // HSA device agent whose loaded code object is being searched.
  hsa_agent_t device_agent;
  // Loaded code object matching |device_agent| when found.
  hsa_loaded_code_object_t loaded_code_object;
} iree_hal_amdgpu_loaded_code_object_find_state_t;

iree_status_t iree_hal_amdgpu_loaded_code_object_query_host_address(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint64_t device_address,
    const void** out_host_pointer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_host_pointer);
  *out_host_pointer = NULL;
  if (IREE_UNLIKELY(device_address == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loaded code-object device address is required");
  }

  hsa_status_t hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_query_host_address(
          (void*)(uintptr_t)device_address, out_host_pointer);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status, "hsa_ven_amd_loader_query_host_address",
        "translating loaded code-object device address");
  }
  if (IREE_UNLIKELY(!*out_host_pointer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loaded code-object host address translation returned NULL");
  }
  return iree_ok_status();
}

static hsa_status_t iree_hal_amdgpu_loaded_code_object_iterate(
    hsa_executable_t executable, hsa_loaded_code_object_t loaded_code_object,
    void* user_data) {
  (void)executable;
  iree_hal_amdgpu_loaded_code_object_find_state_t* find_state =
      (iree_hal_amdgpu_loaded_code_object_find_state_t*)user_data;
  hsa_agent_t device_agent = {0};
  hsa_status_t hsa_status =
      find_state->libhsa->amd_loader
          .hsa_ven_amd_loader_loaded_code_object_get_info(
              loaded_code_object,
              HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &device_agent);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (device_agent.handle == find_state->device_agent.handle) {
    find_state->loaded_code_object = loaded_code_object;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_amdgpu_loaded_code_object_find(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t device_agent,
    hsa_loaded_code_object_t* out_loaded_code_object) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_loaded_code_object);
  *out_loaded_code_object = (hsa_loaded_code_object_t){0};

  iree_hal_amdgpu_loaded_code_object_find_state_t find_state = {
      .libhsa = libhsa,
      .device_agent = device_agent,
      .loaded_code_object = {0},
  };
  hsa_status_t hsa_status =
      libhsa->amd_loader
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              executable, iree_hal_amdgpu_loaded_code_object_iterate,
              &find_state);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no loaded code object found for agent");
  }
  if (hsa_status != HSA_STATUS_INFO_BREAK) {
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
        "iterating loaded executable code objects");
  }
  *out_loaded_code_object = find_state.loaded_code_object;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_loaded_code_object_query_range(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    hsa_loaded_code_object_t loaded_code_object,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_range);
  memset(out_range, 0, sizeof(*out_range));

  iree_hal_amdgpu_loaded_code_object_range_t range = {0};
  hsa_status_t hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
          &range.device_pointer);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_loaded_code_object_get_info",
        "querying loaded executable code-object load base");
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_loaded_code_object_query_host_address(
      libhsa, range.device_pointer, (const void**)&range.host_pointer));

  hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
          &range.byte_length);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_loaded_code_object_get_info",
        "querying loaded executable code-object load size");
  }

  if (IREE_UNLIKELY(!range.host_pointer || range.device_pointer == 0 ||
                    range.byte_length == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loaded executable code-object range is empty or unmapped");
  }

  *out_range = range;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_loaded_code_object_query_agent_range(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(out_range);
  memset(out_range, 0, sizeof(*out_range));

  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_loaded_code_object_find(
      libhsa, executable, device_agent, &loaded_code_object));
  return iree_hal_amdgpu_loaded_code_object_query_range(
      libhsa, loaded_code_object, out_range);
}
