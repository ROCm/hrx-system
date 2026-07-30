// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/agent_target.h"

// Bounds HSA ISA enumeration before target-record allocation.
#define IREE_HAL_AMDGPU_MAX_AGENT_ISA_COUNT 32

// Public HSA AMD agent attribute used to distinguish ASIC revisions. The local
// name keeps this driver buildable against older HSA SDK headers while using
// the stable numeric ABI accepted by hsa_agent_get_info.
enum {
  IREE_HAL_AMDGPU_AGENT_INFO_ASIC_REVISION = 0xA012,
};

typedef struct iree_hal_amdgpu_agent_isa_list_t {
  // Number of valid entries in |values|.
  iree_host_size_t count;
  // True when HSA reported more entries than |values| can hold.
  bool overflowed;
  // HSA ISA handles in reported priority order.
  hsa_isa_t values[IREE_HAL_AMDGPU_MAX_AGENT_ISA_COUNT];
} iree_hal_amdgpu_agent_isa_list_t;

static hsa_status_t iree_hal_amdgpu_append_agent_isa(hsa_isa_t isa,
                                                     void* user_data) {
  iree_hal_amdgpu_agent_isa_list_t* isa_list =
      (iree_hal_amdgpu_agent_isa_list_t*)user_data;
  if (isa_list->count >= IREE_ARRAYSIZE(isa_list->values)) {
    isa_list->overflowed = true;
    return HSA_STATUS_SUCCESS;
  }
  isa_list->values[isa_list->count++] = isa;
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_agent_isa_target_initialize(
    iree_hal_amdgpu_agent_isa_value_t value,
    iree_hal_amdgpu_agent_isa_target_t* out_target) {
  if (iree_string_view_is_empty(value.name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSA GPU agent ISA name is empty");
  }
  if (value.name.size >= sizeof(out_target->isa_name_storage)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSA GPU agent ISA name length %" PRIhsz
                            " exceeds target record capacity %" PRIhsz,
                            value.name.size,
                            sizeof(out_target->isa_name_storage) - 1);
  }

  out_target->isa = value.isa;
  memcpy(out_target->isa_name_storage, value.name.data, value.name.size);
  out_target->isa_name_storage[value.name.size] = 0;
  out_target->isa_name =
      iree_make_string_view(out_target->isa_name_storage, value.name.size);
  return iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
      out_target->isa_name, &out_target->identity);
}

void iree_hal_amdgpu_agent_target_deinitialize(
    iree_hal_amdgpu_agent_target_t* target) {
  if (!target) return;
  iree_allocator_free(target->host_allocator, target->additional_isas);
  memset(target, 0, sizeof(*target));
}

const iree_hal_amdgpu_agent_isa_target_t*
iree_hal_amdgpu_agent_target_find_compatible_isa(
    const iree_hal_amdgpu_agent_target_t* target,
    const iree_hal_amdgpu_target_identity_t* required_identity) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(required_identity);
  for (iree_host_size_t i = 0; i < target->isa_count; ++i) {
    const iree_hal_amdgpu_agent_isa_target_t* isa_target =
        iree_hal_amdgpu_agent_target_isa_at(target, i);
    if (iree_hal_amdgpu_target_identity_check_compatible(
            required_identity, &isa_target->identity) ==
        IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      return isa_target;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_amdgpu_agent_target_initialize_identity(
    hsa_agent_t agent, iree_host_size_t isa_count,
    const iree_hal_amdgpu_agent_isa_value_t* isa_values,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_target) {
  memset(out_target, 0, sizeof(*out_target));
  if (isa_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "HSA GPU agent has no reported ISA");
  }
  if (!isa_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSA GPU agent ISA values are required");
  }

  out_target->agent = agent;
  out_target->isa_count = isa_count;
  out_target->host_allocator = host_allocator;
  iree_status_t status = iree_ok_status();
  if (isa_count > 1) {
    status = iree_allocator_malloc_array(host_allocator, isa_count - 1,
                                         sizeof(out_target->additional_isas[0]),
                                         (void**)&out_target->additional_isas);
  }
  for (iree_host_size_t i = 0; i < isa_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_agent_isa_target_t* isa_target =
        i == 0 ? &out_target->primary_isa : &out_target->additional_isas[i - 1];
    status =
        iree_hal_amdgpu_agent_isa_target_initialize(isa_values[i], isa_target);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_agent_target_deinitialize(out_target);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_agent_target_resolve_physical_targets(
    uint32_t asic_revision, iree_hal_amdgpu_agent_target_t* target) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < target->isa_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_agent_isa_target_t* isa_target =
        i == 0 ? &target->primary_isa : &target->additional_isas[i - 1];
    status = iree_hal_amdgpu_target_identity_resolve_physical_target(
        asic_revision, &isa_target->identity);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_agent_target_initialize(
    hsa_agent_t agent, iree_host_size_t isa_count,
    const iree_hal_amdgpu_agent_isa_value_t* isa_values, uint32_t asic_revision,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_agent_target_initialize_identity(
      agent, isa_count, isa_values, host_allocator, out_target));
  iree_status_t status = iree_hal_amdgpu_agent_target_resolve_physical_targets(
      asic_revision, out_target);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_agent_target_deinitialize(out_target);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_agent_target_query(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_target);

  iree_hal_amdgpu_agent_isa_list_t isa_list = {0};
  IREE_RETURN_IF_ERROR(iree_hsa_agent_iterate_isas(
      IREE_LIBHSA(libhsa), agent, iree_hal_amdgpu_append_agent_isa, &isa_list));
  if (isa_list.overflowed) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSA GPU agent reports more than %d ISAs",
                            IREE_HAL_AMDGPU_MAX_AGENT_ISA_COUNT);
  }
  if (isa_list.count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "HSA GPU agent has no reported ISA");
  }

  char isa_name_storage[IREE_HAL_AMDGPU_MAX_AGENT_ISA_COUNT][128];
  iree_hal_amdgpu_agent_isa_value_t
      isa_values[IREE_HAL_AMDGPU_MAX_AGENT_ISA_COUNT];
  for (iree_host_size_t i = 0; i < isa_list.count; ++i) {
    uint32_t isa_name_length = 0;
    IREE_RETURN_IF_ERROR(
        iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_list.values[i],
                                  HSA_ISA_INFO_NAME_LENGTH, &isa_name_length));
    if (isa_name_length == 0 || isa_name_length > sizeof(isa_name_storage[i])) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HSA GPU agent ISA[%" PRIhsz
          "] name length %u exceeds target record capacity %" PRIhsz,
          i, isa_name_length, sizeof(isa_name_storage[i]) - 1);
    }
    IREE_RETURN_IF_ERROR(
        iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_list.values[i],
                                  HSA_ISA_INFO_NAME, isa_name_storage[i]));
    isa_values[i] = (iree_hal_amdgpu_agent_isa_value_t){
        .isa = isa_list.values[i],
        .name = iree_make_string_view(isa_name_storage[i],
                                      isa_name_length - /*NUL*/ 1),
    };
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_agent_target_initialize_identity(
      agent, isa_list.count, isa_values, host_allocator, out_target));
  bool requires_physical_resolution = false;
  for (iree_host_size_t i = 0; i < out_target->isa_count; ++i) {
    const iree_hal_amdgpu_agent_isa_target_t* isa_target =
        iree_hal_amdgpu_agent_target_isa_at(out_target, i);
    if (iree_hal_amdgpu_target_identity_requires_physical_resolution(
            &isa_target->identity)) {
      requires_physical_resolution = true;
      break;
    }
  }

  iree_status_t status = iree_ok_status();
  uint32_t asic_revision = 0;
  if (requires_physical_resolution) {
    status = iree_hsa_agent_get_info(
        IREE_LIBHSA(libhsa), agent,
        (hsa_agent_info_t)IREE_HAL_AMDGPU_AGENT_INFO_ASIC_REVISION,
        &asic_revision);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_agent_target_resolve_physical_targets(
        asic_revision, out_target);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_agent_target_deinitialize(out_target);
  }
  return status;
}
