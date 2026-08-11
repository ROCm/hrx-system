// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/asan_state.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/executable_global_resolver.h"
#include "iree/hal/drivers/amdgpu/executable_metadata_hsaco.h"
#include "iree/hal/drivers/amdgpu/feedback_state.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/source_context.h"
#include "iree/hal/drivers/amdgpu/tsan_state.h"
#include "iree/hal/drivers/amdgpu/util/global_table.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/loaded_code_object.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/executable/amdgpu/code_object_target.h"
#include "iree/hal/executable/amdgpu/target_id.h"

//===----------------------------------------------------------------------===//
// ISA Support
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_executable_resolve_isa_target(
    const iree_hal_amdgpu_agent_target_t* agent_target,
    iree_string_view_t target_key,
    const iree_hal_amdgpu_agent_isa_target_t** out_isa_target) {
  IREE_ASSERT_ARGUMENT(agent_target);
  IREE_ASSERT_ARGUMENT(out_isa_target);
  *out_isa_target = NULL;

  if (!iree_string_view_starts_with(target_key, IREE_SV("gfx"))) {
    return iree_ok_status();
  }

  iree_hal_amdgpu_target_identity_t requested_target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_artifact_key(
      target_key, &requested_target_id));

  *out_isa_target = iree_hal_amdgpu_agent_target_find_compatible_isa(
      agent_target, &requested_target_id);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Verification
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_limits_t {
  // Maximum total workgroup size from HSA_ISA_INFO_WORKGROUP_MAX_SIZE.
  uint32_t max_workgroup_size;
  // Maximum workgroup size per dimension from HSA_ISA_INFO_WORKGROUP_MAX_DIM.
  uint16_t max_workgroup_size_per_dim[3];
} iree_hal_amdgpu_device_limits_t;
static iree_status_t iree_hal_amdgpu_query_device_limits(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_isa_t isa, iree_hal_amdgpu_device_limits_t* out_limits) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                    HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                                    &out_limits->max_workgroup_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                    HSA_ISA_INFO_WORKGROUP_MAX_DIM,
                                    &out_limits->max_workgroup_size_per_dim));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Loading
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_physical_device_mask_contains(
    uint64_t physical_device_mask, iree_host_size_t physical_device_ordinal) {
  return physical_device_ordinal < IREE_HAL_MAX_QUEUES &&
         iree_all_bits_set(physical_device_mask,
                           ((uint64_t)1) << physical_device_ordinal);
}

static iree_status_t iree_hal_amdgpu_executable_select_physical_devices(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        out_physical_devices) {
  memset(out_physical_devices, 0, sizeof(*out_physical_devices));

  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain = {
      .supported_affinity = 0,
      .physical_device_count = topology->gpu_agent_count,
      .queue_count_per_physical_device = topology->gpu_agent_queue_count,
  };

  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count;
       ++physical_device_ordinal) {
    iree_hal_queue_affinity_t physical_device_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        queue_affinity_domain, physical_device_ordinal,
        &physical_device_affinity));
    iree_hal_queue_affinity_or_into(queue_affinity_domain.supported_affinity,
                                    physical_device_affinity);
  }

  return iree_hal_amdgpu_queue_affinity_select_physical_devices(
      queue_affinity_domain, requested_affinity, out_physical_devices);
}

static iree_status_t iree_hal_amdgpu_executable_format_target_id_for_message(
    const iree_hal_amdgpu_target_identity_t* target_id,
    iree_host_size_t buffer_capacity, char* buffer) {
  return iree_hal_amdgpu_target_identity_format_artifact_key(
      target_id, buffer_capacity, buffer,
      /*out_buffer_length=*/NULL);
}

static iree_status_t iree_hal_amdgpu_executable_preflight_agent_code_object(
    const iree_hal_amdgpu_physical_device_t* physical_device,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_target_identity_t* code_object_target_id) {
  const iree_hal_amdgpu_agent_target_t* agent_target =
      physical_device->agent_target;
  if (iree_hal_amdgpu_agent_target_find_compatible_isa(agent_target,
                                                       code_object_target_id)) {
    return iree_ok_status();
  }
  const iree_hal_amdgpu_target_compatibility_t primary_compatibility =
      iree_hal_amdgpu_target_identity_check_compatible(
          code_object_target_id, &agent_target->primary_isa.identity);

  char code_object_target[128] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_format_target_id_for_message(
      code_object_target_id, sizeof(code_object_target), code_object_target));
  char primary_agent_target[128] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_format_target_id_for_message(
      &agent_target->primary_isa.identity, sizeof(primary_agent_target),
      primary_agent_target));
  char mismatch_reasons[128] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_compatibility_format(
      primary_compatibility, sizeof(mismatch_reasons), mismatch_reasons,
      /*out_buffer_length=*/NULL));
  return iree_make_status(
      IREE_STATUS_INCOMPATIBLE,
      "AMDGPU code object target `%s` is not compatible with GPU agent[%" PRIhsz
      "] ISA targets (primary `%s` mismatched %s; %" PRIhsz " targets checked)",
      code_object_target, physical_device_ordinal, primary_agent_target,
      mismatch_reasons, agent_target->isa_count);
}

static iree_status_t iree_hal_amdgpu_executable_preflight_code_object(
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    iree_const_byte_span_t code_object_data,
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_device_list) {
  iree_hal_amdgpu_target_identity_t code_object_target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_code_object_target_id_from_elf(
      code_object_data, &code_object_target_id));
  for (iree_host_size_t i = 0; i < physical_device_count; ++i) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            physical_devices->physical_device_mask, i)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_preflight_agent_code_object(
        physical_device_list[i], i, &code_object_target_id));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_query_agent_profile(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_profile_t* out_profile,
    hsa_default_float_rounding_mode_t* out_rounding_mode) {
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent, HSA_AGENT_INFO_PROFILE, out_profile));
  return iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                                 HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                                 out_rounding_mode);
}

static iree_status_t iree_hal_amdgpu_executable_select_agent_profile(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    hsa_profile_t* out_profile,
    hsa_default_float_rounding_mode_t* out_rounding_mode) {
  bool has_selected_agent = false;
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            physical_devices->physical_device_mask, i)) {
      continue;
    }

    hsa_profile_t profile = HSA_PROFILE_BASE;
    hsa_default_float_rounding_mode_t rounding_mode =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_query_agent_profile(
        libhsa, topology->gpu_agents[i], &profile, &rounding_mode));
    if (!has_selected_agent) {
      *out_profile = profile;
      *out_rounding_mode = rounding_mode;
      has_selected_agent = true;
      continue;
    }
    if (profile != *out_profile || rounding_mode != *out_rounding_mode) {
      return iree_make_status(
          IREE_STATUS_INCOMPATIBLE,
          "selected GPU agents must have matching HSA executable profiles and "
          "default rounding modes; agent[%" PRIhsz
          "] has profile %d and rounding mode %d but the executable requires "
          "profile %d and rounding mode %d",
          i, (int)profile, (int)rounding_mode, (int)*out_profile,
          (int)*out_rounding_mode);
    }
  }
  if (!has_selected_agent) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no GPU agents selected for executable load");
  }
  return iree_ok_status();
}

// Loads an executable ELF from |code_object_reader| for selected agents in
// |topology| and stores the frozen executable in |out_handle|.
static iree_status_t iree_hal_amdgpu_executable_load_module(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_load_params_t* load_params,
    hsa_code_object_reader_t code_object_reader, hsa_executable_t* out_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_handle = (hsa_executable_t){0};

  // TODO(#18877): support executable constants in HSA executables.
  // We currently don't support executable constants but we could by way of
  // global symbols. We should be using externs for the constants and then
  // hsa_executable_readonly_variable_define to specify each. We have to
  // allocate one copy of the constant table per agent. I don't know if it's
  // best to have one base symbol pointing to the table or one symbol per
  // constant in the table.
  if (load_params->constant_count != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "executable constants not yet implemented"));
  }

  // TODO(benvanik): figure out what options we could pass? Documentation is ...
  // lacking. These may have only been used for HSAIL anyway.
  const char* options = NULL;

  // Create the executable that will hold all of the loaded code objects.
  hsa_profile_t executable_profile = HSA_PROFILE_BASE;
  hsa_default_float_rounding_mode_t executable_rounding_mode =
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
  iree_status_t status = iree_hal_amdgpu_executable_select_agent_profile(
      libhsa, topology, physical_devices, &executable_profile,
      &executable_rounding_mode);
  hsa_executable_t handle = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_create_alt(
        IREE_LIBHSA(libhsa), executable_profile, executable_rounding_mode,
        options, &handle);
  }

  // Load the code object for each selected agent.
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < topology->gpu_agent_count; ++i) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            physical_devices->physical_device_mask, i)) {
      continue;
    }
    status = iree_hsa_executable_load_agent_code_object(
        IREE_LIBHSA(libhsa), handle, topology->gpu_agents[i],
        code_object_reader, options, NULL);
  }

  // Freeze the executable now that loading has completed. Most queries require
  // that the executable be frozen.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_freeze(IREE_LIBHSA(libhsa), handle, options);
  }

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else if (handle.handle) {
    status = iree_status_join(
        status, iree_hsa_executable_destroy(IREE_LIBHSA(libhsa), handle));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_amdgpu_executable_populate_profile_code_object_load_info(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    uint32_t physical_device_ordinal, hsa_agent_t device_agent,
    iree_hal_amdgpu_profile_code_object_load_info_t* out_load_info) {
  memset(out_load_info, 0, sizeof(*out_load_info));
  out_load_info->physical_device_ordinal = physical_device_ordinal;

  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_loaded_code_object_find(
      libhsa, executable, device_agent, &loaded_code_object));

  hsa_status_t hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &out_load_info->load_delta);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    hsa_status =
        libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
            &out_load_info->load_size);
  }
  return iree_status_from_hsa_status(
      __FILE__, __LINE__, hsa_status,
      "hsa_ven_amd_loader_loaded_code_object_get_info",
      "querying loaded executable code-object profile metadata");
}

#define IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH \
  ((iree_host_size_t)(4 * 1024))

static iree_status_t iree_hal_amdgpu_executable_get_symbol_by_cstring(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const char* symbol_name, hsa_agent_t device_agent,
    hsa_executable_symbol_t* out_symbol) {
  // Kernel callers pass the `.kd` suffix; globals use their source symbol name.
  return iree_hsa_executable_get_symbol_by_name(
      IREE_LIBHSA(libhsa), executable, symbol_name, &device_agent, out_symbol);
}

static iree_status_t iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_string_view_t symbol_name, hsa_agent_t device_agent,
    hsa_executable_symbol_t* out_symbol) {
  if (iree_string_view_is_empty(symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable kernel symbol name is empty");
  }
  if (symbol_name.size > IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable kernel symbol name `%.*s` exceeds maximum length %" PRIhsz,
        (int)symbol_name.size, symbol_name.data,
        IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH);
  }

  // AMDGPU MessagePack strings are length-delimited and not NUL-terminated.
  // Copy only at the HSA API boundary so ROCR can use its internal symbol map.
  char* symbol_name_storage = (char*)iree_alloca(symbol_name.size + 1);
  memcpy(symbol_name_storage, symbol_name.data, symbol_name.size);
  symbol_name_storage[symbol_name.size] = 0;
  return iree_hal_amdgpu_executable_get_symbol_by_cstring(
      libhsa, executable, symbol_name_storage, device_agent, out_symbol);
}

// Resolves the uniform kernel arguments that are the same on all GPU device
// agents in the topology (since we assume all are the same device type).
// All fields besides `kernel_object` will have valid values.
static iree_status_t iree_hal_amdgpu_executable_resolve_kernel_args_from_symbol(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_symbol_t symbol,
    const uint32_t workgroup_size[3],
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  IREE_ASSERT_ARGUMENT(out_kernel_args);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_kernel_args, 0, sizeof(*out_kernel_args));

  // All of our kernels assume 3 dimensions.
  out_kernel_args->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  out_kernel_args->workgroup_size[0] = workgroup_size[0];
  out_kernel_args->workgroup_size[1] = workgroup_size[1];
  out_kernel_args->workgroup_size[2] = workgroup_size[2];

  // NOTE: the object pointer is per-device and we populate that when uploading
  // device tables.
  out_kernel_args->kernel_object = 0;

  // Segment size information used to populate dispatch packets and reserve
  // space.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &out_kernel_args->private_segment_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &out_kernel_args->group_segment_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &out_kernel_args->kernarg_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
              &out_kernel_args->kernarg_alignment));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_resolve_kernel_object(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_symbol_t symbol,
    uint64_t* out_kernel_object) {
  return iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
      out_kernel_object);
}

// Uploads the provided kernel table to |device_agent| and returns the pointer.
// |host_kernel_args| must already have device-specific `kernel_object` fields.
static iree_status_t iree_hal_amdgpu_executable_upload_resolved_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t kernel_count,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device_kernel_args = NULL;

  if (kernel_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Find a memory pool on the agent where we can upload the table.
  hsa_amd_memory_pool_t memory_pool = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_find_coarse_global_memory_pool(libhsa, device_agent,
                                                     &memory_pool),
      "finding memory pool for storing kernel arg tables");

  // Allocate device kernel argument storage.
  const iree_host_size_t kernel_args_table_size =
      kernel_count * sizeof(host_kernel_args[0]);
  iree_hal_amdgpu_device_kernel_args_t* device_kernel_args = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(libhsa), memory_pool, kernel_args_table_size,
              HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&device_kernel_args));

  // Copy the entire table to device memory.
  iree_status_t status =
      iree_hsa_memory_copy(IREE_LIBHSA(libhsa), device_kernel_args,
                           host_kernel_args, kernel_args_table_size);

  if (iree_status_is_ok(status)) {
    *out_device_kernel_args = device_kernel_args;
  } else {
    status = iree_status_join(
        status,
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), device_kernel_args));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_upload_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  const iree_host_size_t kernel_count = metadata->export_count;
  for (iree_host_size_t kernel_ordinal = 0; kernel_ordinal < kernel_count;
       ++kernel_ordinal) {
    iree_string_view_t symbol_name =
        metadata->reflection[kernel_ordinal].symbol_name;
    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
            libhsa, executable, symbol_name, device_agent, &symbol),
        "resolving `%.*s`", (int)symbol_name.size, symbol_name.data);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_resolve_kernel_object(
        libhsa, symbol, &host_kernel_args[kernel_ordinal].kernel_object));
  }
  return iree_hal_amdgpu_executable_upload_resolved_kernel_table(
      libhsa, kernel_count, host_kernel_args, device_agent,
      out_device_kernel_args);
}

static iree_status_t
iree_hal_amdgpu_executable_calculate_kernarg_block_count_from_byte_length(
    iree_host_size_t total_kernarg_size, uint32_t* out_kernarg_block_count) {
  iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      total_kernarg_size, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (kernarg_block_count == 0) {
    kernarg_block_count = 1;
  }
  if (IREE_UNLIKELY(kernarg_block_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch kernargs require too many blocks (%" PRIhsz ", max=%u)",
        kernarg_block_count, UINT32_MAX);
  }
  *out_kernarg_block_count = (uint32_t)kernarg_block_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_calculate_kernarg_block_count(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    uint32_t* out_kernarg_block_count) {
  return iree_hal_amdgpu_executable_calculate_kernarg_block_count_from_byte_length(
      layout->total_kernarg_size, out_kernarg_block_count);
}

static iree_status_t
iree_hal_amdgpu_executable_query_kernel_descriptor_host_ptr(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint64_t kernel_object,
    const iree_hal_amdgpu_kernel_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (kernel_object == 0) return iree_ok_status();

  const void* host_ptr = NULL;
  const hsa_status_t hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_query_host_address(
          (void*)(uintptr_t)kernel_object, &host_ptr);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    return iree_status_from_hsa_status(__FILE__, __LINE__, hsa_status,
                                       "hsa_ven_amd_loader_query_host_address",
                                       "querying kernel object host address");
  }
  if (IREE_UNLIKELY(!host_ptr)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel object host address translation returned NULL");
  }

  *out_descriptor = (const iree_hal_amdgpu_kernel_descriptor_t*)host_ptr;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_initialize_dispatch_descriptor(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_workgroup_cluster_capabilities_t* workgroup_cluster,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_kernarg_layout_t* kernarg_layout,
    iree_host_size_t custom_explicit_kernarg_size,
    uint16_t custom_implicit_args_offset,
    iree_hal_amdgpu_executable_dispatch_descriptor_t* out_descriptor) {
  memset(out_descriptor, 0, sizeof(*out_descriptor));

  if (IREE_UNLIKELY(
          !iree_host_size_is_power_of_two(kernel_args->kernarg_alignment))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment must be a power of two (got %u)",
        kernel_args->kernarg_alignment);
  }
  if (IREE_UNLIKELY(kernel_args->kernarg_alignment >
                    iree_alignof(iree_hal_amdgpu_kernarg_block_t))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment %u exceeds queue kernarg ring "
        "alignment %" PRIhsz,
        kernel_args->kernarg_alignment,
        (iree_host_size_t)iree_alignof(iree_hal_amdgpu_kernarg_block_t));
  }

  out_descriptor->kernel_args = *kernel_args;
  out_descriptor->workgroup_cluster_count_limits =
      workgroup_cluster->cluster_count;
  out_descriptor->physical_device_ordinal = physical_device_ordinal;
  out_descriptor->kernarg_layout = kernarg_layout;
  if (kernarg_layout) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_calculate_kernarg_block_count_from_byte_length(
            kernarg_layout->kernarg_byte_length,
            &out_descriptor->kernarg_block_count));
  }

  if (custom_implicit_args_offset == UINT16_MAX && kernarg_layout &&
      iree_any_bit_set(kernarg_layout->flags,
                       IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS)) {
    custom_implicit_args_offset = kernarg_layout->implicit_args_byte_offset;
  }

  // Dynamic custom-direct layout: callers provide a prepacked ABI blob and its
  // length at dispatch time. Fixed fields are only needed when the metadata
  // requires an implicit suffix we must synthesize.
  out_descriptor->custom_kernarg_layout =
      (iree_hal_amdgpu_device_dispatch_kernarg_layout_t){
          .explicit_kernarg_size = custom_explicit_kernarg_size,
      };
  if (custom_implicit_args_offset != UINT16_MAX) {
    // Custom-direct callers own every visible native kernarg byte. Hidden
    // metadata marks where the queue code must synthesize the implicit suffix.
    // A zero explicit size remains dynamic: some HIP callers omit ABI padding
    // before the suffix, and the command buffer zero-fills that gap.
    out_descriptor->custom_kernarg_layout.implicit_args_offset =
        custom_implicit_args_offset;
    out_descriptor->custom_kernarg_layout.total_kernarg_size =
        iree_max((size_t)kernel_args->kernarg_size,
                 iree_max((size_t)custom_explicit_kernarg_size,
                          (size_t)custom_implicit_args_offset +
                              IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE));
    out_descriptor->custom_kernarg_layout.has_implicit_args = true;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_calculate_kernarg_block_count(
      &out_descriptor->custom_kernarg_layout,
      &out_descriptor->custom_kernarg_block_count));

  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(kernel_args->workgroup_size[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable kernel workgroup size dimension %" PRIhsz " is zero", i);
    }
    out_descriptor->max_workgroup_count[i] =
        UINT32_MAX / kernel_args->workgroup_size[i];
  }
  out_descriptor->max_dynamic_workgroup_local_memory =
      UINT32_MAX - kernel_args->group_segment_size;

  // HSA reports the kernel object as the dispatch handle, but CPU-side
  // descriptor reads must go through the AMD loader host-address translation.
  const uint64_t kernel_object = kernel_args->kernel_object;
  if (IREE_UNLIKELY(kernel_object == 0)) {
    return iree_ok_status();
  }
  const iree_hal_amdgpu_kernel_descriptor_t* amdhsa_descriptor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_query_kernel_descriptor_host_ptr(
          libhsa, kernel_object, &amdhsa_descriptor));
  out_descriptor->pm4_group_segment_fixed_size =
      amdhsa_descriptor->group_segment_fixed_size;
  const bool uses_workgroup_clusters =
      kernel_args->workgroup_cluster_size[0] != 0 ||
      kernel_args->workgroup_cluster_size[1] != 0 ||
      kernel_args->workgroup_cluster_size[2] != 0;
  out_descriptor->pm4_launch_state_valid =
      !uses_workgroup_clusters &&
      iree_hal_amdgpu_pm4_dispatch_launch_state_is_supported(
          gfxip_version, amdhsa_descriptor, kernel_object,
          kernel_args->workgroup_size,
          IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_NONE);
  if (out_descriptor->pm4_launch_state_valid) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize(
        gfxip_version, amdhsa_descriptor, kernel_object,
        kernel_args->workgroup_size,
        IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_NONE,
        &out_descriptor->pm4_launch_state));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_setup(
        &out_descriptor->pm4_launch_state,
        IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT,
        out_descriptor->pm4_setup_dwords,
        &out_descriptor->pm4_setup_dword_count));
    if (IREE_UNLIKELY(out_descriptor->pm4_setup_dword_count !=
                      IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 dispatch setup emission changed size");
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_executable_load_variant_t {
  // Loaded HSA executable handle for this variant.
  hsa_executable_t handle;

  // HSA-backed resolver state borrowed by |global_table|.
  iree_hal_amdgpu_executable_global_resolver_t global_resolver;

  // Executable global lookup and per-device buffer alias cache for |handle|.
  iree_hal_amdgpu_global_table_t global_table;

  // Physical queue ordinal represented by this variant.
  iree_host_size_t physical_queue_ordinal;
} iree_hal_amdgpu_executable_load_variant_t;

typedef struct iree_hal_amdgpu_executable_t {
  // HAL executable resource header.
  iree_hal_resource_t resource;
  // Host allocator used for executable-owned metadata tables.
  iree_allocator_t host_allocator;

  // Unowned HSA API handle. Must remain valid for the lifetime of the
  // executable.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Borrowed HAL device used for global-buffer placement metadata.
  iree_hal_device_t* device;

  // Executable-owned copy of the exact HSACO bytes backing
  // |code_object_reader|.
  iree_byte_span_t code_object_data;
  // HSA code-object reader retained until all associated executable variants
  // have been destroyed.
  hsa_code_object_reader_t code_object_reader;

  // Logical-device-local executable identifier assigned at creation.
  uint64_t executable_id;
  // Stable content hash for the exact loaded HSACO/code-object bytes.
  uint64_t code_object_hash[2];
  // Executable-owned source context used by feedback packet attribution.
  iree_hal_amdgpu_source_context_t source_context;
  // Provider-neutral metadata borrowing from |handle|'s loaded code object.
  iree_hal_amdgpu_executable_metadata_t* metadata;

  // Total number of exports in the executable.
  iree_host_size_t kernel_count;
  // Host-resident reflection information for each export.
  iree_hal_executable_function_info_t* export_infos /*[kernel_count]*/;
  // Prefix-sum offsets into |export_parameters| for each export plus a
  // sentinel.
  iree_host_size_t* export_parameter_offsets /*[kernel_count + 1]*/;
  // Host-resident parameter reflection records for all exports.
  iree_hal_executable_function_parameter_t* export_parameters;
  // True for exports discovered from ELF symbols without AMDGPU metadata.
  bool* custom_direct_only_exports /*[kernel_count]*/;
  // Table of kernel args stored in host memory. We have them local so that
  // host-side command buffer recording doesn't need to access device memory.
  // The kernel object specified in each is invalid as it's agent-specific.
  iree_hal_amdgpu_device_kernel_args_t* host_kernel_args /*[kernel_count]*/;
  // Host-resident dispatch descriptors stored as
  // [load_variant_capacity][device_count][kernel_count].
  iree_hal_amdgpu_executable_dispatch_descriptor_t* host_dispatch_descriptors
      /*[load_variant_capacity * device_count * kernel_count]*/;

  // Queue affinity this executable was loaded for after normalization.
  iree_hal_queue_affinity_t queue_affinity;
  // Queue-affinity domain used to resolve per-device globals.
  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain;
  // Bitmask of physical GPU device ordinals with loaded code objects.
  uint64_t loaded_physical_device_mask;
  // Number of loaded physical GPU devices in |loaded_physical_device_mask|.
  iree_host_size_t loaded_physical_device_count;
  // Total number of GPU devices in the topology used for per-device tables.
  iree_host_size_t device_count;
  // Number of allocated entries in |load_variants|.
  iree_host_size_t load_variant_capacity;
  // Number of initialized entries in |load_variants|.
  iree_host_size_t load_variant_count;
  // Loaded executable handles and global tables.
  iree_hal_amdgpu_executable_load_variant_t* load_variants;
  // True when dispatch metadata must be selected by queue.
  bool requires_queue_scope;
  // Number of entries in |queue_scopes|.
  iree_host_size_t queue_scope_count;
  // Immutable queue identities captured from the owning logical device.
  iree_hal_amdgpu_queue_scope_t* queue_scopes;
  // Loaded code-object host/device ranges indexed by physical device ordinal.
  iree_hal_amdgpu_loaded_code_object_range_t*
      loaded_code_object_ranges /*[device_count]*/;
  // HSA GPU agents indexed by physical device ordinal.
  hsa_agent_t* device_agents /*[device_count]*/;
  // Table of kernel args stored in device memory.
  //
  // Entries are stored as [load_variant_capacity][device_count]. Selected
  // variant/device pairs have an entire `kernel_count` set of args; unselected
  // pairs remain NULL and fail lookup.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*
      device_kernel_args[/*load_variant_capacity * device_count*/];
} iree_hal_amdgpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_amdgpu_executable_vtable;

static iree_hal_amdgpu_executable_t* iree_hal_amdgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_executable_vtable);
  return (iree_hal_amdgpu_executable_t*)base_value;
}

static const iree_hal_amdgpu_executable_t*
iree_hal_amdgpu_executable_const_cast(const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_executable_vtable);
  return (const iree_hal_amdgpu_executable_t*)base_value;
}

static iree_status_t iree_hal_amdgpu_executable_create_empty(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    iree_host_size_t export_count, iree_host_size_t load_variant_capacity,
    iree_host_size_t queue_scope_count,
    const iree_hal_amdgpu_queue_scope_t* queue_scopes,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_executable_t** out_executable) {
  *out_executable = NULL;

  iree_host_size_t dispatch_descriptor_count = 0;
  if (!iree_host_size_checked_mul(load_variant_capacity,
                                  topology->gpu_agent_count,
                                  &dispatch_descriptor_count) ||
      !iree_host_size_checked_mul(dispatch_descriptor_count, export_count,
                                  &dispatch_descriptor_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch descriptor table size overflow");
  }

  iree_host_size_t device_kernel_args_count = 0;
  if (!iree_host_size_checked_mul(load_variant_capacity,
                                  topology->gpu_agent_count,
                                  &device_kernel_args_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device kernel args table size overflow");
  }

  iree_host_size_t export_parameter_offset_count = 0;
  if (!iree_host_size_checked_add(export_count, 1,
                                  &export_parameter_offset_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export parameter offset table size overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t export_infos_offset = 0;
  iree_host_size_t export_parameter_offsets_offset = 0;
  iree_host_size_t custom_direct_only_exports_offset = 0;
  iree_host_size_t host_kernel_args_offset = 0;
  iree_host_size_t host_dispatch_descriptors_offset = 0;
  iree_host_size_t load_variants_offset = 0;
  iree_host_size_t queue_scopes_offset = 0;
  iree_host_size_t loaded_code_object_ranges_offset = 0;
  iree_host_size_t device_agents_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_executable_t), &total_size,
      IREE_STRUCT_FIELD_FAM(
          device_kernel_args_count,
          IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*),
      IREE_STRUCT_FIELD_ALIGNED(
          export_count, iree_hal_executable_function_info_t,
          iree_alignof(iree_hal_executable_function_info_t),
          &export_infos_offset),
      IREE_STRUCT_FIELD_ALIGNED(export_parameter_offset_count, iree_host_size_t,
                                iree_alignof(iree_host_size_t),
                                &export_parameter_offsets_offset),
      IREE_STRUCT_FIELD_ALIGNED(export_count, bool, iree_alignof(bool),
                                &custom_direct_only_exports_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          export_count, iree_hal_amdgpu_device_kernel_args_t,
          iree_alignof(iree_hal_amdgpu_device_kernel_args_t),
          &host_kernel_args_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          dispatch_descriptor_count,
          iree_hal_amdgpu_executable_dispatch_descriptor_t,
          iree_alignof(iree_hal_amdgpu_executable_dispatch_descriptor_t),
          &host_dispatch_descriptors_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          load_variant_capacity, iree_hal_amdgpu_executable_load_variant_t,
          iree_alignof(iree_hal_amdgpu_executable_load_variant_t),
          &load_variants_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          queue_scope_count, iree_hal_amdgpu_queue_scope_t,
          iree_alignof(iree_hal_amdgpu_queue_scope_t), &queue_scopes_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          topology->gpu_agent_count, iree_hal_amdgpu_loaded_code_object_range_t,
          iree_alignof(iree_hal_amdgpu_loaded_code_object_range_t),
          &loaded_code_object_ranges_offset),
      IREE_STRUCT_FIELD_ALIGNED(topology->gpu_agent_count, hsa_agent_t,
                                iree_alignof(hsa_agent_t),
                                &device_agents_offset)));

  iree_hal_amdgpu_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->libhsa = libhsa;
  executable->device = device;
  executable->kernel_count = export_count;
  uint8_t* executable_storage = (uint8_t*)executable;
  executable->export_infos =
      (iree_hal_executable_function_info_t*)(executable_storage +
                                             export_infos_offset);
  executable->export_parameter_offsets =
      (iree_host_size_t*)(executable_storage + export_parameter_offsets_offset);
  executable->custom_direct_only_exports =
      (bool*)(executable_storage + custom_direct_only_exports_offset);
  executable->host_kernel_args =
      (iree_hal_amdgpu_device_kernel_args_t*)(executable_storage +
                                              host_kernel_args_offset);
  executable->host_dispatch_descriptors =
      (iree_hal_amdgpu_executable_dispatch_descriptor_t*)(executable_storage +
                                                          host_dispatch_descriptors_offset);
  executable->loaded_code_object_ranges =
      (iree_hal_amdgpu_loaded_code_object_range_t*)(executable_storage +
                                                    loaded_code_object_ranges_offset);
  executable->device_agents =
      (hsa_agent_t*)(executable_storage + device_agents_offset);
  memcpy(executable->device_agents, topology->gpu_agents,
         topology->gpu_agent_count * sizeof(executable->device_agents[0]));
  executable->queue_affinity = physical_devices->queue_affinity;
  executable->queue_affinity_domain = (iree_hal_amdgpu_queue_affinity_domain_t){
      .supported_affinity = physical_devices->queue_affinity,
      .physical_device_count = topology->gpu_agent_count,
      .queue_count_per_physical_device = topology->gpu_agent_queue_count,
  };
  executable->loaded_physical_device_mask =
      physical_devices->physical_device_mask;
  executable->loaded_physical_device_count =
      physical_devices->physical_device_count;
  executable->device_count = topology->gpu_agent_count;
  executable->load_variant_capacity = load_variant_capacity;
  executable->load_variant_count = 1;
  executable->load_variants =
      (iree_hal_amdgpu_executable_load_variant_t*)(executable_storage +
                                                   load_variants_offset);
  executable->load_variants[0].physical_queue_ordinal = 0;
  executable->queue_scope_count = queue_scope_count;
  executable->queue_scopes =
      (iree_hal_amdgpu_queue_scope_t*)(executable_storage +
                                       queue_scopes_offset);
  if (queue_scope_count != 0) {
    memcpy(executable->queue_scopes, queue_scopes,
           queue_scope_count * sizeof(executable->queue_scopes[0]));
  }

  *out_executable = executable;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_executable_variant_device_ordinal(
    const iree_hal_amdgpu_executable_t* executable,
    iree_host_size_t variant_ordinal, iree_host_size_t device_ordinal) {
  return variant_ordinal * executable->device_count + device_ordinal;
}

static iree_host_size_t iree_hal_amdgpu_executable_descriptor_ordinal(
    const iree_hal_amdgpu_executable_t* executable,
    iree_host_size_t variant_ordinal, iree_host_size_t device_ordinal,
    iree_host_size_t kernel_ordinal) {
  return (iree_hal_amdgpu_executable_variant_device_ordinal(
              executable, variant_ordinal, device_ordinal) *
          executable->kernel_count) +
         kernel_ordinal;
}

static iree_hal_amdgpu_executable_load_variant_t*
iree_hal_amdgpu_executable_primary_variant(
    iree_hal_amdgpu_executable_t* executable) {
  return &executable->load_variants[0];
}

static iree_status_t iree_hal_amdgpu_executable_select_queue_scope(
    const iree_hal_amdgpu_executable_t* executable,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved,
    const iree_hal_amdgpu_queue_scope_t** out_queue_scope) {
  *out_queue_scope = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve(
      executable->queue_affinity_domain, queue_affinity, out_resolved));
  for (iree_host_size_t i = 0; i < executable->queue_scope_count; ++i) {
    const iree_hal_amdgpu_queue_scope_t* queue_scope =
        &executable->queue_scopes[i];
    if (queue_scope->physical_device_ordinal ==
            out_resolved->physical_device_ordinal &&
        queue_scope->physical_queue_ordinal ==
            out_resolved->physical_queue_ordinal) {
      *out_queue_scope = queue_scope;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU executable queue scope not available for "
                          "queue affinity 0x%" PRIx64,
                          queue_affinity);
}

static iree_status_t iree_hal_amdgpu_executable_select_load_variant_for_queue(
    const iree_hal_amdgpu_executable_t* executable,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved,
    iree_host_size_t* out_variant_ordinal) {
  *out_variant_ordinal = 0;
  if (!executable->requires_queue_scope) {
    return iree_hal_amdgpu_queue_affinity_resolve(
        executable->queue_affinity_domain, queue_affinity, out_resolved);
  }

  const iree_hal_amdgpu_queue_scope_t* queue_scope = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_select_queue_scope(
      executable, queue_affinity, out_resolved, &queue_scope));
  const iree_host_size_t variant_ordinal = queue_scope->physical_queue_ordinal;
  if (IREE_UNLIKELY(variant_ordinal >= executable->load_variant_count)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU executable queue variant %" PRIhsz
                            " is not loaded; loaded variant count is %" PRIhsz,
                            variant_ordinal, executable->load_variant_count);
  }
  *out_variant_ordinal = variant_ordinal;
  return iree_ok_status();
}

static const iree_hal_amdgpu_queue_scope_t*
iree_hal_amdgpu_executable_find_queue_scope(
    const iree_hal_amdgpu_executable_t* executable,
    iree_host_size_t physical_device_ordinal,
    iree_host_size_t physical_queue_ordinal) {
  for (iree_host_size_t i = 0; i < executable->queue_scope_count; ++i) {
    const iree_hal_amdgpu_queue_scope_t* queue_scope =
        &executable->queue_scopes[i];
    if (queue_scope->physical_device_ordinal == physical_device_ordinal &&
        queue_scope->physical_queue_ordinal == physical_queue_ordinal) {
      return queue_scope;
    }
  }
  return NULL;
}

static void iree_hal_amdgpu_executable_invalidate_host_kernel_objects(
    iree_hal_amdgpu_executable_t* executable) {
  if (!executable) return;
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < executable->kernel_count; ++kernel_ordinal) {
    executable->host_kernel_args[kernel_ordinal].kernel_object = 0;
  }
}

static iree_status_t
iree_hal_amdgpu_executable_try_lookup_required_config_global(
    iree_hal_amdgpu_global_table_t* global_table, const char* global_name,
    iree_host_size_t expected_byte_length, bool capability_enabled,
    const char* capability_name, const char* enable_hint, bool* out_found,
    iree_hal_executable_global_t* out_global) {
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_try_lookup(
      global_table, iree_make_cstring_view(global_name), out_found,
      out_global));
  if (!*out_found) return iree_ok_status();

  if (IREE_UNLIKELY(!capability_enabled)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable references runtime feature global `%s`, but %s is "
        "not enabled on the logical device; %s",
        global_name, capability_name, enable_hint);
  }

  iree_hal_executable_global_info_t info;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_global_table_info(global_table, *out_global, &info),
      "querying AMDGPU runtime feature global info");
  if (IREE_UNLIKELY(info.byte_length != expected_byte_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU runtime feature global `%s` has length %" PRIu64
        " but the runtime ABI requires %" PRIhsz,
        global_name, (uint64_t)info.byte_length, expected_byte_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_publish_asan_config(
    iree_hal_amdgpu_executable_t* executable,
    iree_hal_amdgpu_executable_load_variant_t* load_variant,
    const iree_hal_amdgpu_asan_state_t* asan_state) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_try_lookup_required_config_global(
          &load_variant->global_table, IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME,
          sizeof(iree_hal_amdgpu_asan_config_t),
          iree_hal_amdgpu_asan_state_is_enabled(asan_state),
          "AMDGPU ASAN shadow memory",
          "enable HAL ASAN runtime support before loading this executable",
          &found, &global));
  if (!found) return iree_ok_status();

  iree_hal_amdgpu_asan_config_t config;
  iree_hal_amdgpu_asan_state_populate_config(asan_state, &config);

  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < executable->device_count; ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }

    iree_hal_queue_affinity_t queue_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        executable->queue_affinity_domain, device_ordinal, &queue_affinity));

    iree_hal_buffer_t* global_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_global_table_buffer(&load_variant->global_table, global,
                                            queue_affinity, &global_buffer),
        "resolving ASAN config global buffer");

    // Global buffers are executable-owned aliases and are not released here.
    void* target_ptr = iree_hal_amdgpu_buffer_device_pointer(global_buffer);
    if (IREE_UNLIKELY(!target_ptr)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU ASAN config global `%s` did not resolve to device memory",
          IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME);
    }
    IREE_RETURN_IF_ERROR(
        iree_hsa_memory_copy(IREE_LIBHSA(executable->libhsa), target_ptr,
                             &config, sizeof(config)),
        "publishing ASAN config global on physical device %" PRIhsz,
        device_ordinal);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_publish_tsan_config(
    iree_hal_amdgpu_executable_t* executable,
    iree_hal_amdgpu_executable_load_variant_t* load_variant,
    const iree_hal_amdgpu_tsan_state_t* tsan_state) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_try_lookup_required_config_global(
          &load_variant->global_table, IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME,
          sizeof(iree_hal_amdgpu_tsan_config_t),
          iree_hal_amdgpu_tsan_state_is_enabled(tsan_state),
          "AMDGPU TSAN shadow memory",
          "enable HAL TSAN runtime support before loading this executable",
          &found, &global));
  if (!found) return iree_ok_status();

  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < executable->device_count; ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }

    iree_hal_amdgpu_tsan_config_t config;
    const iree_hal_amdgpu_queue_scope_t* queue_scope =
        iree_hal_amdgpu_executable_find_queue_scope(
            executable, device_ordinal, load_variant->physical_queue_ordinal);
    if (IREE_UNLIKELY(!queue_scope)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU TSAN queue scope missing for physical device %" PRIhsz
          " queue ordinal %" PRIhsz,
          device_ordinal, load_variant->physical_queue_ordinal);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_tsan_state_populate_queue_config(
                             tsan_state, queue_scope, &config),
                         "populating TSAN config for physical device %" PRIhsz
                         " queue ordinal %" PRIhsz,
                         device_ordinal, load_variant->physical_queue_ordinal);
    iree_hal_queue_affinity_t queue_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        executable->queue_affinity_domain, device_ordinal, &queue_affinity));

    iree_hal_buffer_t* global_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_global_table_buffer(&load_variant->global_table, global,
                                            queue_affinity, &global_buffer),
        "resolving TSAN config global buffer");

    // Global buffers are executable-owned aliases and are not released here.
    void* target_ptr = iree_hal_amdgpu_buffer_device_pointer(global_buffer);
    if (IREE_UNLIKELY(!target_ptr)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU TSAN config global `%s` did not resolve to device memory",
          IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME);
    }
    IREE_RETURN_IF_ERROR(
        iree_hsa_memory_copy(IREE_LIBHSA(executable->libhsa), target_ptr,
                             &config, sizeof(config)),
        "publishing TSAN config global on physical device %" PRIhsz,
        device_ordinal);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_query_loaded_code_object_ranges(
    iree_hal_amdgpu_executable_t* executable) {
  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < executable->device_count; ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }
    iree_hal_amdgpu_loaded_code_object_range_t range = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_loaded_code_object_query_agent_range(
            executable->libhsa,
            iree_hal_amdgpu_executable_primary_variant(executable)->handle,
            executable->device_agents[device_ordinal], &range),
        "querying AMDGPU executable loaded code-object range for physical "
        "device %" PRIhsz,
        device_ordinal);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_source_context_set_loaded_code_object_range(
            &executable->source_context, device_ordinal, range));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_try_attach_sanitizer_site_table(
    iree_hal_amdgpu_executable_t* executable,
    iree_host_size_t physical_device_ordinal) {
  iree_hal_amdgpu_executable_load_variant_t* load_variant =
      iree_hal_amdgpu_executable_primary_variant(executable);
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_try_lookup(
      &load_variant->global_table,
      iree_make_cstring_view(
          IREE_HAL_AMDGPU_LOOM_SANITIZER_SITE_TABLE_GLOBAL_NAME),
      &found, &global));
  if (!found) return iree_ok_status();

  iree_hal_executable_global_info_t info;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_info(
                           &load_variant->global_table, global, &info),
                       "querying AMDGPU sanitizer site table global info");

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
      executable->queue_affinity_domain, physical_device_ordinal,
      &queue_affinity));

  iree_hal_buffer_t* global_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_global_table_buffer(&load_variant->global_table, global,
                                          queue_affinity, &global_buffer),
      "resolving AMDGPU sanitizer site table global buffer");

  const uint64_t device_pointer =
      (uint64_t)(uintptr_t)iree_hal_amdgpu_buffer_device_pointer(global_buffer);
  if (IREE_UNLIKELY(device_pointer == 0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer site table global `%s` did not resolve to device "
        "memory",
        IREE_HAL_AMDGPU_LOOM_SANITIZER_SITE_TABLE_GLOBAL_NAME);
  }

  iree_const_byte_span_t host_span = iree_const_byte_span_empty();
  if (IREE_UNLIKELY(!iree_hal_amdgpu_source_context_try_translate_device_span(
          &executable->source_context, physical_device_ordinal, device_pointer,
          info.byte_length, &host_span))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU sanitizer site table global `%s` is outside the loaded "
        "code-object range",
        IREE_HAL_AMDGPU_LOOM_SANITIZER_SITE_TABLE_GLOBAL_NAME);
  }

  return iree_hal_amdgpu_source_context_set_sanitizer_site_table(
      &executable->source_context, host_span);
}

static iree_status_t iree_hal_amdgpu_executable_publish_feedback_config(
    iree_hal_amdgpu_executable_t* executable,
    iree_hal_amdgpu_executable_load_variant_t* load_variant,
    const iree_hal_amdgpu_feedback_state_t* feedback_state) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_try_lookup_required_config_global(
          &load_variant->global_table,
          IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME,
          sizeof(iree_hal_amdgpu_feedback_config_t),
          iree_hal_amdgpu_feedback_state_is_enabled(feedback_state),
          "the AMDGPU feedback channel",
          "enable HAL feedback runtime support before loading this executable",
          &found, &global));
  if (!found) return iree_ok_status();

  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < executable->device_count; ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }

    iree_hal_amdgpu_feedback_config_t config;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_feedback_state_populate_config(feedback_state,
                                                       device_ordinal, &config),
        "populating feedback config for physical device %" PRIhsz,
        device_ordinal);
    config.source_context = (uint64_t)(uintptr_t)&executable->source_context;

    iree_hal_queue_affinity_t queue_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        executable->queue_affinity_domain, device_ordinal, &queue_affinity));

    iree_hal_buffer_t* global_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_global_table_buffer(&load_variant->global_table, global,
                                            queue_affinity, &global_buffer),
        "resolving feedback config global buffer");

    // Global buffers are executable-owned aliases and are not released here.
    void* target_ptr = iree_hal_amdgpu_buffer_device_pointer(global_buffer);
    if (IREE_UNLIKELY(!target_ptr)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU feedback config global `%s` did not resolve to device memory",
          IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME);
    }
    // The context pointer crosses the device global and feedback packet before
    // an HSA signal wakes the host reader. TSAN cannot observe that path.
    IREE_TSAN_RELEASE(&executable->source_context);
    IREE_RETURN_IF_ERROR(
        iree_hsa_memory_copy(IREE_LIBHSA(executable->libhsa), target_ptr,
                             &config, sizeof(config)),
        "publishing feedback config global on physical device %" PRIhsz,
        device_ordinal);
  }

  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_executable_initialize_dispatch_descriptors_for_device(
    iree_hal_amdgpu_executable_t* executable,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    iree_host_size_t variant_ordinal, iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_workgroup_cluster_capabilities_t* workgroup_cluster,
    const iree_hal_amdgpu_executable_metadata_t* dispatch_metadata,
    const iree_host_size_t* custom_explicit_kernarg_sizes,
    const uint16_t* custom_implicit_args_offsets) {
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < executable->kernel_count; ++kernel_ordinal) {
    const iree_host_size_t descriptor_ordinal =
        iree_hal_amdgpu_executable_descriptor_ordinal(
            executable, variant_ordinal, device_ordinal, kernel_ordinal);
    const iree_host_size_t custom_explicit_kernarg_size =
        custom_explicit_kernarg_sizes
            ? custom_explicit_kernarg_sizes[kernel_ordinal]
            : executable->host_kernel_args[kernel_ordinal].kernarg_size;
    const uint16_t custom_implicit_args_offset =
        custom_implicit_args_offsets
            ? custom_implicit_args_offsets[kernel_ordinal]
            : UINT16_MAX;
    const iree_hal_amdgpu_kernarg_layout_t* kernarg_layout = NULL;
    if (!executable->custom_direct_only_exports[kernel_ordinal]) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_executable_metadata_resolve_layout(
              dispatch_metadata,
              dispatch_metadata->exports[kernel_ordinal].kernarg_layout,
              &kernarg_layout),
          "resolving dispatch kernarg layout for export %" PRIhsz,
          kernel_ordinal);
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_initialize_dispatch_descriptor(
            executable->libhsa, gfxip_version, device_ordinal,
            workgroup_cluster, &executable->host_kernel_args[kernel_ordinal],
            kernarg_layout, custom_explicit_kernarg_size,
            custom_implicit_args_offset,
            &executable->host_dispatch_descriptors[descriptor_ordinal]),
        "initializing dispatch descriptor for device %" PRIhsz
        " export %" PRIhsz,
        device_ordinal, kernel_ordinal);
    executable->host_dispatch_descriptors[descriptor_ordinal]
        .custom_direct_only =
        executable->custom_direct_only_exports[kernel_ordinal];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_register_profile_artifacts(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_const_byte_span_t code_object_data,
    iree_hal_amdgpu_executable_t* executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t load_info_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &load_info_storage_size,
              IREE_STRUCT_FIELD(executable->loaded_physical_device_count,
                                iree_hal_amdgpu_profile_code_object_load_info_t,
                                NULL)));

  iree_hal_amdgpu_profile_code_object_load_info_t* load_infos = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(executable->host_allocator,
                                load_info_storage_size, (void**)&load_infos));

  iree_status_t status = iree_ok_status();
  iree_host_size_t load_info_ordinal = 0;
  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < topology->gpu_agent_count && iree_status_is_ok(status);
       ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }
    if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable physical device ordinal exceeds uint32_t");
    } else {
      status =
          iree_hal_amdgpu_executable_populate_profile_code_object_load_info(
              libhsa,
              iree_hal_amdgpu_executable_primary_variant(executable)->handle,
              (uint32_t)device_ordinal, topology->gpu_agents[device_ordinal],
              &load_infos[load_info_ordinal]);
      if (iree_status_is_ok(status)) ++load_info_ordinal;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_register_executable_artifacts(
        profile_metadata, executable->executable_id, code_object_data,
        executable->code_object_hash, executable->loaded_physical_device_count,
        load_infos);
  }

  iree_allocator_free(executable->host_allocator, load_infos);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_register_profile_metadata(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_const_byte_span_t code_object_data,
    iree_hal_amdgpu_executable_t* executable) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_register_executable(
      profile_metadata, executable->kernel_count, executable->export_infos,
      executable->export_parameter_offsets, executable->code_object_hash,
      executable->executable_id));

  // Executable trace profiling may begin after executable preparation. Preserve
  // exact code-object bytes and loader load ranges while |code_object_data| is
  // still in scope so later ATT capture can always emit a self-contained
  // profile.
  return iree_hal_amdgpu_executable_register_profile_artifacts(
      libhsa, topology, profile_metadata, code_object_data, executable);
}

static iree_status_t iree_hal_amdgpu_executable_create_metadata_from_hsaco(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t any_device_agent,
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_executable_metadata_t** out_metadata) {
  IREE_ASSERT_ARGUMENT(out_metadata);
  *out_metadata = NULL;

  iree_hal_amdgpu_loaded_code_object_range_t loaded_code_object_range = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_loaded_code_object_query_agent_range(
      libhsa, executable, any_device_agent, &loaded_code_object_range));
  if (IREE_UNLIKELY(loaded_code_object_range.byte_length >
                    IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "loaded code object byte length exceeds host size");
  }
  const iree_const_byte_span_t loaded_code_object_data =
      iree_make_const_byte_span(
          loaded_code_object_range.host_pointer,
          (iree_host_size_t)loaded_code_object_range.byte_length);

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(hsaco_metadata,
                                                                 &counts));

  iree_hal_amdgpu_executable_metadata_t* metadata = NULL;
  iree_status_t status = iree_hal_amdgpu_executable_metadata_allocate(
      &counts, host_allocator, &metadata);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
        hsaco_metadata, loaded_code_object_data, metadata);
  }
  if (iree_status_is_ok(status)) {
    *out_metadata = metadata;
  } else {
    iree_hal_amdgpu_executable_metadata_free(metadata);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_validate_workgroup_size(
    iree_string_view_t symbol_name, const uint32_t workgroup_size[3],
    const iree_hal_amdgpu_device_limits_t* limits) {
  if (workgroup_size[0] > limits->max_workgroup_size_per_dim[0] ||
      workgroup_size[1] > limits->max_workgroup_size_per_dim[1] ||
      workgroup_size[2] > limits->max_workgroup_size_per_dim[2]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel `%.*s` workgroup size dims %ux%ux%u exceed device "
        "maximum %ux%ux%u",
        (int)symbol_name.size, symbol_name.data, workgroup_size[0],
        workgroup_size[1], workgroup_size[2],
        limits->max_workgroup_size_per_dim[0],
        limits->max_workgroup_size_per_dim[1],
        limits->max_workgroup_size_per_dim[2]);
  }
  const uint64_t total_workgroup_size =
      (uint64_t)workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
  if (total_workgroup_size > limits->max_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel `%.*s` workgroup size total %" PRIu64
                            " exceeds device maximum %u",
                            (int)symbol_name.size, symbol_name.data,
                            total_workgroup_size, limits->max_workgroup_size);
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_executable_validate_workgroup_cluster_sizes(
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        selected_devices,
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_device_list) {
  for (iree_host_size_t export_ordinal = 0;
       export_ordinal < metadata->export_count; ++export_ordinal) {
    const iree_hal_amdgpu_executable_export_t* metadata_export =
        &metadata->exports[export_ordinal];
    const iree_string_view_t symbol_name =
        metadata->reflection[export_ordinal].symbol_name;
    for (iree_host_size_t device_ordinal = 0;
         device_ordinal < physical_device_count; ++device_ordinal) {
      if (!iree_hal_amdgpu_physical_device_mask_contains(
              selected_devices->physical_device_mask, device_ordinal)) {
        continue;
      }
      const iree_hal_amdgpu_physical_device_t* physical_device =
          physical_device_list[device_ordinal];
      if (IREE_UNLIKELY(!physical_device ||
                        physical_device->device_ordinal != device_ordinal)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "AMDGPU physical device list entry %" PRIhsz
                                " does not match its topology ordinal",
                                device_ordinal);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_workgroup_cluster_size(
          symbol_name, metadata_export->workgroup_cluster_size, device_ordinal,
          &physical_device->workgroup_cluster));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_initialize_export_infos(
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    const iree_hal_amdgpu_device_limits_t* limits,
    iree_hal_amdgpu_executable_t* executable) {
  executable->export_parameters = metadata->parameters;
  for (iree_host_size_t i = 0; i < metadata->export_count; ++i) {
    const iree_hal_amdgpu_executable_export_t* metadata_export =
        &metadata->exports[i];
    const iree_hal_amdgpu_executable_reflection_t* reflection =
        &metadata->reflection[i];
    iree_hal_executable_function_info_t* info = &executable->export_infos[i];
    const bool requires_dispatch_workgroup_size = iree_any_bit_set(
        metadata_export->flags,
        IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE);
    const bool custom_direct_only = iree_any_bit_set(
        metadata_export->flags,
        IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY);

    const iree_hal_amdgpu_kernarg_layout_t* layout = NULL;
    if (!custom_direct_only) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_executable_metadata_resolve_layout(
              metadata, metadata_export->kernarg_layout, &layout),
          "resolving dispatch kernarg layout for export %" PRIhsz, i);
    }
    if (!requires_dispatch_workgroup_size) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_validate_workgroup_size(
          reflection->symbol_name, metadata_export->workgroup_size, limits));
    }
    const uint32_t maximum_workgroup_invocations =
        metadata_export->max_workgroup_size != 0
            ? metadata_export->max_workgroup_size
            : limits->max_workgroup_size;
    if (IREE_UNLIKELY(maximum_workgroup_invocations >
                      limits->max_workgroup_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` maximum workgroup size %u exceeds device "
          "maximum %u",
          (int)reflection->symbol_name.size, reflection->symbol_name.data,
          maximum_workgroup_invocations, limits->max_workgroup_size);
    }

    executable->custom_direct_only_exports[i] = custom_direct_only;
    executable->export_parameter_offsets[i] = reflection->parameter_offset;

    memset(info, 0, sizeof(*info));
    info->name = reflection->name;
    info->flags = requires_dispatch_workgroup_size
                      ? IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC
                      : IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE;
    info->constant_byte_length = layout ? layout->constant_byte_length : 0;
    info->binding_count = layout ? layout->binding_count : 0;
    info->parameter_count = reflection->parameter_count;
    if (requires_dispatch_workgroup_size) {
      info->workgroup_size[0] = 1;
      info->workgroup_size[1] = 1;
      info->workgroup_size[2] = 1;
    } else {
      info->workgroup_size[0] = metadata_export->workgroup_size[0];
      info->workgroup_size[1] = metadata_export->workgroup_size[1];
      info->workgroup_size[2] = metadata_export->workgroup_size[2];
    }
  }
  executable->export_parameter_offsets[metadata->export_count] =
      metadata->parameter_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_resolve_metadata_kernel_args(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_host_size_t export_ordinal, hsa_agent_t any_device_agent,
    iree_hal_amdgpu_device_kernel_args_t* out_host_kernel_args) {
  const iree_hal_amdgpu_executable_export_t* metadata_export =
      &metadata->exports[export_ordinal];
  const iree_hal_amdgpu_executable_reflection_t* reflection =
      &metadata->reflection[export_ordinal];
  iree_string_view_t symbol_name = reflection->symbol_name;

  hsa_executable_symbol_t symbol = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
          libhsa, executable, symbol_name, any_device_agent, &symbol),
      "looking up HSA symbol for raw kernel `%.*s`", (int)symbol_name.size,
      symbol_name.data);

  uint32_t workgroup_size[3] = {1, 1, 1};
  if (!iree_any_bit_set(
          metadata_export->flags,
          IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE)) {
    workgroup_size[0] = metadata_export->workgroup_size[0];
    workgroup_size[1] = metadata_export->workgroup_size[1];
    workgroup_size[2] = metadata_export->workgroup_size[2];
  }
  const iree_hal_amdgpu_kernarg_layout_t* layout = NULL;
  if (!iree_any_bit_set(
          metadata_export->flags,
          IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_metadata_resolve_layout(
            metadata, metadata_export->kernarg_layout, &layout),
        "resolving dispatch kernarg layout for export %" PRIhsz,
        export_ordinal);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_resolve_kernel_args_from_symbol(
          libhsa, symbol, workgroup_size, out_host_kernel_args),
      "resolving kernel args for raw kernel `%.*s`", (int)symbol_name.size,
      symbol_name.data);

  if (layout) {
    out_host_kernel_args->kernarg_size = iree_max(
        out_host_kernel_args->kernarg_size, layout->kernarg_byte_length);
    out_host_kernel_args->kernarg_alignment = iree_max(
        out_host_kernel_args->kernarg_alignment, layout->kernarg_alignment);
  }
  memcpy(out_host_kernel_args->workgroup_cluster_size,
         metadata_export->workgroup_cluster_size,
         sizeof(out_host_kernel_args->workgroup_cluster_size));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_initialize_variant_global_table(
    iree_hal_amdgpu_executable_t* executable,
    iree_hal_amdgpu_executable_load_variant_t* load_variant,
    iree_allocator_t host_allocator) {
  load_variant->global_resolver =
      (iree_hal_amdgpu_executable_global_resolver_t){
          .host_allocator = host_allocator,
          .queue_affinity_domain = executable->queue_affinity_domain,
          .loaded_physical_device_mask =
              executable->loaded_physical_device_mask,
          .libhsa = executable->libhsa,
          .device = executable->device,
          .executable = load_variant->handle,
          .device_agent_count = executable->device_count,
          .device_agents = executable->device_agents,
      };
  return iree_hal_amdgpu_executable_global_resolver_initialize_table(
      &load_variant->global_resolver, &load_variant->global_table);
}

static iree_status_t iree_hal_amdgpu_executable_load_variant(
    iree_hal_amdgpu_executable_t* executable,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t physical_queue_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_executable_load_variant_t* out_load_variant) {
  out_load_variant->physical_queue_ordinal = physical_queue_ordinal;
  iree_status_t status = iree_hal_amdgpu_executable_load_module(
      executable->libhsa, topology, physical_devices, load_params,
      executable->code_object_reader, &out_load_variant->handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_initialize_variant_global_table(
        executable, out_load_variant, host_allocator);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_create_from_raw_hsaco(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_load_params_t* load_params,
    uint64_t executable_id, const iree_hal_amdgpu_device_limits_t* limits,
    const iree_hal_amdgpu_feedback_state_t* feedback_state,
    const iree_hal_amdgpu_asan_state_t* asan_state,
    const iree_hal_amdgpu_tsan_state_t* tsan_state,
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_device_list,
    iree_host_size_t queue_scope_count,
    const iree_hal_amdgpu_queue_scope_t* queue_scopes,
    hsa_agent_t any_device_agent, iree_hal_amdgpu_gfxip_version_t gfxip_version,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  if (IREE_UNLIKELY(executable_id == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU executable id is required");
  }

  iree_const_byte_span_t code_object_data = load_params->executable_data;
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {0};
  iree_status_t status = iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      code_object_data, host_allocator, &hsaco_metadata);

  iree_hal_amdgpu_executable_metadata_counts_t metadata_counts;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
        &hsaco_metadata, &metadata_counts);
  }

  iree_hal_amdgpu_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t load_variant_capacity = 1;
    if (iree_hal_amdgpu_tsan_state_is_enabled(tsan_state) &&
        topology->gpu_agent_queue_count > 1) {
      load_variant_capacity = topology->gpu_agent_queue_count;
    }
    status = iree_hal_amdgpu_executable_create_empty(
        device, libhsa, topology, physical_devices,
        metadata_counts.export_count, load_variant_capacity, queue_scope_count,
        queue_scopes, host_allocator, &executable);
  }
  if (iree_status_is_ok(status)) {
    executable->executable_id = executable_id;
    iree_hal_amdgpu_profile_metadata_hash_code_object(
        code_object_data, executable->code_object_hash);
    iree_hal_amdgpu_source_context_initialize(
        executable->executable_id, executable->code_object_hash,
        executable->device_count, executable->loaded_physical_device_mask,
        executable->loaded_code_object_ranges, &executable->source_context);
  }

  // HSA requires the source memory to outlive its code-object reader and the
  // reader to outlive every executable loaded from it. HAL load inputs are
  // call-scoped, so retain one exact copy and reader for all load variants.
  if (iree_status_is_ok(status)) {
    void* owned_code_object_data = NULL;
    status = iree_allocator_clone(host_allocator, code_object_data,
                                  &owned_code_object_data);
    if (iree_status_is_ok(status)) {
      executable->code_object_data = iree_make_byte_span(
          owned_code_object_data, code_object_data.data_length);
      status = iree_hsa_code_object_reader_create_from_memory(
          IREE_LIBHSA(libhsa), executable->code_object_data.data,
          executable->code_object_data.data_length,
          &executable->code_object_reader);
    }
  }

  // Load executable and register it with selected GPU agents.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_load_variant(
        executable, topology, physical_devices, load_params,
        /*physical_queue_ordinal=*/0, host_allocator,
        iree_hal_amdgpu_executable_primary_variant(executable));
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_executable_query_loaded_code_object_ranges(executable);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_tsan_state_is_enabled(tsan_state)) {
    iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
    bool found = false;
    status = iree_hal_amdgpu_global_table_try_lookup(
        &iree_hal_amdgpu_executable_primary_variant(executable)->global_table,
        iree_make_cstring_view(IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME), &found,
        &global);
    if (iree_status_is_ok(status) && found) {
      executable->requires_queue_scope = true;
      if (IREE_UNLIKELY(executable->queue_scope_count == 0)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU TSAN executable requires initialized queue scopes");
      }
      const iree_host_size_t required_load_variant_count =
          executable->load_variant_capacity;
      for (iree_host_size_t physical_queue_ordinal = 1;
           physical_queue_ordinal < required_load_variant_count &&
           iree_status_is_ok(status);
           ++physical_queue_ordinal) {
        status = iree_hal_amdgpu_executable_load_variant(
            executable, topology, physical_devices, load_params,
            physical_queue_ordinal, host_allocator,
            &executable->load_variants[physical_queue_ordinal]);
        if (iree_status_is_ok(status)) {
          executable->load_variant_count = physical_queue_ordinal + 1;
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_try_attach_sanitizer_site_table(
        executable, physical_devices->first_physical_device_ordinal);
  }
  for (iree_host_size_t variant_ordinal = 0;
       variant_ordinal < executable->load_variant_count &&
       iree_status_is_ok(status);
       ++variant_ordinal) {
    iree_hal_amdgpu_executable_load_variant_t* load_variant =
        &executable->load_variants[variant_ordinal];
    status = iree_hal_amdgpu_executable_publish_asan_config(
        executable, load_variant, asan_state);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_executable_publish_tsan_config(
          executable, load_variant, tsan_state);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_executable_publish_feedback_config(
          executable, load_variant, feedback_state);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_create_metadata_from_hsaco(
        libhsa, iree_hal_amdgpu_executable_primary_variant(executable)->handle,
        any_device_agent, &hsaco_metadata, host_allocator,
        &executable->metadata);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_validate_workgroup_cluster_sizes(
        executable->metadata, physical_devices, physical_device_count,
        physical_device_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_initialize_export_infos(
        executable->metadata, limits, executable);
  }

  // Resolve kernel args for each export.
  // These parameters should be the same for all devices as we require all
  // devices have the same ISA. The only thing that will differ is the
  // kernel_object pointer and we handle that per-device during table upload.
  if (iree_status_is_ok(status)) {
    iree_host_size_t* custom_explicit_kernarg_sizes = NULL;
    uint16_t* custom_implicit_args_offsets = NULL;
    if (executable->kernel_count > 0) {
      custom_explicit_kernarg_sizes = (iree_host_size_t*)iree_alloca(
          executable->kernel_count * sizeof(custom_explicit_kernarg_sizes[0]));
      custom_implicit_args_offsets = (uint16_t*)iree_alloca(
          executable->kernel_count * sizeof(custom_implicit_args_offsets[0]));
    }
    for (iree_host_size_t kernel_ordinal = 0;
         iree_status_is_ok(status) && kernel_ordinal < executable->kernel_count;
         ++kernel_ordinal) {
      iree_hal_amdgpu_device_kernel_args_t* host_kernel_args =
          &executable->host_kernel_args[kernel_ordinal];
      status = iree_hal_amdgpu_executable_resolve_metadata_kernel_args(
          libhsa,
          iree_hal_amdgpu_executable_primary_variant(executable)->handle,
          executable->metadata, kernel_ordinal, any_device_agent,
          host_kernel_args);
      if (iree_status_is_ok(status)) {
        if (executable->custom_direct_only_exports[kernel_ordinal]) {
          custom_explicit_kernarg_sizes[kernel_ordinal] =
              host_kernel_args->kernarg_size;
          custom_implicit_args_offsets[kernel_ordinal] = UINT16_MAX;
        } else {
          custom_explicit_kernarg_sizes[kernel_ordinal] = 0;
          custom_implicit_args_offsets[kernel_ordinal] = UINT16_MAX;
        }
      }
    }

    // Upload copies of kernel arguments for each loaded variant and device.
    // We reuse the host storage we already allocated to make it possible to
    // memcpy the entire table in one go from host memory.
    for (iree_host_size_t variant_ordinal = 0;
         variant_ordinal < executable->load_variant_count &&
         iree_status_is_ok(status);
         ++variant_ordinal) {
      iree_hal_amdgpu_executable_load_variant_t* load_variant =
          &executable->load_variants[variant_ordinal];
      for (iree_host_size_t device_ordinal = 0;
           iree_status_is_ok(status) &&
           device_ordinal < executable->device_count;
           ++device_ordinal) {
        if (!iree_hal_amdgpu_physical_device_mask_contains(
                executable->loaded_physical_device_mask, device_ordinal)) {
          continue;
        }
        const iree_host_size_t variant_device_ordinal =
            iree_hal_amdgpu_executable_variant_device_ordinal(
                executable, variant_ordinal, device_ordinal);
        status = iree_hal_amdgpu_executable_upload_kernel_table(
            libhsa, load_variant->handle, executable->metadata,
            executable->host_kernel_args, topology->gpu_agents[device_ordinal],
            &executable->device_kernel_args[variant_device_ordinal]);
        if (iree_status_is_ok(status)) {
          status =
              iree_hal_amdgpu_executable_initialize_dispatch_descriptors_for_device(
                  executable, gfxip_version, variant_ordinal, device_ordinal,
                  &physical_device_list[device_ordinal]->workgroup_cluster,
                  executable->metadata, custom_explicit_kernarg_sizes,
                  custom_implicit_args_offsets);
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_register_profile_metadata(
        libhsa, topology, profile_metadata, code_object_data, executable);
  }

  // Invalidate the kernel object pointer in all host args so that we don't
  // accidentally use it instead of the device-specific one.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_executable_invalidate_host_kernel_objects(executable);
  }

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&hsaco_metadata);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_executable_create(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    uint64_t executable_id, iree_hal_amdgpu_feedback_state_t* feedback_state,
    iree_hal_amdgpu_asan_state_t* asan_state,
    iree_hal_amdgpu_tsan_state_t* tsan_state,
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_device_list,
    iree_host_size_t queue_scope_count,
    const iree_hal_amdgpu_queue_scope_t* queue_scopes,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(profile_metadata);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(executable_id == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "AMDGPU executable id is required"));
  }
  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "topology must have at least one GPU device"));
  }
  if (IREE_UNLIKELY(physical_device_count != topology->gpu_agent_count ||
                    !physical_device_list)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "physical device list count %" PRIhsz
                             " must match topology GPU agent count %" PRIhsz,
                             physical_device_count, topology->gpu_agent_count));
  }
  if (IREE_UNLIKELY(queue_scope_count != 0 && !queue_scopes)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "queue scopes are required when count is nonzero"));
  }

  // Resolve the executable queue affinity to the physical devices that need
  // code-object loads and per-device kernel tables.
  iree_hal_amdgpu_queue_affinity_physical_device_set_t physical_devices;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_select_physical_devices(
              topology, queue_affinity, &physical_devices));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(iree_hal_device_spec(device));
  iree_hal_physical_device_affinity_t resolved_physical_device_affinity = 0;
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            physical_devices.physical_device_mask, i)) {
      continue;
    }
    if (IREE_UNLIKELY(i >= identity->physical_device_count)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "AMDGPU topology physical device ordinal %" PRIhsz
                           " is absent from the immutable device spec",
                           i));
    }
    resolved_physical_device_affinity |=
        identity->physical_devices[i].physical_device_affinity;
  }
  if (!iree_all_bits_set(target->physical_device_affinity,
                         resolved_physical_device_affinity)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "AMDGPU target `%.*s` physical-device affinity 0x%016" PRIx64
                " does not cover the queue-selected affinity 0x%016" PRIx64,
                (int)target->target_key.size, target->target_key.data,
                target->physical_device_affinity,
                resolved_physical_device_affinity));
  }

  if (!iree_string_view_equal(target->family, IREE_SV("amdgpu"))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "executable target family `%.*s` is not AMDGPU",
                             (int)target->family.size, target->family.data));
  }

  // Pick a selected device to be our template for device queries. All devices
  // in the topology have the same immutable target identity.
  const iree_host_size_t template_device_ordinal =
      physical_devices.first_physical_device_ordinal;
  hsa_agent_t any_device_agent = topology->gpu_agents[template_device_ordinal];
  const iree_hal_amdgpu_agent_target_t* agent_target =
      physical_device_list[template_device_ordinal]->agent_target;

  // Resolve the selected target to the compatible HSA ISA used for device
  // limits and executable metadata.
  const iree_hal_amdgpu_agent_isa_target_t* isa_target = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_resolve_isa_target(
              agent_target, target->target_key, &isa_target));
  if (!isa_target) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "executable target `%.*s` is not supported by the "
                             "devices in the topology",
                             (int)target->target_key.size,
                             target->target_key.data));
  }

  iree_hal_amdgpu_device_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_device_limits(libhsa, any_device_agent,
                                              isa_target->isa, &limits));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_preflight_code_object(
              &physical_devices, load_params->executable_data,
              physical_device_count, physical_device_list));
  const iree_hal_amdgpu_gfxip_version_t gfxip_version =
      isa_target->identity.version;

  iree_status_t status = iree_hal_amdgpu_executable_create_from_raw_hsaco(
      device, libhsa, topology, &physical_devices, load_params, executable_id,
      &limits, feedback_state, asan_state, tsan_state, physical_device_count,
      physical_device_list, queue_scope_count, queue_scopes, any_device_agent,
      gfxip_version, profile_metadata, host_allocator, out_executable);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_release(*out_executable);
    *out_executable = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

uint64_t iree_hal_amdgpu_executable_id(iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return executable->executable_id;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_executable_query_function_attributes(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    uint32_t* out_maximum_workgroup_invocations,
    uint32_t* out_workgroup_local_memory_size,
    uint32_t* out_private_memory_size, uint32_t* out_register_count,
    uint32_t* out_maximum_dynamic_workgroup_local_memory_size,
    bool* out_requires_uniform_workgroups) {
  IREE_ASSERT_ARGUMENT(base_executable);
  IREE_ASSERT_ARGUMENT(out_maximum_workgroup_invocations);
  IREE_ASSERT_ARGUMENT(out_workgroup_local_memory_size);
  IREE_ASSERT_ARGUMENT(out_private_memory_size);
  IREE_ASSERT_ARGUMENT(out_register_count);
  IREE_ASSERT_ARGUMENT(out_maximum_dynamic_workgroup_local_memory_size);
  IREE_ASSERT_ARGUMENT(out_requires_uniform_workgroups);
  *out_maximum_workgroup_invocations = 0;
  *out_workgroup_local_memory_size = 0;
  *out_private_memory_size = 0;
  *out_register_count = 0;
  *out_maximum_dynamic_workgroup_local_memory_size = 0;
  *out_requires_uniform_workgroups = false;

  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  const iree_hal_amdgpu_executable_export_t* metadata_export =
      &executable->metadata->exports[export_ordinal];
  *out_maximum_workgroup_invocations = metadata_export->max_workgroup_size;
  *out_workgroup_local_memory_size = metadata_export->fixed_group_segment_size;
  *out_private_memory_size = metadata_export->fixed_private_segment_size;
  *out_register_count = metadata_export->vgpr_count;
  *out_maximum_dynamic_workgroup_local_memory_size =
      metadata_export->max_dynamic_workgroup_local_memory;
  *out_requires_uniform_workgroups = iree_all_bits_set(
      metadata_export->flags,
      IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_UNIFORM_WORKGROUPS);
  return iree_ok_status();
}

static void iree_hal_amdgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t variant_ordinal = 0;
       variant_ordinal < executable->load_variant_count; ++variant_ordinal) {
    for (iree_host_size_t device_ordinal = 0;
         device_ordinal < executable->device_count; ++device_ordinal) {
      const iree_host_size_t variant_device_ordinal =
          iree_hal_amdgpu_executable_variant_device_ordinal(
              executable, variant_ordinal, device_ordinal);
      void* kernel_args =
          (void*)executable->device_kernel_args[variant_device_ordinal];
      if (kernel_args) {
        iree_hal_amdgpu_hsa_cleanup_assert_success(
            iree_hsa_amd_memory_pool_free_raw(executable->libhsa, kernel_args));
      }
    }
  }

  for (iree_host_size_t variant_ordinal = 0;
       variant_ordinal < executable->load_variant_count; ++variant_ordinal) {
    iree_hal_amdgpu_executable_load_variant_t* load_variant =
        &executable->load_variants[variant_ordinal];
    iree_hal_amdgpu_global_table_deinitialize(&load_variant->global_table);
    if (load_variant->handle.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_executable_destroy_raw(executable->libhsa,
                                          load_variant->handle));
    }
  }
  if (executable->code_object_reader.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_code_object_reader_destroy_raw(
            executable->libhsa, executable->code_object_reader));
  }
  iree_hal_amdgpu_executable_metadata_free(executable->metadata);
  iree_allocator_free(host_allocator, executable->code_object_data.data);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);

  *out_kernel_args = &executable->host_kernel_args[export_ordinal];

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  } else if (IREE_UNLIKELY(device_ordinal >= executable->device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device ordinal %" PRIhsz
                            " out of range; executable topology has %" PRIhsz
                            " physical devices",
                            device_ordinal, executable->device_count);
  } else if (IREE_UNLIKELY(!iree_hal_amdgpu_physical_device_mask_contains(
                 executable->loaded_physical_device_mask, device_ordinal))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device ordinal %" PRIhsz
                            " is not in executable queue affinity 0x%" PRIx64,
                            device_ordinal, executable->queue_affinity);
  } else if (IREE_UNLIKELY(executable->requires_queue_scope)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable requires queue-scoped kernel args lookup");
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);

  const iree_host_size_t variant_device_ordinal =
      iree_hal_amdgpu_executable_variant_device_ordinal(
          executable, /*variant_ordinal=*/0, device_ordinal);
  *out_kernel_args =
      &executable->device_kernel_args[variant_device_ordinal][export_ordinal];

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_descriptor = NULL;

  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  } else if (IREE_UNLIKELY(device_ordinal >= executable->device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device ordinal %" PRIhsz
                            " out of range; executable topology has %" PRIhsz
                            " physical devices",
                            device_ordinal, executable->device_count);
  } else if (IREE_UNLIKELY(!iree_hal_amdgpu_physical_device_mask_contains(
                 executable->loaded_physical_device_mask, device_ordinal))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device ordinal %" PRIhsz
                            " is not in executable queue affinity 0x%" PRIx64,
                            device_ordinal, executable->queue_affinity);
  } else if (IREE_UNLIKELY(executable->requires_queue_scope)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable requires queue-scoped dispatch descriptor lookup");
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);

  const iree_host_size_t descriptor_ordinal =
      iree_hal_amdgpu_executable_descriptor_ordinal(
          executable, /*variant_ordinal=*/0, device_ordinal, export_ordinal);
  *out_descriptor = &executable->host_dispatch_descriptors[descriptor_ordinal];
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_descriptor = NULL;

  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  }

  iree_hal_amdgpu_queue_affinity_resolved_t resolved = {0};
  iree_host_size_t variant_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_select_load_variant_for_queue(
      executable, queue_affinity, &resolved, &variant_ordinal));

  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  const iree_host_size_t descriptor_ordinal =
      iree_hal_amdgpu_executable_descriptor_ordinal(
          executable, variant_ordinal, resolved.physical_device_ordinal,
          export_ordinal);
  *out_descriptor = &executable->host_dispatch_descriptors[descriptor_ordinal];
  return iree_ok_status();
}

bool iree_hal_amdgpu_executable_requires_queue_scope(
    iree_hal_executable_t* base_executable) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  return executable->requires_queue_scope;
}

iree_status_t
iree_hal_amdgpu_executable_lookup_pm4_dispatch_launch_state_for_device(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* dispatch_descriptor =
      NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          base_executable, function, device_ordinal, &dispatch_descriptor));
  if (IREE_UNLIKELY(!dispatch_descriptor->pm4_launch_state_valid)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "function id %" PRIu64 " on device ordinal %" PRIhsz
                            " does not have PM4 dispatch launch metadata",
                            function.value, device_ordinal);
  }

  *out_state = &dispatch_descriptor->pm4_launch_state;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return executable->kernel_count;
}

static iree_status_t iree_hal_amdgpu_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  memset(out_info, 0, sizeof(*out_info));
  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  *out_info = executable->export_infos[export_ordinal];
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  if (IREE_UNLIKELY(!iree_hal_executable_function_is_index_in_range(
          function, executable->kernel_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->kernel_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  const iree_host_size_t parameter_begin =
      executable->export_parameter_offsets[export_ordinal];
  const iree_host_size_t parameter_end =
      executable->export_parameter_offsets[export_ordinal + 1];
  const iree_host_size_t parameter_count = parameter_end - parameter_begin;
  const iree_host_size_t copy_count = iree_min(capacity, parameter_count);
  if (copy_count > 0) {
    memcpy(out_parameters, &executable->export_parameters[parameter_begin],
           copy_count * sizeof(out_parameters[0]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->kernel_count; ++i) {
    iree_string_view_t export_name = executable->export_infos[i].name;
    if (iree_string_view_equal(export_name, name)) {
      *out_export_ordinal =
          iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_amdgpu_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return iree_hal_amdgpu_global_table_try_lookup(
      &iree_hal_amdgpu_executable_primary_variant(executable)->global_table,
      name, out_found, out_global);
}

static iree_status_t iree_hal_amdgpu_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return iree_hal_amdgpu_global_table_info(
      &iree_hal_amdgpu_executable_primary_variant(executable)->global_table,
      global, out_info);
}

static iree_status_t iree_hal_amdgpu_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  iree_hal_amdgpu_executable_load_variant_t* load_variant =
      iree_hal_amdgpu_executable_primary_variant(executable);
  iree_hal_executable_global_t selected_global = global;
  if (executable->requires_queue_scope) {
    iree_hal_executable_global_info_t info;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_info(
        &load_variant->global_table, global, &info));

    iree_hal_amdgpu_queue_affinity_resolved_t resolved = {0};
    iree_host_size_t variant_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_select_load_variant_for_queue(
            executable, queue_affinity, &resolved, &variant_ordinal));
    load_variant = &executable->load_variants[variant_ordinal];

    bool found = false;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_try_lookup(
        &load_variant->global_table, info.name, &found, &selected_global));
    if (IREE_UNLIKELY(!found)) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "queue-scoped AMDGPU executable global `%.*s` not found",
          (int)info.name.size, info.name.data);
    }
  }
  return iree_hal_amdgpu_global_table_buffer(
      &load_variant->global_table, selected_global, queue_affinity, out_buffer);
}

static const iree_hal_executable_vtable_t iree_hal_amdgpu_executable_vtable = {
    .destroy = iree_hal_amdgpu_executable_destroy,
    .function_count = iree_hal_amdgpu_executable_export_count,
    .function_info = iree_hal_amdgpu_executable_export_info,
    .function_parameters = iree_hal_amdgpu_executable_export_parameters,
    .lookup_function_by_name = iree_hal_amdgpu_executable_lookup_export_by_name,
    .try_lookup_global_by_name =
        iree_hal_amdgpu_executable_try_lookup_global_by_name,
    .global_info = iree_hal_amdgpu_executable_global_info,
    .global_buffer = iree_hal_amdgpu_executable_global_buffer,
};
