// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library.h"

#include "iree/hal/drivers/amdgpu/device/binaries/toc.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/util/device_library_target.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_library_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_file_toc_append_names_to_builder(
    const iree_file_toc_t* file_toc, size_t file_count,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < file_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, file_toc[i].name));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_device_library_query_agent_profile(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_profile_t* out_profile,
    hsa_default_float_rounding_mode_t* out_rounding_mode) {
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent, HSA_AGENT_INFO_PROFILE, out_profile));
  return iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                                 HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                                 out_rounding_mode);
}

static iree_status_t iree_hal_amdgpu_device_library_select_agent_profile(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, hsa_profile_t* out_profile,
    hsa_default_float_rounding_mode_t* out_rounding_mode) {
  if (topology->gpu_agent_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no GPU agents available for device library load");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_library_query_agent_profile(
      libhsa, topology->gpu_agents[0], out_profile, out_rounding_mode));
  for (iree_host_size_t i = 1; i < topology->gpu_agent_count; ++i) {
    hsa_profile_t profile = HSA_PROFILE_BASE;
    hsa_default_float_rounding_mode_t rounding_mode =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_library_query_agent_profile(
        libhsa, topology->gpu_agents[i], &profile, &rounding_mode));
    if (profile != *out_profile || rounding_mode != *out_rounding_mode) {
      return iree_make_status(
          IREE_STATUS_INCOMPATIBLE,
          "GPU agents must have matching HSA executable profiles and default "
          "rounding modes; agent[%" PRIhsz
          "] has profile %d and rounding mode %d but the device library "
          "requires profile %d and rounding mode %d",
          i, (int)profile, (int)rounding_mode, (int)*out_profile,
          (int)*out_rounding_mode);
    }
  }
  return iree_ok_status();
}

static const iree_file_toc_t* iree_hal_amdgpu_device_library_find_file_for_arch(
    iree_string_view_t arch) {
  const iree_string_view_t isa_prefix = IREE_SVL("amdgcn-amd-amdhsa--");
  for (iree_host_size_t i = 0; i < iree_hal_amdgpu_device_binaries_size();
       ++i) {
    const iree_file_toc_t* file_toc =
        &iree_hal_amdgpu_device_binaries_create()[i];
    iree_string_view_t file_name = iree_make_cstring_view(file_toc->name);
    if (!iree_string_view_starts_with(file_name, isa_prefix)) continue;
    iree_string_view_t file_arch = iree_string_view_substr(
        file_name, isa_prefix.size, IREE_STRING_VIEW_NPOS);
    if (iree_hal_amdgpu_device_library_target_matches_file_arch(file_arch,
                                                                arch)) {
      return file_toc;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_amdgpu_device_library_find_file_for_target(
    const iree_hal_amdgpu_target_identity_t* physical_identity,
    const iree_hal_amdgpu_target_identity_t* isa_identity,
    const iree_file_toc_t** out_file_toc) {
  *out_file_toc = NULL;
  iree_hal_amdgpu_device_library_target_candidate_list_t candidates = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_candidates_from_agent_isa(
          physical_identity, isa_identity, &candidates));
  for (iree_host_size_t i = 0; i < candidates.count; ++i) {
    const iree_file_toc_t* file_toc =
        iree_hal_amdgpu_device_library_find_file_for_arch(
            candidates.values[i].value);
    if (file_toc) {
      *out_file_toc = file_toc;
      break;
    }
  }
  return iree_ok_status();
}

// Selects a device library binary file for a physical agent target set.
static iree_status_t iree_hal_amdgpu_device_library_select_file(
    const iree_hal_amdgpu_agent_target_t* agent_target,
    iree_allocator_t host_allocator, const iree_file_toc_t** out_file_toc) {
  IREE_ASSERT_ARGUMENT(agent_target);
  IREE_ASSERT_ARGUMENT(out_file_toc);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_file_toc = NULL;

  const iree_file_toc_t* best_file_toc = NULL;
  for (iree_host_size_t i = 0; i < agent_target->isa_count && !best_file_toc;
       ++i) {
    const iree_hal_amdgpu_agent_isa_target_t* isa_target =
        iree_hal_amdgpu_agent_target_isa_at(agent_target, i);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_device_library_find_file_for_target(
                &agent_target->primary_isa.identity, &isa_target->identity,
                &best_file_toc));
  }

  // If we found a matching file return that for loading. It should work but is
  // not guaranteed until HSA accepts and freezes the embedded code object.
  iree_status_t status = iree_ok_status();
  if (best_file_toc) {
    *out_file_toc = best_file_toc;
    IREE_TRACE_ZONE_APPEND_TEXT(z0, best_file_toc->name);
  } else {
    char target_id[128] = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_target_identity_format_artifact_key(
                &agent_target->primary_isa.identity, sizeof(target_id),
                target_id, /*out_buffer_length=*/NULL));
    status =
        iree_make_status(IREE_STATUS_INCOMPATIBLE,
                         "no device library binary found for any of %" PRIhsz
                         " GPU agent ISA targets (primary `%s`)",
                         agent_target->isa_count, target_id);
#if IREE_STATUS_MODE >= 2
    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator, &builder);
    iree_status_t annotation_status = iree_string_builder_append_string(
        &builder, IREE_SV("available in runtime build: ["));
    if (iree_status_is_ok(annotation_status)) {
      annotation_status = iree_file_toc_append_names_to_builder(
          iree_hal_amdgpu_device_binaries_create(),
          iree_hal_amdgpu_device_binaries_size(), &builder);
    }
    if (iree_status_is_ok(annotation_status)) {
      annotation_status =
          iree_string_builder_append_string(&builder, IREE_SV("]"));
    }
    if (iree_status_is_ok(annotation_status)) {
      status = iree_status_annotate_f(status, "%.*s",
                                      (int)iree_string_builder_size(&builder),
                                      iree_string_builder_buffer(&builder));
    } else {
      status = iree_status_join(status, annotation_status);
    }
    iree_string_builder_deinitialize(&builder);
#endif  // IREE_STATUS_MODE >= 2
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_device_library_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_agent_target_t* gpu_agent_targets,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_device_library_t* out_library) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(gpu_agent_targets);
  IREE_ASSERT_ARGUMENT(out_library);

  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "topology must have at least one GPU agent");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_library, 0, sizeof(*out_library));
  out_library->libhsa = libhsa;

  // Select (or try to) the binary file for the leading GPU agent.
  // Today we require a single device ISA for all devices as heterogeneous
  // multi-device HAL usage is expected for different devices.
  const iree_file_toc_t* file_toc = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_device_library_select_file(
              &gpu_agent_targets[0], host_allocator, &file_toc));

  // TODO(benvanik): figure out what options we could pass? Documentation is ...
  // lacking. These may have only been used for HSAIL anyway.
  const char* options = NULL;

  // Bind a code object reader to the memory sourced from our rodata.
  iree_status_t status = iree_hsa_code_object_reader_create_from_memory(
      IREE_LIBHSA(libhsa), file_toc->data, file_toc->size,
      &out_library->code_object_reader);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Create the executable that will hold all of the loaded code objects.
  hsa_profile_t executable_profile = HSA_PROFILE_BASE;
  hsa_default_float_rounding_mode_t executable_rounding_mode =
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
  status = iree_hal_amdgpu_device_library_select_agent_profile(
      libhsa, topology, &executable_profile, &executable_rounding_mode);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_create_alt(
        IREE_LIBHSA(libhsa), executable_profile, executable_rounding_mode,
        options, &out_library->executable);
  }

  // Load the code object for each agent.
  // Note that we could save off the loaded_code_object per-agent here but then
  // we'd need big fixed storage or dynamically allocated storage - instead we
  // take the hit of doing the n^2 resolve because it's only done once per
  // HAL device initialization. Everything that needs the information from the
  // loaded_code_objects caches the results.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
      status = iree_hsa_executable_load_agent_code_object(
          IREE_LIBHSA(libhsa), out_library->executable, topology->gpu_agents[i],
          out_library->code_object_reader, options, NULL);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Freeze the executable now that loading has completed. Most queries require
  // that the executable be frozen.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_freeze(IREE_LIBHSA(libhsa),
                                        out_library->executable, options);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_device_library_deinitialize(out_library);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_device_library_deinitialize(
    iree_hal_amdgpu_device_library_t* library) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (library->executable.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_executable_destroy_raw(library->libhsa, library->executable));
  }
  if (library->code_object_reader.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_code_object_reader_destroy_raw(library->libhsa,
                                                library->code_object_reader));
  }

  memset(library, 0, sizeof(*library));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t
iree_hal_amdgpu_device_library_populate_agent_loaded_code_object_range(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_range);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_range, 0, sizeof(*out_range));

  iree_status_t status = iree_hal_amdgpu_loaded_code_object_query_agent_range(
      library->libhsa, library->executable, device_agent, out_range);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_device_library_populate_kernel_args(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    const char* symbol_name, uint16_t workgroup_size_x,
    uint16_t workgroup_size_y, uint16_t workgroup_size_z,
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, symbol_name);

  const iree_hal_amdgpu_libhsa_t* libhsa = library->libhsa;

  // Lookup the symbol. The `.kd` suffix is required and should have been passed
  // by the caller.
  hsa_executable_symbol_t symbol = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_executable_get_symbol_by_name(IREE_LIBHSA(libhsa),
                                             library->executable, symbol_name,
                                             &device_agent, &symbol),
      "resolving `%s`", symbol_name);

  // All of our kernels assume 3 dimensions.
  out_kernel_args->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  // TODO(benvanik): embed this as a custom section or attributes that we could
  // somehow query? For now we hardcode and take directly. This may be fine as
  // we aren't doing anything but blits and probably don't need to tightly
  // optimize workgroup size across architectures. Unfortunately the
  // `reqd_work_group_size` attribute is exactly what we want but clang only
  // allows it on OpenCL kernels (not C ones). Reading it is a PITA (need to
  // crack open the ELF, find the AMDGPU notes section, and decode the msgpack)
  // so unless we absolutely need it alternatives (like extracting from shared
  // headers as part of a build step) may be better.
  out_kernel_args->workgroup_size[0] = workgroup_size_x;
  out_kernel_args->workgroup_size[1] = workgroup_size_y;
  out_kernel_args->workgroup_size[2] = workgroup_size_z;

  // The object pointer is used in dispatch packets from either the host or
  // device.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_executable_symbol_get_info(
          IREE_LIBHSA(libhsa), symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &out_kernel_args->kernel_object));

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

iree_status_t iree_hal_amdgpu_device_library_populate_agent_kernels(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_device_kernels_t* out_kernels) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_kernels);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_kernels, 0, sizeof(*out_kernels));

#define IREE_HAL_AMDGPU_DEVICE_KERNEL(name, workgroup_size_x,             \
                                      workgroup_size_y, workgroup_size_z) \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                      \
      z0,                                                                 \
      iree_hal_amdgpu_device_library_populate_kernel_args(                \
          library, device_agent, #name ".kd", workgroup_size_x,           \
          workgroup_size_y, workgroup_size_z, &out_kernels->name),        \
      #name);
#include "iree/hal/drivers/amdgpu/device/kernel_tables.h"
#undef IREE_HAL_AMDGPU_DEVICE_KERNEL

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
