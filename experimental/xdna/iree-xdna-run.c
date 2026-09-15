// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "experimental/xdna/amdf_status.h"
#include "experimental/xdna/executable.h"
#include "experimental/xdna/prepared_command.h"
#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, image, "", "Complete canonical .xdna executable file.");
IREE_FLAG(string, entry, "",
          "Exported entry name; an empty name selects entry ordinal zero.");
IREE_FLAG(int32_t, columns, 0,
          "Logical context column count required by the image, in [1, 8].");
IREE_FLAG(int32_t, device, 0, "XDNA endpoint ordinal in the native inventory.");
IREE_FLAG(int32_t, invocation_count, 1,
          "Number of checked invocations; the first activates the program and "
          "later invocations use control-only commands.");
IREE_FLAG(string, binding_memory, "system",
          "Backing for every binding: system or registered_host.");
IREE_FLAG_LIST(string, binding,
               "Initial raw buffer file, repeated in entry binding order. "
               "Each file length is the logical binding length, including "
               "output buffers initialized with sentinel data.");
IREE_FLAG_LIST(string, output,
               "Buffer to write after completion, as binding_ordinal=path. "
               "May be repeated for distinct bindings.");

// A native attachment and the HAL view borrowing its explicit host mapping.
typedef struct iree_xdna_run_binding_t {
  // Initial logical bytes, retained until cleanup even after native failure.
  iree_io_file_contents_t* initial_contents;
  // Optional output path borrowed from the parsed command line.
  iree_string_view_t output_path;
  // Aligned caller allocation retained while registered with the provider.
  void* registered_allocation;
  // Logical caller range within `registered_allocation`, or NULL for system
  // memory.
  void* registered_host_pointer;
  // Stable XDNA attachment to provider-owned or registered physical backing.
  amdf_memory_t* memory;
  // Explicit host view retained through prepared-command destruction.
  amdf_host_mapping_t* mapping;
  // Immutable properties of the host view.
  amdf_host_mapping_info_t mapping_info;
  // Logical HAL buffer borrowing the host view.
  iree_hal_buffer_t* buffer;
} iree_xdna_run_binding_t;

// One invocation's ownership tree. Cleanup stops at a failed native boundary.
typedef struct iree_xdna_run_t {
  // Allocator owning host metadata and executable storage.
  iree_allocator_t host_allocator;
  // Borrowed API table from the linked libamdf provider.
  const amdf_api_t* api;
  // Borrowed XDNA extension table from the same provider.
  const amdf_xdna_api_t* xdna_api;
  // Independent provider instance owning the native endpoint namespace.
  amdf_instance_t* instance;
  // Query endpoint selected before device materialization.
  amdf_endpoint_t* endpoint;
  // Explicit native consumer kept live through memory destruction.
  amdf_device_t* device;
  // Borrowed system storage descriptor owned by the instance.
  amdf_memory_scope_t* memory_scope;
  // Device access required by every binding.
  amdf_memory_device_access_t memory_access;
  // Selected binding construction and host-mapping profile.
  amdf_memory_profile_t memory_profile;
  // Allocation or caller-registration acquisition selected by --binding_memory.
  amdf_memory_profile_roles_t binding_memory_role;
  // Native scheduling and placement context borrowed by memory and queues.
  amdf_xdna_context_t* context;
  // Caller-owned cold instruction storage, released before its context.
  struct {
    // Endpoint-required alignment of submitted instruction addresses.
    uint32_t alignment;
    // One private allocation backing this executable instance.
    amdf_memory_t* memory;
    // Explicit writable view used only during cold instantiation.
    amdf_host_mapping_t* mapping;
  } instructions;
  // Executable-only family metadata; native queues are managed through libamdf.
  iree_hal_queue_family_spec_t queue_family_spec;
  // HAL family identity borrowed by the executable.
  iree_hal_queue_family_t queue_family;
  // Parsed and lowered executable retaining the immutable image bytes.
  iree_hal_executable_t* executable;
  // Number of dense input bindings and resolved command bindings.
  iree_host_size_t binding_count;
  // Allocation owning binding state and the trailing resolved-binding array.
  iree_xdna_run_binding_t* bindings;
  // Resolved bindings borrowed during prepared-command construction.
  iree_hal_amd_xdna_prepared_command_binding_t* prepared_bindings;
  // Prepared command retaining the executable and every HAL buffer.
  iree_hal_amd_xdna_prepared_command_t* prepared_command;
  // Exclusive native queue lease retained through accepted work retirement.
  amdf_kernel_queue_t* queue;
} iree_xdna_run_t;

static iree_status_t iree_xdna_run_select_binding_memory(iree_xdna_run_t* run) {
  if (strcmp(FLAG_binding_memory, "system") == 0) {
    run->binding_memory_role = AMDF_MEMORY_PROFILE_ROLE_CREATE;
    return iree_ok_status();
  }
  if (strcmp(FLAG_binding_memory, "registered_host") == 0) {
    run->binding_memory_role = AMDF_MEMORY_PROFILE_ROLE_REGISTER;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--binding_memory must be system or registered_host; got '%s'",
      FLAG_binding_memory);
}

static iree_status_t iree_xdna_run_load_image(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_byte_sequence_t** out_sequence) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(path, host_allocator, &contents));
  iree_byte_span_t span = iree_byte_span_empty();
  iree_byte_sequence_t* sequence = NULL;
  iree_status_t status = iree_allocator_clone(
      host_allocator, contents->const_buffer, (void**)&span.data);
  if (iree_status_is_ok(status)) {
    span.data_length = contents->buffer.data_length;
    status = iree_byte_sequence_create_from_span_move(&span, host_allocator,
                                                      &sequence);
  }
  iree_allocator_free(host_allocator, span.data);
  iree_io_file_contents_free(contents);
  if (iree_status_is_ok(status)) *out_sequence = sequence;
  return status;
}

static iree_status_t iree_xdna_run_load_bindings(iree_xdna_run_t* run) {
  const iree_flag_string_list_t inputs = FLAG_binding_list();
  const iree_flag_string_list_t outputs = FLAG_output_list();
  iree_host_size_t total_size = 0;
  iree_host_size_t prepared_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD(inputs.count, iree_xdna_run_binding_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(
          inputs.count, iree_hal_amd_xdna_prepared_command_binding_t,
          iree_alignof(iree_hal_amd_xdna_prepared_command_binding_t),
          &prepared_offset)));
  if (inputs.count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(run->host_allocator, total_size,
                                               (void**)&run->bindings));
    run->prepared_bindings =
        (iree_hal_amd_xdna_prepared_command_binding_t*)((uint8_t*)
                                                            run->bindings +
                                                        prepared_offset);
  }
  run->binding_count = inputs.count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < inputs.count;
       ++i) {
    status = iree_io_file_contents_read(inputs.values[i], run->host_allocator,
                                        &run->bindings[i].initial_contents);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < outputs.count;
       ++i) {
    iree_string_view_t ordinal_string;
    iree_string_view_t path;
    iree_string_view_split(outputs.values[i], '=', &ordinal_string, &path);
    uint32_t ordinal = 0;
    if (!iree_string_view_atoi_uint32(ordinal_string, &ordinal) ||
        ordinal >= run->binding_count || iree_string_view_is_empty(path)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "--output requires a valid binding_ordinal=path");
    } else if (!iree_string_view_is_empty(run->bindings[ordinal].output_path)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "binding %u has more than one output path", ordinal);
    } else {
      run->bindings[ordinal].output_path = path;
    }
  }
  return status;
}

static iree_status_t iree_xdna_run_select_memory_scope(iree_xdna_run_t* run) {
  uint32_t count = 0;
  const amdf_status_t count_status = run->api->instance_enumerate_memory_scopes(
      run->instance, 0, NULL, &count);
  if (count_status != amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        count_status, "instance_enumerate_memory_scopes(count)"));
  }
  iree_host_size_t scopes_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scopes_size, IREE_STRUCT_FIELD(count, amdf_memory_scope_t*, NULL)));
  amdf_memory_scope_t** scopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(run->host_allocator, scopes_size, (void**)&scopes));
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(run->api->instance_enumerate_memory_scopes(
                                        run->instance, count, scopes, &count),
                                    "instance_enumerate_memory_scopes");
  for (uint32_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->memory_scope_query_info(scopes[i], &info),
        "memory_scope_query_info");
    if (iree_status_is_ok(status) &&
        info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      run->memory_scope = scopes[i];
      break;
    }
  }
  iree_allocator_free(run->host_allocator, scopes);
  if (iree_status_is_ok(status) && run->memory_scope == NULL) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "no system memory scope");
  }
  return status;
}

static iree_status_t iree_xdna_run_open_endpoint(iree_xdna_run_t* run) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &run->api),
      "query_api"));
  const void* extension = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->query_extension(AMDF_EXTENSION_XDNA,
                                AMDF_XDNA_EXTENSION_VERSION_1,
                                AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
      "query_extension(XDNA)"));
  run->xdna_api = extension;
  const amdf_instance_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .structure_size = sizeof(create_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->instance_create(&create_info, &run->instance),
      "instance_create"));
  IREE_RETURN_IF_ERROR(iree_xdna_run_select_memory_scope(run));
  uint32_t count = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->endpoint_enumerate(run->instance, 0, NULL, &count),
      "endpoint_enumerate(count)"));
  if (count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "no native AMD endpoints");
  }
  iree_host_size_t summaries_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &summaries_size,
      IREE_STRUCT_FIELD(count, amdf_endpoint_summary_t, NULL)));
  amdf_endpoint_summary_t* summaries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      run->host_allocator, summaries_size, (void**)&summaries));
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->endpoint_enumerate(run->instance, count, summaries, &count),
      "endpoint_enumerate");
  uint32_t xdna_ordinal = 0;
  for (uint32_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
    if (xdna_ordinal++ != (uint32_t)FLAG_device) continue;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->endpoint_open(run->instance, &summaries[i].id,
                                &run->endpoint),
        "endpoint_open");
    break;
  }
  iree_allocator_free(run->host_allocator, summaries);
  if (iree_status_is_ok(status) && run->endpoint == NULL) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "XDNA endpoint ordinal %d is unavailable",
                              FLAG_device);
  }
  return status;
}

static iree_status_t iree_xdna_run_create_device(
    iree_xdna_run_t* run, iree_hal_amd_xdna_aie2p_target_t* out_target) {
  amdf_xdna_endpoint_info_t xdna_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
      .structure_size = sizeof(xdna_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->xdna_api->endpoint_query_info(run->endpoint, &xdna_info),
      "xdna.endpoint_query_info"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      iree_make_cstring_view(xdna_info.target_id), (uint16_t)FLAG_columns,
      out_target));
  if (xdna_info.instruction.maximum_byte_length == 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "endpoint does not support native instruction submission");
  }
  run->instructions.alignment = xdna_info.instruction.address_alignment;
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->endpoint_query_info(run->endpoint, &endpoint_info),
      "endpoint_query_info"));
  uint32_t family_ordinal = UINT32_MAX;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < endpoint_info.queue_family_count; ++i) {
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->endpoint_query_queue_family_info(run->endpoint, i, &family),
        "endpoint_query_queue_family_info");
    if (iree_status_is_ok(status) &&
        family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0) {
      family_ordinal = i;
      break;
    }
  }
  if (!iree_status_is_ok(status)) return status;
  if (family_ordinal == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "endpoint has no XDNA kernel queue family");
  }
  run->queue_family_spec = (iree_hal_queue_family_spec_t){
      .name = IREE_SV("xdna"),
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  iree_hal_queue_family_initialize(family_ordinal, &run->queue_family_spec,
                                   &run->queue_family);
  const amdf_xdna_device_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create_info),
  };
  fprintf(stderr, "Creating %s device and %d-column context\n",
          xdna_info.target_id, FLAG_columns);
  fflush(stderr);
  run->memory_profile = (amdf_memory_profile_t){
      .ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN,
  };
  const amdf_memory_profile_roles_t required_roles =
      run->binding_memory_role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
  const amdf_memory_flags_t required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  run->memory_access.requirements = (amdf_memory_access_requirements_t){
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA,
  };
  const amdf_memory_endpoint_access_t endpoint_access = {
      .endpoint = run->endpoint,
      .requirements = run->memory_access.requirements,
  };
  for (uint32_t ordinal = 0;; ++ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t query_status = run->api->memory_scope_query_profile(
        run->memory_scope, ordinal, 1, &endpoint_access, &profile,
        &capabilities);
    if (amdf_status_code(query_status) == AMDF_STATUS_CODE_OUT_OF_RANGE) break;
    if (query_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
      continue;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        query_status, "memory_scope_query_profile"));
    if ((profile.roles & required_roles) == required_roles &&
        (required_flags & ~profile.supported_flags) == 0) {
      run->memory_profile = profile;
      break;
    }
  }
  if (run->memory_profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "device has no host-visible %s profile with read/write device access",
        FLAG_binding_memory);
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->xdna_api->device_create(run->endpoint, &create_info, &run->device),
      "xdna.device_create"));
  run->memory_access.device = run->device;
  amdf_memory_access_capabilities_t memory_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(memory_capabilities),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_scope_query_device_profile(
          run->memory_scope, run->memory_profile.ordinal, 1,
          &run->memory_access, &run->memory_profile, &memory_capabilities),
      "memory_scope_query_device_profile"));
  if ((run->memory_profile.roles & required_roles) != required_roles ||
      (required_flags & ~run->memory_profile.supported_flags) != 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "live device cannot provide the selected host-visible %s profile",
        FLAG_binding_memory);
  }
  const amdf_xdna_context_create_info_t context_create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
      .structure_size = sizeof(context_create_info),
      .logical_column_count = (uint32_t)FLAG_columns,
      .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
      .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
  };
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->xdna_api->context_create(run->device, &context_create_info,
                                    &run->context),
      "xdna.context_create");
}

static iree_status_t iree_xdna_run_prepare_registered_binding(
    iree_xdna_run_t* run,
    const iree_hal_amd_xdna_elf_binding_record_t* contract,
    iree_const_byte_span_t initial, iree_xdna_run_binding_t* binding) {
  const uint64_t cover_granularity =
      run->memory_profile.registration.native_byte_length_granularity;
  const uint64_t maximum_alignment =
      run->memory_profile.registration.maximum_alignment;
  if (cover_granularity > IREE_HOST_SIZE_MAX ||
      maximum_alignment > IREE_HOST_SIZE_MAX ||
      initial.data_length > IREE_HOST_SIZE_MAX - cover_granularity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "registered binding storage exceeds host limits");
  }
  if (contract->minimum_alignment > maximum_alignment) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "binding %u requires alignment %" PRIu64
        " beyond the registered-host profile limit %" PRIu64,
        contract->binding_ordinal, contract->minimum_alignment,
        maximum_alignment);
  }

  const iree_host_size_t pointer_offset =
      contract->minimum_alignment < cover_granularity
          ? (iree_host_size_t)contract->minimum_alignment
          : 0;
  const iree_host_size_t allocation_length =
      (iree_host_size_t)cover_granularity + initial.data_length;
  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      run->host_allocator, allocation_length,
      (iree_host_size_t)maximum_alignment, 0, &allocation));
  binding->registered_allocation = allocation;
  binding->registered_host_pointer = (uint8_t*)allocation + pointer_offset;
  memcpy(binding->registered_host_pointer, initial.data, initial.data_length);
  return iree_ok_status();
}

static iree_status_t iree_xdna_run_prepare_binding(
    iree_xdna_run_t* run, iree_hal_executable_function_t function,
    iree_host_size_t ordinal) {
  iree_xdna_run_binding_t* binding = &run->bindings[ordinal];
  const iree_const_byte_span_t initial =
      binding->initial_contents->const_buffer;
  iree_hal_amd_xdna_elf_binding_record_t contract;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_query_binding(
      run->executable, function, ordinal, &contract));
  if (run->binding_memory_role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
    IREE_RETURN_IF_ERROR(iree_xdna_run_prepare_registered_binding(
        run, &contract, initial, binding));
  }
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = run->memory_profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = initial.data_length,
      .minimum_alignment = contract.minimum_alignment,
      .registered_host_pointer = binding->registered_host_pointer,
      .accesses = &run->memory_access,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_create(run->memory_scope, &create_info,
                              &binding->memory),
      "memory_create"));
  amdf_memory_info_t memory_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(memory_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_query_info(binding->memory, &memory_info),
      "memory_query_info"));
  amdf_memory_access_info_t access_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(access_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_query_access_info(binding->memory, 0, &access_info),
      "memory_query_access_info"));
  uint64_t dma_address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_query_address(
          binding->memory, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &dma_address),
      "memory_query_address(XDNA_DMA)"));
  if (binding->registered_host_pointer != NULL) {
    const uint64_t expected_source_byte_offset =
        (uintptr_t)binding->registered_host_pointer %
        run->memory_profile.registration.native_byte_length_granularity;
    if (memory_info.memory_class != AMDF_MEMORY_CLASS_SYSTEM ||
        memory_info.byte_length != initial.data_length ||
        memory_info.source_byte_offset != expected_source_byte_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "registered binding %u returned inconsistent logical geometry",
          contract.binding_ordinal);
    }
  }
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = initial.data_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_map(binding->memory, &map_info, &binding->mapping),
      "memory_map"));
  binding->mapping_info = (amdf_host_mapping_info_t){
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(binding->mapping_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_query_info(binding->mapping,
                                        &binding->mapping_info),
      "host_mapping_query_info"));
  if (binding->registered_host_pointer != NULL &&
      binding->mapping_info.pointer != binding->registered_host_pointer) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "registered binding %u did not preserve its logical host address",
        contract.binding_ordinal);
  }
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                       IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if ((access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  if (binding->mapping_info.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  }
  iree_hal_memory_access_t memory_access =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  if (!iree_host_size_has_alignment((uintptr_t)binding->mapping_info.pointer,
                                    IREE_HAL_HEAP_BUFFER_ALIGNMENT)) {
    memory_access |= IREE_HAL_MEMORY_ACCESS_UNALIGNED;
  }
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), memory_type, memory_access,
      IREE_HAL_BUFFER_USAGE_STORAGE, initial.data_length,
      iree_make_byte_span(binding->mapping_info.pointer, initial.data_length),
      iree_hal_buffer_release_callback_null(), run->host_allocator,
      &binding->buffer));
  if (binding->registered_host_pointer == NULL) {
    memcpy(binding->mapping_info.pointer, initial.data, initial.data_length);
  } else {
    fprintf(stderr, "Registered binding %u at native page offset %" PRIu64 "\n",
            contract.binding_ordinal, memory_info.source_byte_offset);
    fflush(stderr);
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_cache_control(binding->mapping,
                                           AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                           initial.data_length),
      "host_mapping_cache_control(FLUSH)"));
  run->prepared_bindings[ordinal] =
      (iree_hal_amd_xdna_prepared_command_binding_t){
          .buffer_ref =
              iree_hal_make_buffer_ref(binding->buffer, 0, initial.data_length),
          .memory = binding->memory,
          .device_address = dma_address,
      };
  return iree_ok_status();
}

static iree_status_t iree_xdna_run_prepare_invocation(
    iree_xdna_run_t* run, iree_hal_executable_function_t function) {
  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_query_storage_size(
      run->executable, function, run->instructions.alignment, &byte_length));
  amdf_memory_scope_t* scope = NULL;
  uint32_t count = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->xdna_api->context_enumerate_memory_scopes(run->context, 1, &scope,
                                                     &count),
      "xdna.context_enumerate_memory_scopes"));
  const amdf_memory_device_access_t access = {
      .device = run->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        AMDF_MEMORY_ACCESS_EXECUTE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
          },
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_scope_query_device_profile(scope, 0, 1, &access,
                                                  &profile, &capabilities),
      "memory_scope_query_device_profile(instructions)"));
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  uint64_t rounded_length = 0;
  if (!iree_checked_add_u64(byte_length, granularity - 1, &rounded_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "instruction allocation size overflows");
  }
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = (rounded_length / granularity) * granularity,
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &access,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_create(scope, &create_info, &run->instructions.memory),
      "memory_create(instructions)"));
  uint64_t firmware_address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_query_address(run->instructions.memory, 0,
                                     AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                     &firmware_address),
      "memory_query_address(instructions)"));
  if (firmware_address % run->instructions.alignment != 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "private allocation cannot satisfy instruction alignment");
  }
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = byte_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_map(run->instructions.memory, &map_info,
                           &run->instructions.mapping),
      "memory_map(instructions)"));
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_query_info(run->instructions.mapping,
                                        &mapping_info),
      "host_mapping_query_info(instructions)"));
  const amdf_xdna_kernel_command_t storage_range = {
      .memory = run->instructions.memory,
      .byte_length = byte_length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_create(
      run->executable, function, run->instructions.alignment, &storage_range,
      iree_make_byte_span(mapping_info.pointer, byte_length),
      run->binding_count, run->prepared_bindings, run->host_allocator,
      &run->prepared_command));
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_cache_control(run->instructions.mapping,
                                           AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                           byte_length),
      "host_mapping_cache_control(instructions)");
}

static iree_status_t iree_xdna_run_execute(iree_xdna_run_t* run,
                                           iree_byte_sequence_t* image) {
  IREE_RETURN_IF_ERROR(iree_xdna_run_open_endpoint(run));
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_RETURN_IF_ERROR(iree_xdna_run_create_device(run, &target));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_create(
      &run->queue_family, image, &target, run->host_allocator,
      &run->executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  if (FLAG_entry[0] != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
        run->executable, iree_make_cstring_view(FLAG_entry), &function));
  }
  iree_hal_executable_function_info_t function_info;
  IREE_RETURN_IF_ERROR(iree_hal_executable_function_info(
      run->executable, function, &function_info));
  if (function_info.binding_count != run->binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry requires %u bindings but %" PRIhsz
                            " were supplied",
                            function_info.binding_count, run->binding_count);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < run->binding_count; ++i) {
    status = iree_xdna_run_prepare_binding(run, function, i);
  }
  if (!iree_status_is_ok(status)) return status;
  IREE_RETURN_IF_ERROR(iree_xdna_run_prepare_invocation(run, function));
  const amdf_xdna_kernel_queue_create_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_info),
      .queue_family_ordinal = run->queue_family.ordinal,
  };
  fprintf(stderr, "Prepared %.*s; acquiring queue and admitting firmware\n",
          (int)function_info.name.size, function_info.name.data);
  fflush(stderr);
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(run->xdna_api->kernel_queue_create(
                                        run->context, &queue_info, &run->queue),
                                    "xdna.kernel_queue_create"));
  for (int32_t invocation_ordinal = 0;
       invocation_ordinal < FLAG_invocation_count; ++invocation_ordinal) {
    const amdf_xdna_kernel_queue_submission_info_t submission_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submission_info),
        .command_count = 1,
        .commands = invocation_ordinal == 0
                        ? iree_hal_amd_xdna_prepared_command_initialization(
                              run->prepared_command)
                        : iree_hal_amd_xdna_prepared_command_execution(
                              run->prepared_command),
    };
    uint64_t submission = 0;
    fprintf(stderr, "Publishing invocation %d/%d (%s)\n",
            invocation_ordinal + 1, FLAG_invocation_count,
            invocation_ordinal == 0 ? "load and control" : "control only");
    fflush(stderr);
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->xdna_api->kernel_queue_submit(run->queue, &submission_info,
                                           &submission),
        "xdna.kernel_queue_submit"));
    fprintf(stderr, "Waiting for submission %" PRIu64 "\n", submission);
    fflush(stderr);
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->kernel_queue_wait(run->queue, submission,
                                    AMDF_TIMEOUT_INFINITE, 0),
        "kernel_queue_wait"));
    fprintf(stderr, "Submission %" PRIu64 " completed and retired\n",
            submission);
    fflush(stderr);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < run->binding_count; ++i) {
    iree_xdna_run_binding_t* binding = &run->bindings[i];
    if (iree_string_view_is_empty(binding->output_path)) continue;
    const iree_host_size_t length =
        binding->initial_contents->buffer.data_length;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->host_mapping_cache_control(
            binding->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, length),
        "host_mapping_cache_control(INVALIDATE)");
    if (iree_status_is_ok(status)) {
      status = iree_io_file_contents_write(
          binding->output_path,
          iree_make_const_byte_span(binding->mapping_info.pointer, length),
          run->host_allocator);
    }
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "Wrote binding %" PRIhsz ": %" PRIhsz " bytes to %.*s\n",
              i, length, (int)binding->output_path.size,
              binding->output_path.data);
      fflush(stderr);
    }
  }
  return status;
}

static iree_status_t iree_xdna_run_deinitialize(iree_xdna_run_t* run) {
  if (run->queue != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->kernel_queue_destroy(run->queue), "kernel_queue_destroy"));
    run->queue = NULL;
  }
  iree_hal_amd_xdna_prepared_command_destroy(run->prepared_command);
  run->prepared_command = NULL;
  if (run->instructions.mapping != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->host_mapping_destroy(run->instructions.mapping),
        "host_mapping_destroy(instructions)"));
    run->instructions.mapping = NULL;
  }
  if (run->instructions.memory != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->memory_destroy(run->instructions.memory),
        "memory_destroy(instructions)"));
    run->instructions.memory = NULL;
  }
  iree_hal_executable_release(run->executable);
  run->executable = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < run->binding_count; ++i) {
    iree_xdna_run_binding_t* binding = &run->bindings[i];
    iree_hal_buffer_release(binding->buffer);
    binding->buffer = NULL;
    if (binding->mapping != NULL) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          run->api->host_mapping_destroy(binding->mapping),
          "host_mapping_destroy");
      if (iree_status_is_ok(status)) binding->mapping = NULL;
    }
    if (iree_status_is_ok(status) && binding->memory != NULL) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          run->api->memory_destroy(binding->memory), "memory_destroy");
      if (iree_status_is_ok(status)) {
        binding->memory = NULL;
        if (binding->registered_host_pointer != NULL) {
          volatile uint8_t* caller_bytes =
              (volatile uint8_t*)binding->registered_host_pointer;
          const iree_host_size_t byte_length =
              binding->initial_contents->buffer.data_length;
          for (iree_host_size_t j = 0; j < byte_length; ++j) {
            const uint8_t value = caller_bytes[j];
            caller_bytes[j] = value;
          }
          fprintf(stderr,
                  "Released binding %" PRIhsz
                  "; all caller bytes remain accessible\n",
                  i);
          fflush(stderr);
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) return status;
  for (iree_host_size_t i = 0; i < run->binding_count; ++i) {
    iree_allocator_free_aligned(run->host_allocator,
                                run->bindings[i].registered_allocation);
    iree_io_file_contents_free(run->bindings[i].initial_contents);
  }
  iree_allocator_free(run->host_allocator, run->bindings);
  run->bindings = NULL;
  run->binding_count = 0;
  if (run->context != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->xdna_api->context_destroy(run->context), "xdna.context_destroy"));
    run->context = NULL;
  }
  if (run->device != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->device_destroy(run->device), "device_destroy"));
    run->device = NULL;
  }
  if (run->endpoint != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->endpoint_close(run->endpoint), "endpoint_close"));
    run->endpoint = NULL;
  }
  if (run->instance != NULL) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->instance_destroy(run->instance), "instance_destroy"));
    run->instance = NULL;
  }
  return iree_ok_status();
}

static iree_status_t iree_xdna_run_main(void) {
  if (FLAG_image[0] == 0 || FLAG_columns < 1 || FLAG_columns > 8 ||
      FLAG_device < 0 || FLAG_invocation_count < 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--image, --columns in [1, 8], a nonnegative "
                            "--device ordinal, and --invocation_count >= 1 "
                            "are required");
  }
  iree_xdna_run_t run = {.host_allocator = iree_allocator_system()};
  IREE_RETURN_IF_ERROR(iree_xdna_run_select_binding_memory(&run));
  iree_byte_sequence_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_xdna_run_load_image(
      iree_make_cstring_view(FLAG_image), run.host_allocator, &image));
  iree_status_t status = iree_xdna_run_load_bindings(&run);
  if (iree_status_is_ok(status)) status = iree_xdna_run_execute(&run, image);
  const iree_status_t cleanup_status = iree_xdna_run_deinitialize(&run);
  if (iree_status_is_ok(cleanup_status)) {
    fprintf(stderr, "All native resources released\n");
    fflush(stderr);
  }
  iree_byte_sequence_release(image);
  return iree_status_join(status, cleanup_status);
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "iree-xdna-run",
      "Runs one canonical XDNA entry through libamdf with ordered raw "
      "buffers.\n"
      "Example: --image=copy.xdna --columns=1 --entry=copy_i32\n"
      "  --binding_memory=registered_host --binding=input.bin\n"
      "  --binding=sentinel.bin --output=1=result.bin\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status =
      argc == 1 ? iree_xdna_run_main()
                : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "unexpected positional argument");
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
