// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/executable.h"

#include "iree/hal/drivers/amd/status.h"

struct iree_hal_amd_xdna_function_t {
  // Borrowed native device owner.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed admitted image owning immutable metadata.
  iree_hal_amd_xdna_image_t* image;
  // Dense image entry ordinal.
  uint32_t ordinal;
  // Decoded immutable function contract.
  iree_xdna_elf_entry_record_t record;
  // Owned backing in entry-relative allocation-use order.
  iree_hal_amd_xdna_memory_t* allocations;
  // Materializer views borrowing the allocations.
  iree_hal_amd_xdna_executable_storage_t* storage;
  // Independent invocation establishing tile state on every submission.
  amdf_xdna_kernel_command_t command;
};

typedef struct iree_hal_amd_xdna_executable_t {
  // HAL identity and borrowed family.
  iree_hal_executable_t base;
  // Allocator owning all host metadata.
  iree_allocator_t host_allocator;
  // Immutable image retaining its copied source bytes.
  iree_hal_amd_xdna_image_t* image;
  // Number of entries in the admitted image.
  uint32_t function_count;
  // Owned function array; partially initialized during construction.
  iree_hal_amd_xdna_function_t* functions;
} iree_hal_amd_xdna_executable_t;

static const iree_hal_executable_vtable_t iree_hal_amd_xdna_executable_vtable;

static void iree_hal_amd_xdna_executable_destroy(iree_hal_executable_t* base) {
  iree_hal_amd_xdna_executable_t* executable =
      (iree_hal_amd_xdna_executable_t*)base;
  if (executable->functions) {
    for (uint32_t i = 0; i < executable->function_count; ++i) {
      iree_hal_amd_xdna_function_t* function = &executable->functions[i];
      if (function->allocations) {
        for (uint32_t j = 0; j < function->record.allocation_use_count; ++j) {
          iree_hal_amd_xdna_memory_deinitialize(function->context,
                                                &function->allocations[j]);
        }
      }
      iree_allocator_free(executable->host_allocator, function->allocations);
    }
  }
  iree_allocator_free(executable->host_allocator, executable->functions);
  iree_hal_amd_xdna_image_destroy(executable->image);
  iree_allocator_free(executable->host_allocator, executable);
}

static iree_status_t iree_hal_amd_xdna_function_initialize(
    iree_hal_amd_xdna_executable_t* executable,
    iree_hal_amd_xdna_context_t* context, uint32_t ordinal) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(executable->image);
  iree_hal_amd_xdna_function_t* function = &executable->functions[ordinal];
  function->context = context;
  function->image = executable->image;
  function->ordinal = ordinal;
  function->record = iree_hal_amd_xdna_image_tables_entry(tables, ordinal);
  if (function->record.binding_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA entry exceeds HAL binding count limit");
  }
  const uint32_t count = function->record.allocation_use_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executable->host_allocator,
      count * (sizeof(*function->allocations) + sizeof(*function->storage)),
      (void**)&function->allocations));
  function->storage =
      (iree_hal_amd_xdna_executable_storage_t*)(function->allocations + count);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const uint32_t allocation_ordinal =
        iree_hal_amd_xdna_image_tables_allocation_use(
            tables, function->record.first_allocation_use + i);
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, allocation_ordinal);
    const iree_hal_amd_xdna_memory_source_t* source =
        allocation.domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND
            ? &context->command_source
            : &context->data_source;
    // The image alignment constrains the native address domain. Physical
    // allocation alignment is a distinct provider construction contract;
    // the materializer checks the resulting firmware/DMA address below.
    status = iree_hal_amd_xdna_memory_allocate(
        context, source, allocation.byte_length,
        source->profile.allocation.minimum_alignment,
        &function->allocations[i]);
    if (iree_status_is_ok(status)) {
      const iree_hal_amd_xdna_memory_t* memory = &function->allocations[i];
      function->storage[i] = (iree_hal_amd_xdna_executable_storage_t){
          .mapping = memory->contents,
          .memory = memory->handle,
          .device_address = memory->device_address,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_storage_load(
        function->image, ordinal, count, function->storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_storage_query_invocation(
        function->image, ordinal, count, function->storage, &function->command);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_executable_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  if (params->constant_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA images have no load-time constants");
  }
  iree_hal_amd_xdna_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  iree_hal_executable_initialize(family, &iree_hal_amd_xdna_executable_vtable,
                                 &executable->base);
  executable->host_allocator = host_allocator;
  iree_byte_span_t source = {NULL, params->executable_data.data_length};
  iree_status_t status = iree_allocator_clone(
      host_allocator, params->executable_data, (void**)&source.data);
  iree_byte_sequence_t* sequence = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_byte_sequence_create_from_span_move(&source, host_allocator,
                                                      &sequence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_create(sequence, &context->target,
                                            host_allocator, &executable->image);
  }
  iree_byte_sequence_release(sequence);
  iree_allocator_free(host_allocator, source.data);
  if (iree_status_is_ok(status)) {
    executable->function_count =
        iree_hal_amd_xdna_image_tables(executable->image)->header.entry_count;
    status = iree_allocator_malloc(
        host_allocator,
        executable->function_count * sizeof(*executable->functions),
        (void**)&executable->functions);
  }
  for (uint32_t i = 0;
       i < executable->function_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_xdna_function_initialize(executable, context, i);
  }
  if (iree_status_is_ok(status)) {
    *out_executable = &executable->base;
  } else {
    iree_hal_executable_release(&executable->base);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_executable_lookup(
    iree_hal_executable_t* base, iree_hal_executable_function_t token,
    iree_hal_amd_xdna_function_t** out_function) {
  iree_hal_amd_xdna_executable_t* executable =
      (iree_hal_amd_xdna_executable_t*)base;
  if (!iree_hal_executable_function_is_index_in_range(
          token, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA function token is out of range");
  }
  *out_function =
      &executable->functions[iree_hal_executable_function_index(token)];
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_resolve(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amd_xdna_executable_binding_t* out_bindings,
    iree_hal_amd_xdna_function_t** out_function) {
  iree_hal_amd_xdna_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_lookup(executable, token, &function));
  if (bindings.count != function->record.binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA function requires %u bindings",
                            function->record.binding_count);
  }
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(function->image);
  for (uint32_t i = 0; i < bindings.count; ++i) {
    const iree_xdna_elf_binding_record_t contract =
        iree_hal_amd_xdna_image_tables_binding(
            tables, function->record.first_binding + i);
    out_bindings[i] = (iree_hal_amd_xdna_executable_binding_t){0};
    if (contract.kind != IREE_XDNA_ELF_BINDING_KIND_NONE) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_resolve(
          function->context, bindings.values[i], &out_bindings[i]));
    }
  }
  *out_function = function;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_function_prepare(
    iree_hal_amd_xdna_function_t* function,
    const iree_hal_amd_xdna_executable_binding_t* bindings,
    amdf_xdna_kernel_command_t* out_command) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_storage_bind(
      function->image, function->ordinal, function->record.allocation_use_count,
      function->storage, function->record.binding_count, bindings));
  for (uint32_t i = 0; i < function->record.allocation_use_count; ++i) {
    const iree_hal_amd_xdna_memory_t* memory = &function->allocations[i];
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        function->context->api->host_mapping_cache_control(
            memory->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            memory->contents.data_length),
        "host_mapping_cache_control(command)"));
  }
  *out_command = function->command;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amd_xdna_executable_function_count(
    iree_hal_executable_t* executable) {
  return ((iree_hal_amd_xdna_executable_t*)executable)->function_count;
}

static iree_status_t iree_hal_amd_xdna_executable_function_info(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_amd_xdna_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_lookup(executable, token, &function));
  *out_info = (iree_hal_executable_function_info_t){
      .name = iree_hal_amd_xdna_image_tables_entry_name(
          iree_hal_amd_xdna_image_tables(function->image), &function->record),
      .binding_count = function->record.binding_count,
      .parameter_count = function->record.binding_count,
      .maximum_workgroup_invocations = 1,
      .workgroup_size = {1, 1, 1},
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_function_parameters(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_amd_xdna_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_lookup(executable, token, &function));
  if (capacity < function->record.binding_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  for (uint32_t i = 0; i < function->record.binding_count; ++i) {
    out_parameters[i] = (iree_hal_executable_function_parameter_t){
        .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
        .offset = i,
    };
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_lookup_function_by_name(
    iree_hal_executable_t* base, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_amd_xdna_executable_t* executable =
      (iree_hal_amd_xdna_executable_t*)base;
  uint32_t ordinal = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_image_find_entry(executable->image, name, &ordinal));
  *out_function = iree_hal_executable_function_from_index(ordinal);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_try_lookup_global_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name, bool* out_found,
    iree_hal_executable_global_t* out_global) {
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_global_info(
    iree_hal_executable_t* executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "XDNA image has no HAL globals");
}

static iree_status_t iree_hal_amd_xdna_executable_global_buffer(
    iree_hal_executable_t* executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "XDNA image has no HAL globals");
}

static const iree_hal_executable_vtable_t iree_hal_amd_xdna_executable_vtable =
    {
        .destroy = iree_hal_amd_xdna_executable_destroy,
        .function_count = iree_hal_amd_xdna_executable_function_count,
        .function_info = iree_hal_amd_xdna_executable_function_info,
        .function_parameters = iree_hal_amd_xdna_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_amd_xdna_executable_lookup_function_by_name,
        .try_lookup_global_by_name =
            iree_hal_amd_xdna_executable_try_lookup_global_by_name,
        .global_info = iree_hal_amd_xdna_executable_global_info,
        .global_buffer = iree_hal_amd_xdna_executable_global_buffer,
};
