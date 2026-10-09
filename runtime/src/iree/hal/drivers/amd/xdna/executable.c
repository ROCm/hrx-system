// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/executable.h"

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/status.h"

// Cold allocation plan retained in entry-relative allocation-use order.
typedef struct iree_hal_amd_xdna_allocation_plan_t {
  // Decoded storage requirement from the admitted image.
  iree_xdna_elf_allocation_record_t record;
  // True only for immutable backing whose static address closure is shared.
  bool shared;
  // Smallest range containing this allocation's dynamic address patches.
  struct {
    // First patched byte, or UINT64_MAX when there are no dynamic patches.
    uint64_t begin;
    // One past the final patched byte, or zero when there are no patches.
    uint64_t end;
  } patches;
} iree_hal_amd_xdna_allocation_plan_t;

// Cold ownership behind one executable storage row. Command rows own a range
// in the device arena; DMA rows own an independent native allocation.
typedef struct iree_hal_amd_xdna_owned_allocation_t {
  // Independent native allocation used only for a DMA-domain row.
  iree_hal_amd_xdna_memory_t memory;
  // Device command-arena range used only for a command-domain row.
  iree_hal_amd_xdna_command_allocation_t command;
} iree_hal_amd_xdna_owned_allocation_t;

struct iree_hal_amd_xdna_invocation_t {
  // Borrowed function, kept alive by the submitting operation's executable.
  iree_hal_amd_xdna_function_t* function;
  // All allocated invocations, owned until executable destruction.
  iree_hal_amd_xdna_invocation_t* next;
  // Available storage protected by the function's pool mutex.
  iree_hal_amd_xdna_invocation_t* next_available;
  // Backing ownership for each row; shared entries remain empty.
  iree_hal_amd_xdna_owned_allocation_t* allocations;
  // Complete views, including borrowed immutable backing from the prototype.
  iree_hal_amd_xdna_executable_storage_t* storage;
  // Independent invocation establishing tile state on every submission.
  amdf_xdna_kernel_command_t command;
};

struct iree_hal_amd_xdna_function_t {
  // Borrowed native device owner.
  iree_hal_amd_xdna_context_t* context;
  // Host owner for cold invocation growth.
  iree_allocator_t host_allocator;
  // Borrowed device-owned instruction arena.
  iree_hal_amd_xdna_command_arena_t* command_arena;
  // Borrowed admitted image owning immutable metadata.
  iree_hal_amd_xdna_image_t* image;
  // Dense image entry ordinal.
  uint32_t ordinal;
  // Decoded immutable function contract.
  iree_xdna_elf_entry_record_t record;
  // Cold sharing and publication plan, with one row per allocation use.
  iree_hal_amd_xdna_allocation_plan_t* plans;
  // Protects only invocation list links; never held across allocation/native
  // IO.
  iree_slim_mutex_t pool_mutex;
  // First prepared invocation owning the shared immutable allocation closure.
  iree_hal_amd_xdna_invocation_t* prototype;
  // Owned invocations, including the prototype.
  iree_hal_amd_xdna_invocation_t* invocations;
  // Available exclusive invocation storage.
  iree_hal_amd_xdna_invocation_t* available;
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

static iree_hal_memory_access_t iree_hal_amd_xdna_required_binding_access(
    iree_xdna_elf_binding_access_t access) {
  iree_hal_memory_access_t required_access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  return required_access;
}

static iree_hal_buffer_usage_t iree_hal_amd_xdna_required_binding_usage(
    iree_xdna_elf_binding_access_t access) {
  iree_hal_buffer_usage_t required_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  }
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  }
  return required_usage;
}

static iree_hal_memory_type_t iree_hal_amd_xdna_required_binding_memory_type(
    const iree_xdna_elf_binding_record_t* contract) {
  iree_hal_memory_type_t required_memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  if (iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE)) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }
  const bool requires_host_visibility =
      contract->address_space == IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST ||
      iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE);
  if (requires_host_visibility) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    if (iree_any_bit_set(contract->usage,
                         IREE_XDNA_ELF_BINDING_USAGE_COHERENT)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    }
    if (iree_any_bit_set(contract->usage, IREE_XDNA_ELF_BINDING_USAGE_CACHED)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    }
  }
  return required_memory_type;
}

static void iree_hal_amd_xdna_invocation_destroy(
    iree_hal_amd_xdna_invocation_t* invocation) {
  iree_hal_amd_xdna_function_t* function = invocation->function;
  for (uint32_t i = 0; i < function->record.allocation_use_count; ++i) {
    iree_hal_amd_xdna_owned_allocation_t* allocation =
        &invocation->allocations[i];
    iree_hal_amd_xdna_command_arena_release(function->command_arena,
                                            &allocation->command);
    iree_hal_amd_xdna_memory_deinitialize(function->context,
                                          &allocation->memory);
  }
  iree_allocator_free(function->host_allocator, invocation);
}

static void iree_hal_amd_xdna_executable_destroy(iree_hal_executable_t* base) {
  iree_hal_amd_xdna_executable_t* executable =
      (iree_hal_amd_xdna_executable_t*)base;
  if (executable->functions) {
    for (uint32_t i = 0; i < executable->function_count; ++i) {
      iree_hal_amd_xdna_function_t* function = &executable->functions[i];
      iree_hal_amd_xdna_invocation_t* invocation = function->invocations;
      while (invocation) {
        iree_hal_amd_xdna_invocation_t* next = invocation->next;
        iree_hal_amd_xdna_invocation_destroy(invocation);
        invocation = next;
      }
      iree_slim_mutex_deinitialize(&function->pool_mutex);
      iree_allocator_free(executable->host_allocator, function->plans);
    }
  }
  iree_allocator_free(executable->host_allocator, executable->functions);
  iree_hal_amd_xdna_image_destroy(executable->image);
  iree_allocator_free(executable->host_allocator, executable);
}

// Creates storage only when the observed in-flight working set grows. Shared
// views are already published and loading never writes them, even while another
// invocation is reading them on the device.
static iree_status_t iree_hal_amd_xdna_invocation_create(
    iree_hal_amd_xdna_function_t* function,
    iree_hal_amd_xdna_invocation_t** out_invocation) {
  *out_invocation = NULL;
  const uint32_t count = function->record.allocation_use_count;
  iree_host_size_t size = 0, allocations_offset = 0, storage_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_invocation_t), &size,
      IREE_STRUCT_FIELD_ALIGNED(
          count, iree_hal_amd_xdna_owned_allocation_t,
          iree_alignof(iree_hal_amd_xdna_owned_allocation_t),
          &allocations_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          count, iree_hal_amd_xdna_executable_storage_t,
          iree_alignof(iree_hal_amd_xdna_executable_storage_t),
          &storage_offset)));
  iree_hal_amd_xdna_invocation_t* invocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(function->host_allocator, size,
                                             (void**)&invocation));
  invocation->function = function;
  invocation->allocations =
      (iree_hal_amd_xdna_owned_allocation_t*)((uint8_t*)invocation +
                                              allocations_offset);
  invocation->storage =
      (iree_hal_amd_xdna_executable_storage_t*)((uint8_t*)invocation +
                                                storage_offset);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_xdna_allocation_plan_t* plan = &function->plans[i];
    if (function->prototype && plan->shared) {
      invocation->storage[i] = function->prototype->storage[i];
      invocation->storage[i].flags =
          IREE_HAL_AMD_XDNA_EXECUTABLE_STORAGE_FLAG_SHARED;
      continue;
    }
    iree_hal_amd_xdna_owned_allocation_t* allocation =
        &invocation->allocations[i];
    if (plan->record.domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND) {
      status = iree_hal_amd_xdna_command_arena_allocate(
          function->command_arena, plan->record.byte_length,
          plan->record.alignment, &allocation->command);
      if (iree_status_is_ok(status)) {
        invocation->storage[i] = (iree_hal_amd_xdna_executable_storage_t){
            .mapping = allocation->command.contents,
            .memory = allocation->command.memory,
            .memory_byte_offset = allocation->command.memory_byte_offset,
            .device_address = allocation->command.device_address,
        };
      }
    } else {
      const iree_hal_amd_xdna_memory_source_t* source =
          &function->context->data_source;
      // Image alignment constrains the native address domain. Physical
      // allocation alignment is a distinct provider construction contract.
      status = iree_hal_amd_xdna_memory_allocate(
          function->context, source, plan->record.byte_length,
          source->profile.allocation.minimum_alignment, &allocation->memory);
      if (iree_status_is_ok(status)) {
        const iree_hal_amd_xdna_memory_t* memory = &allocation->memory;
        invocation->storage[i] = (iree_hal_amd_xdna_executable_storage_t){
            .mapping = memory->contents,
            .memory = memory->handle,
            .device_address = memory->device_address,
        };
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_storage_load(
        function->image, function->ordinal, count, invocation->storage);
  }
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_xdna_owned_allocation_t* allocation =
        &invocation->allocations[i];
    amdf_host_mapping_t* mapping = allocation->command.mapping
                                       ? allocation->command.mapping
                                       : allocation->memory.mapping;
    if (mapping) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          function->context->api->host_mapping_cache_control(
              mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
              invocation->storage[i].memory_byte_offset,
              invocation->storage[i].mapping.data_length),
          "host_mapping_cache_control(load)");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_storage_query_invocation(
        function->image, function->ordinal, count, invocation->storage,
        &invocation->command);
  }
  if (iree_status_is_ok(status)) {
    *out_invocation = invocation;
  } else {
    iree_hal_amd_xdna_invocation_destroy(invocation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_function_initialize(
    iree_hal_amd_xdna_executable_t* executable,
    iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena, uint32_t ordinal) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(executable->image);
  iree_hal_amd_xdna_function_t* function = &executable->functions[ordinal];
  function->context = context;
  function->host_allocator = executable->host_allocator;
  function->command_arena = command_arena;
  function->image = executable->image;
  function->ordinal = ordinal;
  function->record = iree_hal_amd_xdna_image_tables_entry(tables, ordinal);
  if (function->record.binding_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA entry exceeds HAL binding count limit");
  }
  const uint32_t count = function->record.allocation_use_count;
  iree_host_size_t plans_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &plans_size,
      IREE_STRUCT_FIELD_FAM(count, iree_hal_amd_xdna_allocation_plan_t)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executable->host_allocator, plans_size, (void**)&function->plans));
  for (uint32_t i = 0; i < count; ++i) {
    iree_hal_amd_xdna_allocation_plan_t* plan = &function->plans[i];
    const uint32_t allocation_ordinal =
        iree_hal_amd_xdna_image_tables_allocation_use(
            tables, function->record.first_allocation_use + i);
    plan->record =
        iree_hal_amd_xdna_image_tables_allocation(tables, allocation_ordinal);
    plan->shared = iree_any_bit_set(plan->record.flags,
                                    IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE);
    plan->patches.begin = UINT64_MAX;
  }
  // Immutable bytes containing a private allocation's address are themselves
  // private. Compute this transitive closure once, before any storage exists.
  bool changed = true;
  while (changed) {
    changed = false;
    for (uint32_t i = 0; i < function->record.static_relocation_count; ++i) {
      const iree_xdna_elf_relocation_record_t relocation =
          iree_hal_amd_xdna_image_tables_relocation(
              tables, function->record.first_static_relocation + i);
      iree_hal_amd_xdna_allocation_plan_t* destination =
          &function->plans[relocation.destination_use];
      if (destination->shared &&
          !function->plans[relocation.source_ordinal].shared) {
        destination->shared = false;
        changed = true;
      }
    }
  }
  for (uint32_t i = 0; i < function->record.dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, function->record.first_dynamic_relocation + i);
    iree_hal_amd_xdna_allocation_plan_t* plan =
        &function->plans[relocation.destination_use];
    plan->patches.begin = iree_min(plan->patches.begin, relocation.byte_offset);
    plan->patches.end = iree_max(plan->patches.end, relocation.byte_offset + 8);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_invocation_create(function, &function->prototype));
  function->invocations = function->prototype;
  function->available = function->prototype;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
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
    iree_host_size_t functions_size = 0;
    status =
        IREE_STRUCT_LAYOUT(0, &functions_size,
                           IREE_STRUCT_FIELD_FAM(executable->function_count,
                                                 iree_hal_amd_xdna_function_t));
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(host_allocator, functions_size,
                                     (void**)&executable->functions);
    }
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < executable->function_count; ++i) {
      iree_slim_mutex_initialize(&executable->functions[i].pool_mutex);
    }
  }
  for (uint32_t i = 0;
       i < executable->function_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_xdna_function_initialize(executable, context,
                                                   command_arena, i);
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

static iree_status_t iree_hal_amd_xdna_executable_capture_binding(
    iree_hal_amd_xdna_function_t* function,
    const iree_hal_queue_family_t* family, uint32_t binding_ordinal,
    const iree_xdna_elf_binding_record_t* contract,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_amd_xdna_executable_binding_t* out_binding) {
  *out_binding = (iree_hal_amd_xdna_executable_binding_t){0};
  if (buffer_ref.reserved != 0 || buffer_ref.buffer_slot != 0 ||
      !buffer_ref.buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA binding %u must be a direct non-null buffer reference",
        binding_ordinal);
  }

  iree_device_size_t resource_byte_offset = 0;
  iree_device_size_t resource_byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(buffer_ref.buffer),
      buffer_ref.offset, buffer_ref.length, &resource_byte_offset,
      &resource_byte_length));
  const iree_hal_buffer_usage_t required_usage =
      iree_hal_amd_xdna_required_binding_usage(contract->access);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_family_usage(
      buffer_ref.buffer, family, required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer),
      iree_hal_amd_xdna_required_binding_access(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer_ref.buffer),
      iree_hal_amd_xdna_required_binding_memory_type(contract)));

  if (resource_byte_offset < contract->minimum_byte_offset ||
      resource_byte_offset > contract->maximum_byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u offset %" PRIu64
                            " is outside [%" PRIu64 ", %" PRIu64 "]",
                            binding_ordinal, (uint64_t)resource_byte_offset,
                            contract->minimum_byte_offset,
                            contract->maximum_byte_offset);
  }
  if (resource_byte_length < contract->minimum_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u length %" PRIu64
                            " is smaller than required %" PRIu64,
                            binding_ordinal, (uint64_t)resource_byte_length,
                            contract->minimum_byte_length);
  }

  iree_hal_buffer_native_binding_slot_t slot;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_resolve_binding_slot(
      function->context, family, buffer_ref.buffer, &slot));
  buffer_ref.offset = resource_byte_offset;
  buffer_ref.length = resource_byte_length;
  *out_binding = (iree_hal_amd_xdna_executable_binding_t){
      .buffer_ref = buffer_ref,
      .slot = slot,
      .byte_length = resource_byte_length,
  };
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
    if (contract.kind == IREE_XDNA_ELF_BINDING_KIND_NONE) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_capture_binding(
        function, iree_hal_executable_queue_family(executable), i, &contract,
        bindings.values[i], &out_bindings[i]));
  }
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_function_resolve_binding_addresses(
    iree_hal_amd_xdna_function_t* function,
    iree_hal_amd_xdna_executable_binding_t* bindings) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(function->image);
  for (uint32_t i = 0; i < function->record.binding_count; ++i) {
    const iree_xdna_elf_binding_record_t contract =
        iree_hal_amd_xdna_image_tables_binding(
            tables, function->record.first_binding + i);
    if (contract.kind == IREE_XDNA_ELF_BINDING_KIND_NONE) {
      continue;
    }
    iree_hal_amd_xdna_executable_binding_t* binding = &bindings[i];
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_load_device_address(
        binding->buffer_ref, binding->slot, &binding->device_address));
    if ((binding->device_address & (contract.minimum_alignment - 1)) != 0) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA binding %u device address 0x%" PRIx64
                              " does not satisfy %" PRIu64 "-byte alignment",
                              i, binding->device_address,
                              contract.minimum_alignment);
    }
    if (binding->byte_length != 0 &&
        binding->device_address > UINT64_MAX - (binding->byte_length - 1)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA binding %u native range overflows", i);
    }
  }
  for (uint32_t i = 0; i < function->record.dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, function->record.first_dynamic_relocation + i);
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_executable_storage_validate_relocation(
            &relocation, bindings[relocation.source_ordinal].device_address));
  }
  return iree_ok_status();
}

void iree_hal_amd_xdna_invocation_release(
    iree_hal_amd_xdna_invocation_t* invocation) {
  if (!invocation) {
    return;
  }
  iree_hal_amd_xdna_function_t* function = invocation->function;
  iree_slim_mutex_lock(&function->pool_mutex);
  invocation->next_available = function->available;
  function->available = invocation;
  iree_slim_mutex_unlock(&function->pool_mutex);
}

iree_status_t iree_hal_amd_xdna_function_prepare(
    iree_hal_amd_xdna_function_t* function,
    iree_hal_amd_xdna_executable_binding_t* bindings,
    iree_hal_amd_xdna_invocation_t** out_invocation,
    amdf_xdna_kernel_command_t* out_command) {
  *out_invocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_function_resolve_binding_addresses(function, bindings));
  iree_slim_mutex_lock(&function->pool_mutex);
  iree_hal_amd_xdna_invocation_t* invocation = function->available;
  if (invocation) {
    function->available = invocation->next_available;
  }
  iree_slim_mutex_unlock(&function->pool_mutex);
  if (!invocation) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_invocation_create(function, &invocation));
    iree_slim_mutex_lock(&function->pool_mutex);
    invocation->next = function->invocations;
    function->invocations = invocation;
    iree_slim_mutex_unlock(&function->pool_mutex);
  }
  iree_hal_amd_xdna_executable_storage_patch(function->image, function->ordinal,
                                             invocation->storage, bindings);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < function->record.allocation_use_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_xdna_allocation_plan_t* plan = &function->plans[i];
    if (plan->patches.end) {
      const iree_hal_amd_xdna_owned_allocation_t* allocation =
          &invocation->allocations[i];
      amdf_host_mapping_t* mapping = allocation->command.mapping
                                         ? allocation->command.mapping
                                         : allocation->memory.mapping;
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          function->context->api->host_mapping_cache_control(
              mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
              invocation->storage[i].memory_byte_offset + plan->patches.begin,
              plan->patches.end - plan->patches.begin),
          "host_mapping_cache_control(bind)");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_invocation = invocation;
    *out_command = invocation->command;
  } else {
    iree_hal_amd_xdna_invocation_release(invocation);
  }
  return status;
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
