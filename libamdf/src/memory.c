// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/structure.h"

static bool amdf_memory_is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static bool amdf_external_memory_type_is_valid(
    amdf_external_memory_type_t type) {
  return type >= AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
         type <= AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS;
}

static amdf_status_t amdf_external_memory_validate(
    const amdf_external_memory_t* value) {
  if (value == NULL || !amdf_external_memory_type_is_valid(value->type) ||
      value->reserved != 0 || value->byte_length == 0 ||
      value->source_byte_offset > UINT64_MAX - value->byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  switch (value->type) {
    case AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD:
      if (value->payload.file_descriptor < 0 ||
          amdf_external_memory_provenance_is_valid(&value->provenance)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD:
      if (value->payload.file_descriptor < 0 ||
          !amdf_external_memory_provenance_is_valid(&value->provenance)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_NT_HANDLE:
      if (value->payload.native_handle == NULL ||
          amdf_external_memory_provenance_is_valid(&value->provenance)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_HOST_POINTER:
      if (value->payload.host_pointer == NULL ||
          amdf_external_memory_provenance_is_valid(&value->provenance)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS:
      if (value->payload.device_address == 0 ||
          !amdf_external_memory_provenance_is_valid(&value->provenance)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      break;
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static const amdf_external_memory_support_t*
amdf_memory_profile_find_external_support(
    uint32_t count, const amdf_external_memory_support_t* supports,
    amdf_external_memory_type_t type) {
  for (uint32_t i = 0; i < count; ++i) {
    if (supports[i].type == type) {
      return &supports[i];
    }
  }
  return NULL;
}

static amdf_status_t amdf_memory_validate_external_range(
    const amdf_external_memory_support_t* support, uint64_t source_byte_offset,
    uint64_t byte_length) {
  if (source_byte_offset != 0 &&
      ((support->flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET) ==
           0 ||
       support->source_offset_alignment == 0 ||
       source_byte_offset % support->source_offset_alignment != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_assert(support->byte_length_alignment != 0 &&
              "external memory byte-length alignment must be nonzero");
  if (byte_length % support->byte_length_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (support->maximum_byte_length != 0 &&
      byte_length > support->maximum_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_validate_profile_request(
    const amdf_memory_profile_t* profile,
    amdf_memory_profile_roles_t required_role,
    const amdf_memory_construction_capabilities_t* capabilities,
    amdf_memory_flags_t required_flags, uint64_t byte_length,
    uint64_t minimum_alignment) {
  if ((profile->roles & required_role) == 0 ||
      (required_flags & ~profile->supported_flags) != 0 ||
      capabilities->maximum_byte_length == 0 ||
      byte_length > capabilities->maximum_byte_length ||
      capabilities->byte_length_granularity == 0 ||
      byte_length % capabilities->byte_length_granularity != 0 ||
      (minimum_alignment != 0 &&
       (capabilities->maximum_alignment == 0 ||
        minimum_alignment > capabilities->maximum_alignment))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_query_external_support(
    uint32_t count, const amdf_external_memory_support_t* supports,
    amdf_external_memory_type_t external_memory_type,
    amdf_external_memory_support_flags_t required_support_flag,
    uint64_t source_byte_offset, uint64_t byte_length,
    amdf_external_memory_support_t* out_support) {
  const amdf_external_memory_support_t* support =
      amdf_memory_profile_find_external_support(count, supports,
                                                external_memory_type);
  if (support == NULL || (support->flags & required_support_flag) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_status_t status = amdf_memory_validate_external_range(
      support, source_byte_offset, byte_length);
  if (amdf_status_is_ok(status)) {
    *out_support = *support;
  }
  return status;
}

static amdf_status_t amdf_memory_validate_create_info(
    const amdf_memory_create_info_t* create_info) {
  const amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      (uint32_t)sizeof(amdf_memory_create_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if ((create_info->required_flags & ~AMDF_MEMORY_BACKING_FLAGS) != 0 ||
      create_info->byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (create_info->minimum_alignment != 0 &&
      !amdf_memory_is_power_of_two(create_info->minimum_alignment)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_validate_import_info(
    const amdf_memory_import_info_t* import_info) {
  const amdf_status_t status = amdf_structure_validate_input(
      import_info, AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
      (uint32_t)sizeof(amdf_memory_import_info_t));
  if (!amdf_status_is_ok(status)) return status;
  if ((import_info->required_flags & ~AMDF_MEMORY_BACKING_FLAGS) != 0 ||
      (import_info->minimum_alignment != 0 &&
       !amdf_memory_is_power_of_two(import_info->minimum_alignment))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_validate_export_info(
    const amdf_memory_t* memory, const amdf_memory_export_info_t* export_info) {
  const amdf_status_t status = amdf_structure_validate_input(
      export_info, AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      (uint32_t)sizeof(amdf_memory_export_info_t));
  if (!amdf_status_is_ok(status)) return status;
  if (!amdf_external_memory_type_is_valid(export_info->external_memory_type) ||
      export_info->reserved != 0 || export_info->byte_length == 0 ||
      export_info->byte_offset > memory->info.byte_length ||
      export_info->byte_length >
          memory->info.byte_length - export_info->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_validate_site(const amdf_memory_site_t* site,
                                               amdf_memory_t** out_memory) {
  const amdf_status_t status =
      amdf_structure_validate_input(site, AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                    (uint32_t)sizeof(amdf_memory_site_t));
  if (!amdf_status_is_ok(status)) return status;
  if (site->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  switch (site->kind) {
    case AMDF_MEMORY_SITE_KIND_DEVICE:
      if (site->value.device.memory == NULL) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      if (site->value.device.access_ordinal >=
          site->value.device.memory->info.access_count) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      *out_memory = site->value.device.memory;
      break;
    case AMDF_MEMORY_SITE_KIND_HOST:
      if (site->value.host_mapping == NULL) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      *out_memory = site->value.host_mapping->memory;
      break;
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_allocate(
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_device_access_t* accesses, amdf_memory_t** out_memory) {
  amdf_memory_t* memory = NULL;
  const amdf_status_t status = amdf_memory_resource_allocate(
      plan->host_allocator, plan->access_count, &memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->backing_access_ordinal = plan->backing_access_ordinal;
  memory->scope = plan->scope;
  memory->host_mapping = plan->profile.host_mapping;
  for (uint32_t i = 0; i < plan->access_count; ++i) {
    memory->accesses[i].device = accesses[i].device;
    memory->accesses[i].native_owner_ordinal = plan->native_owner_ordinals[i];
    memory->accesses[i].info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    memory->accesses[i].info.structure_size = sizeof(memory->accesses[i].info);
    memory->accesses[i].info.ordinal = i;
    memory->accesses[i].native_profile_ordinal =
        plan->native_profiles[i].ordinal;
  }
  *out_memory = memory;
  return AMDF_STATUS_OK;
}

static amdf_memory_flags_t amdf_memory_required_access_flags(
    const amdf_memory_access_requirements_t* requirements) {
  return requirements->flags |
         (requirements->address_kinds != 0 ? AMDF_MEMORY_FLAG_DEVICE_ADDRESS
                                           : 0);
}

static amdf_status_t amdf_memory_prepare_import_access(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_flags_t required_flags, uint64_t minimum_alignment,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info) {
  const amdf_memory_native_import_info_t native_info = {
      .device_access = accesses[ordinal].requirements.access,
      .required_flags = required_flags | amdf_memory_required_access_flags(
                                             &accesses[ordinal].requirements),
      .minimum_alignment = minimum_alignment,
  };
  return accesses[ordinal].device->vtable->memory_prepare_import(
      memory, ordinal, &plan->native_profiles[ordinal], &native_info,
      external_memory, out_info);
}

static amdf_status_t amdf_memory_prepare_access(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  const amdf_memory_access_requirements_t* requirements =
      &create_info->accesses[ordinal].requirements;
  const amdf_memory_native_group_t group = {
      .access_count = ordinal == plan->backing_access_ordinal
                          ? plan->backing_access_count
                          : 1,
      .access_ordinals = ordinal == plan->backing_access_ordinal
                             ? plan->backing_access_ordinals
                             : &ordinal,
      .profiles = plan->native_profiles,
  };
  amdf_memory_native_create_info_t native_info = {
      .device_access = requirements->access,
      .required_flags = create_info->required_flags |
                        amdf_memory_required_access_flags(requirements),
      .byte_length = create_info->byte_length,
      .minimum_alignment = create_info->minimum_alignment,
      .registered_host_pointer = create_info->registered_host_pointer,
  };
  for (uint32_t i = 1; i < group.access_count; ++i) {
    native_info.required_flags |= amdf_memory_required_access_flags(
        &create_info->accesses[group.access_ordinals[i]].requirements);
  }
  if (plan->scope->kind == AMDF_MEMORY_SCOPE_KIND_PRIVATE) {
    return plan->scope->owner.private_storage.vtable->prepare(
        plan->scope, memory, &plan->native_profiles[ordinal], &native_info,
        out_info);
  }
  return create_info->accesses[ordinal].device->vtable->memory_prepare(
      memory, &group, &native_info, out_info);
}

static amdf_status_t amdf_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_scope_plan_t* plan,
    const amdf_memory_create_info_t* create_info) {
  if (plan->access_count == 0) {
    return amdf_host_memory_prepare(memory, &plan->native_profiles[0],
                                    create_info, &memory->info);
  }
  const uint32_t backing = plan->backing_access_ordinal;
  amdf_status_t status = amdf_memory_prepare_access(memory, backing, plan,
                                                    create_info, &memory->info);
  if (amdf_status_is_ok(status)) {
    // Native allocation rounding is not an access guarantee for the other
    // consumers. Every access and export names exactly the requested range.
    memory->info.byte_length = create_info->byte_length;
  }
  amdf_external_memory_t shared_memory = {0};
  if (amdf_status_is_ok(status) && plan->shared_external_type != 0) {
    const amdf_memory_export_info_t export_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
        .structure_size = sizeof(export_info),
        .external_memory_type = plan->shared_external_type,
        .byte_length = create_info->byte_length,
    };
    status = amdf_memory_export(memory, &export_info, &shared_memory);
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < plan->access_count;
       ++i) {
    if (plan->native_owner_ordinals[i] == backing) continue;
    amdf_memory_info_t native_info = {0};
    if (plan->shared_external_type != 0) {
      status = amdf_memory_prepare_import_access(
          memory, i, plan, create_info->accesses, 0,
          create_info->minimum_alignment, &shared_memory, &native_info);
    } else {
      status = amdf_memory_prepare_access(memory, i, plan, create_info,
                                          &native_info);
    }
    if (amdf_status_is_ok(status) &&
        native_info.alignment < memory->info.alignment)
      memory->info.alignment = native_info.alignment;
  }
  amdf_external_memory_release(&shared_memory);
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_create(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* create_info,
    amdf_memory_t** out_memory) {
  if (scope == NULL || out_memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_memory_validate_create_info(create_info);
  if (!amdf_status_is_ok(status)) return status;
  const amdf_memory_access_query_t query = {
      .kind = AMDF_MEMORY_ACCESS_QUERY_LIVE,
      .count = create_info->access_count,
      .accesses.devices = create_info->accesses,
  };
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(
      scope, create_info->memory_profile_ordinal, &query, NULL, &plan);
  if (!amdf_status_is_ok(status)) return status;
  const bool registration =
      (plan.profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  const amdf_memory_construction_capabilities_t* capabilities =
      registration ? &plan.profile.registration : &plan.profile.allocation;
  status = amdf_memory_validate_profile_request(
      &plan.profile,
      registration ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
                   : AMDF_MEMORY_PROFILE_ROLE_CREATE,
      capabilities, create_info->required_flags, create_info->byte_length,
      create_info->minimum_alignment);
  if (amdf_status_is_ok(status) &&
      registration != (create_info->registered_host_pointer != NULL)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_status_is_ok(status) && registration) {
    const uintptr_t pointer = (uintptr_t)create_info->registered_host_pointer;
    if (pointer % capabilities->registered_host_pointer_alignment != 0 ||
        (create_info->minimum_alignment != 0 &&
         (pointer & (create_info->minimum_alignment - 1)) != 0)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else if (create_info->byte_length > UINTPTR_MAX - pointer) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  amdf_memory_t* memory = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_allocate(&plan, create_info->accesses, &memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_prepare(memory, &plan, create_info);
  }
  if (amdf_status_is_ok(status)) {
    memory->info.memory_profile_ordinal = plan.profile.ordinal;
    *out_memory = memory;
  } else if (memory != NULL) {
    const amdf_status_t release_status = amdf_memory_discard(memory);
    if (!amdf_status_is_ok(release_status)) status = release_status;
  }
  amdf_memory_scope_plan_deinitialize(&plan);
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_import(
    amdf_memory_scope_t* scope, const amdf_memory_import_info_t* import_info,
    amdf_external_memory_t* inout_external_memory, amdf_memory_t** out_memory) {
  if (scope == NULL || out_memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_memory_validate_import_info(import_info);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_external_memory_validate(inout_external_memory);
  if (!amdf_status_is_ok(status)) return status;
  const amdf_memory_access_query_t query = {
      .kind = AMDF_MEMORY_ACCESS_QUERY_LIVE,
      .count = import_info->access_count,
      .accesses.devices = import_info->accesses,
  };
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(
      scope, import_info->memory_profile_ordinal, &query, inout_external_memory,
      &plan);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_memory_validate_profile_request(
      &plan.profile, AMDF_MEMORY_PROFILE_ROLE_IMPORT, &plan.profile.import,
      import_info->required_flags, inout_external_memory->byte_length,
      import_info->minimum_alignment);
  if (amdf_status_is_ok(status) && import_info->minimum_alignment != 0 &&
      (inout_external_memory->source_byte_offset &
       (import_info->minimum_alignment - 1)) != 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_external_memory_support_t support;
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_query_external_support(
        plan.profile.external_memory_support_count,
        plan.profile.external_memory_support, inout_external_memory->type,
        AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT,
        inout_external_memory->source_byte_offset,
        inout_external_memory->byte_length, &support);
  }
  amdf_memory_t* memory = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_allocate(&plan, import_info->accesses, &memory);
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < plan.access_count;
       ++i) {
    amdf_memory_info_t native_info = {0};
    status = amdf_memory_prepare_import_access(
        memory, i, &plan, import_info->accesses, import_info->required_flags,
        import_info->minimum_alignment, inout_external_memory,
        i == 0 ? &memory->info : &native_info);
    if (amdf_status_is_ok(status) && i != 0 &&
        native_info.alignment < memory->info.alignment) {
      memory->info.alignment = native_info.alignment;
    }
  }
  if (amdf_status_is_ok(status)) {
    memory->info.memory_profile_ordinal = plan.profile.ordinal;
    amdf_external_memory_release(inout_external_memory);
    *out_memory = memory;
  } else if (memory != NULL) {
    const amdf_status_t release_status = amdf_memory_discard(memory);
    if (!amdf_status_is_ok(release_status)) status = release_status;
  }
  amdf_memory_scope_plan_deinitialize(&plan);
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_query_address(
    amdf_memory_t* memory, uint32_t access_ordinal,
    amdf_memory_address_kind_t kind, uint64_t* out_address) {
  if (memory == NULL || out_address == NULL ||
      kind > AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (access_ordinal >= memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_access_state_t* access = &memory->accesses[access_ordinal];
  if ((access->info.address_kinds & (UINT64_C(1) << kind)) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_address = access->addresses[kind];
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_memory_query_access_info(amdf_memory_t* memory, uint32_t access_ordinal,
                              amdf_memory_access_info_t* out_info) {
  if (memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      (uint32_t)sizeof(amdf_memory_access_info_t));
  if (!amdf_status_is_ok(status)) return status;
  if (access_ordinal >= memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = memory->accesses[access_ordinal].info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_memory_query_info(amdf_memory_t* memory,
                                               amdf_memory_info_t* out_info) {
  if (memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status =
      amdf_structure_validate_output(out_info, AMDF_STRUCTURE_TYPE_MEMORY_INFO,
                                     (uint32_t)sizeof(amdf_memory_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = memory->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_memory_export(
    amdf_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  if (out_value == NULL || memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t validation_status =
      amdf_memory_validate_export_info(memory, export_info);
  if (!amdf_status_is_ok(validation_status)) return validation_status;
  if ((memory->info.flags & AMDF_MEMORY_FLAG_SHAREABLE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  if (memory->info.memory_profile_ordinal ==
      AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t source_byte_offset =
      memory->info.source_byte_offset + export_info->byte_offset;
  const uint32_t ordinal = memory->backing_access_ordinal;
  amdf_memory_access_state_t* backing = &memory->accesses[ordinal];
  amdf_memory_native_profile_t profile;
  amdf_status_t status = backing->device->vtable->query_memory_profile(
      backing->device, backing->native_profile_ordinal, &profile);
  if (!amdf_status_is_ok(status)) return status;
  if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_EXPORT) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_external_memory_support_t support;
  status = amdf_memory_query_external_support(
      profile.external_memory_support_count, profile.external_memory_support,
      export_info->external_memory_type,
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT, source_byte_offset,
      export_info->byte_length, &support);
  if (!amdf_status_is_ok(status)) return status;

  amdf_external_memory_t value = {0};
  status =
      backing->vtable->export_external(memory, ordinal, export_info, &value);
  if (amdf_status_is_ok(status)) {
    amdf_assert(value.release != NULL &&
                "successful memory export must own its payload lifetime");
    value.type = export_info->external_memory_type;
    value.reserved = 0;
    value.provenance = support.provenance;
    value.source_byte_offset = source_byte_offset;
    value.byte_length = export_info->byte_length;
    value.physical_backing_id = memory->info.physical_backing_id;
    *out_value = value;
  }
  return status;
}

void AMDF_CALL amdf_external_memory_release(amdf_external_memory_t* value) {
  if (value == NULL) return;
  if (value->release != NULL) {
    value->release(value->release_user_data, value->type, value->payload);
  }
  memset(value, 0, sizeof(*value));
}

// Host visibility depends on the exact peer, not the resource's other accesses.
// Native API operations remain required even when CPU lines are coherent: they
// can also publish an allocation to the native device driver.
static amdf_cache_transition_t amdf_memory_host_transition(
    const amdf_cache_transition_t* available,
    amdf_host_cacheability_t cacheability, bool coherent) {
  if (coherent && cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK &&
      available->executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) {
    return (amdf_cache_transition_t){.kind = AMDF_CACHE_TRANSITION_KIND_NONE};
  }
  return *available;
}

static amdf_status_t amdf_memory_describe_site(
    const amdf_memory_site_t* site, const amdf_memory_site_t* peer,
    amdf_memory_site_description_t* out_description) {
  if (site->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    amdf_memory_t* memory = site->value.device.memory;
    const uint32_t ordinal = site->value.device.access_ordinal;
    return memory->accesses[ordinal].vtable->describe_site(
        memory, ordinal, site->value.device.queue_family_ordinal,
        out_description);
  }

  const amdf_host_mapping_info_t* mapping = &site->value.host_mapping->info;
  const bool coherent =
      peer->kind == AMDF_MEMORY_SITE_KIND_HOST ||
      (peer->value.device.memory->accesses[peer->value.device.access_ordinal]
           .info.flags &
       AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  amdf_memory_site_description_t description = {0};
  if ((mapping->flags & AMDF_MEMORY_MAP_FLAG_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((mapping->flags & AMDF_MEMORY_MAP_FLAG_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  description.release = amdf_memory_host_transition(
      &mapping->flush, mapping->cacheability, coherent);
  description.acquire = amdf_memory_host_transition(
      &mapping->invalidate, mapping->cacheability, coherent);
  if (description.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN;
  }
  if (description.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
  }
  *out_description = description;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_memory_query_pair_info(const amdf_memory_site_t* producer_site,
                            const amdf_memory_site_t* consumer_site,
                            amdf_memory_pair_info_t* out_info) {
  amdf_memory_t* producer_memory = NULL;
  amdf_memory_t* consumer_memory = NULL;
  amdf_status_t status =
      amdf_memory_validate_site(producer_site, &producer_memory);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_memory_validate_site(consumer_site, &consumer_memory);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      (uint32_t)sizeof(amdf_memory_pair_info_t));
  if (!amdf_status_is_ok(status)) return status;

  const amdf_physical_memory_id_t* producer_id =
      &producer_memory->info.physical_backing_id;
  const amdf_physical_memory_id_t* consumer_id =
      &consumer_memory->info.physical_backing_id;
  if (producer_memory != consumer_memory) {
    // CPU-only owners have no external backing identity. Device-backed owners
    // reach their instance through the already-borrowed backing access.
    if (!amdf_physical_memory_id_is_valid(producer_id) ||
        !amdf_physical_memory_id_is_valid(consumer_id)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (!amdf_device_shares_provider_instance(
            producer_memory->accesses[producer_memory->backing_access_ordinal]
                .device,
            consumer_memory->accesses[consumer_memory->backing_access_ordinal]
                .device) ||
        !amdf_physical_memory_id_is_equal(producer_id, consumer_id)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
    }
  }

  amdf_memory_site_description_t producer = {0};
  status = amdf_memory_describe_site(producer_site, consumer_site, &producer);
  if (!amdf_status_is_ok(status)) return status;
  amdf_memory_site_description_t consumer = {0};
  status = amdf_memory_describe_site(consumer_site, producer_site, &consumer);
  if (!amdf_status_is_ok(status)) return status;

  if ((producer.capabilities & AMDF_MEMORY_SITE_CAPABILITY_WRITE) == 0 ||
      (consumer.capabilities & AMDF_MEMORY_SITE_CAPABILITY_READ) == 0 ||
      producer.release.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN ||
      consumer.acquire.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_memory_pair_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = out_info->structure_size,
      .next = out_info->next,
      .flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
      .release = producer.release,
      .acquire = consumer.acquire,
  };
  if ((producer.capabilities & AMDF_MEMORY_SITE_CAPABILITY_MAPPING_SOURCE) !=
          0 &&
      (consumer.capabilities & AMDF_MEMORY_SITE_CAPABILITY_MAPPING_TARGET) !=
          0 &&
      amdf_memory_compatibility_domain_is_valid(&producer.mapping_domain) &&
      amdf_memory_compatibility_domain_is_equal(&producer.mapping_domain,
                                                &consumer.mapping_domain)) {
    info.flags |= AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE;
  }
  if (amdf_memory_compatibility_domain_is_valid(&producer.atomic_domain) &&
      amdf_memory_compatibility_domain_is_equal(&producer.atomic_domain,
                                                &consumer.atomic_domain)) {
    info.atomic_reach.scope_32 =
        producer.atomic_reach.scope_32 < consumer.atomic_reach.scope_32
            ? producer.atomic_reach.scope_32
            : consumer.atomic_reach.scope_32;
    info.atomic_reach.scope_64 =
        producer.atomic_reach.scope_64 < consumer.atomic_reach.scope_64
            ? producer.atomic_reach.scope_64
            : consumer.atomic_reach.scope_64;
  }
  if ((producer.capabilities &
       AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN) != 0 &&
      (consumer.capabilities &
       AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN) != 0) {
    if (producer.release_fixed_cost_nanoseconds >
        UINT64_MAX - consumer.acquire_fixed_cost_nanoseconds) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    info.flags |= AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN;
    info.estimated_fixed_cost_nanoseconds =
        producer.release_fixed_cost_nanoseconds +
        consumer.acquire_fixed_cost_nanoseconds;
  }
  *out_info = info;
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_map(amdf_memory_t* memory,
                                        const amdf_memory_map_info_t* map_info,
                                        amdf_host_mapping_t** out_mapping) {
  if (out_mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_input(
      map_info, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      (uint32_t)sizeof(amdf_memory_map_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_memory_map_flags_t known_flags =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  if (map_info->flags == 0 || (map_info->flags & ~known_flags) != 0 ||
      map_info->byte_length == 0 ||
      map_info->byte_offset > memory->info.byte_length ||
      map_info->byte_length >
          memory->info.byte_length - map_info->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_host_mapping_capabilities_t* capabilities = &memory->host_mapping;
  if ((map_info->flags & ~capabilities->supported_access) != 0 ||
      capabilities->maximum_byte_length == 0 ||
      map_info->byte_length > capabilities->maximum_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (map_info->byte_offset % capabilities->byte_offset_granularity != 0 ||
      map_info->byte_length % capabilities->byte_length_granularity != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_host_mapping_t* mapping = NULL;
  const uint32_t ordinal = memory->backing_access_ordinal;
  const amdf_status_t map_status = memory->accesses[ordinal].vtable->map(
      memory, ordinal, capabilities, map_info, &mapping);
  if (amdf_status_is_ok(map_status)) {
    amdf_assert(mapping != NULL &&
                "successful memory map must return a mapping");
    *out_mapping = mapping;
  }
  return map_status;
}

amdf_status_t AMDF_CALL amdf_memory_destroy(amdf_memory_t* memory) {
  if (memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&memory->children) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_memory_release_native(memory);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = memory->host_allocator;
    amdf_free(host_allocator, memory);
  }
  return status;
}
