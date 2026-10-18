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
    case AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE:
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
  if (!amdf_status_is_ok(profile->availability)) {
    return profile->availability;
  }
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
  if (create_info->reserved != 0 ||
      (create_info->required_flags & ~AMDF_MEMORY_BACKING_FLAGS) != 0 ||
      create_info->byte_length == 0 ||
      create_info->placement.kind > AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA ||
      (create_info->placement.kind == AMDF_MEMORY_PLACEMENT_KIND_NONE &&
       create_info->placement.node != 0)) {
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
    case AMDF_MEMORY_SITE_KIND_FILE_IO: {
      if (site->value.host.mapping == NULL) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      const amdf_host_mapping_t* mapping = site->value.host.mapping;
      if (site->kind == AMDF_MEMORY_SITE_KIND_FILE_IO &&
          (mapping->info.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO) == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      }
      if (mapping->memory != NULL) {
        if (site->value.host.source_memory != NULL) {
          return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
        }
        *out_memory = mapping->memory;
      } else {
        amdf_memory_t* source = site->value.host.source_memory;
        if (source == NULL) {
          return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
        }
        if ((source->info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) == 0 ||
            (mapping->info.flags & ~source->host_mapping.supported_access) !=
                0) {
          return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
        }
        *out_memory = source;
      }
      break;
    }
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_allocate(
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_device_access_t* accesses, amdf_memory_t** out_memory) {
  amdf_memory_t* memory = NULL;
  const amdf_status_t status =
      amdf_memory_resource_allocate(plan->host_allocator, plan->access_count,
                                    plan->backing_access_ordinal, &memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
    memory->accesses[i].describe_site =
        plan->native_profiles[i].visibility.describe_site;
  }
  *out_memory = memory;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_prepare_import_access(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_native_import_origin_t origin,
    amdf_memory_flags_t required_flags, uint64_t minimum_alignment,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info) {
  const amdf_memory_native_import_info_t native_info = {
      .device_access = accesses[ordinal].requirements.access,
      .required_flags =
          required_flags |
          amdf_memory_scope_plan_required_access_flags(plan, accesses, ordinal),
      .origin = origin,
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
      .access_count = ordinal == plan->coordinated.ordinals[0]
                          ? plan->coordinated.count
                          : 1,
      .access_ordinals = ordinal == plan->coordinated.ordinals[0]
                             ? plan->coordinated.ordinals
                             : &ordinal,
      .profiles = plan->native_profiles,
  };
  amdf_memory_native_create_info_t native_info = {
      .device_access = requirements->access,
      .required_flags = create_info->required_flags |
                        amdf_memory_scope_plan_required_access_flags(
                            plan, create_info->accesses, ordinal),
      .byte_length = create_info->byte_length,
      .minimum_alignment = create_info->minimum_alignment,
      .registered_host_pointer = create_info->registered_host_pointer,
  };
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
  const uint32_t backing = plan->backing_access_ordinal;
  if (backing == plan->access_count) {
    amdf_status_t status = amdf_host_memory_prepare(
        memory, &plan->native_profiles[backing], create_info, &memory->info);
    amdf_memory_create_info_t registration = *create_info;
    if (amdf_status_is_ok(status)) {
      registration.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      registration.registered_host_pointer = amdf_host_memory_pointer(memory);
      registration.registered_host_cacheability =
          AMDF_HOST_CACHEABILITY_WRITE_BACK;
    }
    const amdf_platform_host_export_t* service =
        plan->shared_external_type != 0
            ? amdf_platform_instance_host_export(
                  amdf_memory_scope_instance(plan->scope)->platform)
            : NULL;
    for (uint32_t i = 0; amdf_status_is_ok(status) && i < plan->access_count;
         ++i) {
      if (plan->native_owner_ordinals[i] != i) {
        continue;
      }
      amdf_memory_info_t native_info = {0};
      if (service != NULL) {
        amdf_external_memory_t exported = {0};
        status = amdf_host_memory_export(memory, service, i == 0, &exported);
        if (amdf_status_is_ok(status)) {
          status = amdf_memory_prepare_import_access(
              memory, i, plan, create_info->accesses,
              AMDF_MEMORY_NATIVE_IMPORT_ORIGIN_CONSTRUCTION, 0,
              create_info->minimum_alignment, &exported, &native_info);
        }
        amdf_external_memory_release(&exported);
      } else {
        status = amdf_memory_prepare_access(memory, i, plan, &registration,
                                            &native_info);
      }
      if (amdf_status_is_ok(status) &&
          native_info.alignment < memory->info.alignment) {
        memory->info.alignment = native_info.alignment;
      }
    }
    return status;
  }
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
    if (plan->native_owner_ordinals[i] == backing) {
      continue;
    }
    amdf_memory_info_t native_info = {0};
    if (plan->shared_external_type != 0) {
      status = amdf_memory_prepare_import_access(
          memory, i, plan, create_info->accesses,
          AMDF_MEMORY_NATIVE_IMPORT_ORIGIN_CONSTRUCTION, 0,
          create_info->minimum_alignment, &shared_memory, &native_info);
    } else {
      status = amdf_memory_prepare_access(memory, i, plan, create_info,
                                          &native_info);
    }
    if (amdf_status_is_ok(status) &&
        native_info.alignment < memory->info.alignment) {
      memory->info.alignment = native_info.alignment;
    }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_memory_access_query_t query = {
      .count = create_info->access_count,
      .accesses = create_info->accesses,
  };
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(
      scope, create_info->memory_profile_ordinal, &query, NULL, &plan);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
      create_info->placement.kind != AMDF_MEMORY_PLACEMENT_KIND_NONE &&
      (capabilities->placement_kinds &
       (UINT32_C(1) << create_info->placement.kind)) == 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status) &&
      registration != (create_info->registered_host_pointer != NULL)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_scope_plan_validate_registration(
        &plan, create_info->registered_host_cacheability);
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
    const amdf_status_t release_status = amdf_memory_resource_destroy(memory);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_external_memory_validate(inout_external_memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_memory_access_query_t query = {
      .count = import_info->access_count,
      .accesses = import_info->accesses,
  };
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(
      scope, import_info->memory_profile_ordinal, &query, inout_external_memory,
      &plan);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
        memory, i, &plan, import_info->accesses,
        AMDF_MEMORY_NATIVE_IMPORT_ORIGIN_EXTERNAL, import_info->required_flags,
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
    const amdf_status_t release_status = amdf_memory_resource_destroy(memory);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
  if (!amdf_status_is_ok(validation_status)) {
    return validation_status;
  }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_EXPORT) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_external_memory_support_t support;
  status = amdf_memory_query_external_support(
      profile.external_memory_support_count, profile.external_memory_support,
      export_info->external_memory_type,
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT, source_byte_offset,
      export_info->byte_length, &support);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

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
  if (value == NULL) {
    return;
  }
  if (value->release != NULL) {
    value->release(value->release_user_data, value->type, value->payload);
  }
  memset(value, 0, sizeof(*value));
}

static amdf_status_t amdf_memory_describe_site(
    const amdf_memory_site_t* site, const amdf_memory_site_t* peer,
    const amdf_memory_site_description_t* peer_description,
    amdf_memory_site_description_t* out_description) {
  if (site->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    amdf_memory_t* memory = site->value.device.memory;
    const uint32_t ordinal = site->value.device.access_ordinal;
    return memory->accesses[ordinal].vtable->describe_site(
        memory, ordinal, site->value.device.queue_family_ordinal,
        out_description);
  }

  const amdf_host_mapping_info_t* mapping = &site->value.host.mapping->info;
  if (site->kind == AMDF_MEMORY_SITE_KIND_FILE_IO) {
    *out_description = amdf_memory_describe_file_site(mapping->flags);
    return AMDF_STATUS_OK;
  }
  const bool coherent =
      peer->kind != AMDF_MEMORY_SITE_KIND_DEVICE ||
      (peer->value.device.memory->accesses[peer->value.device.access_ordinal]
           .info.flags &
       AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  const amdf_memory_host_description_t host = {
      .cacheability = mapping->cacheability,
      .cache_line_size = mapping->cache_line_size,
      .flush = mapping->flush,
      .invalidate = mapping->invalidate,
  };
  *out_description = amdf_memory_describe_host_site(
      &host, mapping->flags, coherent,
      peer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ? peer_description : NULL);
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_memory_validate_site(consumer_site, &consumer_memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      (uint32_t)sizeof(amdf_memory_pair_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

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

  // Describe a device before its host peer so the host can use the qualified
  // SYSTEM domain without querying the device a second time.
  const bool consumer_first =
      producer_site->kind != AMDF_MEMORY_SITE_KIND_DEVICE;
  const amdf_memory_site_t* first_site =
      consumer_first ? consumer_site : producer_site;
  const amdf_memory_site_t* second_site =
      consumer_first ? producer_site : consumer_site;
  amdf_memory_site_description_t first = {0};
  status = amdf_memory_describe_site(first_site, second_site, NULL, &first);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_memory_site_description_t second = {0};
  status = amdf_memory_describe_site(second_site, first_site, &first, &second);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  return amdf_memory_pair_compose(consumer_first ? &second : &first,
                                  consumer_first ? &first : &second, out_info);
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
    mapping->info.features = capabilities->features;
    *out_mapping = mapping;
  }
  return map_status;
}

amdf_status_t AMDF_CALL amdf_memory_destroy(amdf_memory_t* memory) {
  if (memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_memory_resource_destroy(memory);
}
