// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/source.h"

#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <string.h>
#include <sys/random.h>
#include <sys/resource.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/umd/rocjitsu/device.h"
#include "libamdf/src/gpu/umd/rocjitsu/event.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/structure.h"

static amdf_status_t amdf_rocjitsu_source_destroy(
    amdf_rocjitsu_source_t* source) {
  if (source->thread_started) {
    source->library.rj_vm_request_exit(source->machine,
                                       "libamdf instance teardown");
    const int error = pthread_join(source->thread, NULL);
    if (error != 0) {
      return amdf_linux_error(error);
    }
    source->thread_started = false;
  }
  const amdf_status_t event_status = amdf_rocjitsu_event_source_release(source);
  if (!amdf_status_is_ok(event_status)) {
    return event_status;
  }
  if (source->process != 0) {
    const amdf_status_t status = amdf_rocjitsu_status(
        source->library.rj_vm_device_close(source->machine, source->process));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    source->process = 0;
  }
  if (source->machine != NULL) {
    source->library.rj_vm_destroy(source->machine);
    source->machine = NULL;
  }
  if (source->configuration != NULL) {
    source->library.rj_vm_config_destroy(source->configuration);
    source->configuration = NULL;
  }
  amdf_rocjitsu_library_close(&source->library);
  amdf_free(source->host_allocator, source);
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_rocjitsu_gpu_initialize(
    amdf_rocjitsu_source_t* source, uint32_t source_ordinal,
    uint32_t gpu_ordinal) {
  amdf_rocjitsu_gpu_t* gpu = &source->gpus[gpu_ordinal];
  gpu->source = source;
  amdf_status_t status =
      amdf_rocjitsu_status(source->library.rj_vm_config_gpu_info_at(
          source->configuration, gpu_ordinal, &gpu->native));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const rj_vm_gpu_info_t* native = &gpu->native;
  if (native->present == 0 || native->simd_per_cu == 0 ||
      native->simd_count % native->simd_per_cu != 0 || native->num_xcc == 0 ||
      native->num_shader_engines == 0 ||
      native->max_waves_per_simd > UINT32_MAX / native->simd_per_cu ||
      memchr(native->marketing_name, 0, sizeof(native->marketing_name)) ==
          NULL) {
    return amdf_linux_error(EPROTO);
  }
  const uint32_t major = native->gfx_target_version / 10000;
  const uint32_t minor = (native->gfx_target_version / 100) % 100;
  const uint32_t stepping = native->gfx_target_version % 100;
  const bool cdna = major == 9 && ((minor == 0 && stepping == 10) ||
                                   (minor == 4 && stepping <= 2) ||
                                   (minor == 5 && stepping == 0));
  const bool rdna = major == 11 || (major == 12 && (minor == 0 || minor == 5));
  if ((!cdna && !rdna) || native->wave_front_size != (cdna ? 64u : 32u)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const bool pm4 = major == 11 || (major == 12 && minor == 0);
  amdf_gpu_endpoint_properties_t properties = {
      .gfx_ip = {major, minor, stepping},
      .asic_revision = native->revision_id,
      .unique_id = native->unique_id,
      .compute =
          {
              .wavefront_size = native->wave_front_size,
              .compute_unit_count = native->simd_count / native->simd_per_cu,
              .maximum_wave_count_per_compute_unit =
                  native->max_waves_per_simd * native->simd_per_cu,
              .maximum_scratch_wave_count_per_compute_unit =
                  native->max_slots_scratch_cu,
              .local_data_share_byte_length =
                  (uint64_t)native->lds_size_kb * 1024,
          },
      .topology = {native->num_xcc, native->num_shader_engines},
      .queue_family_count = pm4 ? 1u : 0u,
      .queue_families = {{
          .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
          .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
          .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR |
                             (native->num_xcc == 1
                                  ? AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH
                                  : 0),
          .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
          .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
                   AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
          .atomic_capabilities =
              {
                  .operations_32 =
                      AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
                  .operations_64 = AMDF_ATOMIC_OPERATION_STORE,
                  .operations_without_dispatch_32 =
                      AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
                  .operations_without_dispatch_64 = AMDF_ATOMIC_OPERATION_STORE,
                  .wait_conditions_32 =
                      AMDF_ATOMIC_WAIT_CONDITION_EQUAL |
                      AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL |
                      AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
              },
          .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                              AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
          .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
          .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                     AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
          .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
          .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
          .minimum_ring_byte_length = 4096,
          .maximum_ring_byte_length = UINT64_C(1) << 31,
          .ring_byte_length_alignment = 4096,
      }},
  };
  if (native->num_cp_queues == 0 || !amdf_gpu_aql_descriptor_profile_initialize(
                                        &properties, 0, 0, &gpu->aql)) {
    return amdf_linux_error(EPROTO);
  }
  properties.queue_families[properties.queue_family_count++] =
      (amdf_gpu_queue_family_properties_t){
          .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
          .format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1,
          .format_features = AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE |
                             AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE,
          .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
          .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
                   AMDF_QUEUE_ROLE_CACHE_CONTROL,
          .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                              AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
          .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
          .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                     AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
          .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE |
                            AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI,
          .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
          .minimum_ring_byte_length = 4096,
          .maximum_ring_byte_length = UINT64_C(1) << 31,
          .ring_byte_length_alignment = 4096,
      };
  if (!amdf_gpu_endpoint_profile_initialize(&properties, &gpu->profile)) {
    return amdf_linux_error(EPROTO);
  }
  gpu->info = (amdf_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(gpu->info),
      .id = {{((uint64_t)source_ordinal << 32) | gpu_ordinal,
              source->generation}},
      .engine_kind = AMDF_ENGINE_KIND_GPU,
      .type_flags = AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE |
                    AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY,
      .pci = {.vendor_id = native->vendor_id,
              .device_id = native->device_id,
              .revision_id = native->pci_revision_id},
  };
  const size_t name_length = strlen(native->marketing_name);
  if (name_length >= sizeof(gpu->info.name)) {
    return amdf_linux_error(EOVERFLOW);
  }
  memcpy(gpu->info.name, native->marketing_name, name_length + 1);
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_rocjitsu_source_create(
    amdf_instance_t* instance, uint32_t source_ordinal,
    const amdf_rocjitsu_machine_create_info_t* create_info,
    amdf_rocjitsu_source_t** out_source) {
  const amdf_allocator_t host_allocator = instance->host_allocator;
  amdf_rocjitsu_library_t library = {0};
  amdf_status_t status =
      amdf_rocjitsu_library_open(create_info->library_path, &library);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  rj_vm_config_t* configuration = NULL;
  status = amdf_rocjitsu_status(library.rj_vm_config_create_from_string(
      create_info->configuration_json, &configuration));
  uint32_t gpu_count = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_status(
        library.rj_vm_config_gpu_count(configuration, &gpu_count));
  }
  amdf_rocjitsu_source_t* source = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_calloc(
        host_allocator,
        sizeof(*source) + (size_t)gpu_count * sizeof(source->gpus[0]),
        amdf_alignof(amdf_rocjitsu_source_t), (void**)&source);
  }
  if (!amdf_status_is_ok(status)) {
    if (configuration != NULL) {
      library.rj_vm_config_destroy(configuration);
    }
    amdf_rocjitsu_library_close(&library);
    return status;
  }
  source->host_allocator = host_allocator;
  source->platform = instance->platform;
  source->library = library;
  source->configuration = configuration;
  source->gpu_count = gpu_count;
  amdf_atomic_uint64_initialize(&source->terminal_status, AMDF_STATUS_OK);
  size_t random_length = 0;
  while (amdf_status_is_ok(status) &&
         random_length < sizeof(source->generation)) {
    const ssize_t count =
        getrandom((uint8_t*)&source->generation + random_length,
                  sizeof(source->generation) - random_length, 0);
    if (count < 0 && errno != EINTR) {
      status = amdf_linux_error(errno);
    } else if (count > 0) {
      random_length += (size_t)count;
    }
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < gpu_count; ++i) {
    status = amdf_rocjitsu_gpu_initialize(source, source_ordinal, i);
  }
  if (amdf_status_is_ok(status)) {
    *out_source = source;
  } else {
    const amdf_status_t cleanup_status = amdf_rocjitsu_source_destroy(source);
    if (!amdf_status_is_ok(cleanup_status)) {
      status = cleanup_status;
    }
  }
  return status;
}

static amdf_status_t AMDF_CALL amdf_rocjitsu_instance_add_machine(
    amdf_instance_t* instance,
    const amdf_rocjitsu_machine_create_info_t* create_info) {
  if (instance == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->configuration_json == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&instance->children) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  if (instance->gpu_api != NULL && instance->gpu_api != &amdf_rocjitsu_api) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (amdf_platform_instance_host_allocation_granularity(instance->platform) !=
      4096) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (instance->rocjitsu_source_count == UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  amdf_rocjitsu_source_t** sources = NULL;
  status = amdf_malloc(
      instance->host_allocator,
      ((size_t)instance->rocjitsu_source_count + 1) * sizeof(*sources),
      amdf_alignof(amdf_rocjitsu_source_t*), (void**)&sources);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_rocjitsu_source_t* source = NULL;
  status = amdf_rocjitsu_source_create(
      instance, instance->rocjitsu_source_count, create_info, &source);
  if (amdf_status_is_ok(status)) {
    uint64_t total = source->gpu_count;
    for (uint32_t i = 0; i < instance->rocjitsu_source_count; ++i) {
      total += instance->rocjitsu_sources[i]->gpu_count;
    }
    if (total > UINT32_MAX) {
      status = amdf_rocjitsu_source_destroy(source);
      if (amdf_status_is_ok(status)) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    if (instance->rocjitsu_source_count != 0) {
      memcpy(sources, instance->rocjitsu_sources,
             instance->rocjitsu_source_count * sizeof(*sources));
    }
    sources[instance->rocjitsu_source_count++] = source;
    amdf_free(instance->host_allocator, instance->rocjitsu_sources);
    instance->rocjitsu_sources = sources;
  } else {
    amdf_free(instance->host_allocator, sources);
  }
  return status;
}

amdf_status_t amdf_rocjitsu_extension_query(uint32_t minimum_version,
                                            uint32_t maximum_version,
                                            const void** out_api) {
  static const amdf_rocjitsu_api_t api = {
      .structure_size = sizeof(api),
      .extension_version = AMDF_ROCJITSU_EXTENSION_VERSION_1,
      .instance_add_machine = amdf_rocjitsu_instance_add_machine,
  };
  if (minimum_version > api.extension_version ||
      maximum_version < api.extension_version) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_api = &api;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_sources_destroy(amdf_instance_t* instance) {
  amdf_status_t status = AMDF_STATUS_OK;
  while (amdf_status_is_ok(status) && instance->rocjitsu_source_count != 0) {
    status = amdf_rocjitsu_source_destroy(
        instance->rocjitsu_sources[instance->rocjitsu_source_count - 1]);
    if (amdf_status_is_ok(status)) {
      --instance->rocjitsu_source_count;
    }
  }
  if (amdf_status_is_ok(status)) {
    amdf_free(instance->host_allocator, instance->rocjitsu_sources);
    instance->rocjitsu_sources = NULL;
  }
  return status;
}

uint32_t amdf_rocjitsu_endpoint_enumerate(amdf_instance_t* instance,
                                          uint32_t capacity,
                                          amdf_endpoint_summary_t* summaries) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < instance->rocjitsu_source_count; ++i) {
    const amdf_rocjitsu_source_t* source = instance->rocjitsu_sources[i];
    for (uint32_t j = 0; j < source->gpu_count; ++j, ++count) {
      if (count >= capacity) {
        continue;
      }
      const amdf_endpoint_info_t* info = &source->gpus[j].info;
      summaries[count] =
          (amdf_endpoint_summary_t){.id = info->id,
                                    .engine_kind = info->engine_kind,
                                    .type_flags = info->type_flags};
      memcpy(summaries[count].name, info->name, sizeof(info->name));
    }
  }
  return count;
}

amdf_status_t amdf_rocjitsu_endpoint_open(amdf_instance_t* instance,
                                          const amdf_endpoint_id_t* id,
                                          amdf_endpoint_t** out_endpoint) {
  const uint32_t source_ordinal = (uint32_t)(id->words[0] >> 32);
  const uint32_t gpu_ordinal = (uint32_t)id->words[0];
  if (source_ordinal >= instance->rocjitsu_source_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  amdf_rocjitsu_source_t* source = instance->rocjitsu_sources[source_ordinal];
  if (source->generation != id->words[1] || gpu_ordinal >= source->gpu_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  amdf_rocjitsu_gpu_t* gpu = &source->gpus[gpu_ordinal];
  return amdf_endpoint_create(instance, &gpu->info, &amdf_rocjitsu_api, gpu,
                              out_endpoint);
}

static void* amdf_rocjitsu_source_run(void* argument) {
  amdf_rocjitsu_source_t* source = argument;
  amdf_status_t status =
      amdf_rocjitsu_status(source->library.rj_vm_run(source->machine, NULL));
  if (amdf_status_is_ok(status)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  }
  amdf_atomic_uint64_store_release(&source->terminal_status, status);
  return NULL;
}

static amdf_status_t amdf_rocjitsu_source_prepare_apertures(
    amdf_rocjitsu_source_t* source) {
  struct kfd_ioctl_get_process_apertures_new_args* query = NULL;
  const size_t command_length =
      sizeof(*query) +
      (size_t)source->gpu_count * sizeof(struct kfd_process_device_apertures);
  amdf_status_t status = amdf_calloc(
      source->host_allocator, command_length,
      amdf_alignof(struct kfd_process_device_apertures), (void**)&query);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  // The explicit command API reconstructs pointer fields into inline payload.
  struct kfd_process_device_apertures* apertures = (void*)(query + 1);
  query->num_of_nodes = source->gpu_count;
  status = amdf_rocjitsu_source_command(
      source, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, query, command_length);
  if (amdf_status_is_ok(status) && query->num_of_nodes != source->gpu_count) {
    status = amdf_linux_error(EPROTO);
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < source->gpu_count;
       ++i) {
    const struct kfd_process_device_apertures* aperture = NULL;
    for (uint32_t j = 0; j < query->num_of_nodes; ++j) {
      if (apertures[j].gpu_id == source->gpus[i].native.gpu_id) {
        aperture = &apertures[j];
        break;
      }
    }
    if (aperture == NULL || (uint32_t)aperture->lds_base != 0 ||
        (uint32_t)aperture->scratch_base != 0 ||
        aperture->lds_base > aperture->lds_limit ||
        aperture->scratch_base > aperture->scratch_limit) {
      status = amdf_linux_error(EPROTO);
    } else {
      source->gpus[i].aql.apertures.group_base_hi =
          (uint32_t)(aperture->lds_base >> 32);
      source->gpus[i].aql.apertures.private_base_hi =
          (uint32_t)(aperture->scratch_base >> 32);
    }
  }
  amdf_free(source->host_allocator, query);
  return status;
}

amdf_status_t amdf_rocjitsu_source_prepare(amdf_rocjitsu_source_t* source) {
  amdf_status_t status = amdf_rocjitsu_source_status(source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (source->thread_started) {
    return AMDF_STATUS_OK;
  }
  if (source->machine == NULL) {
    // rocjitsu's local driver otherwise changes this process-wide soft limit.
    // The host owns its descriptor budget.
    struct rlimit descriptors;
    if (getrlimit(RLIMIT_NOFILE, &descriptors) != 0) {
      return amdf_linux_error(errno);
    }
    if (descriptors.rlim_cur < 8192) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    status = amdf_rocjitsu_status(source->library.rj_vm_create_from_config(
        source->configuration, RJ_VM_MODE_LOCAL, &source->machine));
  }
  if (amdf_status_is_ok(status) && source->process == 0) {
    status = amdf_rocjitsu_status(source->library.rj_vm_device_open(
        source->machine, 0, &source->process));
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_source_prepare_apertures(source);
  }
  if (amdf_status_is_ok(status)) {
    const int error =
        pthread_create(&source->thread, NULL, amdf_rocjitsu_source_run, source);
    if (error != 0) {
      status = amdf_linux_error(error);
    } else {
      source->thread_started = true;
    }
  }
  return status;
}

amdf_status_t amdf_rocjitsu_source_command(amdf_rocjitsu_source_t* source,
                                           uint32_t opcode, void* arguments,
                                           size_t byte_length) {
  rj_vm_cmd_t command = {
      .cmd = opcode,
      .buf = arguments,
      .buf_size = byte_length,
      .shared_handle = -1,
      .in_handle = -1,
      .in_mem_handle = -1,
      .in_proc_handle = -1,
  };
  const amdf_status_t status =
      amdf_rocjitsu_status(source->library.rj_vm_execute_as(
          source->machine, source->process, &command));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return command.result == 0 ? AMDF_STATUS_OK
                             : amdf_linux_error(-command.result);
}

amdf_status_t amdf_rocjitsu_source_map(amdf_rocjitsu_source_t* source,
                                       rj_vm_map_t* mapping) {
  const amdf_status_t status =
      amdf_rocjitsu_status(source->library.rj_vm_device_map_as(
          source->machine, source->process, mapping));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return mapping->map_errno == 0 ? AMDF_STATUS_OK
                                 : amdf_linux_error(mapping->map_errno);
}
