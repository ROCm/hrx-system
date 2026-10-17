// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/instance.h"

#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/file.h"
#include "libamdf/src/gpu/umd/kfd/vm.h"
#include "libamdf/src/platform/linux/endpoint.h"

// One native KFD per-device VM binding, including incomplete preparation.
typedef struct amdf_gpu_kfd_vm_binding_t {
  // Next binding owned by the same KFD context.
  struct amdf_gpu_kfd_vm_binding_t* next;
  // Kernel GPU identity used by ACQUIRE_VM and memory mapping operations.
  uint32_t gpu_id;
  // Exact render file whose VM KFD acquired, or -1 before opening succeeds.
  int descriptor;
  // Whether KFD has accepted this VM; bootstrap release may still be pending.
  bool acquired;
  // Native VM handover progress retained through successful release.
  amdf_gpu_kfd_vm_bootstrap_t bootstrap;
  // Shared signal-page access in this exact VM, including native map progress.
  struct {
    // Native map prefix; one means the backing has been attached to this VM.
    uint32_t mapped_count;
    // Native unmap prefix retained across final-synchronization failures.
    uint32_t unmapped_count;
    // Whether the mapping's final native synchronization completed.
    bool complete;
  } event_mapping;
} amdf_gpu_kfd_vm_binding_t;

// CREATE_EVENT can install its backing before the event allocation fails.
typedef enum amdf_gpu_kfd_signal_page_installation_e {
  AMDF_GPU_KFD_SIGNAL_PAGE_UNINSTALLED = 0,
  AMDF_GPU_KFD_SIGNAL_PAGE_UNCERTAIN,
  AMDF_GPU_KFD_SIGNAL_PAGE_INSTALLED,
} amdf_gpu_kfd_signal_page_installation_t;

struct amdf_gpu_umd_instance_t {
  // Allocator for this connection and its concrete VM bindings.
  amdf_allocator_t host_allocator;
  // Native lifetime selected by the public instance.
  amdf_native_lifetime_t native_lifetime;
  // Owned KFD context descriptor, or -1 before its open succeeds.
  int descriptor;
  // ABI of the owned descriptor, established together with its open.
  struct kfd_ioctl_get_version_args version;
  // Whether version qualification and requested context selection completed.
  bool prepared;
  // Acquired or partially prepared per-GPU VM bindings.
  amdf_gpu_kfd_vm_binding_t* bindings;
  // Serializes foreground VM mutations through their final native TLB flush.
  // One connection-wide lock covers peer operations without lock ordering.
  amdf_gpu_kfd_memory_lock_t memory_lock;
  // Native event bootstrap, serialized by memory_lock while devices are live.
  struct {
    // Owned VA reservation, retained until every GPU mapping has been removed.
    void* reservation;
    // Page-covered extent of the KFD signal table in bytes.
    size_t byte_length;
    // Native backing handle; installed backing is reclaimed by its context.
    uint64_t handle;
    // Whether native construction has conclusively installed this backing.
    amdf_gpu_kfd_signal_page_installation_t installation;
    // Context-wide manual-reset memory exception source, or zero when absent.
    uint32_t memory_exception_id;
    // Context-wide manual-reset hardware exception source, or zero when absent.
    uint32_t hardware_exception_id;
  } events;
};

static int amdf_gpu_kfd_instance_event_ioctl(amdf_gpu_umd_instance_t* instance,
                                             unsigned long request,
                                             void* args) {
  int result;
  do {
    result = ioctl(instance->descriptor, request, args);
  } while (result == -1 && errno == EINTR);
  return result;
}

// Fault events are manual-reset because every device waiter must observe a
// terminal context failure. No thread or device registry routes the wake.
static amdf_status_t amdf_gpu_kfd_instance_prepare_fault_event(
    amdf_gpu_umd_instance_t* instance, uint32_t type, uint32_t* event_id) {
  if (*event_id != 0) {
    return AMDF_STATUS_OK;
  }
  struct kfd_ioctl_create_event_args create = {.event_type = type};
  if (amdf_gpu_kfd_instance_event_ioctl(instance, AMDKFD_IOC_CREATE_EVENT,
                                        &create) != 0) {
    return amdf_linux_error(errno);
  }
  *event_id = create.event_id;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_instance_prepare_signal_page(
    amdf_gpu_umd_instance_t* instance, const amdf_gpu_kfd_topology_t* topology,
    size_t page_size, amdf_gpu_kfd_vm_binding_t* binding) {
  if (instance->events.reservation == NULL) {
    const size_t table_size = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t);
    instance->events.byte_length =
        (table_size + page_size - 1) & ~(page_size - 1);
    void* reservation = mmap(NULL, instance->events.byte_length, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reservation == MAP_FAILED) {
      return amdf_linux_error(errno);
    }
    instance->events.reservation = reservation;
  }
  const uint64_t address = (uintptr_t)instance->events.reservation;
  if (address < topology->virtual_address.begin ||
      address >= topology->virtual_address.end ||
      instance->events.byte_length > topology->virtual_address.end - address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (instance->events.handle == 0) {
    // The native WRITABLE macro uses a signed shift; bit 31 is unsigned here.
    struct kfd_ioctl_alloc_memory_of_gpu_args allocate = {
        .va_addr = address,
        .size = instance->events.byte_length,
        .gpu_id = topology->gpu_id,
        .flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                 KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | (UINT32_C(1) << 31),
    };
    if (amdf_gpu_kfd_instance_event_ioctl(
            instance, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocate) != 0) {
      return amdf_linux_error(errno);
    }
    instance->events.handle = allocate.handle;
  }
  if (!binding->event_mapping.complete) {
    uint32_t gpu_id = topology->gpu_id;
    struct kfd_ioctl_map_memory_to_gpu_args map = {
        .handle = instance->events.handle,
        .device_ids_array_ptr = (uintptr_t)&gpu_id,
        .n_devices = 1,
        .n_success = binding->event_mapping.mapped_count,
    };
    const int result = amdf_gpu_kfd_instance_event_ioctl(
        instance, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
    binding->event_mapping.mapped_count = map.n_success;
    if (result != 0) {
      return amdf_linux_error(errno);
    }
    if (map.n_success != 1) {
      return amdf_linux_error(EPROTO);
    }
    binding->event_mapping.complete = true;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_instance_create_event(
    amdf_gpu_umd_instance_t* instance, const amdf_gpu_kfd_topology_t* topology,
    size_t page_size, amdf_gpu_kfd_event_info_t* out_info) {
  amdf_gpu_kfd_vm_binding_t* binding = instance->bindings;
  while (binding->gpu_id != topology->gpu_id) {
    binding = binding->next;
  }
  amdf_status_t status = amdf_gpu_kfd_instance_prepare_signal_page(
      instance, topology, page_size, binding);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_gpu_kfd_instance_prepare_fault_event(
      instance, KFD_IOC_EVENT_MEMORY, &instance->events.memory_exception_id);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_gpu_kfd_instance_prepare_fault_event(
      instance, KFD_IOC_EVENT_HW_EXCEPTION,
      &instance->events.hardware_exception_id);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  struct kfd_ioctl_create_event_args create = {
      .event_page_offset =
          instance->events.installation == AMDF_GPU_KFD_SIGNAL_PAGE_INSTALLED
              ? 0
              : instance->events.handle,
      .event_type = KFD_IOC_EVENT_SIGNAL,
      .auto_reset = 1,
  };
  int result = amdf_gpu_kfd_instance_event_ioctl(
      instance, AMDKFD_IOC_CREATE_EVENT, &create);
  if (result != 0 && errno == EINVAL &&
      instance->events.installation == AMDF_GPU_KFD_SIGNAL_PAGE_UNCERTAIN) {
    // A previous allocation failure may follow page installation. Reinstalling
    // an existing page returns EINVAL; a successful SIGNAL creation without a
    // page then proves installation. This transition is allowed only after our
    // own ambiguous installation, never after encountering a foreign page.
    create.event_page_offset = 0;
    result = amdf_gpu_kfd_instance_event_ioctl(
        instance, AMDKFD_IOC_CREATE_EVENT, &create);
  }
  if (result != 0) {
    // These allocation errors can occur after successful page installation.
    // In contrast, EINVAL from an existing foreign page does not authorize
    // later use of that page with this connection's different GPU backing.
    if ((errno == ENOMEM || errno == ENOSPC) &&
        instance->events.installation != AMDF_GPU_KFD_SIGNAL_PAGE_INSTALLED) {
      instance->events.installation = AMDF_GPU_KFD_SIGNAL_PAGE_UNCERTAIN;
    }
    return amdf_linux_error(errno);
  }
  instance->events.installation = AMDF_GPU_KFD_SIGNAL_PAGE_INSTALLED;
  *out_info = (amdf_gpu_kfd_event_info_t){
      .mailbox_address = (uintptr_t)instance->events.reservation +
                         sizeof(uint64_t) * create.event_slot_index,
      .signal_id = create.event_id,
      .trigger_data = create.event_trigger_data,
      .memory_exception_id = instance->events.memory_exception_id,
      .hardware_exception_id = instance->events.hardware_exception_id,
  };
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_instance_release_event(
    amdf_gpu_umd_instance_t* instance, uint32_t* event_id) {
  if (*event_id == 0) {
    return AMDF_STATUS_OK;
  }
  struct kfd_ioctl_destroy_event_args destroy = {.event_id = *event_id};
  if (amdf_gpu_kfd_instance_event_ioctl(instance, AMDKFD_IOC_DESTROY_EVENT,
                                        &destroy) != 0) {
    return amdf_linux_error(errno);
  }
  *event_id = 0;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_instance_release_events(
    amdf_gpu_umd_instance_t* instance) {
  amdf_status_t status = amdf_gpu_kfd_instance_release_event(
      instance, &instance->events.hardware_exception_id);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_gpu_kfd_instance_release_event(
      instance, &instance->events.memory_exception_id);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (amdf_gpu_kfd_vm_binding_t* binding = instance->bindings;
       binding != NULL && amdf_status_is_ok(status); binding = binding->next) {
    if (binding->event_mapping.mapped_count == 0) {
      continue;
    }
    struct kfd_ioctl_unmap_memory_from_gpu_args unmap = {
        .handle = instance->events.handle,
        .device_ids_array_ptr = (uintptr_t)&binding->gpu_id,
        .n_devices = 1,
        .n_success = binding->event_mapping.unmapped_count,
    };
    const int result = amdf_gpu_kfd_instance_event_ioctl(
        instance, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);
    binding->event_mapping.unmapped_count = unmap.n_success;
    if (result != 0) {
      status = amdf_linux_error(errno);
    } else if (unmap.n_success != 1) {
      status = amdf_linux_error(EPROTO);
    } else {
      binding->event_mapping.mapped_count = 0;
      binding->event_mapping.unmapped_count = 0;
      binding->event_mapping.complete = false;
    }
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (instance->events.handle != 0 &&
      instance->events.installation == AMDF_GPU_KFD_SIGNAL_PAGE_UNINSTALLED) {
    struct kfd_ioctl_free_memory_of_gpu_args release = {
        .handle = instance->events.handle,
    };
    if (amdf_gpu_kfd_instance_event_ioctl(
            instance, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release) != 0) {
      return amdf_linux_error(errno);
    }
    instance->events.handle = 0;
  }
  // Installed (or ambiguously installed) backing belongs to the native
  // context. Its GPU mappings are gone, so the VA reservation can end now.
  // INSTANCE closes that context below; PROCESS permits native exit cleanup.
  if (instance->events.reservation != NULL) {
    if (munmap(instance->events.reservation, instance->events.byte_length) !=
        0) {
      return amdf_linux_error(errno);
    }
    instance->events.reservation = NULL;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_instance_prepare(
    amdf_gpu_umd_instance_t** state, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator) {
  amdf_gpu_umd_instance_t* instance = *state;
  if (instance == NULL) {
    const amdf_status_t status =
        amdf_calloc(host_allocator, sizeof(*instance),
                    amdf_alignof(amdf_gpu_umd_instance_t), (void**)&instance);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const int mutex_error =
        amdf_gpu_kfd_memory_lock_initialize(&instance->memory_lock);
    if (mutex_error != 0) {
      amdf_free(host_allocator, instance);
      return amdf_linux_error(mutex_error);
    }
    instance->host_allocator = host_allocator;
    instance->native_lifetime = native_lifetime;
    instance->descriptor = -1;
    *state = instance;
  }
  if (instance->prepared) {
    return AMDF_STATUS_OK;
  }
  if (instance->descriptor < 0) {
    const amdf_status_t status =
        amdf_gpu_kfd_file_open(&instance->descriptor, &instance->version);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (instance->version.major_version != 1 ||
      instance->version.minor_version < 18 ||
      (instance->native_lifetime == AMDF_NATIVE_LIFETIME_INSTANCE &&
       instance->version.minor_version < 19)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (instance->native_lifetime == AMDF_NATIVE_LIFETIME_INSTANCE &&
      ioctl(instance->descriptor, AMDKFD_IOC_CREATE_PROCESS, NULL) != 0) {
    return amdf_linux_error(errno);
  }
  instance->prepared = true;
  return AMDF_STATUS_OK;
}

int amdf_gpu_kfd_instance_descriptor(const amdf_gpu_umd_instance_t* instance) {
  return instance->descriptor;
}

amdf_gpu_kfd_memory_lock_t* amdf_gpu_kfd_instance_memory_lock(
    amdf_gpu_umd_instance_t* instance) {
  return &instance->memory_lock;
}

amdf_status_t amdf_gpu_kfd_instance_prepare_vm(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology, size_t page_size,
    int* out_render_descriptor) {
  amdf_gpu_kfd_vm_binding_t* binding = instance->bindings;
  while (binding != NULL && binding->gpu_id != topology->gpu_id) {
    binding = binding->next;
  }
  if (binding == NULL) {
    const amdf_status_t status =
        amdf_calloc(instance->host_allocator, sizeof(*binding),
                    amdf_alignof(amdf_gpu_kfd_vm_binding_t), (void**)&binding);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    binding->gpu_id = topology->gpu_id;
    binding->descriptor = -1;
    binding->next = instance->bindings;
    instance->bindings = binding;
  }
  // A previous preparation may have acquired the VM before bootstrap cleanup
  // failed. Finish cleanup before reuse; a converted VM is never bootstrapped
  // a second time through the DRM handover path.
  amdf_status_t status = amdf_gpu_kfd_vm_bootstrap_release(&binding->bootstrap);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (binding->descriptor < 0) {
    status = amdf_linux_endpoint_open_file(endpoint, &binding->descriptor);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  // Bootstrap and later memory mappings consume this connection's installed
  // aperture, including native reservations and administrator-selected limits.
  amdf_gpu_kfd_topology_t native_topology = *topology;
  status = amdf_gpu_kfd_topology_query_memory(
      binding->descriptor, endpoint->info.pci.device_id, &native_topology);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (page_size != native_topology.virtual_address.alignment) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (!binding->acquired) {
    status = amdf_gpu_kfd_vm_acquire(
        instance->descriptor, binding->descriptor, &native_topology, page_size,
        amdf_gpu_kfd_vm_default_native_api(), &binding->bootstrap);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    binding->acquired = true;
    status = amdf_gpu_kfd_vm_bootstrap_release(&binding->bootstrap);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  *topology = native_topology;
  *out_render_descriptor = binding->descriptor;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_instance_destroy(amdf_gpu_umd_instance_t* instance) {
  amdf_status_t status = amdf_gpu_kfd_instance_release_events(instance);
  for (amdf_gpu_kfd_vm_binding_t* binding = instance->bindings;
       binding != NULL && amdf_status_is_ok(status); binding = binding->next) {
    status = amdf_gpu_kfd_vm_bootstrap_release(&binding->bootstrap);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_file_close(&instance->descriptor);
  }
  while (instance->bindings != NULL && amdf_status_is_ok(status)) {
    amdf_gpu_kfd_vm_binding_t* binding = instance->bindings;
    status = amdf_linux_file_close(&binding->descriptor);
    if (amdf_status_is_ok(status)) {
      instance->bindings = binding->next;
      amdf_free(instance->host_allocator, binding);
    }
  }
  if (amdf_status_is_ok(status)) {
    amdf_gpu_kfd_memory_lock_deinitialize(&instance->memory_lock);
    amdf_free(instance->host_allocator, instance);
  }
  return status;
}
