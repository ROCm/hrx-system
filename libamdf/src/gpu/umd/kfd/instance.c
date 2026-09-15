// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/instance.h"

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
} amdf_gpu_kfd_vm_binding_t;

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
};

amdf_status_t amdf_gpu_umd_instance_prepare(
    amdf_gpu_umd_instance_t** state, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator) {
  amdf_gpu_umd_instance_t* instance = *state;
  if (instance == NULL) {
    const amdf_status_t status =
        amdf_calloc(host_allocator, sizeof(*instance),
                    amdf_alignof(amdf_gpu_umd_instance_t), (void**)&instance);
    if (!amdf_status_is_ok(status)) return status;
    instance->host_allocator = host_allocator;
    instance->native_lifetime = native_lifetime;
    instance->descriptor = -1;
    *state = instance;
  }
  if (instance->prepared) return AMDF_STATUS_OK;
  if (instance->descriptor < 0) {
    const amdf_status_t status =
        amdf_gpu_kfd_file_open(&instance->descriptor, &instance->version);
    if (!amdf_status_is_ok(status)) return status;
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
    if (!amdf_status_is_ok(status)) return status;
    binding->gpu_id = topology->gpu_id;
    binding->descriptor = -1;
    binding->next = instance->bindings;
    instance->bindings = binding;
  }
  // A previous preparation may have acquired the VM before bootstrap cleanup
  // failed. Finish cleanup before reuse; a converted VM is never bootstrapped
  // a second time through the DRM handover path.
  amdf_status_t status = amdf_gpu_kfd_vm_bootstrap_release(&binding->bootstrap);
  if (!amdf_status_is_ok(status)) return status;
  if (binding->descriptor < 0) {
    status =
        amdf_linux_endpoint_open_file(endpoint, &binding->descriptor, NULL);
    if (!amdf_status_is_ok(status)) return status;
  }
  // Bootstrap and later memory mappings consume this connection's installed
  // aperture, including native reservations and administrator-selected limits.
  amdf_gpu_kfd_topology_t native_topology = *topology;
  status = amdf_gpu_kfd_topology_refine_memory(
      binding->descriptor, endpoint->info.pci.device_id, &native_topology);
  if (!amdf_status_is_ok(status)) return status;
  if (page_size != native_topology.virtual_address.alignment) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (!binding->acquired) {
    status = amdf_gpu_kfd_vm_acquire(
        instance->descriptor, binding->descriptor, &native_topology, page_size,
        amdf_gpu_kfd_vm_default_native_api(), &binding->bootstrap);
    if (!amdf_status_is_ok(status)) return status;
    binding->acquired = true;
    status = amdf_gpu_kfd_vm_bootstrap_release(&binding->bootstrap);
    if (!amdf_status_is_ok(status)) return status;
  }
  *topology = native_topology;
  *out_render_descriptor = binding->descriptor;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_instance_destroy(amdf_gpu_umd_instance_t* instance) {
  amdf_status_t status = AMDF_STATUS_OK;
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
    amdf_free(instance->host_allocator, instance);
  }
  return status;
}
