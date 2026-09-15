// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_KMT_API_H_
#define AMDF_SRC_PLATFORM_WINDOWS_KMT_API_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dynamically resolved subset of the public Windows KMT API.
typedef struct amdf_kmt_api_t {
  // System GDI module owning every procedure in this table.
  HMODULE module;
  // Optional system syscall module owning the preferred queue submission stub.
  HMODULE win32u_module;
  // Enumerates application-visible display and compute-only adapters.
  PFND3DKMT_ENUMADAPTERS3 enumerate_adapters;
  // Opens one adapter directly from its locally stable LUID.
  PFND3DKMT_OPENADAPTERFROMLUID open_adapter_from_luid;
  // Queries immutable adapter and physical-device properties.
  PFND3DKMT_QUERYADAPTERINFO query_adapter_info;
  // Creates one logical KMT device from an opened adapter.
  PFND3DKMT_CREATEDEVICE create_device;
  // Destroys one logical KMT device.
  PFND3DKMT_DESTROYDEVICE destroy_device;
  // Diagnoses device execution failure after a rejected native operation.
  PFND3DKMT_GETDEVICESTATE get_device_state;
  // Creates the paging queue used by a logical device.
  PFND3DKMT_CREATEPAGINGQUEUE create_paging_queue;
  // Destroys one paging queue and its associated synchronization object.
  PFND3DKMT_DESTROYPAGINGQUEUE destroy_paging_queue;
  // Creates one virtual execution context.
  PFND3DKMT_CREATECONTEXTVIRTUAL create_context_virtual;
  // Destroys one virtual execution context.
  PFND3DKMT_DESTROYCONTEXT destroy_context;
  // Creates one or more physical allocations.
  PFND3DKMT_CREATEALLOCATION2 create_allocation;
  // Destroys one physical allocation or resource.
  PFND3DKMT_DESTROYALLOCATION2 destroy_allocation;
  // Reserves one device virtual-address range without physical backing.
  PFND3DKMT_RESERVEGPUVIRTUALADDRESS reserve_gpu_virtual_address;
  // Releases one unmapped device virtual-address reservation.
  PFND3DKMT_FREEGPUVIRTUALADDRESS free_gpu_virtual_address;
  // Establishes one allocation mapping in a device address space.
  PFND3DKMT_MAPGPUVIRTUALADDRESS map_gpu_virtual_address;
  // Establishes residency without forcing a device error on exhaustion.
  PFND3DKMT_MAKERESIDENT make_resident;
  // Releases residency for one or more physical allocations.
  PFND3DKMT_EVICT evict;
  // Locks one allocation for explicit host access.
  PFND3DKMT_LOCK2 lock;
  // Releases one explicit allocation lock.
  PFND3DKMT_UNLOCK2 unlock;
  // Publishes host cache changes to one native allocation range.
  PFND3DKMT_INVALIDATECACHE invalidate_cache;
  // Waits for a monitored paging fence from the host.
  PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait_from_cpu;
  // Creates one hardware queue from a virtual context.
  PFND3DKMT_CREATEHWQUEUE create_hardware_queue;
  // Destroys one hardware queue.
  PFND3DKMT_DESTROYHWQUEUE destroy_hardware_queue;
  // Publishes one command to a hardware queue.
  PFND3DKMT_SUBMITCOMMANDTOHWQUEUE submit_command_to_hardware_queue;
  // Closes an adapter handle.
  PFND3DKMT_CLOSEADAPTER close_adapter;
} amdf_kmt_api_t;

// Loads the primary system KMT module and resolves available procedures.
// Failure leaves `out_api` unchanged. Procedure availability is reported by
// operation-specific capability queries on the successfully initialized table.
amdf_status_t amdf_kmt_api_initialize(amdf_kmt_api_t* out_api);

// Unloads system KMT modules after every child handle has been closed.
amdf_status_t amdf_kmt_api_deinitialize(amdf_kmt_api_t* api);

// Returns true when the complete endpoint discovery procedure set exists.
bool amdf_kmt_api_supports_endpoint_discovery(const amdf_kmt_api_t* api);

// Returns true when the complete logical-device and paging procedure set
// exists.
bool amdf_kmt_api_supports_paging_devices(const amdf_kmt_api_t* api);

// Returns true when paging devices and virtual contexts are supported.
bool amdf_kmt_api_supports_device_contexts(const amdf_kmt_api_t* api);

// Returns true when the complete physical-memory procedure set exists.
bool amdf_kmt_api_supports_memory(const amdf_kmt_api_t* api);

// Returns true when the complete GPU allocation and virtual-memory procedure
// set exists.
bool amdf_kmt_api_supports_gpu_memory(const amdf_kmt_api_t* api);

// Returns true when the complete GPU kernel-submission procedure set exists.
bool amdf_kmt_api_supports_gpu_kernel_execution(const amdf_kmt_api_t* api);

// Returns true when the complete XDNA kernel-submission procedure set exists.
bool amdf_kmt_api_supports_xdna_kernel_execution(const amdf_kmt_api_t* api);

// Returns true for immediate success and queued asynchronous acceptance.
bool amdf_kmt_status_is_success_or_pending(NTSTATUS status);

// Waits for one paging point unless its mapped fence has already retired.
amdf_status_t amdf_kmt_wait_for_paging(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE device,
    D3DKMT_HANDLE paging_sync_object,
    const volatile uint64_t* current_paging_fence, uint64_t target_value);

// Queries one typed block of adapter information.
amdf_status_t amdf_kmt_query_adapter_info(const amdf_kmt_api_t* api,
                                          D3DKMT_HANDLE adapter,
                                          KMTQUERYADAPTERINFOTYPE type,
                                          void* data, uint32_t data_size);

// Converts a native KMT status without discarding its domain.
amdf_status_t amdf_kmt_make_status(NTSTATUS status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_KMT_API_H_
