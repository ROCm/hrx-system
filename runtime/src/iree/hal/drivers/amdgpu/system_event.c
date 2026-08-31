// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// dl_iterate_phdr, dladdr and the RTLD_NOLOAD/RTLD_NODELETE flags used to pin
// the callback's module are GNU extensions, so this must precede any system
// header.
#define _GNU_SOURCE

#include "iree/hal/drivers/amdgpu/system_event.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/api.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"

#if defined(IREE_PLATFORM_LINUX)
#include <dlfcn.h>
#include <link.h>
#endif  // IREE_PLATFORM_LINUX

struct iree_hal_amdgpu_system_event_agent_target_t {
  // HSA agent handle copied at registration. Immutable for the registration.
  hsa_agent_t agent;
  // Queue storage for |agent|, captured when the queues are published. Interior
  // pointer into the physical device's inline queue array, which lives inside
  // the logical device allocation and outlives this target.
  iree_hal_amdgpu_host_queue_t* host_queues;
  // Number of entries addressable by |live_queue_mask| in |host_queues|.
  // Written only under the registry mutex.
  iree_host_size_t queue_capacity;
  // Bit i selects |host_queues[i]| for failure delivery. Zero before
  // publication and after retirement. Written only under the registry mutex,
  // by frontier assignment, sparse queue materialization, and deassignment.
  uint64_t live_queue_mask;
};

struct iree_hal_amdgpu_system_event_registration_t {
  // Next live registration in the process-wide registry list.
  struct iree_hal_amdgpu_system_event_registration_t* next;
  // Logical device receiving the sticky device failure status for these agents,
  // or NULL once that status has been retired as a delivery target. Borrowed;
  // retirement happens while the device is still allocated.
  iree_hal_amdgpu_logical_device_t* logical_device;
  // Allocator owning this registration.
  iree_allocator_t host_allocator;
  // Number of entries in |agent_targets|.
  iree_host_size_t agent_count;
  // One delivery target per GPU agent of |logical_device|.
  iree_hal_amdgpu_system_event_agent_target_t agent_targets[/*agent_count*/];
};

typedef struct iree_hal_amdgpu_system_event_registry_t {
  // Serializes registration list mutation, target publication/retirement, and
  // callback traversal. Never held across an HSA entry point an event can be
  // dispatched through, so a callback waiting for it can never be waiting on a
  // thread the HSA runtime has to call back into; delivery does hold it across
  // the stop-signal store that records a queue failure, which dispatches
  // nothing. Deliberately process-lifetime: it guards a file-static list that
  // outlives every device, so it is initialized once and never deinitialized.
  iree_mutex_t mutex;
  // Serializes the one-time handler registration with the HSA runtime and is
  // held across that call. The callback never acquires it, which is what keeps
  // the rule above true without giving registration an exemption from it.
  iree_mutex_t hsa_handler_mutex;
  // Head of the live registration list.
  iree_hal_amdgpu_system_event_registration_t* registration_list;
  // Whether the process-wide callback is registered with the HSA runtime.
  // Written under |hsa_handler_mutex| by registration and cleared by the
  // shutdown-event branch of the callback, which takes no lock at all: that
  // branch runs on the thread inside the final hsa_shut_down, which is the one
  // thread that must never block on a lock an HSA caller may hold.
  iree_atomic_int32_t is_hsa_handler_registered;
} iree_hal_amdgpu_system_event_registry_t;

static iree_once_flag iree_hal_amdgpu_system_event_once = IREE_ONCE_FLAG_INIT;
static iree_hal_amdgpu_system_event_registry_t
    iree_hal_amdgpu_system_event_registry;

static void iree_hal_amdgpu_system_event_initialize(void) {
  iree_mutex_initialize(&iree_hal_amdgpu_system_event_registry.mutex);
  iree_mutex_initialize(
      &iree_hal_amdgpu_system_event_registry.hsa_handler_mutex);
  iree_atomic_store(
      &iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered, 0,
      iree_memory_order_relaxed);
}

// Returns the GPU agent an event blames, for the event types this driver
// converts into device failures.
//
// HSA_AMD_GPU_MEMORY_ERROR_EVENT is deliberately not one of them. It is not a
// GPU fault: the runtime raises it synchronously on the thread inside a memory
// pool free, its agent is the owner of the pool being freed rather than a
// faulting device, and claiming it makes that free return an error with the
// allocation record already purged - converting the runtime's abort into a
// permanent leak plus an allocator whose bookkeeping no longer matches reality.
// Leaving it unclaimed keeps the abort, which is the defensible outcome for an
// inconsistent allocator.
static bool iree_hal_amdgpu_system_event_agent(const hsa_amd_event_t* event,
                                               hsa_agent_t* out_agent) {
  *out_agent = (hsa_agent_t){0};
  switch (event->event_type) {
    case HSA_AMD_GPU_MEMORY_FAULT_EVENT:
      *out_agent = event->memory_fault.agent;
      return true;
    case HSA_AMD_GPU_HW_EXCEPTION_EVENT:
      *out_agent = event->hw_exception.agent;
      return true;
    default:
      return false;
  }
}

static iree_status_t iree_hal_amdgpu_system_event_make_status(
    const hsa_amd_event_t* event) {
  switch (event->event_type) {
    case HSA_AMD_GPU_MEMORY_FAULT_EVENT:
      return iree_make_status(
          IREE_STATUS_ABORTED,
          "AMDGPU memory access fault at device address 0x%016" PRIx64
          " (reason mask 0x%08" PRIx32 ")",
          event->memory_fault.virtual_address,
          event->memory_fault.fault_reason_mask);
    case HSA_AMD_GPU_HW_EXCEPTION_EVENT:
      return iree_make_status(
          IREE_STATUS_ABORTED,
          "AMDGPU hardware exception (reset type 0x%08" PRIx32
          ", cause 0x%08" PRIx32 ")",
          (uint32_t)event->hw_exception.reset_type,
          (uint32_t)event->hw_exception.reset_cause);
    default:
      return iree_make_status(IREE_STATUS_ABORTED,
                              "unrecognized AMDGPU system event");
  }
}

// Returns true if |registration| holds a target for |event_agent|.
static bool iree_hal_amdgpu_system_event_registration_matches(
    const iree_hal_amdgpu_system_event_registration_t* registration,
    hsa_agent_t event_agent) {
  for (iree_host_size_t i = 0; i < registration->agent_count; ++i) {
    if (registration->agent_targets[i].agent.handle == event_agent.handle) {
      return true;
    }
  }
  return false;
}

// Delivers |event| to every registration holding a target for its agent.
// Returns true if at least one registration accepted delivery.
//
// The failing unit is the registration, not the agent. Queues of one logical
// device share an epoch signal table spanning every one of its GPU agents, so a
// queue on a healthy agent can be parked on a device-side barrier against the
// epoch signal of a queue on the faulting one, which will never advance again.
// Failing only the matched agent's queues would leave that queue live with an
// epoch nothing can retire, and its teardown wait would never return - while
// the sticky device status this same delivery latches already covers the whole
// logical device.
//
// Runs on an HSA runtime thread and can run while a thread of this driver is
// inside HSA teardown. The only lock it takes is the registry mutex, which
// every path in this file holds across a list walk or a store and never across
// a blocking call, so it cannot be parked behind a teardown that is itself
// waiting. The rest is a compare-exchange per failure slot, host status
// allocation and release, and the stop-signal store that recording a queue
// failure makes into a signal the delivery targets keep valid.
static bool iree_hal_amdgpu_system_event_deliver(const hsa_amd_event_t* event,
                                                 hsa_agent_t event_agent) {
  bool delivered = false;
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  for (iree_hal_amdgpu_system_event_registration_t* registration =
           iree_hal_amdgpu_system_event_registry.registration_list;
       registration != NULL; registration = registration->next) {
    if (!iree_hal_amdgpu_system_event_registration_matches(registration,
                                                           event_agent)) {
      continue;
    }
    for (iree_host_size_t i = 0; i < registration->agent_count; ++i) {
      iree_hal_amdgpu_system_event_agent_target_t* target =
          &registration->agent_targets[i];
      for (iree_host_size_t j = 0; j < target->queue_capacity; ++j) {
        if ((target->live_queue_mask & (UINT64_C(1) << j)) == 0) {
          continue;
        }
        iree_hal_amdgpu_host_queue_record_failure(
            &target->host_queues[j],
            iree_hal_amdgpu_system_event_make_status(event));
        delivered = true;
      }
    }
    // The device's sticky failure status is a delivery target in its own right,
    // outliving the queue targets: it is still readable through the HAL after a
    // frontier deassignment retires every queue, and stops being readable when
    // the device does. A registration that holds neither has nowhere to write,
    // and a write nothing can read is not a delivery and must not be claimed.
    if (registration->logical_device) {
      iree_hal_amdgpu_logical_device_error_handler(
          registration->logical_device,
          iree_hal_amdgpu_system_event_make_status(event));
      delivered = true;
    }
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
  return delivered;
}

static hsa_status_t iree_hal_amdgpu_system_event_callback(
    const hsa_amd_event_t* event, void* user_data) {
  (void)user_data;
  if (event->event_type == HSA_AMD_SYSTEM_SHUTDOWN_EVENT) {
    // The final hsa_shut_down destroys the runtime's callback registry. Reset
    // our process state so a later HSA lifetime registers this handler again.
    iree_atomic_store(
        &iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered, 0,
        iree_memory_order_release);
    return HSA_STATUS_SUCCESS;
  }
  hsa_agent_t event_agent;
  if (!iree_hal_amdgpu_system_event_agent(event, &event_agent)) {
    return HSA_STATUS_ERROR;
  }
  // Claiming is delivering: an event matched but written nowhere readable would
  // suppress the HSA runtime's abort while leaving nothing in the process able
  // to report the fault.
  return iree_hal_amdgpu_system_event_deliver(event, event_agent)
             ? HSA_STATUS_SUCCESS
             : HSA_STATUS_ERROR;
}

#if defined(IREE_PLATFORM_LINUX)

// State threaded through the dl_iterate_phdr walk that classifies an address.
typedef struct iree_hal_amdgpu_module_search_t {
  // Address being located. Set by the caller.
  const void* address;
  // Classification of the object containing |address|. Left unset when no
  // loaded object contains it.
  iree_hal_amdgpu_module_pin_t result;
  // Whether any loaded object contains |address|.
  bool found;
} iree_hal_amdgpu_module_search_t;

// dl_iterate_phdr callback locating the loaded object that contains an address.
static int iree_hal_amdgpu_module_search_visit(struct dl_phdr_info* info,
                                               size_t info_size,
                                               void* user_data) {
  (void)info_size;
  iree_hal_amdgpu_module_search_t* search =
      (iree_hal_amdgpu_module_search_t*)user_data;
  const uintptr_t address = (uintptr_t)search->address;
  for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)* phdr = &info->dlpi_phdr[i];
    if (phdr->p_type != PT_LOAD) continue;
    const uintptr_t segment_start = (uintptr_t)info->dlpi_addr + phdr->p_vaddr;
    if (address < segment_start || address >= segment_start + phdr->p_memsz) {
      continue;
    }
    search->found = true;
    // The loader names the main program with an empty string. Nothing in a
    // process can unload the main program, so an address in it needs no pin,
    // and that includes a link with no dynamic loader at all, where this is
    // the only object there is.
    search->result = (info->dlpi_name == NULL || info->dlpi_name[0] == '\0')
                         ? IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED
                         : IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED;
    return 1;
  }
  return 0;
}

#endif  // IREE_PLATFORM_LINUX

iree_status_t iree_hal_amdgpu_system_event_pin_module_containing(
    const void* address, iree_hal_amdgpu_module_pin_t* out_pin) {
  IREE_ASSERT_ARGUMENT(address);
  IREE_ASSERT_ARGUMENT(out_pin);
  *out_pin = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED;
#if defined(IREE_PLATFORM_WINDOWS)
  // Pinning the main program's own module is legal and is what happens when the
  // driver is linked into the executable, so there is no not-required case to
  // report here.
  HMODULE module = NULL;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_PIN,
                        (LPCSTR)address, &module) == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no loaded module contains address %p", address);
  }
  *out_pin = IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED;
  return iree_ok_status();
#elif defined(IREE_PLATFORM_LINUX)
  // dl_iterate_phdr reports every object the loader has mapped, including the
  // main program, and it works in a link with no dynamic loader where dladdr
  // reports nothing at all. Classifying from the walk is what keeps the three
  // outcomes apart: dlopen returns NULL both for the main program and for a
  // real failure, so inferring from it would report a pin that never happened.
  iree_hal_amdgpu_module_search_t search = {
      .address = address,
      .result = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED,
      .found = false,
  };
  dl_iterate_phdr(iree_hal_amdgpu_module_search_visit, &search);
  if (!search.found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no loaded module contains address %p", address);
  }
  if (search.result == IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED) {
    *out_pin = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED;
    return iree_ok_status();
  }
  // RTLD_NOLOAD resolves the already-resident object without loading anything
  // and RTLD_NODELETE makes its mapping permanent, so the handle taken here can
  // be released rather than leaked. The walk above proved the object is mapped
  // and named, so a NULL here is a real failure to pin and is reported as one.
  Dl_info dl_info;
  if (dladdr(address, &dl_info) == 0 || dl_info.dli_fname == NULL ||
      dl_info.dli_fname[0] == '\0') {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "loaded module containing address %p cannot be "
                            "named for pinning",
                            address);
  }
  void* handle =
      dlopen(dl_info.dli_fname, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
  if (handle == NULL) {
    const char* message = dlerror();
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "module '%s' containing address %p cannot be "
                            "pinned into the process: %s",
                            dl_info.dli_fname, address,
                            message ? message : "no error reported");
  }
  dlclose(handle);
  *out_pin = IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED;
  return iree_ok_status();
#else
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no module pin is available on this platform, so the AMDGPU system event "
      "callback cannot be given a lifetime the HSA runtime will respect");
#endif  // IREE_PLATFORM_WINDOWS
}

iree_status_t iree_hal_amdgpu_system_event_register_device(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_event_registration_t** out_registration) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_registration);
  *out_registration = NULL;
  iree_call_once(&iree_hal_amdgpu_system_event_once,
                 iree_hal_amdgpu_system_event_initialize);

  iree_hal_amdgpu_module_pin_t pin = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_system_event_pin_module_containing(
      (const void*)iree_hal_amdgpu_system_event_callback, &pin));

  const iree_host_size_t agent_count = logical_device->physical_device_count;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_system_event_registration_t), &total_size,
      IREE_STRUCT_FIELD_FAM(agent_count,
                            iree_hal_amdgpu_system_event_agent_target_t)));
  iree_hal_amdgpu_system_event_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&registration));
  memset(registration, 0, total_size);
  registration->logical_device = logical_device;
  registration->host_allocator = host_allocator;
  registration->agent_count = agent_count;
  for (iree_host_size_t i = 0; i < agent_count; ++i) {
    registration->agent_targets[i].agent =
        logical_device->physical_devices[i]->device_agent;
  }

  // The HSA registration runs under its own mutex because it is a call into the
  // same handler registry an event is dispatched from, and the traversal mutex
  // must not be held across one of those. Registering the same handler twice
  // silently succeeds and doubles delivery, so the one-shot has to be exact,
  // which is what serializing the whole check-call-record sequence gives.
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.hsa_handler_mutex);
  iree_status_t status = iree_ok_status();
  if (!iree_atomic_load(
          &iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered,
          iree_memory_order_acquire)) {
    status = iree_hsa_amd_register_system_event_handler(
        IREE_LIBHSA(libhsa), iree_hal_amdgpu_system_event_callback,
        /*data=*/NULL);
    if (iree_status_is_ok(status)) {
      iree_atomic_store(
          &iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered, 1,
          iree_memory_order_release);
    }
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.hsa_handler_mutex);

  if (iree_status_is_ok(status)) {
    iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
    registration->next =
        iree_hal_amdgpu_system_event_registry.registration_list;
    iree_hal_amdgpu_system_event_registry.registration_list = registration;
    iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
  }

  if (iree_status_is_ok(status)) {
    *out_registration = registration;
  } else {
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

void iree_hal_amdgpu_system_event_retire_device_status(
    iree_hal_amdgpu_system_event_registration_t* registration) {
  if (!registration) return;

  // Bounded to the store, on the same reasoning as queue target retirement.
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  registration->logical_device = NULL;
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
}

void iree_hal_amdgpu_system_event_unregister_device(
    iree_hal_amdgpu_system_event_registration_t* registration) {
  if (!registration) return;

  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  iree_hal_amdgpu_system_event_registration_t** registration_ptr =
      &iree_hal_amdgpu_system_event_registry.registration_list;
  while (*registration_ptr != NULL) {
    if (*registration_ptr == registration) {
      *registration_ptr = registration->next;
      break;
    }
    registration_ptr = &(*registration_ptr)->next;
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);

  // Unlinking under the mutex the callback holds across its whole traversal is
  // what makes this a quiescence barrier: no callback can be inside the
  // registration once the mutex is released.
  //
  // The walk finds it. Registration returns a handle only after linking it and
  // frees it itself otherwise, so the sole way to reach here with an unlinked
  // registration is to unregister one twice - which has already read freed
  // memory to get through the walk. Freeing only on a hit would trade that for
  // a leak rather than detect anything.
  iree_allocator_free(registration->host_allocator, registration);
}

iree_hal_amdgpu_system_event_agent_target_t*
iree_hal_amdgpu_system_event_registration_lookup_agent(
    iree_hal_amdgpu_system_event_registration_t* registration,
    hsa_agent_t agent) {
  if (!registration) return NULL;
  for (iree_host_size_t i = 0; i < registration->agent_count; ++i) {
    if (registration->agent_targets[i].agent.handle == agent.handle) {
      return &registration->agent_targets[i];
    }
  }
  return NULL;
}

void iree_hal_amdgpu_system_event_publish_queue_targets(
    iree_hal_amdgpu_system_event_agent_target_t* target,
    iree_hal_amdgpu_host_queue_t* host_queues,
    iree_host_size_t live_queue_count) {
  IREE_ASSERT_TRUE(live_queue_count <= IREE_HAL_MAX_QUEUES);
  live_queue_count =
      iree_min(live_queue_count, (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  const uint64_t live_queue_mask = live_queue_count == IREE_HAL_MAX_QUEUES
                                       ? UINT64_MAX
                                       : (UINT64_C(1) << live_queue_count) - 1u;
  iree_hal_amdgpu_system_event_publish_queue_target_mask(
      target, host_queues, live_queue_count, live_queue_mask);
}

void iree_hal_amdgpu_system_event_publish_queue_target_mask(
    iree_hal_amdgpu_system_event_agent_target_t* target,
    iree_hal_amdgpu_host_queue_t* host_queues, iree_host_size_t queue_capacity,
    uint64_t live_queue_mask) {
  if (!target) return;
  IREE_ASSERT_TRUE(queue_capacity <= IREE_HAL_MAX_QUEUES);
  queue_capacity =
      iree_min(queue_capacity, (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  const uint64_t capacity_mask = queue_capacity == IREE_HAL_MAX_QUEUES
                                     ? UINT64_MAX
                                     : (UINT64_C(1) << queue_capacity) - 1u;

  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  target->host_queues = host_queues;
  target->queue_capacity = queue_capacity;
  target->live_queue_mask = live_queue_mask & capacity_mask;
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
}

void iree_hal_amdgpu_system_event_retire_queue_targets(
    iree_hal_amdgpu_system_event_agent_target_t* target) {
  if (!target) return;

  // Bounded to the store so the mutex acquisition is a quiescence barrier
  // rather than a scope: holding it across queue destruction would serialize
  // every device's teardown against every other device's fault delivery.
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  target->host_queues = NULL;
  target->queue_capacity = 0;
  target->live_queue_mask = 0;
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
}
