// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off: must be included before all other headers.
#include "iree/base/threading/thread_impl.h"
// clang-format on

#if defined(IREE_PLATFORM_WINDOWS)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"

// Great documentation:
// https://www.microsoftpressstore.com/articles/article.aspx?p=2233328

struct iree_thread_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[32];
  HANDLE handle;
  DWORD id;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t is_suspended;

  // Thread-safe (has its own synchronization).
  iree_thread_override_list_t qos_override_list;
};

static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

// Sets the thread's name to the given NUL-terminated string.
//
// See:
// https://docs.microsoft.com/en-us/visualstudio/debugger/how-to-set-a-thread-name-in-native-code
static void iree_thread_set_name(HANDLE handle, const char* name) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try first to use the modern SetThreadDescription API.
  // This will work even if a debugger is not attached meaning that tools that
  // don't use the debugger API can still query thread names. It's only
  // available on Win10+.
  typedef HRESULT(WINAPI * SetThreadDescriptionFn)(HANDLE hThread,
                                                   PCWSTR lpThreadDescription);
  SetThreadDescriptionFn pSetThreadDescription =
      (SetThreadDescriptionFn)GetProcAddress(GetModuleHandleW(L"Kernel32.dll"),
                                             "SetThreadDescription");
  if (pSetThreadDescription) {
    wchar_t name_wide[16] = {0};
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, -1, name_wide,
                        IREE_ARRAYSIZE(name_wide) - 1);
    pSetThreadDescription(handle, name_wide);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  if (!IsDebuggerPresent()) {
    // The name is only captured if a debugger is attached so we can avoid
    // doing any of the work if none is present. This means that a debugger
    // attached to the process after thread creation won't see thread names but
    // that's a rare case anyway.
    IREE_TRACE_ZONE_END(z0);
    return;
  }

#pragma pack(push, 8)
  struct THREADNAME_INFO {
    DWORD dwType;      // Must be 0x1000.
    LPCSTR szName;     // Pointer to name (in user addr space).
    DWORD dwThreadID;  // Thread ID (-1=caller thread).
    DWORD dwFlags;     // Reserved for future use, must be zero.
  };
#pragma pack(pop)

#pragma warning(push)
#pragma warning(disable : 6320 6322)
  struct THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name;
  info.dwThreadID = GetThreadId(handle);
  info.dwFlags = 0;
  __try {
    RaiseException(0x406D1388u, 0, sizeof(info) / sizeof(ULONG_PTR),
                   (ULONG_PTR*)(&info));
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
#pragma warning(pop)

  IREE_TRACE_ZONE_END(z0);
}

static DWORD WINAPI iree_thread_start_routine(LPVOID param) {
  iree_thread_t* thread = (iree_thread_t*)param;

  // Set the thread name used by tracy (which must be called on the thread).
  IREE_TRACE_SET_THREAD_NAME(thread->name);

  // "Consume" the entry info so that we don't see it again (as we don't own
  // its lifetime).
  iree_thread_entry_t entry = thread->entry;
  void* entry_arg = thread->entry_arg;
  thread->entry = NULL;
  thread->entry_arg = NULL;

  // Call the user thread entry point function.
  // Note that this can be a tail-call which saves a stack frame in all threads
  // (which is really just to make call stacks in debuggers much cleaner).
  // Failed startup resumes the native thread with no user entry to invoke.
  return entry ? (DWORD)entry(entry_arg) : 0;
}

iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_thread = NULL;

  // Allocate our thread struct; we'll use it to shuttle params into the thread
  // (including the user-specified entry_arg).
  iree_thread_t* thread = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*thread), (void**)&thread);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_atomic_ref_count_init(&thread->ref_count);
  thread->allocator = allocator;
  thread->entry = entry;
  thread->entry_arg = entry_arg;
  strncpy_s(thread->name, IREE_ARRAYSIZE(thread->name), params.name.data,
            min(params.name.size, IREE_ARRAYSIZE(thread->name) - 1));
  iree_atomic_store(&thread->is_suspended, 1, iree_memory_order_relaxed);
  iree_thread_override_list_initialize(iree_thread_set_priority_class,
                                       params.priority_class, thread->allocator,
                                       &thread->qos_override_list);

  // Configure affinity before allowing the native thread to invoke user code.
  thread->handle =
      CreateThread(NULL, params.stack_size, iree_thread_start_routine, thread,
                   CREATE_SUSPENDED, &thread->id);
  if (!thread->handle) {
    DWORD error = GetLastError();
    status = iree_make_status(iree_status_code_from_win32_error(error),
                              "thread creation failed with %lu", error);
  } else {
    if (!iree_string_view_is_empty(params.name)) {
      iree_thread_set_name(thread->handle, thread->name);
    }
    if (params.priority_class != IREE_THREAD_PRIORITY_CLASS_NORMAL) {
      iree_thread_set_priority_class(thread, params.priority_class);
    }
    status = iree_thread_request_affinity(thread, params.initial_affinity,
                                          params.initial_processor_mask);
  }

  if (iree_status_is_ok(status)) {
    *out_thread = thread;
    if (!params.create_suspended) {
      iree_thread_resume(thread);
    }
  } else if (thread->handle) {
    thread->entry = NULL;
    thread->entry_arg = NULL;
    iree_thread_release(thread);
  } else {
    iree_thread_override_list_deinitialize(&thread->qos_override_list);
    iree_allocator_free(allocator, thread);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_thread_delete(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_thread_resume(thread);

  if (thread->id != GetCurrentThreadId()) {
    // Join with the thread. Since threads can delete themselves we must ensure
    // they don't try to join with themselves and deadlock.
    WaitForSingleObject(thread->handle, INFINITE);
  }
  CloseHandle(thread->handle);
  iree_thread_override_list_deinitialize(&thread->qos_override_list);
  iree_allocator_free(thread->allocator, thread);

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_retain(iree_thread_t* thread) {
  if (thread) {
    iree_atomic_ref_count_inc(&thread->ref_count);
  }
}

void iree_thread_release(iree_thread_t* thread) {
  if (thread && iree_atomic_ref_count_dec(&thread->ref_count) == 1) {
    iree_thread_delete(thread);
  }
}

uintptr_t iree_thread_id(iree_thread_t* thread) {
  return (uintptr_t)thread->id;
}

// Sets the thread priority to the given |priority_class| immediately.
static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

  DWORD priority = THREAD_PRIORITY_NORMAL;
  switch (priority_class) {
    case IREE_THREAD_PRIORITY_CLASS_LOWEST:
      priority = THREAD_PRIORITY_LOWEST;
      break;
    case IREE_THREAD_PRIORITY_CLASS_LOW:
      priority = THREAD_PRIORITY_BELOW_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_NORMAL:
      priority = THREAD_PRIORITY_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      priority = THREAD_PRIORITY_ABOVE_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      priority = THREAD_PRIORITY_HIGHEST;
      break;
  }
  SetThreadPriority(thread->handle, priority);

  IREE_TRACE_ZONE_END(z0);
}

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_t* override = iree_thread_override_list_add(
      &thread->qos_override_list, thread, priority_class);
  IREE_TRACE_ZONE_END(z0);
  return override;
}

void iree_thread_override_end(iree_thread_override_t* override) {
  if (!override) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_remove_self(override);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_thread_request_affinity(iree_thread_t* thread,
                                           iree_thread_affinity_t affinity,
                                           iree_bitmap_t processor_mask) {
  if (iree_thread_affinity_is_unspecified(affinity) &&
      !processor_mask.bit_count) {
    return iree_ok_status();
  }
  if (processor_mask.bit_count && !processor_mask.words) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "processor affinity mask has no storage");
  }
  GROUP_AFFINITY group_affinity = {0};
  if (!iree_thread_affinity_is_unspecified(affinity) &&
      affinity.group > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity group %u exceeds Windows node/group IDs",
                            affinity.group);
  }
  if (affinity.group_any) {
    // NUMA node IDs and processor group IDs are different namespaces. A node
    // spanning several groups uses its primary group's node-local processors.
    if (!GetNumaNodeProcessorMaskEx((USHORT)affinity.group, &group_affinity)) {
      DWORD error = GetLastError();
      return iree_make_status(iree_status_code_from_win32_error(error),
                              "cannot resolve NUMA node %u CPUs: %lu",
                              affinity.group, error);
    }
    if (!group_affinity.Mask) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "NUMA node %u has no CPUs", affinity.group);
    }
  } else if (affinity.id_assigned) {
    if (affinity.id + affinity.smt >= sizeof(KAFFINITY) * 8) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "CPU %u exceeds Windows processor group mask",
                              affinity.id);
    }
    group_affinity.Group = (WORD)affinity.group;
    group_affinity.Mask = (KAFFINITY)1 << affinity.id;
    if (affinity.smt) {
      group_affinity.Mask |= (KAFFINITY)1 << (affinity.id + 1);
    }
  } else {
    const iree_host_size_t first_cpu =
        iree_bitmap_find_first_set(processor_mask, 0);
    if (first_cpu == processor_mask.bit_count) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "processor affinity mask contains no CPUs");
    }
    if (first_cpu / 64 > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "processor affinity exceeds Windows group IDs");
    }
    group_affinity.Group = (WORD)(first_cpu / 64);
    group_affinity.Mask = ~(KAFFINITY)0;
    const iree_host_size_t next_group_cpu = (first_cpu / 64 + 1) * 64;
    if (next_group_cpu < processor_mask.bit_count &&
        iree_bitmap_find_first_set(processor_mask, next_group_cpu) !=
            processor_mask.bit_count) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "Windows hard affinity requires one processor group; "
          "select a CPU or NUMA node within the processor budget");
    }
  }
  if (processor_mask.bit_count) {
    KAFFINITY permitted_mask = 0;
    const iree_host_size_t first_cpu =
        (iree_host_size_t)group_affinity.Group * 64;
    const iree_host_size_t native_end = first_cpu + sizeof(KAFFINITY) * 8;
    if (iree_thread_affinity_is_unspecified(affinity) &&
        sizeof(KAFFINITY) < sizeof(uint64_t) &&
        native_end < processor_mask.bit_count &&
        iree_bitmap_find_first_set(processor_mask, native_end) <
            iree_min(first_cpu + 64, processor_mask.bit_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "processor affinity exceeds native group mask");
    }
    for (iree_host_size_t bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit) {
      const iree_host_size_t cpu = first_cpu + bit;
      if (cpu < processor_mask.bit_count &&
          iree_bitmap_test(processor_mask, cpu)) {
        permitted_mask |= (KAFFINITY)1 << bit;
      }
    }
    group_affinity.Mask &= permitted_mask;
    if (!group_affinity.Mask) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "requested affinity has no CPUs in the processor budget");
    }
  }
  if (!SetThreadGroupAffinity(thread->handle, &group_affinity, NULL)) {
    DWORD error = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "cannot apply thread affinity: %lu", error);
  }
  return iree_ok_status();
}

void iree_thread_resume(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: we don't track the suspend/resume depth here because we don't
  // expose suspend as an operation (yet). If we did we'd want to make sure we
  // always balance suspend/resume or else we'll mess with any
  // debuggers/profilers that may be suspending threads for their own uses.
  int32_t expected = 1;
  if (iree_atomic_compare_exchange_strong(
          &thread->is_suspended, &expected, 0, iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* expected is unused */)) {
    ResumeThread(thread->handle);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_join(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  WaitForSingleObject(thread->handle, INFINITE);
  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_yield(void) { YieldProcessor(); }

#endif  // IREE_PLATFORM_WINDOWS
