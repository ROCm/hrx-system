// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GetProcessInformation and SetProcessInformation require Windows 8.
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif

#include "iree/base/threading/latency.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <wchar.h>
#include <windows.h>

// Windows exposes one process policy without independently owned native
// requests. A named mutex and mapping coordinate statically embedded runtime
// copies. Names identify the process instance, including creation time to keep
// reused process IDs from sharing policy state. Only first/last participation
// by an image touches these native objects; overlapping local scopes use an
// SRW lock and count, with no per-scope allocation or work while held.

enum {
  IREE_LATENCY_SHARED_VERSION = 1,
  IREE_LATENCY_SHARED_ABANDONED = 0xFFFFFFFFu,
};

// This record has a fixed native layout shared across independently linked
// runtime copies. The name identifies the policy; version mismatch rejects
// acquisition instead of opening an independent policy owner.
typedef struct iree_latency_shared_state_t {
  // Shared protocol version; zero in a newly created mapping.
  DWORD version;
  // Number of library images with at least one active scope.
  DWORD participant_count;
  // Execution-speed control bit captured before the first participant.
  DWORD saved_control_mask;
  // Execution-speed state bit captured before the first participant.
  DWORD saved_state_mask;
} iree_latency_shared_state_t;

typedef struct iree_latency_connection_t {
  // Named native mutex serializing process-wide participation transitions.
  HANDLE mutex;
  // Shared policy-state mapping, retained while this image participates.
  HANDLE mapping;
  // This image's mapped view; contains no pointers or image-owned storage.
  iree_latency_shared_state_t* shared;
} iree_latency_connection_t;

static struct {
  // Serializes local scope transitions and connection lifetime.
  SRWLOCK mutex;
  // Number of active scopes in this library image.
  uintptr_t scope_count;
  // One shared-policy participation while scope_count is nonzero.
  iree_latency_connection_t connection;
} iree_latency_module_state = {SRWLOCK_INIT, 0, {0}};

static void iree_latency_connection_close(
    iree_latency_connection_t* connection) {
  if (connection->shared) {
    UnmapViewOfFile(connection->shared);
  }
  if (connection->mapping) {
    CloseHandle(connection->mapping);
  }
  if (connection->mutex) {
    CloseHandle(connection->mutex);
  }
  *connection = (iree_latency_connection_t){0};
}

static DWORD iree_latency_connection_begin(
    iree_latency_connection_t* connection) {
  FILETIME creation, exit, kernel, user;
  if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
    return GetLastError();
  }
  wchar_t mutex_name[128];
  wchar_t mapping_name[128];
  swprintf_s(mutex_name, IREE_ARRAYSIZE(mutex_name),
             L"Local\\IREE.LatencyScope.%lu.%08lx%08lx.mutex",
             GetCurrentProcessId(), creation.dwHighDateTime,
             creation.dwLowDateTime);
  swprintf_s(mapping_name, IREE_ARRAYSIZE(mapping_name),
             L"Local\\IREE.LatencyScope.%lu.%08lx%08lx.state",
             GetCurrentProcessId(), creation.dwHighDateTime,
             creation.dwLowDateTime);
  connection->mutex =
      CreateMutexExW(NULL, mutex_name, 0, SYNCHRONIZE | MUTEX_MODIFY_STATE);
  if (!connection->mutex) {
    return GetLastError();
  }
  DWORD wait_result = WaitForSingleObject(connection->mutex, INFINITE);
  if (wait_result != WAIT_OBJECT_0 && wait_result != WAIT_ABANDONED) {
    return GetLastError();
  }
  DWORD error = ERROR_SUCCESS;
  SetLastError(ERROR_SUCCESS);
  connection->mapping =
      CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
                         sizeof(iree_latency_shared_state_t), mapping_name);
  if (!connection->mapping) {
    error = GetLastError();
  }
  const bool created =
      connection->mapping && GetLastError() != ERROR_ALREADY_EXISTS;
  if (error == ERROR_SUCCESS) {
    connection->shared =
        MapViewOfFile(connection->mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0,
                      sizeof(iree_latency_shared_state_t));
    if (!connection->shared) {
      error = GetLastError();
    }
  }
  if (error == ERROR_SUCCESS) {
    iree_latency_shared_state_t* shared = connection->shared;
    if (created) {
      shared->version = IREE_LATENCY_SHARED_VERSION;
    }
    if (wait_result == WAIT_ABANDONED) {
      // A terminated owner may have changed policy without committing the
      // ownership transition. Preserve the request and reject new ownership.
      shared->version = IREE_LATENCY_SHARED_ABANDONED;
      error = ERROR_ABANDONED_WAIT_0;
    } else if (shared->version != IREE_LATENCY_SHARED_VERSION) {
      error = ERROR_REVISION_MISMATCH;
    } else if (shared->participant_count == 0) {
      PROCESS_POWER_THROTTLING_STATE policy = {
          .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
      };
      if (!GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                 &policy, sizeof(policy))) {
        error = GetLastError();
      } else {
        const DWORD mask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        shared->saved_control_mask = policy.ControlMask & mask;
        shared->saved_state_mask = policy.StateMask & mask;
        policy.ControlMask |= mask;
        policy.StateMask &= ~mask;
        if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                   &policy, sizeof(policy))) {
          error = GetLastError();
        }
      }
    }
    if (error == ERROR_SUCCESS) {
      ++shared->participant_count;
    }
  }
  ReleaseMutex(connection->mutex);
  return error;
}

static DWORD iree_latency_connection_end(
    iree_latency_connection_t* connection) {
  DWORD wait_result = WaitForSingleObject(connection->mutex, INFINITE);
  if (wait_result != WAIT_OBJECT_0 && wait_result != WAIT_ABANDONED) {
    return GetLastError();
  }
  DWORD error = ERROR_SUCCESS;
  iree_latency_shared_state_t* shared = connection->shared;
  if (wait_result == WAIT_ABANDONED) {
    shared->version = IREE_LATENCY_SHARED_ABANDONED;
    error = ERROR_ABANDONED_WAIT_0;
  } else if (shared->version != IREE_LATENCY_SHARED_VERSION) {
    error = ERROR_REVISION_MISMATCH;
  } else if (shared->participant_count == 1) {
    PROCESS_POWER_THROTTLING_STATE policy = {
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
    };
    if (!GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                               &policy, sizeof(policy))) {
      error = GetLastError();
    } else {
      // Restore only execution-speed policy, preserving independently changed
      // controls such as timer-resolution handling.
      const DWORD mask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      policy.ControlMask =
          (policy.ControlMask & ~mask) | shared->saved_control_mask;
      policy.StateMask = (policy.StateMask & ~mask) | shared->saved_state_mask;
      if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                 &policy, sizeof(policy))) {
        error = GetLastError();
      }
    }
  }
  if (error == ERROR_SUCCESS) {
    --shared->participant_count;
  }
  ReleaseMutex(connection->mutex);
  if (error == ERROR_SUCCESS) {
    iree_latency_connection_close(connection);
  }
  return error;
}

bool iree_latency_scope_begin(iree_latency_scope_options_t options,
                              iree_latency_scope_t* out_scope) {
  IREE_ASSERT_ARGUMENT(out_scope);
  out_scope->handle = 0;
  if (options.maximum_cpu_idle_latency_us != -1) {
    return false;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  DWORD error = ERROR_SUCCESS;
  AcquireSRWLockExclusive(&iree_latency_module_state.mutex);
  if (iree_latency_module_state.scope_count == 0) {
    error =
        iree_latency_connection_begin(&iree_latency_module_state.connection);
    if (error != ERROR_SUCCESS) {
      iree_latency_connection_close(&iree_latency_module_state.connection);
    }
  }
  if (error == ERROR_SUCCESS) {
    ++iree_latency_module_state.scope_count;
    out_scope->handle = 1;
  }
  ReleaseSRWLockExclusive(&iree_latency_module_state.mutex);
  IREE_TRACE_ZONE_END(z0);
  return error == ERROR_SUCCESS;
}

iree_status_t iree_latency_scope_end(iree_latency_scope_t* scope) {
  if (!scope->handle) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  DWORD error = ERROR_SUCCESS;
  AcquireSRWLockExclusive(&iree_latency_module_state.mutex);
  if (iree_latency_module_state.scope_count == 1) {
    error = iree_latency_connection_end(&iree_latency_module_state.connection);
  }
  if (error == ERROR_SUCCESS) {
    --iree_latency_module_state.scope_count;
    scope->handle = 0;
  }
  ReleaseSRWLockExclusive(&iree_latency_module_state.mutex);
  IREE_TRACE_ZONE_END(z0);
  return error == ERROR_SUCCESS
             ? iree_ok_status()
             : iree_make_status(iree_status_code_from_win32_error(error),
                                "releasing process latency policy failed (%lu)",
                                error);
}

#endif  // IREE_PLATFORM_WINDOWS
