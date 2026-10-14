// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/symbol_resolver_win32.h"

#if defined(IREE_PLATFORM_WINDOWS)

static SRWLOCK iree_symbol_resolver_mutex = SRWLOCK_INIT;
static INIT_ONCE iree_symbol_resolver_once = INIT_ONCE_STATIC_INIT;
static iree_symbol_resolver_t iree_symbol_resolver_instance;
static DWORD iree_symbol_resolver_error = ERROR_SUCCESS;

void iree_symbol_resolver_lock(void) {
  AcquireSRWLockExclusive(&iree_symbol_resolver_mutex);
}

void iree_symbol_resolver_unlock(void) {
  ReleaseSRWLockExclusive(&iree_symbol_resolver_mutex);
}

static DWORD iree_symbol_resolver_initialize(iree_symbol_resolver_t* resolver) {
  HMODULE library = LoadLibraryA("dbghelp.dll");
  if (!library) {
    return GetLastError();
  }

  typedef BOOL(WINAPI * initialize_fn_t)(HANDLE, PCSTR, BOOL);
  typedef DWORD(WINAPI * get_options_fn_t)(void);
  typedef DWORD(WINAPI * set_options_fn_t)(DWORD);
  initialize_fn_t initialize =
      (initialize_fn_t)GetProcAddress(library, "SymInitialize");
  get_options_fn_t get_options =
      (get_options_fn_t)GetProcAddress(library, "SymGetOptions");
  set_options_fn_t set_options =
      (set_options_fn_t)GetProcAddress(library, "SymSetOptions");
  resolver->SymGetModuleInfo64 =
      (BOOL(WINAPI*)(HANDLE, DWORD64, PIMAGEHLP_MODULE64))GetProcAddress(
          library, "SymGetModuleInfo64");
  resolver->SymFromAddr =
      (BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO))GetProcAddress(
          library, "SymFromAddr");
  resolver->SymGetLineFromAddr64 =
      (BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD, PIMAGEHLP_LINE64))GetProcAddress(
          library, "SymGetLineFromAddr64");
  resolver->SymLoadModuleEx = (DWORD64(WINAPI*)(
      HANDLE, HANDLE, PCSTR, PCSTR, DWORD64, DWORD, PMODLOAD_DATA,
      DWORD))GetProcAddress(library, "SymLoadModuleEx");
  if (!initialize || !get_options || !set_options ||
      !resolver->SymGetModuleInfo64 || !resolver->SymFromAddr ||
      !resolver->SymGetLineFromAddr64 || !resolver->SymLoadModuleEx) {
    FreeLibrary(library);
    return ERROR_PROC_NOT_FOUND;
  }

  // The GetCurrentProcess pseudo-handle would collide with independent users
  // of DbgHelp. A real duplicate remains valid for process enumeration while
  // giving this session a unique identity.
  HANDLE process = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(),
                       GetCurrentProcess(), &process, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD error = GetLastError();
    FreeLibrary(library);
    return error;
  }
  set_options(get_options() | SYMOPT_LOAD_LINES);
  if (!initialize(process, /*UserSearchPath=*/NULL, /*fInvadeProcess=*/TRUE)) {
    DWORD error = GetLastError();
    CloseHandle(process);
    FreeLibrary(library);
    return error ? error : ERROR_DLL_INIT_FAILED;
  }

  // DbgHelp can be queried during static destruction and by delayed profiler
  // requests. Retain the DLL and session until process exit instead of ordering
  // cleanup against every diagnostic producer and consumer.
  resolver->process = process;
  return ERROR_SUCCESS;
}

static BOOL CALLBACK iree_symbol_resolver_setup(PINIT_ONCE once,
                                                PVOID parameter,
                                                PVOID* context) {
  (void)once;
  (void)parameter;
  (void)context;
  iree_symbol_resolver_lock();
  iree_symbol_resolver_error =
      iree_symbol_resolver_initialize(&iree_symbol_resolver_instance);
  iree_symbol_resolver_unlock();
  return TRUE;
}

DWORD iree_symbol_resolver_get(const iree_symbol_resolver_t** out_resolver) {
  InitOnceExecuteOnce(&iree_symbol_resolver_once, iree_symbol_resolver_setup,
                      NULL, NULL);
  *out_resolver = iree_symbol_resolver_error == ERROR_SUCCESS
                      ? &iree_symbol_resolver_instance
                      : NULL;
  return iree_symbol_resolver_error;
}

#endif  // IREE_PLATFORM_WINDOWS
