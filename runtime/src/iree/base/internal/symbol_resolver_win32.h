// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_SYMBOL_RESOLVER_WIN32_H_
#define IREE_BASE_INTERNAL_SYMBOL_RESOLVER_WIN32_H_

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4091)
#endif  // _MSC_VER
#include <dbghelp.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif  // _MSC_VER

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The process-lifetime DbgHelp session shared by diagnostics and tracing.
// All symbol-handler calls, including use of borrowed result strings, require
// iree_symbol_resolver_lock. The table and process handle are immutable after
// initialization. Callers neither close the handle nor call SymCleanup.
typedef struct iree_symbol_resolver_t {
  // Unique owned process handle identifying this symbol session.
  HANDLE process;
  // Resolves an address to its containing module.
  BOOL(WINAPI* SymGetModuleInfo64)(HANDLE, DWORD64, PIMAGEHLP_MODULE64);
  // Resolves an address to its symbol and displacement.
  BOOL(WINAPI* SymFromAddr)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
  // Resolves an address to a source location with a borrowed filename.
  BOOL(WINAPI* SymGetLineFromAddr64)(HANDLE, DWORD64, PDWORD, PIMAGEHLP_LINE64);
  // Registers a module or explicit PDB in the shared session.
  DWORD64(WINAPI* SymLoadModuleEx)(HANDLE, HANDLE, PCSTR, PCSTR, DWORD64, DWORD,
                                   PMODLOAD_DATA, DWORD);
} iree_symbol_resolver_t;

// Initializes the session once and returns a borrowed process-lifetime table.
// Returns ERROR_SUCCESS on success or a cached Win32 initialization error with
// a NULL table. Uses no IREE status allocation: this is called while formatting
// allocation failures and other status payloads. Must be called outside the
// lock.
DWORD iree_symbol_resolver_get(const iree_symbol_resolver_t** out_resolver);

// Serializes DbgHelp access. Available before session initialization so offline
// tracing can protect its module cache without loading the symbol handler.
void iree_symbol_resolver_lock(void);
void iree_symbol_resolver_unlock(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_BASE_INTERNAL_SYMBOL_RESOLVER_WIN32_H_
