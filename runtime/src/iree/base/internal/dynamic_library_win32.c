// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/csprng.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

#if defined(IREE_PLATFORM_WINDOWS)

// PDB overlays are registered when the profiling symbol store is active.
#if defined(TRACY_ENABLE)
#define IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT 1
#pragma warning(disable : 4091)
#include <psapi.h>

#include "iree/base/internal/symbol_resolver_win32.h"
#endif  // TRACY_ENABLE

struct iree_dynamic_library_t {
  // Number of references to the library wrapper.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the wrapper and its path strings.
  iree_allocator_t allocator;

  // Flags controlling module lifetime.
  iree_dynamic_library_flags_t flags;

  // Base module name used as an identifier. When loaded from a file this must
  // be the basename for dbghelp to be able to find symbols.
  // Owned and allocated as part of the struct upon creation.
  // Has NUL terminator for compatibility with Windows APIs.
  char* identifier;

  // File path of the loaded module, if loaded from one.
  // Owned and allocated as part of the struct upon creation.
  // Has NUL terminator for compatibility with Windows APIs.
  char* module_path;

  // Windows module handle.
  HMODULE module;

  // Delete-on-close handle retaining an extracted module until it is unmapped.
  // NULL when the module was loaded from a caller-owned file.
  HANDLE module_file;

  // Delete-on-close handle retaining an extracted debug database.
  // NULL when no debug database was extracted from memory.
  HANDLE symbol_file;
};

// Writes an exclusive temporary file and returns a read-only delete-on-close
// handle. The caller keeps the handle open while the loader or symbol handler
// may need to reopen the path. Output ownership transfers only on success.
static iree_status_t iree_dynamic_library_write_temp_file(
    iree_const_byte_span_t source_data, const char* extension,
    iree_host_size_t file_path_capacity, char* file_path, HANDLE* out_file) {
  *out_file = NULL;
  file_path[0] = 0;

  char temp_path[MAX_PATH + 1];
  DWORD temp_path_length = GetTempPathA(sizeof(temp_path), temp_path);
  if (!temp_path_length) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "querying temporary directory");
  }
  if (temp_path_length >= sizeof(temp_path)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary directory path is too long");
  }

  uint64_t unique_id[2];
  IREE_RETURN_IF_ERROR(
      iree_csprng_fill(iree_make_byte_span(unique_id, sizeof(unique_id))));
  int file_path_length =
      iree_snprintf(file_path, file_path_capacity,
                    "%siree_dylib_%016" PRIX64 "%016" PRIX64 ".%s", temp_path,
                    unique_id[0], unique_id[1], extension);
  if (file_path_length < 0 ||
      (iree_host_size_t)file_path_length >= file_path_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary library path is too long");
  }

  // Never overwrite an existing path. The write handle prevents other writers
  // until all bytes have been installed; executable mappings require it closed.
  HANDLE writer = CreateFileA(file_path, GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                              CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, NULL);
  if (writer == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "creating temporary file '%s'", file_path);
  }

  // Reopen the same file, without another path lookup, before writing.
  // The read handle owns cleanup on both write failure and eventual unload.
  HANDLE file =
      ReOpenFile(writer, GENERIC_READ | DELETE,
                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                 FILE_FLAG_DELETE_ON_CLOSE);
  iree_status_t status = iree_ok_status();
  if (file == INVALID_HANDLE_VALUE) {
    status = iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "retaining temporary file '%s'", file_path);
    if (!DeleteFileA(file_path)) {
      status = iree_status_join(
          status,
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "removing temporary file '%s'", file_path));
    }
  }

  iree_host_size_t position = 0;
  while (iree_status_is_ok(status) && position < source_data.data_length) {
    DWORD write_length = (DWORD)iree_min(source_data.data_length - position,
                                         (iree_host_size_t)MAXDWORD);
    DWORD written = 0;
    if (!WriteFile(writer, source_data.data + position, write_length, &written,
                   NULL)) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "writing temporary file '%s'", file_path);
    } else if (!written) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "no progress writing temporary file '%s'",
                                file_path);
    } else {
      position += written;
    }
  }
  CloseHandle(writer);

  if (iree_status_is_ok(status)) {
    *out_file = file;
  } else if (file != INVALID_HANDLE_VALUE) {
    CloseHandle(file);
  }
  return status;
}

// Allocates an iree_dynamic_library_t with the given allocator.
static iree_status_t iree_dynamic_library_create(
    iree_string_view_t identifier, iree_string_view_t module_path,
    HMODULE module, iree_dynamic_library_flags_t flags,
    iree_allocator_t allocator, iree_dynamic_library_t** out_library) {
  *out_library = NULL;

  iree_host_size_t identifier_storage_size = 0;
  iree_host_size_t module_path_storage_size = 0;
  if (!iree_host_size_checked_add(identifier.size, 1,
                                  &identifier_storage_size) ||
      !iree_host_size_checked_add(module_path.size, 1,
                                  &module_path_storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path length overflow");
  }
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t module_path_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_dynamic_library_t), &total_size,
      IREE_STRUCT_FIELD(identifier_storage_size, char, &identifier_offset),
      IREE_STRUCT_FIELD(module_path_storage_size, char, &module_path_offset)));

  iree_dynamic_library_t* library = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&library));
  memset(library, 0, total_size);
  iree_atomic_ref_count_init(&library->ref_count);
  library->allocator = allocator;
  library->flags = flags;
  library->module = module;

  library->identifier = (char*)library + identifier_offset;
  memcpy(library->identifier, identifier.data, identifier.size);
  library->identifier[identifier.size] = 0;  // NUL

  library->module_path = (char*)library + module_path_offset;
  memcpy(library->module_path, module_path.data, module_path.size);
  library->module_path[module_path.size] = 0;  // NUL

  *out_library = library;
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_load_from_file(
    const char* file_path, iree_dynamic_library_flags_t flags,
    iree_allocator_t allocator, iree_dynamic_library_t** out_library) {
  return iree_dynamic_library_load_from_files(1, &file_path, flags, allocator,
                                              out_library);
}

iree_status_t iree_dynamic_library_load_from_files(
    iree_host_size_t search_path_count, const char* const* search_paths,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  // Try to load the module from the set of search paths provided.
  HMODULE module = NULL;
  iree_host_size_t i = 0;
  for (i = 0; i < search_path_count; ++i) {
    module = LoadLibraryA(search_paths[i]);
    if (module) {
      break;
    }
  }
  if (!module) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "dynamic library not found on any search path");
  }

  iree_string_view_t file_path = iree_make_cstring_view(search_paths[i]);
  iree_string_view_t identifier = iree_file_path_basename(file_path);

  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_dynamic_library_create(
      identifier, file_path, module, flags, allocator, &library);

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    FreeLibrary(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_dynamic_library_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  char temp_path[MAX_PATH + 64];
  HANDLE file = NULL;
  iree_status_t status = iree_dynamic_library_write_temp_file(
      buffer, "dll", sizeof(temp_path), temp_path, &file);
  if (iree_status_is_ok(status)) {
    status = iree_dynamic_library_load_from_file(temp_path, flags, allocator,
                                                 out_library);
  }
  if (iree_status_is_ok(status)) {
    (*out_library)->module_file = file;
  } else if (file) {
    CloseHandle(file);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_dynamic_library_delete(iree_dynamic_library_t* library) {
  iree_allocator_t allocator = library->allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  bool retain_mapping =
      iree_any_bit_set(library->flags, IREE_DYNAMIC_LIBRARY_FLAG_NODELETE);
#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION) || \
    defined(IREE_SANITIZER_ADDRESS)
  // Profilers and sanitizer reports can inspect the image after wrapper
  // release.
  retain_mapping = true;
#endif
  if (!retain_mapping) {
    FreeLibrary(library->module);
    if (library->module_file) {
      CloseHandle(library->module_file);
    }
    if (library->symbol_file) {
      CloseHandle(library->symbol_file);
    }
  }
  // Retained mappings and their non-inheritable backing handles live until
  // process exit. Windows closes the handles and retires their temporary files.

  iree_allocator_free(allocator, library);

  IREE_TRACE_ZONE_END(z0);
}

void iree_dynamic_library_retain(iree_dynamic_library_t* library) {
  if (library) {
    iree_atomic_ref_count_inc(&library->ref_count);
  }
}

void iree_dynamic_library_release(iree_dynamic_library_t* library) {
  if (library && iree_atomic_ref_count_dec(&library->ref_count) == 1) {
    iree_dynamic_library_delete(library);
  }
}

iree_status_t iree_dynamic_library_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(symbol_name);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = NULL;
  void* fn = GetProcAddress(library->module, symbol_name);
  if (!fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "symbol '%s' not found in library", symbol_name);
  }
  *out_fn = fn;
  return iree_ok_status();
}

void* iree_dynamic_library_try_lookup_symbol(iree_dynamic_library_t* library,
                                             const char* symbol_name) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(symbol_name);
  return (void*)GetProcAddress(library->module, symbol_name);
}

iree_status_t iree_dynamic_library_append_symbol_path_to_builder(
    void* symbol, iree_string_builder_t* builder) {
  HMODULE hm = NULL;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                        (LPCSTR)symbol, &hm) == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND);
  }

  char* path_buffer = NULL;
  iree_host_size_t path_capacity = 0;
  iree_host_size_t path_size = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      builder, 64, &path_buffer, &path_capacity));
  while (1) {
    path_size = GetModuleFileNameA(hm, path_buffer, path_capacity);
    if (path_size == 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND);
    }
    if (path_size == path_capacity ||
        GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
      IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
          builder, path_capacity + MAX_PATH, &path_buffer, &path_capacity));
      continue;
    }
    break;
  }
  iree_string_builder_commit_append(builder, path_size);
  return iree_ok_status();
}

#if defined(IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT)

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  // The loader already owns the module handle; query its mapping directly.
  MODULEINFO module_info;
  if (!GetModuleInformation(GetCurrentProcess(), library->module, &module_info,
                            sizeof(module_info))) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "querying module mapping for '%s'",
                            library->identifier);
  }
  const iree_symbol_resolver_t* resolver = NULL;
  DWORD error = iree_symbol_resolver_get(&resolver);
  if (error != ERROR_SUCCESS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "initializing symbol session");
  }

  iree_symbol_resolver_lock();
  DWORD64 module_base = resolver->SymLoadModuleEx(
      resolver->process, NULL, file_path, library->identifier,
      (DWORD64)module_info.lpBaseOfDll, module_info.SizeOfImage, NULL, 0);
  // An already registered module returns zero with ERROR_SUCCESS.
  error = module_base ? ERROR_SUCCESS : GetLastError();
  iree_symbol_resolver_unlock();

  IREE_TRACE_ZONE_END(z0);
  if (error != ERROR_SUCCESS) {
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "loading debug symbols from '%s'", file_path);
  }
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  char temp_path[MAX_PATH + 64];
  HANDLE file = NULL;
  iree_status_t status = iree_dynamic_library_write_temp_file(
      buffer, "pdb", sizeof(temp_path), temp_path, &file);
  if (iree_status_is_ok(status)) {
    status = iree_dynamic_library_attach_symbols_from_file(library, temp_path);
  }
  if (iree_status_is_ok(status)) {
    if (library->symbol_file) {
      CloseHandle(library->symbol_file);
    }
    library->symbol_file = file;
  } else if (file) {
    CloseHandle(file);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  return iree_ok_status();
}

#endif  // IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT

#endif  // IREE_PLATFORM_WINDOWS
