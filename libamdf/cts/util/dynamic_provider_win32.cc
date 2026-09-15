// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "util/provider.h"

static HMODULE amdf_cts_library_handle = nullptr;
static amdf_query_api_fn_t amdf_cts_query_api = nullptr;

static const char* amdf_cts_consume_library_argument(int* argument_count,
                                                     char*** argument_values) {
  static const char prefix[] = "--amdf_library=";
  char** values = *argument_values;
  for (int i = 1; i < *argument_count; ++i) {
    if (std::strncmp(values[i], prefix, sizeof(prefix) - 1) != 0) {
      continue;
    }
    const char* path = values[i] + sizeof(prefix) - 1;
    for (int j = i; j + 1 < *argument_count; ++j) {
      values[j] = values[j + 1];
    }
    values[--*argument_count] = nullptr;
    return path;
  }
  return nullptr;
}

static HMODULE amdf_cts_load_library(const char* path) {
  const int character_count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
  if (character_count == 0) {
    return nullptr;
  }
  std::vector<wchar_t> wide_path(character_count);
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
                          wide_path.data(), character_count) == 0) {
    return nullptr;
  }
  return LoadLibraryW(wide_path.data());
}

int amdf_cts_provider_initialize(int* argument_count, char*** argument_values) {
  const char* library_path =
      amdf_cts_consume_library_argument(argument_count, argument_values);
  if (library_path == nullptr || library_path[0] == 0) {
    std::fprintf(stderr, "missing --amdf_library=<path>\n");
    return 0;
  }

  amdf_cts_library_handle = amdf_cts_load_library(library_path);
  if (amdf_cts_library_handle == nullptr) {
    std::fprintf(stderr, "unable to load %s: Win32 error %lu\n", library_path,
                 static_cast<unsigned long>(GetLastError()));
    return 0;
  }
  amdf_cts_query_api = reinterpret_cast<amdf_query_api_fn_t>(
      GetProcAddress(amdf_cts_library_handle, AMDF_QUERY_API_SYMBOL));
  if (amdf_cts_query_api == nullptr) {
    std::fprintf(stderr, "unable to resolve %s: Win32 error %lu\n",
                 AMDF_QUERY_API_SYMBOL,
                 static_cast<unsigned long>(GetLastError()));
    FreeLibrary(amdf_cts_library_handle);
    amdf_cts_library_handle = nullptr;
    return 0;
  }
  return 1;
}

int amdf_cts_provider_deinitialize(void) {
  amdf_cts_query_api = nullptr;
  if (amdf_cts_library_handle != nullptr) {
    if (!FreeLibrary(amdf_cts_library_handle)) {
      std::fprintf(stderr, "unable to unload libamdf: Win32 error %lu\n",
                   static_cast<unsigned long>(GetLastError()));
      return 0;
    }
    amdf_cts_library_handle = nullptr;
  }
  return 1;
}

amdf_query_api_fn_t amdf_cts_provider_query_api(void) {
  return amdf_cts_query_api;
}
