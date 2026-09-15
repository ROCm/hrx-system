// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

#include "util/provider.h"

static void* amdf_cts_library_handle = nullptr;
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

int amdf_cts_provider_initialize(int* argument_count, char*** argument_values) {
  const char* library_path =
      amdf_cts_consume_library_argument(argument_count, argument_values);
  if (library_path == nullptr || library_path[0] == 0) {
    std::fprintf(stderr, "missing --amdf_library=<path>\n");
    return 0;
  }

  amdf_cts_library_handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (amdf_cts_library_handle == nullptr) {
    std::fprintf(stderr, "unable to load %s: %s\n", library_path, dlerror());
    return 0;
  }
  dlerror();
  amdf_cts_query_api = reinterpret_cast<amdf_query_api_fn_t>(
      dlsym(amdf_cts_library_handle, AMDF_QUERY_API_SYMBOL));
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || amdf_cts_query_api == nullptr) {
    std::fprintf(stderr, "unable to resolve %s: %s\n", AMDF_QUERY_API_SYMBOL,
                 symbol_error == nullptr ? "symbol not found" : symbol_error);
    dlclose(amdf_cts_library_handle);
    amdf_cts_library_handle = nullptr;
    amdf_cts_query_api = nullptr;
    return 0;
  }
  return 1;
}

int amdf_cts_provider_deinitialize(void) {
  amdf_cts_query_api = nullptr;
  if (amdf_cts_library_handle != nullptr) {
    if (dlclose(amdf_cts_library_handle) != 0) {
      std::fprintf(stderr, "unable to unload libamdf: %s\n", dlerror());
      return 0;
    }
    amdf_cts_library_handle = nullptr;
  }
  return 1;
}

amdf_query_api_fn_t amdf_cts_provider_query_api(void) {
  return amdf_cts_query_api;
}
