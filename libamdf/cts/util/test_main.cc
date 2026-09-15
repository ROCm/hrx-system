// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

int main(int argument_count, char** argument_values) {
  const char prefix[] = "--amdf_native_lifetime=";
  for (int i = 1; i < argument_count; ++i) {
    if (std::strncmp(argument_values[i], prefix, sizeof(prefix) - 1) != 0) {
      continue;
    }
    const char* value = argument_values[i] + sizeof(prefix) - 1;
    if (std::strcmp(value, "process") == 0) {
      GetCtsDeviceCache().SetNativeLifetime(AMDF_NATIVE_LIFETIME_PROCESS);
    } else if (std::strcmp(value, "instance") == 0) {
      GetCtsDeviceCache().SetNativeLifetime(AMDF_NATIVE_LIFETIME_INSTANCE);
    } else {
      std::fprintf(stderr, "invalid native lifetime: %s\n", value);
      return EXIT_FAILURE;
    }
    for (int j = i; j + 1 < argument_count; ++j) {
      argument_values[j] = argument_values[j + 1];
    }
    argument_values[--argument_count] = nullptr;
    --i;
  }
  if (!amdf_cts_provider_initialize(&argument_count, &argument_values)) {
    return EXIT_FAILURE;
  }
  testing::InitGoogleTest(&argument_count, argument_values);
  const int result = RUN_ALL_TESTS();
  const amdf_status_t cleanup_status = GetCtsDeviceCache().Deinitialize();
  if (!amdf_status_is_ok(cleanup_status)) {
    std::fprintf(stderr, "CTS device cleanup failed: domain=%u code=%u\n",
                 amdf_status_domain(cleanup_status),
                 amdf_status_code(cleanup_status));
    // Failed native cleanup retains children and their provider code.
    return EXIT_FAILURE;
  }
  const int deinitialize_succeeded = amdf_cts_provider_deinitialize();
  return result == EXIT_SUCCESS && deinitialize_succeeded ? EXIT_SUCCESS
                                                          : EXIT_FAILURE;
}
