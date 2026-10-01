// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

// Matches the two opaque 64-bit words recorded in GPU qualification results.
bool ParseEndpointId(const char* value, amdf_endpoint_id_t* out_id) {
  if (std::strlen(value) != 33 || value[16] != ':') {
    return false;
  }
  amdf_endpoint_id_t id = {};
  for (size_t i = 0; i < 2; ++i) {
    const char* begin = value + i * 17;
    const auto result = std::from_chars(begin, begin + 16, id.words[i], 16);
    if (result.ec != std::errc{} || result.ptr != begin + 16) {
      return false;
    }
  }
  *out_id = id;
  return true;
}

// Required names are independent of discovery, filtering, and runtime skips.
// A qualification invocation cannot pass by silently omitting its witness.
bool CheckRequiredTests(const std::vector<std::string>& required_tests) {
  const auto* unit_test = testing::UnitTest::GetInstance();
  bool all_passed = true;
  for (const std::string& required : required_tests) {
    const testing::TestInfo* match = nullptr;
    for (int i = 0; i < unit_test->total_test_suite_count(); ++i) {
      const auto* suite = unit_test->GetTestSuite(i);
      for (int j = 0; j < suite->total_test_count(); ++j) {
        const auto* test = suite->GetTestInfo(j);
        if (required == std::string(suite->name()) + "." + test->name()) {
          match = test;
        }
      }
    }
    if (match == nullptr || !match->should_run() ||
        match->result()->start_timestamp() == 0 || match->result()->Skipped() ||
        !match->result()->Passed()) {
      std::fprintf(stderr, "required CTS case did not pass: %s\n",
                   required.c_str());
      all_passed = false;
    }
  }
  return all_passed;
}

}  // namespace

int main(int argument_count, char** argument_values) {
  const char prefix[] = "--amdf_native_lifetime=";
  const char required_prefix[] = "--amdf_require_test=";
  const char gpu_target_prefix[] = "--amdf_gpu_target=";
  const char gpu_endpoint_prefix[] = "--amdf_gpu_endpoint_id=";
  const char gpu_peer_endpoint_prefix[] = "--amdf_gpu_peer_endpoint_id=";
  std::vector<std::string> required_tests;
  for (int i = 1; i < argument_count; ++i) {
    if (std::strncmp(argument_values[i], prefix, sizeof(prefix) - 1) == 0) {
      const char* value = argument_values[i] + sizeof(prefix) - 1;
      if (std::strcmp(value, "process") == 0) {
        GetCtsDeviceCache().SetNativeLifetime(AMDF_NATIVE_LIFETIME_PROCESS);
      } else if (std::strcmp(value, "instance") == 0) {
        GetCtsDeviceCache().SetNativeLifetime(AMDF_NATIVE_LIFETIME_INSTANCE);
      } else {
        std::fprintf(stderr, "invalid native lifetime: %s\n", value);
        return EXIT_FAILURE;
      }
    } else if (std::strncmp(argument_values[i], required_prefix,
                            sizeof(required_prefix) - 1) == 0) {
      const char* name = argument_values[i] + sizeof(required_prefix) - 1;
      if (name[0] == 0) {
        std::fprintf(stderr, "--amdf_require_test needs a full test name\n");
        return EXIT_FAILURE;
      }
      required_tests.emplace_back(name);
    } else if (std::strncmp(argument_values[i], gpu_endpoint_prefix,
                            sizeof(gpu_endpoint_prefix) - 1) == 0) {
      const char* value = argument_values[i] + sizeof(gpu_endpoint_prefix) - 1;
      amdf_endpoint_id_t id = {};
      if (!ParseEndpointId(value, &id)) {
        std::fprintf(stderr,
                     "--amdf_gpu_endpoint_id needs two 16-digit hexadecimal "
                     "words separated by ':'\n");
        return EXIT_FAILURE;
      }
      if (GetCtsDeviceCache().gpu_endpoint_id().has_value()) {
        std::fprintf(stderr, "--amdf_gpu_endpoint_id was specified twice\n");
        return EXIT_FAILURE;
      }
      GetCtsDeviceCache().SetGpuEndpointId(id);
    } else if (std::strncmp(argument_values[i], gpu_peer_endpoint_prefix,
                            sizeof(gpu_peer_endpoint_prefix) - 1) == 0) {
      const char* value =
          argument_values[i] + sizeof(gpu_peer_endpoint_prefix) - 1;
      amdf_endpoint_id_t id = {};
      if (!ParseEndpointId(value, &id)) {
        std::fprintf(stderr,
                     "--amdf_gpu_peer_endpoint_id needs two 16-digit "
                     "hexadecimal words separated by ':'\n");
        return EXIT_FAILURE;
      }
      if (GetCtsDeviceCache().gpu_peer_endpoint_id().has_value()) {
        std::fprintf(stderr,
                     "--amdf_gpu_peer_endpoint_id was specified twice\n");
        return EXIT_FAILURE;
      }
      GetCtsDeviceCache().SetGpuPeerEndpointId(id);
    } else if (std::strncmp(argument_values[i], gpu_target_prefix,
                            sizeof(gpu_target_prefix) - 1) == 0) {
      const char* target = argument_values[i] + sizeof(gpu_target_prefix) - 1;
      if (target[0] == 0) {
        std::fprintf(stderr,
                     "--amdf_gpu_target needs a canonical gfx target\n");
        return EXIT_FAILURE;
      }
      GetCtsDeviceCache().SetGpuTarget(target);
    } else {
      continue;
    }
    for (int j = i; j + 1 < argument_count; ++j) {
      argument_values[j] = argument_values[j + 1];
    }
    argument_values[--argument_count] = nullptr;
    --i;
  }
  const auto& primary = GetCtsDeviceCache().gpu_endpoint_id();
  const auto& peer = GetCtsDeviceCache().gpu_peer_endpoint_id();
  if (peer.has_value() && !primary.has_value()) {
    std::fprintf(
        stderr,
        "--amdf_gpu_peer_endpoint_id requires --amdf_gpu_endpoint_id\n");
    return EXIT_FAILURE;
  }
  if (peer.has_value() && amdf_endpoint_id_is_equal(&*primary, &*peer)) {
    std::fprintf(stderr, "primary and peer GPU endpoint IDs must differ\n");
    return EXIT_FAILURE;
  }
  if (!amdf_cts_provider_initialize(&argument_count, &argument_values)) {
    return EXIT_FAILURE;
  }
  testing::InitGoogleTest(&argument_count, argument_values);
  const int result = RUN_ALL_TESTS();
  const bool required_tests_passed = CheckRequiredTests(required_tests);
  const amdf_status_t cleanup_status = GetCtsDeviceCache().Deinitialize();
  if (!amdf_status_is_ok(cleanup_status)) {
    std::fprintf(stderr, "CTS device cleanup failed: domain=%u code=%u\n",
                 amdf_status_domain(cleanup_status),
                 amdf_status_code(cleanup_status));
    // Failed native cleanup retains children and their provider code.
    return EXIT_FAILURE;
  }
  const int deinitialize_succeeded = amdf_cts_provider_deinitialize();
  return result == EXIT_SUCCESS && required_tests_passed &&
                 deinitialize_succeeded
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
