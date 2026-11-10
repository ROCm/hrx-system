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
#include "libamdf/cts/util/native_identity.h"
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

bool ResolveRunnerGpuSelection() {
  // Native runner allocation is independent of the configured simulator fabric.
  if (GetCtsDeviceCache().has_rocjitsu_configurations()) {
    return true;
  }
  const char* value = std::getenv("AMDF_CTS_GPU_NATIVE_IDENTITY");
  if (value == nullptr) {
    return true;
  }
  amdf_endpoint_native_identity_t identity = {};
  if (!amdf::cts::ParseNativeIdentity(value, &identity)) {
    std::fprintf(stderr, "invalid AMDF_CTS_GPU_NATIVE_IDENTITY: %s\n", value);
    return false;
  }
  const amdf_status_t status = GetCtsDeviceCache().ResolveGpuEndpoint(identity);
  if (!amdf_status_is_ok(status)) {
    std::fprintf(stderr, "cannot select runner GPU %s: domain=%u code=%u\n",
                 value, amdf_status_domain(status), amdf_status_code(status));
    return false;
  }
  return true;
}

bool ValidatePeerSelection() {
  const auto& primary = GetCtsDeviceCache().gpu_endpoint_id();
  const auto& peer = GetCtsDeviceCache().gpu_peer_endpoint_id();
  if (peer.has_value() && !primary.has_value()) {
    std::fprintf(
        stderr,
        "--amdf_gpu_peer_endpoint_id requires a primary GPU selection\n");
    return false;
  }
  if (peer.has_value() && amdf_endpoint_id_is_equal(&*primary, &*peer)) {
    std::fprintf(stderr, "primary and peer GPU endpoint IDs must differ\n");
    return false;
  }
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
  const char rocjitsu_library_prefix[] = "--amdf_rocjitsu_library=";
  const char rocjitsu_configuration_prefix[] = "--amdf_rocjitsu_config=";
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
      GetCtsDeviceCache().AddGpuTarget(target);
    } else if (std::strncmp(argument_values[i], rocjitsu_library_prefix,
                            sizeof(rocjitsu_library_prefix) - 1) == 0) {
      const char* path =
          argument_values[i] + sizeof(rocjitsu_library_prefix) - 1;
      if (path[0] == 0 || GetCtsDeviceCache().has_rocjitsu_library()) {
        std::fprintf(
            stderr,
            "--amdf_rocjitsu_library requires one explicit library path\n");
        return EXIT_FAILURE;
      }
      GetCtsDeviceCache().SetRocjitsuLibrary(path);
    } else if (std::strncmp(argument_values[i], rocjitsu_configuration_prefix,
                            sizeof(rocjitsu_configuration_prefix) - 1) == 0) {
      const char* path =
          argument_values[i] + sizeof(rocjitsu_configuration_prefix) - 1;
      if (path[0] == 0) {
        std::fprintf(stderr,
                     "--amdf_rocjitsu_config requires a configuration path\n");
        return EXIT_FAILURE;
      }
      GetCtsDeviceCache().AddRocjitsuConfiguration(path);
    } else {
      continue;
    }
    for (int j = i; j + 1 < argument_count; ++j) {
      argument_values[j] = argument_values[j + 1];
    }
    argument_values[--argument_count] = nullptr;
    --i;
  }
  if (GetCtsDeviceCache().has_rocjitsu_library() &&
      !GetCtsDeviceCache().has_rocjitsu_configurations()) {
    std::fprintf(stderr,
                 "--amdf_rocjitsu_library requires --amdf_rocjitsu_config\n");
    return EXIT_FAILURE;
  }
  if (!amdf_cts_provider_initialize(&argument_count, &argument_values)) {
    return EXIT_FAILURE;
  }
  testing::InitGoogleTest(&argument_count, argument_values);
  // Cases share native devices. A failed case may retain an unretired queue,
  // so later cases must not add more work to the same native owner.
  GTEST_FLAG_SET(fail_fast, true);
  const int result = ResolveRunnerGpuSelection() && ValidatePeerSelection()
                         ? RUN_ALL_TESTS()
                         : EXIT_FAILURE;
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
