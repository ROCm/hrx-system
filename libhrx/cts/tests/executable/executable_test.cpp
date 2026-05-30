// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

std::string gpu_architecture(hrx_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch.size()));
  std::string value(arch.data());
  for (char c : value) {
    REQUIRE((std::isalnum(static_cast<unsigned char>(c)) || c == '_'));
  }
  return value;
}

void append_candidate(std::vector<std::filesystem::path> &candidates,
                      const char *root,
                      std::initializer_list<const char *> fragments) {
  if (!root || !root[0]) {
    return;
  }
  std::filesystem::path path(root);
  for (const char *fragment : fragments) {
    path /= fragment;
  }
  candidates.push_back(path);
}

std::filesystem::path find_noop_hsaco() {
  std::vector<std::filesystem::path> candidates;
  append_candidate(candidates, std::getenv("HRX_CTS_EXECUTABLE_HSACO"), {});
  append_candidate(candidates, std::getenv("TEST_SRCDIR"),
                   {"libhrx", "cts", "hrx_cts_noop_kernel.so"});
  append_candidate(candidates, std::getenv("RUNFILES_DIR"),
                   {"libhrx", "cts", "hrx_cts_noop_kernel.so"});
  append_candidate(candidates, std::getenv("IREE_BINARY_DIR"),
                   {"libhrx", "cts", "hrx_cts_noop_kernel.so"});

  for (const auto &candidate : candidates) {
    if (std::filesystem::exists(candidate) &&
        std::filesystem::file_size(candidate) > 0) {
      return candidate;
    }
  }
  return {};
}

} // namespace

TEST_CASE_METHOD(HrxTestFixture, "executable_load_lookup_dispatch_noop") {
  if (!is_gpu()) {
    SUCCEED("Native executable dispatch is GPU-only");
    return;
  }

  std::string arch = gpu_architecture(device_);
  if (arch.empty() || arch == "unknown") {
    SUCCEED("Skipping native executable CTS: unknown GPU architecture");
    return;
  }

  std::filesystem::path hsaco_path = find_noop_hsaco();
  if (hsaco_path.empty()) {
    SUCCEED("Skipping native executable CTS: no build-time HSACO test asset");
    return;
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_file(device_, hsaco_path.c_str(), nullptr,
                                        &executable));
  REQUIRE(executable != nullptr);

  hrx().executable_retain(executable);
  hrx().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(hrx().executable_export_count(executable, &export_count));
  REQUIRE(export_count == 1);

  uint32_t ordinal = UINT32_MAX;
  REQUIRE_OK(
      hrx().executable_lookup_export_by_name(executable, "hrx_noop", &ordinal));
  REQUIRE(ordinal == 0);

  hrx_executable_export_info_t info = {};
  REQUIRE_OK(hrx().executable_export_info(executable, ordinal, &info));
  REQUIRE(info.name != nullptr);
  REQUIRE(std::string(info.name) == "hrx_noop");
  REQUIRE(info.constant_count == 0);
  REQUIRE(info.binding_count == 0);
  REQUIRE(info.workgroup_size[0] >= 1);
  REQUIRE(info.workgroup_size[1] >= 1);
  REQUIRE(info.workgroup_size[2] >= 1);

  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);

  hrx_dispatch_config_t config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          info.workgroup_size[0],
          info.workgroup_size[1],
          info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(stream, executable, ordinal, &config,
                                   /*constants=*/nullptr, /*constants_size=*/0,
                                   /*bindings=*/nullptr, /*binding_count=*/0,
                                   HRX_DISPATCH_FLAG_NONE));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  hrx().stream_release(stream);
  hrx().executable_release(executable);
}
