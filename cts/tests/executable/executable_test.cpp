// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string gpu_architecture(pyre_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(pyre().device_get_property(
      device, PYRE_DEVICE_PROPERTY_ARCHITECTURE, arch.data(), arch.size()));
  std::string value(arch.data());
  for (char c : value) {
    REQUIRE((std::isalnum(static_cast<unsigned char>(c)) || c == '_'));
  }
  return value;
}

std::filesystem::path write_noop_kernel_source() {
  std::filesystem::path src_path =
      std::filesystem::temp_directory_path() / "pyre_noop_kernel.hip";
  std::ofstream src(src_path);
  REQUIRE(src.good());
  src << "extern \"C\" __attribute__((global)) void pyre_noop() {}\n";
  REQUIRE(src.good());
  return src_path;
}

std::filesystem::path build_noop_hsaco(const std::string& arch) {
  std::filesystem::path src_path = write_noop_kernel_source();
  std::filesystem::path hsaco_path =
      std::filesystem::temp_directory_path() / "pyre_noop_kernel.hsaco";

  std::string command =
      "clang++ -x hip --offload-device-only --offload-arch=" + arch +
      " -nogpuinc -nogpulib -c " + src_path.string() +
      " -o " + hsaco_path.string();
  int rc = std::system(command.c_str());
  INFO("command: " << command);
  REQUIRE(rc == 0);
  REQUIRE(std::filesystem::exists(hsaco_path));
  REQUIRE(std::filesystem::file_size(hsaco_path) > 0);
  return hsaco_path;
}

}  // namespace

TEST_CASE_METHOD(PyreTestFixture, "executable_load_lookup_dispatch_noop") {
  if (!is_gpu()) {
    SUCCEED("Native executable dispatch is GPU-only");
    return;
  }

  std::string arch = gpu_architecture(device_);
  if (arch.empty() || arch == "unknown") {
    SUCCEED("Skipping native executable CTS: unknown GPU architecture");
    return;
  }

  std::filesystem::path hsaco_path = build_noop_hsaco(arch);

  pyre_executable_t executable = nullptr;
  REQUIRE_OK(pyre().executable_load_file(
      device_, hsaco_path.c_str(), nullptr, &executable));
  REQUIRE(executable != nullptr);

  pyre().executable_retain(executable);
  pyre().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(pyre().executable_export_count(executable, &export_count));
  REQUIRE(export_count == 1);

  uint32_t ordinal = UINT32_MAX;
  REQUIRE_OK(pyre().executable_lookup_export_by_name(
      executable, "pyre_noop", &ordinal));
  REQUIRE(ordinal == 0);

  pyre_executable_export_info_t info = {};
  REQUIRE_OK(pyre().executable_export_info(executable, ordinal, &info));
  REQUIRE(info.name != nullptr);
  REQUIRE(std::string(info.name) == "pyre_noop");
  REQUIRE(info.constant_count == 0);
  REQUIRE(info.binding_count == 0);
  REQUIRE(info.workgroup_size[0] >= 1);
  REQUIRE(info.workgroup_size[1] >= 1);
  REQUIRE(info.workgroup_size[2] >= 1);

  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);

  pyre_dispatch_config_t config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */ {
          info.workgroup_size[0],
          info.workgroup_size[1],
          info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(pyre().stream_dispatch(
      stream, executable, ordinal, &config,
      /*constants=*/nullptr, /*constants_size=*/0,
      /*bindings=*/nullptr, /*binding_count=*/0,
      PYRE_DISPATCH_FLAG_NONE));
  REQUIRE_OK(pyre().stream_synchronize(stream));

  pyre().stream_release(stream);
  pyre().executable_release(executable);
}
