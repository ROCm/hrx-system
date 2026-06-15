// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Custom Catch2 main for hrx CTS. Initializes hrx and selects a device.

#include <catch2/catch_session.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "hrx_loader.hpp"
#include "hrx_test_fixture.hpp"

hrx_device_t g_test_device = nullptr;
hrx_accelerator_type_t g_test_device_type = HRX_ACCELERATOR_CPU;

namespace {

void printStatusAndConsume(HrxLoader& loader, const char* context,
                           hrx_status_t status) {
  char* message = nullptr;
  size_t message_length = 0;
  loader.status_to_string(status, &message, &message_length);
  fprintf(stderr, "%s: %s\n", context, message ? message : "(unknown error)");
  loader.status_free_message(message);
  loader.status_ignore(status);
}

void seedInstalledCtsSourceDir(const char* argv0) {
  if (std::getenv("HRX_CTS_SOURCE_DIR") || !argv0) {
    return;
  }

  std::filesystem::path executable_dir =
      std::filesystem::path(argv0).parent_path();
  if (executable_dir.empty()) {
    return;
  }

  std::error_code ec;
  if (std::filesystem::is_directory(executable_dir / "testdata", ec)) {
    std::string source_dir = executable_dir.string();
    setenv("HRX_CTS_SOURCE_DIR", source_dir.c_str(), /*overwrite=*/0);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Catch::Session session;

  std::string hrx_library;
  const char* hrx_device_env = std::getenv("HRX_CTS_DEVICE");
  std::string hrx_device_spec =
      hrx_device_env && hrx_device_env[0] ? hrx_device_env : "cpu:0";

  // Parse custom args.
  auto cli = session.cli() |
             Catch::Clara::Opt(hrx_library,
                               "path")["--hrx-library"]("Path to libhrx.so") |
             Catch::Clara::Opt(hrx_device_spec, "spec")["--hrx-device"](
                 "Device spec (gpu:N or cpu:N)");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) return ret;

  seedInstalledCtsSourceDir(argc > 0 ? argv[0] : nullptr);

  // Load library.
  if (!hrx_library.empty()) {
    HrxLoader::setLibraryPath(hrx_library);
  }

  try {
    auto& loader = HrxLoader::instance();

    // Print version.
    int major, minor, patch;
    loader.runtime_version(&major, &minor, &patch);
    printf("HRX CTS using libhrx v%d.%d.%d\n", major, minor, patch);

    // Parse device spec.
    hrx_accelerator_type_t type = HRX_ACCELERATOR_CPU;
    int index = 0;
    if (hrx_device_spec.substr(0, 4) == "gpu:") {
      type = HRX_ACCELERATOR_GPU;
      index = std::atoi(hrx_device_spec.c_str() + 4);
    } else if (hrx_device_spec.substr(0, 4) == "cpu:") {
      type = HRX_ACCELERATOR_CPU;
      index = std::atoi(hrx_device_spec.c_str() + 4);
    }

    // Initialize accelerator.
    hrx_status_t status;
    if (type == HRX_ACCELERATOR_GPU) {
      status = loader.gpu_initialize(0);
      if (!hrx_status_is_ok(status)) {
        printStatusAndConsume(loader, "Failed to initialize GPU accelerator",
                              status);
        return 1;
      }
      status = loader.gpu_device_get(index, &g_test_device);
    } else {
      status = loader.cpu_initialize(0);
      if (!hrx_status_is_ok(status)) {
        printStatusAndConsume(loader, "Failed to initialize CPU accelerator",
                              status);
        return 1;
      }
      status = loader.cpu_device_get(index, &g_test_device);
    }
    if (!hrx_status_is_ok(status)) {
      std::string context = "Failed to get device " + hrx_device_spec;
      printStatusAndConsume(loader, context.c_str(), status);
      return 1;
    }
    g_test_device_type = type;

    char name[128] = {0};
    loader.device_get_property(g_test_device, HRX_DEVICE_PROPERTY_NAME, name,
                               sizeof(name));
    printf("Test device: %s (%s)\n", hrx_device_spec.c_str(), name);

  } catch (const HrxLoaderError& e) {
    fprintf(stderr, "Loader error: %s\n", e.what());
    return 1;
  }

  ret = session.run();

  // Shutdown.
  if (g_test_device_type == HRX_ACCELERATOR_GPU) {
    hrx().status_ignore(hrx().gpu_shutdown());
  } else {
    hrx().status_ignore(hrx().cpu_shutdown());
  }

  return ret;
}
