// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Custom Catch2 main for hrx CTS. Initializes hrx and selects a device.

#include "hrx_loader.hpp"
#include "hrx_test_fixture.hpp"

#include <catch2/catch_session.hpp>

#include <cstdio>
#include <cstring>

hrx_device_t g_test_device = nullptr;
hrx_accelerator_type_t g_test_device_type = HRX_ACCELERATOR_CPU;

int main(int argc, char *argv[]) {
  Catch::Session session;

  std::string hrx_library;
  std::string hrx_device_spec = "cpu:0";

  // Parse custom args.
  auto cli = session.cli() |
             Catch::Clara::Opt(hrx_library,
                               "path")["--hrx-library"]("Path to libhrx.so") |
             Catch::Clara::Opt(hrx_device_spec, "spec")["--hrx-device"](
                 "Device spec (gpu:N or cpu:N)");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0)
    return ret;

  // Load library.
  if (!hrx_library.empty()) {
    HrxLoader::setLibraryPath(hrx_library);
  }

  try {
    auto &loader = HrxLoader::instance();

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
        fprintf(stderr, "Failed to initialize GPU accelerator\n");
        loader.status_ignore(status);
        return 1;
      }
      status = loader.gpu_device_get(index, &g_test_device);
    } else {
      status = loader.cpu_initialize(0);
      if (!hrx_status_is_ok(status)) {
        fprintf(stderr, "Failed to initialize CPU accelerator\n");
        loader.status_ignore(status);
        return 1;
      }
      status = loader.cpu_device_get(index, &g_test_device);
    }
    if (!hrx_status_is_ok(status)) {
      fprintf(stderr, "Failed to get device %s\n", hrx_device_spec.c_str());
      loader.status_ignore(status);
      return 1;
    }
    g_test_device_type = type;

    char name[128] = {0};
    loader.device_get_property(g_test_device, HRX_DEVICE_PROPERTY_NAME, name,
                               sizeof(name));
    printf("Test device: %s (%s)\n", hrx_device_spec.c_str(), name);

  } catch (const HrxLoaderError &e) {
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
