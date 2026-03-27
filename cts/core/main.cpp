// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Custom Catch2 main for pyre CTS. Initializes pyre and selects a device.

#include "pyre_loader.hpp"
#include "pyre_test_fixture.hpp"

#include <catch2/catch_session.hpp>

#include <cstdio>
#include <cstring>

pyre_device_t g_test_device = nullptr;
pyre_accelerator_type_t g_test_device_type = PYRE_ACCELERATOR_CPU;

int main(int argc, char* argv[]) {
  Catch::Session session;

  std::string pyre_library;
  std::string pyre_device_spec = "cpu:0";

  // Parse custom args.
  auto cli = session.cli()
      | Catch::Clara::Opt(pyre_library, "path")
            ["--pyre-library"]("Path to libpyre.so")
      | Catch::Clara::Opt(pyre_device_spec, "spec")
            ["--pyre-device"]("Device spec (gpu:N or cpu:N)");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) return ret;

  // Load library.
  if (!pyre_library.empty()) {
    PyreLoader::setLibraryPath(pyre_library);
  }

  try {
    auto& loader = PyreLoader::instance();

    // Print version.
    int major, minor, patch;
    loader.runtime_version(&major, &minor, &patch);
    printf("Pyre CTS using libpyre v%d.%d.%d\n", major, minor, patch);

    // Parse device spec.
    pyre_accelerator_type_t type = PYRE_ACCELERATOR_CPU;
    int index = 0;
    if (pyre_device_spec.substr(0, 4) == "gpu:") {
      type = PYRE_ACCELERATOR_GPU;
      index = std::atoi(pyre_device_spec.c_str() + 4);
    } else if (pyre_device_spec.substr(0, 4) == "cpu:") {
      type = PYRE_ACCELERATOR_CPU;
      index = std::atoi(pyre_device_spec.c_str() + 4);
    }

    // Initialize accelerator.
    pyre_status_t status;
    if (type == PYRE_ACCELERATOR_GPU) {
      status = loader.gpu_initialize(0);
      if (!pyre_status_is_ok(status)) {
        fprintf(stderr, "Failed to initialize GPU accelerator\n");
        loader.status_ignore(status);
        return 1;
      }
      status = loader.gpu_device_get(index, &g_test_device);
    } else {
      status = loader.cpu_initialize(0);
      if (!pyre_status_is_ok(status)) {
        fprintf(stderr, "Failed to initialize CPU accelerator\n");
        loader.status_ignore(status);
        return 1;
      }
      status = loader.cpu_device_get(index, &g_test_device);
    }
    if (!pyre_status_is_ok(status)) {
      fprintf(stderr, "Failed to get device %s\n", pyre_device_spec.c_str());
      loader.status_ignore(status);
      return 1;
    }
    g_test_device_type = type;

    char name[128] = {0};
    loader.device_get_property(g_test_device, PYRE_DEVICE_PROPERTY_NAME,
                               name, sizeof(name));
    printf("Test device: %s (%s)\n", pyre_device_spec.c_str(), name);

  } catch (const PyreLoaderError& e) {
    fprintf(stderr, "Loader error: %s\n", e.what());
    return 1;
  }

  ret = session.run();

  // Shutdown.
  if (g_test_device_type == PYRE_ACCELERATOR_GPU) {
    pyre().status_ignore(pyre().gpu_shutdown());
  } else {
    pyre().status_ignore(pyre().cpu_shutdown());
  }

  return ret;
}
