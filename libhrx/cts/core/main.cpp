// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Custom Catch2 main for HRX CTS. Owns one explicitly selected device group.

#include <atomic>
#include <catch2/catch_session.hpp>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#include "hrx_loader.hpp"
#include "hrx_test_fixture.hpp"

hrx_device_t g_test_device = nullptr;
hrx_accelerator_type_t g_test_device_type = HRX_ACCELERATOR_CPU;
std::string g_test_hip_library_path;

namespace {

// Includes cleanup failures that occur outside Catch's assertion boundary.
std::atomic<bool> g_test_failure = false;

void printStatusAndConsume(HrxLoader& loader, const char* context,
                           hrx_status_t status) {
  char* message = nullptr;
  size_t message_length = 0;
  hrx_status_t formatting_status =
      loader.status_to_string(status, &message, &message_length);
  if (hrx_status_is_ok(formatting_status)) {
    fprintf(stderr, "%s: %s\n", context, message);
  } else {
    fprintf(stderr, "%s: status %d (formatting failed with status %d)\n",
            context, static_cast<int>(loader.status_code(status)),
            static_cast<int>(loader.status_code(formatting_status)));
    loader.status_ignore(formatting_status);
  }
  loader.status_free_message(message);
  loader.status_ignore(status);
}

hrx_status_t createTestGroup(HrxLoader& loader, const std::string& spec,
                             hrx_device_group_t* out_group) {
  *out_group = nullptr;
  const hrx_runtime_options_t options = {
      .event_sink =
          {
              .fn = ReportHrxDeviceEvent,
          },
  };
  hrx_runtime_t runtime = nullptr;
  hrx_status_t status =
      loader.runtime_create(&options, loader.host_allocator_system(), &runtime);
  hrx_device_info_t* infos = nullptr;
  const char* uri = spec.c_str();
  // These are CTS selection shorthands, not native HRX device inventories.
  if (hrx_status_is_ok(status) &&
      (spec.starts_with("gpu:") || spec.starts_with("cpu:"))) {
    size_t ordinal = 0;
    const auto parsed =
        std::from_chars(spec.data() + 4, spec.data() + spec.size(), ordinal);
    if (parsed.ec != std::errc{} || parsed.ptr != spec.data() + spec.size()) {
      status = loader.make_status(HRX_STATUS_INVALID_ARGUMENT,
                                  "invalid device ordinal");
    } else {
      const char* gpu_driver = std::getenv("HRX_GPU_DRIVER");
      const char* scheme =
          spec.starts_with("cpu:")
              ? "task"
              : (gpu_driver && gpu_driver[0] ? gpu_driver : "amdgpu");
      size_t count = 0;
      status = loader.runtime_query_available_devices(
          runtime, scheme, loader.host_allocator_system(), &count, &infos);
      if (hrx_status_is_ok(status)) {
        if (ordinal >= count) {
          status = loader.make_status(HRX_STATUS_OUT_OF_RANGE,
                                      "device ordinal is not available");
        } else {
          uri = infos[ordinal].uri;
        }
      }
    }
  }
  if (hrx_status_is_ok(status)) {
    const hrx_device_create_params_t params = {
        .uri = uri,
        .completion_wait_flags = HRX_WAIT_FLAG_YIELD,
    };
    status = loader.device_group_create(runtime, 1, &params, out_group);
  }
  loader.host_allocator_free(loader.host_allocator_system(), infos);
  loader.runtime_release(runtime);
  return status;
}

}  // namespace

// Driver failures reported while releasing devices must affect process exit,
// including events delivered after the Catch session has finished.
void ReportHrxDeviceEvent(void*, const hrx_device_event_t* event) noexcept {
  if (event->severity < HRX_DEVICE_EVENT_SEVERITY_ERROR) {
    return;
  }
  fprintf(stderr,
          "Device event: type %u, severity %u, device %.*s, sequence %llu\n",
          event->type, event->severity,
          static_cast<int>(event->source.device_id.size),
          event->source.device_id.size ? event->source.device_id.data : "",
          static_cast<unsigned long long>(event->sequence));
  g_test_failure.store(true, std::memory_order_relaxed);
}

void ReportHrxTestFailure(const char* context, hrx_status_t status) noexcept {
  if (hrx_status_is_ok(status)) {
    return;
  }
  printStatusAndConsume(hrx(), context, status);
  g_test_failure.store(true, std::memory_order_relaxed);
}

int main(int argc, char* argv[]) {
  Catch::Session session;

  std::string hrx_library;
  std::string hip_library;
  const char* hrx_device_env = std::getenv("HRX_CTS_DEVICE");
  std::string hrx_device_spec =
      hrx_device_env && hrx_device_env[0] ? hrx_device_env : "cpu:0";

  // Parse custom args.
  auto cli = session.cli() |
             Catch::Clara::Opt(hrx_library,
                               "path")["--hrx-library"]("Path to libhrx.so") |
             Catch::Clara::Opt(hip_library, "path")["--hip-library"](
                 "Path to libamdhip64.so") |
             Catch::Clara::Opt(hrx_device_spec, "spec")["--hrx-device"](
                 "HAL URI, gpu:N, cpu:N, or none");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) {
    return ret;
  }
  g_test_hip_library_path = hip_library;

  // Load library.
  if (!hrx_library.empty()) {
    HrxLoader::setLibraryPath(hrx_library);
  }

  hrx_device_group_t group = nullptr;
  try {
    auto& loader = HrxLoader::instance();

    // Print version.
    int major, minor, patch;
    loader.runtime_version(&major, &minor, &patch);
    printf("HRX CTS using libhrx v%d.%d.%d\n", major, minor, patch);

    if (hrx_device_spec == "none") {
      ret = session.run();
      return ret ? ret : static_cast<int>(g_test_failure.load());
    }

    hrx_status_t status = createTestGroup(loader, hrx_device_spec, &group);
    if (!hrx_status_is_ok(status)) {
      printStatusAndConsume(loader, "Failed to create test device group",
                            status);
      return 1;
    }
    g_test_device = loader.device_group_get_device(group, 0);
    ReportHrxTestFailure(
        "Device type query",
        loader.device_get_type(g_test_device, &g_test_device_type));

    char name[128] = {0};
    ReportHrxTestFailure(
        "Device name query",
        loader.device_get_property(g_test_device, HRX_DEVICE_PROPERTY_NAME,
                                   name, sizeof(name)));
    printf("Test device: %s (%s)\n", hrx_device_spec.c_str(), name);

  } catch (const HrxLoaderError& e) {
    fprintf(stderr, "Loader error: %s\n", e.what());
    return 1;
  }

  ret = session.run();

  // Release execution resources before reporting the final result. Runtime
  // events and fixture cleanup failures remain observable after session.run().
  g_test_device = nullptr;
  hrx().device_group_release(group);

  return ret ? ret : static_cast<int>(g_test_failure.load());
}
