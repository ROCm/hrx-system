// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// pyre-info: CLI tool for device enumeration and smoke testing.
// Dual-links libpyre.so (public API) + IREE static libs (for CLI flags).
// This validates the build architecture: two copies of IREE in one process
// (hidden inside libpyre.so and static in this binary) must not conflict.

#include "pyre_runtime.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// IREE flags for CLI parsing (linked statically into this binary).
#include "iree/base/tooling/flags.h"

#define CHECK_STATUS(expr)                              \
  do {                                                  \
    pyre_status_t _s = (expr);                          \
    if (!pyre_status_is_ok(_s)) {                       \
      char* msg = NULL;                                 \
      size_t len = 0;                                   \
      pyre_status_to_string(_s, &msg, &len);            \
      fprintf(stderr, "ERROR: %s\n", msg ? msg : "?");  \
      pyre_status_free_message(msg);                    \
      pyre_status_ignore(_s);                           \
      return 1;                                         \
    }                                                   \
  } while (0)

static int print_devices(void) {
  int major, minor, patch;
  pyre_runtime_version(&major, &minor, &patch);
  printf("Pyre Runtime v%d.%d.%d\n", major, minor, patch);

  // Try GPU.
  pyre_status_t gpu_status = pyre_gpu_initialize(0);
  if (pyre_status_is_ok(gpu_status)) {
    int gpu_count = 0;
    CHECK_STATUS(pyre_gpu_device_count(&gpu_count));
    printf("GPU accelerator: %d device%s\n", gpu_count,
           gpu_count != 1 ? "s" : "");
    for (int i = 0; i < gpu_count; i++) {
      pyre_device_t dev = NULL;
      CHECK_STATUS(pyre_gpu_device_get(i, &dev));
      char name[128] = {0};
      CHECK_STATUS(pyre_device_get_property(
          dev, PYRE_DEVICE_PROPERTY_NAME, name, sizeof(name)));
      char arch[64] = {0};
      CHECK_STATUS(pyre_device_get_property(
          dev, PYRE_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
      printf("  [%d] %s (%s)\n", i, name, arch);
    }
  } else {
    printf("GPU accelerator: unavailable\n");
    pyre_status_ignore(gpu_status);
  }

  // Try CPU.
  pyre_status_t cpu_status = pyre_cpu_initialize(0);
  if (pyre_status_is_ok(cpu_status)) {
    int cpu_count = 0;
    CHECK_STATUS(pyre_cpu_device_count(&cpu_count));
    printf("CPU accelerator: %d device%s\n", cpu_count,
           cpu_count != 1 ? "s" : "");
    for (int i = 0; i < cpu_count; i++) {
      pyre_device_t dev = NULL;
      CHECK_STATUS(pyre_cpu_device_get(i, &dev));
      char name[128] = {0};
      CHECK_STATUS(pyre_device_get_property(
          dev, PYRE_DEVICE_PROPERTY_NAME, name, sizeof(name)));
      printf("  [%d] %s\n", i, name);
    }
  } else {
    printf("CPU accelerator: unavailable\n");
    pyre_status_ignore(cpu_status);
  }

  return 0;
}

static int run_device_smoke_test(pyre_device_t device, const char* label) {
  printf("Opening %s...\n", label);

  // Create a stream.
  pyre_stream_t stream = NULL;
  CHECK_STATUS(pyre_stream_create(device, 0, &stream));

  // Allocate a 1 MiB host-visible buffer.
  const size_t size = 1024 * 1024;
  pyre_buffer_t buffer = NULL;
  CHECK_STATUS(
      pyre_buffer_allocate(stream, size, PYRE_MEMORY_HOST_LOCAL, &buffer));
  printf("  Allocate 1 MiB buffer: OK\n");

  // Fill with pattern.
  uint32_t pattern = 0xDEADBEEF;
  CHECK_STATUS(
      pyre_stream_fill_buffer(stream, buffer, 0, size, &pattern, sizeof(pattern)));
  // Flush fill before copy — the heap allocator's task CB doesn't guarantee
  // intra-CB ordering for transfer ops on the same buffer.
  CHECK_STATUS(pyre_stream_flush(stream));
  CHECK_STATUS(pyre_stream_synchronize(stream));
  printf("  Fill buffer: OK\n");

  // Allocate a second buffer and copy.
  pyre_buffer_t buffer2 = NULL;
  CHECK_STATUS(
      pyre_buffer_allocate(stream, size, PYRE_MEMORY_HOST_LOCAL, &buffer2));
  CHECK_STATUS(pyre_stream_copy_buffer(stream, buffer, 0, buffer2, 0, size));
  CHECK_STATUS(pyre_stream_flush(stream));
  CHECK_STATUS(pyre_stream_synchronize(stream));
  printf("  Copy buffer: OK\n");

  // Verify data.
  void* mapped = NULL;
  CHECK_STATUS(pyre_buffer_map(buffer2, PYRE_MAP_READ, 0, size, &mapped));
  uint32_t* data = (uint32_t*)mapped;
  bool ok = true;
  for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
    if (data[i] != 0xDEADBEEF) {
      fprintf(stderr, "  VERIFY FAILED at offset %zu: got 0x%08X\n",
              i * sizeof(uint32_t), data[i]);
      ok = false;
      break;
    }
  }
  CHECK_STATUS(pyre_buffer_unmap(buffer2));
  if (ok) printf("  Verify data: OK\n");

  // Release.
  CHECK_STATUS(pyre_buffer_release(buffer2));
  CHECK_STATUS(pyre_buffer_release(buffer));
  CHECK_STATUS(pyre_stream_release(stream));
  printf("  Release: OK\n");

  return ok ? 0 : 1;
}

static int parse_device_spec(const char* spec,
                             pyre_accelerator_type_t* type, int* index) {
  if (strncmp(spec, "gpu:", 4) == 0) {
    *type = PYRE_ACCELERATOR_GPU;
    *index = atoi(spec + 4);
    return 0;
  } else if (strncmp(spec, "cpu:", 4) == 0) {
    *type = PYRE_ACCELERATOR_CPU;
    *index = atoi(spec + 4);
    return 0;
  }
  fprintf(stderr, "Invalid device spec: %s (expected gpu:N or cpu:N)\n", spec);
  return 1;
}

int main(int argc, char** argv) {
  const char* device_spec = NULL;
  bool test_all = false;

  // Parse our args first, then let IREE have the rest.
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--device=", 9) == 0) {
      device_spec = argv[i] + 9;
    } else if (strcmp(argv[i], "--test=all") == 0) {
      test_all = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: pyre-info [--device=gpu:0|cpu:0] [--test=all]\n");
      printf("  (no args)      Print version and device list\n");
      printf("  --device=DEV   Run smoke test on specific device\n");
      printf("  --test=all     Run smoke tests on all devices\n");
      return 0;
    }
  }

  // Parse IREE flags (validates dual-linkage — this uses IREE's static
  // flag registry linked directly into this binary). UNDEFINED_OK lets
  // our custom flags pass through without error.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);

  if (!device_spec && !test_all) {
    return print_devices();
  }

  // Initialize both accelerators.
  pyre_status_t gpu_status = pyre_gpu_initialize(0);
  pyre_status_t cpu_status = pyre_cpu_initialize(0);

  int result = 0;

  if (device_spec) {
    pyre_accelerator_type_t type;
    int index;
    if (parse_device_spec(device_spec, &type, &index) != 0) return 1;

    pyre_device_t dev = NULL;
    if (type == PYRE_ACCELERATOR_GPU) {
      if (!pyre_status_is_ok(gpu_status)) {
        fprintf(stderr, "GPU accelerator not available\n");
        pyre_status_ignore(gpu_status);
        pyre_status_ignore(cpu_status);
        return 1;
      }
      CHECK_STATUS(pyre_gpu_device_get(index, &dev));
    } else {
      if (!pyre_status_is_ok(cpu_status)) {
        fprintf(stderr, "CPU accelerator not available\n");
        pyre_status_ignore(gpu_status);
        pyre_status_ignore(cpu_status);
        return 1;
      }
      CHECK_STATUS(pyre_cpu_device_get(index, &dev));
    }
    result = run_device_smoke_test(dev, device_spec);
  } else if (test_all) {
    // Test all available devices.
    if (pyre_status_is_ok(gpu_status)) {
      int count = 0;
      CHECK_STATUS(pyre_gpu_device_count(&count));
      for (int i = 0; i < count; i++) {
        pyre_device_t dev = NULL;
        CHECK_STATUS(pyre_gpu_device_get(i, &dev));
        char label[32];
        snprintf(label, sizeof(label), "gpu:%d", i);
        int r = run_device_smoke_test(dev, label);
        if (r != 0) result = r;
      }
    } else {
      pyre_status_ignore(gpu_status);
      gpu_status = pyre_ok_status();
    }

    if (pyre_status_is_ok(cpu_status)) {
      int count = 0;
      CHECK_STATUS(pyre_cpu_device_count(&count));
      for (int i = 0; i < count; i++) {
        pyre_device_t dev = NULL;
        CHECK_STATUS(pyre_cpu_device_get(i, &dev));
        char label[32];
        snprintf(label, sizeof(label), "cpu:%d", i);
        int r = run_device_smoke_test(dev, label);
        if (r != 0) result = r;
      }
    } else {
      pyre_status_ignore(cpu_status);
      cpu_status = pyre_ok_status();
    }
  }

  // Shutdown.
  if (pyre_status_is_ok(gpu_status)) {
    pyre_status_ignore(pyre_gpu_shutdown());
  }
  if (pyre_status_is_ok(cpu_status)) {
    pyre_status_ignore(pyre_cpu_shutdown());
  }

  if (result == 0) {
    printf("\nAll tests PASSED\n");
  } else {
    printf("\nSome tests FAILED\n");
  }
  return result;
}
