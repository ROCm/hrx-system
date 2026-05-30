// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// hrx-info: CLI tool for device enumeration and smoke testing.
// Dual-links libhrx.so (public API) + IREE static libs (for CLI flags).
// This validates the build architecture: two copies of IREE in one process
// (hidden inside libhrx.so and static in this binary) must not conflict.

#include "hrx_runtime.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// IREE flags for CLI parsing (linked statically into this binary).
#include "iree/base/tooling/flags.h"

#define CHECK_STATUS(expr)                                                     \
  do {                                                                         \
    hrx_status_t _s = (expr);                                                  \
    if (!hrx_status_is_ok(_s)) {                                               \
      char *msg = NULL;                                                        \
      size_t len = 0;                                                          \
      hrx_status_to_string(_s, &msg, &len);                                    \
      fprintf(stderr, "ERROR: %s\n", msg ? msg : "?");                         \
      hrx_status_free_message(msg);                                            \
      hrx_status_ignore(_s);                                                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_STATUS_CLEANUP(expr)                                             \
  do {                                                                         \
    hrx_status_t _s = (expr);                                                  \
    if (!hrx_status_is_ok(_s)) {                                               \
      char *msg = NULL;                                                        \
      size_t len = 0;                                                          \
      hrx_status_to_string(_s, &msg, &len);                                    \
      fprintf(stderr, "ERROR: %s\n", msg ? msg : "?");                         \
      hrx_status_free_message(msg);                                            \
      hrx_status_ignore(_s);                                                   \
      result = 1;                                                              \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

static void print_status_message(FILE *stream, const char *prefix,
                                 hrx_status_t status) {
  char *msg = NULL;
  size_t len = 0;
  hrx_status_to_string(status, &msg, &len);
  fprintf(stream, "%s%s", prefix, msg ? msg : "?");
  hrx_status_free_message(msg);
}

static int print_devices(void) {
  int result = 0;
  bool gpu_initialized = false;
  bool cpu_initialized = false;
  hrx_status_t gpu_status = hrx_ok_status();
  hrx_status_t cpu_status = hrx_ok_status();
  int major, minor, patch;
  hrx_runtime_version(&major, &minor, &patch);
  printf("HRX (HIP Runtime Extended) v%d.%d.%d\n", major, minor, patch);

  // Try GPU.
  gpu_status = hrx_gpu_initialize(0);
  if (hrx_status_is_ok(gpu_status)) {
    gpu_initialized = true;
    int gpu_count = 0;
    CHECK_STATUS_CLEANUP(hrx_gpu_device_count(&gpu_count));
    printf("GPU accelerator: %d device%s\n", gpu_count,
           gpu_count != 1 ? "s" : "");
    for (int i = 0; i < gpu_count; i++) {
      hrx_device_t dev = NULL;
      CHECK_STATUS_CLEANUP(hrx_gpu_device_get(i, &dev));
      char name[128] = {0};
      CHECK_STATUS_CLEANUP(hrx_device_get_property(
          dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof(name)));
      char arch[64] = {0};
      CHECK_STATUS_CLEANUP(hrx_device_get_property(
          dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
      printf("  [%d] %s (%s)\n", i, name, arch);
    }
  } else {
    printf("GPU accelerator: unavailable (");
    print_status_message(stdout, "", gpu_status);
    printf(")\n");
    hrx_status_ignore(gpu_status);
  }

  // Try CPU.
  cpu_status = hrx_cpu_initialize(0);
  if (hrx_status_is_ok(cpu_status)) {
    cpu_initialized = true;
    int cpu_count = 0;
    CHECK_STATUS_CLEANUP(hrx_cpu_device_count(&cpu_count));
    printf("CPU accelerator: %d device%s\n", cpu_count,
           cpu_count != 1 ? "s" : "");
    for (int i = 0; i < cpu_count; i++) {
      hrx_device_t dev = NULL;
      CHECK_STATUS_CLEANUP(hrx_cpu_device_get(i, &dev));
      char name[128] = {0};
      CHECK_STATUS_CLEANUP(hrx_device_get_property(
          dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof(name)));
      printf("  [%d] %s\n", i, name);
    }
  } else {
    printf("CPU accelerator: unavailable (");
    print_status_message(stdout, "", cpu_status);
    printf(")\n");
    hrx_status_ignore(cpu_status);
  }

cleanup:
  if (gpu_initialized) {
    hrx_status_t shutdown_status = hrx_gpu_shutdown();
    if (!hrx_status_is_ok(shutdown_status)) {
      print_status_message(
          stderr, "GPU accelerator shutdown failed: ", shutdown_status);
      fprintf(stderr, "\n");
      hrx_status_ignore(shutdown_status);
      result = 1;
    }
  }
  if (cpu_initialized) {
    hrx_status_t shutdown_status = hrx_cpu_shutdown();
    if (!hrx_status_is_ok(shutdown_status)) {
      print_status_message(
          stderr, "CPU accelerator shutdown failed: ", shutdown_status);
      fprintf(stderr, "\n");
      hrx_status_ignore(shutdown_status);
      result = 1;
    }
  }
  return result;
}

static int run_device_smoke_test(hrx_device_t device, const char *label) {
  printf("Opening %s...\n", label);

  // Create a stream.
  hrx_stream_t stream = NULL;
  CHECK_STATUS(hrx_stream_create(device, 0, &stream));

  // Allocate a 1 MiB host-visible buffer.
  const size_t size = 1024 * 1024;
  hrx_buffer_t buffer = NULL;
  CHECK_STATUS(hrx_buffer_allocate(
      stream, size, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buffer));
  printf("  Allocate 1 MiB buffer: OK\n");

  // Fill with pattern.
  uint32_t pattern = 0xDEADBEEF;
  CHECK_STATUS(hrx_stream_fill_buffer(stream, buffer, 0, size, &pattern,
                                      sizeof(pattern)));
  // Flush fill before copy — the heap allocator's task CB doesn't guarantee
  // intra-CB ordering for transfer ops on the same buffer.
  CHECK_STATUS(hrx_stream_flush(stream));
  CHECK_STATUS(hrx_stream_synchronize(stream));
  printf("  Fill buffer: OK\n");

  // Allocate a second buffer and copy.
  hrx_buffer_t buffer2 = NULL;
  CHECK_STATUS(hrx_buffer_allocate(
      stream, size, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buffer2));
  CHECK_STATUS(hrx_stream_copy_buffer(stream, buffer, 0, buffer2, 0, size));
  CHECK_STATUS(hrx_stream_flush(stream));
  CHECK_STATUS(hrx_stream_synchronize(stream));
  printf("  Copy buffer: OK\n");

  // Verify data.
  void *mapped = NULL;
  CHECK_STATUS(hrx_buffer_map(buffer2, HRX_MAP_READ, 0, size, &mapped));
  uint32_t *data = (uint32_t *)mapped;
  bool ok = true;
  for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
    if (data[i] != 0xDEADBEEF) {
      fprintf(stderr, "  VERIFY FAILED at offset %zu: got 0x%08X\n",
              i * sizeof(uint32_t), data[i]);
      ok = false;
      break;
    }
  }
  CHECK_STATUS(hrx_buffer_unmap(buffer2));
  if (ok)
    printf("  Verify data: OK\n");

  // Release.
  hrx_buffer_release(buffer2);
  hrx_buffer_release(buffer);
  hrx_stream_release(stream);
  printf("  Release: OK\n");

  return ok ? 0 : 1;
}

static int parse_device_spec(const char *spec, hrx_accelerator_type_t *type,
                             int *index) {
  if (strncmp(spec, "gpu:", 4) == 0) {
    *type = HRX_ACCELERATOR_GPU;
    *index = atoi(spec + 4);
    return 0;
  } else if (strncmp(spec, "cpu:", 4) == 0) {
    *type = HRX_ACCELERATOR_CPU;
    *index = atoi(spec + 4);
    return 0;
  }
  fprintf(stderr, "Invalid device spec: %s (expected gpu:N or cpu:N)\n", spec);
  return 1;
}

int main(int argc, char **argv) {
  const char *device_spec = NULL;
  bool test_all = false;
  hrx_accelerator_type_t requested_type = HRX_ACCELERATOR_CPU;
  int requested_index = 0;

  // Parse our args first, then let IREE have the rest.
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--device=", 9) == 0) {
      device_spec = argv[i] + 9;
    } else if (strcmp(argv[i], "--test=all") == 0) {
      test_all = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: hrx-info [--device=gpu:0|cpu:0] [--test=all]\n");
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
  if (device_spec &&
      parse_device_spec(device_spec, &requested_type, &requested_index) != 0) {
    return 1;
  }

  const bool need_gpu =
      test_all || (device_spec && requested_type == HRX_ACCELERATOR_GPU);
  const bool need_cpu =
      test_all || (device_spec && requested_type == HRX_ACCELERATOR_CPU);
  hrx_status_t gpu_status =
      need_gpu ? hrx_gpu_initialize(0) : hrx_ok_status();
  hrx_status_t cpu_status =
      need_cpu ? hrx_cpu_initialize(0) : hrx_ok_status();
  bool gpu_initialized = need_gpu && hrx_status_is_ok(gpu_status);
  bool cpu_initialized = need_cpu && hrx_status_is_ok(cpu_status);

  int result = 0;

  if (device_spec) {
    hrx_device_t dev = NULL;
    if (requested_type == HRX_ACCELERATOR_GPU) {
      if (!hrx_status_is_ok(gpu_status)) {
        print_status_message(stderr,
                             "GPU accelerator not available: ", gpu_status);
        fprintf(stderr, "\n");
        result = 1;
        goto cleanup;
      }
      CHECK_STATUS_CLEANUP(hrx_gpu_device_get(requested_index, &dev));
    } else {
      if (!hrx_status_is_ok(cpu_status)) {
        print_status_message(stderr,
                             "CPU accelerator not available: ", cpu_status);
        fprintf(stderr, "\n");
        result = 1;
        goto cleanup;
      }
      CHECK_STATUS_CLEANUP(hrx_cpu_device_get(requested_index, &dev));
    }
    result = run_device_smoke_test(dev, device_spec);
  } else if (test_all) {
    // Test all available devices.
    if (hrx_status_is_ok(gpu_status)) {
      int count = 0;
      CHECK_STATUS_CLEANUP(hrx_gpu_device_count(&count));
      for (int i = 0; i < count; i++) {
        hrx_device_t dev = NULL;
        CHECK_STATUS_CLEANUP(hrx_gpu_device_get(i, &dev));
        char label[32];
        snprintf(label, sizeof(label), "gpu:%d", i);
        int r = run_device_smoke_test(dev, label);
        if (r != 0)
          result = r;
      }
    } else {
      hrx_status_ignore(gpu_status);
      gpu_status = hrx_ok_status();
    }

    if (hrx_status_is_ok(cpu_status)) {
      int count = 0;
      CHECK_STATUS_CLEANUP(hrx_cpu_device_count(&count));
      for (int i = 0; i < count; i++) {
        hrx_device_t dev = NULL;
        CHECK_STATUS_CLEANUP(hrx_cpu_device_get(i, &dev));
        char label[32];
        snprintf(label, sizeof(label), "cpu:%d", i);
        int r = run_device_smoke_test(dev, label);
        if (r != 0)
          result = r;
      }
    } else {
      hrx_status_ignore(cpu_status);
      cpu_status = hrx_ok_status();
    }
  }

  // Shutdown.
cleanup:
  if (gpu_initialized) {
    hrx_status_ignore(hrx_gpu_shutdown());
  } else {
    hrx_status_ignore(gpu_status);
  }
  if (cpu_initialized) {
    hrx_status_ignore(hrx_cpu_shutdown());
  } else {
    hrx_status_ignore(cpu_status);
  }

  if (result == 0) {
    printf("\nAll tests PASSED\n");
  } else {
    printf("\nSome tests FAILED\n");
  }
  return result;
}
