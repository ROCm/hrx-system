// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// hrx-info: CLI tool for device enumeration and smoke testing.
// Dual-links libhrx.so (public API) + IREE static libs (for CLI flags).
// This validates the build architecture: two copies of IREE in one process
// (hidden inside libhrx.so and static in this binary) must not conflict.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrx_runtime.h"

// IREE flags for CLI parsing (linked statically into this binary).
#include "iree/base/tooling/flags.h"

static void print_status_message(FILE* stream, const char* prefix,
                                 hrx_status_t status) {
  char* msg = NULL;
  size_t len = 0;
  hrx_status_to_string(status, &msg, &len);
  fprintf(stream, "%s%s", prefix, msg ? msg : "?");
  hrx_status_free_message(msg);
}

static int report_status_error(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return 0;
  print_status_message(stderr, "ERROR: ", status);
  fprintf(stderr, "\n");
  hrx_status_ignore(status);
  return 1;
}

static void consume_status(hrx_status_t* status) {
  hrx_status_ignore(*status);
  *status = hrx_ok_status();
}

static int report_shutdown_error(const char* accelerator_name,
                                 hrx_status_t status) {
  if (hrx_status_is_ok(status)) return 0;
  fprintf(stderr, "%s accelerator shutdown failed: ", accelerator_name);
  print_status_message(stderr, "", status);
  fprintf(stderr, "\n");
  hrx_status_ignore(status);
  return 1;
}

static int print_gpu_devices(void) {
  hrx_status_t status = hrx_gpu_initialize(0);
  if (!hrx_status_is_ok(status)) {
    printf("GPU accelerator: unavailable (");
    print_status_message(stdout, "", status);
    printf(")\n");
    hrx_status_ignore(status);
    return 0;
  }

  int result = 0;
  int gpu_count = 0;
  if (report_status_error(hrx_gpu_device_count(&gpu_count))) {
    result = 1;
  }
  if (result == 0) {
    printf("GPU accelerator: %d device%s\n", gpu_count,
           gpu_count != 1 ? "s" : "");
    for (int i = 0; i < gpu_count && result == 0; i++) {
      hrx_device_t dev = NULL;
      if (report_status_error(hrx_gpu_device_get(i, &dev))) {
        result = 1;
        continue;
      }
      char name[128] = {0};
      if (report_status_error(hrx_device_get_property(
              dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof(name)))) {
        result = 1;
        continue;
      }
      char arch[64] = {0};
      if (report_status_error(hrx_device_get_property(
              dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)))) {
        result = 1;
        continue;
      }
      printf("  [%d] %s (%s)\n", i, name, arch);
    }
  }
  if (report_shutdown_error("GPU", hrx_gpu_shutdown())) {
    result = 1;
  }
  return result;
}

static int print_cpu_devices(void) {
  hrx_status_t status = hrx_cpu_initialize(0);
  if (!hrx_status_is_ok(status)) {
    printf("CPU accelerator: unavailable (");
    print_status_message(stdout, "", status);
    printf(")\n");
    hrx_status_ignore(status);
    return 0;
  }

  int result = 0;
  int cpu_count = 0;
  if (report_status_error(hrx_cpu_device_count(&cpu_count))) {
    result = 1;
  }
  if (result == 0) {
    printf("CPU accelerator: %d device%s\n", cpu_count,
           cpu_count != 1 ? "s" : "");
    for (int i = 0; i < cpu_count && result == 0; i++) {
      hrx_device_t dev = NULL;
      if (report_status_error(hrx_cpu_device_get(i, &dev))) {
        result = 1;
        continue;
      }
      char name[128] = {0};
      if (report_status_error(hrx_device_get_property(
              dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof(name)))) {
        result = 1;
        continue;
      }
      printf("  [%d] %s\n", i, name);
    }
  }
  if (report_shutdown_error("CPU", hrx_cpu_shutdown())) {
    result = 1;
  }
  return result;
}

static int print_devices(void) {
  int major, minor, patch;
  hrx_runtime_version(&major, &minor, &patch);
  printf("HRX (HIP Runtime Extended) v%d.%d.%d\n", major, minor, patch);

  int result = print_gpu_devices();
  if (print_cpu_devices() != 0) result = 1;
  return result;
}

static int run_device_smoke_test(hrx_device_t device, const char* label) {
  printf("Opening %s...\n", label);

  hrx_stream_t stream = NULL;
  hrx_buffer_t buffer = NULL;
  hrx_buffer_t buffer2 = NULL;
  void* mapped = NULL;
  bool buffer_mapped = false;
  int result = 0;

  if (report_status_error(hrx_stream_create(device, 0, &stream))) result = 1;

  const size_t size = 1024 * 1024;
  if (result == 0 &&
      report_status_error(hrx_buffer_allocate(
          stream, size,
          HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
          HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
          &buffer))) {
    result = 1;
  }
  if (result == 0) printf("  Allocate 1 MiB buffer: OK\n");

  uint32_t pattern = 0xDEADBEEF;
  if (result == 0 && report_status_error(hrx_stream_fill_buffer(
                         stream, buffer, 0, size, &pattern, sizeof(pattern)))) {
    result = 1;
  }
  // Flush fill before copy — the heap allocator's task CB doesn't guarantee
  // intra-CB ordering for transfer ops on the same buffer.
  if (result == 0 && report_status_error(hrx_stream_flush(stream))) result = 1;
  if (result == 0 && report_status_error(hrx_stream_synchronize(stream))) {
    result = 1;
  }
  if (result == 0) printf("  Fill buffer: OK\n");

  if (result == 0 &&
      report_status_error(hrx_buffer_allocate(
          stream, size,
          HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
          HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
          &buffer2))) {
    result = 1;
  }
  if (result == 0 && report_status_error(hrx_stream_copy_buffer(
                         stream, buffer, 0, buffer2, 0, size))) {
    result = 1;
  }
  if (result == 0 && report_status_error(hrx_stream_flush(stream))) result = 1;
  if (result == 0 && report_status_error(hrx_stream_synchronize(stream))) {
    result = 1;
  }
  if (result == 0) printf("  Copy buffer: OK\n");

  if (result == 0 && report_status_error(hrx_buffer_map(buffer2, HRX_MAP_READ,
                                                        0, size, &mapped))) {
    result = 1;
  }
  if (result == 0) buffer_mapped = true;
  if (result == 0) {
    uint32_t* data = (uint32_t*)mapped;
    for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
      if (data[i] != 0xDEADBEEF) {
        fprintf(stderr, "  VERIFY FAILED at offset %zu: got 0x%08X\n",
                i * sizeof(uint32_t), data[i]);
        result = 1;
        break;
      }
    }
  }
  if (buffer_mapped && report_status_error(hrx_buffer_unmap(buffer2))) {
    result = 1;
  }
  if (result == 0) printf("  Verify data: OK\n");

  hrx_buffer_release(buffer2);
  hrx_buffer_release(buffer);
  hrx_stream_release(stream);
  if (result == 0) printf("  Release: OK\n");

  return result;
}

static int parse_device_spec(const char* spec, hrx_accelerator_type_t* type,
                             int* index) {
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

int main(int argc, char** argv) {
  const char* device_spec = NULL;
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
  hrx_status_t gpu_status = need_gpu ? hrx_gpu_initialize(0) : hrx_ok_status();
  hrx_status_t cpu_status = need_cpu ? hrx_cpu_initialize(0) : hrx_ok_status();
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
      } else if (report_status_error(
                     hrx_gpu_device_get(requested_index, &dev))) {
        result = 1;
      } else {
        result = run_device_smoke_test(dev, device_spec);
      }
    } else {
      if (!hrx_status_is_ok(cpu_status)) {
        print_status_message(stderr,
                             "CPU accelerator not available: ", cpu_status);
        fprintf(stderr, "\n");
        result = 1;
      } else if (report_status_error(
                     hrx_cpu_device_get(requested_index, &dev))) {
        result = 1;
      } else {
        result = run_device_smoke_test(dev, device_spec);
      }
    }
  } else if (test_all) {
    // Test all available devices.
    if (hrx_status_is_ok(gpu_status)) {
      int count = 0;
      bool enumerate_gpu =
          report_status_error(hrx_gpu_device_count(&count)) == 0;
      if (!enumerate_gpu) result = 1;
      for (int i = 0; i < count && enumerate_gpu; i++) {
        hrx_device_t dev = NULL;
        if (report_status_error(hrx_gpu_device_get(i, &dev))) {
          result = 1;
          enumerate_gpu = false;
        } else {
          char label[32];
          snprintf(label, sizeof(label), "gpu:%d", i);
          int r = run_device_smoke_test(dev, label);
          if (r != 0) result = r;
        }
      }
    } else {
      consume_status(&gpu_status);
    }

    if (hrx_status_is_ok(cpu_status)) {
      int count = 0;
      bool enumerate_cpu =
          report_status_error(hrx_cpu_device_count(&count)) == 0;
      if (!enumerate_cpu) result = 1;
      for (int i = 0; i < count && enumerate_cpu; i++) {
        hrx_device_t dev = NULL;
        if (report_status_error(hrx_cpu_device_get(i, &dev))) {
          result = 1;
          enumerate_cpu = false;
        } else {
          char label[32];
          snprintf(label, sizeof(label), "cpu:%d", i);
          int r = run_device_smoke_test(dev, label);
          if (r != 0) result = r;
        }
      }
    } else {
      consume_status(&cpu_status);
    }
  }

  // Shutdown.
  if (gpu_initialized) {
    hrx_status_ignore(hrx_gpu_shutdown());
  } else {
    consume_status(&gpu_status);
  }
  if (cpu_initialized) {
    hrx_status_ignore(hrx_cpu_shutdown());
  } else {
    consume_status(&cpu_status);
  }

  if (result == 0) {
    printf("\nAll tests PASSED\n");
  } else {
    printf("\nSome tests FAILED\n");
  }
  return result;
}
