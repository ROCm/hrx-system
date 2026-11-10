// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// hrx-info: CLI tool for device enumeration and smoke testing.
// Device selection uses the public HRX discovery and sealed-group APIs.

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrx_runtime.h"

// IREE flags for CLI parsing (linked statically into this binary).
#include "iree/base/tooling/flags.h"

static void print_status_message(FILE* stream, const char* prefix,
                                 hrx_status_t status) {
  char* message = NULL;
  size_t length = 0;
  hrx_status_to_string(status, &message, &length);
  fprintf(stream, "%s%s", prefix, message ? message : "?");
  hrx_status_free_message(message);
}

static int report_status_error(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return 0;
  }
  print_status_message(stderr, "ERROR: ", status);
  fprintf(stderr, "\n");
  hrx_status_ignore(status);
  return 1;
}

static int run_device_smoke_test(hrx_device_t device, const char* label) {
  printf("Opening %s...\n", label);

  hrx_stream_t stream = NULL;
  hrx_buffer_t buffer = NULL;
  hrx_buffer_t buffer2 = NULL;
  void* mapped = NULL;
  bool buffer_mapped = false;
  int result = 0;

  hrx_queue_t queue = NULL;
  for (uint32_t i = 0;
       result == 0 && !queue && i < hrx_device_queue_family_count(device);
       ++i) {
    hrx_queue_family_info_t family;
    if (report_status_error(
            hrx_device_query_queue_family(device, i, &family))) {
      result = 1;
    } else if ((family.roles & HRX_QUEUE_ROLE_TRANSFER) &&
               family.provisioned_queue_count) {
      queue = hrx_device_get_queue(device, i, 0);
    }
  }
  if (result == 0 &&
      report_status_error(hrx_stream_create(queue, 0, &stream))) {
    result = 1;
  }

  const size_t size = 1024 * 1024;
  if (result == 0 &&
      report_status_error(hrx_buffer_allocate(
          stream, size,
          HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
          HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
          &buffer))) {
    result = 1;
  }
  if (result == 0) {
    printf("  Allocate 1 MiB buffer: OK\n");
  }

  uint32_t pattern = 0xDEADBEEF;
  if (result == 0 && report_status_error(hrx_stream_fill_buffer(
                         stream, buffer, 0, size, &pattern, sizeof(pattern)))) {
    result = 1;
  }
  // Complete each smoke operation before reporting success.
  if (result == 0 && report_status_error(hrx_stream_flush(stream))) {
    result = 1;
  }
  if (result == 0 && report_status_error(hrx_stream_synchronize(stream))) {
    result = 1;
  }
  if (result == 0) {
    printf("  Fill buffer: OK\n");
  }

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
  if (result == 0 && report_status_error(hrx_stream_flush(stream))) {
    result = 1;
  }
  if (result == 0 && report_status_error(hrx_stream_synchronize(stream))) {
    result = 1;
  }
  if (result == 0) {
    printf("  Copy buffer: OK\n");
  }

  if (result == 0 && report_status_error(hrx_buffer_map(buffer2, HRX_MAP_READ,
                                                        0, size, &mapped))) {
    result = 1;
  }
  if (result == 0) {
    buffer_mapped = true;
  }
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
  if (result == 0) {
    printf("  Verify data: OK\n");
  }

  hrx_buffer_release(buffer2);
  hrx_buffer_release(buffer);
  hrx_stream_release(stream);
  if (result == 0) {
    printf("  Release: OK\n");
  }

  return result;
}

// GPU scheme selection is CLI policy, independent of native HRX construction.
static const char* gpu_driver_name(void) {
  const char* value = getenv("HRX_GPU_DRIVER");
  return value && value[0] ? value : "amdgpu";
}

static int inspect_driver(hrx_runtime_t runtime, const char* driver_name,
                          bool run_tests, int selected_index, bool optional) {
  const hrx_host_allocator_t allocator = hrx_host_allocator_system();
  size_t count = 0;
  hrx_device_info_t* infos = NULL;
  hrx_status_t status = hrx_runtime_query_available_devices(
      runtime, driver_name, allocator, &count, &infos);
  if (!hrx_status_is_ok(status)) {
    if (optional) {
      printf("%s: unavailable (", driver_name);
      print_status_message(stdout, "", status);
      printf(")\n");
      hrx_status_ignore(status);
      return 0;
    }
    fprintf(stderr, "%s discovery: ", driver_name);
    return report_status_error(status);
  }
  // Testing the aggregate and each physical entry would create overlapping
  // execution devices. This tool lists and selects physical legacy GPUs only.
  if (strcmp(driver_name, "legacy-amdgpu") == 0) {
    size_t physical_count = 0;
    for (size_t i = 0; i < count; ++i) {
      if (strcmp(infos[i].uri, "legacy-amdgpu://") != 0) {
        infos[physical_count++] = infos[i];
      }
    }
    count = physical_count;
  }
  printf("%s: %zu device%s\n", driver_name, count, count != 1 ? "s" : "");
  for (size_t i = 0; i < count; ++i) {
    printf("  [%zu] %s (%s)\n", i, infos[i].uri, infos[i].name);
  }
  if (!run_tests) {
    hrx_host_allocator_free(allocator, infos);
    return 0;
  }
  if (!count || (selected_index >= 0 && (size_t)selected_index >= count)) {
    fprintf(stderr, "No selected device in %s\n", driver_name);
    hrx_host_allocator_free(allocator, infos);
    return 1;
  }

  const size_t selected_count = selected_index >= 0 ? 1 : count;
  hrx_device_create_params_t* params = NULL;
  status = hrx_host_allocator_malloc(
      allocator, selected_count * sizeof(*params), (void**)&params);
  hrx_device_group_t group = NULL;
  if (hrx_status_is_ok(status)) {
    for (size_t i = 0; i < selected_count; ++i) {
      params[i].uri =
          infos[selected_index >= 0 ? (size_t)selected_index : i].uri;
    }
    status = hrx_device_group_create(runtime, selected_count, params, &group);
  }
  int result = report_status_error(status);
  if (!result) {
    for (size_t i = 0; i < selected_count; ++i) {
      result |= run_device_smoke_test(hrx_device_group_get_device(group, i),
                                      params[i].uri);
    }
  }
  hrx_device_group_release(group);
  hrx_host_allocator_free(allocator, params);
  hrx_host_allocator_free(allocator, infos);
  return result;
}

int main(int argc, char** argv) {
  const char* device_uri = NULL;
  const char* driver_name = NULL;
  bool test_all = false;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--device=", 9) == 0) {
      device_uri = argv[i] + 9;
    } else if (strncmp(argv[i], "--driver=", 9) == 0) {
      driver_name = argv[i] + 9;
    } else if (strcmp(argv[i], "--test=all") == 0) {
      test_all = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: hrx-info [--driver=SCHEME] [--device=URI | --test=all]\n");
      printf(
          "  (no args)       List GPU and CPU devices without creating "
          "them\n");
      printf(
          "  --driver=NAME   List a HAL scheme (task, amdgpu, rdna, cdna, "
          "xdna)\n");
      printf("  --device=URI    Run smoke work on an exact HAL URI\n");
      printf(
          "  --test=all      Test all devices listed by the selected "
          "scheme(s)\n");
      return 0;
    }
  }
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  if (device_uri && (driver_name || test_all)) {
    fprintf(
        stderr,
        "--device selects one exact device; use --driver with --test=all\n");
    return 1;
  }

  int major, minor, patch;
  hrx_runtime_version(&major, &minor, &patch);
  printf("HRX v%d.%d.%d\n", major, minor, patch);
  hrx_runtime_t runtime = NULL;
  if (report_status_error(
          hrx_runtime_create(NULL, hrx_host_allocator_system(), &runtime))) {
    return 1;
  }
  int result = 0;
  if (device_uri) {
    // Existing CLI shorthands select an ordinal in one explicit scheme.
    const bool gpu = strncmp(device_uri, "gpu:", 4) == 0;
    const bool cpu = strncmp(device_uri, "cpu:", 4) == 0;
    if (gpu || cpu) {
      char* end = NULL;
      const long index = strtol(device_uri + 4, &end, 10);
      if (end == device_uri + 4 || *end || index < 0 || index > INT_MAX) {
        fprintf(stderr, "Invalid device ordinal: %s\n", device_uri);
        result = 1;
      } else {
        result = inspect_driver(runtime, gpu ? gpu_driver_name() : "task", true,
                                (int)index, false);
      }
    } else {
      const hrx_device_create_params_t params = {.uri = device_uri};
      hrx_device_group_t group = NULL;
      result = report_status_error(
          hrx_device_group_create(runtime, 1, &params, &group));
      if (!result) {
        result = run_device_smoke_test(hrx_device_group_get_device(group, 0),
                                       device_uri);
      }
      hrx_device_group_release(group);
    }
  } else if (driver_name) {
    result = inspect_driver(runtime, driver_name, test_all, -1, false);
  } else {
    result = inspect_driver(runtime, gpu_driver_name(), test_all, -1, true);
    result |= inspect_driver(runtime, "task", test_all, -1, false);
  }
  hrx_runtime_release(runtime);
  return result;
}
