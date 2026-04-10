// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP Buffer Tracer Interception Library
//
// This library tracks all HIP memory allocations and can dump buffer contents
// before and after kernel launches for debugging/comparison.
//
// Environment variables:
// - HIP_TRACE_FILE: Path to trace output file (default: stderr)
// - HIP_TRACE_LEVEL: 0=off, 1=errors, 2=calls, 3=buffers, 4=verbose (default:
// 2)
// - HIP_TRACE_SYNC: 1=sync before/after kernel launches (default: 0)
// - HIP_TRACE_DUMP: 1=dump buffer contents, 2=dump hash only (default: 0)
// - HIP_TRACE_DUMP_MAX: Max bytes to dump per buffer (default: 1024)
// - HIP_TRACE_KERNEL_FILTER: Substring to match kernel names (empty=all)
// - HIP_TRACE_KERNEL_COUNT: Only trace first N kernels (default: 0=unlimited)
// - HIP_TRACE_KERNEL_FULL_DUMP: Colon-separated list of kernel names for full
// buffer dumps
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "passthrough/hip_function_table.h"

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

#define MAX_TRACKED_BUFFERS 65536
#define MAX_KERNEL_NAME_LEN 512

//===----------------------------------------------------------------------===//
// Buffer Tracking
//===----------------------------------------------------------------------===//

typedef enum {
  BUFFER_TYPE_DEVICE = 0,
  BUFFER_TYPE_HOST = 1,
  BUFFER_TYPE_MANAGED = 2,
} buffer_type_t;

typedef struct {
  void *ptr;
  size_t size;
  buffer_type_t type;
  bool in_use;
  uint64_t alloc_id; // Sequential allocation ID
} tracked_buffer_t;

typedef struct {
  tracked_buffer_t buffers[MAX_TRACKED_BUFFERS];
  size_t count;
  uint64_t next_alloc_id;
  pthread_mutex_t mutex;
} buffer_table_t;

//===----------------------------------------------------------------------===//
// Kernel Tracking
//===----------------------------------------------------------------------===//

typedef struct {
  void *host_func;
  char name[MAX_KERNEL_NAME_LEN];
} kernel_info_t;

typedef struct {
  kernel_info_t *entries;
  size_t count;
  size_t capacity;
  pthread_mutex_t mutex;
} kernel_table_t;

//===----------------------------------------------------------------------===//
// Global State
//===----------------------------------------------------------------------===//

static hip_function_table_t *g_real = NULL;
static hip_function_table_t g_wrapper = {0};
static FILE *g_trace_file = NULL;
static int g_trace_level = 2;
static bool g_trace_sync = false;
static int g_trace_dump = 0; // 0=none, 1=full, 2=hash
static size_t g_trace_dump_max = 1024;
static const char *g_kernel_filter = NULL;
static int g_kernel_count_limit = 0;
static int g_kernel_count = 0;
static char *g_kernel_full_dump_list =
    NULL; // Colon-separated list of kernel names
static pthread_mutex_t g_trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static buffer_table_t g_buffer_table = {0};
static kernel_table_t g_kernel_table = {0};

//===----------------------------------------------------------------------===//
// Logging Helpers
//===----------------------------------------------------------------------===//

static void trace_msg(int level, const char *fmt, ...) {
  if (level > g_trace_level || !g_trace_file)
    return;

  pthread_mutex_lock(&g_trace_mutex);

  // Timestamp
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  fprintf(g_trace_file, "[%ld.%06ld] ", (long)ts.tv_sec, ts.tv_nsec / 1000);

  // Message
  va_list args;
  va_start(args, fmt);
  vfprintf(g_trace_file, fmt, args);
  va_end(args);

  fprintf(g_trace_file, "\n");
  fflush(g_trace_file);

  pthread_mutex_unlock(&g_trace_mutex);
}

static const char *memcpy_kind_name(hipMemcpyKind kind) {
  switch (kind) {
  case hipMemcpyHostToHost:
    return "H2H";
  case hipMemcpyHostToDevice:
    return "H2D";
  case hipMemcpyDeviceToHost:
    return "D2H";
  case hipMemcpyDeviceToDevice:
    return "D2D";
  case hipMemcpyDefault:
    return "Default";
  default:
    return "Unknown";
  }
}

static const char *buffer_type_name(buffer_type_t type) {
  switch (type) {
  case BUFFER_TYPE_DEVICE:
    return "device";
  case BUFFER_TYPE_HOST:
    return "host";
  case BUFFER_TYPE_MANAGED:
    return "managed";
  default:
    return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// Hash Computation (simple FNV-1a)
//===----------------------------------------------------------------------===//

static uint64_t compute_hash(const void *data, size_t size) {
  const uint8_t *bytes = (const uint8_t *)data;
  uint64_t hash = 0xcbf29ce484222325ULL; // FNV offset basis
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL; // FNV prime
  }
  return hash;
}

//===----------------------------------------------------------------------===//
// Buffer Table Operations
//===----------------------------------------------------------------------===//

static void buffer_table_init(void) {
  pthread_mutex_init(&g_buffer_table.mutex, NULL);
  g_buffer_table.count = 0;
  g_buffer_table.next_alloc_id = 1;
  memset(g_buffer_table.buffers, 0, sizeof(g_buffer_table.buffers));
}

static bool buffer_table_add(void *ptr, size_t size, buffer_type_t type) {
  if (!ptr || size == 0)
    return true; // NULL/zero-size allocs are valid

  pthread_mutex_lock(&g_buffer_table.mutex);

  // Find free slot
  int slot = -1;
  for (size_t i = 0; i < MAX_TRACKED_BUFFERS; ++i) {
    if (!g_buffer_table.buffers[i].in_use) {
      slot = (int)i;
      break;
    }
  }

  if (slot < 0) {
    pthread_mutex_unlock(&g_buffer_table.mutex);
    trace_msg(1, "WARNING: buffer table full, cannot track %p", ptr);
    return false;
  }

  g_buffer_table.buffers[slot].ptr = ptr;
  g_buffer_table.buffers[slot].size = size;
  g_buffer_table.buffers[slot].type = type;
  g_buffer_table.buffers[slot].in_use = true;
  g_buffer_table.buffers[slot].alloc_id = g_buffer_table.next_alloc_id++;
  g_buffer_table.count++;

  pthread_mutex_unlock(&g_buffer_table.mutex);
  return true;
}

static bool buffer_table_remove(void *ptr) {
  if (!ptr)
    return true; // Free(NULL) is valid

  pthread_mutex_lock(&g_buffer_table.mutex);

  for (size_t i = 0; i < MAX_TRACKED_BUFFERS; ++i) {
    if (g_buffer_table.buffers[i].in_use &&
        g_buffer_table.buffers[i].ptr == ptr) {
      g_buffer_table.buffers[i].in_use = false;
      g_buffer_table.count--;
      pthread_mutex_unlock(&g_buffer_table.mutex);
      return true;
    }
  }

  pthread_mutex_unlock(&g_buffer_table.mutex);
  return false; // Not found (might be external allocation)
}

static tracked_buffer_t *buffer_table_find(void *ptr) {
  // Caller must hold mutex
  for (size_t i = 0; i < MAX_TRACKED_BUFFERS; ++i) {
    if (g_buffer_table.buffers[i].in_use &&
        g_buffer_table.buffers[i].ptr == ptr) {
      return &g_buffer_table.buffers[i];
    }
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Kernel Table Operations
//===----------------------------------------------------------------------===//

static void kernel_table_init(void) {
  pthread_mutex_init(&g_kernel_table.mutex, NULL);
  g_kernel_table.count = 0;
  g_kernel_table.capacity = 256;
  g_kernel_table.entries =
      (kernel_info_t *)calloc(g_kernel_table.capacity, sizeof(kernel_info_t));
}

static void kernel_table_add(void *host_func, const char *name) {
  if (!host_func)
    return;

  pthread_mutex_lock(&g_kernel_table.mutex);

  // Check if already exists
  for (size_t i = 0; i < g_kernel_table.count; ++i) {
    if (g_kernel_table.entries[i].host_func == host_func) {
      pthread_mutex_unlock(&g_kernel_table.mutex);
      return;
    }
  }

  // Grow if needed
  if (g_kernel_table.count >= g_kernel_table.capacity) {
    size_t new_cap = g_kernel_table.capacity * 2;
    kernel_info_t *new_entries = (kernel_info_t *)realloc(
        g_kernel_table.entries, new_cap * sizeof(kernel_info_t));
    if (!new_entries) {
      pthread_mutex_unlock(&g_kernel_table.mutex);
      return;
    }
    g_kernel_table.entries = new_entries;
    g_kernel_table.capacity = new_cap;
  }

  kernel_info_t *entry = &g_kernel_table.entries[g_kernel_table.count++];
  entry->host_func = host_func;
  if (name) {
    strncpy(entry->name, name, MAX_KERNEL_NAME_LEN - 1);
    entry->name[MAX_KERNEL_NAME_LEN - 1] = '\0';
  } else {
    snprintf(entry->name, MAX_KERNEL_NAME_LEN, "kernel_%p", host_func);
  }

  pthread_mutex_unlock(&g_kernel_table.mutex);
}

static const char *kernel_table_get_name(void *host_func) {
  pthread_mutex_lock(&g_kernel_table.mutex);

  for (size_t i = 0; i < g_kernel_table.count; ++i) {
    if (g_kernel_table.entries[i].host_func == host_func) {
      const char *name = g_kernel_table.entries[i].name;
      pthread_mutex_unlock(&g_kernel_table.mutex);
      return name;
    }
  }

  pthread_mutex_unlock(&g_kernel_table.mutex);
  return NULL;
}

//===----------------------------------------------------------------------===//
// Buffer Dumping
//===----------------------------------------------------------------------===//

static void dump_buffer_hex(const void *data, size_t size, size_t max_bytes) {
  const uint8_t *bytes = (const uint8_t *)data;
  size_t dump_size = (size > max_bytes) ? max_bytes : size;

  fprintf(g_trace_file, "    ");
  for (size_t i = 0; i < dump_size; ++i) {
    fprintf(g_trace_file, "%02x", bytes[i]);
    if ((i + 1) % 32 == 0) {
      fprintf(g_trace_file, "\n    ");
    } else if ((i + 1) % 4 == 0) {
      fprintf(g_trace_file, " ");
    }
  }
  if (size > max_bytes) {
    fprintf(g_trace_file, "... (%zu more bytes)", size - max_bytes);
  }
  fprintf(g_trace_file, "\n");
}

static void dump_all_buffers_ex(const char *label, bool force_full_dump) {
  if (g_trace_dump == 0)
    return;

  pthread_mutex_lock(&g_buffer_table.mutex);

  trace_msg(3, "=== Buffer Dump: %s (%zu buffers)%s ===", label,
            g_buffer_table.count, force_full_dump ? " [FULL DUMP]" : "");

  // Allocate host staging buffer for device memory reads
  void *host_staging = NULL;
  size_t staging_size = 0;

  for (size_t i = 0; i < MAX_TRACKED_BUFFERS; ++i) {
    tracked_buffer_t *buf = &g_buffer_table.buffers[i];
    if (!buf->in_use)
      continue;

    // Skip zero-size buffers
    if (buf->size == 0)
      continue;

    // Determine how many bytes to read - full dump reads everything
    size_t read_size = buf->size;
    size_t max_dump = force_full_dump ? buf->size : g_trace_dump_max;
    if (!force_full_dump && g_trace_dump_max > 0 &&
        read_size > g_trace_dump_max) {
      read_size = g_trace_dump_max;
    }

    // For device buffers, we need to copy to host
    const void *read_ptr = buf->ptr;

    if (buf->type == BUFFER_TYPE_DEVICE) {
      // Ensure staging buffer is large enough
      if (read_size > staging_size) {
        if (host_staging)
          free(host_staging);
        staging_size = read_size;
        host_staging = malloc(staging_size);
        if (!host_staging) {
          trace_msg(1, "  [alloc %lu] %s %p size=%zu - FAILED TO ALLOC STAGING",
                    buf->alloc_id, buffer_type_name(buf->type), buf->ptr,
                    buf->size);
          continue;
        }
      }

      // Copy from device to host
      hipError_t err = g_real->hipMemcpy(host_staging, buf->ptr, read_size,
                                         hipMemcpyDeviceToHost);
      if (err != 0) {
        trace_msg(1, "  [alloc %lu] %s %p size=%zu - MEMCPY FAILED: %d",
                  buf->alloc_id, buffer_type_name(buf->type), buf->ptr,
                  buf->size, err);
        continue;
      }
      read_ptr = host_staging;
    }

    if (g_trace_dump == 2 && !force_full_dump) {
      // Hash only (unless forced to full dump)
      uint64_t hash = compute_hash(read_ptr, read_size);
      fprintf(g_trace_file, "  [alloc %lu] %s %p size=%zu hash=0x%016lx\n",
              buf->alloc_id, buffer_type_name(buf->type), buf->ptr, buf->size,
              hash);
    } else {
      // Full dump
      fprintf(g_trace_file, "  [alloc %lu] %s %p size=%zu:\n", buf->alloc_id,
              buffer_type_name(buf->type), buf->ptr, buf->size);
      dump_buffer_hex(read_ptr, buf->size, max_dump);
    }
  }

  if (host_staging)
    free(host_staging);

  pthread_mutex_unlock(&g_buffer_table.mutex);

  fflush(g_trace_file);
}

static void dump_all_buffers(const char *label) {
  dump_all_buffers_ex(label, false);
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Memory Allocation
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipMalloc(void **ptr, size_t size) {
  hipError_t err = g_real->hipMalloc(ptr, size);
  if (err == 0 && ptr && *ptr) {
    buffer_table_add(*ptr, size, BUFFER_TYPE_DEVICE);
  }
  trace_msg(2, "hipMalloc(size=%zu) -> ptr=%p, ret=%d", size, ptr ? *ptr : NULL,
            err);
  return err;
}

static hipError_t wrap_hipFree(void *ptr) {
  buffer_table_remove(ptr);
  hipError_t err = g_real->hipFree(ptr);
  trace_msg(2, "hipFree(%p) -> %d", ptr, err);
  return err;
}

static hipError_t wrap_hipHostMalloc(void **ptr, size_t size,
                                     unsigned int flags) {
  hipError_t err = g_real->hipHostMalloc(ptr, size, flags);
  if (err == 0 && ptr && *ptr) {
    buffer_table_add(*ptr, size, BUFFER_TYPE_HOST);
  }
  trace_msg(2, "hipHostMalloc(size=%zu, flags=0x%x) -> ptr=%p, ret=%d", size,
            flags, ptr ? *ptr : NULL, err);
  return err;
}

static hipError_t wrap_hipHostFree(void *ptr) {
  buffer_table_remove(ptr);
  hipError_t err = g_real->hipHostFree(ptr);
  trace_msg(2, "hipHostFree(%p) -> %d", ptr, err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Memory Operations
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                                 hipMemcpyKind kind) {
  hipError_t err = g_real->hipMemcpy(dst, src, sizeBytes, kind);
  trace_msg(2, "hipMemcpy(dst=%p, src=%p, size=%zu, kind=%s) -> %d", dst, src,
            sizeBytes, memcpy_kind_name(kind), err);
  return err;
}

static hipError_t wrap_hipMemcpyAsync(void *dst, const void *src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream) {
  hipError_t err = g_real->hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  trace_msg(
      2, "hipMemcpyAsync(dst=%p, src=%p, size=%zu, kind=%s, stream=%p) -> %d",
      dst, src, sizeBytes, memcpy_kind_name(kind), (void *)stream, err);
  return err;
}

static hipError_t wrap_hipMemset(void *dst, int value, size_t sizeBytes) {
  hipError_t err = g_real->hipMemset(dst, value, sizeBytes);
  trace_msg(2, "hipMemset(dst=%p, value=0x%02x, size=%zu) -> %d", dst, value,
            sizeBytes, err);
  return err;
}

static hipError_t wrap_hipMemsetAsync(void *dst, int value, size_t sizeBytes,
                                      hipStream_t stream) {
  hipError_t err = g_real->hipMemsetAsync(dst, value, sizeBytes, stream);
  trace_msg(2,
            "hipMemsetAsync(dst=%p, value=0x%02x, size=%zu, stream=%p) -> %d",
            dst, value, sizeBytes, (void *)stream, err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Device Management
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipInit(unsigned int flags) {
  hipError_t err = g_real->hipInit(flags);
  trace_msg(2, "hipInit(flags=0x%x) -> %d", flags, err);
  return err;
}

static hipError_t wrap_hipGetDevice(int *deviceId) {
  hipError_t err = g_real->hipGetDevice(deviceId);
  trace_msg(2, "hipGetDevice() -> device=%d, ret=%d", deviceId ? *deviceId : -1,
            err);
  return err;
}

static hipError_t wrap_hipGetDeviceCount(int *count) {
  hipError_t err = g_real->hipGetDeviceCount(count);
  trace_msg(2, "hipGetDeviceCount() -> count=%d, ret=%d", count ? *count : -1,
            err);
  return err;
}

static hipError_t wrap_hipSetDevice(int deviceId) {
  hipError_t err = g_real->hipSetDevice(deviceId);
  trace_msg(2, "hipSetDevice(%d) -> %d", deviceId, err);
  return err;
}

static hipError_t wrap_hipDeviceSynchronize(void) {
  hipError_t err = g_real->hipDeviceSynchronize();
  trace_msg(2, "hipDeviceSynchronize() -> %d", err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Stream Management
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipStreamCreate(hipStream_t *stream) {
  hipError_t err = g_real->hipStreamCreate(stream);
  trace_msg(2, "hipStreamCreate() -> stream=%p, ret=%d",
            stream ? (void *)*stream : NULL, err);
  return err;
}

static hipError_t wrap_hipStreamCreateWithFlags(hipStream_t *stream,
                                                unsigned int flags) {
  hipError_t err = g_real->hipStreamCreateWithFlags(stream, flags);
  trace_msg(2, "hipStreamCreateWithFlags(flags=0x%x) -> stream=%p, ret=%d",
            flags, stream ? (void *)*stream : NULL, err);
  return err;
}

static hipError_t wrap_hipStreamDestroy(hipStream_t stream) {
  hipError_t err = g_real->hipStreamDestroy(stream);
  trace_msg(2, "hipStreamDestroy(stream=%p) -> %d", (void *)stream, err);
  return err;
}

static hipError_t wrap_hipStreamSynchronize(hipStream_t stream) {
  hipError_t err = g_real->hipStreamSynchronize(stream);
  trace_msg(2, "hipStreamSynchronize(stream=%p) -> %d", (void *)stream, err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Event Management
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipEventCreate(hipEvent_t *event) {
  hipError_t err = g_real->hipEventCreate(event);
  trace_msg(2, "hipEventCreate() -> event=%p, ret=%d",
            event ? (void *)*event : NULL, err);
  return err;
}

static hipError_t wrap_hipEventCreateWithFlags(hipEvent_t *event,
                                               unsigned int flags) {
  hipError_t err = g_real->hipEventCreateWithFlags(event, flags);
  trace_msg(2, "hipEventCreateWithFlags(flags=0x%x) -> event=%p, ret=%d", flags,
            event ? (void *)*event : NULL, err);
  return err;
}

static hipError_t wrap_hipEventDestroy(hipEvent_t event) {
  hipError_t err = g_real->hipEventDestroy(event);
  trace_msg(2, "hipEventDestroy(event=%p) -> %d", (void *)event, err);
  return err;
}

static hipError_t wrap_hipEventRecord(hipEvent_t event, hipStream_t stream) {
  hipError_t err = g_real->hipEventRecord(event, stream);
  trace_msg(2, "hipEventRecord(event=%p, stream=%p) -> %d", (void *)event,
            (void *)stream, err);
  return err;
}

static hipError_t wrap_hipEventSynchronize(hipEvent_t event) {
  hipError_t err = g_real->hipEventSynchronize(event);
  trace_msg(2, "hipEventSynchronize(event=%p) -> %d", (void *)event, err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Kernel Launch (with buffer tracing)
//===----------------------------------------------------------------------===//

static bool should_trace_kernel(const char *kernel_name) {
  // Check kernel count limit
  if (g_kernel_count_limit > 0 && g_kernel_count >= g_kernel_count_limit) {
    return false;
  }

  // Check kernel filter
  if (g_kernel_filter && *g_kernel_filter) {
    if (!kernel_name || !strstr(kernel_name, g_kernel_filter)) {
      return false;
    }
  }

  return true;
}

// Check if kernel is in the full dump list (colon-separated)
static bool should_full_dump_kernel(const char *kernel_name) {
  if (!g_kernel_full_dump_list || !*g_kernel_full_dump_list || !kernel_name) {
    return false;
  }

  // Make a copy since strtok modifies the string
  size_t list_len = strlen(g_kernel_full_dump_list);
  char *list_copy = (char *)malloc(list_len + 1);
  if (!list_copy)
    return false;
  strcpy(list_copy, g_kernel_full_dump_list);

  bool found = false;
  char *saveptr = NULL;
  char *token = strtok_r(list_copy, ":", &saveptr);
  while (token) {
    // Check if the kernel name contains this token as substring
    if (strstr(kernel_name, token)) {
      found = true;
      trace_msg(3, "Full dump match: kernel='%s' matches token='%s'",
                kernel_name, token);
      break;
    }
    token = strtok_r(NULL, ":", &saveptr);
  }

  free(list_copy);
  return found;
}

static hipError_t wrap_hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void **kernelParams, void **extra) {

  // Get kernel name if available
  const char *kernel_name = kernel_table_get_name((void *)f);

  bool do_trace = should_trace_kernel(kernel_name);
  bool do_full_dump = should_full_dump_kernel(kernel_name);

  if (do_trace && g_trace_sync) {
    // Sync before launch
    g_real->hipDeviceSynchronize();

    char before_label[256];
    snprintf(before_label, sizeof(before_label), "BEFORE kernel #%d: %s",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)");
    dump_all_buffers_ex(before_label, do_full_dump);
  }

  trace_msg(
      2,
      "hipModuleLaunchKernel(func=%p [%s], grid=(%u,%u,%u), block=(%u,%u,%u), "
      "shared=%u, stream=%p)",
      (void *)f, kernel_name ? kernel_name : "?", gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, (void *)stream);

  hipError_t err = g_real->hipModuleLaunchKernel(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);

  if (do_trace && g_trace_sync) {
    // Sync after launch
    g_real->hipDeviceSynchronize();

    char after_label[256];
    snprintf(after_label, sizeof(after_label), "AFTER kernel #%d: %s (ret=%d)",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)", err);
    dump_all_buffers_ex(after_label, do_full_dump);
  }

  if (do_trace) {
    g_kernel_count++;
  }

  trace_msg(2, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipLaunchKernel(const void *function_address,
                                       dim3 numBlocks, dim3 dimBlocks,
                                       void **args, size_t sharedMemBytes,
                                       hipStream_t stream) {
  // Get kernel name if available
  const char *kernel_name = kernel_table_get_name((void *)function_address);

  bool do_trace = should_trace_kernel(kernel_name);
  bool do_full_dump = should_full_dump_kernel(kernel_name);

  if (do_trace && g_trace_sync) {
    // Sync before launch
    g_real->hipDeviceSynchronize();

    char before_label[256];
    snprintf(before_label, sizeof(before_label), "BEFORE kernel #%d: %s",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)");
    dump_all_buffers_ex(before_label, do_full_dump);
  }

  trace_msg(2,
            "hipLaunchKernel(func=%p [%s], grid=(%u,%u,%u), block=(%u,%u,%u), "
            "shared=%zu, stream=%p)",
            function_address, kernel_name ? kernel_name : "?", numBlocks.x,
            numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z,
            sharedMemBytes, (void *)stream);

  hipError_t err = g_real->hipLaunchKernel(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);

  if (do_trace && g_trace_sync) {
    // Sync after launch
    g_real->hipDeviceSynchronize();

    char after_label[256];
    snprintf(after_label, sizeof(after_label), "AFTER kernel #%d: %s (ret=%d)",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)", err);
    dump_all_buffers_ex(after_label, do_full_dump);
  }

  if (do_trace) {
    g_kernel_count++;
  }

  trace_msg(2, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipExtModuleLaunchKernel(
    hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ, unsigned int localWorkSizeX,
    unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t stream, void **kernelParams,
    void **extra, hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags) {

  const char *kernel_name = kernel_table_get_name((void *)f);
  bool do_trace = should_trace_kernel(kernel_name);
  bool do_full_dump = should_full_dump_kernel(kernel_name);

  if (do_trace && g_trace_sync) {
    g_real->hipDeviceSynchronize();
    char before_label[256];
    snprintf(before_label, sizeof(before_label), "BEFORE kernel #%d: %s",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)");
    dump_all_buffers_ex(before_label, do_full_dump);
  }

  trace_msg(2,
            "hipExtModuleLaunchKernel(func=%p [%s], globalSize=(%u,%u,%u), "
            "localSize=(%u,%u,%u), shared=%zu, stream=%p, flags=0x%x)",
            (void *)f, kernel_name ? kernel_name : "?", globalWorkSizeX,
            globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
            localWorkSizeZ, sharedMemBytes, (void *)stream, flags);

  hipError_t err = g_real->hipExtModuleLaunchKernel(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
      localWorkSizeY, localWorkSizeZ, sharedMemBytes, stream, kernelParams,
      extra, startEvent, stopEvent, flags);

  if (do_trace && g_trace_sync) {
    g_real->hipDeviceSynchronize();
    char after_label[256];
    snprintf(after_label, sizeof(after_label), "AFTER kernel #%d: %s (ret=%d)",
             g_kernel_count, kernel_name ? kernel_name : "(unknown)", err);
    dump_all_buffers_ex(after_label, do_full_dump);
  }

  if (do_trace) {
    g_kernel_count++;
  }

  trace_msg(2, "  -> %d", err);
  return err;
}

//===----------------------------------------------------------------------===//
// Wrapper Functions - Fat Binary Registration
//===----------------------------------------------------------------------===//

static void **wrap___hipRegisterFatBinary(const void *data) {
  void **result = g_real->__hipRegisterFatBinary(data);
  trace_msg(2, "__hipRegisterFatBinary(data=%p) -> handle=%p", data,
            (void *)result);
  return result;
}

static void wrap___hipUnregisterFatBinary(void **fatCubinHandle) {
  trace_msg(2, "__hipUnregisterFatBinary(handle=%p)", (void *)fatCubinHandle);
  g_real->__hipUnregisterFatBinary(fatCubinHandle);
}

static void wrap___hipRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       void *tid, void *bid, dim3 *blockDim,
                                       dim3 *gridDim, int *wSize) {
  // Track kernel name
  kernel_table_add((void *)hostFun, deviceName);

  trace_msg(2, "__hipRegisterFunction(handle=%p, host=%p, device=%s)",
            (void *)fatCubinHandle, hostFun,
            deviceName ? deviceName : "(null)");
  g_real->__hipRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                thread_limit, tid, bid, blockDim, gridDim,
                                wSize);
}

static void wrap___hipRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  trace_msg(2, "__hipRegisterVar(handle=%p, name=%s, size=%zu)",
            (void *)fatCubinHandle, deviceName ? deviceName : "(null)", size);
  g_real->__hipRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
                           ext, size, constant, global);
}

//===----------------------------------------------------------------------===//
// Interceptor Interface
//===----------------------------------------------------------------------===//

__attribute__((visibility("default"))) hip_function_table_t *
hip_interceptor_init(hip_function_table_t *real_functions) {
  g_real = real_functions;

  // Initialize tables
  buffer_table_init();
  kernel_table_init();

  // Initialize tracing
  const char *trace_path = getenv("HIP_TRACE_FILE");
  if (trace_path && *trace_path) {
    g_trace_file = fopen(trace_path, "w");
    if (!g_trace_file) {
      g_trace_file = stderr;
      fprintf(stderr, "hip_buffer_tracer: failed to open trace file: %s\n",
              trace_path);
    }
  } else {
    g_trace_file = stderr;
  }

  const char *trace_level_str = getenv("HIP_TRACE_LEVEL");
  if (trace_level_str) {
    g_trace_level = atoi(trace_level_str);
  }

  const char *trace_sync_str = getenv("HIP_TRACE_SYNC");
  if (trace_sync_str) {
    g_trace_sync = atoi(trace_sync_str) != 0;
  }

  const char *trace_dump_str = getenv("HIP_TRACE_DUMP");
  if (trace_dump_str) {
    g_trace_dump = atoi(trace_dump_str);
  }

  const char *trace_dump_max_str = getenv("HIP_TRACE_DUMP_MAX");
  if (trace_dump_max_str) {
    g_trace_dump_max = (size_t)atol(trace_dump_max_str);
  }

  g_kernel_filter = getenv("HIP_TRACE_KERNEL_FILTER");

  const char *kernel_count_str = getenv("HIP_TRACE_KERNEL_COUNT");
  if (kernel_count_str) {
    g_kernel_count_limit = atoi(kernel_count_str);
  }

  // Colon-separated list of kernel names for full buffer dumps
  const char *full_dump_str = getenv("HIP_TRACE_KERNEL_FULL_DUMP");
  if (full_dump_str && *full_dump_str) {
    g_kernel_full_dump_list = strdup(full_dump_str);
    trace_msg(1, "Full dump enabled for kernels (len=%zu): %s",
              strlen(g_kernel_full_dump_list), g_kernel_full_dump_list);
  }

  trace_msg(1, "HIP Buffer Tracer initialized");
  trace_msg(1, "  trace_level=%d, trace_sync=%d, trace_dump=%d, dump_max=%zu",
            g_trace_level, g_trace_sync, g_trace_dump, g_trace_dump_max);
  if (g_kernel_filter && *g_kernel_filter) {
    trace_msg(1, "  kernel_filter=%s", g_kernel_filter);
  }
  if (g_kernel_count_limit > 0) {
    trace_msg(1, "  kernel_count_limit=%d", g_kernel_count_limit);
  }

  // Copy real table to wrapper
  memcpy(&g_wrapper, g_real, sizeof(g_wrapper));

  // Replace functions with wrappers
  // Device Management
  g_wrapper.hipInit = wrap_hipInit;
  g_wrapper.hipGetDevice = wrap_hipGetDevice;
  g_wrapper.hipGetDeviceCount = wrap_hipGetDeviceCount;
  g_wrapper.hipSetDevice = wrap_hipSetDevice;
  g_wrapper.hipDeviceSynchronize = wrap_hipDeviceSynchronize;

  // Memory Management
  g_wrapper.hipMalloc = wrap_hipMalloc;
  g_wrapper.hipFree = wrap_hipFree;
  g_wrapper.hipHostMalloc = wrap_hipHostMalloc;
  g_wrapper.hipHostFree = wrap_hipHostFree;
  g_wrapper.hipMemcpy = wrap_hipMemcpy;
  g_wrapper.hipMemcpyAsync = wrap_hipMemcpyAsync;
  g_wrapper.hipMemset = wrap_hipMemset;
  g_wrapper.hipMemsetAsync = wrap_hipMemsetAsync;

  // Stream Management
  g_wrapper.hipStreamCreate = wrap_hipStreamCreate;
  g_wrapper.hipStreamCreateWithFlags = wrap_hipStreamCreateWithFlags;
  g_wrapper.hipStreamDestroy = wrap_hipStreamDestroy;
  g_wrapper.hipStreamSynchronize = wrap_hipStreamSynchronize;

  // Event Management
  g_wrapper.hipEventCreate = wrap_hipEventCreate;
  g_wrapper.hipEventCreateWithFlags = wrap_hipEventCreateWithFlags;
  g_wrapper.hipEventDestroy = wrap_hipEventDestroy;
  g_wrapper.hipEventRecord = wrap_hipEventRecord;
  g_wrapper.hipEventSynchronize = wrap_hipEventSynchronize;

  // Kernel Launch
  g_wrapper.hipModuleLaunchKernel = wrap_hipModuleLaunchKernel;
  g_wrapper.hipLaunchKernel = wrap_hipLaunchKernel;
  g_wrapper.hipExtModuleLaunchKernel = wrap_hipExtModuleLaunchKernel;

  // Fat Binary Registration
  g_wrapper.__hipRegisterFatBinary = wrap___hipRegisterFatBinary;
  g_wrapper.__hipUnregisterFatBinary = wrap___hipUnregisterFatBinary;
  g_wrapper.__hipRegisterFunction = wrap___hipRegisterFunction;
  g_wrapper.__hipRegisterVar = wrap___hipRegisterVar;

  return &g_wrapper;
}

__attribute__((visibility("default"))) void hip_interceptor_shutdown(void) {
  trace_msg(1, "HIP Buffer Tracer shutting down");
  trace_msg(1, "  Total kernels traced: %d", g_kernel_count);
  trace_msg(1, "  Total buffers tracked: %zu", g_buffer_table.count);

  if (g_trace_file && g_trace_file != stderr) {
    fclose(g_trace_file);
    g_trace_file = NULL;
  }

  if (g_kernel_table.entries) {
    free(g_kernel_table.entries);
    g_kernel_table.entries = NULL;
  }
}
