// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cinttypes>
#include <cstdio>
#include <vector>

#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

namespace {

constexpr iree_device_size_t kChunkBytes = 1u << 20;
constexpr iree_device_size_t kMissBytes = 33u << 10;

void DestroyBuffers(std::vector<iree_hal_amdxdna_native_buffer_t*>* buffers) {
  for (iree_hal_amdxdna_native_buffer_t* buffer : *buffers) {
    iree_hal_amdxdna_native_buffer_c_destroy(buffer);
  }
  buffers->clear();
}

int Fail(const char* phase, iree_status_t status,
         std::vector<iree_hal_amdxdna_native_buffer_t*>* buffers,
         iree_hal_amdxdna_native_device_t* device) {
  const int status_code = static_cast<int>(iree_status_code(status));
  iree_status_ignore(status);
  DestroyBuffers(buffers);
  if (device) iree_hal_amdxdna_native_device_c_destroy(device);
  std::printf(
      "{\"schema_version\":1,\"status\":\"fail\",\"phase\":\"%s\","
      "\"status_code\":%d}\n",
      phase, status_code);
  return 1;
}

}  // namespace

int main() {
  iree_hal_amdxdna_device_params params;
  iree_hal_amdxdna_device_options_initialize(&params);
  iree_hal_amdxdna_native_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_device_c_create(
      &params, iree_allocator_system(), &device);
  std::vector<iree_hal_amdxdna_native_buffer_t*> buffers;
  if (!iree_status_is_ok(status)) {
    return Fail("open-device", status, &buffers, device);
  }

  iree_hal_amdxdna_native_c_device_caps_t caps = {};
  status = iree_hal_amdxdna_native_device_c_query_caps(device, &caps);
  if (!iree_status_is_ok(status)) {
    return Fail("query-caps", status, &buffers, device);
  }

  // Native backends without a bounded shared code-memory heap still validate
  // that the instruction-buffer path is usable. This keeps one portable test
  // contract while applying destructive pressure only where the backend
  // advertises a finite reclaimable resource.
  if (caps.max_shared_code_memory_bytes == 0) {
    iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        device, kMissBytes, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
        &buffer);
    if (!iree_status_is_ok(status)) {
      return Fail("instruction-smoke", status, &buffers, device);
    }
    iree_hal_amdxdna_native_buffer_c_destroy(buffer);
    iree_hal_amdxdna_native_device_c_destroy(device);
    std::printf(
        "{\"schema_version\":1,\"status\":\"pass\","
        "\"bounded_shared_code_memory\":false,"
        "\"budget_bytes\":0,\"allocated_bytes\":%" PRIu64 ","
        "\"exhaustion_observed\":false,\"reclaimed_bytes\":0,"
        "\"post_reclaim_allocation_bytes\":0}\n",
        static_cast<uint64_t>(kMissBytes));
    return 0;
  }

  iree_device_size_t allocated_bytes = 0;
  for (;;) {
    iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        device, kChunkBytes, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
        &buffer);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      break;
    }
    if (!iree_status_is_ok(status)) {
      return Fail("fill-shared-code-memory", status, &buffers, device);
    }
    buffers.push_back(buffer);
    allocated_bytes += kChunkBytes;
    if (allocated_bytes >= 2 * caps.max_shared_code_memory_bytes) {
      return Fail("shared-code-memory-not-bounded",
                  iree_make_status(
                      IREE_STATUS_FAILED_PRECONDITION,
                      "shared code-memory allocation exceeded advertised cap"),
                  &buffers, device);
    }
  }

  if (buffers.empty()) {
    return Fail("shared-code-memory-empty",
                iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                 "shared code-memory was already exhausted"),
                &buffers, device);
  }
  if (allocated_bytes < (caps.max_shared_code_memory_bytes / 4) * 3) {
    return Fail(
        "shared-code-memory-premature-exhaustion",
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "less than 75%% of the advertised budget was usable"),
        &buffers, device);
  }

  iree_hal_amdxdna_native_buffer_t* miss_buffer = nullptr;
  status = iree_hal_amdxdna_native_device_c_alloc_buffer(
      device, kMissBytes, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
      &miss_buffer);
  if (!iree_status_is_unavailable(status)) {
    if (iree_status_is_ok(status)) {
      iree_hal_amdxdna_native_buffer_c_destroy(miss_buffer);
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "allocation unexpectedly succeeded at cap");
    }
    return Fail("verify-exhaustion", status, &buffers, device);
  }
  iree_status_ignore(status);

  iree_hal_amdxdna_native_buffer_c_destroy(buffers.back());
  buffers.pop_back();
  status = iree_hal_amdxdna_native_device_c_alloc_buffer(
      device, kMissBytes, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
      &miss_buffer);
  if (!iree_status_is_ok(status)) {
    return Fail("retry-after-reclaim", status, &buffers, device);
  }
  iree_hal_amdxdna_native_buffer_c_destroy(miss_buffer);

  DestroyBuffers(&buffers);
  iree_hal_amdxdna_native_device_c_destroy(device);
  std::printf(
      "{\"schema_version\":1,\"status\":\"pass\","
      "\"bounded_shared_code_memory\":true,"
      "\"budget_bytes\":%" PRIu64 ",\"allocated_bytes\":%" PRIu64 ","
      "\"exhaustion_observed\":true,\"reclaimed_bytes\":%" PRIu64 ","
      "\"post_reclaim_allocation_bytes\":%" PRIu64 "}\n",
      static_cast<uint64_t>(caps.max_shared_code_memory_bytes),
      static_cast<uint64_t>(allocated_bytes), static_cast<uint64_t>(kChunkBytes),
      static_cast<uint64_t>(kMissBytes));
  return 0;
}
