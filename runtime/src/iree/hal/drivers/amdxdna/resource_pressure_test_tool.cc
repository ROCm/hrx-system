// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <vector>

#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

namespace {

constexpr iree_device_size_t kChunkBytes = 1u << 20;
constexpr iree_device_size_t kMissBytes = 33u << 10;

// Retained single and chain commands carry 36-44 KiB of control code each.
constexpr iree_device_size_t kRetainedCommandBytes = 40u << 10;
// Buffers created between commands that outlive the retained ones and land in
// the holes eviction leaves behind.
constexpr iree_device_size_t kLongLivedBytes = 40u << 10;
constexpr size_t kLongLivedCount = 64;
constexpr iree_device_size_t kMinChurnBytes = 16u << 20;

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
      "{\"schema_version\":2,\"status\":\"fail\",\"phase\":\"%s\","
      "\"status_code\":%d}\n",
      phase, status_code);
  return 1;
}

iree_status_t AllocInstruction(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    std::vector<iree_hal_amdxdna_native_buffer_t*>* buffers) {
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_c_alloc_buffer(
      device, size, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &buffer));
  buffers->push_back(buffer);
  return iree_ok_status();
}

// Replays the placement pattern of a command cache held at the advertised
// retention ceiling: commands are created and the oldest evicted while
// long-lived buffers are created in between, then one large control program
// must still fit. Retention beyond the ceiling spreads live code across the
// shared heap and leaves no range large enough.
iree_status_t RunFragmentedLargeInstruction(
    iree_hal_amdxdna_native_device_t* device,
    iree_device_size_t retained_ceiling_bytes, iree_device_size_t churned_bytes,
    iree_device_size_t large_bytes) {
  std::vector<iree_hal_amdxdna_native_buffer_t*> long_lived;
  std::deque<iree_hal_amdxdna_native_buffer_t*> retained;
  iree_device_size_t retained_bytes = 0;
  const size_t command_count =
      static_cast<size_t>(churned_bytes / kRetainedCommandBytes);
  const size_t long_lived_every =
      std::max<size_t>(1, command_count / kLongLivedCount);
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < command_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdxdna_native_buffer_t* command = nullptr;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        device, kRetainedCommandBytes,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &command);
    if (!iree_status_is_ok(status)) break;
    retained.push_back(command);
    retained_bytes += kRetainedCommandBytes;
    while (retained_bytes > retained_ceiling_bytes) {
      iree_hal_amdxdna_native_buffer_c_destroy(retained.front());
      retained.pop_front();
      retained_bytes -= kRetainedCommandBytes;
    }
    if (i % long_lived_every == 0 && long_lived.size() < kLongLivedCount) {
      status = AllocInstruction(device, kLongLivedBytes, &long_lived);
    }
  }
  // The device reclaims every idle command before failing an allocation, so
  // only the long-lived buffers retention scattered can block the program.
  for (iree_hal_amdxdna_native_buffer_t* command : retained) {
    iree_hal_amdxdna_native_buffer_c_destroy(command);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_buffer_t* large = nullptr;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        device, large_bytes, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
        &large);
    if (iree_status_is_ok(status)) {
      iree_hal_amdxdna_native_buffer_c_destroy(large);
    }
  }
  DestroyBuffers(&long_lived);
  return status;
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
        "{\"schema_version\":2,\"status\":\"pass\","
        "\"bounded_shared_code_memory\":false,"
        "\"budget_bytes\":0,\"allocated_bytes\":%" PRIu64
        ",\"exhaustion_observed\":false,\"reclaimed_bytes\":0,"
        "\"post_reclaim_allocation_bytes\":0,"
        "\"retained_command_code_bytes\":0,\"churned_command_code_bytes\":0,"
        "\"fragmented_large_allocation_bytes\":0}\n",
        static_cast<uint64_t>(kMissBytes));
    return 0;
  }

  // A bounded shared heap needs a retention ceiling; without one, idle
  // command code is limited only by the retained-cache budget.
  const iree_device_size_t retained_ceiling_bytes =
      caps.max_retained_command_code_bytes;
  if (retained_ceiling_bytes == 0) {
    return Fail("retained-command-code-unbounded",
                iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                 "bounded shared code memory advertises no "
                                 "retained command-code ceiling"),
                &buffers, device);
  }
  const iree_device_size_t churned_bytes =
      std::max(4 * retained_ceiling_bytes, kMinChurnBytes);
  const iree_device_size_t large_bytes =
      (caps.max_shared_code_memory_bytes / 4) * 3;
  status = RunFragmentedLargeInstruction(device, retained_ceiling_bytes,
                                         churned_bytes, large_bytes);
  if (!iree_status_is_ok(status)) {
    return Fail("fragmented-large-instruction", status, &buffers, device);
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
      "{\"schema_version\":2,\"status\":\"pass\","
      "\"bounded_shared_code_memory\":true,"
      "\"budget_bytes\":%" PRIu64 ",\"allocated_bytes\":%" PRIu64
      ",\"exhaustion_observed\":true,\"reclaimed_bytes\":%" PRIu64
      ",\"post_reclaim_allocation_bytes\":%" PRIu64
      ",\"retained_command_code_bytes\":%" PRIu64
      ",\"churned_command_code_bytes\":%" PRIu64
      ",\"fragmented_large_allocation_bytes\":%" PRIu64 "}\n",
      static_cast<uint64_t>(caps.max_shared_code_memory_bytes),
      static_cast<uint64_t>(allocated_bytes),
      static_cast<uint64_t>(kChunkBytes), static_cast<uint64_t>(kMissBytes),
      static_cast<uint64_t>(retained_ceiling_bytes),
      static_cast<uint64_t>(churned_bytes), static_cast<uint64_t>(large_bytes));
  return 0;
}
