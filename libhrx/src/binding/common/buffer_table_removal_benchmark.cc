// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "buffer_table.h"
#include "hrx_internal.h"
#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

namespace {

enum class RemovalLayout {
  kDeviceOnly,
  kAdjacentAliases,
  kSeparateAliases,
  kHostRegistration,
};

enum class RemovalOrder {
  kAscendingAddress,
  kDescendingAddress,
  kRandomAddress,
  kReverseShuffledRegistration,
};

struct RemovalBenchmarkConfig {
  // Number of live allocations at the start of each complete drain.
  size_t entry_count;
  // Placement of host aliases relative to the device-address ranges.
  RemovalLayout layout;
  // Address or registration order used to drain the table.
  RemovalOrder order;
};

// Each benchmark iteration accounts for one removed allocation. Registration,
// capacity growth, and fixture construction are excluded from the timed batch.
// Public unregister rows include reservation cancellation and wrapper release.
static iree_status_t RunRemovalBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const auto& config =
      *static_cast<const RemovalBenchmarkConfig*>(benchmark_def->user_data);
  const size_t count = config.entry_count;
  const bool host_registration =
      config.layout == RemovalLayout::kHostRegistration;
  constexpr size_t kAllocationSize = 16;
  constexpr size_t kHostStride = 64;

  std::vector<size_t> insertion_order(count);
  std::iota(insertion_order.begin(), insertion_order.end(), 0);
  std::mt19937 random(0x2865);
  if (config.order == RemovalOrder::kReverseShuffledRegistration) {
    std::shuffle(insertion_order.begin(), insertion_order.end(), random);
  }
  auto removal_order = insertion_order;
  if (config.order == RemovalOrder::kDescendingAddress ||
      config.order == RemovalOrder::kReverseShuffledRegistration) {
    std::reverse(removal_order.begin(), removal_order.end());
  } else if (config.order == RemovalOrder::kRandomAddress) {
    std::shuffle(removal_order.begin(), removal_order.end(), random);
  }

  std::vector<uint8_t> host_storage(host_registration ? count * kHostStride
                                                      : 0);
  std::vector<uint64_t> device_addresses(count);
  std::vector<void*> host_aliases(count, nullptr);
  std::vector<hrx_buffer_t> buffers(count, nullptr);
  for (size_t index = 0; index < count; ++index) {
    device_addresses[index] = UINT64_C(0x10000000) + index * 128 + 64;
    if (config.layout == RemovalLayout::kAdjacentAliases) {
      host_aliases[index] =
          reinterpret_cast<void*>(device_addresses[index] - 64);
    } else if (config.layout == RemovalLayout::kSeparateAliases) {
      host_aliases[index] =
          reinterpret_cast<void*>(UINT64_C(0x30000000) + index * 128);
    }
    if (!host_registration) {
      IREE_CHECK_OK(hrx_buffer_create_from_hal(
          /*hal_buffer=*/nullptr, /*device=*/nullptr,
          HRX_MEMORY_TYPE_HOST_LOCAL, kAllocationSize,
          reinterpret_cast<void*>(device_addresses[index]), &buffers[index]));
    }
  }

  hrx_device_t device = nullptr;
  if (host_registration) {
    IREE_CHECK_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_CHECK_OK(hrx_status_to_iree(hrx_cpu_device_get(0, &device)));
  }
  hrx_buffer_table_t table;
  hrx_buffer_table_initialize(&table);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    for (size_t index : insertion_order) {
      if (host_registration) {
        IREE_CHECK_OK(hrx_status_to_iree(hrx_host_memory_register(
            device, host_storage.data() + index * kHostStride, kAllocationSize,
            /*flags=*/0)));
      } else {
        IREE_CHECK_OK(hrx_status_to_iree(hrx_buffer_table_insert(
            &table, device_addresses[index], host_aliases[index],
            kAllocationSize, buffers[index], /*user_data=*/nullptr)));
      }
    }
    iree_benchmark_resume_timing(benchmark_state);

    if (host_registration) {
      for (size_t index : removal_order) {
        IREE_CHECK_OK(hrx_status_to_iree(hrx_host_memory_unregister(
            device, host_storage.data() + index * kHostStride)));
      }
    } else {
      for (size_t index : removal_order) {
        IREE_CHECK_OK(hrx_status_to_iree(
            hrx_buffer_table_remove(&table, device_addresses[index])));
      }
    }
  }

  hrx_buffer_table_deinitialize(&table);
  for (hrx_buffer_t buffer : buffers) {
    hrx_buffer_release(buffer);
  }
  if (host_registration) {
    IREE_CHECK_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }
  return iree_ok_status();
}

static bool RegisterRemovalBenchmarks() {
  const struct {
    // Benchmark path describing the operation and alias placement.
    const char* name;
    // Layout selected by the corresponding benchmark path.
    RemovalLayout layout;
  } layouts[] = {
      {"BufferTableRemove/DeviceOnly", RemovalLayout::kDeviceOnly},
      {"BufferTableRemove/AdjacentAliases", RemovalLayout::kAdjacentAliases},
      {"BufferTableRemove/SeparateAliases", RemovalLayout::kSeparateAliases},
      {"HostMemoryUnregister/EqualAliases", RemovalLayout::kHostRegistration},
  };
  const struct {
    // Benchmark path identifying the actual address/registration order.
    const char* name;
    // Order selected by the corresponding benchmark path.
    RemovalOrder order;
  } orders[] = {
      {"AscendingAddress", RemovalOrder::kAscendingAddress},
      {"DescendingAddress", RemovalOrder::kDescendingAddress},
      {"RandomAddress", RemovalOrder::kRandomAddress},
      {"ReverseShuffledRegistration",
       RemovalOrder::kReverseShuffledRegistration},
  };
  constexpr size_t counts[] = {16, 256, 1000, 10000};
  // Registered definitions borrow these descriptors for the process lifetime.
  static std::array<RemovalBenchmarkConfig, IREE_ARRAYSIZE(layouts) *
                                                IREE_ARRAYSIZE(orders) *
                                                IREE_ARRAYSIZE(counts)>
      configs;
  size_t config_index = 0;
  for (size_t count : counts) {
    for (const auto& layout : layouts) {
      for (const auto& order : orders) {
        auto& config = configs[config_index++];
        config = {count, layout.layout, order.order};
        const std::string name = std::string(layout.name) + "/" + order.name +
                                 "/" + std::to_string(count);
        // Bound setup work while giving short rows enough measured removals.
        const size_t batches =
            std::max(size_t{8}, (size_t{32768} + count - 1) / count);
        const iree_benchmark_def_t definition = {
            .flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME,
            .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
            .minimum_duration_ns = 0,
            .iteration_count = count * batches,
            .run = RunRemovalBenchmark,
            .user_data = &config,
        };
        iree_benchmark_register(iree_make_cstring_view(name.c_str()),
                                &definition);
      }
    }
  }
  return true;
}

static const bool registered IREE_ATTRIBUTE_UNUSED =
    RegisterRemovalBenchmarks();

}  // namespace
