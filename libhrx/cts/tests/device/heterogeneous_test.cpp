// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "hrx_test_fixture.hpp"
#include "iree/hal/driver/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/driver/amd/xdna/image/testdata/mul_i32_npu4.h"

namespace {

struct HeterogeneousResources {
  ~HeterogeneousResources() {
    // Every accepted operation depends only on the gate or an earlier accepted
    // producer. An assertion failure can therefore unblock and drain its tails.
    if (gate && !gate_released) {
      ReportHrxTestFailure("release submission gate",
                           hrx().semaphore_signal(gate, 1));
    }
    for (size_t i = 0; i < semaphores.size(); ++i) {
      if (tails[i]) {
        ReportHrxTestFailure(
            "heterogeneous timeline cleanup",
            hrx().semaphore_wait(semaphores[i], tails[i], UINT64_MAX));
      }
    }
    for (auto buffer : buffers) {
      hrx().buffer_release(buffer);
    }
    for (auto pool : pools) {
      hrx().mem_pool_release(pool);
    }
    hrx().executable_release(executable);
    for (auto semaphore : semaphores) {
      hrx().semaphore_release(semaphore);
    }
    hrx().semaphore_release(gate);
    hrx().device_group_release(group);
    hrx().runtime_release(runtime);
    for (auto infos : device_infos) {
      hrx().host_allocator_free(hrx().host_allocator_system(), infos);
    }
  }

  // Explicit owner of the task executor and native execution services.
  hrx_runtime_t runtime = nullptr;
  // One sealed CPU, GPU, and NPU set, created once for the test process.
  hrx_device_group_t group = nullptr;
  // Passive discovery results retained until group creation has consumed them.
  std::array<hrx_device_info_t*, 3> device_infos = {};
  // Native NPU program with three buffer bindings and no scalar arguments.
  hrx_executable_t executable = nullptr;
  // Source, slab cache, and two independently suballocating siblings.
  std::array<hrx_mem_pool_t, 4> pools = {};
  // Five buffers per generation: two inputs and one output per execution site.
  std::array<hrx_buffer_t, 10> buffers = {};
  // Independent CPU, GPU, and NPU progress timelines.
  std::array<hrx_semaphore_t, 3> semaphores = {};
  // Highest accepted signal value on each timeline.
  std::array<uint64_t, 3> tails = {};
  // Host-controlled prerequisite released after the initial work is enqueued.
  hrx_semaphore_t gate = nullptr;
  // True once the normal path has fulfilled the cleanup-owned prerequisite.
  bool gate_released = false;
};

const iree_file_toc_t* MultiplyImage(std::string_view architecture) {
  if (architecture == "amd.xdna.strix_halo.17f0_11") {
    return iree_hal_amd_xdna_test_mul_i32_create();
  }
  if (architecture == "amd.xdna.strix.17f0_10" ||
      architecture == "amd.xdna.krackan.17f0_20") {
    return iree_hal_amd_xdna_test_mul_i32_npu4_create();
  }
  return nullptr;
}

}  // namespace

TEST_CASE("CPU NPU GPU work shares storage and explicit queue dependencies",
          "[group][heterogeneous]") {
  constexpr size_t kElementCount = 16;
  constexpr size_t kPayloadLength = kElementCount * sizeof(uint32_t);
  constexpr size_t kBufferLength = 2 * kPayloadLength;
  constexpr size_t kOutputOffset = kPayloadLength / 2;
  constexpr uint32_t kGuard = 0xC3C3BEEF;
  HeterogeneousResources resources;
  const hrx_runtime_options_t options = {
      .event_sink = {.fn = ReportHrxDeviceEvent},
  };
  REQUIRE_OK(hrx().runtime_create(&options, hrx().host_allocator_system(),
                                  &resources.runtime));
  const char* schemes[] = {"task", "amdgpu", "xdna"};
  std::array<hrx_device_create_params_t, 3> params = {};
  for (size_t i = 0; i < params.size(); ++i) {
    size_t count = 0;
    REQUIRE_OK(hrx().runtime_query_available_devices(
        resources.runtime, schemes[i], hrx().host_allocator_system(), &count,
        &resources.device_infos[i]));
    REQUIRE(count > 0);
    params[i].uri = resources.device_infos[i][0].uri;
  }
  REQUIRE_OK(hrx().device_group_create(resources.runtime, params.size(),
                                       params.data(), &resources.group));
  std::array<hrx_device_t, 3> devices;
  std::array<hrx_queue_t, 3> queues;
  std::array<hrx_memory_scope_t, 3> scopes;
  std::vector<hrx_pool_family_access_t> families;
  for (size_t i = 0; i < devices.size(); ++i) {
    devices[i] = hrx().device_group_get_device(resources.group, i);
    queues[i] = FindProvisionedQueue(
        devices[i], i == 2 ? HRX_QUEUE_ROLE_DISPATCH : HRX_QUEUE_ROLE_TRANSFER);
    REQUIRE(queues[i] != nullptr);
    const auto family = hrx().queue_get_family_ordinal(queues[i]);
    families.push_back({
        .device = devices[i],
        .family_ordinal = family,
        .usage = i == 2 ? HRX_BUFFER_USAGE_STORAGE : HRX_BUFFER_USAGE_TRANSFER,
    });
    REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
        devices[i], family,
        i == 2 ? HRX_MEMORY_SITE_PROGRAM : HRX_MEMORY_SITE_QUEUE, &scopes[i]));
    REQUIRE_OK(hrx().semaphore_create(devices[i], 0, &resources.semaphores[i]));
  }
  REQUIRE_OK(hrx().semaphore_create(devices[0], 0, &resources.gate));
  char architecture[64] = {};
  REQUIRE_OK(hrx().device_get_property(devices[2],
                                       HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       architecture, sizeof(architecture)));
  const auto* image = MultiplyImage(architecture);
  INFO("NPU architecture: " << architecture);
  REQUIRE(image != nullptr);
  REQUIRE_OK(hrx().executable_load_data(devices[2], image[0].data,
                                        image[0].size, "xdna", architecture,
                                        &resources.executable));
  uint32_t export_ordinal = 0;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(
      resources.executable, "mul_i32", &export_ordinal));

  const hrx_pool_scope_t scope = {
      .family_count = families.size(),
      .families = families.data(),
      .host =
          {
              .access = HRX_MEMORY_ACCESS_READ,
              .modes = HRX_MAPPING_MODE_SCOPED,
              .properties = HRX_HOST_MEMORY_PROPERTY_READ_CACHED,
          },
  };
  REQUIRE_OK(hrx().slab_pool_create(resources.group, scope, nullptr,
                                    &resources.pools[0]));
  const hrx_tlsf_pool_options_t tlsf = {.slab_size = 4096, .alignment = 64};
  hrx_slab_cache_options_t cache = {.max_count = 1};
  REQUIRE_OK(hrx().tlsf_pool_query_backing_request(resources.pools[0], &tlsf,
                                                   &cache.slab));
  REQUIRE_OK(
      hrx().slab_cache_create(resources.pools[0], &cache, &resources.pools[1]));
  for (size_t i = 2; i < resources.pools.size(); ++i) {
    REQUIRE_OK(
        hrx().tlsf_pool_create(resources.pools[1], &tlsf, &resources.pools[i]));
  }
  // Dependent resources retain the complete group and its execution services.
  hrx().device_group_release(resources.group);
  resources.group = nullptr;
  hrx().runtime_release(resources.runtime);
  resources.runtime = nullptr;

  uint64_t original_address = 0;
  for (size_t generation = 0; generation < 2; ++generation) {
    INFO("storage generation: " << generation);
    auto* buffers = resources.buffers.data() + generation * 5;
    std::array<hrx_pool_reservation_request_t, 5> requests;
    requests.fill({kBufferLength, 64});
    const uint64_t gate_value = 1;
    hrx_semaphore_t prerequisites[] = {resources.gate, resources.semaphores[2]};
    uint64_t prerequisite_values[] = {gate_value, resources.tails[2]};
    const hrx_semaphore_list_t allocation_waits = {prerequisites,
                                                   prerequisite_values, 2};
    uint64_t cpu_value = resources.tails[0] + 1;
    hrx_semaphore_list_t cpu_signal = {&resources.semaphores[0], &cpu_value, 1};
    REQUIRE_OK(hrx().queue_alloca(queues[0], &allocation_waits, &cpu_signal,
                                  resources.pools[2 + generation],
                                  requests.size(), requests.data(), buffers));
    resources.tails[0] = cpu_value;
    const uint32_t input_values[] = {
        11u + static_cast<uint32_t>(generation) * 17u,
        7u + static_cast<uint32_t>(generation) * 23u,
        kGuard,
    };
    for (size_t i = 0; i < 3; ++i) {
      const hrx_semaphore_list_t waits = {&resources.semaphores[0],
                                          &resources.tails[0], 1};
      ++cpu_value;
      REQUIRE_OK(hrx().queue_fill(queues[0], &waits, &cpu_signal, buffers[i], 0,
                                  kBufferLength, &input_values[i],
                                  sizeof(input_values[i])));
      resources.tails[0] = cpu_value;
    }
    uint64_t gpu_value = resources.tails[1] + 1;
    hrx_semaphore_list_t gpu_signal = {&resources.semaphores[1], &gpu_value, 1};
    REQUIRE_OK(hrx().queue_fill(queues[1], &cpu_signal, &gpu_signal, buffers[3],
                                0, kBufferLength, &kGuard, sizeof(kGuard)));
    resources.tails[1] = gpu_value;
    const hrx_dispatch_config_t dispatch = {
        .workgroup_count = {1, 1, 1},
        .workgroup_size = {1, 1, 1},
    };
    const hrx_buffer_ref_t bindings[] = {
        {buffers[0], 0, kPayloadLength},
        {buffers[1], 0, kPayloadLength},
        {buffers[2], 0, kPayloadLength},
    };
    uint64_t npu_value = resources.tails[2] + 1;
    const hrx_semaphore_list_t npu_signal = {&resources.semaphores[2],
                                             &npu_value, 1};
    REQUIRE_OK(hrx().queue_dispatch(queues[2], &gpu_signal, &npu_signal,
                                    resources.executable, export_ordinal,
                                    &dispatch, nullptr, 0, bindings, 3, 0));
    resources.tails[2] = npu_value;
    ++gpu_value;
    REQUIRE_OK(hrx().queue_copy(queues[1], &npu_signal, &gpu_signal, buffers[2],
                                0, buffers[3], kOutputOffset, kPayloadLength));
    resources.tails[1] = gpu_value;
    ++cpu_value;
    REQUIRE_OK(hrx().queue_copy(queues[0], &gpu_signal, &cpu_signal, buffers[3],
                                0, buffers[4], 0, kBufferLength));
    resources.tails[0] = cpu_value;
    if (!generation) {
      // All buffers and dependent operations were accepted before allocation
      // could materialize. No host wait lies between these execution sites.
      REQUIRE_OK(hrx().semaphore_signal(resources.gate, gate_value));
      resources.gate_released = true;
    }
    REQUIRE_OK(
        hrx().semaphore_wait(resources.semaphores[0], cpu_value, UINT64_MAX));

    uint64_t address = 0;
    REQUIRE_OK(hrx().buffer_query_address(buffers[0], scopes[0],
                                          HRX_BUFFER_INTERFACE_HOST, &address));
    REQUIRE(address != 0);
    if (!generation) {
      original_address = address;
    }
    REQUIRE(address == original_address);
    REQUIRE_OK(hrx().buffer_query_address(
        buffers[0], scopes[1], HRX_BUFFER_INTERFACE_DEVICE_ADDRESS, &address));
    REQUIRE(address != 0);
    REQUIRE_OK(hrx().buffer_query_address(
        buffers[0], scopes[2], HRX_BUFFER_INTERFACE_XDNA_SHIM_DMA, &address));
    REQUIRE(address != 0);
    void* mapped = nullptr;
    REQUIRE_OK(
        hrx().buffer_map(buffers[4], HRX_MAP_READ, 0, kBufferLength, &mapped));
    const auto* result = static_cast<const uint32_t*>(mapped);
    for (size_t i = 0; i < kBufferLength / sizeof(uint32_t); ++i) {
      const bool payload =
          i >= kOutputOffset / sizeof(uint32_t) &&
          i < (kOutputOffset + kPayloadLength) / sizeof(uint32_t);
      CHECK(result[i] ==
            (payload ? input_values[0] * input_values[1] : kGuard));
    }
    REQUIRE_OK(hrx().buffer_unmap(buffers[4]));
    REQUIRE_OK(
        hrx().buffer_map(buffers[2], HRX_MAP_READ, 0, kBufferLength, &mapped));
    result = static_cast<const uint32_t*>(mapped);
    for (size_t i = 0; i < kBufferLength / sizeof(uint32_t); ++i) {
      CHECK(result[i] ==
            (i < kElementCount ? input_values[0] * input_values[1] : kGuard));
    }
    REQUIRE_OK(hrx().buffer_unmap(buffers[2]));
    hrx_pool_stats_t stats;
    hrx().mem_pool_query_stats(resources.pools[0], &stats);
    REQUIRE(stats.reserve_count == 1);
    REQUIRE(stats.reservation_count == 1);
    hrx().mem_pool_trim(resources.pools[1], 0, HRX_POOL_TRIM_FLAG_ALL);
    hrx().mem_pool_query_stats(resources.pools[0], &stats);
    REQUIRE(stats.reservation_count == 1);

    ++npu_value;
    REQUIRE_OK(
        hrx().queue_dealloca(queues[2], &cpu_signal, &npu_signal, 5, buffers));
    resources.tails[2] = npu_value;
  }
  REQUIRE_OK(hrx().semaphore_wait(resources.semaphores[2], resources.tails[2],
                                  UINT64_MAX));
  hrx().mem_pool_trim(resources.pools[1], 0, HRX_POOL_TRIM_FLAG_ALL);
  hrx_pool_stats_t stats;
  hrx().mem_pool_query_stats(resources.pools[0], &stats);
  REQUIRE(stats.reservation_count == 0);
}
