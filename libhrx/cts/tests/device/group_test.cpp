// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "hrx_test_fixture.hpp"

namespace {

// Owns accepted timeline tails as well as handles, so an assertion failure
// still retires work before releasing buffers, pools, or execution owners.
struct GroupResources {
  ~GroupResources() {
    for (size_t i = 0; i < semaphores.size(); ++i) {
      if (tails[i]) {
        ReportHrxTestFailure(
            "group timeline cleanup",
            hrx().semaphore_wait(semaphores[i], tails[i], UINT64_MAX));
      }
    }
    for (auto buffer : buffers) {
      hrx().buffer_release(buffer);
    }
    for (auto pool : pools) {
      hrx().mem_pool_release(pool);
    }
    for (auto semaphore : semaphores) {
      hrx().semaphore_release(semaphore);
    }
    hrx().device_group_release(group);
    hrx().runtime_release(runtime);
    for (auto infos : device_infos) {
      hrx().host_allocator_free(hrx().host_allocator_system(), infos);
    }
  }
  // Explicit execution-service owner.
  hrx_runtime_t runtime = nullptr;
  // Caller-owned CPU and GPU discovery results, independent of the group.
  std::array<hrx_device_info_t*, 2> device_infos = {};
  // One sealed CPU+GPU group, created once for this test process.
  hrx_device_group_t group = nullptr;
  // Source, GPU-written intermediate, and CPU output in each reuse generation.
  std::array<hrx_buffer_t, 6> buffers = {};
  // CPU and GPU timeline semaphores.
  std::array<hrx_semaphore_t, 2> semaphores = {};
  // Last successfully accepted signal on each timeline.
  std::array<uint64_t, 2> tails = {};
  // Shared native source, idle cache, and two independently suballocating
  // pools.
  std::array<hrx_mem_pool_t, 4> pools = {};
};

}  // namespace

TEST_CASE("Explicit CPU GPU pools share backing across queue timelines",
          "[group]") {
  constexpr size_t kRegionLength = 4096;
  constexpr size_t kRegionCount = 8;
  constexpr size_t kLength = kRegionLength * kRegionCount;
  GroupResources resources;
  const hrx_runtime_options_t options = {
      .event_sink = {.fn = ReportHrxDeviceEvent},
  };
  REQUIRE_OK(hrx().runtime_create(&options, hrx().host_allocator_system(),
                                  &resources.runtime));
  const char* driver = std::getenv("HRX_GPU_DRIVER");
  size_t cpu_count = 0;
  size_t gpu_count = 0;
  REQUIRE_OK(hrx().runtime_query_available_devices(
      resources.runtime, "task", hrx().host_allocator_system(), &cpu_count,
      &resources.device_infos[0]));
  REQUIRE_OK(hrx().runtime_query_available_devices(
      resources.runtime, driver && driver[0] ? driver : "amdgpu",
      hrx().host_allocator_system(), &gpu_count, &resources.device_infos[1]));
  REQUIRE(cpu_count == 1);
  REQUIRE(gpu_count > 0);
  const hrx_device_create_params_t devices[] = {
      {.uri = resources.device_infos[0][0].uri},
      {.uri = resources.device_infos[1][0].uri},
  };
  REQUIRE_OK(hrx().device_group_create(resources.runtime, 2, devices,
                                       &resources.group));
  for (auto& infos : resources.device_infos) {
    hrx().host_allocator_free(hrx().host_allocator_system(), infos);
    infos = nullptr;
  }
  REQUIRE(hrx().device_group_device_count(resources.group) == 2);
  hrx_device_t cpu = hrx().device_group_get_device(resources.group, 0);
  hrx_device_t gpu = hrx().device_group_get_device(resources.group, 1);
  hrx_queue_t cpu_queue = FindProvisionedQueue(cpu, HRX_QUEUE_ROLE_TRANSFER);
  hrx_queue_t gpu_queue = FindProvisionedQueue(gpu, HRX_QUEUE_ROLE_TRANSFER);
  REQUIRE_OK(hrx().semaphore_create(cpu, 0, &resources.semaphores[0]));
  REQUIRE_OK(hrx().semaphore_create(gpu, 0, &resources.semaphores[1]));

  std::vector<hrx_pool_family_access_t> families = {
      {
          .device = cpu,
          .family_ordinal = hrx().queue_get_family_ordinal(cpu_queue),
          .usage = HRX_BUFFER_USAGE_DEFAULT,
      },
      {
          .device = gpu,
          .family_ordinal = hrx().queue_get_family_ordinal(gpu_queue),
          .usage = HRX_BUFFER_USAGE_DEFAULT,
      },
  };
  const uint32_t program_family = hrx().queue_get_family_ordinal(
      FindProvisionedQueue(gpu, HRX_QUEUE_ROLE_DISPATCH));
  if (program_family != families[1].family_ordinal) {
    families.push_back({
        .device = gpu,
        .family_ordinal = program_family,
        .usage = HRX_BUFFER_USAGE_DEFAULT,
    });
  }
  const hrx_pool_scope_t pool_scope = {
      .family_count = families.size(),
      .families = families.data(),
      .host =
          {
              .access = HRX_MEMORY_ACCESS_ALL,
              .modes = HRX_MAPPING_MODE_SCOPED,
              .properties = HRX_HOST_MEMORY_PROPERTY_READ_CACHED |
                            HRX_HOST_MEMORY_PROPERTY_WRITE_BACK,
          },
  };
  // Shared access and CPU mapping guarantees qualify one native route before
  // any payload allocation. No host NUMA node is assumed on platforms that do
  // not publish CPU locality in their device topology.
  REQUIRE_OK(hrx().slab_pool_create(resources.group, pool_scope, nullptr,
                                    &resources.pools[0]));
  const hrx_pool_host_access_t host =
      hrx().mem_pool_query_host_access(resources.pools[0]);
  REQUIRE(host.access == HRX_MEMORY_ACCESS_ALL);
  REQUIRE((host.properties & pool_scope.host.properties) ==
          pool_scope.host.properties);
  const hrx_tlsf_pool_options_t tlsf = {
      .slab_size = 4 * kLength,
      .alignment = 64,
  };
  hrx_slab_cache_options_t cache = {.max_count = 1};
  REQUIRE_OK(hrx().tlsf_pool_query_backing_request(resources.pools[0], &tlsf,
                                                   &cache.slab));
  REQUIRE_OK(
      hrx().slab_cache_create(resources.pools[0], &cache, &resources.pools[1]));
  for (size_t i = 2; i < resources.pools.size(); ++i) {
    REQUIRE_OK(
        hrx().tlsf_pool_create(resources.pools[1], &tlsf, &resources.pools[i]));
  }
  hrx_memory_scope_t cpu_scope, gpu_scope;
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      cpu, families[0].family_ordinal, HRX_MEMORY_SITE_PROGRAM, &cpu_scope));
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      gpu, program_family, HRX_MEMORY_SITE_PROGRAM, &gpu_scope));
  hrx().device_group_release(resources.group);
  resources.group = nullptr;
  hrx().runtime_release(resources.runtime);
  resources.runtime = nullptr;

  uint64_t first_address = 0;
  for (size_t generation = 0; generation < 2; ++generation) {
    hrx_buffer_t* buffers = resources.buffers.data() + 3 * generation;
    const hrx_pool_reservation_request_t requests[] = {
        {kLength, 64},
        {kLength, 64},
        {kLength, 64},
    };
    const hrx_semaphore_list_t previous = {&resources.semaphores[0],
                                           &resources.tails[0], 1};
    uint64_t allocated = resources.tails[0] + 1;
    const hrx_semaphore_list_t allocation_signal = {&resources.semaphores[0],
                                                    &allocated, 1};
    REQUIRE_OK(hrx().queue_alloca(cpu_queue, &previous, &allocation_signal,
                                  resources.pools[2 + generation], 3, requests,
                                  buffers));
    resources.tails[0] = allocated;
    for (size_t i = 0; i < kRegionCount; ++i) {
      const size_t offset = i * kRegionLength;
      const uint32_t pattern =
          0x10325476u ^
          static_cast<uint32_t>((i + generation * kRegionCount) * 0x01010101u);
      uint64_t produced = resources.tails[0] + 1;
      const hrx_semaphore_list_t cpu_wait = {&resources.semaphores[0],
                                             &resources.tails[0], 1};
      const hrx_semaphore_list_t cpu_signal = {&resources.semaphores[0],
                                               &produced, 1};
      REQUIRE_OK(hrx().queue_fill(cpu_queue, &cpu_wait, &cpu_signal, buffers[0],
                                  offset, kRegionLength, &pattern,
                                  sizeof(pattern)));
      resources.tails[0] = produced;
      uint64_t copied = resources.tails[1] + 1;
      uint64_t gpu_wait_values[] = {produced, resources.tails[1]};
      const hrx_semaphore_list_t gpu_wait = {resources.semaphores.data(),
                                             gpu_wait_values, 2};
      const hrx_semaphore_list_t gpu_signal = {&resources.semaphores[1],
                                               &copied, 1};
      REQUIRE_OK(hrx().queue_copy(gpu_queue, &gpu_wait, &gpu_signal, buffers[0],
                                  offset, buffers[1], offset, kRegionLength));
      resources.tails[1] = copied;
    }
    const hrx_semaphore_list_t gpu_complete = {&resources.semaphores[1],
                                               &resources.tails[1], 1};
    uint64_t consumed = resources.tails[0] + 1;
    const hrx_semaphore_list_t cpu_signal = {&resources.semaphores[0],
                                             &consumed, 1};
    REQUIRE_OK(hrx().queue_copy(cpu_queue, &gpu_complete, &cpu_signal,
                                buffers[1], 0, buffers[2], 0, kLength));
    resources.tails[0] = consumed;
    REQUIRE_OK(
        hrx().semaphore_wait(resources.semaphores[0], consumed, UINT64_MAX));

    uint64_t address = 0;
    REQUIRE_OK(hrx().buffer_query_address(buffers[0], cpu_scope,
                                          HRX_BUFFER_INTERFACE_HOST, &address));
    REQUIRE(address != 0);
    if (!generation) {
      first_address = address;
    } else {
      REQUIRE(address == first_address);
    }
    REQUIRE_OK(hrx().buffer_query_address(
        buffers[0], gpu_scope, HRX_BUFFER_INTERFACE_DEVICE_ADDRESS, &address));
    REQUIRE(address != 0);
    hrx_pool_stats_t stats;
    hrx().mem_pool_query_stats(resources.pools[0], &stats);
    REQUIRE(stats.reserve_count == 1);
    REQUIRE(stats.reservation_count == 1);
    // Trimming the empty sibling and shared cache cannot reclaim live backing.
    hrx().mem_pool_trim(resources.pools[2 + (1 - generation)], 0,
                        HRX_POOL_TRIM_FLAG_ALL);
    hrx().mem_pool_trim(resources.pools[1], 0, HRX_POOL_TRIM_FLAG_ALL);
    void* output = nullptr;
    REQUIRE_OK(hrx().buffer_map(buffers[2], HRX_MAP_READ, 0, kLength, &output));
    for (size_t i = 0; i < kLength / sizeof(uint32_t); ++i) {
      const size_t region = i * sizeof(uint32_t) / kRegionLength;
      CHECK(static_cast<const uint32_t*>(output)[i] ==
            (0x10325476u ^
             static_cast<uint32_t>((region + generation * kRegionCount) *
                                   0x01010101u)));
    }
    REQUIRE_OK(hrx().buffer_unmap(buffers[2]));
    uint64_t released = consumed + 1;
    const hrx_semaphore_list_t release_signal = {&resources.semaphores[0],
                                                 &released, 1};
    REQUIRE_OK(hrx().queue_dealloca(cpu_queue, &cpu_signal, &release_signal, 3,
                                    buffers));
    resources.tails[0] = released;
    // The next sibling allocation consumes this frontier without a host join.
  }
  REQUIRE_OK(hrx().semaphore_wait(resources.semaphores[0], resources.tails[0],
                                  UINT64_MAX));
  hrx().mem_pool_trim(resources.pools[1], 0, HRX_POOL_TRIM_FLAG_ALL);
  hrx_pool_stats_t stats;
  hrx().mem_pool_query_stats(resources.pools[1], &stats);
  REQUIRE(stats.bytes_committed == 0);
  hrx().mem_pool_query_stats(resources.pools[0], &stats);
  REQUIRE(stats.reservation_count == 0);
}
