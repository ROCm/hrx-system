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
  // Caller-owned task and GPU discovery results, independent of the group.
  std::array<hrx_device_info_t*, 2> device_infos = {};
  // One sealed producer/consumer group, created once for this test process.
  hrx_device_group_t group = nullptr;
  // Source, consumer-written intermediate, and producer output in each reuse
  // generation.
  std::array<hrx_buffer_t, 6> buffers = {};
  // Producer and consumer timeline semaphores.
  std::array<hrx_semaphore_t, 2> semaphores = {};
  // Last successfully accepted signal on each timeline.
  std::array<uint64_t, 2> tails = {};
  // Shared native source, idle cache, and two independently suballocating
  // pools.
  std::array<hrx_mem_pool_t, 4> pools = {};
};

}  // namespace

TEST_CASE("Explicit group pools share backing across queue timelines",
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
  const char* producer_uri = std::getenv("HRX_CTS_GROUP_PRODUCER");
  const char* consumer_uri = std::getenv("HRX_CTS_GROUP_CONSUMER");
  const hrx_device_create_params_t devices[] = {
      {
          .uri = producer_uri && producer_uri[0]
                     ? producer_uri
                     : resources.device_infos[0][0].uri,
      },
      {
          .uri = consumer_uri && consumer_uri[0]
                     ? consumer_uri
                     : resources.device_infos[1][0].uri,
      },
  };
  INFO("producer=" << devices[0].uri << " consumer=" << devices[1].uri);
  REQUIRE_OK(hrx().device_group_create(resources.runtime, 2, devices,
                                       &resources.group));
  for (auto& infos : resources.device_infos) {
    hrx().host_allocator_free(hrx().host_allocator_system(), infos);
    infos = nullptr;
  }
  REQUIRE(hrx().device_group_device_count(resources.group) == 2);
  hrx_device_t producer = hrx().device_group_get_device(resources.group, 0);
  hrx_device_t consumer = hrx().device_group_get_device(resources.group, 1);
  hrx_queue_t producer_queue =
      FindProvisionedQueue(producer, HRX_QUEUE_ROLE_TRANSFER);
  hrx_queue_t consumer_queue =
      FindProvisionedQueue(consumer, HRX_QUEUE_ROLE_TRANSFER);
  REQUIRE_OK(hrx().semaphore_create(producer, 0, &resources.semaphores[0]));
  REQUIRE_OK(hrx().semaphore_create(consumer, 0, &resources.semaphores[1]));

  std::vector<hrx_pool_family_access_t> families = {
      {
          .device = producer,
          .family_ordinal = hrx().queue_get_family_ordinal(producer_queue),
          .usage = HRX_BUFFER_USAGE_DEFAULT,
      },
      {
          .device = consumer,
          .family_ordinal = hrx().queue_get_family_ordinal(consumer_queue),
          .usage = HRX_BUFFER_USAGE_DEFAULT,
      },
  };
  const hrx_device_t participants[] = {producer, consumer};
  std::array<uint32_t, 2> program_families;
  for (size_t i = 0; i < program_families.size(); ++i) {
    program_families[i] = hrx().queue_get_family_ordinal(
        FindProvisionedQueue(participants[i], HRX_QUEUE_ROLE_DISPATCH));
    if (program_families[i] != families[i].family_ordinal) {
      families.push_back({
          .device = participants[i],
          .family_ordinal = program_families[i],
          .usage = HRX_BUFFER_USAGE_DEFAULT,
      });
    }
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
  hrx_memory_scope_t producer_scope, consumer_scope;
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      producer, program_families[0], HRX_MEMORY_SITE_PROGRAM, &producer_scope));
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      consumer, program_families[1], HRX_MEMORY_SITE_PROGRAM, &consumer_scope));
  hrx_memory_scope_t host_scope, consumer_queue_scope;
  REQUIRE_OK(hrx().device_query_host_memory_scope(producer, &host_scope));
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      consumer, families[1].family_ordinal, HRX_MEMORY_SITE_QUEUE,
      &consumer_queue_scope));
  const auto transition_table =
      hrx().mem_pool_transition_table(resources.pools[0]);
  hrx_memory_transition_pair_t upload_pair, download_pair;
  REQUIRE_OK(hrx().memory_transition_prepare_pair(
      transition_table, host_scope, consumer_queue_scope,
      HRX_MEMORY_TRANSITION_RELEASE, &upload_pair));
  REQUIRE_OK(hrx().memory_transition_prepare_pair(
      transition_table, consumer_queue_scope, host_scope,
      HRX_MEMORY_TRANSITION_ACQUIRE, &download_pair));
  hrx().device_group_release(resources.group);
  resources.group = nullptr;
  hrx().runtime_release(resources.runtime);
  resources.runtime = nullptr;

  hrx_accelerator_type_t producer_type;
  REQUIRE_OK(hrx().device_get_type(producer, &producer_type));
  const hrx_buffer_interface_t producer_interface =
      producer_type == HRX_ACCELERATOR_CPU
          ? HRX_BUFFER_INTERFACE_HOST
          : HRX_BUFFER_INTERFACE_DEVICE_ADDRESS;
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
    REQUIRE_OK(hrx().queue_alloca(producer_queue, &previous, &allocation_signal,
                                  resources.pools[2 + generation], 3, requests,
                                  buffers));
    resources.tails[0] = allocated;
    for (size_t i = 0; i < kRegionCount; ++i) {
      const size_t offset = i * kRegionLength;
      const uint32_t pattern =
          0x10325476u ^
          static_cast<uint32_t>((i + generation * kRegionCount) * 0x01010101u);
      uint64_t produced = resources.tails[0] + 1;
      const hrx_semaphore_list_t producer_wait = {&resources.semaphores[0],
                                                  &resources.tails[0], 1};
      const hrx_semaphore_list_t producer_signal = {&resources.semaphores[0],
                                                    &produced, 1};
      REQUIRE_OK(hrx().queue_fill(
          producer_queue, &producer_wait, &producer_signal, buffers[0], offset,
          kRegionLength, &pattern, sizeof(pattern), nullptr));
      resources.tails[0] = produced;
      uint64_t copied = resources.tails[1] + 1;
      uint64_t consumer_wait_values[] = {produced, resources.tails[1]};
      const hrx_semaphore_list_t consumer_wait = {resources.semaphores.data(),
                                                  consumer_wait_values, 2};
      const hrx_semaphore_list_t consumer_signal = {&resources.semaphores[1],
                                                    &copied, 1};
      REQUIRE_OK(hrx().queue_copy(consumer_queue, &consumer_wait,
                                  &consumer_signal, buffers[0], offset,
                                  buffers[1], offset, kRegionLength, nullptr));
      resources.tails[1] = copied;
    }
    const hrx_semaphore_list_t consumer_complete = {&resources.semaphores[1],
                                                    &resources.tails[1], 1};
    uint64_t consumed = resources.tails[0] + 1;
    const hrx_semaphore_list_t producer_signal = {&resources.semaphores[0],
                                                  &consumed, 1};
    REQUIRE_OK(hrx().queue_copy(producer_queue, &consumer_complete,
                                &producer_signal, buffers[1], 0, buffers[2], 0,
                                kLength, nullptr));
    resources.tails[0] = consumed;
    REQUIRE_OK(
        hrx().semaphore_wait(resources.semaphores[0], consumed, UINT64_MAX));

    uint64_t address = 0;
    REQUIRE_OK(hrx().buffer_query_address(buffers[0], host_scope,
                                          HRX_BUFFER_INTERFACE_HOST, &address));
    REQUIRE(address != 0);
    if (!generation) {
      first_address = address;
    } else {
      REQUIRE(address == first_address);
    }
    REQUIRE_OK(hrx().buffer_query_address(buffers[0], producer_scope,
                                          producer_interface, &address));
    REQUIRE(address != 0);
    REQUIRE_OK(hrx().buffer_query_address(buffers[0], consumer_scope,
                                          HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                          &address));
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
    // Keep both mappings active across the consumer copy. Explicit maintenance,
    // rather than unmap/remap side effects, publishes and acquires these bytes.
    const auto table = hrx().buffer_transition_table(buffers[0]);
    const auto upload = hrx_memory_transition_query(table, upload_pair);
    const auto download = hrx_memory_transition_query(table, download_pair);
    REQUIRE(hrx_memory_effects_is_supported(upload.release));
    REQUIRE(hrx_memory_effects_is_supported(download.acquire));
    void* input = nullptr;
    REQUIRE_OK(hrx().buffer_map(buffers[0], HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS,
                                64, 256, &input));
    REQUIRE_OK(hrx().buffer_map(buffers[2], HRX_MAP_READ | HRX_MAP_MAY_ALIAS,
                                128, 256, &output));
    const uint32_t host_pattern =
        0x76543210u + static_cast<uint32_t>(generation);
    for (size_t i = 0; i < 256 / sizeof(uint32_t); ++i) {
      static_cast<uint32_t*>(input)[i] = host_pattern;
    }
    const hrx_buffer_mapping_transition_t upload_mapping = {
        .buffer = buffers[0],
        .length = 256,
        .recipe = hrx().memory_transition_recipe(table, upload_pair,
                                                 HRX_MEMORY_TRANSITION_RELEASE),
    };
    REQUIRE_OK(hrx().buffer_mapping_memory_barrier(
        upload.release, upload_mapping.recipe ? 1 : 0, &upload_mapping));
    uint64_t uploaded = resources.tails[1] + 1;
    const hrx_semaphore_list_t upload_signal = {&resources.semaphores[1],
                                                &uploaded, 1};
    REQUIRE_OK(hrx().queue_copy(consumer_queue, &producer_signal,
                                &upload_signal, buffers[0], 64, buffers[2], 128,
                                256, nullptr));
    resources.tails[1] = uploaded;
    REQUIRE_OK(
        hrx().semaphore_wait(resources.semaphores[1], uploaded, UINT64_MAX));
    const hrx_buffer_mapping_transition_t download_mapping = {
        .buffer = buffers[2],
        .length = 256,
        .recipe = hrx().memory_transition_recipe(table, download_pair,
                                                 HRX_MEMORY_TRANSITION_ACQUIRE),
    };
    REQUIRE_OK(hrx().buffer_mapping_memory_barrier(
        download.acquire, download_mapping.recipe ? 1 : 0, &download_mapping));
    for (size_t i = 0; i < 256 / sizeof(uint32_t); ++i) {
      CHECK(static_cast<const uint32_t*>(output)[i] == host_pattern);
    }
    REQUIRE_OK(hrx().buffer_unmap(buffers[2]));
    REQUIRE_OK(hrx().buffer_unmap(buffers[0]));
    uint64_t released = consumed + 1;
    const hrx_semaphore_list_t release_signal = {&resources.semaphores[0],
                                                 &released, 1};
    REQUIRE_OK(hrx().queue_dealloca(producer_queue, &upload_signal,
                                    &release_signal, 3, buffers));
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
