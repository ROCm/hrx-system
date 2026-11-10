// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "hrx_test_fixture.hpp"

namespace {

// Owns accepted timeline tails as well as handles, so an assertion failure
// still retires work before releasing its imported host allocation.
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
    for (auto semaphore : semaphores) {
      hrx().semaphore_release(semaphore);
    }
    hrx().device_group_release(group);
    hrx().runtime_release(runtime);
    hrx().host_allocator_free_aligned(hrx().host_allocator_system(),
                                      host_storage);
  }
  // Explicit execution-service owner.
  hrx_runtime_t runtime = nullptr;
  // One sealed CPU+GPU group, created once for this test process.
  hrx_device_group_t group = nullptr;
  // Imported CPU/GPU aliases, GPU intermediate, and CPU output.
  std::array<hrx_buffer_t, 6> buffers = {};
  // CPU and GPU timeline semaphores.
  std::array<hrx_semaphore_t, 2> semaphores = {};
  // Last successfully accepted signal on each timeline.
  std::array<uint64_t, 2> tails = {};
  // Caller-owned source/readback pages, kept alive through both imports.
  void* host_storage = nullptr;
};

}  // namespace

TEST_CASE("Explicit CPU GPU group carries data across queue timelines",
          "[group]") {
  constexpr size_t kRegionLength = 4096;
  constexpr size_t kRegionCount = 8;
  constexpr size_t kLength = kRegionLength * kRegionCount;
  GroupResources resources;
  REQUIRE_OK(hrx().runtime_create(nullptr, hrx().host_allocator_system(),
                                  &resources.runtime));
  const char* driver = std::getenv("HRX_GPU_DRIVER");
  const std::string gpu_uri =
      std::string(driver && driver[0] ? driver : "amdgpu") + "://0";
  const hrx_device_create_params_t devices[] = {
      {.uri = "task://"},
      {.uri = gpu_uri.c_str()},
  };
  REQUIRE_OK(hrx().device_group_create(resources.runtime, 2, devices,
                                       &resources.group));
  REQUIRE(hrx().device_group_device_count(resources.group) == 2);
  hrx_device_t cpu = hrx().device_group_get_device(resources.group, 0);
  hrx_device_t gpu = hrx().device_group_get_device(resources.group, 1);
  hrx_queue_t cpu_queue = FindProvisionedQueue(cpu, HRX_QUEUE_ROLE_TRANSFER);
  hrx_queue_t gpu_queue = FindProvisionedQueue(gpu, HRX_QUEUE_ROLE_TRANSFER);
  REQUIRE_OK(hrx().semaphore_create(cpu, 0, &resources.semaphores[0]));
  REQUIRE_OK(hrx().semaphore_create(gpu, 0, &resources.semaphores[1]));

  REQUIRE_OK(hrx().host_allocator_malloc_aligned(hrx().host_allocator_system(),
                                                 2 * kLength, 64 * 1024, 0,
                                                 &resources.host_storage));
  auto* host_bytes = static_cast<uint8_t*>(resources.host_storage);
  const hrx_buffer_params_t host_params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
  };
  // Imports alias caller-owned pages. Queue dependencies order access to those
  // pages without a host wait between the producer and consumer operations.
  REQUIRE_OK(hrx().allocator_import_buffer(hrx().device_allocator(cpu),
                                           host_params, host_bytes, kLength,
                                           &resources.buffers[0]));
  REQUIRE_OK(hrx().allocator_import_buffer(hrx().device_allocator(gpu),
                                           host_params, host_bytes, kLength,
                                           &resources.buffers[1]));
  REQUIRE_OK(hrx().allocator_import_buffer(hrx().device_allocator(gpu),
                                           host_params, host_bytes + kLength,
                                           kLength, &resources.buffers[2]));
  REQUIRE_OK(hrx().allocator_import_buffer(hrx().device_allocator(cpu),
                                           host_params, host_bytes + kLength,
                                           kLength, &resources.buffers[3]));
  const hrx_buffer_params_t gpu_params = {
      .type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
  };
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      hrx().device_allocator(gpu), gpu_params, kLength, &resources.buffers[4]));
  REQUIRE_OK(hrx().allocator_allocate_buffer(hrx().device_allocator(cpu),
                                             host_params, kLength,
                                             &resources.buffers[5]));

  hrx_memory_scope_t cpu_scope, gpu_scope;
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      cpu, hrx().queue_get_family_ordinal(cpu_queue), HRX_MEMORY_SITE_PROGRAM,
      &cpu_scope));
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      gpu,
      hrx().queue_get_family_ordinal(
          FindProvisionedQueue(gpu, HRX_QUEUE_ROLE_DISPATCH)),
      HRX_MEMORY_SITE_PROGRAM, &gpu_scope));

  // Retaining resources keep the complete group alive after the caller drops
  // both construction owners. Queue handles remain borrowed from that group.
  hrx().device_group_release(resources.group);
  resources.group = nullptr;
  hrx().runtime_release(resources.runtime);
  resources.runtime = nullptr;

  uint64_t address = 0;
  REQUIRE_OK(hrx().buffer_query_address(resources.buffers[0], cpu_scope,
                                        HRX_BUFFER_INTERFACE_HOST, &address));
  REQUIRE(address == reinterpret_cast<uintptr_t>(host_bytes));
  REQUIRE_OK(hrx().buffer_query_address(resources.buffers[1], gpu_scope,
                                        HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                        &address));
  REQUIRE(address != 0);

  for (size_t i = 0; i < kRegionCount; ++i) {
    const size_t offset = i * kRegionLength;
    const uint32_t pattern =
        0x10325476u ^ static_cast<uint32_t>(i * 0x01010101u);
    uint64_t produced = i + 1;
    const hrx_semaphore_list_t cpu_wait = {&resources.semaphores[0],
                                           &resources.tails[0], 1};
    hrx_semaphore_list_t cpu_signal = {&resources.semaphores[0], &produced, 1};
    REQUIRE_OK(hrx().queue_fill(cpu_queue, &cpu_wait, &cpu_signal,
                                resources.buffers[0], offset, kRegionLength,
                                &pattern, sizeof(pattern)));
    resources.tails[0] = produced;

    uint64_t copied = 2 * i + 1;
    // HAL queues may schedule independent operations out of order. Each copy
    // depends on its CPU producer and the preceding GPU timeline tail.
    uint64_t gpu_wait_values[] = {produced, resources.tails[1]};
    const hrx_semaphore_list_t gpu_wait = {resources.semaphores.data(),
                                           gpu_wait_values, 2};
    hrx_semaphore_list_t gpu_signal = {&resources.semaphores[1], &copied, 1};
    REQUIRE_OK(hrx().queue_copy(gpu_queue, &gpu_wait, &gpu_signal,
                                resources.buffers[1], offset,
                                resources.buffers[4], offset, kRegionLength));
    resources.tails[1] = copied;
    uint64_t returned = copied + 1;
    hrx_semaphore_list_t gpu_returned = {&resources.semaphores[1], &returned,
                                         1};
    REQUIRE_OK(hrx().queue_copy(gpu_queue, &gpu_signal, &gpu_returned,
                                resources.buffers[4], offset,
                                resources.buffers[2], offset, kRegionLength));
    resources.tails[1] = returned;
  }
  hrx_semaphore_list_t gpu_wait = {&resources.semaphores[1],
                                   &resources.tails[1], 1};
  uint64_t consumed = kRegionCount + 1;
  hrx_semaphore_list_t cpu_signal = {&resources.semaphores[0], &consumed, 1};
  REQUIRE_OK(hrx().queue_copy(cpu_queue, &gpu_wait, &cpu_signal,
                              resources.buffers[3], 0, resources.buffers[5], 0,
                              kLength));
  resources.tails[0] = consumed;
  REQUIRE_OK(
      hrx().semaphore_wait(resources.semaphores[0], consumed, UINT64_MAX));

  void* output = nullptr;
  REQUIRE_OK(hrx().buffer_map(resources.buffers[5], HRX_MAP_READ, 0, kLength,
                              &output));
  for (size_t i = 0; i < kLength / sizeof(uint32_t); ++i) {
    const size_t region = i * sizeof(uint32_t) / kRegionLength;
    CHECK(static_cast<const uint32_t*>(output)[i] ==
          (0x10325476u ^ static_cast<uint32_t>(region * 0x01010101u)));
  }
  REQUIRE_OK(hrx().buffer_unmap(resources.buffers[5]));
}
