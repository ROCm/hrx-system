// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/topology.h"

#include <fcntl.h>
#include <stdlib.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

// The filesystem is the native dependency. The production query reads it
// through the instance's sysfs root without a render node or native device.
class KfdTopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string directory = ::testing::TempDir() + "/amdf-topology-XXXXXX";
    ASSERT_NE(mkdtemp(directory.data()), nullptr);
    directory_ = directory;
    const auto topology = directory_ / "class/kfd/kfd/topology";
    const auto node = topology / "nodes/6";
    const auto device = directory_ / "dev/char/226:160/device";
    const auto sdma = device / "ip_discovery/die/0/42/0";
    std::filesystem::create_directories(node);
    std::filesystem::create_directories(sdma);
    WriteAttribute(topology / "generation_id", "1\n");
    WriteAttribute(node / "gpu_id", "53458\n");
    WriteAttribute(device / "mem_info_vram_total", "206141652992\n");
    WriteAttribute(device / "mem_info_vis_vram_total", "206141652992\n");
    WriteAttribute(sdma / "major", "4\n");
    WriteAttribute(sdma / "minor", "4\n");
    WriteAttribute(sdma / "revision", "2\n");
    instance_.sysfs_descriptor =
        open(directory_.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
    ASSERT_GE(instance_.sysfs_descriptor, 0);
    endpoint_.instance = &instance_;
    endpoint_.info.id.words[0] = (UINT64_C(226) << 32) | 160;
    endpoint_.info.pci.vendor_id = 0x1002;
    endpoint_.info.pci.device_id = 0x74a1;
  }

  void TearDown() override {
    EXPECT_EQ(amdf_linux_file_close(&instance_.sysfs_descriptor),
              AMDF_STATUS_OK);
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }

  void WriteAttribute(const std::filesystem::path& path,
                      const std::string& value) {
    std::ofstream stream(path);
    stream << value;
    stream.close();
    ASSERT_TRUE(stream.good()) << path;
  }

  void WriteProperties(const char* optional_properties,
                       uint32_t peer_count = 0) {
    WriteAttribute(directory_ / "class/kfd/kfd/topology/nodes/6/properties",
                   std::string("drm_render_minor 160\n"
                               "vendor_id 4098\n"
                               "device_id 29857\n"
                               "gfx_target_version 90402\n"
                               "capability 2893521536\n"
                               "simd_count 1216\n"
                               "simd_per_cu 4\n"
                               "max_waves_per_simd 8\n"
                               "max_slots_scratch_cu 32\n"
                               "wave_front_size 64\n"
                               "lds_size_in_kb 64\n"
                               "array_count 32\n"
                               "simd_arrays_per_engine 1\n"
                               "num_xcc 8\n"
                               "num_cp_queues 24\n") +
                       "p2p_links_count " + std::to_string(peer_count) + "\n" +
                       optional_properties);
  }

  void WritePeer(uint32_t ordinal, uint32_t backing_node, uint32_t gpu_id,
                 uint32_t flags) {
    const auto nodes = directory_ / "class/kfd/kfd/topology/nodes";
    const auto link = nodes / "6/p2p_links" / std::to_string(ordinal);
    const auto backing = nodes / std::to_string(backing_node);
    std::filesystem::create_directories(link);
    std::filesystem::create_directories(backing);
    WriteAttribute(link / "properties", "type 2\nnode_from 6\nnode_to " +
                                            std::to_string(backing_node) +
                                            "\nflags " + std::to_string(flags) +
                                            "\n");
    WriteAttribute(backing / "gpu_id", std::to_string(gpu_id) + "\n");
    if (gpu_id != 0) {
      WriteAttribute(backing / "properties", "drm_render_minor 168\n");
    }
  }

  // Temporary native metadata directory owned by this case.
  std::filesystem::path directory_;
  // Instance whose discovery root names only the test-owned directory.
  amdf_platform_instance_t instance_ = {.sysfs_descriptor = -1};
  // Identity-checked endpoint supplied to the native topology consumer.
  amdf_platform_endpoint_t endpoint_ = {};
};

TEST_F(KfdTopologyTest, DiscoversMemoryAndSdmaWithoutComputeStorageMetadata) {
  WriteProperties(
      "num_sdma_engines 2\n"
      "num_sdma_xgmi_engines 14\n"
      "num_sdma_queues_per_engine 8\n");
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_EQ(topology.gpu_id, 53458u);
  EXPECT_EQ(topology.properties.gfx_ip.major, 9u);
  EXPECT_EQ(topology.properties.gfx_ip.minor, 4u);
  EXPECT_EQ(topology.properties.gfx_ip.stepping, 2u);
  EXPECT_EQ(topology.properties.compute.compute_unit_count, 304u);
  EXPECT_EQ(topology.properties.compute.wavefront_size, 64u);
  EXPECT_EQ(topology.properties.topology.xcc_count, 8u);
  EXPECT_EQ(topology.vram.total_byte_length, UINT64_C(206141652992));
  EXPECT_EQ(topology.vram.visible_byte_length, UINT64_C(206141652992));
  EXPECT_EQ(topology.sdma.engine_count, 2u);
  EXPECT_EQ(topology.sdma.xgmi_engine_count, 14u);
  EXPECT_EQ(topology.sdma.queue_count_per_engine, 8u);
  EXPECT_TRUE(topology.sdma.ip.exact);
  EXPECT_EQ(topology.sdma.ip.major, 4u);
  EXPECT_EQ(topology.sdma.ip.minor, 4u);
  EXPECT_EQ(topology.sdma.ip.revision, 2u);
  EXPECT_EQ(topology.context_save_restore_byte_length, 0u);
  EXPECT_EQ(topology.control_stack_byte_length, 0u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, PreservesComputeStorageWithoutSdmaMetadata) {
  WriteProperties("cwsr_size 19185664\nctl_stack_size 16384\n");
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_EQ(topology.context_save_restore_byte_length, 19185664u);
  EXPECT_EQ(topology.control_stack_byte_length, 16384u);
  EXPECT_EQ(topology.sdma.engine_count, 0u);
  EXPECT_EQ(topology.sdma.xgmi_engine_count, 0u);
  EXPECT_EQ(topology.sdma.queue_count_per_engine, 0u);
  EXPECT_FALSE(topology.sdma.ip.exact);
  EXPECT_EQ(topology.vram.total_byte_length, UINT64_C(206141652992));
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, RejectsPartialOptionalGroupsWithoutPublication) {
  for (const char* properties : {
           "cwsr_size 19185664\n",
           "ctl_stack_size 16384\n",
           "num_sdma_engines 2\n",
           "num_sdma_engines 2\nnum_sdma_xgmi_engines 14\n",
           "cwsr_size 19185664\nctl_stack_size 16384\n"
           "num_sdma_engines 2\n",
       }) {
    SCOPED_TRACE(properties);
    WriteProperties(properties);
    amdf_gpu_kfd_topology_t topology;
    std::memset(&topology, 0xA5, sizeof(topology));
    const amdf_gpu_kfd_topology_t original = topology;
    EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              amdf_linux_error(EPROTO));
    EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  }
}

TEST_F(KfdTopologyTest, PreservesFullNativeHiveAndItsIndependentPolicy) {
  WriteProperties("");
  const auto hive = directory_ / "dev/char/226:160/device/xgmi_hive_info";
  const auto policy = directory_ / "module/amdgpu/parameters";
  std::filesystem::create_directories(hive);
  std::filesystem::create_directories(policy);
  WriteAttribute(hive / "xgmi_hive_id", "11827785098739261628\n");
  for (uint32_t enabled : {0u, 1u}) {
    WriteAttribute(policy / "use_xgmi_p2p", std::to_string(enabled) + "\n");
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_EQ(topology.memory_peers.hive_id, UINT64_C(11827785098739261628));
    EXPECT_EQ(topology.memory_peers.hive_sharing_enabled, enabled != 0);
    EXPECT_EQ(topology.memory_peers.count, 0u);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, KeepsOnlyAdmittedGpuBackingsInThePublishedDirection) {
  WriteProperties("", 4);
  WritePeer(0, 2, 12345, 1);
  WritePeer(1, 3, 0, 1);
  WritePeer(2, 4, 54321, 0);
  WritePeer(3, 5, 32145, 17);
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_EQ(topology.gpu_id, 53458u);
  ASSERT_EQ(topology.memory_peers.count, 1u);
  EXPECT_EQ(topology.memory_peers.gpu_ids[0], 12345u);
  EXPECT_EQ(topology.memory_peers.hive_id, 0u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, MalformedPeerRecordsNeverPublishPartialTopology) {
  WriteProperties("", 1);
  WritePeer(0, 2, 12345, 1);
  for (const char* properties : {
           "type 2\nnode_from 7\nnode_to 2\nflags 1\n",
           "type 2\nnode_from 6\nflags 1\n",
           "type 2\nnode_from 6\nnode_to 2\nnode_to 3\nflags 1\n",
           "type 2\nnode_from 6\nnode_to 4294967296\nflags 1\n",
           "type 2\nnode_from 6\nnode_to -1\nflags 1\n",
       }) {
    SCOPED_TRACE(properties);
    WriteAttribute(
        directory_ / "class/kfd/kfd/topology/nodes/6/p2p_links/0/properties",
        properties);
    amdf_gpu_kfd_topology_t topology;
    std::memset(&topology, 0xA5, sizeof(topology));
    const amdf_gpu_kfd_topology_t original = topology;
    EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              amdf_linux_error(EPROTO));
    EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  }
}

TEST_F(KfdTopologyTest, EndpointOwnsItsCompleteImmutablePeerSnapshot) {
  WriteProperties("", 1);
  WritePeer(0, 2, 12345, 1);
  amdf_gpu_endpoint_profile_t* profile = nullptr;
  ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                &endpoint_, AMDF_NATIVE_LIFETIME_PROCESS,
                amdf_allocator_system(), &profile),
            AMDF_STATUS_OK);
  const auto* topology = static_cast<const amdf_gpu_kfd_topology_t*>(
      profile->memory.values[0].construction.data);
  WritePeer(0, 2, 23456, 1);
  ASSERT_EQ(topology->memory_peers.count, 1u);
  EXPECT_EQ(topology->memory_peers.gpu_ids[0], 12345u);
  for (uint32_t i = 0; i < profile->memory.count; ++i) {
    EXPECT_EQ(profile->memory.values[i].construction.data, topology);
  }
  amdf_free(amdf_allocator_system(), profile);
}

TEST_F(KfdTopologyTest, ChangedGenerationReleasesUnpublishedPeerMetadata) {
  WriteProperties("", 1);
  WritePeer(0, 2, 12345, 1);
  struct AllocationState {
    // Native generation file changed while the snapshot is being constructed.
    std::filesystem::path generation_path;
    // Metadata allocations that have not yet been released.
    uint32_t live_count = 0;
  } state = {directory_ / "class/kfd/kfd/topology/generation_id"};
  amdf_allocator_t allocator = {};
  allocator.user_data = &state;
  allocator.allocate = [](void* user_data, uint64_t length,
                          uint64_t alignment) -> void* {
    auto* state = static_cast<AllocationState*>(user_data);
    std::ofstream stream(state->generation_path);
    stream << "2\n";
    stream.close();
    EXPECT_TRUE(stream.good());
    const auto system = amdf_allocator_system();
    void* pointer = system.allocate(system.user_data, length, alignment);
    if (pointer != nullptr) ++state->live_count;
    return pointer;
  };
  allocator.free = [](void* user_data, void* pointer) {
    auto* state = static_cast<AllocationState*>(user_data);
    EXPECT_GT(state->live_count, 0u);
    --state->live_count;
    const auto system = amdf_allocator_system();
    system.free(system.user_data, pointer);
  };
  amdf_gpu_kfd_topology_t topology;
  std::memset(&topology, 0xA5, sizeof(topology));
  const amdf_gpu_kfd_topology_t original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(&endpoint_, allocator, &topology),
            amdf_linux_error(EAGAIN));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  EXPECT_EQ(state.live_count, 0u);
}

TEST_F(KfdTopologyTest, FailedSnapshotAllocationsLeaveNoMetadataOrOutput) {
  WriteProperties("", 1);
  WritePeer(0, 2, 12345, 1);
  struct AllocationState {
    // Allocation ordinal at which the external allocator reports exhaustion.
    uint32_t failure_ordinal;
    // Number of attempted metadata allocations.
    uint32_t count = 0;
    // Number of allocations still owned by the constructor.
    uint32_t live_count = 0;
  };
  for (uint32_t failure_ordinal : {0u, 1u, 2u}) {
    AllocationState state = {failure_ordinal};
    amdf_allocator_t allocator = {};
    allocator.user_data = &state;
    allocator.allocate = [](void* user_data, uint64_t length,
                            uint64_t alignment) -> void* {
      auto* state = static_cast<AllocationState*>(user_data);
      if (state->count++ == state->failure_ordinal) return nullptr;
      const auto system = amdf_allocator_system();
      void* pointer = system.allocate(system.user_data, length, alignment);
      if (pointer != nullptr) ++state->live_count;
      return pointer;
    };
    allocator.free = [](void* user_data, void* pointer) {
      auto* state = static_cast<AllocationState*>(user_data);
      EXPECT_GT(state->live_count, 0u);
      --state->live_count;
      const auto system = amdf_allocator_system();
      system.free(system.user_data, pointer);
    };
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_endpoint_profile_t*>(uintptr_t{1});
    amdf_gpu_endpoint_profile_t* profile = sentinel;
    const amdf_status_t status = amdf_gpu_umd_create_endpoint_profile(
        &endpoint_, AMDF_NATIVE_LIFETIME_PROCESS, allocator, &profile);
    if (failure_ordinal < 2) {
      EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      EXPECT_EQ(profile, sentinel);
    } else {
      ASSERT_EQ(status, AMDF_STATUS_OK);
      EXPECT_EQ(state.live_count, 1u);
      amdf_free(allocator, profile);
    }
    EXPECT_EQ(state.live_count, 0u);
  }
}

}  // namespace
