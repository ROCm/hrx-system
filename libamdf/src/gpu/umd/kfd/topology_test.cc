// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/topology.h"

#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/utsname.h>

#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

struct NativeDeviceInfo {
  // Kernel release supplied by the native identity dependency.
  const char* kernel_release = "7.3.0";
  // Cached native facts returned by the read-only render query.
  drm_amdgpu_info_device info = {
      .virtual_address_offset = 0x10000,
      .virtual_address_max = UINT64_C(1) << 47,
      .virtual_address_alignment = 4096,
      .pte_fragment_size = 2 * 1024 * 1024,
  };
  // Native local-memory capacity, reserved bytes, and global usage snapshot.
  drm_amdgpu_memory_info memory = {
      .vram =
          {
              .total_heap_size = UINT64_C(2) << 30,
              .usable_heap_size = UINT64_C(1536) << 20,
              .heap_usage = UINT64_C(512) << 20,
          },
  };
  // Real descriptor borrowed by an explicit native query.
  int descriptor = -1;
  // Execution-file opens attempted by the operation under test.
  uint32_t open_count = 0;
  // Error at native file opening, or zero for success.
  int open_error = 0;
  // Error at the metadata query, or zero for success.
  int query_error = 0;
  // Number of native metadata queries.
  uint32_t query_count = 0;
};

thread_local NativeDeviceInfo* native_device_info = nullptr;

struct AttributeReadFailure {
  // Filesystem device containing the selected sysfs fixture attribute.
  dev_t device = 0;
  // Inode of the selected attribute, independent of its temporary descriptor.
  ino_t inode = 0;
  // Native read error returned for that attribute.
  int error = 0;
  // Reads of the selected attribute attempted by the production reader.
  uint32_t attempts = 0;
};

thread_local AttributeReadFailure* attribute_read_failure = nullptr;

}  // namespace

extern "C" amdf_status_t __real_amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t*, int*);
extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __real_uname(struct utsname*);
extern "C" ssize_t __real_read(int, void*, size_t);

extern "C" ssize_t __wrap_read(int descriptor, void* buffer, size_t length) {
  if (attribute_read_failure != nullptr) {
    struct stat info = {};
    if (fstat(descriptor, &info) == 0 &&
        info.st_dev == attribute_read_failure->device &&
        info.st_ino == attribute_read_failure->inode) {
      ++attribute_read_failure->attempts;
      errno = attribute_read_failure->error;
      return -1;
    }
  }
  return __real_read(descriptor, buffer, length);
}

extern "C" int __wrap_uname(struct utsname* out_kernel) {
  if (native_device_info == nullptr) {
    return __real_uname(out_kernel);
  }
  std::memset(out_kernel, 0, sizeof(*out_kernel));
  std::strcpy(out_kernel->release, native_device_info->kernel_release);
  return 0;
}

extern "C" amdf_status_t __wrap_amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t* endpoint, int* out_descriptor) {
  if (native_device_info == nullptr) {
    return __real_amdf_linux_endpoint_open_file(endpoint, out_descriptor);
  }
  ++native_device_info->open_count;
  if (native_device_info->open_error != 0) {
    return amdf_linux_error(native_device_info->open_error);
  }
  const int descriptor = open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  native_device_info->descriptor = descriptor;
  native_device_info->info.device_id = endpoint->info.pci.device_id;
  *out_descriptor = descriptor;
  return AMDF_STATUS_OK;
}

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_device_info == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(descriptor, native_device_info->descriptor);
  EXPECT_EQ(request, DRM_IOCTL_AMDGPU_INFO);
  const auto* query = static_cast<const drm_amdgpu_info*>(argument);
  ++native_device_info->query_count;
  if (native_device_info->query_error != 0) {
    errno = native_device_info->query_error;
    return -1;
  }
  if (query->query == AMDGPU_INFO_MEMORY) {
    EXPECT_EQ(query->return_size, sizeof(drm_amdgpu_memory_info));
    *reinterpret_cast<drm_amdgpu_memory_info*>(query->return_pointer) =
        native_device_info->memory;
  } else {
    EXPECT_EQ(query->query, AMDGPU_INFO_DEV_INFO);
    EXPECT_EQ(query->return_size, sizeof(drm_amdgpu_info_device));
    *reinterpret_cast<drm_amdgpu_info_device*>(query->return_pointer) =
        native_device_info->info;
  }
  return 0;
}

namespace {

TEST(KfdExecutionResourceTest, QualifiedNativeMappingGeometry) {
  struct Case {
    // Compiler-visible architecture, distinct from GC discovery.
    uint32_t gfx_ip[3];
    // Exact native IP identity when reported by sysfs.
    amdf_gpu_kfd_ip_version_t gc_ip;
    // Total active native compute units after harvesting.
    uint32_t compute_unit_count;
    // Number of command-processor groups.
    uint32_t xcc_count;
    // Expected selectable unit size, or zero for an unsupported geometry.
    uint32_t units_per_resource;
  };
  const Case cases[] = {
      {{9, 0, 10}, {9, 0, 10, true}, 104, 1, 1},
      {{9, 4, 0}, {9, 4, 3, true}, 304, 8, 1},
      {{9, 4, 1}, {9, 4, 3, true}, 76, 2, 1},
      {{9, 4, 2}, {9, 4, 3, true}, 304, 8, 1},
      {{9, 5, 0}, {9, 5, 0, true}, 256, 8, 1},
      {{11, 0, 0}, {11, 0, 0, true}, 96, 1, 2},
      {{11, 5, 1}, {11, 5, 1, true}, 40, 1, 2},
      {{11, 7, 0}, {11, 7, 0, true}, 32, 1, 2},
      {{12, 0, 0}, {12, 0, 0, true}, 56, 1, 2},
      {{12, 0, 1}, {12, 0, 1, true}, 64, 1, 2},
      {{12, 5, 0}, {12, 1, 0, true}, 256, 8, 1},
      {{12, 5, 1}, {12, 1, 0, true}, 256, 8, 1},
      {{12, 5, 0}, {12, 5, 0, true}, 256, 8, 0},
      {{12, 5, 0}, {}, 256, 8, 0},
      {{13, 0, 0}, {13, 0, 0, true}, 64, 1, 0},
      {{11, 0, 0}, {11, 0, 0, true}, 95, 1, 0},
      {{9, 4, 2}, {9, 4, 3, true}, 303, 8, 0},
      {{9, 4, 2}, {9, 4, 3, true}, 1024, 8, 1},
      {{9, 4, 2}, {9, 4, 3, true}, 2048, 8, 0},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.gfx_ip[0]);
    SCOPED_TRACE(test.gfx_ip[1]);
    SCOPED_TRACE(test.gfx_ip[2]);
    amdf_gpu_kfd_topology_t topology = {};
    topology.properties.gfx_ip = {test.gfx_ip[0], test.gfx_ip[1],
                                  test.gfx_ip[2]};
    topology.gc_ip = test.gc_ip;
    topology.properties.compute.compute_unit_count = test.compute_unit_count;
    topology.properties.topology.xcc_count = test.xcc_count;
    const auto geometry = amdf_gpu_kfd_topology_execution_resources(&topology);
    EXPECT_EQ(geometry.compute_units_per_resource, test.units_per_resource);
    EXPECT_EQ(geometry.group_count,
              test.units_per_resource ? test.xcc_count : 0);
    EXPECT_EQ(geometry.minimum_selected_resources_per_group,
              test.units_per_resource ? 1u : 0u);
  }
}

// The filesystem is the native dependency. The production query reads it
// through the instance's sysfs root. Profile construction additionally queries
// cached render metadata; neither path acquires a KFD VM or execution storage.
class KfdTopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_device_info = &native_;
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
    WriteAttribute(device / "mem_info_gtt_total", "68719476736\n");
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
    native_device_info = nullptr;
    attribute_read_failure = nullptr;
    if (native_.descriptor >= 0) {
      EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), -1);
      EXPECT_EQ(errno, EBADF);
    }
    EXPECT_EQ(amdf_linux_file_close(&instance_.sysfs_descriptor),
              AMDF_STATUS_OK);
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  void WriteAttribute(const std::filesystem::path& path,
                      const std::string& value) {
    std::ofstream stream(path);
    stream << value;
    stream.close();
    ASSERT_TRUE(stream.good()) << path;
  }

  void WriteProperties(const char* optional_properties, uint32_t peer_count = 0,
                       uint32_t gfx_target_version = 90402) {
    WriteAttribute(
        directory_ / "class/kfd/kfd/topology/nodes/6/properties",
        std::string("drm_render_minor 160\n"
                    "vendor_id 4098\n"
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
            "device_id " + std::to_string(endpoint_.info.pci.device_id) + "\n" +
            "gfx_target_version " + std::to_string(gfx_target_version) + "\n" +
            "p2p_links_count " + std::to_string(peer_count) + "\n" +
            optional_properties);
  }

  void FailAttributeReads(const std::filesystem::path& path, int error) {
    struct stat info = {};
    ASSERT_EQ(stat(path.c_str(), &info), 0);
    read_failure_ = {
        .device = info.st_dev,
        .inode = info.st_ino,
        .error = error,
    };
    attribute_read_failure = &read_failure_;
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

  void WriteIoLink(uint32_t ordinal, uint32_t destination, uint32_t gpu_id,
                   uint32_t flags, uint32_t type = 2) {
    const auto nodes = directory_ / "class/kfd/kfd/topology/nodes";
    const auto link = nodes / "6/io_links" / std::to_string(ordinal);
    const auto target = nodes / std::to_string(destination);
    std::filesystem::create_directories(link);
    std::filesystem::create_directories(target);
    WriteAttribute(link / "properties",
                   "type " + std::to_string(type) + "\nnode_from 6\nnode_to " +
                       std::to_string(destination) + "\nflags " +
                       std::to_string(flags) + "\n");
    WriteAttribute(target / "gpu_id", std::to_string(gpu_id) + "\n");
    if (gpu_id != 0) {
      WriteAttribute(target / "properties", "drm_render_minor 168\n");
    }
  }

  // Read-only native query dependency; descriptor release remains real.
  NativeDeviceInfo native_;
  // Optional failure injected at the native attribute-read dependency.
  AttributeReadFailure read_failure_;
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
      "num_sdma_queues_per_engine 8\n"
      "num_gws 64\n");
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_EQ(topology.gpu_id, 53458u);
  EXPECT_EQ(topology.properties.unique_id, 0u);
  EXPECT_EQ(topology.properties.gfx_ip.major, 9u);
  EXPECT_EQ(topology.properties.gfx_ip.minor, 4u);
  EXPECT_EQ(topology.properties.gfx_ip.stepping, 2u);
  EXPECT_FALSE(topology.gc_ip.exact);
  EXPECT_TRUE(topology.sdma.device_producer_restricted);
  EXPECT_EQ(topology.properties.compute.compute_unit_count, 304u);
  EXPECT_EQ(topology.properties.compute.wavefront_size, 64u);
  EXPECT_EQ(topology.properties.topology.xcc_count, 8u);
  EXPECT_EQ(topology.vram.total_byte_length, UINT64_C(206141652992));
  EXPECT_EQ(topology.vram.visible_byte_length, UINT64_C(206141652992));
  EXPECT_EQ(topology.gtt_byte_length, UINT64_C(68719476736));
  EXPECT_EQ(topology.sdma.engine_count, 2u);
  EXPECT_EQ(topology.sdma.xgmi_engine_count, 14u);
  EXPECT_EQ(topology.sdma.queue_count_per_engine, 8u);
  EXPECT_EQ(topology.global_wave_sync_count, 64u);
  EXPECT_TRUE(topology.sdma.ip.exact);
  EXPECT_EQ(topology.sdma.ip.major, 4u);
  EXPECT_EQ(topology.sdma.ip.minor, 4u);
  EXPECT_EQ(topology.sdma.ip.revision, 2u);
  EXPECT_EQ(topology.context_save_restore_byte_length, 0u);
  EXPECT_EQ(topology.control_stack_byte_length, 0u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, PreservesFullWidthNativeUniqueId) {
  for (uint64_t unique_id :
       {UINT64_C(0), UINT64_C(0xfedcba9876543210), UINT64_MAX}) {
    const std::string properties =
        "unique_id " + std::to_string(unique_id) + "\n";
    WriteProperties(properties.c_str());
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_EQ(topology.properties.unique_id, unique_id);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, RejectsMalformedOrDuplicateNativeUniqueId) {
  for (const char* properties :
       {"unique_id -1\n", "unique_id 18446744073709551616\n",
        "unique_id 1tail\n", "unique_id 0\nunique_id 1\n"}) {
    SCOPED_TRACE(properties);
    WriteProperties(properties);
    amdf_gpu_kfd_topology_t topology = {.gpu_id = 123};
    EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              amdf_linux_error(EPROTO));
    EXPECT_EQ(topology.gpu_id, 123u);
  }
}

TEST_F(KfdTopologyTest, LocalAdmissionUsesInitializedAllocationDomains) {
  WriteProperties("");
  const auto device = directory_ / "dev/char/226:160/device";
  struct Configuration {
    // Native package identity, independent of host aperture size.
    bool integrated;
    // Initialized physical VRAM capacity in MiB.
    uint64_t vram_mib;
    // Initialized GTT manager capacity in MiB.
    uint64_t gtt_mib;
    // CPU-visible VRAM aperture in MiB.
    uint64_t visible_mib;
    // Whether the native allocator honors a LOCAL request.
    bool local;
    // Whether all admitted LOCAL memory can be mapped on the CPU.
    bool host_visible;
  };
  const Configuration configurations[] = {
      {true, 512, 98304, 512, false, false},
      {true, 98304, 16384, 98304, true, true},
      {true, 16384, 16384, 16384, true, true},
      {true, 16384, 16385, 16384, false, false},
      {true, 16384, 16384, 256, true, false},
      {true, 0, 16384, 0, false, false},
      {true, 16384, 0, 16384, true, true},
      {false, 16384, 98304, 16384, true, true},
      {false, 16384, 98304, 256, true, false},
      {false, 0, 98304, 0, false, false},
  };
  for (const auto& configuration : configurations) {
    SCOPED_TRACE(::testing::Message()
                 << "integrated=" << configuration.integrated << " vram="
                 << configuration.vram_mib << " gtt=" << configuration.gtt_mib
                 << " visible=" << configuration.visible_mib);
    WriteAttribute(device / "mem_info_vram_total",
                   std::to_string(configuration.vram_mib << 20) + "\n");
    WriteAttribute(device / "mem_info_gtt_total",
                   std::to_string(configuration.gtt_mib << 20) + "\n");
    WriteAttribute(device / "mem_info_vis_vram_total",
                   std::to_string(configuration.visible_mib << 20) + "\n");
    native_.info.ids_flags =
        configuration.integrated ? AMDGPU_IDS_FLAGS_FUSION : 0;
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    int descriptor = -1;
    ASSERT_EQ(amdf_linux_endpoint_open_file(&endpoint_, &descriptor),
              AMDF_STATUS_OK);
    ASSERT_EQ(amdf_gpu_kfd_topology_query_memory(
                  descriptor, endpoint_.info.pci.device_id, &topology),
              AMDF_STATUS_OK);
    EXPECT_EQ(topology.integrated, configuration.integrated);
    EXPECT_EQ(topology.memory_features,
              (configuration.local ? AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY : 0) |
                  (configuration.host_visible
                       ? AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY
                       : 0));
    EXPECT_EQ(amdf_linux_file_close(&descriptor), AMDF_STATUS_OK);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
  EXPECT_EQ(native_.query_count, std::size(configurations));
}

TEST_F(KfdTopologyTest, MalformedHeapMetadataDoesNotBecomeAbsentCapability) {
  WriteProperties("");
  WriteAttribute(directory_ / "dev/char/226:160/device/mem_info_gtt_total",
                 "invalid\n");
  amdf_gpu_kfd_topology_t topology = {.gpu_id = 123};
  const auto original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_linux_error(EPROTO));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  EXPECT_EQ(native_.query_count, 0u);
}

TEST_F(KfdTopologyTest, NativeGcIdentityDoesNotUseCompilerTargetAliases) {
  WriteProperties("");
  const auto gc =
      directory_ / "dev/char/226:160/device/ip_discovery/die/0/11/0";
  std::filesystem::create_directories(gc);
  WriteAttribute(gc / "major", "9\n");
  WriteAttribute(gc / "minor", "4\n");
  WriteAttribute(gc / "revision", "3\n");
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_TRUE(topology.gc_ip.exact);
  EXPECT_EQ(topology.gc_ip.major, 9u);
  EXPECT_EQ(topology.gc_ip.minor, 4u);
  EXPECT_EQ(topology.gc_ip.revision, 3u);
  EXPECT_EQ(topology.properties.gfx_ip.stepping, 2u);
  EXPECT_FALSE(topology.sdma.ip.exact);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, ShaderDoorbellSupportFollowsFixedKernelReleases) {
  WriteProperties(
      "cwsr_size 19185664\nctl_stack_size 16384\n"
      "num_sdma_engines 2\nnum_sdma_xgmi_engines 14\n"
      "num_sdma_queues_per_engine 8\n");
  const auto module = directory_ / "module/amdgpu";
  std::filesystem::create_directories(module);
  WriteAttribute(module / "taint", "\n");
  struct Case {
    // Release reported by the upstream module's running kernel.
    const char* release;
    // Whether this release carries the native doorbell alignment fix.
    bool device_producer;
  };
  const Case cases[] = {
      {"6.6.159", false},
      {"6.6.160", true},
      {"6.12.112", false},
      {"6.12.113", true},
      {"6.18.56", false},
      {"6.18.57", true},
      {"7.2.10", false},
      {"7.2.11", true},
      {"7.0.99-generic", false},
      {"7.1.99-custom", false},
      {"7.3.0-rc6", false},
      {"7.3.0-rc7", true},
      {"7.3.0", true},
      {"7.4.0-rc1", true},
      {"6.12.113-custom", true},
      {"6.18.57-rc1", false},
      {"7.2.12-rc1", true},
      {"7.2.11.31400100", false},
      {"7.3.0-rc", false},
      {"7.3.0-rc7extra", false},
      {"999999999999999999999.0.0", false},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.release);
    native_.kernel_release = test.release;
    amdf_gpu_endpoint_profile_t* profile = nullptr;
    ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                  &endpoint_, amdf_allocator_system(), &profile),
              AMDF_STATUS_OK);
    ASSERT_EQ(profile->queue_family_count, 2u);
    EXPECT_EQ(profile->queue_families[0].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    EXPECT_NE(profile->queue_families[0].user_queue_capabilities &
                  AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
              0u);
    EXPECT_EQ(profile->queue_families[1].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    EXPECT_NE(profile->queue_families[1].user_queue_capabilities &
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
              0u);
    EXPECT_EQ((profile->queue_families[1].user_queue_capabilities &
               AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) != 0,
              test.device_producer);
    EXPECT_EQ(native_.open_count, 0u);
    amdf_free(amdf_allocator_system(), profile);
  }
}

TEST_F(KfdTopologyTest, VendorModulesDoNotInheritTheHostKernelFix) {
  WriteProperties(
      "num_sdma_engines 2\nnum_sdma_xgmi_engines 0\n"
      "num_sdma_queues_per_engine 8\n");
  const auto module = directory_ / "module/amdgpu";
  std::filesystem::create_directories(module);
  native_.kernel_release = "7.4.0";
  WriteAttribute(module / "taint", "OE\n");
  for (const char* version : {"6.16.13-custom\n", "6.19.14.31400100\n"}) {
    WriteAttribute(module / "version", version);
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_TRUE(topology.sdma.device_producer_restricted);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
  std::filesystem::remove(module / "version");
  for (bool expose_module : {true, false}) {
    if (!expose_module) {
      std::filesystem::remove(module / "taint");
    }
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_TRUE(topology.sdma.device_producer_restricted);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, ShaderDoorbellRestrictionUsesNativeHardwareIdentity) {
  const auto gc =
      directory_ / "dev/char/226:160/device/ip_discovery/die/0/11/0";
  std::filesystem::create_directories(gc);
  struct Case {
    // Compiler-facing target for this physical GPU.
    uint32_t target;
    // Exact GC IP supplied by native discovery.
    uint32_t gc[3];
    // Whether shader doorbell routing needs the native driver fix.
    bool restricted;
  };
  const Case cases[] = {
      {90402, {9, 4, 3}, true},    {90402, {9, 4, 4}, true},
      {90500, {9, 5, 0}, true},    {90010, {9, 4, 2}, false},
      {110501, {11, 5, 1}, false},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.target);
    WriteProperties(
        "num_sdma_engines 2\nnum_sdma_xgmi_engines 0\n"
        "num_sdma_queues_per_engine 8\n",
        0, test.target);
    WriteAttribute(gc / "major", std::to_string(test.gc[0]) + "\n");
    WriteAttribute(gc / "minor", std::to_string(test.gc[1]) + "\n");
    WriteAttribute(gc / "revision", std::to_string(test.gc[2]) + "\n");
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_EQ(topology.sdma.device_producer_restricted, test.restricted);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, ModuleReadFailureDoesNotPublishTopology) {
  WriteProperties("");
  // Opening a directory succeeds, but reading it as an attribute fails.
  std::filesystem::create_directories(directory_ / "module/amdgpu/version");
  amdf_gpu_kfd_topology_t topology;
  std::memset(&topology, 0xA5, sizeof(topology));
  const auto original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_linux_error(EISDIR));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
}

TEST_F(KfdTopologyTest, ScalarReservationFollowsNativeTrapPolicy) {
  const auto parameters = directory_ / "module/amdgpu/parameters";
  std::filesystem::create_directories(parameters);
  for (uint32_t target :
       {90010u, 90400u, 90401u, 90402u, 90500u, 110000u, 120000u, 120500u}) {
    SCOPED_TRACE(target);
    WriteProperties("", 0, target);
    for (uint32_t enabled : {0u, 1u}) {
      SCOPED_TRACE(enabled);
      WriteAttribute(parameters / "cwsr_enable",
                     std::to_string(enabled) + "\n");
      amdf_gpu_kfd_topology_t topology = {};
      ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                    &endpoint_, amdf_allocator_system(), &topology),
                AMDF_STATUS_OK);
      const amdf_queue_format_features_t expected =
          target >= 100000 ? 0
          : enabled        ? AMDF_GPU_AQL_FORMAT_FEATURE_SGPR_TRAP_RESERVE_16
                           : AMDF_GPU_AQL_FORMAT_FEATURE_SGPR_TRAP_RESERVE_0;
      EXPECT_EQ(topology.aql_scalar_register_features, expected);
      amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
    }
  }
}

TEST_F(KfdTopologyTest, MissingTrapPolicyDoesNotGuessAReservation) {
  WriteProperties("");
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_EQ(topology.aql_scalar_register_features, 0u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, MalformedTrapPolicyDoesNotPublishTopology) {
  WriteProperties("");
  const auto parameters = directory_ / "module/amdgpu/parameters";
  std::filesystem::create_directories(parameters);
  WriteAttribute(parameters / "cwsr_enable", "invalid\n");
  amdf_gpu_kfd_topology_t topology = {.gpu_id = 73};
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_linux_error(EPROTO));
  EXPECT_EQ(topology.gpu_id, 73u);
}

TEST_F(KfdTopologyTest, PartialGcDiscoveryDoesNotPublishTopology) {
  WriteProperties("");
  const auto gc =
      directory_ / "dev/char/226:160/device/ip_discovery/die/0/11/0";
  std::filesystem::create_directories(gc);
  WriteAttribute(gc / "major", "11\n");
  WriteAttribute(gc / "minor", "5\n");
  amdf_gpu_kfd_topology_t topology;
  std::memset(&topology, 0xA5, sizeof(topology));
  const auto original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_linux_error(ENOENT));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
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

TEST_F(KfdTopologyTest, DeviceCgroupDenialSkipsGpuBeforeSelectedEndpoint) {
  WriteProperties("");
  const auto nodes = directory_ / "class/kfd/kfd/topology/nodes";
  std::filesystem::create_directories(nodes / "8");
  // Choose the denied node in actual directory order. Populating files within
  // these two directories leaves that order intact and tests scan continuation.
  const auto denied = std::filesystem::directory_iterator(nodes)->path();
  const auto allowed = nodes / (denied.filename() == "6" ? "8" : "6");
  if (allowed.filename() != "6") {
    std::filesystem::copy_file(nodes / "6/properties", allowed / "properties");
  }
  WriteAttribute(allowed / "gpu_id", "53458\n");
  WriteAttribute(denied / "gpu_id", "12345\n");
  FailAttributeReads(denied / "gpu_id", EPERM);
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_GT(read_failure_.attempts, 0u);
  EXPECT_EQ(topology.gpu_id, 53458u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, DeviceCgroupDenialOmitsPeerBacking) {
  WriteProperties("", 2);
  WritePeer(0, 2, 12345, 1);
  WritePeer(1, 8, 54321, 1);
  FailAttributeReads(directory_ / "class/kfd/kfd/topology/nodes/2/gpu_id",
                     EPERM);
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            AMDF_STATUS_OK);
  EXPECT_GT(read_failure_.attempts, 0u);
  ASSERT_EQ(topology.memory_peers.count, 1u);
  EXPECT_EQ(topology.memory_peers.gpu_ids[0], 54321u);
  amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
}

TEST_F(KfdTopologyTest, DeniedEndpointAndNativeReadFailuresPreserveOutput) {
  WriteProperties("");
  for (const int error : {EPERM, EACCES, EIO}) {
    SCOPED_TRACE(error);
    FailAttributeReads(directory_ / "class/kfd/kfd/topology/nodes/6/gpu_id",
                       error);
    amdf_gpu_kfd_topology_t topology;
    std::memset(&topology, 0xA5, sizeof(topology));
    const auto original = topology;
    EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              error == EPERM
                  ? amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)
                  : amdf_linux_error(error));
    EXPECT_GT(read_failure_.attempts, 0u);
    EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  }
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

TEST_F(KfdTopologyTest, DiscoversNewPciAndGfxIdentitiesWithoutMemoryTables) {
  endpoint_.info.pci.device_id = 0xFFFF;
  for (uint32_t gfx_target_version : {90402u, 130000u}) {
    SCOPED_TRACE(gfx_target_version);
    WriteProperties("", 0, gfx_target_version);
    amdf_gpu_endpoint_profile_t* profile = nullptr;
    ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                  &endpoint_, amdf_allocator_system(), &profile),
              AMDF_STATUS_OK);
    EXPECT_EQ(profile->info.gfx_ip.major, gfx_target_version / 10000);
    EXPECT_EQ(profile->info.compute.compute_unit_count, 304u);
    amdf_free(amdf_allocator_system(), profile);
  }
}

TEST_F(KfdTopologyTest, VirtualFunctionsKeepOnlyHostPublishedSystemRings) {
  const auto gc =
      directory_ / "dev/char/226:160/device/ip_discovery/die/0/11/0";
  std::filesystem::create_directories(gc);
  WriteAttribute(gc / "major", "11\n");
  WriteAttribute(gc / "minor", "0\n");
  WriteAttribute(gc / "revision", "3\n");
  const auto sdma =
      directory_ / "dev/char/226:160/device/ip_discovery/die/0/42/0";
  WriteAttribute(sdma / "major", "6\n");
  WriteAttribute(sdma / "minor", "0\n");
  WriteAttribute(sdma / "revision", "1\n");
  WriteAttribute(directory_ / "class/kfd/kfd/topology/nodes/6/properties",
                 "drm_render_minor 160\nvendor_id 4098\ndevice_id 29857\n"
                 "capability 2893521536\nsimd_count 216\nsimd_per_cu 4\n"
                 "max_waves_per_simd 8\nmax_slots_scratch_cu 32\n"
                 "wave_front_size 32\nlds_size_in_kb 64\narray_count 3\n"
                 "simd_arrays_per_engine 1\nnum_xcc 1\nnum_cp_queues 24\n"
                 "gfx_target_version 110001\np2p_links_count 0\n"
                 "cwsr_size 4096\nctl_stack_size 4096\nnum_sdma_engines 2\n"
                 "num_sdma_xgmi_engines 0\nnum_sdma_queues_per_engine 8\n");
  for (uint64_t mode : {AMDGPU_IDS_FLAGS_MODE_PF, AMDGPU_IDS_FLAGS_MODE_VF,
                        AMDGPU_IDS_FLAGS_MODE_PT}) {
    SCOPED_TRACE(mode);
    native_.info.ids_flags = mode << AMDGPU_IDS_FLAGS_MODE_SHIFT;
    amdf_gpu_endpoint_profile_t* profile = nullptr;
    ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                  &endpoint_, amdf_allocator_system(), &profile),
              AMDF_STATUS_OK);
    EXPECT_EQ(profile->queue_family_count, 3u);
    amdf_free(amdf_allocator_system(), profile);

    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    int descriptor = -1;
    ASSERT_EQ(amdf_linux_endpoint_open_file(&endpoint_, &descriptor),
              AMDF_STATUS_OK);
    ASSERT_EQ(amdf_gpu_kfd_topology_query_memory(
                  descriptor, endpoint_.info.pci.device_id, &topology),
              AMDF_STATUS_OK);
    const auto placement =
        amdf_gpu_kfd_target_user_queue_placement(&topology, 4096);
    EXPECT_EQ(placement.placement_capabilities,
              AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
                  (mode != AMDGPU_IDS_FLAGS_MODE_VF
                       ? AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL
                       : 0));
    EXPECT_EQ(placement.host_publication[AMDF_QUEUE_PLACEMENT_HOST_LOCAL],
              AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN);
    EXPECT_EQ(placement.host_publication[AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL] ==
                  AMDF_QUEUE_HOST_PUBLICATION_FLAG_UNKNOWN,
              mode == AMDGPU_IDS_FLAGS_MODE_VF);
    EXPECT_EQ(amdf_linux_file_close(&descriptor), AMDF_STATUS_OK);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
  EXPECT_EQ(native_.query_count, 3u);
  EXPECT_EQ(native_.open_count, 3u);
}

TEST_F(KfdTopologyTest, DiscoveryDoesNotOpenExecutionFilesOrQueryLiveMemory) {
  WriteProperties("");
  for (bool fail_open : {true, false}) {
    SCOPED_TRACE(fail_open);
    native_.open_error = fail_open ? EACCES : 0;
    native_.query_error = fail_open ? 0 : EIO;
    amdf_gpu_endpoint_profile_t* output = nullptr;
    ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                  &endpoint_, amdf_allocator_system(), &output),
              AMDF_STATUS_OK);
    EXPECT_EQ(output->info.compute.compute_unit_count, 304u);
    EXPECT_GT(output->queue_family_count, 0u);
    amdf_free(amdf_allocator_system(), output);
  }
  EXPECT_EQ(native_.query_count, 0u);
  EXPECT_EQ(native_.open_count, 0u);
  EXPECT_EQ(native_.descriptor, -1);
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
  amdf_allocator_t allocator = {
      .user_data = &state,
      .allocate = [](void* user_data, uint64_t length,
                     uint64_t alignment) -> void* {
        auto* state = static_cast<AllocationState*>(user_data);
        std::ofstream stream(state->generation_path);
        stream << "2\n";
        stream.close();
        EXPECT_TRUE(stream.good());
        const auto system = amdf_allocator_system();
        void* pointer = system.allocate(system.user_data, length, alignment);
        if (pointer != nullptr) {
          ++state->live_count;
        }
        return pointer;
      },
      .free =
          [](void* user_data, void* pointer) {
            auto* state = static_cast<AllocationState*>(user_data);
            EXPECT_GT(state->live_count, 0u);
            --state->live_count;
            const auto system = amdf_allocator_system();
            system.free(system.user_data, pointer);
          },
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
    amdf_allocator_t allocator = {
        .user_data = &state,
        .allocate = [](void* user_data, uint64_t length,
                       uint64_t alignment) -> void* {
          auto* state = static_cast<AllocationState*>(user_data);
          if (state->count++ == state->failure_ordinal) {
            return nullptr;
          }
          const auto system = amdf_allocator_system();
          void* pointer = system.allocate(system.user_data, length, alignment);
          if (pointer != nullptr) {
            ++state->live_count;
          }
          return pointer;
        },
        .free =
            [](void* user_data, void* pointer) {
              auto* state = static_cast<AllocationState*>(user_data);
              EXPECT_GT(state->live_count, 0u);
              --state->live_count;
              const auto system = amdf_allocator_system();
              system.free(system.user_data, pointer);
            },
    };
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_endpoint_profile_t*>(uintptr_t{1});
    amdf_gpu_endpoint_profile_t* profile = sentinel;
    const amdf_status_t status =
        amdf_gpu_umd_create_endpoint_profile(&endpoint_, allocator, &profile);
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

TEST_F(KfdTopologyTest, HostAtomicWidthsFollowEveryCpuRoute) {
  WriteProperties("io_links_count 3\n");
  WriteIoLink(0, 0, 0, 1);
  // GPU peer atomic flags cannot qualify or restrict the CPU route.
  WriteIoLink(2, 8, 42, 13);
  for (uint32_t flags : {0u, 1u, 3u, 5u, 9u, 13u}) {
    SCOPED_TRACE(flags);
    WriteIoLink(1, 1, 0, flags);
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_EQ(topology.host_atomics.supports_32, (flags & 5u) == 1u);
    EXPECT_EQ(topology.host_atomics.supports_64, (flags & 9u) == 1u);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, MissingCpuRouteDoesNotImplyAtomicSupport) {
  for (const char* properties :
       {"", "io_links_count 0\n", "io_links_count 1\n"}) {
    SCOPED_TRACE(properties);
    WriteProperties(properties);
    WriteIoLink(0, 8, 42, 1);
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_FALSE(topology.host_atomics.supports_32);
    EXPECT_FALSE(topology.host_atomics.supports_64);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, DeviceCgroupDeniedGpuIsNotACpuRoute) {
  // An enabled XGMI GPU edge cannot provide CPU atomics or cached VRAM, even
  // when the destination's gpu_id is unreadable under the job's device cgroup.
  WriteIoLink(0, 8, 42, 1, 11);
  WriteIoLink(1, 0, 0, 1);
  FailAttributeReads(directory_ / "class/kfd/kfd/topology/nodes/8/gpu_id",
                     EPERM);
  for (const bool cpu_route : {false, true}) {
    SCOPED_TRACE(cpu_route);
    WriteProperties(cpu_route ? "io_links_count 2\n" : "io_links_count 1\n");
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                  &endpoint_, amdf_allocator_system(), &topology),
              AMDF_STATUS_OK);
    EXPECT_GT(read_failure_.attempts, 0u);
    EXPECT_EQ(topology.host_atomics.supports_32, cpu_route);
    EXPECT_EQ(topology.host_atomics.supports_64, cpu_route);
    EXPECT_FALSE(topology.vram.host_cached);
    amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
  }
}

TEST_F(KfdTopologyTest, CachedVramRequiresEnabledCpuXgmiRoute) {
  WriteProperties("io_links_count 1\n");
  // Native link types: PCIe = 2, XGMI = 11. A GPU peer is not a CPU route.
  for (uint32_t type : {2u, 11u}) {
    for (uint32_t gpu_id : {0u, 42u}) {
      for (uint32_t flags : {0u, 1u}) {
        SCOPED_TRACE(::testing::Message()
                     << type << "/" << gpu_id << "/" << flags);
        WriteIoLink(0, 8, gpu_id, flags, type);
        amdf_gpu_kfd_topology_t topology = {};
        ASSERT_EQ(amdf_gpu_kfd_topology_initialize(
                      &endpoint_, amdf_allocator_system(), &topology),
                  AMDF_STATUS_OK);
        EXPECT_EQ(topology.vram.host_cached,
                  type == 11 && gpu_id == 0 && flags == 1);
        amdf_gpu_kfd_topology_deinitialize(&topology, amdf_allocator_system());
      }
    }
  }
}

TEST_F(KfdTopologyTest, MissingEndpointLeavesOutputUnchanged) {
  WriteProperties("");
  WriteIoLink(0, 8, 42, 1);
  // A missing endpoint makes discovery inspect every GPU node, independent
  // of the filesystem's directory iteration order.
  endpoint_.info.id.words[0] = (UINT64_C(226) << 32) | 176;
  amdf_gpu_kfd_topology_t topology;
  std::memset(&topology, 0xA5, sizeof(topology));
  const auto original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
}

TEST_F(KfdTopologyTest,
       LocalMemoryObservationIncludesNativeReservationsAndUsage) {
  amdf_memory_scope_observation_t observation = {};
  ASSERT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          &endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      AMDF_STATUS_OK);
  EXPECT_EQ(observation.capacity_byte_length, UINT64_C(2) << 30);
  EXPECT_EQ(observation.available_byte_length, UINT64_C(1) << 30);
  EXPECT_EQ(native_.query_count, 1u);
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);

  native_.memory.vram.heap_usage = UINT64_C(2) << 30;
  ASSERT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          &endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      AMDF_STATUS_OK);
  EXPECT_EQ(observation.available_byte_length, 0u);
}

TEST_F(KfdTopologyTest, FailedLocalMemoryObservationPreservesOutput) {
  amdf_memory_scope_observation_t observation = {
      .capacity_byte_length = UINT64_MAX,
      .available_byte_length = UINT64_MAX,
  };
  const auto expected = observation;
  native_.query_error = EIO;
  EXPECT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          &endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      amdf_linux_error(EIO));
  EXPECT_EQ(std::memcmp(&observation, &expected, sizeof(observation)), 0);
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST_F(KfdTopologyTest, UnifiedMemoryObservationDoesNotOpenAnExecutionFile) {
  amdf_memory_scope_observation_t observation = {};
  native_.open_error = EACCES;
  ASSERT_EQ(amdf_gpu_umd_sample_memory_scope(
                &endpoint_,
                AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                    AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES,
                &observation),
            AMDF_STATUS_OK);
  EXPECT_GT(observation.capacity_byte_length, 0u);
  EXPECT_LE(observation.available_byte_length,
            observation.capacity_byte_length);
  EXPECT_EQ(native_.query_count, 0u);
  EXPECT_EQ(native_.descriptor, -1);
}

TEST_F(KfdTopologyTest, IncompleteCpuRouteDoesNotPublishTopology) {
  WriteProperties("io_links_count 1\n");
  amdf_gpu_kfd_topology_t topology;
  std::memset(&topology, 0xA5, sizeof(topology));
  const auto original = topology;
  EXPECT_EQ(amdf_gpu_kfd_topology_initialize(
                &endpoint_, amdf_allocator_system(), &topology),
            amdf_linux_error(ENOENT));
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
}

}  // namespace
