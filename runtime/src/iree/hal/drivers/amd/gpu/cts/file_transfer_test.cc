// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/cts/testdata.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::amd::gpu::cts::NativeMemoryGate;
using iree::hal::cts::Ref;

enum class FileBacking {
  kPhysicalHost,
  kVirtualHost,
  kVirtualPrivate,
  kPhysicalRegistered,
  kVirtualRegistered,
  kPhysicalPrivate,
  kNestedPrivate,
  kNestedHost,
  kVirtualMixed,
  kVirtualMixedRegistered,
};

std::string FileTestName(const ::testing::TestParamInfo<FileBacking>& info) {
  constexpr const char* backings[] = {
      "PhysicalHost",          "VirtualHost",       "VirtualPrivate",
      "PhysicalRegistered",    "VirtualRegistered", "PhysicalPrivate",
      "NestedPrivate",         "NestedHost",        "VirtualMixed",
      "VirtualMixedRegistered"};
  return backings[static_cast<size_t>(info.param)];
}

// Owns one native device construction across all file cases with the same
// provisioning contract. Individual cases only borrow these resources and
// keep all mutable file, pool, buffer, and semaphore state local.
class SharedGpuNativeDevice final : public GpuNativeTest {
 public:
  SharedGpuNativeDevice(amdf_native_lifetime_t native_lifetime,
                        iree_async_proactor_pool_options_t proactor_options,
                        iree_hal_amd_fabric_t* fabric)
      : native_lifetime_(native_lifetime) {
    proactor_options_ = proactor_options;
    fabric_ = fabric;
  }

  void Initialize() {
    attempted_ = true;
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      unavailable_ = IsSkipped();
      return;
    }
    native_device_ = context_->device;
    gfx_major_ = context_->target_info.gfx_ip.major;
    gfx_minor_ = context_->target_info.gfx_ip.minor;
    gfx_stepping_ = context_->target_info.gfx_ip.stepping;
    Create();
    ready_ = !HasFatalFailure() && !IsSkipped();
    unavailable_ = IsSkipped();
  }

  void Deinitialize() {
    if (!attempted_) {
      return;
    }
    GpuNativeTest::TearDown();
    attempted_ = false;
    ready_ = false;
  }

  void TestBody() override {}

  bool ready() const { return ready_; }
  bool unavailable() const { return unavailable_; }
  iree_hal_amd_fabric_t* fabric() const { return fabric_; }
  iree_async_proactor_pool_t* proactors() const { return proactors_; }
  iree_async_frontier_tracker_t* tracker() const { return tracker_; }
  iree_hal_device_t* device() const { return device_; }
  iree_hal_device_group_t* group() const { return group_; }
  const std::array<iree_hal_queue_t*, 2>& queues() const { return queues_; }
  amdf_device_t* native_device() const { return native_device_; }
  uint32_t gfx_major() const { return gfx_major_; }
  uint32_t gfx_minor() const { return gfx_minor_; }
  uint32_t gfx_stepping() const { return gfx_stepping_; }

 protected:
  // File transfer uses the device-wide SDMA owner. Select the production
  // default compute family so RDNA exercises PM4 and CDNA exercises AQL.
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_UNKNOWN;
  }

  amdf_native_lifetime_t native_lifetime() const override {
    return native_lifetime_;
  }

 private:
  amdf_native_lifetime_t native_lifetime_;
  amdf_device_t* native_device_ = nullptr;
  uint32_t gfx_major_ = 0;
  uint32_t gfx_minor_ = 0;
  uint32_t gfx_stepping_ = 0;
  bool attempted_ = false;
  bool ready_ = false;
  bool unavailable_ = false;
};

class SharedFileDeviceCache {
 public:
  SharedFileDeviceCache(NativeMemoryGate* memory_gate,
                        amdf_native_lifetime_t native_lifetime,
                        uint64_t maximum_native_allocation,
                        iree_async_proactor_pool_options_t proactor_options)
      : memory_gate_(memory_gate),
        native_lifetime_(native_lifetime),
        maximum_native_allocation_(maximum_native_allocation),
        proactor_options_(proactor_options) {}

  iree_status_t Acquire(SharedGpuNativeDevice** out_device) {
    Activate();
    if (!device_) {
      iree_hal_amd_fabric_t* fabric = nullptr;
      IREE_RETURN_IF_ERROR(memory_gate_->CreateFabric(
          &fabric, maximum_native_allocation_, native_lifetime_));
      device_ = std::make_unique<SharedGpuNativeDevice>(
          native_lifetime_, proactor_options_, fabric);
      device_->Initialize();
    }
    *out_device = device_.get();
    return iree_ok_status();
  }

  NativeMemoryGate& memory_gate() { return *memory_gate_; }

  void Activate() { memory_gate_->Activate(maximum_native_allocation_); }

  void Reset() {
    Activate();
    if (device_) {
      device_->Deinitialize();
      device_.reset();
    }
  }

 private:
  NativeMemoryGate* memory_gate_;
  amdf_native_lifetime_t native_lifetime_;
  uint64_t maximum_native_allocation_;
  iree_async_proactor_pool_options_t proactor_options_;
  std::unique_ptr<SharedGpuNativeDevice> device_;
};

class FileDeviceCacheEnvironment final : public ::testing::Environment {
 public:
  FileDeviceCacheEnvironment()
      : process_(&memory_gate_, AMDF_NATIVE_LIFETIME_PROCESS, UINT64_MAX,
                 iree_async_proactor_pool_options_default()),
        instance_(&memory_gate_, AMDF_NATIVE_LIFETIME_INSTANCE, UINT64_MAX,
                  iree_async_proactor_pool_options_default()),
        limited_(&memory_gate_, AMDF_NATIVE_LIFETIME_PROCESS, 8 * 1024 * 1024,
                 iree_async_proactor_pool_options_default()) {}

  SharedFileDeviceCache& process() { return process_; }
  SharedFileDeviceCache& instance() { return instance_; }
  SharedFileDeviceCache& limited() { return limited_; }

  void TearDown() override {
    limited_.Reset();
    instance_.Reset();
    process_.Reset();
    memory_gate_.Activate();
    const auto usage = memory_gate_.QueryUsage();
    EXPECT_EQ(usage.live_bytes, 0u);
    EXPECT_EQ(usage.destroy_failures, 0u);
    memory_gate_.Deactivate();
  }

 private:
  NativeMemoryGate memory_gate_;
  SharedFileDeviceCache process_;
  SharedFileDeviceCache instance_;
  SharedFileDeviceCache limited_;
};

FileDeviceCacheEnvironment* file_device_caches = nullptr;

::testing::Environment* RegisterFileDeviceCaches() {
  auto* environment = new FileDeviceCacheEnvironment;
  file_device_caches = environment;
  return ::testing::AddGlobalTestEnvironment(environment);
}

[[maybe_unused]] ::testing::Environment* file_device_cache_environment =
    RegisterFileDeviceCaches();

class GpuNativeFileTest : public GpuNativeTest,
                          public ::testing::WithParamInterface<FileBacking> {
 protected:
  FileBacking backing_kind() const { return GetParam(); }
  bool is_physical() const {
    return backing_kind() == FileBacking::kPhysicalHost ||
           backing_kind() == FileBacking::kPhysicalRegistered ||
           backing_kind() == FileBacking::kPhysicalPrivate || is_nested();
  }

  bool is_nested() const {
    return backing_kind() == FileBacking::kNestedPrivate ||
           backing_kind() == FileBacking::kNestedHost;
  }

  bool is_registered() const {
    return backing_kind() == FileBacking::kPhysicalRegistered ||
           backing_kind() == FileBacking::kVirtualRegistered ||
           backing_kind() == FileBacking::kVirtualMixedRegistered;
  }

  bool is_mixed() const {
    return backing_kind() == FileBacking::kVirtualMixed ||
           backing_kind() == FileBacking::kVirtualMixedRegistered;
  }

  virtual SharedFileDeviceCache& device_cache() const {
    return file_device_caches->process();
  }

  void SetUp() override {
    device_cache_ = &device_cache();
    IREE_ASSERT_OK(device_cache_->Acquire(&shared_device_));
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    if (shared_device_->unavailable()) {
      GTEST_SKIP() << "Native AMD GPU packet family is unavailable";
    }
    ASSERT_TRUE(shared_device_->ready());
    device_cache_->Activate();
    IREE_ASSERT_OK(iree_hal_device_trim(shared_device_->device()));
    fabric_ = shared_device_->fabric();
    proactors_ = shared_device_->proactors();
    tracker_ = shared_device_->tracker();
    device_ = shared_device_->device();
    group_ = shared_device_->group();
    queues_ = shared_device_->queues();
    native_device_ = shared_device_->native_device();
    RecordProperty("gfx_ip_major", shared_device_->gfx_major());
    RecordProperty("gfx_ip_minor", shared_device_->gfx_minor());
    RecordProperty("gfx_ip_stepping", shared_device_->gfx_stepping());
    iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces =
            (is_physical()
                 ? 0
                 : UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE) |
            (is_registered()
                 ? UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO
                 : 0),
    };
    iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access =
                    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    if (backing_kind() == FileBacking::kVirtualPrivate ||
        backing_kind() == FileBacking::kPhysicalPrivate ||
        backing_kind() == FileBacking::kNestedPrivate || is_registered()) {
      scope.host = {};
    }
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    iree_status_t status = iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out());
    if (is_registered() && iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "Native registered file backing is unavailable";
    }
    IREE_ASSERT_OK(status);
    if (is_mixed()) {
      scope.host = {};
      family.interfaces &=
          ~(UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO);
      IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                               iree_allocator_system(),
                                               device_pool_.out()));
    }
    CreateSemaphore(timeline_.out());
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    IREE_ASSERT_OK(iree::hal::amd::gpu::cts::LookupCodeObject(
        iree_hal_amd_gpu_hip_images_create(),
        iree_hal_amd_gpu_hip_images_size(), targets->targets[0].target_key,
        &load.executable_data));
    IREE_ASSERT_OK(iree_hal_executable_load(family.family, &targets->targets[0],
                                            &load, executable_.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("image_accumulate"), &function_));
  }

  void TearDown() override {
    executable_.reset();
    timeline_.reset();
    sources_.clear();
    nested_pool_.reset();
    device_pool_.reset();
    pool_.reset();
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
    if (shared_device_ && shared_device_->ready()) {
      device_cache_->Activate();
      IREE_EXPECT_OK(iree_hal_device_trim(device_));
    }
    fabric_ = nullptr;
    proactors_ = nullptr;
    tracker_ = nullptr;
    device_ = nullptr;
    group_ = nullptr;
    queues_.fill(nullptr);
    shared_device_ = nullptr;
    device_cache_ = nullptr;
  }

  void Allocate(size_t length, iree_hal_buffer_t** out_buffer) {
    if (is_physical()) {
      const bool nested = is_nested();
      const uint64_t block_length = iree_device_align(length, 64);
      Ref<iree_hal_buffer_t> backing;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_,
          iree_hal_make_pool_reservation_request(
              block_length + (nested ? 512 : 0), 64),
          iree_infinite_timeout(), backing.out()));
      if (nested) {
        const iree_hal_fixed_block_pool_options_t options = {
            .block_size = block_length,
        };
        IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
            backing, 256, block_length, &options, iree_allocator_system(),
            nested_pool_.out()));
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            nested_pool_, iree_hal_make_pool_reservation_request(length, 64),
            iree_infinite_timeout(), out_buffer));
      } else {
        *out_buffer = backing.release();
      }
      const auto memory = iree_hal_buffer_memory_view(*out_buffer);
      const auto* facts =
          reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
              memory.bindings + memory.contract->binding_layout.binding_count);
      host_visible_ = facts->memory->mapping != nullptr;
      file_io_capable_ = iree_any_bit_set(facts->memory->host_features,
                                          AMDF_HOST_MAPPING_FEATURE_FILE_IO);
      RecordProperty("physical_sources_host_visible", host_visible_);
      RecordProperty("physical_sources_file_io_capable", file_io_capable_);
      if (backing_kind() == FileBacking::kPhysicalPrivate ||
          backing_kind() == FileBacking::kNestedPrivate) {
        EXPECT_EQ(memory.contract->host.access, 0u);
      }
      if (host_visible_) {
        const auto binding = iree_hal_buffer_memory_native_binding(
            &memory, {memory.contract->binding_layout.host_binding_index,
                      IREE_HAL_BUFFER_INTERFACE_HOST});
        EXPECT_EQ(binding.host_pointer - facts->memory->contents.data,
                  nested ? 256 : 0);
      }
      return;
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    const iree_hal_pool_scope_t scope = {1, &family, {}};
    iree_hal_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        group_, scope, iree_allocator_system(), &capabilities));
    ASSERT_NE(capabilities.max_byte_length, 0u);
    const auto source = iree_hal_pool_query_mapping_requirements(pool_);
    uint64_t unit = iree_max(source.minimum_granularity,
                             iree_max(capabilities.mapping_granularity,
                                      capabilities.reservation_granularity));
    if (is_mixed()) {
      unit =
          iree_max(unit, iree_hal_pool_query_mapping_requirements(device_pool_)
                             .minimum_granularity);
    }
    const uint64_t mapped_length = iree_device_align(length, unit);
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = mapped_length + 2 * unit,
        .mapping_granularity = unit,
    };
    Ref<iree_hal_buffer_t> root;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, scope, &options, iree_allocator_system(), root.out()));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        root, unit, mapped_length, iree_allocator_system(), out_buffer));
    EXPECT_EQ((*out_buffer)->host_binding_index,
              IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE);
    EXPECT_EQ((*out_buffer)->memory.contract->host.access, 0u);
    iree_hal_memory_scope_t program_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family.family}, &program_scope));
    const uint64_t chunk = iree_device_align(4 * 1024 * 1024, unit);
    for (uint64_t offset = 0; offset < mapped_length; offset += chunk) {
      const uint64_t count = iree_min(chunk, mapped_length - offset);
      sources_.emplace_back();
      auto* source_pool =
          is_mixed() && (offset / chunk) % 2 ? device_pool_.get() : pool_.get();
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          source_pool,
          iree_hal_make_pool_reservation_request(chunk + 2 * unit, 0),
          iree_infinite_timeout(), sources_.back().out()));
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          source_pool, program_scope, IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
          &slot));
      const auto binding =
          iree_hal_buffer_native_binding(sources_.back(), slot);
      const auto* native_source =
          static_cast<const iree_hal_amd_fabric_mapping_source_t*>(
              binding.mapping_source.descriptor);
      host_visible_ &= native_source->storage->mapping != nullptr;
      file_io_capable_ &=
          iree_any_bit_set(native_source->storage->host_features,
                           AMDF_HOST_MAPPING_FEATURE_FILE_IO);
      mappings_.push_back({
          .target = *out_buffer,
          .scope = program_scope,
          .target_range = {offset, count},
          .source = sources_.back(),
          .source_offset = unit,
          .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      });
    }
    Ref<iree_hal_semaphore_t> mapped;
    CreateSemaphore(mapped.out());
    auto* semaphore = mapped.get();
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {}, {1, &semaphore, &value},
                                      mappings_.size(), mappings_.data(), 0));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(mapped, value, iree_infinite_timeout(), 0));
    RecordProperty("physical_sources_host_visible", host_visible_);
    RecordProperty("physical_sources_file_io_capable", file_io_capable_);
  }

  void ReleaseSources() {
    // The final captured virtual-buffer reference has already been released.
    // Physical sources remain application-owned until that native detach.
    sources_.clear();
    nested_pool_.reset();
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    pool_.reset();
    if (device_pool_) {
      iree_hal_pool_query_stats(device_pool_, &stats);
      EXPECT_EQ(stats.reservation_count, 0u);
      device_pool_.reset();
    }
  }

  iree_status_t Open(const std::vector<uint32_t>& contents,
                     iree_hal_file_t** out_file, bool direct = false,
                     iree_async_proactor_t* proactor = nullptr) {
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        path_.path_view(),
        iree_make_const_byte_span(contents.data(), contents.size() * 4),
        iree_allocator_system()));
    iree_io_file_handle_t* handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE |
            (direct ? IREE_IO_FILE_MODE_DIRECT : 0),
        path_.path_view(), iree_allocator_system(), &handle));
    iree_status_t status =
        proactor
            ? iree_hal_fd_file_from_handle(
                  IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                  handle, proactor, iree_allocator_system(), out_file)
            : iree_hal_file_import(
                  device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                  IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                  handle, 0, out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  void ExpectFile(const std::vector<uint32_t>& expected) {
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        path_.path_view(), iree_allocator_system(), &contents));
    EXPECT_EQ(contents->buffer.data_length, expected.size() * 4);
    if (contents->buffer.data_length == expected.size() * 4) {
      EXPECT_EQ(std::memcmp(contents->buffer.data, expected.data(),
                            expected.size() * 4),
                0);
    }
    iree_io_file_contents_free(contents);
  }

  void Accumulate(iree_hal_buffer_t* buffer, uint64_t before, uint64_t after,
                  uint32_t value, uint64_t buffer_offset = kBufferOffset) {
    auto* semaphore = timeline_.get();
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const auto binding = iree_hal_make_buffer_ref(buffer, buffer_offset, 4);
    const iree_hal_queue_barriers_t barriers = {nullptr, &empty_};
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {1, &semaphore, &before}, {1, &semaphore, &after},
        executable_, function_, config,
        iree_make_const_byte_span(&value, sizeof(value)), {1, &binding},
        &barriers, 0));
  }

  void Wait(uint64_t value) {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(timeline_, value, iree_infinite_timeout(), 0));
  }

  uint64_t Publications() {
    iree_async_single_frontier_t frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        reinterpret_cast<iree_async_semaphore_t*>(timeline_.get()),
        iree_async_single_frontier_as_frontier(&frontier), 1);
    EXPECT_EQ(count, 1);
    return count == 1 ? frontier.entries[0].epoch : 0;
  }

  void BufferedRoundTrip(size_t length);

  // Logical range begins inside the allocation, independently of file offset.
  static constexpr size_t kBufferOffset = 128;
  // Small non-page-multiple range used by tests whose contract is independent
  // of physical staging-window growth.
  static constexpr size_t kCorrectnessLength = 64 * 1024 + 12;
  // Crosses the 4 MiB virtual-source partition and the 2 MiB staging window.
  static constexpr size_t kMultiWindowLength = 4 * 1024 * 1024 + 12;
  // Exceeds the native allocation limit while remaining streamable in chunks.
  static constexpr size_t kLimitedBackingLength = 8 * 1024 * 1024 + 12;
  // Process-owned device selected by the fixture's construction contract.
  SharedGpuNativeDevice* shared_device_ = nullptr;
  // Cache owning the shared device and its native observation policy.
  SharedFileDeviceCache* device_cache_ = nullptr;
  // Native identity borrowed from the context transferred into the HAL device.
  amdf_device_t* native_device_ = nullptr;
  // Native file owns its path through independent output verification.
  iree::testing::TempFilePath path_{"iree_gpu_file_transfer"};
  // Explicit host/device backing contract, released after captured uses end.
  Ref<iree_hal_pool_t> pool_;
  // Alternating device-preferred backing for mixed virtual-buffer sources.
  Ref<iree_hal_pool_t> device_pool_;
  // Finite allocator retaining a nonzero range of an ordinary physical buffer.
  Ref<iree_hal_pool_t> nested_pool_;
  // Explicit backing owners for the execution-only virtual-buffer variant.
  std::vector<Ref<iree_hal_buffer_t>> sources_;
  // Installed source coordinates, reused for explicit replacement and removal.
  std::vector<iree_hal_memory_mapping_t> mappings_;
  // Every physical source exposes a CPU mapping, independent of file access.
  bool host_visible_ = true;
  // Every physical source exposes a native file-capable view, independent of
  // retained registration and public host-mapping permission.
  bool file_io_capable_ = true;
  // Public operation timeline, independent of private native handoff points.
  Ref<iree_hal_semaphore_t> timeline_;
  // Real shader that leaves dirty lines before file access.
  Ref<iree_hal_executable_t> executable_;
  // One-word read/modify/write entry point.
  iree_hal_executable_function_t function_ = {};
  // Explicitly omits optional caller cache work.
  const iree_hal_barrier_list_t empty_ = {};
  // Internal file handoffs must remain correct with both lists empty.
  const iree_hal_queue_barriers_t barriers_ = {&empty_, &empty_};
};

void GpuNativeFileTest::BufferedRoundTrip(size_t length) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(length + 256, buffer.out()));
  const uint64_t accumulate_offset =
      is_mixed() ? mappings_.back().target_range.byte_offset + kBufferOffset
                 : kBufferOffset;
  const size_t accumulated_index =
      16 + (accumulate_offset - kBufferOffset) / sizeof(uint32_t);
  std::vector<uint32_t> expected(length / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5, six = 6;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, length + 256, &sentinel,
                                     sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7, accumulate_offset));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &semaphore, &two},
                                     {1, &semaphore, &three}, file, 64, buffer,
                                     kBufferOffset, length, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  RecordProperty("read_queue_publications", Publications() - before_read);
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_read, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(
      Accumulate(buffer, three, four, 19, accumulate_offset));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  const uint64_t before_write = Publications();
  expected[accumulated_index] += 19;
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &four}, {1, &semaphore, &five}, buffer,
      kBufferOffset, file, 64, length, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  RecordProperty("write_queue_publications", Publications() - before_write);
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_write, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));

  // Check both allocation guards through the public GPU transfer path.
  std::vector<uint32_t> observed((length + 256) / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &five}, {1, &semaphore, &six}, buffer, 0,
      observed.data(), observed.size() * 4, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(six));
  for (size_t i = 0; i < observed.size(); ++i) {
    ASSERT_EQ(observed[i],
              i >= kBufferOffset / 4 && i < (kBufferOffset + length) / 4
                  ? expected[i - kBufferOffset / 4 + 16]
                  : sentinel)
        << i;
  }
  if (!mappings_.empty()) {
    // Rebind the same virtual coordinates to different physical sources and
    // offsets. A file implementation using stale source metadata would modify
    // the old slabs while this shader and write-back use the new page tables.
    uint64_t seven = 7, eight = 8, nine = 9, ten = 10, eleven = 11, twelve = 12;
    for (size_t i = 0; i < mappings_.size(); ++i) {
      mappings_[i].source = sources_[(i + 1) % sources_.size()];
      mappings_[i].source_offset *= 2;
      mappings_[i].operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE;
    }
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &semaphore, &six},
                                      {1, &semaphore, &seven}, mappings_.size(),
                                      mappings_.data(), 0));
    IREE_ASSERT_OK(iree_hal_queue_read(
        queues_[0], {1, &semaphore, &seven}, {1, &semaphore, &eight}, file, 64,
        buffer, kBufferOffset, length, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(
        Accumulate(buffer, eight, nine, 11, accumulate_offset));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &nine}, {1, &semaphore, &ten}, buffer,
        kBufferOffset, file, 64, length, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(ten));
    expected[accumulated_index] += 11;
    ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));

    // An unmapped middle range rejects the whole operation during preparation,
    // before an earlier readable prefix can alter the destination file.
    if (mappings_.size() > 1) {
      auto hole = mappings_[1];
      hole.source = nullptr;
      hole.source_offset = 0;
      hole.access = 0;
      hole.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
      IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &semaphore, &ten},
                                        {1, &semaphore, &eleven}, 1, &hole, 0));
      IREE_ASSERT_OK(iree_hal_queue_write(
          queues_[0], {1, &semaphore, &eleven}, {1, &semaphore, &twelve},
          buffer, kBufferOffset, file, 64, length, &barriers_, 0));
      IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                            iree_hal_semaphore_wait(
                                timeline_, twelve, iree_infinite_timeout(), 0));
      ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
    }
  }
}

TEST_P(GpuNativeFileTest, PendingWriteRetiresCapturedBufferBeforeSignal) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kCorrectnessLength + 256, buffer.out()));
  std::vector<uint32_t> expected(kCorrectnessLength / 4, 0);
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  Ref<iree_hal_semaphore_t> gate;
  CreateSemaphore(gate.out());
  auto* gate_handle = gate.get();
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3;
  // Submit the file consumer before the shader producer. File and buffer
  // references disappear from the caller before either operation can run.
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &two}, {1, &semaphore, &three}, buffer,
      kBufferOffset, file, 0, kCorrectnessLength, &barriers_, 0));
  uint32_t value = 0x12345678u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &gate_handle, &one}, {1, &semaphore, &one}, buffer, 0,
      kCorrectnessLength + 256, &value, sizeof(value), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7));
  buffer.reset();
  file.reset();
  executable_.reset();
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(timeline_, &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  ReleaseSources();
  std::fill(expected.begin(), expected.end(), value);
  expected[0] += 7;
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
}

TEST_P(GpuNativeFileTest, TruncatedReadEndsCapturedUsesBeforeFailure) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kCorrectnessLength, buffer.out()));
  Ref<iree_hal_file_t> file;
  std::vector<uint32_t> contents(kCorrectnessLength / 4, 0xABCDu);
  IREE_ASSERT_OK(Open(contents, file.out()));
  Ref<iree_hal_semaphore_t> gate;
  CreateSemaphore(gate.out());
  auto* gate_handle = gate.get();
  auto* semaphore = timeline_.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &gate_handle, &one},
                                     {1, &semaphore, &one}, file, 0, buffer, 0,
                                     kCorrectnessLength, &barriers_, 0));
  buffer.reset();
  file.reset();
  // File length can change after public range validation but before the
  // dependency permits I/O. The accepted read must retire its partial prefix.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(), iree_make_const_byte_span(contents.data(), 4096),
      iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_wait(timeline_, one, iree_infinite_timeout(), 0));
  ReleaseSources();
}

class GpuNativeFileAccessTest : public GpuNativeFileTest {
 protected:
  void SetUp() override {
    GpuNativeFileTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const amdf_memory_device_access_t access = {
        .device = native_device_,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(
        fabric_->api->memory_scope_query_info(fabric_->system_scope, &info),
        AMDF_STATUS_OK);
    for (uint32_t i = 0; i < info.memory_profile_count; ++i) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      const auto status = fabric_->api->memory_scope_query_device_profile(
          fabric_->system_scope, i, 1, &access, &profile, &capabilities);
      if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      direct_staging_available_ |=
          amdf_status_is_ok(profile.availability) &&
          iree_all_bits_set(profile.roles,
                            AMDF_MEMORY_PROFILE_ROLE_CREATE |
                                AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
          iree_all_bits_set(
              profile.supported_flags,
              AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_FILE_IO);
    }
    RecordProperty("direct_staging_available", direct_staging_available_);
  }

  void RoundTrip(bool direct);

  // Exact native capability, including the selected lifetime and host services.
  bool direct_staging_available_ = false;
};

// Instance-scoped native connections can lack file-capable staging even when
// compute works. Exercise both successful transfers and explicit rejection
// according to the exact native capability, followed by independent GPU work.
class GpuInstanceFileAccessTest : public GpuNativeFileAccessTest {
 protected:
  amdf_native_lifetime_t native_lifetime() const override {
    return AMDF_NATIVE_LIFETIME_INSTANCE;
  }

  SharedFileDeviceCache& device_cache() const override {
    return file_device_caches->instance();
  }
};

class GpuNativeGlobalFileTest : public GpuNativeFileTest {};

TEST_P(GpuNativeGlobalFileTest, NestedExecutableGlobalRoundTrip) {
  constexpr size_t kGlobalWords = 2048;
  constexpr size_t kRangeOffset = 4096;
  constexpr size_t kRangeLength = 4096;
  bool found = false;
  iree_hal_executable_global_t token;
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV("image_arena"), &found, &token));
  ASSERT_TRUE(found);
  iree_hal_buffer_t* global = nullptr;
  IREE_ASSERT_OK(
      iree_hal_executable_global_buffer(executable_, token, &global));
  ASSERT_EQ(iree_hal_buffer_byte_length(global),
            kGlobalWords * sizeof(uint32_t));
  const iree_hal_fixed_block_pool_options_t options = {
      .block_size = kRangeLength,
  };
  Ref<iree_hal_pool_t> nested;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      global, kRangeOffset, kRangeLength, &options, iree_allocator_system(),
      nested.out()));
  Ref<iree_hal_buffer_t> view;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      nested, iree_hal_make_pool_reservation_request(kRangeLength, 4),
      iree_infinite_timeout(), view.out()));

  std::vector<uint32_t> contents(16);
  for (size_t i = 0; i < contents.size(); ++i) {
    contents[i] = uint32_t(i * 23 + 7);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(contents, file.out()));
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &semaphore, &one},
                                     file, 8, view, 8, 32, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(view, one, two, 19, 8));
  IREE_ASSERT_OK(iree_hal_queue_write(queues_[0], {1, &semaphore, &two},
                                      {1, &semaphore, &three}, view, 8, file, 4,
                                      32, &barriers_, 0));
  view.reset();
  file.reset();
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  nested.reset();

  constexpr size_t kFirstWord = (kRangeOffset + 8) / sizeof(uint32_t);
  std::vector<uint32_t> expected_global(kGlobalWords, 0);
  std::copy_n(contents.begin() + 2, 8, expected_global.begin() + kFirstWord);
  expected_global[kFirstWord] += 19;
  std::vector<uint32_t> expected_file = contents;
  std::copy_n(expected_global.begin() + kFirstWord, 8,
              expected_file.begin() + 1);
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected_file));
  std::vector<uint32_t> observed_global(kGlobalWords);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &three}, {1, &semaphore, &four}, global, 0,
      observed_global.data(), observed_global.size() * sizeof(uint32_t),
      nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  EXPECT_EQ(observed_global, expected_global);
}

void GpuNativeFileAccessTest::RoundTrip(bool direct) {
  constexpr size_t kAlignment = 4096;
  constexpr size_t kBytes = 64 * 1024 + kAlignment;
  const size_t buffer_offset = kAlignment - (is_nested() ? 256 : 0);
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kBytes + 2 * kAlignment, buffer.out()));
  const bool direct_io_available =
      file_io_capable_ || direct_staging_available_;
  RecordProperty("direct_io_available", direct_io_available);
  std::vector<uint32_t> expected(kBytes / 4 + 2 * kAlignment / 4);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  iree::Status file_status = Open(expected, file.out(), direct);
  if (file_status.code() == iree::StatusCode::kUnavailable) {
    GTEST_SKIP() << iree::hal::cts::FormatSkipStatus(file_status);
  }
  IREE_ASSERT_OK(file_status);
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, kBytes + 2 * kAlignment,
                                     &sentinel, sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(
      queues_[0], {1, &semaphore, &one}, {1, &semaphore, &two}, file,
      kAlignment, buffer, buffer_offset, kBytes, &barriers_, 0));
  if (!direct_io_available) {
    EXPECT_EQ(iree::Status(iree_hal_semaphore_wait(timeline_, two,
                                                   iree_infinite_timeout(), 0))
                  .code(),
              iree::StatusCode::kUnavailable);
    Ref<iree_hal_semaphore_t> write_completion;
    CreateSemaphore(write_completion.out());
    auto* write_semaphore = write_completion.get();
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {}, {1, &write_semaphore, &one}, buffer, buffer_offset,
        file, 0, kBytes, &barriers_, 0));
    file.reset();
    EXPECT_EQ(iree::Status(iree_hal_semaphore_wait(write_completion, one,
                                                   iree_infinite_timeout(), 0))
                  .code(),
              iree::StatusCode::kUnavailable);
    // Failure retires the captured operands without publishing even an
    // eligible prefix. Independent GPU work still advances this queue.
    timeline_.reset();
    CreateSemaphore(timeline_.out());
    semaphore = timeline_.get();
    ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, 0, one, 19, buffer_offset));
    ASSERT_NO_FATAL_FAILURE(Wait(one));
    EXPECT_EQ(Publications() - before_read, 1u);
    std::vector<uint32_t> observed(expected.size());
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &semaphore, &one}, {1, &semaphore, &two}, buffer, 0,
        observed.data(), observed.size() * sizeof(uint32_t), nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(two));
    std::vector<uint32_t> unchanged(expected.size(), sentinel);
    unchanged[buffer_offset / sizeof(uint32_t)] += 19;
    EXPECT_EQ(observed, unchanged);
    ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
    return;
  }
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_read, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, two, three, 19, buffer_offset));
  expected[kAlignment / sizeof(uint32_t)] += 19;
  std::vector<uint32_t> observed(kBytes / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &three}, {1, &semaphore, &four}, buffer,
      buffer_offset, observed.data(), kBytes, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  EXPECT_TRUE(std::equal(observed.begin(), observed.end(),
                         expected.begin() + kAlignment / 4));
  const uint64_t before_write = Publications();
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &four}, {1, &semaphore, &five}, buffer,
      buffer_offset, file, 0, kBytes, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_write, 2u);
  }
  std::copy(observed.begin(), observed.end(), expected.begin());
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
}

TEST_P(GpuNativeFileAccessTest, DirectFileRoundTripThroughGpuBacking) {
  ASSERT_NO_FATAL_FAILURE(RoundTrip(/*direct=*/true));
}

TEST_P(GpuNativeFileAccessTest, BufferedFileRoundTripThroughGpuBacking) {
  ASSERT_NO_FATAL_FAILURE(RoundTrip(/*direct=*/false));
}

TEST_P(GpuInstanceFileAccessTest, DirectFileRoundTripThroughGpuBacking) {
  ASSERT_NO_FATAL_FAILURE(RoundTrip(/*direct=*/true));
}

TEST_P(GpuInstanceFileAccessTest, BufferedFileRoundTripThroughGpuBacking) {
  ASSERT_NO_FATAL_FAILURE(RoundTrip(/*direct=*/false));
}

class GpuNativeFileProgressTest : public GpuNativeFileTest {
 protected:
  NativeMemoryGate& memory_gate() { return device_cache().memory_gate(); }
};

TEST_P(GpuNativeFileProgressTest,
       PreparedSmallReadsProgressDuringStagingGrowth) {
  constexpr size_t kSmall = 4096;
  constexpr size_t kLarge = 128 * 1024;
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(
      Allocate(kBufferOffset + kSmall + kLarge, buffer.out()));
  if (file_io_capable_) {
    GTEST_SKIP() << "This backing uses direct file I/O and needs no staging";
  }
  std::vector<uint32_t> expected((kLarge + kSmall) / 4);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  auto* small = timeline_.get();
  uint64_t one = 1, two = 2, three = 3;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &small, &one}, file, 0,
                                     buffer, kBufferOffset, kSmall, &barriers_,
                                     0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));

  Ref<iree_hal_semaphore_t> large_completion;
  CreateSemaphore(large_completion.out());
  auto* large = large_completion.get();
  std::vector<uint32_t> observed(kSmall / 4);
  {
    NativeMemoryGate::Hold hold(&memory_gate());
    IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &large, &one}, file,
                                       0, buffer, kBufferOffset + kSmall,
                                       kLarge, &barriers_, 0));
    hold.WaitEntered();
    IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &small, &one},
                                       {1, &small, &two}, file, kLarge, buffer,
                                       kBufferOffset, kSmall, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(two));
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &small, &two}, {1, &small, &three}, buffer,
        kBufferOffset, observed.data(), kSmall, nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(three));
    EXPECT_TRUE(std::equal(observed.begin(), observed.end(),
                           expected.begin() + kLarge / 4));
    uint64_t value = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(large, &value));
    EXPECT_EQ(value, 0u);
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(large, one, iree_infinite_timeout(), 0));
  observed.resize(kLarge / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &large, &one}, {1, &large, &two}, buffer,
      kBufferOffset + kSmall, observed.data(), kLarge, nullptr));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(large, two, iree_infinite_timeout(), 0));
  EXPECT_TRUE(std::equal(observed.begin(), observed.end(), expected.begin()));
}

class GpuNativeFileLimitTest : public GpuNativeFileTest {
 protected:
  SharedFileDeviceCache& device_cache() const override {
    return file_device_caches->limited();
  }
};

TEST_P(GpuNativeFileTest, BufferedReadWriteRoundTrip) {
  ASSERT_NO_FATAL_FAILURE(
      BufferedRoundTrip(is_mixed() ? kMultiWindowLength : kCorrectnessLength));
}

class GpuNativeFileStagingWindowTest : public GpuNativeFileTest {};

TEST_P(GpuNativeFileStagingWindowTest,
       MultiWindowReadWritePreservesInternalCacheHandoffs) {
  ASSERT_NO_FATAL_FAILURE(BufferedRoundTrip(kMultiWindowLength));
}

TEST_P(GpuNativeFileLimitTest, StagingFitsNativeBackingLimit) {
  ASSERT_NO_FATAL_FAILURE(BufferedRoundTrip(kLimitedBackingLength));
}

TEST_P(GpuNativeFileTest, ConcurrentRangesSurviveRingWrapAndTrim) {
  constexpr size_t kLaneCount = 8;
  constexpr size_t kLaneLength = 4096;
  constexpr size_t kLength = kLaneCount * kLaneLength;
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kBufferOffset + kLength, buffer.out()));
  std::vector<uint32_t> expected(kLength / 4);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  std::array<Ref<iree_hal_semaphore_t>, kLaneCount> completions;
  std::array<iree_hal_semaphore_t*, kLaneCount> semaphores;
  std::array<uint64_t, kLaneCount> values;
  for (size_t i = 0; i < kLaneCount; ++i) {
    CreateSemaphore(completions[i].out());
    semaphores[i] = completions[i].get();
  }
  // 1,024 independent copies wrap the primary command ring while reusing a
  // finite slot set. Each cohort explicitly joins every lane: HAL queues do
  // not imply submission order or make a larger signal prove earlier work.
  for (uint64_t round = 0; round < 128; ++round) {
    values.fill(round + 1);
    for (size_t i = 0; i < kLaneCount; ++i) {
      IREE_ASSERT_OK(iree_hal_queue_read(
          queues_[0], {}, {1, &semaphores[i], &values[i]}, file,
          i * kLaneLength, buffer, kBufferOffset + i * kLaneLength, kLaneLength,
          &barriers_, 0));
    }
    if (round == 31) {
      IREE_ASSERT_OK(iree_hal_device_trim(device_));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {kLaneCount, semaphores.data(), values.data()}, iree_infinite_timeout(),
        0));
  }
  std::vector<uint32_t> observed(expected.size());
  auto* timeline = timeline_.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {kLaneCount, semaphores.data(), values.data()},
      {1, &timeline, &one}, buffer, kBufferOffset, observed.data(), kLength,
      nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  EXPECT_EQ(observed, expected);
}

// Holds a real polling callback and joins its final access on scope exit.
class HeldFileOwner {
 public:
  ~HeldFileOwner() {
    if (!submitted_) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_one();
    condition_.wait(lock, [&] { return exited_; });
  }

  iree_status_t Hold(iree_async_proactor_t* proactor) {
    operation_.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    operation_.base.user_data = this;
    operation_.base.completion_fn =
        +[](void* user_data, iree_async_operation_t*, iree_status_t status,
            iree_async_completion_flags_t) {
          IREE_EXPECT_OK(status);
          auto* self = static_cast<HeldFileOwner*>(user_data);
          std::unique_lock<std::mutex> lock(self->mutex_);
          self->entered_ = true;
          self->condition_.notify_one();
          self->condition_.wait(lock, [&] { return self->released_; });
          self->exited_ = true;
          self->condition_.notify_one();
        };
    IREE_RETURN_IF_ERROR(
        iree_async_proactor_submit_one(proactor, &operation_.base));
    submitted_ = true;
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
    return iree_ok_status();
  }

 private:
  // Accepted operation and callback storage, joined before destruction.
  iree_async_nop_operation_t operation_ = {};
  // Serializes gate publication and the callback's final access.
  std::mutex mutex_;
  // Publishes entry, release permission, and terminal gate exit.
  std::condition_variable condition_;
  // True after the proactor accepts the operation.
  bool submitted_ = false;
  // The actual polling owner has entered its callback.
  bool entered_ = false;
  // Permission from scope exit to return from the callback.
  bool released_ = false;
  // Callback made its final gate access while holding mutex_.
  bool exited_ = false;
};

// Delays one real registration-detach callback without blocking its poll owner.
// The forwarding table is installed before the ordinary runner starts and
// outlives that runner and all native proactor references.
class FileDetachGate {
 public:
  FileDetachGate()
      : runner_(iree_async_proactor_pool_options_default().runner) {}

  iree_async_proactor_pool_runner_factory_t runner() {
    return {
        .user_data = this,
        .create =
            +[](void* user_data, iree_async_proactor_t* proactor,
                uint32_t node_id, iree_allocator_t allocator,
                void** out_runner) {
              auto* self = static_cast<FileDetachGate*>(user_data);
              self->native_ = proactor->vtable;
              self->forwarding_.vtable = *proactor->vtable;
              self->forwarding_.owner = self;
              self->forwarding_.vtable.detach_buffer = Detach;
              proactor->vtable = &self->forwarding_.vtable;
              return self->runner_.create(self->runner_.user_data, proactor,
                                          node_id, allocator, out_runner);
            },
        .request_stop =
            +[](void* user_data, void* runner) {
              auto* self = static_cast<FileDetachGate*>(user_data);
              self->runner_.request_stop(self->runner_.user_data, runner);
            },
        .destroy =
            +[](void* user_data, void* runner) {
              auto* self = static_cast<FileDetachGate*>(user_data);
              self->runner_.destroy(self->runner_.user_data, runner);
            },
    };
  }

  // Reuses the process-lifetime forwarding table after the preceding fixture
  // has released its file owner and joined every intercepted callback.
  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_FALSE(armed_);
    EXPECT_FALSE(closed_);
    EXPECT_EQ(callback_.fn, nullptr);
    IREE_EXPECT_OK(status_);
    iree_status_ignore(status_);
    armed_ = false;
    closed_ = false;
    completed_ = false;
    following_ = false;
    duplicates_ = 0;
    entry_ = nullptr;
    callback_ = {};
    status_ = iree_ok_status();
  }

  class Hold {
   public:
    explicit Hold(FileDetachGate* owner) : owner_(owner) {
      std::lock_guard<std::mutex> lock(owner_->mutex_);
      owner_->armed_ = true;
      owner_->closed_ = true;
    }
    ~Hold() { owner_->Open(); }

   private:
    // Fixture-owned gate, alive until all proactor references are released.
    FileDetachGate* owner_;
  };

  void WaitForCompletedDetach() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return completed_; });
  }

  void WaitForFollowingDetach() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return following_; });
  }

  uint32_t duplicates() {
    std::lock_guard<std::mutex> lock(mutex_);
    return duplicates_;
  }

  void Open() {
    iree_async_buffer_registration_callback_t callback = {};
    iree_status_t status = iree_ok_status();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = false;
      if (completed_) {
        callback = callback_;
        callback_ = {};
        status = status_;
        status_ = iree_ok_status();
      }
    }
    if (callback.fn) {
      callback.fn(callback.user_data, status);
    }
  }

 private:
  static void Detach(iree_async_proactor_t* proactor,
                     iree_async_buffer_registration_entry_t* entry,
                     iree_async_buffer_registration_callback_t callback) {
    auto* self = reinterpret_cast<const Forwarding*>(proactor->vtable)->owner;
    bool held = false;
    {
      std::lock_guard<std::mutex> lock(self->mutex_);
      held = self->armed_;
      if (held) {
        self->armed_ = false;
        self->entry_ = entry;
        self->callback_ = callback;
      } else if (self->entry_) {
        self->following_ = true;
        self->duplicates_ += entry == self->entry_ && self->callback_.fn;
        self->condition_.notify_all();
      }
    }
    self->native_->detach_buffer(
        proactor, entry,
        held ? iree_async_buffer_registration_callback_t{Completed, self}
             : callback);
  }

  static void Completed(void* user_data, iree_status_t status) {
    auto* self = static_cast<FileDetachGate*>(user_data);
    bool deliver = false;
    {
      std::lock_guard<std::mutex> lock(self->mutex_);
      self->status_ = status;
      self->completed_ = true;
      deliver = !self->closed_;
      self->condition_.notify_all();
    }
    if (deliver) {
      self->Open();
    }
  }

  struct Forwarding {
    // Native operations with only registration detachment intercepted.
    iree_async_proactor_vtable_t vtable = {};
    // Fixture owning the callback hold and original native table.
    FileDetachGate* owner = nullptr;
  };
  // Installed before polling and retained through native proactor destruction.
  Forwarding forwarding_;
  // Original backend operations; all work still uses the real proactor.
  const iree_async_proactor_vtable_t* native_ = nullptr;
  // Ordinary runner creation and teardown, including its locality contract.
  iree_async_proactor_pool_runner_factory_t runner_;
  // Serializes callback ownership and test observations.
  std::mutex mutex_;
  // Publishes actual native completion and subsequent detach attempts.
  std::condition_variable condition_;
  // Next detach is intercepted once after Hold begins.
  bool armed_ = false;
  // Prevents delivery of the intercepted completion until Open.
  bool closed_ = false;
  // Native detach has completed and published status_.
  bool completed_ = false;
  // Another detach was requested after the held one.
  bool following_ = false;
  // Number of duplicate requests while the original callback is retained.
  uint32_t duplicates_ = 0;
  // Identity of the intercepted registration, borrowed through completion.
  iree_async_buffer_registration_entry_t* entry_ = nullptr;
  // Original completion, consumed exactly once by Open.
  iree_async_buffer_registration_callback_t callback_ = {};
  // Owned native result until the original callback receives it.
  iree_status_t status_ = iree_ok_status();
};

FileDetachGate& SharedFileDetachGate() {
  // Constructed by the first lifecycle fixture and destroyed only after all
  // global test environments have released their shared devices.
  static FileDetachGate gate;
  return gate;
}

// Independent file owners use the ordinary imported-file contract. The last
// completed file operation permits its caller to stop that file's poll runner.
class GpuSharedFileStagingTest : public GpuNativeFileProgressTest {
 protected:
  void SetUp() override {
    GpuNativeFileTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    detach_gate_ = &SharedFileDetachGate();
    detach_gate_->Reset();
    auto options = extra_proactor_options_;
    options.runner = detach_gate_->runner();
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, options, iree_allocator_system(), &extra_proactors_));
    IREE_ASSERT_OK(
        iree_async_proactor_pool_get(extra_proactors_, 0, &extra_proactor_));
  }

  void TearDown() override {
    GpuNativeFileTest::TearDown();
    iree_async_proactor_pool_release(extra_proactors_);
    extra_proactors_ = nullptr;
    extra_proactor_ = nullptr;
  }

  // Owned independent runner, released after its file operations complete.
  iree_async_proactor_pool_t* extra_proactors_ = nullptr;
  // Capabilities of the independently supplied file I/O owner.
  iree_async_proactor_pool_options_t extra_proactor_options_ =
      iree_async_proactor_pool_options_default();
  // Borrowed independent file I/O owner.
  iree_async_proactor_t* extra_proactor_ = nullptr;
  // Real dependency callback hold, alive through the independent runner.
  // Process-lifetime callback interceptor outliving the shared device.
  FileDetachGate* detach_gate_ = nullptr;
};

class GpuUnregisteredFileOwnerTest : public GpuSharedFileStagingTest {
 public:
  GpuUnregisteredFileOwnerTest() {
    extra_proactor_options_.proactor_options.allowed_capabilities &=
        ~IREE_ASYNC_PROACTOR_CAPABILITY_CONCURRENT_BUFFER_REGISTRATION;
  }
};

TEST_P(GpuUnregisteredFileOwnerTest, SharesBackingWithRegisteredDeviceOwner) {
  constexpr size_t kLength = 4096;
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kBufferOffset + kLength, buffer.out()));
  std::vector<uint32_t> expected(kLength / sizeof(uint32_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> primary_file, extra_file;
  IREE_ASSERT_OK(Open(expected, primary_file.out()));
  IREE_ASSERT_OK(Open(expected, extra_file.out(), false, extra_proactor_));
  auto* timeline = timeline_.get();
  uint64_t one = 1, two = 2, three = 3;
  // Provision staging through the device owner before the independent file
  // owner accesses it without a compatible prepared-registration capability.
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &timeline, &one},
                                     primary_file, 0, buffer, kBufferOffset,
                                     kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  IREE_ASSERT_OK(iree_hal_queue_read(
      queues_[0], {1, &timeline, &one}, {1, &timeline, &two}, extra_file, 0,
      buffer, kBufferOffset, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  std::vector<uint32_t> observed(expected.size());
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &timeline, &two}, {1, &timeline, &three}, buffer,
      kBufferOffset, observed.data(), kLength, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  EXPECT_EQ(observed, expected);
  extra_file.reset();
  iree_async_proactor_pool_release(extra_proactors_);
  extra_proactors_ = nullptr;
  extra_proactor_ = nullptr;
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

TEST_P(GpuSharedFileStagingTest,
       SharedBackingGrowthAndTrimPreservePreparedWork) {
  constexpr uint64_t kSmall = 1024 * 1024;
  constexpr uint64_t kLarge = 4 * 1024 * 1024;
  constexpr uint64_t kTotal = 2 * kSmall + kLarge;
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kBufferOffset + kTotal, buffer.out()));
  if (file_io_capable_) {
    GTEST_SKIP() << "Direct host ranges do not require physical staging";
  }
  std::vector<uint32_t> expected(kTotal / 4);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> primary_file, extra_file;
  IREE_ASSERT_OK(Open(expected, primary_file.out()));
  IREE_ASSERT_OK(Open(expected, extra_file.out(), false, extra_proactor_));
  ASSERT_NE(iree_hal_file_async_handle(primary_file)->proactor,
            iree_hal_file_async_handle(extra_file)->proactor);
  // Provision the oracle before measuring staging. queue_download grows the
  // PM4 argument/readback buffer independently of the file engine's budget.
  const iree_hal_pool_family_access_t readback_family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_pool_scope_t readback_scope = {
      .family_count = 1,
      .families = &readback_family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_slab_pool_options_t readback_options;
  iree_hal_slab_pool_options_initialize(&readback_options);
  Ref<iree_hal_pool_t> readback_pool;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(group_, readback_scope, &readback_options,
                                iree_allocator_system(), readback_pool.out()));
  Ref<iree_hal_buffer_t> readback;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      readback_pool, iree_hal_make_pool_reservation_request(kTotal, 64),
      iree_infinite_timeout(), readback.out()));
  const auto baseline = memory_gate().ResetPeak();
  auto* primary = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &primary, &one},
                                     primary_file, 0, buffer, kBufferOffset,
                                     kSmall, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  const auto prepared = memory_gate().QueryUsage();
  ASSERT_GT(prepared.live_bytes, baseline.live_bytes);
  RecordProperty("staging_file_io_capable",
                 iree_any_bit_set(prepared.last_allocation.flags,
                                  AMDF_MEMORY_FLAG_HOST_FILE_IO));
  const uint64_t granularity =
      prepared.last_allocation.native_allocation_granularity;
  const uint64_t prefix = prepared.last_allocation.source_byte_offset;
  ASSERT_NE(granularity, 0u);
  const uint64_t native_budget =
      (16 * 1024 * 1024 + prefix + granularity - 1) / granularity * granularity;
  Ref<iree_hal_semaphore_t> extra_completion, large_completion;
  CreateSemaphore(extra_completion.out());
  CreateSemaphore(large_completion.out());
  auto* extra = extra_completion.get();
  auto* large = large_completion.get();
  IREE_ASSERT_OK(iree_hal_queue_read(
      queues_[0], {}, {1, &extra, &one}, extra_file, kSmall, buffer,
      kBufferOffset + kSmall, kSmall, &barriers_, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(extra, one, iree_infinite_timeout(), 0));
  const auto shared = memory_gate().QueryUsage();
  EXPECT_EQ(shared.allocation_count, prepared.allocation_count);
  EXPECT_EQ(shared.live_bytes, prepared.live_bytes);
  {
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold native(&memory_gate());
    IREE_ASSERT_OK(iree_hal_queue_read(
        queues_[0], {}, {1, &large, &one}, primary_file, 2 * kSmall, buffer,
        kBufferOffset + 2 * kSmall, kLarge, &barriers_, 0));
    native.WaitEntered();
    {
      HeldFileOwner owner;
      IREE_ASSERT_OK(owner.Hold(extra_proactor_));
      IREE_ASSERT_OK(iree_hal_queue_read(
          queues_[0], {1, &extra, &one}, {1, &extra, &two}, extra_file, kSmall,
          buffer, kBufferOffset + kSmall, kSmall, &barriers_, 0));
      IREE_ASSERT_OK(iree_hal_device_trim(device_));
      IREE_ASSERT_OK(iree_hal_queue_read(
          queues_[0], {1, &primary, &one}, {1, &primary, &two}, primary_file, 0,
          buffer, kBufferOffset, kSmall, &barriers_, 0));
      ASSERT_NO_FATAL_FAILURE(Wait(two));
      EXPECT_EQ(memory_gate().QueryUsage().allocation_count,
                prepared.allocation_count);
      uint64_t value = UINT64_MAX;
      IREE_ASSERT_OK(iree_hal_semaphore_query(large, &value));
      EXPECT_EQ(value, 0u);
    }
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(extra, two, iree_infinite_timeout(), 0));
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(large, one, iree_infinite_timeout(), 0));
  std::vector<uint32_t> observed(expected.size());
  IREE_ASSERT_OK(iree_hal_queue_copy(
      queues_[0], {1, &large, &one}, {1, &primary, &three}, buffer,
      kBufferOffset, readback, 0, kTotal, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(readback, 0, observed.data(), kTotal));
  EXPECT_EQ(observed, expected);
  const auto trimmed = memory_gate().WaitForBytes(baseline.live_bytes);
  EXPECT_EQ(trimmed.destroy_failures, 0u);
  EXPECT_LE(trimmed.peak_bytes - baseline.live_bytes, native_budget);
  RecordProperty("staging_peak_native_bytes",
                 trimmed.peak_bytes - baseline.live_bytes);
  RecordProperty("staging_native_byte_budget", native_budget);
  // A completed trim is one event, not a permanent no-cache policy. Rewarm a
  // small window, then prove another proactor reuses it without native growth.
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &primary, &four},
                                     primary_file, 0, buffer, kBufferOffset,
                                     4096, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  const auto rewarmed = memory_gate().QueryUsage();
  EXPECT_GT(rewarmed.live_bytes, baseline.live_bytes);
  IREE_ASSERT_OK(
      iree_hal_queue_read(queues_[0], {}, {1, &extra, &three}, extra_file, 0,
                          buffer, kBufferOffset + kSmall, 4096, &barriers_, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(extra, three, iree_infinite_timeout(), 0));
  EXPECT_EQ(memory_gate().QueryUsage().allocation_count,
            rewarmed.allocation_count);
  std::vector<uint32_t> small(4096 / 4);
  IREE_ASSERT_OK(iree_hal_queue_copy(
      queues_[0], {1, &extra, &three}, {1, &primary, &five}, buffer,
      kBufferOffset + kSmall, readback, 0, 4096, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(readback, 0, small.data(), 4096));
  EXPECT_TRUE(std::equal(small.begin(), small.end(), expected.begin()));
  // Completion relinquishes the foreign poll owner. Device trim must not
  // require a future detach callback from this now-stopped runner.
  extra_file.reset();
  iree_async_proactor_pool_release(extra_proactors_);
  extra_proactors_ = nullptr;
  extra_proactor_ = nullptr;
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(memory_gate().WaitForBytes(baseline.live_bytes).destroy_failures,
            0u);
}

class GpuSharedVirtualFileStagingTest : public GpuSharedFileStagingTest {};

TEST_P(GpuSharedVirtualFileStagingTest,
       RetiredBackingDetachesOnceBeforeFileCompletion) {
  if (!iree_any_bit_set(
          iree_async_proactor_query_capabilities(extra_proactor_),
          IREE_ASYNC_PROACTOR_CAPABILITY_CONCURRENT_BUFFER_REGISTRATION)) {
    GTEST_SKIP() << "File owner cannot attach prepared buffer pins";
  }
  constexpr uint64_t kSmall = 1024 * 1024;
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kLimitedBackingLength + 256, buffer.out()));
  if (file_io_capable_) {
    GTEST_SKIP() << "Direct host ranges do not require physical staging";
  }
  ASSERT_GE(mappings_.size(), 2u);
  ASSERT_GE(mappings_[1].target_range.byte_offset, kSmall);
  ASSERT_GT(mappings_[1].target_range.byte_length, kSmall);
  const uint64_t offset = mappings_[1].target_range.byte_offset - kSmall;
  const uint64_t length = kSmall + mappings_[1].target_range.byte_length;
  std::vector<uint32_t> expected(length / sizeof(uint32_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> primary_file, extra_file;
  IREE_ASSERT_OK(Open(expected, primary_file.out()));
  IREE_ASSERT_OK(Open(expected, extra_file.out(), false, extra_proactor_));
  auto* primary = timeline_.get();
  uint64_t one = 1, two = 2;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {}, {1, &primary, &one},
                                     primary_file, 0, buffer, offset, kSmall,
                                     &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  if (!iree_any_bit_set(memory_gate().QueryUsage().last_allocation.flags,
                        AMDF_MEMORY_FLAG_HOST_FILE_IO)) {
    GTEST_SKIP() << "Native staging does not provide registered file backing";
  }

  Ref<iree_hal_semaphore_t> completed, allocated;
  CreateSemaphore(completed.out());
  CreateSemaphore(allocated.out());
  auto* completed_handle = completed.get();
  auto* allocated_handle = allocated.get();
  Ref<iree_hal_buffer_t> cold;
  {
    FileDetachGate::Hold detach(detach_gate_);
    // One operation crosses physical ranges: the first uses the prepared small
    // block, and the second grows staging while retaining the first view. Hold
    // old-block detach until final completion also relinquishes the new view.
    IREE_ASSERT_OK(iree_hal_queue_read(
        queues_[0], {}, {1, &completed_handle, &one}, extra_file, 0, buffer,
        offset, length, &barriers_, 0));
    detach_gate_->WaitForCompletedDetach();
    detach_gate_->WaitForFollowingDetach();
    EXPECT_EQ(detach_gate_->duplicates(), 0u);
    uint64_t value = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(completed, &value));
    EXPECT_EQ(value, 0u);

    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold allocation(&memory_gate());
    const auto request = iree_hal_make_pool_reservation_request(4096, 64);
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                         {1, &allocated_handle, &one}, pool_, 1,
                                         &request, cold.out()));
    allocation.WaitEntered();
    detach_gate_->Open();
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(completed, one, iree_infinite_timeout(), 0));
    IREE_ASSERT_OK(iree_hal_semaphore_query(allocated, &value));
    EXPECT_EQ(value, 0u);
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(allocated, one, iree_infinite_timeout(), 0));
  auto* cold_handle = cold.get();
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[1], {1, &allocated_handle, &one},
                              {1, &allocated_handle, &two}, 1, &cold_handle));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(allocated, two, iree_infinite_timeout(), 0));
  std::vector<uint32_t> observed(expected.size());
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &completed_handle, &one}, {1, &primary, &two}, buffer,
      offset, observed.data(), length, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  EXPECT_EQ(observed, expected);
  // Successful file completion has returned the foreign polling lifetime.
  extra_file.reset();
  iree_async_proactor_pool_release(extra_proactors_);
  extra_proactors_ = nullptr;
  extra_proactor_ = nullptr;
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

class GpuNativeRegisteredFileProgressTest : public GpuNativeFileProgressTest {};

TEST_P(GpuNativeRegisteredFileProgressTest,
       ReusedBackingMapsWhileColdRegistrationWaitsForFileOwner) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kCorrectnessLength + 256, buffer.out()));
  auto* timeline = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4;
  auto unmaps = mappings_;
  for (auto& mapping : unmaps) {
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = 0;
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {}, {1, &timeline, &one},
                                    unmaps.size(), unmaps.data(), 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));

  iree_async_proactor_t* file_proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(proactors_, 0, &file_proactor));
  Ref<iree_hal_buffer_t> cold;
  Ref<iree_hal_semaphore_t> acquired;
  CreateSemaphore(acquired.out());
  auto* acquired_handle = acquired.get();
  constexpr uint32_t kPattern = 0x73C04B19;
  {
    HeldFileOwner file_owner;
    IREE_ASSERT_OK(file_owner.Hold(file_proactor));
    {
      iree::hal::amd::gpu::cts::NativeMemoryGate::Hold native(&memory_gate());
      const auto request = iree_hal_make_pool_reservation_request(4096, 64);
      IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                           {1, &acquired_handle, &one}, pool_,
                                           1, &request, cold.out()));
      native.WaitEntered();
      // Mapping reaches the native queue only after the cold allocation has
      // entered its driver call. It cannot pass by winning an admission race.
      IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &timeline, &one},
                                        {1, &timeline, &two}, mappings_.size(),
                                        mappings_.data(), 0));
    }
    // Cold preparation must return without its attachment receipt, allowing
    // these already-owned pages to map and execute while file polling is held.
    ASSERT_NO_FATAL_FAILURE(Wait(two));
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], {1, &timeline, &two}, {1, &timeline, &three}, buffer,
        kBufferOffset, kCorrectnessLength, &kPattern, sizeof(kPattern), nullptr,
        0));
    ASSERT_NO_FATAL_FAILURE(Wait(three));
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(acquired, one, iree_infinite_timeout(), 0));

  std::vector<uint32_t> expected(kCorrectnessLength / 4, 0);
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &timeline, &three}, {1, &timeline, &four}, buffer,
      kBufferOffset, file, 0, kCorrectnessLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  std::fill(expected.begin(), expected.end(), kPattern);
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
  auto* cold_handle = cold.get();
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[1], {1, &acquired_handle, &one},
                              {1, &acquired_handle, &two}, 1, &cold_handle));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(acquired, two, iree_infinite_timeout(), 0));
}

TEST_P(GpuNativeFileProgressTest,
       ReadShaderAndWriteCompleteWhileUnrelatedAllocationIsHeld) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kCorrectnessLength + 256, buffer.out()));
  std::vector<uint32_t> expected(kCorrectnessLength / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  auto read_update_write = [&](uint64_t before) {
    uint64_t read = before + 1, dispatch = before + 2, write = before + 3;
    IREE_ASSERT_OK(iree_hal_queue_read(
        queues_[0], {1, &semaphore, &before}, {1, &semaphore, &read}, file, 64,
        buffer, kBufferOffset, kCorrectnessLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, read, dispatch, 19));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &dispatch}, {1, &semaphore, &write}, buffer,
        kBufferOffset, file, 64, kCorrectnessLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(write));
    expected[16] += 19;
    ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
  };
  // Establish reusable native capacity before modeling another model's growth.
  ASSERT_NO_FATAL_FAILURE(read_update_write(0));
  // Change every source word so the held-allocation cycle must perform its
  // real file read; replaying the already resident bytes cannot pass.
  for (auto& value : expected) {
    value ^= 0xA5F03219u;
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(),
      iree_make_const_byte_span(expected.data(), expected.size() * 4),
      iree_allocator_system()));

  Ref<iree_hal_buffer_t> cold;
  Ref<iree_hal_semaphore_t> allocated;
  CreateSemaphore(allocated.out());
  auto* allocated_handle = allocated.get();
  uint64_t one = 1, two = 2;
  {
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate());
    const auto request = iree_hal_make_pool_reservation_request(4096, 64);
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                         {1, &allocated_handle, &one}, pool_, 1,
                                         &request, cold.out()));
    hold.WaitEntered();
    ASSERT_NO_FATAL_FAILURE(read_update_write(3));
    uint64_t observed = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(allocated, &observed));
    EXPECT_EQ(observed, 0u);
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(allocated, one, iree_infinite_timeout(), 0));
  auto* cold_handle = cold.get();
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[1], {1, &allocated_handle, &one},
                              {1, &allocated_handle, &two}, 1, &cold_handle));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(allocated, two, iree_infinite_timeout(), 0));
}

INSTANTIATE_TEST_SUITE_P(
    PhysicalAndVirtual, GpuNativeFileTest,
    ::testing::Values(
        FileBacking::kPhysicalHost, FileBacking::kVirtualHost,
        FileBacking::kVirtualPrivate, FileBacking::kPhysicalRegistered,
        FileBacking::kVirtualRegistered, FileBacking::kPhysicalPrivate,
        FileBacking::kNestedPrivate, FileBacking::kNestedHost,
        FileBacking::kVirtualMixed, FileBacking::kVirtualMixedRegistered),
    FileTestName);

INSTANTIATE_TEST_SUITE_P(VirtualPrivate, GpuNativeFileStagingWindowTest,
                         ::testing::Values(FileBacking::kVirtualPrivate),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(
    PhysicalAndVirtual, GpuNativeFileProgressTest,
    ::testing::Values(
        FileBacking::kPhysicalHost, FileBacking::kVirtualHost,
        FileBacking::kVirtualPrivate, FileBacking::kPhysicalRegistered,
        FileBacking::kVirtualRegistered, FileBacking::kPhysicalPrivate,
        FileBacking::kNestedPrivate, FileBacking::kNestedHost,
        FileBacking::kVirtualMixed, FileBacking::kVirtualMixedRegistered),
    FileTestName);

INSTANTIATE_TEST_SUITE_P(
    PhysicalAndVirtual, GpuNativeFileAccessTest,
    ::testing::Values(
        FileBacking::kPhysicalHost, FileBacking::kVirtualHost,
        FileBacking::kVirtualPrivate, FileBacking::kPhysicalRegistered,
        FileBacking::kVirtualRegistered, FileBacking::kPhysicalPrivate,
        FileBacking::kNestedPrivate, FileBacking::kNestedHost,
        FileBacking::kVirtualMixed, FileBacking::kVirtualMixedRegistered),
    FileTestName);

INSTANTIATE_TEST_SUITE_P(
    InstancePhysicalAndVirtual, GpuInstanceFileAccessTest,
    ::testing::Values(
        FileBacking::kPhysicalHost, FileBacking::kVirtualHost,
        FileBacking::kVirtualPrivate, FileBacking::kPhysicalRegistered,
        FileBacking::kVirtualRegistered, FileBacking::kPhysicalPrivate,
        FileBacking::kNestedPrivate, FileBacking::kNestedHost,
        FileBacking::kVirtualMixed, FileBacking::kVirtualMixedRegistered),
    FileTestName);

INSTANTIATE_TEST_SUITE_P(PrivateBacking, GpuSharedFileStagingTest,
                         ::testing::Values(FileBacking::kPhysicalPrivate,
                                           FileBacking::kVirtualPrivate),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(PrivateBacking, GpuUnregisteredFileOwnerTest,
                         ::testing::Values(FileBacking::kPhysicalPrivate,
                                           FileBacking::kVirtualPrivate),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(VirtualPrivate, GpuSharedVirtualFileStagingTest,
                         ::testing::Values(FileBacking::kVirtualPrivate),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(RegisteredVirtual, GpuNativeRegisteredFileProgressTest,
                         ::testing::Values(FileBacking::kVirtualRegistered),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(VirtualPrivate, GpuNativeFileLimitTest,
                         ::testing::Values(FileBacking::kVirtualPrivate),
                         FileTestName);

INSTANTIATE_TEST_SUITE_P(Executable, GpuNativeGlobalFileTest,
                         ::testing::Values(FileBacking::kPhysicalHost),
                         FileTestName);

}  // namespace
