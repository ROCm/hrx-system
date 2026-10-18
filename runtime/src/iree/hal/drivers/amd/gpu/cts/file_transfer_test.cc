// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
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
};

class GpuNativeFileTest : public GpuNativeTest,
                          public ::testing::WithParamInterface<FileBacking> {
 protected:
  bool is_physical() const {
    return GetParam() == FileBacking::kPhysicalHost ||
           GetParam() == FileBacking::kPhysicalRegistered ||
           GetParam() == FileBacking::kPhysicalPrivate || is_nested();
  }

  bool is_nested() const {
    return GetParam() == FileBacking::kNestedPrivate ||
           GetParam() == FileBacking::kNestedHost;
  }

  bool is_registered() const {
    return GetParam() == FileBacking::kPhysicalRegistered ||
           GetParam() == FileBacking::kVirtualRegistered;
  }

  bool is_mixed() const { return GetParam() == FileBacking::kVirtualMixed; }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    RecordProperty("gfx_ip_major", gfx_major);
    RecordProperty("gfx_ip_minor", context_->target_info.gfx_ip.minor);
    RecordProperty("gfx_ip_stepping", context_->target_info.gfx_ip.stepping);
    Create();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const iree_hal_pool_family_access_t family = {
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
    if (GetParam() == FileBacking::kVirtualPrivate ||
        GetParam() == FileBacking::kPhysicalPrivate ||
        GetParam() == FileBacking::kNestedPrivate || is_registered()) {
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
      IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                               iree_allocator_system(),
                                               device_pool_.out()));
    }
    CreateSemaphore(timeline_.out());
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    const auto* images = iree_hal_amd_gpu_hip_images_create();
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(images[i].name,
                      gfx_major == 11 ? "gfx11_generic" : "gfx12_generic")) {
        load.executable_data =
            iree_make_const_byte_span(images[i].data, images[i].size);
      }
    }
    ASSERT_NE(load.executable_data.data, nullptr);
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
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
    GpuNativeTest::TearDown();
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
      host_transfer_ = facts->memory->mapping != nullptr;
      file_io_capable_ = iree_any_bit_set(facts->memory->host_features,
                                          AMDF_HOST_MAPPING_FEATURE_FILE_IO);
      RecordProperty("physical_sources_host_visible", host_transfer_);
      RecordProperty("physical_sources_file_io_capable", file_io_capable_);
      if (GetParam() == FileBacking::kPhysicalPrivate ||
          GetParam() == FileBacking::kNestedPrivate) {
        EXPECT_EQ(memory.contract->host.access, 0u);
      }
      if (host_transfer_) {
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
          iree_hal_make_pool_reservation_request(chunk + 2 * unit, unit),
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
      host_transfer_ &= native_source->storage->mapping != nullptr;
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
    RecordProperty("physical_sources_host_visible", host_transfer_);
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
                     iree_hal_file_t** out_file, bool direct = false) {
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
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle, 0,
        out_file);
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

  // Logical range begins inside the allocation, independently of file offset.
  static constexpr size_t kBufferOffset = 128;
  // Several old staging chunks with a non-power-of-two final tail.
  static constexpr size_t kLength = 8 * 1024 * 1024 + 12;
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
  // Prepared native sources admit file transfers without staged GPU copies.
  bool host_transfer_ = true;
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

TEST_P(GpuNativeFileTest, BufferedReadWritePreservesInternalCacheHandoffs) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kLength + 256, buffer.out()));
  const uint64_t accumulate_offset =
      is_mixed() ? mappings_[1].target_range.byte_offset + kBufferOffset
                 : kBufferOffset;
  const size_t accumulated_index =
      16 + (accumulate_offset - kBufferOffset) / sizeof(uint32_t);
  std::vector<uint32_t> expected(kLength / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5, six = 6;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, kLength + 256, &sentinel,
                                     sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7, accumulate_offset));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &semaphore, &two},
                                     {1, &semaphore, &three}, file, 64, buffer,
                                     kBufferOffset, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  RecordProperty("read_queue_publications", Publications() - before_read);
  if (host_transfer_) {
    EXPECT_EQ(Publications() - before_read, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(
      Accumulate(buffer, three, four, 19, accumulate_offset));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  const uint64_t before_write = Publications();
  expected[accumulated_index] += 19;
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &four}, {1, &semaphore, &five}, buffer,
      kBufferOffset, file, 64, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  RecordProperty("write_queue_publications", Publications() - before_write);
  if (host_transfer_) {
    EXPECT_EQ(Publications() - before_write, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));

  // Check both allocation guards through the public GPU transfer path.
  std::vector<uint32_t> observed((kLength + 256) / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &five}, {1, &semaphore, &six}, buffer, 0,
      observed.data(), observed.size() * 4, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(six));
  for (size_t i = 0; i < observed.size(); ++i) {
    EXPECT_EQ(observed[i],
              i >= kBufferOffset / 4 && i < (kBufferOffset + kLength) / 4
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
        buffer, kBufferOffset, kLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(
        Accumulate(buffer, eight, nine, 11, accumulate_offset));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &nine}, {1, &semaphore, &ten}, buffer,
        kBufferOffset, file, 64, kLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(ten));
    expected[accumulated_index] += 11;
    ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));

    // An unmapped middle range rejects the whole operation during preparation,
    // before an earlier readable prefix can alter the destination file.
    auto hole = mappings_[1];
    hole.source = nullptr;
    hole.source_offset = 0;
    hole.access = 0;
    hole.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &semaphore, &ten},
                                      {1, &semaphore, &eleven}, 1, &hole, 0));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &eleven}, {1, &semaphore, &twelve}, buffer,
        kBufferOffset, file, 64, kLength, &barriers_, 0));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_semaphore_wait(timeline_, twelve, iree_infinite_timeout(), 0));
    ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
  }
}

TEST_P(GpuNativeFileTest, PendingWriteRetiresCapturedBufferBeforeSignal) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kLength + 256, buffer.out()));
  std::vector<uint32_t> expected(kLength / 4, 0);
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
      kBufferOffset, file, 0, kLength, &barriers_, 0));
  uint32_t value = 0x12345678u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &gate_handle, &one}, {1, &semaphore, &one}, buffer, 0,
      kLength + 256, &value, sizeof(value), nullptr, 0));
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
  ASSERT_NO_FATAL_FAILURE(Allocate(kLength, buffer.out()));
  Ref<iree_hal_file_t> file;
  std::vector<uint32_t> contents(kLength / 4, 0xABCDu);
  IREE_ASSERT_OK(Open(contents, file.out()));
  Ref<iree_hal_semaphore_t> gate;
  CreateSemaphore(gate.out());
  auto* gate_handle = gate.get();
  auto* semaphore = timeline_.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &gate_handle, &one},
                                     {1, &semaphore, &one}, file, 0, buffer, 0,
                                     kLength, &barriers_, 0));
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

class GpuNativeDirectFileTest : public GpuNativeFileTest {};

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

TEST_P(GpuNativeDirectFileTest, DirectFileRoundTripThroughGpuBacking) {
  constexpr size_t kAlignment = 4096;
  constexpr size_t kBytes = 8 * 1024 * 1024 + kAlignment;
  const size_t buffer_offset = kAlignment - (is_nested() ? 256 : 0);
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kBytes + 2 * kAlignment, buffer.out()));
  std::vector<uint32_t> expected(kBytes / 4 + 2 * kAlignment / 4);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  iree::Status file_status = Open(expected, file.out(), /*direct=*/true);
  if (file_status.code() == iree::StatusCode::kUnavailable) {
    GTEST_SKIP() << file_status.ToString();
  }
  IREE_ASSERT_OK(file_status);
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, kBytes + 2 * kAlignment,
                                     &sentinel, sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(one));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(
      queues_[0], {1, &semaphore, &one}, {1, &semaphore, &two}, file,
      kAlignment, buffer, buffer_offset, kBytes, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_read, 2u);
  }
  std::vector<uint32_t> observed(kBytes / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &two}, {1, &semaphore, &three}, buffer,
      buffer_offset, observed.data(), kBytes, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  EXPECT_TRUE(std::equal(observed.begin(), observed.end(),
                         expected.begin() + kAlignment / 4));
  const uint64_t before_write = Publications();
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &three}, {1, &semaphore, &four}, buffer,
      buffer_offset, file, 0, kBytes, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  if (file_io_capable_) {
    EXPECT_EQ(Publications() - before_write, 2u);
  }
  std::copy(observed.begin(), observed.end(), expected.begin());
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
}

class GpuNativeFileProgressTest : public GpuNativeFileTest {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
    GpuNativeFileTest::SetUp();
  }

  // Native dependency gate, joined by base TearDown before table destruction.
  iree::hal::amd::gpu::cts::NativeMemoryGate memory_gate_;
};

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

class GpuNativeRegisteredFileProgressTest : public GpuNativeFileProgressTest {};

TEST_P(GpuNativeRegisteredFileProgressTest,
       ReusedBackingMapsWhileColdRegistrationWaitsForFileOwner) {
  Ref<iree_hal_buffer_t> buffer;
  ASSERT_NO_FATAL_FAILURE(Allocate(kLength + 256, buffer.out()));
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
      iree::hal::amd::gpu::cts::NativeMemoryGate::Hold native(&memory_gate_);
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
        kBufferOffset, kLength, &kPattern, sizeof(kPattern), nullptr, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(three));
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(acquired, one, iree_infinite_timeout(), 0));

  std::vector<uint32_t> expected(kLength / 4, 0);
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &timeline, &three}, {1, &timeline, &four}, buffer,
      kBufferOffset, file, 0, kLength, &barriers_, 0));
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
  ASSERT_NO_FATAL_FAILURE(Allocate(kLength + 256, buffer.out()));
  std::vector<uint32_t> expected(kLength / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  auto read_update_write = [&](uint64_t before) {
    uint64_t read = before + 1, dispatch = before + 2, write = before + 3;
    IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &semaphore, &before},
                                       {1, &semaphore, &read}, file, 64, buffer,
                                       kBufferOffset, kLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, read, dispatch, 19));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &dispatch}, {1, &semaphore, &write}, buffer,
        kBufferOffset, file, 64, kLength, &barriers_, 0));
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
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
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

INSTANTIATE_TEST_SUITE_P(PhysicalAndVirtual, GpuNativeFileTest,
                         ::testing::Values(FileBacking::kPhysicalHost,
                                           FileBacking::kVirtualHost,
                                           FileBacking::kVirtualPrivate,
                                           FileBacking::kPhysicalRegistered,
                                           FileBacking::kVirtualRegistered,
                                           FileBacking::kPhysicalPrivate,
                                           FileBacking::kNestedPrivate,
                                           FileBacking::kNestedHost,
                                           FileBacking::kVirtualMixed));

INSTANTIATE_TEST_SUITE_P(PhysicalAndVirtual, GpuNativeFileProgressTest,
                         ::testing::Values(FileBacking::kPhysicalHost,
                                           FileBacking::kVirtualHost,
                                           FileBacking::kVirtualPrivate,
                                           FileBacking::kPhysicalRegistered,
                                           FileBacking::kVirtualRegistered,
                                           FileBacking::kPhysicalPrivate,
                                           FileBacking::kNestedPrivate,
                                           FileBacking::kNestedHost,
                                           FileBacking::kVirtualMixed));

INSTANTIATE_TEST_SUITE_P(PhysicalAndVirtual, GpuNativeDirectFileTest,
                         ::testing::Values(FileBacking::kPhysicalHost,
                                           FileBacking::kVirtualHost,
                                           FileBacking::kVirtualPrivate,
                                           FileBacking::kPhysicalRegistered,
                                           FileBacking::kVirtualRegistered,
                                           FileBacking::kPhysicalPrivate,
                                           FileBacking::kNestedPrivate,
                                           FileBacking::kNestedHost,
                                           FileBacking::kVirtualMixed));

INSTANTIATE_TEST_SUITE_P(RegisteredVirtual, GpuNativeRegisteredFileProgressTest,
                         ::testing::Values(FileBacking::kVirtualRegistered));

INSTANTIATE_TEST_SUITE_P(Executable, GpuNativeGlobalFileTest,
                         ::testing::Values(FileBacking::kPhysicalHost));

}  // namespace
