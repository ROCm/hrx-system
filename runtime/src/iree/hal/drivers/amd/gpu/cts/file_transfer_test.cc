// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/async/semaphore.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

enum class FileBacking { kPhysicalHost, kVirtualHost, kVirtualPrivate };

class GpuNativeFileTest : public GpuNativeTest,
                          public ::testing::WithParamInterface<FileBacking> {
 protected:
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
    if (HasFatalFailure()) {
      return;
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = GetParam() != FileBacking::kPhysicalHost
                          ? UINT64_C(1)
                                << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE
                          : 0,
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
    if (GetParam() == FileBacking::kVirtualPrivate) {
      scope.host = {};
    }
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
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
    pool_.reset();
    EXPECT_TRUE(path_.Remove());
    GpuNativeTest::TearDown();
  }

  void Allocate(size_t length, iree_hal_buffer_t** out_buffer) {
    if (GetParam() == FileBacking::kPhysicalHost) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_, iree_hal_make_pool_reservation_request(length, 64),
          iree_infinite_timeout(), out_buffer));
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
    const uint64_t unit =
        iree_max(source.minimum_granularity,
                 iree_max(capabilities.mapping_granularity,
                          capabilities.reservation_granularity));
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
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_, iree_hal_make_pool_reservation_request(chunk + 2 * unit, unit),
          iree_infinite_timeout(), sources_.back().out()));
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
          &slot));
      const auto binding =
          iree_hal_buffer_native_binding(sources_.back(), slot);
      const auto* native_source =
          static_cast<const iree_hal_amd_fabric_mapping_source_t*>(
              binding.mapping_source.descriptor);
      amdf_memory_info_t native_info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
          .structure_size = sizeof(native_info),
      };
      ASSERT_EQ(
          fabric_->api->memory_query_info(native_source->memory, &native_info),
          AMDF_STATUS_OK);
      direct_ &= (native_info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0;
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
    RecordProperty("physical_sources_host_visible", direct_);
  }

  void ReleaseSources() {
    // The final captured virtual-buffer reference has already been released.
    // Physical sources remain application-owned until that native detach.
    sources_.clear();
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    pool_.reset();
  }

  void Open(const std::vector<uint32_t>& contents, iree_hal_file_t** out_file) {
    IREE_ASSERT_OK(iree_io_file_contents_write(
        path_.path_view(),
        iree_make_const_byte_span(contents.data(), contents.size() * 4),
        iree_allocator_system()));
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE,
        path_.path_view(), iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle, 0,
        out_file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
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
                  uint32_t value) {
    auto* semaphore = timeline_.get();
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const auto binding = iree_hal_make_buffer_ref(buffer, kBufferOffset, 4);
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
  // Explicit backing owners for the execution-only virtual-buffer variant.
  std::vector<Ref<iree_hal_buffer_t>> sources_;
  // Installed source coordinates, reused for explicit replacement and removal.
  std::vector<iree_hal_memory_mapping_t> mappings_;
  // Native backing admits direct I/O even if the virtual view has no CPU grant.
  bool direct_ = true;
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
  std::vector<uint32_t> expected(kLength / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  ASSERT_NO_FATAL_FAILURE(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5, six = 6;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, kLength + 256, &sentinel,
                                     sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &semaphore, &two},
                                     {1, &semaphore, &three}, file, 64, buffer,
                                     kBufferOffset, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  if (direct_) {
    EXPECT_EQ(Publications() - before_read, 2u);
  }
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, three, four, 19));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  const uint64_t before_write = Publications();
  expected[16] += 19;
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &four}, {1, &semaphore, &five}, buffer,
      kBufferOffset, file, 64, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  if (direct_) {
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
    ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, eight, nine, 11));
    IREE_ASSERT_OK(iree_hal_queue_write(
        queues_[0], {1, &semaphore, &nine}, {1, &semaphore, &ten}, buffer,
        kBufferOffset, file, 64, kLength, &barriers_, 0));
    ASSERT_NO_FATAL_FAILURE(Wait(ten));
    expected[16] += 11;
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
  ASSERT_NO_FATAL_FAILURE(Open(expected, file.out()));
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
  ASSERT_NO_FATAL_FAILURE(Open(contents, file.out()));
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

INSTANTIATE_TEST_SUITE_P(PhysicalAndVirtual, GpuNativeFileTest,
                         ::testing::Values(FileBacking::kPhysicalHost,
                                           FileBacking::kVirtualHost,
                                           FileBacking::kVirtualPrivate));

}  // namespace
