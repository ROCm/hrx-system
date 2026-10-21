// Copyright 2026 The IREE Authors
//
// Licensed under the Apache v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/replay/dump.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/execute_operation.h"
#include "iree/hal/replay/execute_state.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/slab_pool.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &group));
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return group;
}

class ReplayRecorderVirtualBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    storage_.resize(256 * 1024);
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(storage_.data(), storage_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    auto options = iree_hal_replay_recorder_options_default();
    options.external_file_validation =
        IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
    iree_status_t status = iree_hal_replay_recorder_create(
        file_handle, &options, iree_allocator_system(), &recorder_);
    iree_io_file_handle_release(file_handle);
    IREE_ASSERT_OK(status);
    source_group_ = CreateTaskDeviceGroup();
    IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
        recorder_, source_group_, iree_allocator_system(), &group_));
    device_ = iree_hal_device_group_device_at(group_, 0);
    family_ = iree_hal_device_queue_family(device_, 0);
    queue_ = iree_hal_device_queue(device_, 0, 0);
    all_ = {iree_hal_device_group_memory_domain(group_),
            IREE_HAL_MEMORY_SCOPE_ANY};
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, iree_hal_make_queue_family_affinity(0), 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion_));
  }

  void TearDown() override {
    iree_hal_semaphore_release(completion_);
    iree_hal_device_group_release(group_);
    iree_hal_device_group_release(source_group_);
    IREE_EXPECT_OK(iree_hal_replay_recorder_close(recorder_));
    iree_hal_replay_recorder_release(recorder_);
  }

  iree_hal_pool_scope_t Scope(iree_hal_pool_family_access_t* family) {
    *family = {
        .family = family_,
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    return {
        .family_count = 1,
        .families = family,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
                .cacheability = IREE_HAL_HOST_CACHEABILITY_UNKNOWN,
            },
    };
  }

  iree_hal_semaphore_list_t Point(uint64_t* value) {
    return {1, &completion_, value};
  }

  void Wait(uint64_t value) {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        Point(&value), iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  iree_const_byte_span_t Recording() {
    iree_hal_replay_file_header_t header;
    iree_host_size_t offset = 0;
    IREE_CHECK_OK(iree_hal_replay_file_parse_header(
        iree_make_const_byte_span(storage_.data(), storage_.size()), &header,
        &offset));
    return iree_make_const_byte_span(storage_.data(), header.file_length);
  }

  // Serialized capture bytes retained through replay.
  std::vector<uint8_t> storage_;
  // Shared capture session.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Owned native Task group.
  iree_hal_device_group_t* source_group_ = nullptr;
  // Owned recording wrapper with the same scope coordinates.
  iree_hal_device_group_t* group_ = nullptr;
  // Device borrowed from the recording group.
  iree_hal_device_t* device_ = nullptr;
  // Canonical queue family borrowed from device_.
  const iree_hal_queue_family_t* family_ = nullptr;
  // Provisioned queue borrowed from device_.
  iree_hal_queue_t* queue_ = nullptr;
  // Wildcard within the sealed group.
  iree_hal_memory_scope_t all_ = {};
  // Owned timeline ordering every submitted operation in the witness.
  iree_hal_semaphore_t* completion_ = nullptr;
};

TEST_F(ReplayRecorderVirtualBufferTest,
       ReplaysBatchedAliasesAndReplacementToFile) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const std::array<uint32_t, 16> initial_output = {};
  iree::testing::TempFilePath output_path("iree_replay_mapping_output");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(),
      iree_make_const_byte_span(initial_output.data(), sizeof(initial_output)),
      iree_allocator_system()));
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS,
      output_path.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(device_, nullptr,
                                      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                      IREE_HAL_MEMORY_ACCESS_ALL, file_handle,
                                      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_pool_family_access_t family;
  auto scope = Scope(&family);
  family.atomics.queue.device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  family.atomics.program.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  scope.host.atomics.system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, iree_allocator_system(), &capabilities));
  ASSERT_GT(capabilities.mapping_granularity, 0u);
  const iree_device_size_t unit =
      iree_max(64 * 1024, capabilities.reservation_granularity);
  const iree_hal_virtual_buffer_options_t reservation_options = {
      .byte_length = unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  std::array<iree_hal_buffer_t*, 2> reservations = {};
  for (auto& reservation : reservations) {
    IREE_ASSERT_OK(
        iree_hal_virtual_buffer_create(group_, scope, &reservation_options,
                                       iree_allocator_system(), &reservation));
  }
  family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                           iree_allocator_system(), &pool));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, iree_hal_make_queue_family_affinity(0),
                                1, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_list = {1, &gate, &gate_value};
  uint64_t allocated = 1;
  const auto request = iree_hal_make_pool_reservation_request(2 * unit, unit);
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, gate_list, Point(&allocated),
                                       pool, 1, &request, &source));
  iree_hal_buffer_t* source_tail = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      source, unit, unit, iree_allocator_system(), &source_tail));
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = reservations[i],
        .scope = all_,
        .target_range = iree_hal_make_buffer_range(0, unit),
        .source = source_tail,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  uint64_t mapped = 2;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&allocated), Point(&mapped),
                                    mappings.size(), mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  mappings = {};
  Wait(mapped);

  auto* first = reservations[0];
  auto* second = reservations[1];
  iree_hal_buffer_t* alias = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(second, 32, 16, iree_allocator_system(), &alias));
  const uint32_t first_pattern = 0x12345678;
  uint64_t filled = 3;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queue_, Point(&mapped), Point(&filled), first, 32, 16, &first_pattern,
      sizeof(first_pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  // Host writes through an alias are captured under the stable buffer ID.
  const uint32_t alias_pattern = 0xABCDEF01;
  IREE_ASSERT_OK(iree_hal_buffer_map_write(alias, 4, &alias_pattern,
                                           sizeof(alias_pattern)));
  uint64_t first_written = 4;
  IREE_ASSERT_OK(iree_hal_queue_write(queue_, Point(&filled),
                                      Point(&first_written), alias, 0, file, 0,
                                      16, nullptr, IREE_HAL_WRITE_FLAG_NONE));
  Wait(first_written);

  const uint32_t replacement_pattern = 0x87654321;
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source, 0, &replacement_pattern,
                                           sizeof(replacement_pattern)));
  iree_hal_memory_mapping_t replacement = {
      .target = reservations[0],
      .scope = {all_.domain, family_->memory.queue_scope_id},
      .target_range = iree_hal_make_buffer_range(0, unit),
      .source = source,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE,
  };
  uint64_t replaced = 5;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&first_written),
                                    Point(&replaced), 1, &replacement,
                                    IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS));
  replacement.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
  replacement.access = IREE_HAL_MEMORY_ACCESS_NONE;
  uint64_t revoked = 6;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&replaced), Point(&revoked),
                                    1, &replacement,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  replacement.access = IREE_HAL_MEMORY_ACCESS_READ;
  uint64_t restored = 7;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&revoked), Point(&restored),
                                    1, &replacement,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  uint64_t replacement_written = 8;
  IREE_ASSERT_OK(iree_hal_queue_write(
      queue_, Point(&restored), Point(&replacement_written), first, 0, file, 16,
      sizeof(replacement_pattern), nullptr, IREE_HAL_WRITE_FLAG_NONE));
  uint64_t alias_written = 9;
  IREE_ASSERT_OK(iree_hal_queue_write(queue_, Point(&replacement_written),
                                      Point(&alias_written), alias, 0, file, 20,
                                      16, nullptr, IREE_HAL_WRITE_FLAG_NONE));
  replacement.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  replacement.source = nullptr;
  replacement.access = IREE_HAL_MEMORY_ACCESS_NONE;
  uint64_t detached = 10;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&alias_written),
                                    Point(&detached), 1, &replacement,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  uint64_t complete = 11;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&detached), Point(&complete),
                                    0, nullptr, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Wait(complete);

  // Retained views keep addresses live. Checked removal separately establishes
  // when the physical source can be deallocated.
  iree_hal_buffer_retain(first);
  for (auto* reservation : reservations) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(reservation, all_));
    iree_hal_buffer_release(reservation);
  }
  iree_hal_buffer_release(first);
  iree_hal_buffer_release(alias);
  uint64_t deallocated = 12;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue_, Point(&complete),
                                         Point(&deallocated), 1, &source));
  Wait(deallocated);
  iree_hal_buffer_release(source_tail);
  iree_hal_buffer_release(source);
  iree_hal_pool_release(pool);
  iree_hal_semaphore_release(gate);
  iree_hal_file_release(file);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));

  // Construction records preserve independent host, queue, and program
  // contracts.
  const auto recording = Recording();
  iree_hal_replay_file_header_t header;
  iree_host_size_t record_offset = 0;
  IREE_ASSERT_OK(
      iree_hal_replay_file_parse_header(recording, &header, &record_offset));
  unsigned pool_count = 0, reservation_count = 0;
  while (record_offset < header.file_length) {
    iree_hal_replay_file_record_t record;
    IREE_ASSERT_OK(iree_hal_replay_file_parse_record(recording, record_offset,
                                                     &record, &record_offset));
    iree_host_size_t family_offset = 0;
    if (record.header.payload_type ==
        IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_CREATE_SLAB) {
      iree_hal_replay_pool_create_slab_payload_t payload;
      memcpy(&payload, record.payload.data, sizeof(payload));
      EXPECT_EQ(payload.host_atomic_system_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
      family_offset = sizeof(payload);
      ++pool_count;
    } else if (record.header.payload_type ==
               IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE) {
      iree_hal_replay_virtual_buffer_create_payload_t payload;
      memcpy(&payload, record.payload.data, sizeof(payload));
      EXPECT_EQ(payload.host_atomic_system_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
      family_offset = sizeof(payload);
      ++reservation_count;
    }
    if (family_offset) {
      iree_hal_replay_pool_family_access_payload_t payload;
      memcpy(&payload, record.payload.data + family_offset, sizeof(payload));
      EXPECT_EQ(payload.queue_atomic_device_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
      EXPECT_EQ(payload.program_atomic_device_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD);
    }
  }
  EXPECT_EQ(pool_count, 1u);
  EXPECT_EQ(reservation_count, 2u);

  std::array<uint32_t, 16> expected = {};
  expected[0] = expected[2] = expected[3] = first_pattern;
  expected[1] = alias_pattern;
  expected[4] = replacement_pattern;
  std::copy_n(expected.begin(), 4, expected.begin() + 5);
  auto check_output = [&](const char* phase) {
    SCOPED_TRACE(phase);
    iree_io_file_contents_t* actual = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        output_path.path_view(), iree_allocator_system(), &actual));
    ASSERT_EQ(sizeof(expected), actual->const_buffer.data_length);
    std::array<uint32_t, 16> actual_words;
    memcpy(actual_words.data(), actual->const_buffer.data,
           sizeof(actual_words));
    EXPECT_EQ(expected, actual_words);
    iree_io_file_contents_free(actual);
  };
  check_output("capture");
  for (auto format :
       {IREE_HAL_REPLAY_DUMP_FORMAT_TEXT, IREE_HAL_REPLAY_DUMP_FORMAT_JSONL}) {
    const iree_hal_replay_dump_options_t options = {.format = format};
    std::string dump;
    const iree_hal_replay_dump_write_callback_t sink = {
        .fn =
            [](void* user_data, iree_string_view_t text) {
              static_cast<std::string*>(user_data)->append(text.data,
                                                           text.size);
              return iree_ok_status();
            },
        .user_data = &dump,
    };
    IREE_ASSERT_OK(iree_hal_replay_dump_file(Recording(), &options, sink,
                                             iree_allocator_system()));
    EXPECT_NE(dump.find("virtual_buffer.create"), std::string::npos);
    EXPECT_NE(dump.find("virtual_buffer.detach"), std::string::npos);
    EXPECT_NE(dump.find("mapping_count"), std::string::npos);
    EXPECT_NE(dump.find("target_buffer_id"), std::string::npos);
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(),
      iree_make_const_byte_span(initial_output.data(), sizeof(initial_output)),
      iree_allocator_system()));
  auto* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(
      Recording(), replay_group, nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  check_output("replay");
#else
  GTEST_SKIP() << "File output replay requires POSIX file IO.";
#endif
}

TEST_F(ReplayRecorderVirtualBufferTest,
       CapturedPoolViewSurvivesPendingMapping) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  iree_hal_pool_family_access_t family;
  auto scope = Scope(&family);
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 2 * kUnit,
      .min_alignment = kUnit,
      .mapping_granularity = kUnit,
  };
  iree_hal_buffer_t* arena = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, &options, iree_allocator_system(), &arena));
  iree_hal_fixed_block_pool_options_t block_options;
  iree_hal_fixed_block_pool_options_initialize(&block_options);
  block_options.block_size = kUnit;
  block_options.alignment = kUnit;
  iree_hal_pool_t* child_pool = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      arena, kUnit, kUnit, &block_options, iree_allocator_system(),
      &child_pool));
  iree_hal_buffer_t* child = nullptr;
  const auto request = iree_hal_make_pool_reservation_request(kUnit, kUnit);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      child_pool, request, iree_infinite_timeout(), &child));

  family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &pool_options, iree_allocator_system(), &source_pool));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, request, iree_infinite_timeout(), &backing));
  const std::array<uint32_t, 4> expected = {2, 3, 5, 7};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(backing, 0, expected.data(), sizeof(expected)));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, iree_hal_make_queue_family_affinity(0),
                                0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_list = {1, &gate, &gate_value};
  uint64_t mapped = 1;
  const iree_hal_memory_mapping_t mapping = {
      .target = child,
      .scope = all_,
      .target_range = {0, kUnit},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, gate_list, Point(&mapped), 1,
                                    &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  iree_hal_buffer_release(child);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(child_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  // The later submission is the recorded producer of the gate. Submission
  // order supplies no dependency, so mapping remains pending until this call.
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, iree_hal_semaphore_list_empty(),
                                    gate_list, 0, nullptr,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Wait(mapped);
  iree_hal_pool_query_stats(child_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  std::array<uint32_t, 4> observed = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(arena, kUnit, observed.data(),
                                          sizeof(observed)));
  EXPECT_EQ(observed, expected);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(arena, all_));
  iree_hal_pool_release(child_pool);
  iree_hal_buffer_release(arena);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(source_pool);
  iree_hal_semaphore_release(gate);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
}

TEST_F(ReplayRecorderVirtualBufferTest,
       ScopedHostMapsOutliveQueueDeallocation) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  iree::testing::TempFilePath output_path("iree_replay_host_mapping");
  std::array<uint8_t, 64> initial_output = {};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(),
      iree_make_const_byte_span(initial_output.data(), initial_output.size()),
      iree_allocator_system()));
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS,
      output_path.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(device_, nullptr,
                                      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                      IREE_HAL_MEMORY_ACCESS_ALL, file_handle,
                                      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_pool_family_access_t family;
  auto scope = Scope(&family);
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, iree_allocator_system(), &capabilities));
  const auto unit = capabilities.reservation_granularity;
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, &options, iree_allocator_system(), &reservation));
  family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                           iree_allocator_system(), &pool));
  uint64_t allocated = 1;
  auto request = iree_hal_make_pool_reservation_request(2 * unit, unit);
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, iree_hal_semaphore_list_empty(),
                                       Point(&allocated), pool, 1, &request,
                                       &source));
  iree_hal_memory_mapping_t update = {
      .target = reservation,
      .scope = all_,
      .target_range = iree_hal_make_buffer_range(0, unit),
      .source = source,
      .source_offset = unit,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
  };
  uint64_t mapped = 2;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&allocated), Point(&mapped),
                                    1, &update, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Wait(mapped);
  iree_hal_buffer_t* tail = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(source, unit + 32, 64,
                                         iree_allocator_system(), &tail));
  std::array<iree_hal_buffer_mapping_t, 2> host_maps;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      tail, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 64, &host_maps[0]));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      tail, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 16, 16, &host_maps[1]));
  uint64_t deallocated = 3;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue_, Point(&mapped),
                                         Point(&deallocated), 1, &source));
  Wait(deallocated);
  iree_hal_buffer_release(tail);
  iree_hal_buffer_release(source);

  // Both maps now own committed storage independently from the transient view.
  std::memset(host_maps[0].contents.data, 0xA1, 64);
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&host_maps[0], 4, 8));
  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_begin(
      recorder_, IREE_SV("mapped_after_deallocation")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_end(
      recorder_, IREE_SV("mapped_after_deallocation")));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&host_maps[0]));
  std::memset(host_maps[1].contents.data, 0xB2, 16);
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&host_maps[1], 0, 16));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&host_maps[1], 0, 16));
  uint64_t written = 4;
  IREE_ASSERT_OK(iree_hal_queue_write(queue_, Point(&deallocated),
                                      Point(&written), reservation, 32, file, 0,
                                      64, nullptr, IREE_HAL_WRITE_FLAG_NONE));
  update.source = nullptr;
  update.source_offset = 0;
  update.access = IREE_HAL_MEMORY_ACCESS_NONE;
  update.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  uint64_t detached = 5;
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, Point(&written), Point(&detached),
                                    1, &update, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Wait(detached);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&host_maps[1]));
  iree_hal_buffer_release(reservation);
  iree_hal_pool_release(pool);
  iree_hal_file_release(file);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));

  std::array<uint8_t, 64> expected;
  expected.fill(0xA1);
  std::fill_n(expected.begin() + 16, 16, 0xB2);
  auto check_output = [&]() {
    iree_io_file_contents_t* actual = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        output_path.path_view(), iree_allocator_system(), &actual));
    ASSERT_EQ(expected.size(), actual->const_buffer.data_length);
    EXPECT_EQ(0, std::memcmp(expected.data(), actual->const_buffer.data,
                             expected.size()));
    iree_io_file_contents_free(actual);
  };
  check_output();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(),
      iree_make_const_byte_span(initial_output.data(), initial_output.size()),
      iree_allocator_system()));
  auto* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(
      Recording(), replay_group, nullptr, iree_allocator_system()));
  check_output();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(),
      iree_make_const_byte_span(initial_output.data(), initial_output.size()),
      iree_allocator_system()));
  // Abort while both maps and their alias remain live, then replay normally.
  // Cleanup must unmap, close address owners, and finally release storage.
  bool cancelled = false;
  auto execute_options = iree_hal_replay_execute_options_default();
  execute_options.scope_event_callback = {
      .fn =
          [](void* user_data, const iree_hal_replay_scope_event_t* event) {
            *static_cast<bool*>(user_data) = true;
            return iree_status_from_code(IREE_STATUS_CANCELLED);
          },
      .user_data = &cancelled,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_replay_execute_file(Recording(), replay_group, &execute_options,
                                   iree_allocator_system()));
  EXPECT_TRUE(cancelled);
  IREE_EXPECT_OK(iree_hal_replay_execute_file(
      Recording(), replay_group, nullptr, iree_allocator_system()));
  check_output();
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "File output replay requires POSIX file IO.";
#endif
}

TEST_F(ReplayRecorderVirtualBufferTest,
       RejectsAnUnrecordedHostSignalBeforeExecution) {
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, iree_hal_make_queue_family_affinity(0),
                                0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
  uint64_t gate_value = 1;
  uint64_t completed = 1;
  const iree_hal_semaphore_list_t gate_list = {1, &gate, &gate_value};
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, gate_list, Point(&completed), 0,
                                    nullptr, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
  Wait(completed);
  iree_hal_semaphore_release(gate);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
  // Host semaphore calls are outside the recorded HAL device surface. The
  // planner must reject this wait instead of submitting unreplayable work.
  iree_hal_replay_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_plan_allocate(
                            Recording(), iree_allocator_system(), &plan));
  EXPECT_EQ(nullptr, plan);
}

TEST_F(ReplayRecorderVirtualBufferTest,
       ForeignSemaphoreRemainsLiveAndIsUnsupported) {
  iree_hal_semaphore_t* external = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      iree_hal_device_group_device_at(source_group_, 0),
      iree_hal_make_queue_family_affinity(0), 1,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &external));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t external_list = {1, &external, &value};
  IREE_ASSERT_OK(iree_hal_queue_map(queue_, external_list, Point(&value), 0,
                                    nullptr, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Wait(value);
  iree_hal_semaphore_release(external);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
  auto* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_replay_execute_file(Recording(), replay_group, nullptr,
                                   iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
}

TEST_F(ReplayRecorderVirtualBufferTest, FailedRecordedDetachPreservesBuffer) {
  iree_hal_pool_family_access_t family;
  auto scope = Scope(&family);
  family.family = iree_hal_device_queue_family(
      iree_hal_device_group_device_at(source_group_, 0), 0);
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      source_group_, scope, iree_allocator_system(), &capabilities));
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = capabilities.reservation_granularity,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      source_group_, scope, &options, iree_allocator_system(), &buffer));

  std::array<iree_hal_replay_object_entry_t, 2> objects = {};
  objects[1].type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  objects[1].value.buffer = buffer;
  iree_hal_replay_executor_t executor = {
      .device_group = source_group_,
      .host_allocator = iree_allocator_system(),
      .objects = objects.data(),
      .object_capacity = objects.size(),
  };
  const iree_hal_replay_virtual_buffer_detach_payload_t payload = {};
  iree_hal_replay_file_record_t record = {};
  record.header.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  record.header.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  record.header.object_id = 1;
  record.header.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_DETACH;
  record.header.status_code = IREE_STATUS_UNAVAILABLE;
  record.header.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_DETACH;
  record.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  IREE_EXPECT_OK(iree_hal_replay_executor_replay_operation(&executor, &record));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, objects[1].type);
  EXPECT_EQ(buffer, objects[1].value.buffer);
  iree_hal_virtual_buffer_info_t info;
  IREE_EXPECT_OK(iree_hal_virtual_buffer_query_info(buffer, &info));
  iree_hal_buffer_release(buffer);
}

TEST_F(ReplayRecorderVirtualBufferTest, BufferReleasesAfterRecorderCloses) {
  iree_hal_pool_family_access_t family;
  auto scope = Scope(&family);
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, iree_allocator_system(), &capabilities));
  ASSERT_GT(capabilities.reservation_granularity, 0u);
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = capabilities.reservation_granularity,
  };
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, &options, iree_allocator_system(), &reservation));
  auto* buffer = (reservation);
  iree_hal_buffer_retain(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
  iree_hal_buffer_release(reservation);
  iree_hal_buffer_release(buffer);
  // The capture ends with a live buffer. Executor teardown releases it.
  auto* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(
      Recording(), replay_group, nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
}

}  // namespace
