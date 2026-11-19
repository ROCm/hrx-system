// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/driver/task/registration/driver_module.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

enum class Descriptions { kHost, kIndirect };
enum class Geometry { kRows, kStrided };

class ReplayFileTransferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    iree_async_proactor_pool_t* proactors = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors));
    iree_hal_driver_registry_t* registry = nullptr;
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
    IREE_ASSERT_OK(iree_hal_task_driver_module_register(registry));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors;
    iree_hal_device_t* device = nullptr;
    IREE_ASSERT_OK(iree_hal_create_device(registry, IREE_SV("task"), &params,
                                          iree_allocator_system(), &device));
    iree_hal_driver_registry_free(registry);
    iree_async_proactor_pool_release(proactors);
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device, tracker, iree_allocator_system(), &group_));
    iree_async_frontier_tracker_release(tracker);
    iree_hal_device_release(device);
  }
  static void TearDownTestSuite() { iree_hal_device_group_release(group_); }

  void Run(Descriptions descriptions, Geometry geometry, size_t count,
           iree_hal_replay_recorder_external_file_policy_t file_policy =
               IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_REFERENCE) {
    const bool indirect = descriptions == Descriptions::kIndirect;
    const bool strided = geometry == Geometry::kStrided;
    const bool capture_ranges =
        file_policy ==
        IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_RANGES;
    SCOPED_TRACE(indirect);
    SCOPED_TRACE(strided);
    SCOPED_TRACE(count);
    constexpr size_t kRows = 65, kCapacity = 68, kWidth = 90;
    constexpr size_t kFileStride = 192, kBufferStride = 128, kOrigin = 13;
    std::vector<uint8_t> source(kOrigin + kRows * kFileStride + 7);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = (i * 37 + 19) % 251;
    }
    std::vector<uint8_t> expected(source.size(), 0xCD);
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(expected.data() + kOrigin + i * kFileStride,
                  source.data() + kOrigin + i * kFileStride, kWidth);
    }
    iree::testing::TempFilePath input("replay_finite_input"),
        output("replay_finite_output");
    std::vector<uint8_t> initial(source.size(), 0xCD);
    IREE_ASSERT_OK(iree_io_file_contents_write(
        input.path_view(),
        iree_make_const_byte_span(source.data(), source.size()),
        iree_allocator_system()));
    IREE_ASSERT_OK(iree_io_file_contents_write(
        output.path_view(),
        iree_make_const_byte_span(initial.data(), initial.size()),
        iree_allocator_system()));

    std::vector<uint8_t> recording(256 * 1024);
    iree_io_file_handle_t* recording_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(recording.data(), recording.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &recording_handle));
    auto recorder_options = iree_hal_replay_recorder_options_default();
    recorder_options.external_file_policy = file_policy;
    recorder_options.external_file_validation =
        IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
    iree_hal_replay_recorder_t* recorder = nullptr;
    IREE_ASSERT_OK(
        iree_hal_replay_recorder_create(recording_handle, &recorder_options,
                                        iree_allocator_system(), &recorder));
    iree_io_file_handle_release(recording_handle);
    iree_hal_device_group_t* wrapped = nullptr;
    IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
        recorder, group_, iree_allocator_system(), &wrapped));
    iree_hal_device_t* device = iree_hal_device_group_device_at(wrapped, 0);
    iree_hal_queue_t* queue = iree_hal_device_queue(device, 0, 0);
    iree_hal_file_t* files[2] = {};
    for (size_t i = 0; i < (capture_ranges ? 1u : 2u); ++i) {
      iree_io_file_handle_t* handle = nullptr;
      IREE_ASSERT_OK(iree_io_file_handle_open(
          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
              IREE_IO_FILE_MODE_RANDOM_ACCESS,
          i ? output.path_view() : input.path_view(), iree_allocator_system(),
          &handle));
      IREE_ASSERT_OK(iree_hal_file_import(
          device, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          IREE_HAL_MEMORY_ACCESS_ALL, handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
          &files[i]));
      iree_io_file_handle_release(handle);
    }
    iree_hal_pool_family_access_t family = {iree_hal_queue_family(queue),
                                            IREE_HAL_BUFFER_USAGE_TRANSFER};
    iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(wrapped, scope, &options,
                                             iree_allocator_system(), &pool));
    iree_hal_buffer_t *root = nullptr, *buffer = nullptr, *metadata = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool,
        iree_hal_make_pool_reservation_request(7 + kRows * kBufferStride, 64),
        iree_infinite_timeout(), &root));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, 7, kRows * kBufferStride,
                                           iree_allocator_system(), &buffer));
    iree_hal_buffer_release(root);
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request((1 + kCapacity) * 64, 64),
        iree_infinite_timeout(), &metadata));
    iree_hal_semaphore_t* timeline = nullptr;
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline));
    std::array<uint64_t, 3> values = {1, 2, 3};
    auto point = [&](size_t i) {
      return iree_hal_semaphore_list_t{1, &timeline, &values[i]};
    };
    std::array<iree_hal_file_transfer_t, kRows> rows = {};
    std::array<uint64_t, (1 + kCapacity) * 8> words;
    words.fill(UINT64_MAX);
    const size_t description_count = strided && count ? 1 : count;
    words[0] = description_count;
    for (size_t i = 0; i < description_count; ++i) {
      rows[i] = {
          .file_binding = static_cast<uint32_t>(i % 2),
          .buffer_binding = static_cast<uint32_t>(i % 2),
          .file_offset = i * kFileStride,
          .buffer_offset = i * kBufferStride,
          .row_length = kWidth,
          .row_count = strided ? count / 5 : 1,
          .slice_count = strided ? 5u : 1u,
          .file_strides = {kFileStride, (count / 5) * kFileStride},
          .buffer_strides = {kBufferStride, (count / 5) * kBufferStride},
      };
      uint64_t* record = words.data() + 8 + i * 8;
      record[0] = rows[i].file_offset | (uint64_t(rows[i].file_binding) << 48);
      record[1] =
          rows[i].buffer_offset | (uint64_t(rows[i].buffer_binding) << 48);
      record[2] = kWidth | (uint64_t(strided) << 48);
      if (strided) {
        record[3] = kFileStride | (rows[i].row_count << 48);
        record[4] = kBufferStride;
        record[5] =
            rows[i].file_strides.slice_stride | (rows[i].slice_count << 48);
        record[6] = rows[i].buffer_strides.slice_stride;
      }
    }
    if (!capture_ranges) {
      IREE_ASSERT_OK(iree_hal_queue_update_linear(
          queue, {}, point(0), words.data(), 0, metadata, 0, sizeof(words),
          nullptr, IREE_HAL_UPDATE_FLAG_NONE));
    }
    const iree_hal_file_binding_t sources[] = {
        {files[0], kOrigin, kRows * kFileStride},
        {files[0], kOrigin, kRows * kFileStride}};
    const iree_hal_file_binding_t targets[] = {
        {files[1], kOrigin, kRows * kFileStride},
        {files[1], kOrigin, kRows * kFileStride}};
    const iree_hal_buffer_binding_t buffers[] = {
        {buffer, 0, IREE_HAL_WHOLE_BUFFER}, {buffer, 0, IREE_HAL_WHOLE_BUFFER}};
    iree_hal_file_transfer_list_t list = {};
    if (indirect) {
      list.type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT;
      list.indirect = {{metadata, 64, kCapacity * 64}, metadata, 0};
    } else {
      list.host = {description_count, rows.data()};
    }
    IREE_ASSERT_OK(iree_hal_queue_read(
        queue, capture_ranges ? iree_hal_semaphore_list_t{} : point(0),
        point(1), {2, sources}, {2, buffers}, &list, nullptr,
        IREE_HAL_READ_FLAG_NONE));
    std::vector<uint8_t> downloaded(kRows * kBufferStride);
    if (capture_ranges) {
      IREE_ASSERT_OK(iree_hal_queue_download_linear(
          queue, point(1), point(2), buffer, 0, downloaded.data(),
          downloaded.size(), nullptr));
    } else {
      IREE_ASSERT_OK(iree_hal_queue_write(queue, point(1), point(2),
                                          {2, buffers}, {2, targets}, &list,
                                          nullptr, IREE_HAL_WRITE_FLAG_NONE));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        point(2), iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    if (capture_ranges) {
      for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(
            std::memcmp(downloaded.data() + i * kBufferStride,
                        source.data() + kOrigin + i * kFileStride, kWidth),
            0);
      }
    }
    iree_hal_buffer_release(buffer);
    iree_hal_buffer_release(metadata);
    iree_hal_semaphore_release(timeline);
    iree_hal_file_release(files[0]);
    iree_hal_file_release(files[1]);
    iree_hal_pool_release(pool);
    IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
    iree_hal_device_group_release(wrapped);
    iree_hal_replay_recorder_release(recorder);

    auto verify = [&]() {
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          output.path_view(), iree_allocator_system(), &contents));
      ASSERT_EQ(contents->const_buffer.data_length, expected.size());
      EXPECT_EQ(std::memcmp(contents->const_buffer.data, expected.data(),
                            expected.size()),
                0);
      iree_io_file_contents_free(contents);
    };
    if (!capture_ranges) {
      ASSERT_NO_FATAL_FAILURE(verify());
    }
    IREE_ASSERT_OK(iree_io_file_contents_write(
        output.path_view(),
        iree_make_const_byte_span(initial.data(), initial.size()),
        iree_allocator_system()));
    iree_hal_replay_file_header_t header;
    iree_host_size_t offset = 0;
    IREE_ASSERT_OK(iree_hal_replay_file_parse_header(
        iree_make_const_byte_span(recording.data(), recording.size()), &header,
        &offset));
    if (capture_ranges) {
      const auto contents =
          iree_make_const_byte_span(recording.data(), header.file_length);
      size_t read_count = 0;
      while (offset < contents.data_length) {
        iree_hal_replay_file_record_t record;
        IREE_ASSERT_OK(iree_hal_replay_file_parse_record(contents, offset,
                                                         &record, &offset));
        if (record.header.payload_type !=
            IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ) {
          continue;
        }
        ++read_count;
        iree_hal_replay_queue_file_payload_t payload;
        std::memcpy(&payload, record.payload.data, sizeof(payload));
        EXPECT_EQ(payload.captured_data_length, count * kWidth);
        iree_hal_replay_queue_file_layout_t layout;
        IREE_ASSERT_OK(iree_hal_replay_queue_file_layout(&payload, &layout));
        const uint8_t* data =
            record.payload.data + sizeof(payload) +
            (payload.wait_semaphore_count + payload.signal_semaphore_count) *
                sizeof(iree_hal_replay_semaphore_timepoint_payload_t) +
            layout.data_offset;
        for (size_t i = 0; i < count; ++i) {
          EXPECT_EQ(
              std::memcmp(data + i * kWidth,
                          source.data() + kOrigin + i * kFileStride, kWidth),
              0);
        }
      }
      EXPECT_EQ(read_count, 1u);
      ASSERT_TRUE(input.Remove());
    }
    IREE_ASSERT_OK(iree_hal_replay_execute_file(
        iree_make_const_byte_span(recording.data(), header.file_length), group_,
        nullptr, iree_allocator_system()));
    if (!capture_ranges) {
      ASSERT_NO_FATAL_FAILURE(verify());
    }
  }

  // One native Task group is shared across capture and replay cases.
  static iree_hal_device_group_t* group_;
};
iree_hal_device_group_t* ReplayFileTransferTest::group_ = nullptr;

TEST_F(ReplayFileTransferTest, HostAndIndirectGroupsPreserveSparseBytes) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  for (auto descriptions : {Descriptions::kHost, Descriptions::kIndirect}) {
    for (size_t count : {0u, 1u, 65u}) {
      ASSERT_NO_FATAL_FAILURE(Run(descriptions, Geometry::kRows, count));
    }
    ASSERT_NO_FATAL_FAILURE(Run(descriptions, Geometry::kStrided, 65));
  }
  for (size_t count : {0u, 65u}) {
    ASSERT_NO_FATAL_FAILURE(
        Run(Descriptions::kHost, Geometry::kStrided, count,
            IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_RANGES));
  }
#else
  GTEST_SKIP() << "External-file replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID || IREE_PLATFORM_LINUX)
}

}  // namespace
