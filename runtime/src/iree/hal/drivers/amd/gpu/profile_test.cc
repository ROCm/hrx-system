// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProfileSink {
  // Stack-owned sink resource, retained by each active session.
  iree_hal_resource_t resource;
  // Successful queue-event writes, copied before the callback returns.
  std::vector<iree_hal_profile_queue_event_t> events;
  // Successful device metadata records.
  std::vector<iree_hal_profile_device_record_t> devices;
  // Successful queue metadata records.
  std::vector<iree_hal_profile_queue_record_t> queues;
  // Completed device samples copied before the producer reuses its storage.
  std::vector<iree_hal_profile_dispatch_event_t> dispatches;
  // Host-only identities emitted before referring dispatch records.
  std::vector<uint64_t> executables;
  // Reflected function names, including names after the source is destroyed.
  std::vector<std::string> functions;
  // Host recording identities retained independently of native IBs.
  std::vector<uint64_t> commands;
  // Original code-object records and their payloads, copied at the sink.
  std::vector<iree_hal_profile_executable_code_object_record_t> code_objects;
  // Image bytes corresponding to code_objects in emission order.
  std::vector<std::vector<uint8_t>> images;
  // Every native placement sharing an emitted image.
  std::vector<iree_hal_profile_executable_code_object_load_record_t> loads;
  // Metadata loss with an unknown number of missing unique executables.
  size_t incomplete_metadata = 0;
  // Optional producer work during a sink write, exercising lock separation.
  std::function<void()> during_write;
  // Successfully reported drops across all queue streams.
  uint64_t dropped = 0;
  // Number of session starts.
  int begins = 0;
  // Number of terminal callbacks.
  int ends = 0;
  // Producer status supplied to the most recent terminal callback.
  iree_status_code_t end_status = IREE_STATUS_OK;
  // Dependency failure injected at the sink boundary.
  bool fail_write = false;
  // Optional content type at which a write fails after earlier chunks succeed.
  iree_string_view_t fail_content_type = {};

  iree_hal_profile_sink_t* base() {
    return reinterpret_cast<iree_hal_profile_sink_t*>(this);
  }

  static void Destroy(iree_hal_profile_sink_t*) {}
  static iree_status_t Begin(iree_hal_profile_sink_t* base,
                             const iree_hal_profile_chunk_metadata_t*) {
    ++reinterpret_cast<ProfileSink*>(base)->begins;
    return iree_ok_status();
  }
  static iree_status_t End(iree_hal_profile_sink_t* base,
                           const iree_hal_profile_chunk_metadata_t*,
                           iree_status_code_t code) {
    auto& sink = *reinterpret_cast<ProfileSink*>(base);
    ++sink.ends;
    sink.end_status = code;
    return iree_ok_status();
  }
  static iree_status_t Write(iree_hal_profile_sink_t* base,
                             const iree_hal_profile_chunk_metadata_t* metadata,
                             iree_host_size_t count,
                             const iree_const_byte_span_t* spans) {
    auto& sink = *reinterpret_cast<ProfileSink*>(base);
    if (sink.fail_write || iree_string_view_equal(metadata->content_type,
                                                  sink.fail_content_type)) {
      return iree_status_from_code(IREE_STATUS_DATA_LOSS);
    }
    if (sink.during_write) {
      auto callback = std::move(sink.during_write);
      callback();
    }
    EXPECT_EQ(sink.begins, sink.ends + 1);
    std::vector<uint8_t> contents;
    iree_const_byte_span_t joined;
    if (count > 1) {
      for (iree_host_size_t i = 0; i < count; ++i) {
        contents.insert(contents.end(), spans[i].data,
                        spans[i].data + spans[i].data_length);
      }
      joined = iree_make_const_byte_span(contents.data(), contents.size());
      spans = &joined;
      count = 1;
    }
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (iree_string_view_equal(metadata->content_type,
                                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
        auto* records = reinterpret_cast<const iree_hal_profile_queue_event_t*>(
            spans[i].data);
        for (size_t j = 0; j < spans[i].data_length / sizeof(*records); ++j) {
          EXPECT_EQ(records[j].queue_ordinal, metadata->queue_ordinal);
          EXPECT_EQ(records[j].stream_id, metadata->stream_id);
          sink.events.push_back(records[j]);
        }
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
        auto* record =
            reinterpret_cast<const iree_hal_profile_device_record_t*>(
                spans[i].data);
        sink.devices.push_back(*record);
      } else if (iree_string_view_equal(metadata->content_type,
                                        IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
        auto* record = reinterpret_cast<const iree_hal_profile_queue_record_t*>(
            spans[i].data);
        sink.queues.push_back(*record);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
        auto* record =
            reinterpret_cast<const iree_hal_profile_executable_record_t*>(
                spans[i].data);
        sink.executables.push_back(record->executable_id);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS)) {
        size_t offset = 0;
        while (offset < spans[i].data_length) {
          iree_hal_profile_executable_function_record_t record;
          std::memcpy(&record, spans[i].data + offset, sizeof(record));
          sink.functions.emplace_back(
              reinterpret_cast<const char*>(spans[i].data + offset +
                                            sizeof(record)),
              record.name_length);
          offset += record.record_length;
        }
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS)) {
        iree_hal_profile_executable_code_object_record_t record;
        std::memcpy(&record, spans[i].data, sizeof(record));
        EXPECT_EQ(record.record_length, spans[i].data_length);
        EXPECT_EQ(record.data_length, spans[i].data_length - sizeof(record));
        sink.code_objects.push_back(record);
        sink.images.emplace_back(spans[i].data + sizeof(record),
                                 spans[i].data + spans[i].data_length);
      } else if (
          iree_string_view_equal(
              metadata->content_type,
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS)) {
        for (size_t offset = 0; offset < spans[i].data_length;) {
          iree_hal_profile_executable_code_object_load_record_t record;
          std::memcpy(&record, spans[i].data + offset, sizeof(record));
          sink.loads.push_back(record);
          offset += record.record_length;
        }
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
        auto* record =
            reinterpret_cast<const iree_hal_profile_command_buffer_record_t*>(
                spans[i].data);
        sink.commands.push_back(record->command_buffer_id);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
        auto* records =
            reinterpret_cast<const iree_hal_profile_dispatch_event_t*>(
                spans[i].data);
        for (size_t j = 0; j < spans[i].data_length / sizeof(*records); ++j) {
          EXPECT_NE(std::find(sink.executables.begin(), sink.executables.end(),
                              records[j].executable_id),
                    sink.executables.end());
          sink.dispatches.push_back(records[j]);
        }
      }
    }
    sink.dropped += metadata->dropped_record_count;
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES) &&
        (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
      ++sink.incomplete_metadata;
      EXPECT_EQ(count, 0u);
      EXPECT_EQ(metadata->dropped_record_count, 0u);
    } else {
      EXPECT_EQ(metadata->dropped_record_count != 0,
                (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED) != 0);
    }
    return iree_ok_status();
  }
  ProfileSink() {
    static const iree_hal_profile_sink_vtable_t vtable = {Destroy, Begin, Write,
                                                          End};
    iree_hal_resource_initialize(&vtable, &resource);
  }
};

// Opaque artifact dependency with a separately observable host lifetime.
struct ProfileImage {
  // Refcounted sequence header; metadata retains this instead of the
  // executable.
  iree_byte_sequence_t base;
  // Immutable source bytes exposed by the contiguous native-loader contract.
  std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  // Shared observation survives a test's early return while capture owns bytes.
  std::shared_ptr<bool> destroyed;

  static iree_byte_sequence_t* Create(std::shared_ptr<bool> destroyed) {
    static const iree_byte_sequence_vtable_t vtable = {
        .destroy =
            [](iree_byte_sequence_t* base) {
              auto* self = reinterpret_cast<ProfileImage*>(base);
              *self->destroyed = true;
              delete self;
            },
        .enumerate =
            [](const iree_byte_sequence_t* base,
               iree_byte_sequence_segment_callback_t callback) {
              const auto* self = reinterpret_cast<const ProfileImage*>(base);
              return callback.fn(callback.user_data,
                                 iree_make_const_byte_span(self->bytes.data(),
                                                           self->bytes.size()));
            },
        .try_get_contiguous_span =
            [](const iree_byte_sequence_t* base,
               iree_const_byte_span_t* out_span) {
              const auto* self = reinterpret_cast<const ProfileImage*>(base);
              *out_span = iree_make_const_byte_span(self->bytes.data(),
                                                    self->bytes.size());
              return true;
            },
    };
    auto* self = new ProfileImage;
    self->destroyed = std::move(destroyed);
    iree_byte_sequence_initialize(&vtable, self->bytes.size(), &self->base);
    return &self->base;
  }
};

// Reflection dependency with an observable code-owner destruction boundary.
// The profile implementation must copy names and retain no executable owner.
struct ReflectionExecutable {
  // Actual HAL resource passed through the public reflection API.
  iree_hal_executable_t base;
  // Source names become unavailable as soon as the executable is released.
  std::array<std::string, 2> names = {"first", "second"};
  // Reflection-only family identity; this dependency never executes work.
  iree_hal_queue_family_t family = {};
  // Confirms metadata owns no code or device resource reference.
  bool destroyed = false;
  // Observes final host-source release independently of executable destruction.
  std::shared_ptr<bool> image_destroyed = std::make_shared<bool>(false);
  // Admitted image facts consumed by profiling, with a nonzero original base.
  iree_hal_amd_gpu_image_t image = {
      .source = ProfileImage::Create(image_destroyed),
      .virtual_base = 0x400000,
      .byte_length = 0x1000,
  };
  // Distinct native placements of the same image, as for per-queue TSAN code.
  std::array<uint64_t, 2> load_addresses = {0x100000000ull, 0x200000000ull};

  ReflectionExecutable() {
    static const iree_hal_executable_vtable_t vtable = {
        .destroy =
            [](iree_hal_executable_t* base) {
              auto* self = reinterpret_cast<ReflectionExecutable*>(base);
              self->destroyed = true;
              self->names = {};
              iree_byte_sequence_release(self->image.source);
            },
        .function_count = [](iree_hal_executable_t*) -> iree_host_size_t {
          return 2;
        },
        .function_info =
            [](iree_hal_executable_t* base,
               iree_hal_executable_function_t token,
               iree_hal_executable_function_info_t* out_info) {
              auto* self = reinterpret_cast<ReflectionExecutable*>(base);
              *out_info = {};
              const auto& name = self->names[token.value];
              out_info->name = name.empty() ? iree_string_view_empty()
                                            : iree_make_string_view(
                                                  name.data(), name.size());
              out_info->workgroup_size[0] = 32;
              out_info->workgroup_size[1] = out_info->workgroup_size[2] = 1;
              return iree_ok_status();
            },
    };
    iree_hal_executable_initialize(&family, &vtable, &base);
  }

  iree_status_t Capture(iree_hal_amd_gpu_profile_executable_t** out_metadata) {
    return iree_hal_amd_gpu_profile_executable_create(
        &base, &image, load_addresses.size(), load_addresses.data(),
        iree_allocator_system(), out_metadata);
  }
};

class GpuProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
    options_.sink = sink_.base();
  }
  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(profile_));
  }
  void Begin(uint64_t frequency = 0) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_begin(
        &options_, 37, 2, frequency, iree_allocator_system(), &profile_));
  }
  void Append(uint32_t queue = 0) {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, queue, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
        &operation);
    iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 0);
  }
  // Sink outlives the retained session, including failure cleanup.
  ProfileSink sink_;
  // Validated caller request for the internal session owner.
  iree_hal_device_profiling_options_t options_ = {};
  // Unique session consumed by teardown unless a test explicitly ends it.
  iree_hal_amd_gpu_profile_t* profile_ = nullptr;
};

TEST_F(GpuProfileTest, IdentitiesAndObservedAdmission) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  ASSERT_EQ(sink_.devices.size(), 1u);
  EXPECT_EQ(sink_.devices[0].queue_count, 2u);
  EXPECT_EQ(sink_.devices[0].timestamp_frequency_hz, 0u);
  ASSERT_EQ(sink_.queues.size(), 2u);
  EXPECT_NE(sink_.queues[0].stream_id, sink_.queues[1].stream_id);
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 1, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 2, 3, 1000,
      &operation);
  iree_hal_amd_gpu_profile_operation_defer(&operation);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 2);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 1u);
  const auto& event = sink_.events[0];
  EXPECT_EQ(event.record_length, sizeof(event));
  EXPECT_EQ(event.queue_ordinal, 1u);
  EXPECT_EQ(event.stream_id, sink_.queues[1].stream_id);
  EXPECT_EQ(event.wait_count, 2u);
  EXPECT_EQ(event.signal_count, 3u);
  EXPECT_EQ(event.barrier_count, 2u);
  EXPECT_EQ(event.operation_count, 1000u);
  EXPECT_EQ(event.dependency_strategy,
            IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER);
  EXPECT_GE(event.ready_host_time_ns, event.host_time_ns);

  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, 1, 1, 0,
      &operation);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, false, 0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.events.back().ready_host_time_ns, 0);
}

TEST_F(GpuProfileTest, FailedSnapshotRetriesWithoutBlockingProducer) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append();
  sink_.fail_write = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  sink_.fail_write = false;
  sink_.during_write = [&] { Append(); };
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 1u);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 2u);
  EXPECT_EQ(sink_.events[0].event_id + 1, sink_.events[1].event_id);
  EXPECT_EQ(sink_.events[0].submission_id + 1, sink_.events[1].submission_id);
}

TEST_F(GpuProfileTest, AddedCoordinatePreservesPriorRecordsAndIdentity) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append(0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, 255));
  Append(255);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, 255));
  Append(255);
  Append(0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.queues.size(), 3u);
  EXPECT_EQ(sink_.queues.back().queue_ordinal, 255u);
  EXPECT_EQ(sink_.devices.back().queue_count, 3u);
  ASSERT_EQ(sink_.events.size(), 4u);
  EXPECT_EQ(sink_.events[0].queue_ordinal, 0u);
  EXPECT_EQ(sink_.events[1].submission_id, 2u);
  EXPECT_EQ(sink_.events[2].queue_ordinal, 255u);
  EXPECT_EQ(sink_.events[2].submission_id, 1u);
  EXPECT_EQ(sink_.events[3].submission_id, 2u);
}

TEST_F(GpuProfileTest, FailedRegistrationLeavesPriorStreamsUsable) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  sink_.fail_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_register_queue(profile_, 2));
  Append(0);
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, 2));
  Append(2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 2u);
  EXPECT_EQ(sink_.events[0].queue_ordinal, 0u);
  EXPECT_EQ(sink_.events[1].queue_ordinal, 2u);
  EXPECT_EQ(sink_.devices.back().queue_count, 3u);
}

TEST_F(GpuProfileTest, RegistrationOverlapsFlushAndExistingProducer) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  std::barrier<> start(3);
  std::thread producer([&] {
    start.arrive_and_wait();
    for (uint32_t i = 0; i < 512; ++i) {
      Append(0);
    }
  });
  std::thread registrar([&] {
    start.arrive_and_wait();
    for (uint32_t i = 2; i < 10; ++i) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, i));
      for (uint32_t j = 0; j < 32; ++j) {
        Append(i);
      }
    }
  });
  start.arrive_and_wait();
  for (uint32_t i = 0; i < 64; ++i) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  }
  producer.join();
  registrar.join();
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.events.size(), 512u + 8u * 32u);
  EXPECT_EQ(sink_.dropped, 0u);
  EXPECT_EQ(sink_.queues.size(), 10u);
}

TEST_F(GpuProfileTest,
       BoundedConcurrentStreamsAccountEveryDropAndReuseStorage) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  constexpr size_t kOperations = 8193;
  std::array<std::thread, 2> threads;
  for (uint32_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (size_t j = 0; j < kOperations; ++j) {
        Append(i);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_GT(sink_.dropped, 0u);
  EXPECT_EQ(sink_.events.size() + sink_.dropped, 2 * kOperations);
  const size_t first_count = sink_.events.size();
  const uint64_t first_dropped = sink_.dropped;
  Append(0);
  Append(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.events.size(), first_count + 2);
  EXPECT_EQ(sink_.dropped, first_dropped);
}

TEST_F(GpuProfileTest, FailedEndStillClosesSessionAndReleasesSink) {
  options_.data_families |= IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append();
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      &operation);
  operation.dispatch = {metadata, iree_hal_profile_dispatch_event_default()};
  operation.dispatch.event.executable_id =
      iree_hal_amd_gpu_profile_executable_id(metadata);
  operation.completion.count = 1;
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  iree_hal_amd_gpu_profile_executable_release(metadata);
  EXPECT_FALSE(*executable.image_destroyed);
  sink_.fail_write = true;
  auto* profile = profile_;
  profile_ = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_end(profile));
  EXPECT_EQ(sink_.ends, 1);
  EXPECT_EQ(sink_.end_status, IREE_STATUS_DATA_LOSS);
  EXPECT_TRUE(*executable.image_destroyed);
  sink_.fail_write = false;
  ASSERT_NO_FATAL_FAILURE(Begin());
}

TEST_F(GpuProfileTest, FailedMetadataClosesBegunSession) {
  sink_.fail_write = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, 0,
                                     iree_allocator_system(), &profile_));
  EXPECT_EQ(profile_, nullptr);
  EXPECT_EQ(sink_.begins, 1);
  EXPECT_EQ(sink_.ends, 1);
}

TEST_F(GpuProfileTest, UnsupportedDataDoesNotStartSinkSession) {
  options_.data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, 0,
                                     iree_allocator_system(), &profile_));
  EXPECT_EQ(profile_, nullptr);
  EXPECT_EQ(sink_.begins, 0);
}

TEST_F(GpuProfileTest, DispatchClockMustBeEstablishedBeforeBeginning) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, 0,
                                     iree_allocator_system(), &profile_));
  EXPECT_EQ(sink_.begins, 0);
  EXPECT_EQ(profile_, nullptr);
}

TEST_F(GpuProfileTest, EveryFilterPredicateCombinesAndOwnsItsPattern) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  constexpr uint32_t kPattern =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  constexpr uint32_t kCommandBuffer =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID;
  constexpr uint32_t kCommand =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX;
  constexpr uint32_t kDevice =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL;
  constexpr uint32_t kQueue =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL;
  for (uint32_t flags = 0; flags < 32; ++flags) {
    SCOPED_TRACE(flags);
    std::string pattern = "sec*";
    options_.capture_filter = {
        .flags = flags,
        .executable_function_pattern =
            iree_make_string_view(pattern.data(), pattern.size()),
        .command_buffer_id = 51,
        .command_index = 7,
        .physical_device_ordinal = 0,
        .queue_ordinal = 1,
    };
    ASSERT_NO_FATAL_FAILURE(Begin(100000000));
    pattern.assign("gone");
    EXPECT_TRUE(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 51, 7,
                                                           IREE_SV("second")));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 51, 7,
                                                         IREE_SV("first")),
              !(flags & kPattern));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 52, 7,
                                                         IREE_SV("second")),
              !(flags & kCommandBuffer));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 51, 8,
                                                         IREE_SV("second")),
              !(flags & kCommand));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 0, 51, 7,
                                                         IREE_SV("second")),
              !(flags & kQueue));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 0, 7,
                                                         IREE_SV("second")),
              !(flags & (kCommandBuffer | kCommand)));
    EXPECT_TRUE(iree_hal_amd_gpu_profile_captures_commands(profile_, 1, 51));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_commands(profile_, 0, 52),
              !(flags & (kQueue | kCommandBuffer)));
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_end(profile_));
    profile_ = nullptr;
    options_.capture_filter.physical_device_ordinal = 1;
    options_.capture_filter.executable_function_pattern = IREE_SV("sec*");
    ASSERT_NO_FATAL_FAILURE(Begin(100000000));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_dispatch(profile_, 1, 51, 7,
                                                         IREE_SV("second")),
              !(flags & kDevice));
    EXPECT_EQ(iree_hal_amd_gpu_profile_captures_commands(profile_, 1, 51),
              !(flags & kDevice));
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_end(profile_));
    profile_ = nullptr;
  }
}

TEST_F(GpuProfileTest, ExcludedCommandsUseNoRingCreditOrExecutableLifetime) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  options_.capture_filter = {
      .flags = IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN,
      .executable_function_pattern = IREE_SV("second"),
  };
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  ASSERT_TRUE(executable.destroyed);
  auto record = iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = 51;
  iree_hal_amd_gpu_profile_commands_t* commands = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_commands_create(
      &record, iree_allocator_system(), &commands));
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_amd_gpu_profile_dispatch_t dispatch = {
        .executable = metadata,
        .event = iree_hal_profile_dispatch_event_default(),
    };
    dispatch.event.executable_id =
        iree_hal_amd_gpu_profile_executable_id(metadata);
    dispatch.event.command_buffer_id = 51;
    dispatch.event.command_index = i + 3;
    dispatch.event.function_ordinal = i;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_profile_commands_append(commands, dispatch));
  }
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_amd_gpu_profile_sample_t samples[] = {{111, 222}, {333, 444}};
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 0, 1, 2,
      &operation);
  operation.commands = commands;
  operation.completion.samples = samples;
  operation.completion.sample_stride = sizeof(samples[0]);
  operation.completion.count = 2;
  for (uint32_t i = 0; i < 4096; ++i) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
  }
  operation.completion.count = 1;
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  iree_hal_amd_gpu_profile_commands_release(commands);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dropped, 0u);
  ASSERT_EQ(sink_.dispatches.size(), 4096u);
  for (const auto& event : sink_.dispatches) {
    EXPECT_EQ(event.function_ordinal, 1u);
    EXPECT_EQ(event.command_index, 4u);
    EXPECT_EQ(event.start_tick, 333u);
    EXPECT_EQ(event.end_tick, 444u);
  }
  EXPECT_TRUE(*executable.image_destroyed);
}

TEST_F(GpuProfileTest, CompletedSamplesAndMetadataOutliveTheirOriginalOwners) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  EXPECT_EQ(sink_.devices[0].timestamp_frequency_hz, 100000000u);
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  ASSERT_TRUE(executable.destroyed);
  EXPECT_FALSE(*executable.image_destroyed);
  auto record = iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = 51;
  iree_hal_amd_gpu_profile_commands_t* commands = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_commands_create(
      &record, iree_allocator_system(), &commands));
  iree_hal_amd_gpu_profile_dispatch_t dispatch = {
      .executable = metadata,
      .event = iree_hal_profile_dispatch_event_default(),
  };
  dispatch.event.executable_id =
      iree_hal_amd_gpu_profile_executable_id(metadata);
  dispatch.event.command_buffer_id = 51;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_commands_append(commands, dispatch));
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_amd_gpu_profile_sample_t sample = {123, 456};
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 0, 1, 1,
      &operation);
  operation.commands = commands;
  operation.completion.samples = &sample;
  operation.completion.count = 1;
  operation.completion.sample_stride = sizeof(sample);
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  sample = {};
  iree_hal_amd_gpu_profile_commands_release(commands);
  sink_.fail_content_type =
      IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_FALSE(*executable.image_destroyed);
  EXPECT_TRUE(sink_.dispatches.empty());
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.dispatches.size(), 1u);
  EXPECT_EQ(sink_.dispatches[0].start_tick, 123u);
  EXPECT_EQ(sink_.dispatches[0].end_tick, 456u);
  EXPECT_EQ(sink_.commands, std::vector<uint64_t>({51}));
  EXPECT_EQ(sink_.functions,
            std::vector<std::string>({"first", "second", "first", "second"}));
  EXPECT_TRUE(*executable.image_destroyed);
  ASSERT_EQ(sink_.code_objects.size(), 1u);
  const auto& code_object = sink_.code_objects[0];
  EXPECT_EQ(code_object.executable_id, sink_.dispatches[0].executable_id);
  EXPECT_EQ(sink_.images[0], std::vector<uint8_t>({0x7f, 'E', 'L', 'F'}));
  ASSERT_EQ(sink_.loads.size(), 2u);
  for (size_t i = 0; i < sink_.loads.size(); ++i) {
    EXPECT_EQ(sink_.loads[i].executable_id, code_object.executable_id);
    EXPECT_EQ(sink_.loads[i].code_object_id, code_object.code_object_id);
    EXPECT_EQ(sink_.loads[i].physical_device_ordinal, 0u);
    EXPECT_EQ(sink_.loads[i].load_delta,
              executable.load_addresses[i] - executable.image.virtual_base);
    EXPECT_EQ(sink_.loads[i].load_size, executable.image.byte_length);
  }
  EXPECT_TRUE(sink_.events.empty());
}

TEST_F(GpuProfileTest, PlacementHistoryGrowsDuringFlushWithoutRepeatingImages) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  ASSERT_NO_FATAL_FAILURE(Begin());
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      &operation);
  operation.dispatch = {metadata, iree_hal_profile_dispatch_event_default()};
  operation.completion.count = 1;
  std::barrier<> start(2);
  std::thread placements([&] {
    start.arrive_and_wait();
    for (uint64_t i = 0; i < 64; ++i) {
      IREE_EXPECT_OK(iree_hal_amd_gpu_profile_executable_add_load(
          metadata, executable.image.virtual_base, executable.image.byte_length,
          0x300000000ull + i * 0x1000));
    }
  });
  start.arrive_and_wait();
  for (uint32_t i = 0; i < 64; ++i) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  }
  placements.join();
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  sink_.fail_content_type =
      IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_executable_add_load(
      metadata, executable.image.virtual_base, executable.image.byte_length,
      0x400000000ull));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.code_objects.size(), 1u);
  ASSERT_EQ(sink_.loads.size(), 67u);
  for (uint64_t i = 0; i < 64; ++i) {
    EXPECT_EQ(sink_.loads[i + 2].load_delta,
              0x300000000ull + i * 0x1000 - executable.image.virtual_base);
  }
  EXPECT_EQ(sink_.loads.back().load_delta,
            0x400000000ull - executable.image.virtual_base);
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_executable_release(&executable.base);
}

TEST_F(GpuProfileTest, DispatchDropsAndConcurrentFlushKeepOwnershipBounded) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_profile_sample_t sample = {7, 9};
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      &operation);
  operation.dispatch = {metadata, iree_hal_profile_dispatch_event_default()};
  operation.dispatch.event.executable_id =
      iree_hal_amd_gpu_profile_executable_id(metadata);
  operation.completion.samples = &sample;
  operation.completion.count = 1;
  operation.completion.sample_stride = sizeof(sample);
  for (size_t i = 0; i < 8193; ++i) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
  }
  sink_.during_write = [&] {
    // This append sees the full snapshot still owned by the consumer. It is a
    // counted drop, not an overwrite or a mutex held across the sink callback.
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dispatches.size() + sink_.dropped, 8194u);
  for (size_t i = 0; i < 19; ++i) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dispatches.size() + sink_.dropped, 8213u);
  EXPECT_EQ(sink_.executables.size(), 1u);
  iree_hal_amd_gpu_profile_executable_release(metadata);
}

TEST_F(GpuProfileTest,
       ChunkRetirementCopiesOnlyItsRangeBeforeSamplesAreReused) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  auto record = iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = 72;
  iree_hal_amd_gpu_profile_commands_t* commands = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_commands_create(
      &record, iree_allocator_system(), &commands));
  for (uint32_t i = 0; i < 5; ++i) {
    iree_hal_amd_gpu_profile_dispatch_t dispatch = {
        .executable = metadata,
        .event = iree_hal_profile_dispatch_event_default(),
    };
    dispatch.event.executable_id =
        iree_hal_amd_gpu_profile_executable_id(metadata);
    dispatch.event.command_buffer_id = record.command_buffer_id;
    dispatch.event.command_index = i * 2;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_profile_commands_append(commands, dispatch));
  }
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 0, 1, 5,
      &operation);
  operation.commands = commands;
  operation.completion.count = 5;
  // Native signal timestamp pairs may be interleaved with other device fields.
  // The producer reuses these bytes before the caller exports either chunk.
  struct NativeSample {
    // Native bookkeeping preceding the timestamp fields.
    uint64_t prefix[4];
    // Firmware-owned pair read only after native completion.
    iree_hal_amd_gpu_profile_sample_t sample;
    // Native bookkeeping following the timestamp fields.
    uint64_t suffix[2];
  } samples[3] = {};
  for (uint32_t first : {0u, 2u}) {
    const uint32_t count = first ? 3 : 2;
    for (uint32_t i = 0; i < count; ++i) {
      samples[i].sample = {(first + i) * 10 + 1, (first + i) * 10 + 7};
    }
    const iree_hal_amd_gpu_profile_completion_t completion = {
        .samples = &samples[0].sample,
        .sample_stride = sizeof(samples[0]),
        .first = first,
        .count = count,
    };
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &completion);
    std::memset(samples, 0, sizeof(samples));
  }
  iree_hal_amd_gpu_profile_commands_release(commands);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.dispatches.size(), 5u);
  EXPECT_EQ(sink_.commands, std::vector<uint64_t>({72}));
  for (uint32_t i = 0; i < 5; ++i) {
    const auto& event = sink_.dispatches[i];
    EXPECT_EQ(event.submission_id, operation.event.submission_id);
    EXPECT_EQ(event.command_index, i * 2);
    EXPECT_EQ(event.start_tick, i * 10 + 1);
    EXPECT_EQ(event.end_tick, i * 10 + 7);
  }
}

TEST_F(GpuProfileTest, MetadataOnlyReportsIncompleteIdentityCoverage) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  ASSERT_NO_FATAL_FAILURE(Begin());
  ReflectionExecutable executable;
  executable.names[1].clear();
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      &operation);
  operation.dispatch = {metadata, iree_hal_profile_dispatch_event_default()};
  operation.dispatch.event.executable_id =
      iree_hal_amd_gpu_profile_executable_id(metadata);
  operation.completion.count = 1;
  for (size_t i = 0; i < 4097; ++i) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
  }
  iree_hal_amd_gpu_profile_executable_release(metadata);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.incomplete_metadata, 1u);
  EXPECT_EQ(sink_.executables.size(), 1u);
  EXPECT_EQ(sink_.functions, std::vector<std::string>({"first", ""}));
  EXPECT_TRUE(sink_.dispatches.empty());
}

TEST_F(GpuProfileTest, LiveRetirementAndFlushPreserveSamplesAcrossRingWraps) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(&metadata));
  iree_hal_executable_release(&executable.base);
  const uint64_t executable_id =
      iree_hal_amd_gpu_profile_executable_id(metadata);
  std::atomic<bool> finished = false;
  std::thread retirement([&] {
    for (uint32_t i = 0; i < 10000; ++i) {
      iree_hal_amd_gpu_profile_operation_t operation;
      iree_hal_amd_gpu_profile_operation_initialize(
          profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
          &operation);
      operation.dispatch = {metadata,
                            iree_hal_profile_dispatch_event_default()};
      operation.dispatch.event.executable_id = executable_id;
      const iree_hal_amd_gpu_profile_sample_t sample = {
          operation.event.submission_id * 10,
          operation.event.submission_id * 10 + 5};
      operation.completion.samples = &sample;
      operation.completion.count = 1;
      operation.completion.sample_stride = sizeof(sample);
      iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                  &operation.completion);
    }
    finished.store(true, std::memory_order_release);
  });
  while (!finished.load(std::memory_order_acquire)) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  }
  retirement.join();
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dispatches.size() + sink_.dropped, 10000u);
  for (const auto& event : sink_.dispatches) {
    EXPECT_EQ(event.start_tick, event.submission_id * 10);
    EXPECT_EQ(event.end_tick, event.submission_id * 10 + 5);
  }
  EXPECT_EQ(sink_.executables.size(), 1u);
  iree_hal_amd_gpu_profile_executable_release(metadata);
}

}  // namespace
