// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProfileAllocator {
  // Runs at the real host-allocation boundary, before forwarding or failing.
  std::function<void()> before_allocate;
  // Dependency failure leaves existing allocations and frees operational.
  bool fail = false;
  // Allocation requests, excluding frees, for allocation-free flush checks.
  uint32_t allocation_count = 0;

  iree_allocator_t allocator() {
    return {this, [](void* self, iree_allocator_command_t command,
                     const void* params, void** inout_pointer) {
              auto& allocator = *static_cast<ProfileAllocator*>(self);
              if (command != IREE_ALLOCATOR_COMMAND_FREE) {
                ++allocator.allocation_count;
                if (allocator.before_allocate) {
                  allocator.before_allocate();
                }
                if (allocator.fail) {
                  return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
              }
              const auto system = iree_allocator_system();
              return system.ctl(system.self, command, params, inout_pointer);
            }};
  }
};

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
  // Executable trace records and their raw bytes, in emission order.
  std::vector<iree_hal_profile_executable_trace_record_t> traces;
  std::vector<std::vector<uint8_t>> trace_bytes;
  // Chunk flags of each trace chunk.
  std::vector<uint64_t> trace_flags;
  // How each trace partition was captured, in emission order.
  std::vector<iree_hal_profile_executable_trace_capture_record_t> captures;
  // Counter set records and their names, in emission order.
  std::vector<iree_hal_profile_counter_set_record_t> counter_sets;
  std::vector<std::string> counter_set_names;
  // Counter records with their block name, name and description.
  std::vector<iree_hal_profile_counter_record_t> counters;
  std::vector<std::array<std::string, 3>> counter_texts;
  // Counter sample records and their value vectors.
  std::vector<iree_hal_profile_counter_sample_record_t> counter_samples;
  std::vector<std::vector<uint64_t>> counter_values;
  // Cumulative counter capture records, in emission order.
  std::vector<iree_hal_profile_counter_capture_record_t> counter_captures;
  // Counter samples reported lost.
  uint64_t dropped_counter_samples = 0;
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
      auto callback = std::exchange(sink.during_write, {});
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
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES)) {
        iree_hal_profile_executable_trace_record_t record;
        std::memcpy(&record, spans[i].data, sizeof(record));
        EXPECT_EQ(record.record_length, sizeof(record));
        EXPECT_EQ(record.data_length, spans[i].data_length - sizeof(record));
        EXPECT_EQ(record.queue_ordinal, metadata->queue_ordinal);
        EXPECT_EQ(record.stream_id, metadata->stream_id);
        sink.traces.push_back(record);
        sink.trace_bytes.emplace_back(spans[i].data + sizeof(record),
                                      spans[i].data + spans[i].data_length);
        sink.trace_flags.push_back(metadata->flags);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACE_CAPTURES)) {
        iree_hal_profile_executable_trace_capture_record_t record = {};
        EXPECT_EQ(spans[i].data_length, sizeof(record));
        std::memcpy(&record, spans[i].data,
                    std::min<size_t>(spans[i].data_length, sizeof(record)));
        sink.captures.push_back(record);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
        iree_hal_profile_counter_set_record_t record;
        std::memcpy(&record, spans[i].data, sizeof(record));
        EXPECT_EQ(record.record_length, spans[i].data_length);
        sink.counter_sets.push_back(record);
        sink.counter_set_names.emplace_back(
            reinterpret_cast<const char*>(spans[i].data + sizeof(record)),
            record.name_length);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
        iree_hal_profile_counter_record_t record;
        std::memcpy(&record, spans[i].data, sizeof(record));
        EXPECT_EQ(record.record_length, spans[i].data_length);
        EXPECT_EQ(record.record_length,
                  sizeof(record) + record.block_name_length +
                      record.name_length + record.description_length);
        const char* text =
            reinterpret_cast<const char*>(spans[i].data + sizeof(record));
        sink.counters.push_back(record);
        sink.counter_texts.push_back(
            {std::string(text, record.block_name_length),
             std::string(text + record.block_name_length, record.name_length),
             std::string(text + record.block_name_length + record.name_length,
                         record.description_length)});
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
        for (size_t offset = 0; offset < spans[i].data_length;) {
          iree_hal_profile_counter_sample_record_t record;
          std::memcpy(&record, spans[i].data + offset, sizeof(record));
          EXPECT_EQ(record.record_length,
                    sizeof(record) + 8 * record.sample_value_count);
          EXPECT_EQ(record.queue_ordinal, metadata->queue_ordinal);
          EXPECT_EQ(record.stream_id, metadata->stream_id);
          std::vector<uint64_t> values(record.sample_value_count);
          std::memcpy(values.data(), spans[i].data + offset + sizeof(record),
                      8 * values.size());
          sink.counter_samples.push_back(record);
          sink.counter_values.push_back(std::move(values));
          offset += record.record_length;
        }
        sink.dropped_counter_samples += metadata->dropped_record_count;
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_CAPTURES)) {
        iree_hal_profile_counter_capture_record_t record = {};
        EXPECT_EQ(spans[i].data_length, sizeof(record));
        std::memcpy(&record, spans[i].data,
                    std::min<size_t>(spans[i].data_length, sizeof(record)));
        sink.counter_captures.push_back(record);
      }
    }
    sink.dropped += metadata->dropped_record_count;
    if (iree_string_view_equal(
            metadata->content_type,
            IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES)) {
      // A trace is also truncated by a full buffer or an unfinished stop, where
      // the hardware counts no dropped token.
      if (metadata->dropped_record_count) {
        EXPECT_TRUE(metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
      }
    } else if (iree_string_view_equal(
                   metadata->content_type,
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
    return Capture(iree_allocator_system(), out_metadata);
  }

  iree_status_t Capture(iree_allocator_t host_allocator,
                        iree_hal_amd_gpu_profile_executable_t** out_metadata) {
    return iree_hal_amd_gpu_profile_executable_create(
        &base, &image, load_addresses.size(), load_addresses.data(),
        host_allocator, out_metadata);
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
  // Optional allocation dependency, alive through session failure cleanup.
  ProfileAllocator allocator_;
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

TEST_F(GpuProfileTest, FailedStreamMetadataRetriesBeforeReferringEvents) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, 2));
  Append(2);
  sink_.fail_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.queues.size(), 2u);
  EXPECT_TRUE(sink_.events.empty());
  sink_.fail_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.queues.size(), 3u);
  EXPECT_EQ(sink_.devices.back().queue_count, 2u);
  EXPECT_TRUE(sink_.events.empty());
  Append(0);
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.queues.size(), 3u);
  ASSERT_EQ(sink_.events.size(), 2u);
  EXPECT_EQ(sink_.events[0].queue_ordinal, 0u);
  EXPECT_EQ(sink_.events[1].queue_ordinal, 2u);
  EXPECT_EQ(sink_.devices.back().queue_count, 3u);
}

TEST_F(GpuProfileTest, RegistrationProgressesDuringBlockedFlush) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append(0);
  std::barrier<> sink_entered(2);
  std::barrier<> producer_finished(2);
  std::thread producer([&] {
    sink_entered.arrive_and_wait();
    iree_status_t status =
        iree_hal_amd_gpu_profile_register_queue(profile_, 255);
    if (iree_status_is_ok(status)) {
      Append(255);
    }
    IREE_EXPECT_OK(status);
    Append(0);
    producer_finished.arrive_and_wait();
  });
  sink_.during_write = [&] {
    sink_entered.arrive_and_wait();
    producer_finished.arrive_and_wait();
  };
  IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  producer.join();
  // The current flush owns its original directory and event snapshots.
  EXPECT_EQ(sink_.queues.size(), 2u);
  EXPECT_EQ(sink_.events.size(), 1u);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.queues.size(), 3u);
  EXPECT_EQ(sink_.queues.back().queue_ordinal, 255u);
  EXPECT_EQ(sink_.devices.back().queue_count, 3u);
  ASSERT_EQ(sink_.events.size(), 3u);
  EXPECT_EQ(sink_.events[1].queue_ordinal, 0u);
  EXPECT_EQ(sink_.events[2].queue_ordinal, 255u);
}

TEST_F(GpuProfileTest, RegistrationAllocationDoesNotBlockFlushOrLoseHistory) {
  auto& allocator = allocator_;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_begin(
      &options_, 37, 2, 0, allocator.allocator(), &profile_));
  Append(0);
  allocator.before_allocate = [&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  };
  allocator.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_profile_register_queue(profile_, 255));
  EXPECT_EQ(sink_.events.size(), 1u);
  EXPECT_EQ(sink_.queues.size(), 2u);
  allocator.fail = false;
  allocator.before_allocate = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_register_queue(profile_, 255));
  Append(255);
  const uint32_t allocations = allocator.allocation_count;
  IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(profile_));
  profile_ = nullptr;
  EXPECT_EQ(allocator.allocation_count, allocations);
  ASSERT_EQ(sink_.events.size(), 2u);
  EXPECT_EQ(sink_.events.back().queue_ordinal, 255u);
  EXPECT_EQ(sink_.queues.size(), 3u);
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
  options_.data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
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

TEST_F(GpuProfileTest, PlacementGrowthProgressesDuringBlockedLoadOutput) {
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
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  const auto append_load = [&](uint64_t index) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_executable_add_load(
        metadata, executable.image.virtual_base, executable.image.byte_length,
        0x300000000ull + index * 0x1000));
  };
  for (uint64_t i = 0; i < 65; ++i) {
    append_load(i);
  }
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  std::barrier<> sink_entered(2);
  std::barrier<> producer_finished(2);
  std::thread producer([&] {
    sink_entered.arrive_and_wait();
    // Grow the backing while the sink still borrows its first output batch.
    for (uint64_t i = 65; i < 193; ++i) {
      append_load(i);
    }
    iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                                &operation.completion);
    producer_finished.arrive_and_wait();
  });
  sink_.during_write = [&] {
    sink_entered.arrive_and_wait();
    producer_finished.arrive_and_wait();
  };
  IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  producer.join();
  EXPECT_EQ(sink_.loads.size(), 67u);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.loads.size(), 195u);
  EXPECT_EQ(sink_.images.size(), 1u);
  for (uint64_t i = 0; i < 193; ++i) {
    EXPECT_EQ(sink_.loads[i + 2].load_delta,
              0x300000000ull + i * 0x1000 - executable.image.virtual_base);
  }
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_executable_release(&executable.base);
}

TEST_F(GpuProfileTest, FailedLoadBatchRetriesWithoutRepeatingAcceptedRecords) {
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
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  for (uint64_t i = 0; i < 65; ++i) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_executable_add_load(
        metadata, executable.image.virtual_base, executable.image.byte_length,
        0x300000000ull + i * 0x1000));
  }
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  sink_.during_write = [&] {
    sink_.fail_content_type =
        IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS;
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_GT(sink_.loads.size(), 2u);
  EXPECT_LT(sink_.loads.size(), 67u);
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.loads.size(), 67u);
  EXPECT_EQ(sink_.images.size(), 1u);
  for (uint64_t i = 0; i < 65; ++i) {
    EXPECT_EQ(sink_.loads[i + 2].load_delta,
              0x300000000ull + i * 0x1000 - executable.image.virtual_base);
  }
  iree_hal_amd_gpu_profile_executable_release(metadata);
  iree_hal_executable_release(&executable.base);
}

TEST_F(GpuProfileTest, PlacementAllocationDoesNotBlockFlushOrLoseHistory) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  ASSERT_NO_FATAL_FAILURE(Begin());
  auto& allocator = allocator_;
  ReflectionExecutable executable;
  iree_hal_amd_gpu_profile_executable_t* metadata = nullptr;
  IREE_ASSERT_OK(executable.Capture(allocator.allocator(), &metadata));
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      &operation);
  operation.dispatch = {metadata, iree_hal_profile_dispatch_event_default()};
  operation.completion.count = 1;
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  allocator.before_allocate = [&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  };
  allocator.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_profile_executable_add_load(
                            metadata, executable.image.virtual_base,
                            executable.image.byte_length, 0x300000000ull));
  EXPECT_EQ(sink_.loads.size(), 2u);
  allocator.fail = false;
  allocator.before_allocate = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_executable_add_load(
      metadata, executable.image.virtual_base, executable.image.byte_length,
      0x300000000ull));
  iree_hal_amd_gpu_profile_operation_complete(profile_, &operation,
                                              &operation.completion);
  const uint32_t allocations = allocator.allocation_count;
  IREE_EXPECT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(allocator.allocation_count, allocations);
  ASSERT_EQ(sink_.loads.size(), 3u);
  EXPECT_EQ(sink_.loads.back().load_delta,
            0x300000000ull - executable.image.virtual_base);
  EXPECT_EQ(sink_.images.size(), 1u);
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

// Host stand-in for the device's trace buffers and info words. Tests write the
// values the stop sequence would copy back.
struct TraceStorage {
  // GFX11.5 register table borrowed by the bracket.
  iree_hal_amd_gpu_pm4_thread_trace_profile_t registers;
  // Bracket handed to the session.
  iree_hal_amd_gpu_profile_trace_storage_t storage = {};
  // One buffer per engine with a recognizable byte pattern.
  std::array<std::vector<uint8_t>, 2> buffers;
  // Info words of both engines.
  std::array<uint32_t, 8> info = {};

  explicit TraceStorage(uint32_t limit = 2) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
        {11, 5, 1}, &registers));
    auto& config = storage.config;
    config.profile = &registers;
    config.engine_count = 2;
    config.engine_mask = 3;
    config.workgroup_processor = 4;
    config.simd = 1;
    config.buffer_length = 128 * 1024;
    for (uint32_t i = 0; i < 2; ++i) {
      buffers[i].resize(config.buffer_length);
      for (size_t j = 0; j < buffers[i].size(); ++j) {
        buffers[i][j] = static_cast<uint8_t>(j * 7 + i);
      }
      storage.buffers[i] = buffers[i].data();
      config.buffer_addresses[i] = 0x100000000ull + i * 0x1000000ull;
    }
    config.info_address = 0x200000000ull;
    storage.info_words = info.data();
    storage.trace_limit = limit;
    storage.stable_power_state = true;
  }

  // What the stop sequence leaves for one engine.
  void Stop(uint32_t engine, uint32_t units, uint32_t status,
            uint32_t dropped) {
    info[engine * 4] = units;
    info[engine * 4 + 1] = status;
    info[engine * 4 + 2] = dropped;
  }
};

class GpuProfileTraceTest : public GpuProfileTest {
 protected:
  void BeginTracing(
      uint64_t families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES) {
    options_.data_families = families;
    ASSERT_NO_FATAL_FAILURE(Begin(100000000));
    iree_hal_amd_gpu_profile_set_trace_storage(profile_, &trace_.storage);
    IREE_ASSERT_OK(executable_.Capture(&metadata_));
  }
  void TearDown() override {
    iree_hal_amd_gpu_profile_executable_release(metadata_);
    GpuProfileTest::TearDown();
    iree_hal_executable_release(&executable_.base);
  }
  // A captured direct dispatch of the first function.
  iree_hal_amd_gpu_profile_operation_t Direct() {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
        &operation);
    operation.dispatch = {metadata_, iree_hal_profile_dispatch_event_default()};
    operation.dispatch.event.executable_id =
        iree_hal_amd_gpu_profile_executable_id(metadata_);
    operation.dispatch.event.function_ordinal = 0;
    operation.completion.count = 1;
    return operation;
  }
  // Publishes the claimed bracket and retires the operation.
  void Complete(iree_hal_amd_gpu_profile_operation_t* operation) {
    operation->traced = true;
    iree_hal_amd_gpu_profile_operation_complete(profile_, operation,
                                                &operation->completion);
  }
  // Device trace resources, outliving the session.
  TraceStorage trace_;
  // Source of code-object metadata for traced dispatches.
  ReflectionExecutable executable_;
  // Retained host metadata of executable_.
  iree_hal_amd_gpu_profile_executable_t* metadata_ = nullptr;
};

TEST_F(GpuProfileTraceTest, FamilyIsAdmittedButTracesNothingWithoutStorage) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  EXPECT_TRUE(iree_hal_amd_gpu_profile_traces_executables(profile_));
  EXPECT_TRUE(iree_hal_amd_gpu_profile_captures_dispatches(profile_));
  IREE_ASSERT_OK(executable_.Capture(&metadata_));
  auto operation = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, nullptr), nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(nullptr, &operation), nullptr);
}

TEST_F(GpuProfileTraceTest, StorageIsIgnoredBySessionsThatDoNotTrace) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  iree_hal_amd_gpu_profile_set_trace_storage(profile_, &trace_.storage);
  IREE_ASSERT_OK(executable_.Capture(&metadata_));
  auto operation = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
}

TEST_F(GpuProfileTraceTest, BracketIsExclusiveBoundedAndHeldUntilFlush) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto first = Direct();
  trace_.info.fill(0);
  const auto* config = iree_hal_amd_gpu_profile_trace_claim(profile_, &first);
  ASSERT_EQ(config, &trace_.storage.config);
  // Claiming resets the info words so a stale stop cannot read as a result.
  for (const uint32_t word : trace_.info) {
    EXPECT_EQ(word, 0xffffffffu);
  }
  // Another queue's candidate runs untraced while the bracket is held. Its
  // publication retries count as one untraced operation.
  auto second = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &second), nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &second), nullptr);
  EXPECT_TRUE(second.trace_refused);
  // An unpublished claim returns both the bracket and its budget.
  iree_hal_amd_gpu_profile_trace_cancel(profile_);
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &first), nullptr);
  trace_.Stop(0, 4, 0x90008000, 0);
  trace_.Stop(1, 2, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&first));
  // Stopped but not flushed: the buffers still hold the only copy.
  auto waiting = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &waiting), nullptr);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.traces.size(), 2u);
  // Two operations were refused while the first trace held the bracket.
  ASSERT_EQ(sink_.captures.size(), 2u);
  EXPECT_EQ(sink_.captures[0].skipped_count, 2u);
  EXPECT_EQ(sink_.captures[0].trace_limit, 2u);
  // The second and last trace of the bound.
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &second), nullptr);
  trace_.Stop(0, 1, 0x90008000, 0);
  trace_.Stop(1, 1, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&second));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.traces.size(), 4u);
  EXPECT_EQ(sink_.traces[0].trace_id, 1u);
  EXPECT_EQ(sink_.traces[2].trace_id, 2u);
  EXPECT_EQ(sink_.captures[2].skipped_count, 0u);
  auto third = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &third), nullptr);
}

TEST_F(GpuProfileTraceTest, DirectTraceJoinsItsDispatchEventAndStatesCoverage) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  // Engine 0 wrote 96 bytes cleanly; engine 1 dropped tokens.
  trace_.Stop(0, 3, 0x90008000, 0);
  trace_.Stop(1, 5, 0x90008000, 41268);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.dispatches.size(), 1u);
  ASSERT_EQ(sink_.traces.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& record = sink_.traces[i];
    EXPECT_EQ(record.format,
              IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT);
    EXPECT_EQ(record.flags,
              IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_DISPATCH_EVENT);
    EXPECT_EQ(record.dispatch_event_id, sink_.dispatches[0].event_id);
    EXPECT_EQ(record.submission_id, sink_.dispatches[0].submission_id);
    EXPECT_EQ(record.executable_id, sink_.dispatches[0].executable_id);
    EXPECT_EQ(record.function_ordinal, 0u);
    EXPECT_EQ(record.command_buffer_id, 0u);
    EXPECT_EQ(record.shader_engine, i);
    EXPECT_EQ(record.physical_device_ordinal, 0u);
  }
  EXPECT_EQ(sink_.trace_bytes[0],
            std::vector<uint8_t>(trace_.buffers[0].begin(),
                                 trace_.buffers[0].begin() + 96));
  EXPECT_EQ(sink_.trace_bytes[1],
            std::vector<uint8_t>(trace_.buffers[1].begin(),
                                 trace_.buffers[1].begin() + 160));
  EXPECT_EQ(sink_.trace_flags[0], 0u);
  EXPECT_EQ(sink_.trace_flags[1], IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  EXPECT_EQ(sink_.dropped, 41268u);
  ASSERT_EQ(sink_.captures.size(), 2u);
  const auto& capture = sink_.captures[1];
  EXPECT_EQ(capture.trace_id, sink_.traces[1].trace_id);
  EXPECT_EQ(capture.shader_engine, 1u);
  EXPECT_EQ(capture.shader_engine_mask, 3u);
  EXPECT_EQ(capture.target_compute_unit, 4u);
  EXPECT_EQ(capture.simd_select, 1u);
  EXPECT_EQ(capture.dispatch_count, 1u);
  EXPECT_EQ(capture.buffer_byte_length, 128u * 1024);
  EXPECT_EQ(capture.written_byte_length, 160u);
  EXPECT_EQ(capture.dropped_count, 41268u);
  EXPECT_EQ(capture.hardware_status, 0x90008000u);
  EXPECT_EQ(capture.token_mask, 0x003f0683u);
  EXPECT_EQ(capture.flags,
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_STABLE_POWER_STATE |
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_FINISHED);
  // The code object the trace's addresses resolve against precedes it.
  ASSERT_EQ(sink_.code_objects.size(), 1u);
  EXPECT_EQ(sink_.code_objects[0].executable_id, sink_.traces[0].executable_id);
}

TEST_F(GpuProfileTraceTest, VerifiedTargetNeitherWarnsNorMarksCaptures) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(iree_hal_amd_gpu_profile_warn_unverified(profile_, {11, 5, 1}));
  EXPECT_FALSE(iree_hal_amd_gpu_profile_warn_unverified(nullptr, {11, 0, 0}));
  EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  trace_.Stop(0, 3, 0x90008000, 0);
  trace_.Stop(1, 5, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.captures.size(), 2u);
  for (const auto& capture : sink_.captures) {
    EXPECT_FALSE(
        capture.flags &
        IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNVERIFIED_TARGET);
  }
}

TEST_F(GpuProfileTraceTest, UnverifiedTargetWarnsOnceAndMarksEveryCapture) {
  // GFX11.0: a table that was never run, with a base-relative write pointer.
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
      {11, 0, 3}, &trace_.registers));
  ASSERT_FALSE(trace_.registers.verified);
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  ::testing::internal::CaptureStderr();
  EXPECT_TRUE(iree_hal_amd_gpu_profile_warn_unverified(profile_, {11, 0, 3}));
  const std::string warning = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(warning.find("executable traces on gfx11.0.3 use UNVERIFIED"),
            std::string::npos)
      << warning;
  EXPECT_EQ(warning.find("counter samples"), std::string::npos) << warning;
  EXPECT_NE(warning.find("only gfx11.5.1 was tested"), std::string::npos);
  EXPECT_EQ(std::count(warning.begin(), warning.end(), '\n'), 1);
  EXPECT_EQ(warning.back(), '\n');

  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  // The hardware leaves the buffer base plus the written units.
  const auto& config = trace_.storage.config;
  trace_.Stop(0, static_cast<uint32_t>(config.buffer_addresses[0] >> 5) + 3,
              0x90008000, 0);
  trace_.Stop(1, static_cast<uint32_t>(config.buffer_addresses[1] >> 5) + 5,
              0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.captures.size(), 2u);
  EXPECT_EQ(sink_.captures[0].written_byte_length, 96u);
  EXPECT_EQ(sink_.captures[1].written_byte_length, 160u);
  for (const auto& capture : sink_.captures) {
    EXPECT_EQ(
        capture.flags,
        IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_STABLE_POWER_STATE |
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_FINISHED |
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNVERIFIED_TARGET);
  }
  // The trace records and their bytes are those of a verified target.
  ASSERT_EQ(sink_.traces.size(), 2u);
  EXPECT_EQ(sink_.trace_bytes[0],
            std::vector<uint8_t>(trace_.buffers[0].begin(),
                                 trace_.buffers[0].begin() + 96));
}

TEST_F(GpuProfileTraceTest, UncollectedAndFullEnginesNeverClaimCoverage) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  // Engine 0: the stop sequence never ran. Engine 1: the buffer filled.
  trace_.Stop(1, 128 * 1024 / 32 - 1, 0x91008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.captures.size(), 2u);
  EXPECT_EQ(sink_.captures[0].flags,
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_STABLE_POWER_STATE |
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNCOLLECTED);
  EXPECT_EQ(sink_.captures[0].written_byte_length, 0u);
  EXPECT_EQ(sink_.captures[1].flags,
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_STABLE_POWER_STATE |
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_FINISHED |
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_BUFFER_FULL |
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_WRITE_ERROR);
  // Only the engine that wrote bytes has a trace chunk, marked truncated.
  ASSERT_EQ(sink_.traces.size(), 1u);
  EXPECT_EQ(sink_.traces[0].shader_engine, 1u);
  EXPECT_EQ(sink_.trace_bytes[0].size(), 128u * 1024 - 32);
  EXPECT_EQ(sink_.trace_flags[0], IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
}

TEST_F(GpuProfileTraceTest, RecordingIsTracedWholeWhenAnyDispatchMatches) {
  options_.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  options_.capture_filter.executable_function_pattern = IREE_SV("sec*");
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto record = iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = 51;
  auto make = [&](std::initializer_list<uint32_t> ordinals) {
    iree_hal_amd_gpu_profile_commands_t* commands = nullptr;
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_commands_create(
        &record, iree_allocator_system(), &commands));
    uint32_t index = 0;
    for (const uint32_t ordinal : ordinals) {
      iree_hal_amd_gpu_profile_dispatch_t dispatch = {
          .executable = metadata_,
          .event = iree_hal_profile_dispatch_event_default(),
      };
      dispatch.event.executable_id =
          iree_hal_amd_gpu_profile_executable_id(metadata_);
      dispatch.event.function_ordinal = ordinal;
      dispatch.event.command_buffer_id = 51;
      dispatch.event.command_index = index++;
      IREE_EXPECT_OK(
          iree_hal_amd_gpu_profile_commands_append(commands, dispatch));
    }
    return commands;
  };
  auto operation_for = [&](iree_hal_amd_gpu_profile_commands_t* commands) {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 0, 1, 1,
        &operation);
    operation.event.command_buffer_id = 51;
    operation.commands = commands;
    operation.completion.count =
        iree_hal_amd_gpu_profile_commands_count(commands);
    return operation;
  };
  // No dispatch of this recording matches the function pattern.
  auto* unmatched = make({0, 0});
  auto skipped = operation_for(unmatched);
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &skipped), nullptr);
  iree_hal_amd_gpu_profile_commands_release(unmatched);
  // One of three does: the whole recording is bracketed.
  auto* commands = make({0, 1, 0});
  auto operation = operation_for(commands);
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  trace_.Stop(0, 2, 0x90008000, 0);
  trace_.Stop(1, 2, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  iree_hal_amd_gpu_profile_commands_release(commands);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  // Only the matching dispatch has an event; the trace names the submission.
  ASSERT_EQ(sink_.dispatches.size(), 1u);
  ASSERT_EQ(sink_.traces.size(), 2u);
  const auto& trace = sink_.traces[0];
  EXPECT_EQ(trace.flags, IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_NONE);
  EXPECT_EQ(trace.dispatch_event_id, 0u);
  EXPECT_EQ(trace.command_buffer_id, 51u);
  EXPECT_EQ(trace.command_index, UINT32_MAX);
  EXPECT_EQ(trace.function_ordinal, UINT32_MAX);
  EXPECT_EQ(trace.submission_id, sink_.dispatches[0].submission_id);
  ASSERT_EQ(sink_.captures.size(), 2u);
  EXPECT_EQ(sink_.captures[0].dispatch_count, 3u);
  EXPECT_TRUE(
      sink_.captures[0].flags &
      IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_COMMAND_BUFFER_SCOPE);
  EXPECT_EQ(sink_.commands, std::vector<uint64_t>({51}));
  EXPECT_EQ(sink_.code_objects.size(), 1u);
}

TEST_F(GpuProfileTraceTest, FailedTraceWriteRetriesWithoutRepeatingRecords) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  trace_.Stop(0, 1, 0x90008000, 0);
  trace_.Stop(1, 1, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  sink_.fail_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.captures.size(), 1u);
  EXPECT_TRUE(sink_.traces.empty());
  // The bracket stays held: its buffers are still the only copy.
  auto next = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_trace_claim(profile_, &next), nullptr);
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.captures.size(), 2u);
  EXPECT_EQ(sink_.traces.size(), 2u);
  EXPECT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &next), nullptr);
  iree_hal_amd_gpu_profile_trace_cancel(profile_);
}

TEST_F(GpuProfileTraceTest, EndEmitsACompletedTraceAndReleasesAnUnflushedOne) {
  ASSERT_NO_FATAL_FAILURE(BeginTracing());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &operation),
            nullptr);
  trace_.Stop(0, 1, 0x90008000, 0);
  trace_.Stop(1, 1, 0x90008000, 0);
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  auto* profile = profile_;
  profile_ = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_end(profile));
  EXPECT_EQ(sink_.traces.size(), 2u);
  // A session whose sink fails at the end still releases the trace's owner.
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  iree_hal_amd_gpu_profile_set_trace_storage(profile_, &trace_.storage);
  auto second = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_trace_claim(profile_, &second), nullptr);
  ASSERT_NO_FATAL_FAILURE(Complete(&second));
  sink_.fail_write = true;
  profile = profile_;
  profile_ = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_end(profile));
  sink_.fail_write = false;
}

// Host stand-in for the device's counter selection and state words: two engine
// counters on two shader engines and one instance counter on three workgroup
// processors, so a sample holds seven values.
struct CounterStorage {
  // GFX11.5 register table borrowed by the selection.
  iree_hal_amd_gpu_pm4_perf_counter_profile_t registers;
  // Selection handed to the session.
  iree_hal_amd_gpu_profile_counter_storage_t storage = {};
  // State words every start sequence would copy; reset defaults.
  std::array<uint32_t, 4> state = {0, 0, 0x1ff, 0xffffffff};

  explicit CounterStorage(uint64_t limit = 8) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
        {11, 5, 1}, &registers));
    auto& config = storage.config;
    config.profile = &registers;
    config.engine_count = 2;
    config.engine_counters[config.engine_counter_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(&registers,
                                                       IREE_SV("SQG_WAVES"));
    config.engine_counters[config.engine_counter_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(
            &registers, IREE_SV("SQG_BUSY_CYCLES"));
    config.instance_counters[config.instance_counter_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(&registers,
                                                       IREE_SV("SQ_INSTS_ALL"));
    for (uint32_t i = 0; i < 3; ++i) {
      config.instances[config.instance_count++] =
          iree_hal_amd_gpu_pm4_perf_counter_instance(i / 2, 0, i);
    }
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config));
    storage.signature =
        iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config);
    storage.state_address = 0x300000000ull;
    storage.state_words = state.data();
    storage.sample_limit = limit;
    storage.stable_power_state = true;
  }
};

class GpuProfileCounterTest : public GpuProfileTest {
 protected:
  // Bytes of one dispatch's sample: a timestamp pair and two raw samples.
  static constexpr uint32_t kStride = 16 + 2 * 7 * 8;

  void BeginCounting(
      uint64_t families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES) {
    options_.data_families = families;
    ASSERT_NO_FATAL_FAILURE(Begin(100000000));
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_set_counter_storage(
        profile_, &counters_.storage));
    IREE_ASSERT_OK(executable_.Capture(&metadata_));
  }
  void TearDown() override {
    iree_hal_amd_gpu_profile_executable_release(metadata_);
    GpuProfileTest::TearDown();
    iree_hal_executable_release(&executable_.base);
  }
  // A captured direct dispatch of the first function on |queue|.
  iree_hal_amd_gpu_profile_operation_t Direct(uint32_t queue = 0) {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, queue, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
        &operation);
    operation.dispatch = {metadata_, iree_hal_profile_dispatch_event_default()};
    operation.dispatch.event.executable_id =
        iree_hal_amd_gpu_profile_executable_id(metadata_);
    operation.dispatch.event.function_ordinal = 0;
    operation.completion.count = 1;
    operation.completion.sample_stride = kStride;
    return operation;
  }
  // A recording of |ordinals| with command buffer id 51.
  iree_hal_amd_gpu_profile_commands_t* Recording(
      std::initializer_list<uint32_t> ordinals) {
    auto record = iree_hal_profile_command_buffer_record_default();
    record.command_buffer_id = 51;
    iree_hal_amd_gpu_profile_commands_t* commands = nullptr;
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_commands_create(
        &record, iree_allocator_system(), &commands));
    uint32_t index = 0;
    for (const uint32_t ordinal : ordinals) {
      iree_hal_amd_gpu_profile_dispatch_t dispatch = {
          .executable = metadata_,
          .event = iree_hal_profile_dispatch_event_default(),
      };
      dispatch.event.executable_id =
          iree_hal_amd_gpu_profile_executable_id(metadata_);
      dispatch.event.function_ordinal = ordinal;
      dispatch.event.command_buffer_id = 51;
      dispatch.event.command_index = index++;
      IREE_EXPECT_OK(
          iree_hal_amd_gpu_profile_commands_append(commands, dispatch));
    }
    return commands;
  }
  iree_hal_amd_gpu_profile_operation_t Execution(
      iree_hal_amd_gpu_profile_commands_t* commands) {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 0, 1, 1,
        &operation);
    operation.event.command_buffer_id = 51;
    operation.commands = commands;
    operation.completion.count =
        iree_hal_amd_gpu_profile_commands_count(commands);
    operation.completion.sample_stride = kStride;
    return operation;
  }
  // What the hardware leaves for a published operation: dispatch |i| runs
  // from tick 100 i to 100 i + 50 and every counter value v advances by
  // (i + 1) * (v + 1) over arbitrary begin values.
  void Publish(iree_hal_amd_gpu_profile_operation_t* operation,
               bool write_end = true) {
    samples_.assign(size_t{operation->completion.count} * kStride, 0xff);
    for (uint32_t i = 0; i < operation->completion.count; ++i) {
      uint8_t* sample = samples_.data() + size_t{i} * kStride;
      const uint64_t ticks[2] = {100ull * i, 100ull * i + 50};
      std::memcpy(sample, ticks, sizeof(ticks));
      for (uint32_t v = 0; v < 7; ++v) {
        const uint64_t begin = 0xfffffff0ull + 7 * v;
        const uint64_t end = begin + uint64_t{i + 1} * (v + 1);
        // Engine values are 64 bits wide, instance values 32.
        const size_t width = v < 4 ? 8 : 4;
        std::memcpy(sample + 16 + 8 * v, &begin, width);
        if (write_end) {
          std::memcpy(sample + 16 + 56 + 8 * v, &end, width);
        }
      }
    }
    operation->completion.samples =
        reinterpret_cast<const iree_hal_amd_gpu_profile_sample_t*>(
            samples_.data());
    operation->counted = true;
  }
  void Complete(iree_hal_amd_gpu_profile_operation_t* operation) {
    iree_hal_amd_gpu_profile_operation_complete(profile_, operation,
                                                &operation->completion);
  }
  // Device counter resources, outliving the session.
  CounterStorage counters_;
  // Source of executable metadata for sampled dispatches.
  ReflectionExecutable executable_;
  // Retained host metadata of executable_.
  iree_hal_amd_gpu_profile_executable_t* metadata_ = nullptr;
  // Native sample storage of the operation being completed.
  std::vector<uint8_t> samples_;
};

TEST_F(GpuProfileCounterTest, FamilyIsAdmittedButSamplesNothingWithoutStorage) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                           IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  IREE_ASSERT_OK(executable_.Capture(&metadata_));
  EXPECT_TRUE(iree_hal_amd_gpu_profile_counts_dispatches(profile_));
  EXPECT_TRUE(iree_hal_amd_gpu_profile_timestamps_dispatches(profile_));
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_sample_byte_length(profile_), 0u);
  auto operation = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_TRUE(sink_.counter_sets.empty());
  EXPECT_TRUE(sink_.counter_captures.empty());
}

TEST_F(GpuProfileCounterTest, SamplesNeedADeviceClock) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, 0,
                                     iree_allocator_system(), &profile_));
  EXPECT_EQ(sink_.begins, 0);
}

TEST_F(GpuProfileCounterTest, SelectionWithoutTheFamilyIsRejected) {
  iree_hal_profile_counter_set_selection_t selection = {};
  options_.counter_set_count = 1;
  options_.counter_sets = &selection;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, 100000000,
                                     iree_allocator_system(), &profile_));
  EXPECT_EQ(sink_.begins, 0);
}

TEST_F(GpuProfileCounterTest, StorageIsIgnoredBySessionsThatDoNotCount) {
  options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  ASSERT_NO_FATAL_FAILURE(Begin(100000000));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_set_counter_storage(
      profile_, &counters_.storage));
  IREE_ASSERT_OK(executable_.Capture(&metadata_));
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_sample_byte_length(profile_), 0u);
  auto operation = Direct();
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
}

TEST_F(GpuProfileCounterTest, UnverifiedTargetWarnsOnceAndMarksTheCapture) {
  // GFX11.0 shares the GFX11.5 registers and events and was never run.
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
      {11, 0, 0}, &counters_.registers));
  ASSERT_FALSE(counters_.registers.verified);
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  ::testing::internal::CaptureStderr();
  EXPECT_TRUE(iree_hal_amd_gpu_profile_warn_unverified(profile_, {11, 0, 0}));
  const std::string warning = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(warning.find("counter samples on gfx11.0.0 use UNVERIFIED"),
            std::string::npos)
      << warning;
  EXPECT_EQ(warning.find("executable traces"), std::string::npos) << warning;
  EXPECT_EQ(std::count(warning.begin(), warning.end(), '\n'), 1);

  auto operation = Direct();
  ASSERT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            &counters_.storage);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  // The sample itself is unchanged; the capture record carries the caveat.
  ASSERT_EQ(sink_.counter_samples.size(), 1u);
  EXPECT_EQ(sink_.counter_values[0],
            (std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}));
  ASSERT_EQ(sink_.counter_captures.size(), 1u);
  EXPECT_EQ(sink_.counter_captures[0].flags,
            IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_STABLE_POWER_STATE |
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_EXCLUSIVE_WINDOW |
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_PROCESS_SCOPE |
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_UNVERIFIED_TARGET);
}

TEST_F(GpuProfileCounterTest, VerifiedCountersAreNotNamedInTheWarning) {
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(iree_hal_amd_gpu_profile_warn_unverified(profile_, {11, 5, 1}));
  EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

TEST_F(GpuProfileCounterTest, DirectSampleJoinsItsDispatchAndNamesItsCounters) {
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_sample_byte_length(profile_),
            2u * 7 * 8);
  auto operation = Direct();
  ASSERT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            &counters_.storage);
  EXPECT_EQ(operation.counter_grant, 1u);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));

  ASSERT_EQ(sink_.counter_sets.size(), 1u);
  EXPECT_EQ(sink_.counter_set_names[0], "amdgpu.perfmon");
  EXPECT_EQ(sink_.counter_sets[0].counter_set_id, 1u);
  EXPECT_EQ(sink_.counter_sets[0].counter_count, 3u);
  EXPECT_EQ(sink_.counter_sets[0].sample_value_count, 7u);
  EXPECT_EQ(sink_.counter_sets[0].physical_device_ordinal, 0u);
  ASSERT_EQ(sink_.counters.size(), 3u);
  const std::array<std::array<uint32_t, 3>, 3> layout = {{
      // unit, value offset, value count
      {IREE_HAL_PROFILE_COUNTER_UNIT_COUNT, 0, 2},
      {IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES, 2, 2},
      {IREE_HAL_PROFILE_COUNTER_UNIT_COUNT, 4, 3},
  }};
  const std::array<std::array<std::string, 2>, 3> names = {{
      {"SQG", "SQG_WAVES"},
      {"SQG", "SQG_BUSY_CYCLES"},
      {"SQ", "SQ_INSTS_ALL"},
  }};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(sink_.counters[i].counter_set_id, 1u);
    EXPECT_EQ(sink_.counters[i].counter_ordinal, i);
    EXPECT_EQ(sink_.counters[i].flags, IREE_HAL_PROFILE_COUNTER_FLAG_RAW);
    EXPECT_EQ(sink_.counters[i].unit, layout[i][0]);
    EXPECT_EQ(sink_.counters[i].sample_value_offset, layout[i][1]);
    EXPECT_EQ(sink_.counters[i].sample_value_count, layout[i][2]);
    EXPECT_EQ(sink_.counter_texts[i][0], names[i][0]);
    EXPECT_EQ(sink_.counter_texts[i][1], names[i][1]);
    EXPECT_FALSE(sink_.counter_texts[i][2].empty());
  }

  ASSERT_EQ(sink_.dispatches.size(), 1u);
  ASSERT_EQ(sink_.counter_samples.size(), 1u);
  const auto& sample = sink_.counter_samples[0];
  EXPECT_EQ(sample.scope, IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH);
  EXPECT_EQ(sample.flags,
            IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT |
                IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE);
  EXPECT_EQ(sample.counter_set_id, 1u);
  EXPECT_EQ(sample.sample_id, 1u);
  EXPECT_EQ(sample.dispatch_event_id, sink_.dispatches[0].event_id);
  EXPECT_EQ(sample.submission_id, sink_.dispatches[0].submission_id);
  EXPECT_EQ(sample.executable_id, sink_.dispatches[0].executable_id);
  EXPECT_EQ(sample.function_ordinal, 0u);
  EXPECT_EQ(sample.command_buffer_id, 0u);
  EXPECT_EQ(sample.command_index, UINT32_MAX);
  EXPECT_EQ(sample.start_tick, 0u);
  EXPECT_EQ(sample.end_tick, 50u);
  EXPECT_EQ(sample.physical_device_ordinal, 0u);
  // Differences, including the engine values that crossed 32 bits and the
  // instance values that wrapped.
  EXPECT_EQ(sink_.counter_values[0],
            (std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}));

  ASSERT_EQ(sink_.counter_captures.size(), 1u);
  const auto& capture = sink_.counter_captures[0];
  EXPECT_EQ(capture.flags,
            IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_STABLE_POWER_STATE |
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_EXCLUSIVE_WINDOW |
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_PROCESS_SCOPE);
  EXPECT_EQ(capture.counter_set_id, 1u);
  EXPECT_EQ(capture.shader_engine_count, 2u);
  EXPECT_EQ(capture.instance_count, 3u);
  EXPECT_EQ(capture.sample_limit, 8u);
  EXPECT_EQ(capture.sampled_operation_count, 1u);
  EXPECT_EQ(capture.sampled_dispatch_count, 1u);
  EXPECT_EQ(capture.skipped_operation_count, 0u);
  EXPECT_EQ(capture.dropped_sample_count, 0u);
  EXPECT_EQ(capture.invalid_sample_count, 0u);
  EXPECT_EQ(capture.foreign_state_count, 0u);

  // Nothing changed: metadata and the capture record are not repeated.
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.counter_sets.size(), 1u);
  EXPECT_EQ(sink_.counters.size(), 3u);
  EXPECT_EQ(sink_.counter_captures.size(), 1u);
}

TEST_F(GpuProfileCounterTest, SamplesWithoutDispatchEventsCarryNoEventJoin) {
  ASSERT_NO_FATAL_FAILURE(
      BeginCounting(IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES));
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_TRUE(sink_.dispatches.empty());
  ASSERT_EQ(sink_.counter_samples.size(), 1u);
  EXPECT_EQ(sink_.counter_samples[0].flags,
            IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE);
  EXPECT_EQ(sink_.counter_samples[0].dispatch_event_id, 0u);
  // The executable the sample names is still described.
  EXPECT_EQ(sink_.executables.size(), 1u);
}

TEST_F(GpuProfileCounterTest, RangeIsSharedWithinAQueueAndExclusiveAcross) {
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  auto first = Direct(0);
  auto second = Direct(0);
  auto other = Direct(1);
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &first, NULL),
            nullptr);
  // One ring serializes the owner's operations.
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &second, NULL),
            nullptr);
  // Another queue would be counted together with the owner: refused, and
  // counted once however often publication retries.
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &other, NULL),
            nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &other, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&first));
  ASSERT_NO_FATAL_FAILURE(Complete(&first));
  // The owner still has an operation in flight.
  auto late = Direct(1);
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &late, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&second));
  ASSERT_NO_FATAL_FAILURE(Complete(&second));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_FALSE(sink_.counter_captures.empty());
  EXPECT_EQ(sink_.counter_captures.back().sampled_operation_count, 2u);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 2u);
  // Free again: a refused operation that is admitted on a retry was not
  // skipped after all.
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &other, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&other));
  ASSERT_NO_FATAL_FAILURE(Complete(&other));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.counter_captures.back().sampled_operation_count, 3u);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 1u);
  ASSERT_EQ(sink_.counter_samples.size(), 3u);
  EXPECT_EQ(sink_.counter_samples[2].queue_ordinal, 1u);
  EXPECT_EQ(sink_.counter_samples[2].stream_id, 2u);
}

TEST_F(GpuProfileCounterTest, CancelReturnsTheRangeAndTheBound) {
  counters_.storage.sample_limit = 1;
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  auto operation = Direct(0);
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
  iree_hal_amd_gpu_profile_counter_cancel(profile_, &operation);
  EXPECT_EQ(operation.counter_grant, 0u);
  auto other = Direct(1);
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &other, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&other));
  ASSERT_NO_FATAL_FAILURE(Complete(&other));
  // The bound is spent; the range is free but nothing more is sampled.
  auto third = Direct(0);
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &third, NULL),
            nullptr);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  const auto& capture = sink_.counter_captures.back();
  EXPECT_TRUE(capture.flags &
              IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_LIMIT_REACHED);
  EXPECT_EQ(capture.sampled_dispatch_count, 1u);
  EXPECT_EQ(capture.skipped_operation_count, 0u);
}

TEST_F(GpuProfileCounterTest,
       RecordingIsSampledPerDispatchWithinBoundAndFilter) {
  counters_.storage.sample_limit = 3;
  options_.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  options_.capture_filter.executable_function_pattern = IREE_SV("sec*");
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  const uint64_t signature = counters_.storage.signature;
  const uint64_t other_signature = signature + 1, no_signature = 0;

  // No dispatch matches the function pattern: nothing to sample.
  auto* unmatched = Recording({0, 0});
  auto ignored = Execution(unmatched);
  EXPECT_EQ(
      iree_hal_amd_gpu_profile_counter_claim(profile_, &ignored, &signature),
      nullptr);
  iree_hal_amd_gpu_profile_commands_release(unmatched);

  auto* commands = Recording({1, 0, 1, 1});
  // Recorded without sample sequences, or for another selection: it runs
  // unsampled and is reported as skipped.
  auto plain = Execution(commands);
  EXPECT_EQ(
      iree_hal_amd_gpu_profile_counter_claim(profile_, &plain, &no_signature),
      nullptr);
  auto stale = Execution(commands);
  EXPECT_EQ(iree_hal_amd_gpu_profile_counter_claim(profile_, &stale,
                                                   &other_signature),
            nullptr);

  auto operation = Execution(commands);
  ASSERT_NE(
      iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, &signature),
      nullptr);
  EXPECT_EQ(operation.counter_grant, 3u);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  // Dispatches 0, 2 and 3 match; each sample is its own dispatch's.
  ASSERT_EQ(sink_.dispatches.size(), 3u);
  ASSERT_EQ(sink_.counter_samples.size(), 3u);
  const std::array<uint32_t, 3> indices = {0, 2, 3};
  for (size_t i = 0; i < 3; ++i) {
    const auto& sample = sink_.counter_samples[i];
    EXPECT_EQ(sample.flags,
              IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT |
                  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION |
                  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE);
    EXPECT_EQ(sample.dispatch_event_id, sink_.dispatches[i].event_id);
    EXPECT_EQ(sample.command_buffer_id, 51u);
    EXPECT_EQ(sample.command_index, indices[i]);
    EXPECT_EQ(sample.function_ordinal, 1u);
    EXPECT_EQ(sample.start_tick, 100u * indices[i]);
    EXPECT_EQ(sink_.counter_values[i][0], indices[i] + 1);
    EXPECT_EQ(sink_.counter_values[i][6], 7u * (indices[i] + 1));
  }
  EXPECT_EQ(sink_.counter_captures.back().sampled_operation_count, 1u);
  EXPECT_EQ(sink_.counter_captures.back().sampled_dispatch_count, 3u);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 2u);

  // The bound is spent: a later execution is not sampled and not skipped.
  auto later = Execution(commands);
  EXPECT_EQ(
      iree_hal_amd_gpu_profile_counter_claim(profile_, &later, &signature),
      nullptr);
  iree_hal_amd_gpu_profile_commands_release(commands);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_TRUE(sink_.counter_captures.back().flags &
              IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_LIMIT_REACHED);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 2u);
}

TEST_F(GpuProfileCounterTest, BoundEmitsTheFirstDispatchesOfAnExecution) {
  counters_.storage.sample_limit = 2;
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  const uint64_t signature = counters_.storage.signature;
  auto* commands = Recording({0, 1, 0});
  auto operation = Execution(commands);
  ASSERT_NE(
      iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, &signature),
      nullptr);
  EXPECT_EQ(operation.counter_grant, 2u);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  iree_hal_amd_gpu_profile_commands_release(commands);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dispatches.size(), 3u);
  ASSERT_EQ(sink_.counter_samples.size(), 2u);
  EXPECT_EQ(sink_.counter_samples[1].command_index, 1u);
}

TEST_F(GpuProfileCounterTest, UnwrittenSamplesAreCountedNotReported) {
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
  // The end sample's copies never ran.
  ASSERT_NO_FATAL_FAILURE(Publish(&operation, false));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  // Another collector's programming was in place when the next one started.
  counters_.state = {1, 0x55, 0x14, 0xffffffff};
  auto next = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &next, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&next));
  ASSERT_NO_FATAL_FAILURE(Complete(&next));
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.dispatches.size(), 2u);
  ASSERT_EQ(sink_.counter_samples.size(), 1u);
  EXPECT_EQ(sink_.counter_samples[0].dispatch_event_id,
            sink_.dispatches[1].event_id);
  const auto& capture = sink_.counter_captures.back();
  EXPECT_EQ(capture.sampled_operation_count, 2u);
  EXPECT_EQ(capture.sampled_dispatch_count, 1u);
  EXPECT_EQ(capture.invalid_sample_count, 1u);
  EXPECT_EQ(capture.foreign_state_count, 1u);
  // The invalid sample did not use the bound.
  auto third = Direct();
  EXPECT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &third, NULL),
            nullptr);
  iree_hal_amd_gpu_profile_counter_cancel(profile_, &third);
}

TEST_F(GpuProfileCounterTest,
       FailedSampleWriteRetriesWithoutRepeatingMetadata) {
  ASSERT_NO_FATAL_FAILURE(BeginCounting());
  auto operation = Direct();
  ASSERT_NE(iree_hal_amd_gpu_profile_counter_claim(profile_, &operation, NULL),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Publish(&operation));
  ASSERT_NO_FATAL_FAILURE(Complete(&operation));
  sink_.fail_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.counter_sets.size(), 1u);
  EXPECT_TRUE(sink_.counter_samples.empty());
  sink_.fail_content_type = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.counter_sets.size(), 1u);
  EXPECT_EQ(sink_.counters.size(), 3u);
  EXPECT_EQ(sink_.counter_samples.size(), 1u);
  EXPECT_EQ(sink_.counter_captures.size(), 1u);
}

}  // namespace
