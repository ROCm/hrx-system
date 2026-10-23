// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL queue_dispatch operations.

#include <algorithm>
#include <array>
#include <cstdint>
#include <future>
#include <numeric>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class QueueDispatchTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no provisioned dispatch-capable queue";
    }
    LoadExecutableOrSkipUnsupported(
        "command_buffer_dispatch_constants_bindings_test.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  void RunDeferredFileGather(iree_hal_queue_t* io_queue);

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

static void MakeScaleAndOffsetBindings(iree_hal_buffer_t* input_buffer,
                                       iree_hal_buffer_t* output_buffer,
                                       iree_hal_buffer_ref_t binding_refs[2]) {
  binding_refs[0] = iree_hal_make_buffer_ref(
      input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
  binding_refs[1] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
}

// The callback owns the source and destination until its reads have captured
// them. Completing the callback does not complete those reads or their
// consumer.
struct DeferredFileGather : iree_hal_resource_t {
  // Borrowed fixture device, live through the final consumer completion.
  iree_hal_device_t* device;
  // Borrowed download destination, live through the final consumer completion.
  const uint32_t* rows;
  // Source file retained until each accepted read owns its own reference.
  Ref<iree_hal_file_t> file;
  // Destination retained independently by the callback, reads, and consumer.
  Ref<iree_hal_buffer_t> target;
  // External notification joined before the test destroys its borrowed state.
  std::promise<void>* released;

  DeferredFileGather(iree_hal_device_t* device, const uint32_t* rows,
                     iree_hal_file_t* file, iree_hal_buffer_t* target,
                     std::promise<void>* released)
      : device(device),
        rows(rows),
        file(file),
        target(target),
        released(released) {
    iree_hal_file_retain(file);
    iree_hal_buffer_retain(target);
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            [](iree_hal_resource_t* resource) {
              auto* state = static_cast<DeferredFileGather*>(resource);
              auto* released = state->released;
              delete state;
              released->set_value();
            },
    };
    iree_hal_resource_initialize(&vtable, this);
  }

  static iree_status_t Invoke(void* user_data, const uint64_t[4],
                              iree_hal_host_call_context_t* context) {
    auto* state = static_cast<DeferredFileGather*>(user_data);
    std::array<iree_hal_semaphore_t*, 4> completions = {};
    uint64_t values[] = {1, 1, 1, 1};
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < completions.size() && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_semaphore_create(
          state->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &completions[i]);
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_read(
            context->queue, iree_hal_semaphore_list_empty(),
            {1, &completions[i], &values[i]}, state->file,
            state->rows[i] * sizeof(uint32_t), state->target,
            i * sizeof(uint32_t), sizeof(uint32_t), /*barriers=*/nullptr,
            IREE_HAL_READ_FLAG_NONE);
      }
    }
    if (iree_status_is_ok(status)) {
      // Queue capture retains the signal list beyond the callback's return.
      status = iree_hal_queue_transfer(
          context->queue, {completions.size(), completions.data(), values},
          context->signal_semaphore_list, 0, nullptr, /*barriers=*/nullptr);
    }
    for (auto* completion : completions) {
      iree_hal_semaphore_release(completion);
    }
    return iree_status_is_ok(status)
               ? iree_status_from_code(IREE_STATUS_DEFERRED)
               : status;
  }
};

void QueueDispatchTest::RunDeferredFileGather(iree_hal_queue_t* io_queue) {
  iree::testing::TempFilePath path("iree_hal_cts_deferred_gather");
  std::array<uint32_t, 32> file_contents;
  for (size_t i = 0; i < file_contents.size(); ++i) {
    file_contents[i] = 1009u + 17u * i;
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(file_contents.data(), sizeof(file_contents)),
      iree_allocator_system()));
  struct FileOwner {
    // Owns the native handle until the imported file's final release.
    iree_io_file_handle_t* handle = nullptr;
    // Joined before this owner or the temporary path is destroyed.
    std::promise<void> closed;
  } owner;
  auto file_closed_future = owner.closed.get_future();
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), iree_allocator_system(), &owner.handle));
  iree_io_file_handle_t* handle = nullptr;
  iree_status_t wrap_status = iree_io_file_handle_wrap(
      IREE_IO_FILE_ACCESS_READ, IREE_IO_FILE_MODE_ASYNC,
      iree_io_file_handle_primitive(owner.handle),
      {[](void* user_data, iree_io_file_handle_primitive_t) {
         auto* owner = static_cast<FileOwner*>(user_data);
         iree_io_file_handle_release(owner->handle);
         owner->closed.set_value();
       },
       &owner},
      iree_allocator_system(), &handle);
  if (!iree_status_is_ok(wrap_status)) {
    iree_io_file_handle_release(owner.handle);
    EXPECT_TRUE(path.Remove());
  }
  IREE_ASSERT_OK(wrap_status);
  Ref<iree_hal_file_t> file;
  iree_status_t import_status = iree_hal_file_import(
      device_, /*storage_pool=*/nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ, handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
      file.out());
  iree_io_file_handle_release(handle);
  if (!iree_status_is_ok(import_status)) {
    file_closed_future.wait();
    EXPECT_TRUE(path.Remove());
  }
  IREE_ASSERT_OK(import_status);
  if (!iree_hal_file_async_handle(file)) {
    file.reset();
    file_closed_future.wait();
    EXPECT_TRUE(path.Remove());
    GTEST_SKIP() << "device does not import native files for asynchronous I/O";
  }
  ASSERT_EQ(iree_hal_file_storage_buffer(file), nullptr);

  const uint32_t seeds[] = {1, 2, 3, 4};
  Ref<iree_hal_buffer_t> input, addresses, gathered, output;
  IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(sizeof(seeds), input.out()));
  IREE_ASSERT_OK(
      CreateUninitializedDeviceBuffer(sizeof(seeds), addresses.out()));
  IREE_ASSERT_OK(
      CreateUninitializedDeviceBuffer(sizeof(seeds), gathered.out()));
  IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(sizeof(seeds), output.out()));

  // Gate the producer so that the callback, consumer, and final readback are
  // submitted before any GPU-generated address is available to the callback.
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList uploaded(device_, {0}, {1});
  SemaphoreList produced(device_, {0}, {1});
  SemaphoreList downloaded(device_, {0}, {1});
  SemaphoreList gathered_signal(device_, {0}, {1});
  SemaphoreList consumed(device_, {0}, {1});
  SemaphoreList finished(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_update(
      io_queue, gate, uploaded, seeds, 0, input, 0, sizeof(seeds),
      /*barriers=*/nullptr, IREE_HAL_UPDATE_FLAG_NONE));

  iree_hal_buffer_ref_t producer_bindings[2];
  MakeScaleAndOffsetBindings(input, addresses, producer_bindings);
  const uint32_t producer_constants[] = {3, 10};
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, uploaded, produced, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(producer_constants, sizeof(producer_constants)),
      {IREE_ARRAYSIZE(producer_bindings), producer_bindings},
      /*barriers=*/nullptr, IREE_HAL_DISPATCH_FLAG_NONE));

  uint32_t rows[4] = {};
  IREE_ASSERT_OK(iree_hal_queue_download(io_queue, produced, downloaded,
                                         addresses, 0, rows, sizeof(rows),
                                         /*barriers=*/nullptr));
  std::promise<void> callback_released;
  auto callback_released_future = callback_released.get_future();
  auto* callback =
      new DeferredFileGather(device_, rows, file, gathered, &callback_released);
  const uint64_t args[4] = {};
  iree_status_t call_status = iree_hal_queue_host_call(
      io_queue, downloaded, gathered_signal,
      iree_hal_make_host_call_with_resource(DeferredFileGather::Invoke,
                                            callback, callback),
      args, IREE_HAL_HOST_CALL_FLAG_NONE);
  iree_hal_resource_release(callback);
  IREE_ASSERT_OK(call_status);
  file.reset();

  iree_hal_buffer_ref_t consumer_bindings[2];
  MakeScaleAndOffsetBindings(gathered, output, consumer_bindings);
  const uint32_t consumer_constants[] = {5, 7};
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, gathered_signal, consumed, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(consumer_constants, sizeof(consumer_constants)),
      {IREE_ARRAYSIZE(consumer_bindings), consumer_bindings},
      /*barriers=*/nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
  gathered.reset();
  std::array<uint32_t, 4> result = {};
  IREE_ASSERT_OK(iree_hal_queue_download(io_queue, consumed, finished, output,
                                         0, result.data(), sizeof(result),
                                         /*barriers=*/nullptr));

  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(finished, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  callback_released_future.wait();
  file_closed_future.wait();
  EXPECT_THAT(result,
              ContainerEq(std::array<uint32_t, 4>{6157, 6412, 6667, 6922}));
  EXPECT_TRUE(path.Remove());
}

TEST_P(QueueDispatchTest, DeferredFileGatherWithQueuedConsumerOnAcquiredQueue) {
  ASSERT_NE(transfer_queue_, nullptr);
  const auto* family = iree_hal_queue_family(transfer_queue_);
  const auto* spec = iree_hal_queue_family_spec(family);
  if (!iree_all_bits_set(spec->flags,
                         IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
    GTEST_SKIP()
        << "transfer queue family does not support dynamic acquisition";
  }
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  Ref<iree_hal_queue_t> io_queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, io_queue.out()));
  ASSERT_NE(io_queue.get(), dispatch_queue_);
  RunDeferredFileGather(io_queue);
}

TEST_P(QueueDispatchTest, DeferredFileGatherWithQueuedConsumerOnSameQueue) {
  const auto* spec =
      iree_hal_queue_family_spec(iree_hal_queue_family(dispatch_queue_));
  if (!iree_all_bits_set(spec->role_flags,
                         IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
    GTEST_SKIP() << "dispatch queue does not support transfers";
  }
  RunDeferredFileGather(dispatch_queue_);
}

// Dispatches scale_and_offset directly on the queue:
// output[i] = input[i] * scale + offset.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindings) {
  const uint32_t input_data[] = {1, 2, 3, 4};
  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(
      CreateUninitializedDeviceBuffer(sizeof(input_data), input_buffer.out()));
  SemaphoreList upload_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_update(
      transfer_queue_, iree_hal_semaphore_list_empty(), upload_signal,
      input_data, 0, input_buffer, 0, sizeof(input_data), /*barriers=*/NULL,
      IREE_HAL_UPDATE_FLAG_NONE));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateUninitializedDeviceBuffer(sizeof(input_data), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, upload_signal, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

TEST_P(QueueDispatchTest,
       DynamicallyAcquiredQueuesDispatchAtAdvertisedPriorities) {
  const iree_hal_queue_family_t* queue_family =
      iree_hal_queue_family(dispatch_queue_);
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  if (!iree_any_bit_set(family_spec->flags,
                        IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
    GTEST_SKIP()
        << "dispatch queue family does not support dynamic acquisition";
  }

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {2, 4, 6, 8};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  const iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };
  const uint32_t constant_data[] = {5, 3};
  const iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  for (iree_host_size_t i = 0; i < family_spec->priority_count; ++i) {
    const iree_hal_queue_priority_t priority = family_spec->priorities[i];
    SCOPED_TRACE(priority);
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    params.priority = priority;
    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, queue.out()));

    SemaphoreList signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, iree_hal_semaphore_list_empty(), signal, executable_,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

    std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
    EXPECT_THAT(output_data,
                ContainerEq(std::vector<uint32_t>{13, 23, 33, 43}));
  }
}

// Borrowed resource lifetimes are an optimization hint for callers that keep
// resources live until dispatch completion. The observable dispatch behavior is
// identical to the default retained mode.
TEST_P(QueueDispatchTest, DispatchWithBorrowedResourceLifetimes) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

// Profiling must not perturb direct queue dispatch semantics or completion.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindingsWhileProfiling) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(iree_status_code(profiling_status))) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_GE(sink.device_metadata_count, 1);
  EXPECT_GE(sink.queue_metadata_count, 1);
  EXPECT_GE(sink.dispatch_event_count, 1);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);
  ExpectDispatchEventsHaveClockMetadata(sink);
}

// HAL-native CPU profiling should produce host queue and execution records for
// the direct dispatch submission.
TEST_P(QueueDispatchTest, DispatchHostQueueEventProfiling) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(iree_status_code(profiling_status))) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "host queue profiling unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_GE(sink.host_execution_event_count, 1);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_EQ(0, sink.queue_device_event_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);

  auto queue_event_it = std::find_if(
      sink.queue_events.begin(), sink.queue_events.end(),
      [](const iree_hal_profile_queue_event_t& event) {
        return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      });
  ASSERT_NE(sink.queue_events.end(), queue_event_it);
  EXPECT_EQ(1u, queue_event_it->operation_count);

  auto host_event_it = std::find_if(
      sink.host_execution_events.begin(), sink.host_execution_events.end(),
      [](const iree_hal_profile_host_execution_event_t& event) {
        return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      });
  ASSERT_NE(sink.host_execution_events.end(), host_event_it);
  EXPECT_EQ(IREE_STATUS_OK, host_event_it->status_code);
  EXPECT_EQ(0u, host_event_it->function_ordinal);
  EXPECT_EQ(queue_event_it->submission_id, host_event_it->submission_id);
  EXPECT_GE(host_event_it->end_host_time_ns, host_event_it->start_host_time_ns);
}

// Device queue event profiling should produce one device-domain span for the
// direct dispatch submission without also requiring dispatch-event capture.
TEST_P(QueueDispatchTest, DispatchDeviceQueueEventProfiling) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(iree_status_code(profiling_status))) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device queue profiling unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.clock_correlation_count, 3);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_TRUE(sink.dispatch_events.empty());
  EXPECT_GE(sink.queue_device_event_count, 1);
  auto dispatch_event_it = std::find_if(
      sink.queue_device_events.begin(), sink.queue_device_events.end(),
      [](const iree_hal_profile_queue_device_event_t& event) {
        return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      });
  ASSERT_NE(sink.queue_device_events.end(), dispatch_event_it);
  EXPECT_EQ(1u, dispatch_event_it->operation_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);
  ExpectQueueDeviceEventsWithinClockCorrelationRange(sink);
}

// Capture filters must not perturb direct queue dispatch semantics or
// completion when no dispatch event matches.
TEST_P(QueueDispatchTest, DispatchProfileFilterCanSkipDirectDispatchEvents) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = TestProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  profiling_options.capture_filter.executable_function_pattern =
      IREE_SV("iree-hal-cts-never-matches-*");
  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status = profiling.Begin(&profiling_options);
  if (IsProfilingUnsupported(iree_status_code(profiling_status))) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_TRUE(sink.dispatch_events.empty());
  EXPECT_FALSE(sink.write_after_end);
}

// Zero-workgroup dispatches are no-ops that still participate in semaphore
// ordering and signal completion.
TEST_P(QueueDispatchTest, NoopDispatchSignalsAndDoesNotTouchBuffers) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                                          output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Deferred zero-workgroup dispatches still wait for their dependency before
// signaling completion, but never execute the kernel body.
TEST_P(QueueDispatchTest, DeferredNoopDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                                          output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, dispatch_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Wait-before-signal dispatches must be queued without head-of-line blocking
// and replayed after the wait semaphore advances.
TEST_P(QueueDispatchTest, DeferredWaitBeforeSignalDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, dispatch_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchTest);

class QueueDispatchIndirectParametersTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kDispatchedWorkgroupCount = 4;
  static constexpr iree_host_size_t kOutputElementCount = 32;
  static constexpr iree_device_size_t kOutputByteLength =
      kOutputElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kParameterByteLength =
      3 * sizeof(uint32_t);
  static constexpr uint32_t kSentinelValue = 0xCDCDCDCDu;

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no dispatch-capable queue";
    }
    LoadExecutableOrSkipUnsupported(
        "command_buffer_dispatch_multi_workgroup_test.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_status_t CreateIndirectParameterBuffer(iree_hal_buffer_t** out_buffer) {
    return CreateUninitializedDeviceBuffer(kParameterByteLength, out_buffer);
  }

  void ReloadExecutableFromCallScopedStorage() {
    const iree_const_byte_span_t source_data = executable_data(
        IREE_SV("command_buffer_dispatch_multi_workgroup_test.bin"));
    ASSERT_FALSE(iree_const_byte_span_is_empty(source_data));
    std::vector<uint8_t> storage(source_data.data,
                                 source_data.data + source_data.data_length);

    iree_hal_executable_target_selection_result_t target_result;
    IREE_ASSERT_OK(SelectExecutableTarget(
        iree_hal_queue_family(dispatch_queue_), &target_result));
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              target_result.outcome);

    iree_hal_executable_t* replacement_executable = nullptr;
    IREE_ASSERT_OK(LoadExecutable(
        iree_hal_queue_family(dispatch_queue_), target_result.target,
        IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
        iree_make_const_byte_span(storage.data(), storage.size()),
        &replacement_executable));
    ASSERT_NE(replacement_executable, nullptr);

    // Loading before releasing the prior instance covers implementations that
    // share loader state or code mappings between identical executables.
    iree_hal_executable_release(executable_);
    executable_ = replacement_executable;

    // Load inputs are call-scoped. Destroy their contents before the
    // executable is first used so retained borrowed pointers fail loudly.
    std::fill(storage.begin(), storage.end(), UINT8_C(0xA5));
  }

  void RunIndirectQueueDispatch(
      iree_hal_dispatch_flags_t flags,
      iree_device_size_t parameter_ref_length = kParameterByteLength) {
    Ref<iree_hal_buffer_t> output_buffer;
    IREE_ASSERT_OK(CreateFilledDeviceBuffer(kOutputByteLength, kSentinelValue,
                                            output_buffer.out()));

    Ref<iree_hal_buffer_t> parameter_buffer;
    IREE_ASSERT_OK(CreateIndirectParameterBuffer(parameter_buffer.out()));

    const uint32_t parameter_data[3] = {
        kDispatchedWorkgroupCount,
        1,
        1,
    };
    SemaphoreList update_signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_queue_update(
        transfer_queue_, empty_wait, update_signal, parameter_data,
        /*source_offset=*/0, parameter_buffer, /*target_offset=*/0,
        sizeof(parameter_data), /*barriers=*/NULL, IREE_HAL_UPDATE_FLAG_NONE));

    iree_hal_buffer_ref_t binding_refs[1] = {
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 kOutputByteLength),
    };
    iree_hal_buffer_ref_list_t bindings = {
        .count = IREE_ARRAYSIZE(binding_refs),
        .values = binding_refs,
    };

    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        /*workgroup_count_x=*/kOutputElementCount, /*workgroup_count_y=*/1,
        /*workgroup_count_z=*/1);
    config.workgroup_count_ref = iree_hal_make_buffer_ref(
        parameter_buffer, /*offset=*/0, parameter_ref_length);

    SemaphoreList dispatch_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        dispatch_queue_, update_signal, dispatch_signal, executable_,
        iree_hal_executable_function_from_index(0), config,
        iree_const_byte_span_empty(), bindings, /*barriers=*/NULL, flags));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
    std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
    std::iota(expected.begin(), expected.begin() + kDispatchedWorkgroupCount,
              0u);
    EXPECT_THAT(output_data, ContainerEq(expected));
  }

  void RunIndirectQueueDispatchWhileProfiling(iree_hal_dispatch_flags_t flags) {
    TestProfileSink sink = {};
    TestProfileSinkInitialize(&sink);
    sink.expected_dispatch_flags =
        IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
    sink.validate_dispatch_workgroup_count = false;

    DeviceProfilingScope profiling(device_);
    iree_status_t profiling_status =
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                            IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                        TestProfileSinkAsBase(&sink));
    if (IsProfilingUnsupported(iree_status_code(profiling_status))) {
      iree_status_free(profiling_status);
      GTEST_SKIP() << "device profiling data family unsupported by backend";
    }
    IREE_ASSERT_OK(profiling_status);

    RunIndirectQueueDispatch(flags);

    IREE_ASSERT_OK(profiling.End());
    EXPECT_EQ(1, sink.begin_count);
    EXPECT_EQ(1, sink.end_count);
    EXPECT_GE(sink.device_metadata_count, 1);
    EXPECT_GE(sink.queue_metadata_count, 1);
    EXPECT_GE(sink.clock_correlation_count, 2);
    EXPECT_GE(sink.dispatch_event_count, 1);
    EXPECT_GE(sink.queue_device_event_count, 1);
    const iree_host_size_t dispatch_queue_device_event_count = std::count_if(
        sink.queue_device_events.begin(), sink.queue_device_events.end(),
        [](const iree_hal_profile_queue_device_event_t& event) {
          return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
        });
    EXPECT_GE(dispatch_queue_device_event_count, 1u);
    EXPECT_TRUE(sink.saw_device_metadata);
    EXPECT_TRUE(sink.saw_queue_metadata);
    EXPECT_FALSE(sink.write_after_end);
    ExpectDispatchEventsHaveClockMetadata(sink);
    ExpectQueueDeviceEventsWithinClockCorrelationRange(sink);
  }

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(QueueDispatchIndirectParametersTest, StaticParameters) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, DynamicParameters) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest,
       DynamicParametersWithCallScopedExecutableData) {
  for (int i = 0; i < 2; ++i) {
    ReloadExecutableFromCallScopedStorage();
    ASSERT_NE(executable_, nullptr);
    RunIndirectQueueDispatch(
        IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
  }
}

TEST_P(QueueDispatchIndirectParametersTest, StaticParametersWhileProfiling) {
  RunIndirectQueueDispatchWhileProfiling(
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, DynamicParametersWhileProfiling) {
  RunIndirectQueueDispatchWhileProfiling(
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, WholeBufferParameterRef) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                           IREE_HAL_WHOLE_BUFFER);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchIndirectParametersTest);

}  // namespace iree::hal::cts
