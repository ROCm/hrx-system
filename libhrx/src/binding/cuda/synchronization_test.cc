// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdlib>

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "driver.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"
#include "runtime.h"

namespace {

class CudaSynchronizationTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  using DeviceSynchronizeFn = cudaError_t (*)();
  using EventQueryFn = CUresult (*)(CUevent);
  using GetLastErrorFn = cudaError_t (*)();
  using PeekAtLastErrorFn = cudaError_t (*)();
  using StreamQueryFn = CUresult (*)(CUstream);

  void SetUp() override {
    const char* library_path = std::getenv("HRX_TEST_LIBCUDA");
    ASSERT_NE(nullptr, library_path);
    library_ = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(nullptr, library_) << dlerror();
    device_synchronize_ = reinterpret_cast<DeviceSynchronizeFn>(
        dlsym(library_, "cudaDeviceSynchronize"));
    ASSERT_NE(nullptr, device_synchronize_) << dlerror();
    event_query_ =
        reinterpret_cast<EventQueryFn>(dlsym(library_, "cuEventQuery"));
    ASSERT_NE(nullptr, event_query_) << dlerror();
    get_last_error_ =
        reinterpret_cast<GetLastErrorFn>(dlsym(library_, "cudaGetLastError"));
    ASSERT_NE(nullptr, get_last_error_) << dlerror();
    peek_at_last_error_ = reinterpret_cast<PeekAtLastErrorFn>(
        dlsym(library_, "cudaPeekAtLastError"));
    ASSERT_NE(nullptr, peek_at_last_error_) << dlerror();
    stream_query_ =
        reinterpret_cast<StreamQueryFn>(dlsym(library_, "cuStreamQuery"));
    ASSERT_NE(nullptr, stream_query_) << dlerror();

    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    hrx_device_t device = hrx_device_group_get_device(group_, 0);
    device_entry_.hrx_device = device;
    device_entry_.hal_device = hrx_device_hal(device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    device_entry_initialized_ = true;
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_,
        {.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO},
        iree_allocator_system(), &context_));

    IREE_ASSERT_OK(iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate_));
    uint64_t gate_value = 1;
    const iree_hal_semaphore_list_t waits = {
        .count = 1,
        .semaphores = &gate_,
        .payload_values = &gate_value,
    };
    IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
        context_->default_stream, waits));
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
        iree_allocator_system(), &event_));
    IREE_ASSERT_OK(iree_hal_streaming_event_record(
        event_, context_->default_stream, /*capture_external_record=*/false));
  }

  void TearDown() override {
    if (gate_ && !gate_released_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, /*frontier=*/nullptr));
    }
    if (event_) {
      IREE_EXPECT_OK(iree_hal_streaming_event_synchronize(event_));
      iree_hal_streaming_event_release(event_);
    }
    iree_hal_semaphore_release(gate_);
    iree_hal_streaming_context_release(context_);
    if (device_entry_initialized_) {
      iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
      iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
      iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    }
    hrx_device_group_release(group_);
    group_ = nullptr;
    if (library_) {
      EXPECT_EQ(0, dlclose(library_));
    }
  }

  void ReleaseGate() {
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, /*frontier=*/nullptr));
    gate_released_ = true;
  }

  // Exact CUDA shared object produced by this build.
  void* library_ = nullptr;
  // Public CUDA runtime device synchronization entry point.
  DeviceSynchronizeFn device_synchronize_ = nullptr;
  // Public CUDA event query entry point resolved from |library_|.
  EventQueryFn event_query_ = nullptr;
  // Public CUDA runtime last-error consumer.
  GetLastErrorFn get_last_error_ = nullptr;
  // Public CUDA runtime last-error query.
  PeekAtLastErrorFn peek_at_last_error_ = nullptr;
  // Public CUDA stream query entry point resolved from |library_|.
  StreamQueryFn stream_query_ = nullptr;
  // True after the registry entry's synchronization and pool are initialized.
  bool device_entry_initialized_ = false;
  // Registry entry backing the real task-device context.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context whose default stream owns the pending test work.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Semaphore keeping the submitted stream work pending.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Event recorded after the gate wait.
  iree_hal_streaming_event_t* event_ = nullptr;
  // True after the test releases |gate_|.
  bool gate_released_ = false;
};

TEST_F(CudaSynchronizationTest, DriverQueriesReportPendingAndCompletedWork) {
  iree_hal_streaming_completion_state_t context_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(
      iree_hal_streaming_context_query(context_, &context_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            context_completion_state);
  EXPECT_EQ(CUDA_ERROR_NOT_READY,
            event_query_(reinterpret_cast<CUevent>(event_)));
  EXPECT_EQ(
      CUDA_ERROR_NOT_READY,
      stream_query_(reinterpret_cast<CUstream>(context_->default_stream)));

  ReleaseGate();
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event_));

  IREE_ASSERT_OK(
      iree_hal_streaming_context_query(context_, &context_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE,
            context_completion_state);
  EXPECT_EQ(CUDA_SUCCESS, event_query_(reinterpret_cast<CUevent>(event_)));
  EXPECT_EQ(
      CUDA_SUCCESS,
      stream_query_(reinterpret_cast<CUstream>(context_->default_stream)));
}

TEST_F(CudaSynchronizationTest, RuntimeDeviceSynchronizationReportsNoContext) {
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, device_synchronize_());
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, peek_at_last_error_());
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, get_last_error_());
  EXPECT_EQ(cudaSuccess, peek_at_last_error_());
}

}  // namespace
