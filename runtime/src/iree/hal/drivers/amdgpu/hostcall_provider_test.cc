// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/hostcall_provider.h"

#include "iree/base/threading/notification.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr iree_host_size_t kSharedMemorySize = 4096;
constexpr iree_host_size_t kSharedMemoryAlignment = 64;

struct TestProviderState;

struct TestProviderContext {
  // Provider state shared by every physical-device context.
  TestProviderState* parent = nullptr;

  // Physical-device facts passed to provider initialization.
  iree_hal_amdgpu_hostcall_provider_device_info_t device_info = {};

  // HAL-owned shared allocation passed to the provider.
  iree_byte_span_t shared_memory = {};

  // Stable device-visible address of |shared_memory|.
  uint64_t device_address = 0;

  // Opaque notification token encoded by device-side protocols.
  uint64_t notification_token = 0;

  // Logical-device error callback retained until provider deinitialization.
  iree_hal_amdgpu_error_callback_t error_callback = {};

  // Number of listener service calls observed by this context.
  iree_atomic_int32_t service_count = IREE_ATOMIC_VAR_INIT(0);

  // True while the listener is executing the service callback.
  iree_atomic_int32_t in_service = IREE_ATOMIC_VAR_INIT(0);

  // Service count awaited by the test thread.
  iree_atomic_int32_t expected_service_count = IREE_ATOMIC_VAR_INIT(0);
};

struct TestProviderState {
  TestProviderState() { iree_notification_initialize(&notification); }

  ~TestProviderState() { iree_notification_deinitialize(&notification); }

  // Cross-thread listener service notification.
  iree_notification_t notification;

  // Number of physical-device requirement queries.
  iree_host_size_t query_count = 0;

  // Number of initialized physical-device contexts.
  iree_host_size_t initialize_count = 0;

  // Number of deinitialized physical-device contexts.
  iree_host_size_t deinitialize_count = 0;

  // True when provider initialization should fail after returning a context.
  bool fail_initialize = false;

  // True when the next service call should fail the logical device.
  iree_atomic_int32_t fail_next_service = IREE_ATOMIC_VAR_INIT(0);

  // True when deinitialize observed a concurrently executing service call.
  iree_atomic_int32_t invalid_deinitialize = IREE_ATOMIC_VAR_INIT(0);

  // Per-physical-device provider contexts.
  TestProviderContext contexts[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
};

static iree_status_t TestProviderQueryRequirements(
    void* user_data,
    const iree_hal_amdgpu_hostcall_provider_device_info_t* device_info,
    iree_hal_amdgpu_hostcall_provider_requirements_t* out_requirements) {
  TestProviderState* state = static_cast<TestProviderState*>(user_data);
  if (device_info->physical_device_ordinal != state->query_count ||
      device_info->compute_unit_count == 0 ||
      device_info->maximum_waves_per_compute_unit == 0 ||
      device_info->wavefront_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test provider received invalid device facts");
  }
  ++state->query_count;
  out_requirements->allocation_size = kSharedMemorySize;
  out_requirements->allocation_alignment = kSharedMemoryAlignment;
  return iree_ok_status();
}

static iree_status_t TestProviderInitialize(
    void* user_data,
    const iree_hal_amdgpu_hostcall_provider_device_info_t* device_info,
    iree_byte_span_t shared_memory, uint64_t device_address,
    uint64_t notification_token,
    iree_hal_amdgpu_error_callback_t error_callback, void** out_context) {
  TestProviderState* state = static_cast<TestProviderState*>(user_data);
  if (device_info->physical_device_ordinal >= IREE_HAL_AMDGPU_MAX_GPU_AGENT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "test provider device ordinal is out of range");
  }
  if (shared_memory.data_length != kSharedMemorySize ||
      (reinterpret_cast<uintptr_t>(shared_memory.data) &
       (kSharedMemoryAlignment - 1)) != 0 ||
      device_address != reinterpret_cast<uintptr_t>(shared_memory.data) ||
      notification_token == 0 || !error_callback.fn) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "test provider received invalid physical-device resources");
  }

  TestProviderContext* context =
      &state->contexts[device_info->physical_device_ordinal];
  context->parent = state;
  context->device_info = *device_info;
  context->shared_memory = shared_memory;
  context->device_address = device_address;
  context->notification_token = notification_token;
  context->error_callback = error_callback;
  memset(shared_memory.data, 0xA5, shared_memory.data_length);
  ++state->initialize_count;
  *out_context = context;
  if (state->fail_initialize) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "test provider initialization failure");
  }
  return iree_ok_status();
}

static void TestProviderService(void* context_ptr) {
  TestProviderContext* context = static_cast<TestProviderContext*>(context_ptr);
  TestProviderState* state = context->parent;
  iree_atomic_store(&context->in_service, 1, iree_memory_order_release);
  if (iree_atomic_exchange(&state->fail_next_service, 0,
                           iree_memory_order_acq_rel)) {
    context->error_callback.fn(
        context->error_callback.user_data,
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "test hostcall provider structural failure"));
  }
  iree_atomic_fetch_add(&context->service_count, 1, iree_memory_order_release);
  iree_atomic_store(&context->in_service, 0, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static void TestProviderDeinitialize(void* context_ptr) {
  TestProviderContext* context = static_cast<TestProviderContext*>(context_ptr);
  TestProviderState* state = context->parent;
  if (iree_atomic_load(&context->in_service, iree_memory_order_acquire)) {
    iree_atomic_store(&state->invalid_deinitialize, 1,
                      iree_memory_order_release);
  }
  ++state->deinitialize_count;
}

static bool TestProviderReachedExpectedServiceCount(void* context_ptr) {
  TestProviderContext* context = static_cast<TestProviderContext*>(context_ptr);
  return iree_atomic_load(&context->service_count, iree_memory_order_acquire) >=
         iree_atomic_load(&context->expected_service_count,
                          iree_memory_order_acquire);
}

static iree_hal_amdgpu_hostcall_provider_extension_t MakeProviderExtension(
    TestProviderState* state) {
  iree_hal_amdgpu_hostcall_provider_extension_t extension = {};
  extension.base.type =
      IREE_HAL_AMDGPU_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_HOSTCALL_PROVIDER;
  extension.provider.user_data = state;
  extension.provider.query_requirements = TestProviderQueryRequirements;
  extension.provider.initialize = TestProviderInitialize;
  extension.provider.service = TestProviderService;
  extension.provider.deinitialize = TestProviderDeinitialize;
  return extension;
}

class LiveDevice {
 public:
  ~LiveDevice() { Reset(); }

  iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                           const iree_hal_amdgpu_topology_t* topology,
                           const iree_hal_device_create_params_t* create_params,
                           iree_async_frontier_tracker_t* frontier_tracker,
                           iree_allocator_t host_allocator) {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), &options, libhsa, topology, create_params,
        host_allocator, &device_));
    return iree_hal_device_group_create_from_device(
        device_, frontier_tracker, host_allocator, &device_group_);
  }

  void Reset() {
    iree_hal_device_release(device_);
    device_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(device_);
  }

 private:
  // Test-owned device reference.
  iree_hal_device_t* device_ = nullptr;

  // Device group assigning production-shaped queue topology.
  iree_hal_device_group_t* device_group_ = nullptr;
};

class HostcallProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostcallProviderTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostcallProviderTest::libhsa_;
iree_hal_amdgpu_topology_t HostcallProviderTest::topology_;

TEST_F(HostcallProviderTest, OmittedProviderCreatesNoResources) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));

  LiveDevice device;
  IREE_ASSERT_OK(
      device.Initialize(&libhsa_, &topology_, create_context.params(),
                        create_context.frontier_tracker(), host_allocator_));
  ASSERT_NE(device.logical_device(), nullptr);
  for (iree_host_size_t i = 0;
       i < device.logical_device()->physical_device_count; ++i) {
    EXPECT_EQ(
        device.logical_device()->physical_devices[i]->hostcall_provider_state,
        nullptr);
  }
}

TEST_F(HostcallProviderTest, UnknownExtensionIsSkipped) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));
  iree_hal_device_create_params_extension_t unknown_extension = {};
  unknown_extension.type = UINT32_MAX;
  iree_hal_device_create_params_t create_params = *create_context.params();
  create_params.next = &unknown_extension;

  LiveDevice device;
  IREE_ASSERT_OK(device.Initialize(&libhsa_, &topology_, &create_params,
                                   create_context.frontier_tracker(),
                                   host_allocator_));
  for (iree_host_size_t i = 0;
       i < device.logical_device()->physical_device_count; ++i) {
    EXPECT_EQ(
        device.logical_device()->physical_devices[i]->hostcall_provider_state,
        nullptr);
  }
}

TEST_F(HostcallProviderTest, InvalidProviderFailsBeforeRequirementQuery) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));
  TestProviderState provider_state;
  iree_hal_amdgpu_hostcall_provider_extension_t extension =
      MakeProviderExtension(&provider_state);
  extension.provider.service = nullptr;
  iree_hal_device_create_params_t create_params = *create_context.params();
  create_params.next = &extension;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdgpu_logical_device_create(
      IREE_SV("amdgpu"), &options, &libhsa_, &topology_, &create_params,
      host_allocator_, &device);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(provider_state.query_count, 0u);
}

TEST_F(HostcallProviderTest, DuplicateProviderFailsBeforeRequirementQuery) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));
  TestProviderState provider_state;
  iree_hal_amdgpu_hostcall_provider_extension_t first_extension =
      MakeProviderExtension(&provider_state);
  iree_hal_amdgpu_hostcall_provider_extension_t second_extension =
      MakeProviderExtension(&provider_state);
  first_extension.base.next = &second_extension;
  iree_hal_device_create_params_t create_params = *create_context.params();
  create_params.next = &first_extension;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdgpu_logical_device_create(
      IREE_SV("amdgpu"), &options, &libhsa_, &topology_, &create_params,
      host_allocator_, &device);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(provider_state.query_count, 0u);
}

TEST_F(HostcallProviderTest, InitializationFailureUnwindsProviderContext) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));
  TestProviderState provider_state;
  provider_state.fail_initialize = true;
  iree_hal_amdgpu_hostcall_provider_extension_t extension =
      MakeProviderExtension(&provider_state);
  iree_hal_device_create_params_t create_params = *create_context.params();
  create_params.next = &extension;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdgpu_logical_device_create(
      IREE_SV("amdgpu"), &options, &libhsa_, &topology_, &create_params,
      host_allocator_, &device);
  EXPECT_TRUE(iree_status_is_data_loss(status));
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(provider_state.query_count, 1u);
  EXPECT_EQ(provider_state.initialize_count, 1u);
  EXPECT_EQ(provider_state.deinitialize_count, 1u);
}

TEST_F(HostcallProviderTest, EagerPhysicalLifecycleAndFailurePublication) {
  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));
  TestProviderState provider_state;
  iree_hal_amdgpu_hostcall_provider_extension_t extension =
      MakeProviderExtension(&provider_state);
  iree_hal_device_create_params_t create_params = *create_context.params();
  create_params.next = &extension;

  LiveDevice device;
  IREE_ASSERT_OK(device.Initialize(&libhsa_, &topology_, &create_params,
                                   create_context.frontier_tracker(),
                                   host_allocator_));
  iree_hal_amdgpu_logical_device_t* logical_device = device.logical_device();
  ASSERT_NE(logical_device, nullptr);
  EXPECT_EQ(provider_state.query_count, topology_.gpu_agent_count);
  EXPECT_EQ(provider_state.initialize_count, topology_.gpu_agent_count);
  EXPECT_EQ(provider_state.deinitialize_count, 0u);

  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_hostcall_provider_state_t* hostcall_state =
        logical_device->physical_devices[i]->hostcall_provider_state;
    ASSERT_NE(hostcall_state, nullptr);
    TestProviderContext* provider_context = &provider_state.contexts[i];
    EXPECT_EQ(hostcall_state->provider_context, provider_context);
    EXPECT_EQ(hostcall_state->device_address, provider_context->device_address);
    EXPECT_EQ(hostcall_state->notification_signal.handle,
              provider_context->notification_token);
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    const iree_host_size_t host_queue_count =
        iree_hal_amdgpu_physical_device_host_queue_count(physical_device);
    for (iree_host_size_t j = 0; j < host_queue_count; ++j) {
      EXPECT_EQ(physical_device->host_queues[j].hostcall_buffer,
                reinterpret_cast<void*>(hostcall_state->device_address));
    }
    iree_atomic_store(&provider_context->expected_service_count, 1,
                      iree_memory_order_release);
    iree_hsa_signal_add_screlease(IREE_LIBHSA(&libhsa_),
                                  hostcall_state->notification_signal, 1);
  }

  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    EXPECT_TRUE(iree_notification_await(
        &provider_state.notification, TestProviderReachedExpectedServiceCount,
        &provider_state.contexts[i], iree_infinite_timeout()));
  }

  iree_hal_amdgpu_hostcall_provider_state_t* first_hostcall_state =
      logical_device->physical_devices[0]->hostcall_provider_state;
  iree_atomic_store(&provider_state.fail_next_service, 1,
                    iree_memory_order_release);
  iree_atomic_store(&provider_state.contexts[0].expected_service_count, 2,
                    iree_memory_order_release);
  iree_hsa_signal_add_screlease(IREE_LIBHSA(&libhsa_),
                                first_hostcall_state->notification_signal, 1);
  EXPECT_TRUE(iree_notification_await(
      &provider_state.notification, TestProviderReachedExpectedServiceCount,
      &provider_state.contexts[0], iree_infinite_timeout()));
  EXPECT_NE(iree_atomic_load(&logical_device->failure_status,
                             iree_memory_order_acquire),
            0);

  device.Reset();
  EXPECT_EQ(provider_state.deinitialize_count, topology_.gpu_agent_count);
  EXPECT_EQ(iree_atomic_load(&provider_state.invalid_deinitialize,
                             iree_memory_order_acquire),
            0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
