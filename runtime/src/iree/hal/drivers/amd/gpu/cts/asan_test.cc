// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <semaphore>
#include <string>
#include <tuple>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/cts/testdata.h"
#include "iree/hal/drivers/amd/gpu/image/asan_images.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

class GpuAsanTest : public GpuNativeTest,
                    public ::testing::WithParamInterface<
                        std::tuple<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return std::get<0>(GetParam());
  }

  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_, UINT64_MAX,
                                             AMDF_NATIVE_LIFETIME_PROCESS));
    memory_gate_.RequireCodeUpload();
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    if (!(context_->device_info.features &
          AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS)) {
      GTEST_SKIP() << "Native shader notification is unavailable";
    }
    options_.asan.memory.shadow_slab_size = 1024 * 1024;
    options_.asan.shadow_backing =
        std::get<1>(GetParam())
            ? IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_HOST_LOCAL
            : IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL;
    native_domain_ = context_;
    Create(IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN);
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_READ,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    const iree_hal_pool_import_class_t import_class = {
        .storage_class = {.type = IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER},
    };
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    pool_options.imports = {1, &import_class};
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                             iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t program_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        &slot_));
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    IREE_ASSERT_OK(LookupCodeObject(iree_hal_amd_gpu_asan_images_create(),
                                    iree_hal_amd_gpu_asan_images_size(),
                                    target->target_key, &load.executable_data));
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executable_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("asan_read_shadow"), &function_));
    CreateSemaphore(&done_);
    Allocate(64, &output_);
  }

  void TearDown() override {
    iree_hal_buffer_release(imported_);
    iree_hal_buffer_release(target_);
    iree_hal_buffer_release(output_);
    iree_hal_semaphore_release(done_);
    iree_hal_executable_release(executable_);
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void Allocate(uint64_t length, iree_hal_buffer_t** out_buffer) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(length, 8),
        iree_infinite_timeout(), out_buffer));
  }

  uint64_t Address(iree_hal_buffer_t* buffer) {
    return iree_hal_buffer_native_binding(buffer, slot_).device_address;
  }

  void ExpectShadow(uint64_t address, std::initializer_list<uint8_t> expected) {
    struct {
      // Application address whose shadow bytes the shader reads.
      uint64_t address;
      // Number of consecutive shadow bytes to return.
      uint32_t count;
    } constants = {address, static_cast<uint32_t>(expected.size())};
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const auto binding = iree_hal_make_buffer_ref(output_, 0, 64);
    ++value_;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {}, {1, &done_, &value_}, executable_, function_, config,
        iree_make_const_byte_span(
            &constants, sizeof(constants.address) + sizeof(constants.count)),
        {1, &binding}, nullptr, 0));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        done_, value_, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    std::vector<uint8_t> actual(expected.size());
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(output_, 0, actual.data(), actual.size()));
    EXPECT_EQ(actual, std::vector<uint8_t>(expected));
  }

  iree_hal_device_asan_observation_t Observe() {
    iree_hal_device_observation_t observation;
    IREE_CHECK_OK(iree_hal_device_sample_observation(
        device_, IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER, &observation));
    EXPECT_NE(
        observation.provided_flags & IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER,
        0u);
    return observation.sanitizer.asan;
  }

  // Force configuration patching through staging before real shader execution.
  NativeMemoryGate memory_gate_;
  // Dependency domain borrowed through the application's device ownership.
  const void* native_domain_ = nullptr;
  // Explicit source shared by owned and imported buffers.
  iree_hal_pool_t* pool_ = nullptr;
  // Shader-readable device address slot fixed during pool construction.
  iree_hal_buffer_native_binding_slot_t slot_ = {};
  // Real compiler artifact whose globals receive the production configuration.
  iree_hal_executable_t* executable_ = nullptr;
  // Uninstrumented observer of the device's mapped shadow bytes.
  iree_hal_executable_function_t function_ = {};
  // Host-readable output retained through every dispatch.
  iree_hal_buffer_t* output_ = nullptr;
  // Owned allocation under test.
  iree_hal_buffer_t* target_ = nullptr;
  // Attachment under test; external storage remains owned by this fixture.
  iree_hal_buffer_t* imported_ = nullptr;
  // Publishes completion of native detachment before observing quarantine.
  std::binary_semaphore import_released_{0};
  // Explicit completion timeline for shadow observation.
  iree_hal_semaphore_t* done_ = nullptr;
  // Last submitted timeline value.
  uint64_t value_ = 0;
  // Three pages isolate the authorized middle registration envelope.
  alignas(4096) std::array<uint8_t, 3 * 4096> external_ = {};
};

TEST_P(GpuAsanTest, OwnedSlabReleasePoisonsAndQuarantinesBeforeFinalCleanup) {
  iree_hal_pool_stats_t before;
  iree_hal_pool_query_stats(pool_, &before);
  Allocate(19, &target_);
  ASSERT_FALSE(HasFatalFailure());
  const uint64_t address = Address(target_);
  ExpectShadow(address - 8, {0xfa, 0, 0, 3, 0xfa});
  iree_hal_buffer_release(target_);
  target_ = nullptr;
  iree_hal_pool_trim(pool_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_TRUE(
      iree::hal::cts::WaitPoolCommittedBytes(pool_, before.bytes_committed));
  ExpectShadow(address, {0xfa, 0xfa, 0xfa, 0xfa});
  EXPECT_GE(Observe().quarantine_size, 19u + 2u * 256u);
}

TEST_P(GpuAsanTest, QueuedReleaseWaitsForHostSanitizerBoundary) {
  using iree::hal::cts::Ref;
  ASSERT_TRUE(iree_hal_pool_requires_asan_advice(pool_));
  Ref<iree_hal_semaphore_t> allocated, producer, released, paused;
  CreateSemaphore(allocated.out());
  CreateSemaphore(producer.out());
  CreateSemaphore(released.out());
  CreateSemaphore(paused.out());
  iree_hal_semaphore_t* allocated_handle = allocated;
  iree_hal_semaphore_t* producer_handle = producer;
  iree_hal_semaphore_t* released_handle = released;
  iree_hal_semaphore_t* paused_handle = paused;
  const auto request = iree_hal_make_pool_reservation_request(19, 8);
  uint64_t one = 1;
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                       {1, &allocated_handle, &one}, pool_, 1,
                                       &request, buffer.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      allocated, one, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  const uint64_t address = Address(buffer);
  ExpectShadow(address, {0, 0, 3});
  iree_hal_pool_stats_t before;
  iree_hal_pool_query_stats(pool_, &before);

  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(paused_handle)};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {},
                                        {1, &paused_handle, &one}, nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[1], {}, {1, &producer_handle, &one}, nullptr, 0));
  iree_hal_amd_gpu_semaphore_wait_t dependency = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      producer, one, native_domain_, &dependency));
  EXPECT_EQ(dependency.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  iree_hal_amd_gpu_signal_release(dependency.signal);

  iree_hal_buffer_t* handle = buffer;
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[1], {1, &producer_handle, &one},
                              {1, &released_handle, &one}, 1, &handle));
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      released, one, native_domain_, &dependency));
  EXPECT_EQ(dependency.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED);
  iree_hal_amd_gpu_signal_release(dependency.signal);
  iree_hal_pool_stats_t pending;
  iree_hal_pool_query_stats(pool_, &pending);
  EXPECT_EQ(pending.reservation_count, before.reservation_count);
  EXPECT_EQ(Address(buffer), address);

  pause.Join();
  IREE_ASSERT_OK(iree_hal_semaphore_wait(released, one, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(Address(buffer), 0u);
  ExpectShadow(address, {0xfa, 0xfa, 0xfa});
}

TEST_P(GpuAsanTest, HostImportEnablesRegisteredEnvelopeAndBypassesQuarantine) {
  iree_hal_external_storage_t storage = {
      .storage_class = {.type = IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER},
      .payload = {.host_pointer = external_.data()},
      .byte_length = external_.size(),
      .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
      .maintenance_range = {4096, 4096},
      .release =
          [](void* user_data, iree_hal_external_storage_class_t,
             iree_hal_external_storage_payload_t) {
            static_cast<std::binary_semaphore*>(user_data)->release();
          },
      .release_user_data = &import_released_,
  };
  IREE_ASSERT_OK(
      iree_hal_pool_import_buffer(pool_, &storage, {4099, 13}, &imported_));
  const uint64_t address = Address(imported_) - 3;
  // Registration enables the complete page, beyond the logical buffer range.
  // Sparse shadow outside that page may be unmapped instead of poisoned.
  ExpectShadow(address, {0, 0});
  ExpectShadow(address + 4096 - 16, {0, 0});
  const auto before = Observe();
  iree_hal_buffer_release(imported_);
  imported_ = nullptr;
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  import_released_.acquire();
  EXPECT_EQ(Observe().quarantine_size, before.quarantine_size);
}

INSTANTIATE_TEST_SUITE_P(
    ShadowBacking, GpuAsanTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<
        std::tuple<amdf_queue_command_type_t, bool>>& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (std::get<1>(info.param) ? "System" : "Local");
    });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
