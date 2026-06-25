// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/readback.h"

#include <cstring>

#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  T* value_ = nullptr;
};

iree_hal_semaphore_list_t SingleSemaphoreList(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  return iree_hal_semaphore_list_t{
      // Number of wait or signal edges in this stack-backed list.
      /*.count=*/1,
      // Stack-backed semaphore handle.
      /*.semaphores=*/semaphore_storage,
      // Stack-backed payload value.
      /*.payload_values=*/payload_storage,
  };
}

TEST(Id4ToolingReadbackTest, ReadsBufferBindingAfterWait) {
  Ref<iree_hal_device_group_t, iree_hal_device_group_release> device_group;
  device_group.reset(id4::test::CreateLocalSyncDeviceGroup());
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), 0);

  static const uint32_t kPayload[] = {
      0x12345678u,
      0xA5A5A5A5u,
      0x0BADCAFEu,
      0xDEADBEEFu,
  };
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, sizeof(kPayload),
      buffer.out()));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           update_semaphore.out()));
  iree_hal_semaphore_t* update_signal_storage = nullptr;
  uint64_t update_signal_payload = 1;
  iree_hal_semaphore_list_t update_signal_list =
      SingleSemaphoreList(&update_signal_storage, &update_signal_payload,
                          update_semaphore.get(), update_signal_payload);
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      update_signal_list, kPayload, /*source_offset=*/0, buffer.get(),
      /*target_offset=*/0, sizeof(kPayload), IREE_HAL_UPDATE_FLAG_NONE));

  iree_hal_buffer_binding_t binding = {
      // Source buffer initialized by the queued update.
      /*.buffer=*/buffer.get(),
      // Source byte offset.
      /*.offset=*/0,
      // Source byte length.
      /*.length=*/sizeof(kPayload),
  };
  iree_hal_semaphore_t* update_wait_storage = nullptr;
  uint64_t update_wait_payload = 1;
  iree_hal_semaphore_list_t update_wait_list =
      SingleSemaphoreList(&update_wait_storage, &update_wait_payload,
                          update_semaphore.get(), update_wait_payload);

  id4_tooling_host_bytes_t bytes;
  std::memset(&bytes, 0, sizeof(bytes));
  id4_tooling_readback_buffer_binding_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.device = device;
  options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  options.binding = binding;
  options.wait_semaphore_list = update_wait_list;
  options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_readback_buffer_binding(&options, &bytes));

  ASSERT_EQ(bytes.length, sizeof(kPayload));
  EXPECT_EQ(std::memcmp(bytes.data, kPayload, sizeof(kPayload)), 0);
  id4_tooling_host_bytes_deinitialize(&bytes, iree_allocator_system());
}

TEST(Id4ToolingReadbackTest, RejectsMissingSourceBuffer) {
  id4_tooling_host_bytes_t bytes;
  std::memset(&bytes, 0, sizeof(bytes));
  id4_tooling_readback_buffer_binding_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.device = reinterpret_cast<iree_hal_device_t*>(0x1);
  options.binding.length = 4;
  options.host_allocator = iree_allocator_system();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_tooling_readback_buffer_binding(&options, &bytes));
}

}  // namespace
