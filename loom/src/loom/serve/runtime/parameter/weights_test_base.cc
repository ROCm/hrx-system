// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights_test_base.h"

#include "iree/base/tooling/flags.h"

IREE_FLAG(string, weight_sources, "", "Weight loader source catalog.");
IREE_FLAG(string, weight_device_uri, "", "Weight loader test device URI.");

namespace loom_serve::testing {

const char* WeightSourcePath() { return FLAG_weight_sources; }
const char* WeightDeviceUri() { return FLAG_weight_device_uri; }

namespace {

class WeightDeviceEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    loom_serve_host_t* host = nullptr;
    iree_hal_driver_t* driver = nullptr;
    auto status =
        loom_serve_test_host_create(1, iree_allocator_system(), &host);
    if (iree_status_is_ok(status)) {
      status = loom_serve_test_driver_acquire(host, &driver);
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_device_options_t options = {
          .driver = driver,
          .uri = WeightDeviceUri()[0]
                     ? iree_make_cstring_view(WeightDeviceUri())
                     : loom_serve_test_device()->device_uri,
      };
      status = loom_serve_device_group_allocate(
          host, 1, &options, allocator_.allocator(), &group_);
    }
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);

    // Observe the native allocation worker once. Pool policy and memory budgets
    // vary per case, but neither changes the physical device or its workers.
    IREE_ASSERT_OK(AllocateWeightPool(device(), false, 0, 0, &consumer_pool_));
    IREE_ASSERT_OK(allocator_.Observe(device(), consumer_pool_));
  }

  void TearDown() override {
    // Every fixture retires its own pools first. Final release must join native
    // cleanup before the allocator wrapper can leave scope.
    IREE_CHECK_OK(loom_serve_pool_free(consumer_pool_));
    IREE_CHECK_OK(loom_serve_device_group_free(group_));
    group_ = nullptr;
    allocator_.AwaitReleased();
  }

  loom_serve_device_t* device() {
    return loom_serve_device_group_at(group_, 0);
  }

  iree_hal_pool_t* consumer_pool() {
    return loom_serve_pool_storage(consumer_pool_);
  }

 private:
  // Native allocator dependency outliving every device-owned allocation.
  NativeAllocator allocator_;
  // One physical device and host retained through the complete test process.
  loom_serve_device_group_t* group_ = nullptr;
  // Caller-owned bindings outlive parameter residency and have separate policy.
  loom_serve_pool_t* consumer_pool_ = nullptr;
};

auto* const weight_device_environment = static_cast<WeightDeviceEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new WeightDeviceEnvironment));

}  // namespace

loom_serve_device_t* WeightDevice() {
  return weight_device_environment->device();
}

iree_hal_pool_t* WeightConsumerPool() {
  return weight_device_environment->consumer_pool();
}

iree_status_t AllocateWeightPool(loom_serve_device_t* device, bool elastic,
                                 uint64_t slab_size, uint64_t memory_limit,
                                 loom_serve_pool_t** out_pool) {
  const auto* dispatch =
      iree_hal_queue_family(loom_serve_device_dispatch_queue(device));
  auto* transfer_queue = loom_serve_device_transfer_queue(device);
  const auto* transfer = iree_hal_queue_family(transfer_queue);
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = dispatch,
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      },
      {.family = transfer, .usage = IREE_HAL_BUFFER_USAGE_TRANSFER},
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = dispatch == transfer ? 1u : 2u,
      .families = families,
  };
  const auto* sanitizer = iree_hal_device_spec_sanitizer(
      iree_hal_device_spec(loom_serve_device_handle(device)));
  const loom_serve_pool_options_t options = {
      .elastic = elastic,
      .weight_slab_size = slab_size,
      .state_slab_size = slab_size,
      .memory_limit = elastic ? memory_limit : 0,
      .asan = iree_any_bit_set(sanitizer->flags,
                               IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)
                  ? sanitizer->asan.pool_options
                  : iree_hal_asan_pool_options_t{},
  };
  return loom_serve_pool_allocate(
      loom_serve_device_group(device), scope, transfer_queue, &options,
      NativeAllocator::ForDevice(device).allocator(), out_pool);
}

}  // namespace loom_serve::testing
