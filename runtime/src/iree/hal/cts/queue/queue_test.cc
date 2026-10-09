// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class QueueTest : public CtsTestBase<> {};

struct DynamicQueueFamily {
  // Canonical family ordinal within the device.
  iree_hal_queue_family_ordinal_t ordinal;

  // Pointer-unique family identity exposed by the device.
  const iree_hal_queue_family_t* identity;

  // Immutable family specification row.
  const iree_hal_queue_family_spec_t* spec;
};

static bool FindDynamicQueueFamily(
    iree_hal_device_t* device,
    iree_hal_queue_family_role_flags_t required_roles,
    DynamicQueueFamily* out_family) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
    const iree_hal_queue_family_spec_t* family_spec = &queue_spec->families[i];
    if (!iree_all_bits_set(family_spec->role_flags, required_roles) ||
        !iree_any_bit_set(
            family_spec->flags,
            IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
      continue;
    }
    const iree_hal_queue_family_ordinal_t family_ordinal =
        (iree_hal_queue_family_ordinal_t)i;
    DynamicQueueFamily family = {
        .ordinal = family_ordinal,
        .identity = iree_hal_device_queue_family(device, family_ordinal),
        .spec = family_spec};
    if (!family.identity) {
      return false;
    }
    *out_family = family;
    return true;
  }
  return false;
}

static bool FindUnsupportedPriority(
    const iree_hal_queue_family_spec_t* family_spec,
    iree_hal_queue_priority_t* out_priority) {
  if (family_spec->priorities[0] >
      std::numeric_limits<iree_hal_queue_priority_t>::min()) {
    *out_priority = std::numeric_limits<iree_hal_queue_priority_t>::min();
    return true;
  }
  if (family_spec->priorities[family_spec->priority_count - 1] <
      std::numeric_limits<iree_hal_queue_priority_t>::max()) {
    *out_priority = std::numeric_limits<iree_hal_queue_priority_t>::max();
    return true;
  }
  for (iree_host_size_t i = 1; i < family_spec->priority_count; ++i) {
    if ((int64_t)family_spec->priorities[i] -
            (int64_t)family_spec->priorities[i - 1] >
        1) {
      *out_priority = family_spec->priorities[i - 1] + 1;
      return true;
    }
  }
  return false;
}

static void ExpectQueuePropertiesSupportedByFamily(
    const iree_hal_queue_t* queue,
    const iree_hal_queue_family_spec_t* family_spec) {
  bool priority_supported = false;
  for (iree_host_size_t i = 0; i < family_spec->priority_count; ++i) {
    priority_supported |=
        family_spec->priorities[i] == iree_hal_queue_priority(queue);
  }
  EXPECT_TRUE(priority_supported);
  EXPECT_EQ(0u, iree_hal_queue_features(queue) &
                    ~family_spec->supported_queue_features);

  const iree_hal_queue_execution_resource_list_t resources =
      iree_hal_queue_execution_resources(queue);
  if (!resources.count) {
    return;
  }
  ASSERT_NE(nullptr, resources.ordinals);
  std::vector<iree_host_size_t> group_resource_counts(
      family_spec->execution_resource_group_count, 0);
  for (iree_host_size_t i = 0; i < resources.count; ++i) {
    if (i > 0) {
      EXPECT_LT(resources.ordinals[i - 1], resources.ordinals[i]);
    }
    ASSERT_LT(resources.ordinals[i], family_spec->execution_resource_count);
    const iree_hal_queue_execution_resource_group_ordinal_t group_ordinal =
        family_spec->execution_resources[resources.ordinals[i]].group_ordinal;
    ASSERT_LT(group_ordinal, group_resource_counts.size());
    ++group_resource_counts[group_ordinal];
  }
  for (iree_host_size_t i = 0; i < group_resource_counts.size(); ++i) {
    EXPECT_GE(group_resource_counts[i],
              family_spec->execution_resource_groups[i]
                  .minimum_selected_resource_count);
  }
}

TEST_P(QueueTest, ProvisionedInventoryMatchesDeviceSpec) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  const iree_host_size_t family_count =
      queue_spec ? queue_spec->family_count : 0;

  std::vector<const iree_hal_queue_family_t*> observed_families;
  std::vector<iree_hal_queue_t*> observed_queues;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const iree_hal_queue_family_ordinal_t family_ordinal =
        (iree_hal_queue_family_ordinal_t)i;
    const iree_hal_queue_family_t* queue_family =
        iree_hal_device_queue_family(device_, family_ordinal);
    ASSERT_NE(nullptr, queue_family)
        << "device did not expose advertised queue family " << i;
    EXPECT_EQ(device_, iree_hal_queue_family_device(queue_family));
    EXPECT_EQ(family_ordinal, iree_hal_queue_family_ordinal(queue_family));
    const iree_hal_queue_family_spec_t* family_spec = &queue_spec->families[i];
    EXPECT_EQ(family_spec, iree_hal_queue_family_spec(queue_family));
    EXPECT_EQ(queue_family,
              iree_hal_device_queue_family(device_, family_ordinal));
    EXPECT_EQ(observed_families.end(),
              std::find(observed_families.begin(), observed_families.end(),
                        queue_family))
        << "queue family " << i << " aliases an earlier family identity";
    observed_families.push_back(queue_family);

    const uint32_t queue_count =
        queue_spec->families[i].provisioned_queue_count;
    for (uint32_t j = 0; j < queue_count; ++j) {
      const iree_hal_queue_ordinal_t queue_ordinal =
          (iree_hal_queue_ordinal_t)j;
      iree_hal_queue_t* queue =
          iree_hal_device_queue(device_, family_ordinal, queue_ordinal);
      ASSERT_NE(nullptr, queue)
          << "device did not expose advertised queue " << i << ":" << j;
      EXPECT_EQ(queue,
                iree_hal_device_queue(device_, family_ordinal, queue_ordinal));
      EXPECT_EQ(queue_family, iree_hal_queue_family(queue));
      ExpectQueuePropertiesSupportedByFamily(queue, family_spec);
      EXPECT_EQ(observed_queues.end(), std::find(observed_queues.begin(),
                                                 observed_queues.end(), queue))
          << "queue " << i << ":" << j
          << " aliases an earlier provisioned queue";
      observed_queues.push_back(queue);
    }
    EXPECT_EQ(nullptr,
              iree_hal_device_queue(device_, family_ordinal, queue_count));
  }

  const iree_hal_queue_family_ordinal_t invalid_family_ordinal =
      (iree_hal_queue_family_ordinal_t)family_count;
  EXPECT_EQ(nullptr,
            iree_hal_device_queue_family(device_, invalid_family_ordinal));
  EXPECT_EQ(nullptr, iree_hal_device_queue(device_, invalid_family_ordinal, 0));
}

TEST_P(QueueTest, DynamicallyAcquiredQueueExecutesBarrier) {
  DynamicQueueFamily family;
  if (!FindDynamicQueueFamily(device_, /*required_roles=*/0, &family)) {
    GTEST_SKIP() << "device has no dynamically acquirable queue family";
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(family.identity, &params, queue.out()));
  ASSERT_NE(nullptr, queue.get());
  EXPECT_EQ(family.identity, iree_hal_queue_family(queue));
  EXPECT_EQ(params.priority, iree_hal_queue_priority(queue));
  EXPECT_EQ(params.features, iree_hal_queue_features(queue));
  EXPECT_EQ(params.execution_resources.count,
            iree_hal_queue_execution_resources(queue).count);

  for (iree_hal_queue_ordinal_t i = 0; i < family.spec->provisioned_queue_count;
       ++i) {
    EXPECT_NE(iree_hal_device_queue(device_, family.ordinal, i), queue.get());
  }

  SemaphoreList signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue, iree_hal_semaphore_list_empty(),
                                        signal, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(QueueTest, QueueAcquisitionCanonicalizesCompleteResourceSet) {
  DynamicQueueFamily family;
  if (!FindDynamicQueueFamily(device_, /*required_roles=*/0, &family) ||
      !family.spec->execution_resource_count) {
    GTEST_SKIP()
        << "device has no dynamically acquirable family with execution "
           "resources";
  }

  std::vector<iree_hal_queue_execution_resource_ordinal_t> resource_ordinals(
      family.spec->execution_resource_count);
  for (iree_host_size_t i = 0; i < resource_ordinals.size(); ++i) {
    resource_ordinals[i] = (iree_hal_queue_execution_resource_ordinal_t)i;
  }
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.execution_resources.count = resource_ordinals.size();
  params.execution_resources.ordinals = resource_ordinals.data();

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(family.identity, &params, queue.out()));
  EXPECT_EQ(family.identity, iree_hal_queue_family(queue));
  const iree_hal_queue_execution_resource_list_t achieved_resources =
      iree_hal_queue_execution_resources(queue);
  EXPECT_EQ(0u, achieved_resources.count);
  EXPECT_EQ(nullptr, achieved_resources.ordinals);
}

TEST_P(QueueTest, QueueAcquisitionPreservesResourceSubset) {
  DynamicQueueFamily family;
  if (!FindDynamicQueueFamily(device_, /*required_roles=*/0, &family) ||
      family.spec->execution_resource_count < 2) {
    GTEST_SKIP() << "device has no dynamically acquirable resource subset";
  }

  std::vector<iree_host_size_t> group_counts(
      family.spec->execution_resource_group_count, 0);
  for (iree_host_size_t i = 0; i < family.spec->execution_resource_count; ++i) {
    ++group_counts[family.spec->execution_resources[i].group_ordinal];
  }
  iree_host_size_t omitted_ordinal = family.spec->execution_resource_count;
  for (iree_host_size_t i = 0; i < family.spec->execution_resource_count; ++i) {
    const auto group = family.spec->execution_resources[i].group_ordinal;
    if (group_counts[group] > family.spec->execution_resource_groups[group]
                                  .minimum_selected_resource_count) {
      omitted_ordinal = i;
      break;
    }
  }
  if (omitted_ordinal == family.spec->execution_resource_count) {
    GTEST_SKIP() << "family requires every advertised execution resource";
  }

  std::vector<iree_hal_queue_execution_resource_ordinal_t> expected_ordinals;
  for (iree_host_size_t i = 0; i < family.spec->execution_resource_count; ++i) {
    if (i != omitted_ordinal) {
      expected_ordinals.push_back(
          (iree_hal_queue_execution_resource_ordinal_t)i);
    }
  }
  Ref<iree_hal_queue_t> queue;
  {
    // Acquisition must copy the parameter storage before returning.
    auto requested_ordinals = expected_ordinals;
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    params.execution_resources = {requested_ordinals.size(),
                                  requested_ordinals.data()};
    IREE_ASSERT_OK(
        iree_hal_queue_acquire(family.identity, &params, queue.out()));
  }
  EXPECT_EQ(family.identity, iree_hal_queue_family(queue));
  const auto achieved_resources = iree_hal_queue_execution_resources(queue);
  ASSERT_EQ(expected_ordinals.size(), achieved_resources.count);
  ASSERT_NE(nullptr, achieved_resources.ordinals);
  EXPECT_TRUE(std::equal(expected_ordinals.begin(), expected_ordinals.end(),
                         achieved_resources.ordinals));

  SemaphoreList signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue, iree_hal_semaphore_list_empty(),
                                        signal, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(QueueTest, DynamicallyAcquiredQueueOrdersProvisionedQueue) {
  DynamicQueueFamily family;
  if (!FindDynamicQueueFamily(device_, /*required_roles=*/0, &family)) {
    GTEST_SKIP() << "device has no dynamically acquirable queue family";
  }
  if (!family.spec->provisioned_queue_count) {
    GTEST_SKIP() << "dynamic queue family has no provisioned queues";
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  Ref<iree_hal_queue_t> dynamic_queue;
  IREE_ASSERT_OK(
      iree_hal_queue_acquire(family.identity, &params, dynamic_queue.out()));

  SemaphoreList producer_signal(device_, {0}, {1});
  SemaphoreList completion_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      dynamic_queue, iree_hal_semaphore_list_empty(), producer_signal,
      /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  iree_hal_queue_t* provisioned_queue =
      iree_hal_device_queue(device_, family.ordinal, 0);
  ASSERT_NE(nullptr, provisioned_queue);
  IREE_ASSERT_OK(iree_hal_queue_barrier(provisioned_queue, producer_signal,
                                        completion_signal, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));

  // Releasing the producing queue must not invalidate the dependency edge
  // consumed by another queue.
  dynamic_queue.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      completion_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(QueueTest, QueueAcquisitionRejectsInvalidRequests) {
  DynamicQueueFamily family;
  if (!FindDynamicQueueFamily(device_, /*required_roles=*/0, &family)) {
    GTEST_SKIP() << "device has no dynamically acquirable queue family";
  }

  iree_hal_queue_t* const sentinel =
      reinterpret_cast<iree_hal_queue_t*>(uintptr_t{1});
  iree_hal_queue_t* output = sentinel;

  iree_hal_queue_priority_t unsupported_priority = 0;
  iree_hal_queue_params_t params;
  if (FindUnsupportedPriority(family.spec, &unsupported_priority)) {
    iree_hal_queue_params_initialize(&params);
    params.priority = unsupported_priority;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_acquire(family.identity, &params, &output));
    EXPECT_EQ(sentinel, output);
  }

  iree_hal_queue_params_initialize(&params);
  const iree_hal_queue_feature_flags_t unsupported_features =
      ~family.spec->supported_queue_features;
  if (unsupported_features) {
    params.features = unsupported_features;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_acquire(family.identity, &params, &output));
    EXPECT_EQ(sentinel, output);
  }

  iree_hal_queue_params_initialize(&params);
  const iree_hal_queue_execution_resource_ordinal_t out_of_range_resource =
      (iree_hal_queue_execution_resource_ordinal_t)
          family.spec->execution_resource_count;
  params.execution_resources.count = 1;
  params.execution_resources.ordinals = &out_of_range_resource;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_acquire(family.identity, &params, &output));
  EXPECT_EQ(sentinel, output);

  params.execution_resources.count = 1;
  params.execution_resources.ordinals = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_acquire(family.identity, &params, &output));
  EXPECT_EQ(sentinel, output);

  if (family.spec->execution_resource_count >= 1) {
    const iree_hal_queue_execution_resource_ordinal_t duplicate_resources[] = {
        0, 0};
    params.execution_resources.count = IREE_ARRAYSIZE(duplicate_resources);
    params.execution_resources.ordinals = duplicate_resources;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_acquire(family.identity, &params, &output));
    EXPECT_EQ(sentinel, output);
  }

  if (family.spec->execution_resource_count >= 2) {
    const iree_hal_queue_execution_resource_ordinal_t unordered_resources[] = {
        1, 0};
    params.execution_resources.count = IREE_ARRAYSIZE(unordered_resources);
    params.execution_resources.ordinals = unordered_resources;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_acquire(family.identity, &params, &output));
    EXPECT_EQ(sentinel, output);
  }
}

TEST_P(QueueTest, QueueAcquisitionRequiresDynamicFamily) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
    const iree_hal_queue_family_spec_t* family_spec = &queue_spec->families[i];
    if (iree_any_bit_set(family_spec->flags,
                         IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
      continue;
    }
    const iree_hal_queue_family_t* family = iree_hal_device_queue_family(
        device_, (iree_hal_queue_family_ordinal_t)i);
    ASSERT_NE(nullptr, family);
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    iree_hal_queue_t* const sentinel =
        reinterpret_cast<iree_hal_queue_t*>(uintptr_t{1});
    iree_hal_queue_t* output = sentinel;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          iree_hal_queue_acquire(family, &params, &output));
    EXPECT_EQ(sentinel, output);
    return;
  }
  GTEST_SKIP() << "all device queue families support dynamic acquisition";
}

CTS_REGISTER_TEST_SUITE(QueueTest);

}  // namespace iree::hal::cts
