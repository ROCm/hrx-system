// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/resource_table.h"

#include <array>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr std::array<iree_hal_remote_resource_type_t, 6> kResourceTypes = {
    IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
    IREE_HAL_REMOTE_RESOURCE_TYPE_SEMAPHORE,
    IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE,
    IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER,
    IREE_HAL_REMOTE_RESOURCE_TYPE_FILE,
    IREE_HAL_REMOTE_RESOURCE_TYPE_PHYSICAL_MEMORY,
};

struct TestResource {
  iree_hal_resource_t resource;
  int* destroy_count;
};

static void DestroyTestResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<TestResource*>(base_resource);
  ++*resource->destroy_count;
}

static const iree_hal_resource_vtable_t kTestResourceVTable = {
    DestroyTestResource,
};

static void InitializeTestResource(TestResource* resource, int* destroy_count) {
  iree_hal_resource_initialize(&kTestResourceVTable, &resource->resource);
  resource->destroy_count = destroy_count;
}

static iree_hal_remote_resource_id_t ResourceIdWithType(
    iree_hal_remote_resource_id_t resource_id,
    iree_hal_remote_resource_type_t resource_type) {
  return (resource_id & UINT64_C(0x00FFFFFFFFFFFFFF)) |
         ((uint64_t)resource_type << 56);
}

class ResourceTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_remote_resource_table_initialize(
        /*capacity=*/8, iree_allocator_system(), &table_));
  }

  void TearDown() override {
    iree_hal_remote_resource_table_deinitialize(&table_,
                                                iree_allocator_system());
  }

  iree_hal_remote_resource_table_t table_ = {};
};

class ResourceTableGenerationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_remote_resource_table_initialize(
        /*capacity=*/1, iree_allocator_system(), &table_));
  }

  void TearDown() override {
    iree_hal_remote_resource_table_deinitialize(&table_,
                                                iree_allocator_system());
  }

  iree_hal_remote_resource_table_t table_ = {};
};

TEST_F(ResourceTableTest, LookupRequiresEncodedAndStoredType) {
  for (iree_hal_remote_resource_type_t actual_type : kResourceTypes) {
    int destroy_count = 0;
    TestResource resource;
    InitializeTestResource(&resource, &destroy_count);

    iree_hal_remote_resource_id_t resource_id = 0;
    IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
        &table_, actual_type, &resource, &resource_id));
    iree_hal_resource_release(&resource);

    for (iree_hal_remote_resource_type_t encoded_type : kResourceTypes) {
      iree_hal_remote_resource_id_t encoded_id =
          ResourceIdWithType(resource_id, encoded_type);
      for (iree_hal_remote_resource_type_t expected_type : kResourceTypes) {
        SCOPED_TRACE(::testing::Message()
                     << "actual=" << actual_type << " encoded=" << encoded_type
                     << " expected=" << expected_type);
        void* lookup_result = iree_hal_remote_resource_table_lookup(
            &table_, expected_type, encoded_id);
        if (actual_type == encoded_type && encoded_type == expected_type) {
          EXPECT_EQ(lookup_result, &resource);
        } else {
          EXPECT_EQ(lookup_result, nullptr);
        }
      }
    }

    iree_hal_remote_resource_table_release(&table_, resource_id);
    EXPECT_EQ(destroy_count, 1);
  }
}

TEST_F(ResourceTableTest, ReleaseRejectsCrossTypeIds) {
  for (iree_hal_remote_resource_type_t actual_type : kResourceTypes) {
    int destroy_count = 0;
    TestResource resource;
    InitializeTestResource(&resource, &destroy_count);

    iree_hal_remote_resource_id_t resource_id = 0;
    IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
        &table_, actual_type, &resource, &resource_id));
    iree_hal_resource_release(&resource);

    for (iree_hal_remote_resource_type_t encoded_type : kResourceTypes) {
      if (encoded_type == actual_type) continue;
      SCOPED_TRACE(::testing::Message()
                   << "actual=" << actual_type << " encoded=" << encoded_type);
      iree_hal_remote_resource_table_release(
          &table_, ResourceIdWithType(resource_id, encoded_type));
      EXPECT_EQ(destroy_count, 0);
      EXPECT_EQ(iree_hal_remote_resource_table_lookup(&table_, actual_type,
                                                      resource_id),
                &resource);
    }

    iree_hal_remote_resource_table_release(&table_, resource_id);
    EXPECT_EQ(destroy_count, 1);
  }
}

TEST_F(ResourceTableTest, DetachRejectsCrossTypeIds) {
  for (iree_hal_remote_resource_type_t actual_type : kResourceTypes) {
    int destroy_count = 0;
    TestResource resource;
    InitializeTestResource(&resource, &destroy_count);

    iree_hal_remote_resource_id_t resource_id = 0;
    IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
        &table_, actual_type, &resource, &resource_id));
    iree_hal_resource_release(&resource);

    for (iree_hal_remote_resource_type_t encoded_type : kResourceTypes) {
      if (encoded_type == actual_type) continue;
      SCOPED_TRACE(::testing::Message()
                   << "actual=" << actual_type << " encoded=" << encoded_type);
      EXPECT_EQ(iree_hal_remote_resource_table_detach(
                    &table_, ResourceIdWithType(resource_id, encoded_type)),
                nullptr);
    }

    void* detached_resource =
        iree_hal_remote_resource_table_detach(&table_, resource_id);
    ASSERT_EQ(detached_resource, &resource);
    EXPECT_EQ(iree_hal_remote_resource_table_lookup(&table_, actual_type,
                                                    resource_id),
              nullptr);
    iree_hal_resource_release(detached_resource);
    EXPECT_EQ(destroy_count, 1);
  }
}

TEST_F(ResourceTableGenerationTest, StaleGenerationCannotReleaseReusedSlot) {
  int first_destroy_count = 0;
  TestResource first_resource;
  InitializeTestResource(&first_resource, &first_destroy_count);
  iree_hal_remote_resource_id_t first_id = 0;
  IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
      &table_, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, &first_resource,
      &first_id));
  iree_hal_resource_release(&first_resource);
  iree_hal_remote_resource_table_release(&table_, first_id);
  EXPECT_EQ(first_destroy_count, 1);

  int second_destroy_count = 0;
  TestResource second_resource;
  InitializeTestResource(&second_resource, &second_destroy_count);
  iree_hal_remote_resource_id_t second_id = 0;
  IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
      &table_, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, &second_resource,
      &second_id));
  iree_hal_resource_release(&second_resource);
  EXPECT_EQ(IREE_HAL_REMOTE_RESOURCE_ID_SLOT(first_id),
            IREE_HAL_REMOTE_RESOURCE_ID_SLOT(second_id));
  EXPECT_NE(IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(first_id),
            IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(second_id));

  EXPECT_EQ(iree_hal_remote_resource_table_lookup(
                &table_, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, first_id),
            nullptr);
  iree_hal_remote_resource_table_release(&table_, first_id);
  EXPECT_EQ(second_destroy_count, 0);

  iree_hal_remote_resource_table_release(&table_, second_id);
  EXPECT_EQ(second_destroy_count, 1);
}

TEST_F(ResourceTableGenerationTest, SlotRetiresBeforeGenerationWraparound) {
  int destroy_count = 0;
  TestResource resource;
  InitializeTestResource(&resource, &destroy_count);
  for (uint32_t generation = 1; generation <= UINT16_MAX; ++generation) {
    iree_hal_remote_resource_id_t resource_id = 0;
    IREE_ASSERT_OK(iree_hal_remote_resource_table_assign(
        &table_, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, &resource,
        &resource_id));
    ASSERT_EQ(IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(resource_id), generation);
    iree_hal_remote_resource_table_release(&table_, resource_id);
  }

  iree_hal_remote_resource_id_t wrapped_id = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_remote_resource_table_assign(
                            &table_, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
                            &resource, &wrapped_id));
  EXPECT_EQ(wrapped_id, 0u);

  iree_hal_resource_release(&resource);
  EXPECT_EQ(destroy_count, 1);
}

}  // namespace
