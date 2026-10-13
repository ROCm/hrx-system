// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/module.h"

#include <vector>

#include "iree/module/hal/native_schema.h"
#include "iree/module/hal/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HALModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
  }
  void TearDown() override {
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }
  void Create() {
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_module_register_types(environment_, &table));
    IREE_ASSERT_OK(iree_hal_module_create(environment_, iree_allocator_system(),
                                          &module_));
  }
  // Construction registry, which the native module does not retain.
  iree_vm_environment_t* environment_ = nullptr;
  // Optional native module under test.
  iree_vm_module_t* module_ = nullptr;
};

TEST_F(HALModuleTest, RequiresRegisteredTypes) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_module_create(environment_, iree_allocator_system(), &module_));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(HALModuleTest, RequiresEnvironmentAndOutput) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_module_create(nullptr, iree_allocator_system(), &module_));
  EXPECT_EQ(module_, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_module_create(environment_, iree_allocator_system(), nullptr));
}

TEST_F(HALModuleTest, InvalidAllocatorDoesNotPublish) {
  const iree_vm_ref_type_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_module_register_types(environment_, &table));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_module_create(environment_, iree_allocator_null(), &module_));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(HALModuleTest, NativeDirectoryOutlivesConstructionRegistry) {
  Create();
  iree_vm_environment_free(environment_);
  environment_ = nullptr;
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_module_name(module_), IREE_SV("hal")));
  EXPECT_EQ(iree_vm_module_export_count(module_), 0u);
  EXPECT_EQ(iree_vm_module_function_count(module_), 0u);
  EXPECT_EQ(module_->descriptor->native->export_count,
            IREE_HAL_NATIVE_METHOD_COUNT);
  const auto& entry =
      module_->descriptor->native->exports[IREE_HAL_NATIVE_QUEUE_COPY];
  EXPECT_TRUE(iree_string_view_equal(entry.name, IREE_SV("queue.copy")));
  EXPECT_EQ(entry.version, 1u);
  IREE_ASSERT_OK(iree_vm_native_call_schema_validate(
      entry.schema, IREE_HAL_MODULE_TYPE_COUNT));
}

TEST_F(HALModuleTest, LinksOptionalPresenceOnlyAfterPhysicalAdmission) {
  Create();
  const auto& entry =
      module_->descriptor->native->exports[IREE_HAL_NATIVE_QUEUE_COPY];
  const auto* types = module_->descriptor->ref_types.data;
  // Each optional before boundary contains an empty barrier list. Zero and
  // one are distinct valid encodings; two is physically valid but not HAL.
  for (uint32_t presence : {0u, 1u, 2u}) {
    std::vector<uint32_t> words(IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT +
                                2 * presence);
    words[IREE_HAL_NATIVE_QUEUE_COPY_WAITS + 1] =
        IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT;
    words[IREE_HAL_NATIVE_QUEUE_COPY_SIGNALS + 1] =
        IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT;
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE] = presence;
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE + 1] =
        IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT;
    for (uint32_t i = 0; i < presence; ++i) {
      words[IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT + i * 2 + 1] = words.size();
    }
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_AFTER + 1] = words.size();
    iree_vm_native_callsite_t callsite = {IREE_SV("hal"),
                                          IREE_SV("queue.copy"),
                                          1,
                                          *entry.schema,
                                          0,
                                          words.data(),
                                          static_cast<uint32_t>(words.size()),
                                          {1, 1, 0}};
    IREE_ASSERT_OK(
        iree_vm_native_call_verify(&callsite, iree_allocator_system()));
    iree_host_size_t scratch_size = 123;
    iree_status_t status =
        iree_vm_native_call_link(&entry, types, &callsite, types,
                                 iree_allocator_system(), &scratch_size);
    if (presence <= 1) {
      IREE_ASSERT_OK(status);
      EXPECT_EQ(scratch_size, 0u);
      ++callsite.version;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INCOMPATIBLE,
          iree_vm_native_call_link(&entry, types, &callsite, types,
                                   iree_allocator_system(), &scratch_size));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
      EXPECT_EQ(scratch_size, 123u);
    }
  }
}

TEST_F(HALModuleTest, RejectsMultipleRecipesInsideNestedBarrier) {
  Create();
  const auto& entry =
      module_->descriptor->native->exports[IREE_HAL_NATIVE_QUEUE_COPY];
  const auto* types = module_->descriptor->ref_types.data;
  const uint32_t boundary = IREE_HAL_NATIVE_QUEUE_COPY_WORD_COUNT;
  const uint32_t barrier = boundary + 2;
  const uint32_t buffer = barrier + IREE_HAL_NATIVE_BARRIER_WORD_COUNT;
  const uint32_t recipe = buffer + IREE_HAL_NATIVE_BUFFER_BARRIER_WORD_COUNT;
  for (uint32_t presence : {0u, 1u, 2u}) {
    std::vector<uint32_t> words(
        recipe + presence * IREE_HAL_NATIVE_TRANSITION_RECIPE_WORD_COUNT);
    words[IREE_HAL_NATIVE_QUEUE_COPY_WAITS + 1] = boundary;
    words[IREE_HAL_NATIVE_QUEUE_COPY_SIGNALS + 1] = boundary;
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE] = 1;
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_BEFORE + 1] = boundary;
    words[boundary] = 1;
    words[boundary + 1] = barrier;
    words[barrier + IREE_HAL_NATIVE_BARRIER_MEMORY + 1] = buffer;
    words[barrier + IREE_HAL_NATIVE_BARRIER_BUFFERS] = 1;
    words[barrier + IREE_HAL_NATIVE_BARRIER_BUFFERS + 1] = buffer;
    words[buffer + IREE_HAL_NATIVE_BUFFER_BARRIER_RECIPE] = presence;
    words[buffer + IREE_HAL_NATIVE_BUFFER_BARRIER_RECIPE + 1] = recipe;
    for (uint32_t i = 0; i < presence; ++i) {
      words[recipe + i * IREE_HAL_NATIVE_TRANSITION_RECIPE_WORD_COUNT +
            IREE_HAL_NATIVE_TRANSITION_RECIPE_OPERATIONS + 1] = words.size();
    }
    words[IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_AFTER + 1] = words.size();
    const iree_vm_native_callsite_t callsite = {
        IREE_SV("hal"),
        IREE_SV("queue.copy"),
        1,
        *entry.schema,
        0,
        words.data(),
        static_cast<uint32_t>(words.size()),
        {1, 1, 0}};
    IREE_ASSERT_OK(
        iree_vm_native_call_verify(&callsite, iree_allocator_system()));
    iree_host_size_t scratch_size = 123;
    iree_status_t status =
        iree_vm_native_call_link(&entry, types, &callsite, types,
                                 iree_allocator_system(), &scratch_size);
    if (presence <= 1) {
      IREE_ASSERT_OK(status);
      EXPECT_GT(scratch_size, 0u);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
      EXPECT_EQ(scratch_size, 123u);
    }
  }
}

}  // namespace
