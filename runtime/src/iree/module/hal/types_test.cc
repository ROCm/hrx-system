// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/types.h"

#include <array>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HALModuleTypesTest, RegistersCanonicalHandlesAcrossEnvironments) {
  const iree_vm_ref_type_table_t* first = nullptr;
  const iree_vm_ref_type_table_t* second = nullptr;
  for (auto* output : {&first, &second}) {
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    IREE_ASSERT_OK(iree_hal_module_register_types(environment, output));
    EXPECT_EQ(*output, iree_vm_environment_lookup_ref_type_table(
                           environment, IREE_SV("hal")));
    const iree_vm_ref_type_table_t* duplicate = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ALREADY_EXISTS,
        iree_hal_module_register_types(environment, &duplicate));
    EXPECT_EQ(duplicate, nullptr);
    iree_vm_environment_free(environment);
  }
  EXPECT_EQ(first, second);
  iree_hal_module_types_t types = {};
  IREE_ASSERT_OK(iree_hal_module_types_resolve(first, &types));
  EXPECT_NE(types.buffer, types.buffer_view);
  EXPECT_EQ(types.buffer->table, first);
  EXPECT_EQ(types.buffer_view->table, first);
}

// Valid alternate provider tables exercise consumer prefix compatibility at
// the public registration boundary, including providers from newer libraries.
struct Provider {
  // Complete provider table, borrowed by its environment.
  iree_vm_ref_type_table_t table = {};
  // Immutable descriptors after registration.
  std::array<iree_vm_ref_type_descriptor_t, IREE_HAL_MODULE_TYPE_COUNT + 1>
      descriptors = {};
  // Dense append-order type handles.
  std::array<iree_vm_ref_type_t, IREE_HAL_MODULE_TYPE_COUNT + 1> types = {};

  Provider(iree_string_view_t namespace_name,
           std::initializer_list<iree_string_view_t> names) {
    table = {sizeof(table),
             IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
             namespace_name,
             {types.data(), names.size()}};
    size_t ordinal = 0;
    for (auto name : names) {
      descriptors[ordinal] = {nullptr, &table, name};
      types[ordinal] = &descriptors[ordinal];
      ++ordinal;
    }
  }
};

TEST(HALModuleTypesTest, ResolvesKnownPrefixFromNewerProvider) {
  Provider provider(
      IREE_SV("hal"),
      {IREE_SV("buffer"), IREE_SV("buffer_view"), IREE_SV("device_group"),
       IREE_SV("queue"), IREE_SV("command_buffer"), IREE_SV("semaphore"),
       IREE_SV("pool"), IREE_SV("device")});
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(environment,
                                                             &provider.table));
  iree_hal_module_types_t types = {};
  IREE_ASSERT_OK(iree_hal_module_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("hal")),
      &types));
  EXPECT_EQ(types.buffer, provider.types[0]);
  EXPECT_EQ(types.buffer_view, provider.types[1]);
  EXPECT_EQ(types.device_group, provider.types[2]);
  EXPECT_EQ(types.queue, provider.types[3]);
  EXPECT_EQ(types.command_buffer, provider.types[4]);
  EXPECT_EQ(types.semaphore, provider.types[5]);
  EXPECT_EQ(types.pool, provider.types[6]);
  iree_vm_environment_free(environment);
}

TEST(HALModuleTypesTest, RejectsIncompatiblePrefixesWithoutPublishing) {
  Provider short_prefix(IREE_SV("hal"), {IREE_SV("buffer")});
  Provider reordered(IREE_SV("hal"), {IREE_SV("buffer_view"), IREE_SV("buffer"),
                                      IREE_SV("device_group"), IREE_SV("queue"),
                                      IREE_SV("command_buffer"),
                                      IREE_SV("semaphore"), IREE_SV("pool")});
  Provider other(IREE_SV("other"), {IREE_SV("buffer"), IREE_SV("buffer_view")});
  for (auto* provider : {&short_prefix, &reordered, &other}) {
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
        environment, &provider->table));
    iree_hal_module_types_t types = {provider->types[0], provider->types[0]};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_module_types_resolve(&provider->table, &types));
    EXPECT_EQ(types.buffer, provider->types[0]);
    EXPECT_EQ(types.buffer_view, provider->types[0]);
    iree_vm_environment_free(environment);
  }
}

}  // namespace
