// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/environment.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"

namespace {

struct CountingAllocator {
  // Allocator performing the actual memory operations.
  iree_allocator_t delegate;
  // Number of allocation-like commands forwarded.
  iree_host_size_t allocation_count;
  // Number of free commands forwarded.
  iree_host_size_t free_count;
};

iree_status_t CountingAllocatorControl(void* self,
                                       iree_allocator_command_t command,
                                       const void* params, void** inout_ptr) {
  auto* allocator = static_cast<CountingAllocator*>(self);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_count;
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_count;
      break;
    default:
      break;
  }
  return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                 inout_ptr);
}

iree_allocator_t MakeCountingAllocator(CountingAllocator* allocator) {
  return iree_allocator_t{
      allocator,
      CountingAllocatorControl,
  };
}

struct TestProvider {
  // Provider table published to the environment.
  iree_vm_ref_type_table_t table;
  // Sole type descriptor owned by this provider.
  iree_vm_ref_type_descriptor_t descriptor;
  // Dense append-order provider handle storage.
  struct {
    // Sole type handle at ordinal zero.
    iree_vm_ref_type_t object;
  } types;

  void Initialize(iree_string_view_t namespace_name) {
    table = {
        sizeof(table),
        IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
        namespace_name,
        {&types, 1},
    };
    descriptor = {
        nullptr,
        &table,
        IREE_SV("object"),
    };
    types.object = &descriptor;
  }
};

struct DuplicateNameProvider {
  // Provider table published to the environment.
  iree_vm_ref_type_table_t table;
  // First type descriptor.
  iree_vm_ref_type_descriptor_t first_descriptor;
  // Second type descriptor with the same local name.
  iree_vm_ref_type_descriptor_t second_descriptor;
  // Dense append-order provider handle storage.
  struct {
    // First type handle.
    iree_vm_ref_type_t first;
    // Second type handle.
    iree_vm_ref_type_t second;
  } types;

  void Initialize() {
    table = {
        sizeof(table),
        IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
        IREE_SV("duplicate_types"),
        {&types, 2},
    };
    first_descriptor = {
        nullptr,
        &table,
        IREE_SV("object"),
    };
    second_descriptor = {
        nullptr,
        &table,
        IREE_SV("object"),
    };
    types = {
        &first_descriptor,
        &second_descriptor,
    };
  }
};

TEST(VMEnvironmentTest, AllocateUsesOneAllocationAndPublishesCoreTypes) {
  CountingAllocator allocator = {
      iree_allocator_system(),
      0,
      0,
  };
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(iree_vm_environment_allocate(MakeCountingAllocator(&allocator),
                                              &environment));
  ASSERT_NE(environment, nullptr);
  EXPECT_EQ(allocator.allocation_count, 1u);

  const iree_vm_ref_type_table_t* table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
  ASSERT_NE(table, nullptr);
  iree_vm_ref_types_t types = {};
  IREE_ASSERT_OK(iree_vm_ref_types_resolve(table, &types));
  EXPECT_NE(types.buffer, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(types.buffer->type_name, IREE_SV("buffer")));

  iree_vm_environment_free(environment);
  EXPECT_EQ(allocator.free_count, 1u);
}

TEST(VMEnvironmentTest, CoreBufferTypeSupportsTypedOwnership) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  const iree_vm_ref_type_table_t* table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
  ASSERT_NE(table, nullptr);
  iree_vm_ref_types_t types = {};
  IREE_ASSERT_OK(iree_vm_ref_types_resolve(table, &types));

  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, iree_allocator_system(), &buffer));
  iree_vm_ref_t ref = iree_vm_buffer_ref_from_ptr_move(&types, &buffer);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_TRUE(iree_vm_buffer_ref_isa(&types, ref));
  iree_vm_environment_free(environment);

  iree_vm_buffer_t* moved_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_ref_move(&types, &ref, &moved_buffer));
  EXPECT_TRUE(iree_vm_ref_is_null(ref));
  ASSERT_NE(moved_buffer, nullptr);
  iree_vm_buffer_release(moved_buffer);
}

TEST(VMEnvironmentTest, RegistrationIsFailureAtomicAndNamespaceUnique) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  TestProvider provider;
  provider.Initialize(IREE_SV("test"));
  provider.descriptor.table = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));
  EXPECT_EQ(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("test")),
      nullptr);

  provider.descriptor.table = &provider.table;
  IREE_EXPECT_OK(iree_vm_environment_register_ref_type_table(environment,
                                                             &provider.table));
  EXPECT_EQ(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("test")),
      &provider.table);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));

  TestProvider collision;
  collision.Initialize(IREE_SV("test"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_vm_environment_register_ref_type_table(
                            environment, &collision.table));
  EXPECT_EQ(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("test")),
      &provider.table);

  iree_vm_environment_free(environment);
}

TEST(VMEnvironmentTest, RegistrationRejectsIncompatibleTablePrefixes) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  TestProvider provider;
  provider.Initialize(IREE_SV("test"));

  provider.table.structure_size = IREE_VM_REF_TYPE_TABLE_V0_REQUIRED_SIZE - 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));
  provider.table.structure_size = sizeof(provider.table);
  provider.table.flags = IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));
  EXPECT_EQ(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("test")),
      nullptr);

  iree_vm_environment_free(environment);
}

TEST(VMEnvironmentTest, RegistrationRejectsMalformedDescriptorGraphs) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  TestProvider provider;
  provider.Initialize(IREE_SV("empty"));
  provider.table.types.count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));

  provider.Initialize(IREE_SV("missing"));
  provider.types.object = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));

  provider.Initialize(IREE_SV("unnamed"));
  provider.descriptor.type_name = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_environment_register_ref_type_table(
                            environment, &provider.table));

  DuplicateNameProvider duplicate_provider;
  duplicate_provider.Initialize();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_environment_register_ref_type_table(
                            environment, &duplicate_provider.table));

  iree_vm_environment_free(environment);
}

TEST(VMEnvironmentTest, FixedCapacityIncludesTheCoreFamily) {
  static const char* const kNamespaceNames[] = {
      "family_00", "family_01", "family_02", "family_03",
      "family_04", "family_05", "family_06", "family_07",
      "family_08", "family_09", "family_10", "family_11",
      "family_12", "family_13", "family_14", "family_15",
  };
  std::array<TestProvider, 16> providers;
  for (iree_host_size_t i = 0; i < providers.size(); ++i) {
    providers[i].Initialize(iree_make_cstring_view(kNamespaceNames[i]));
  }

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  for (iree_host_size_t i = 0; i < providers.size() - 1; ++i) {
    IREE_EXPECT_OK(iree_vm_environment_register_ref_type_table(
        environment, &providers[i].table));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_vm_environment_register_ref_type_table(
                            environment, &providers.back().table));
  EXPECT_EQ(iree_vm_environment_lookup_ref_type_table(
                environment, providers.back().table.namespace_name),
            nullptr);

  iree_vm_environment_free(environment);
}

TEST(VMEnvironmentTest, ConcurrentRegistrationAndLookupRemainCoherent) {
  static const char* const kNamespaceNames[] = {
      "concurrent_0", "concurrent_1", "concurrent_2", "concurrent_3",
      "concurrent_4", "concurrent_5", "concurrent_6", "concurrent_7",
  };
  constexpr iree_host_size_t kProviderCount = IREE_ARRAYSIZE(kNamespaceNames);
  std::array<TestProvider, kProviderCount> providers;
  for (iree_host_size_t i = 0; i < providers.size(); ++i) {
    providers[i].Initialize(iree_make_cstring_view(kNamespaceNames[i]));
  }

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  std::array<iree_status_t, kProviderCount> statuses = {};
  std::atomic<bool> start = false;
  std::atomic<bool> lookup_started = false;
  std::atomic<iree_host_size_t> remaining = kProviderCount;
  std::atomic<iree_host_size_t> successful_lookups = 0;

  std::vector<std::thread> threads;
  for (iree_host_size_t i = 0; i < providers.size(); ++i) {
    threads.emplace_back([&, i]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!lookup_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      statuses[i] = iree_vm_environment_register_ref_type_table(
          environment, &providers[i].table);
      remaining.fetch_sub(1, std::memory_order_release);
    });
  }
  std::thread lookup_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"))) {
      successful_lookups.fetch_add(1, std::memory_order_relaxed);
    }
    lookup_started.store(true, std::memory_order_release);
    while (remaining.load(std::memory_order_acquire) != 0) {
      for (const TestProvider& provider : providers) {
        if (iree_vm_environment_lookup_ref_type_table(
                environment, provider.table.namespace_name)) {
          successful_lookups.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  });

  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }
  lookup_thread.join();
  for (iree_host_size_t i = 0; i < statuses.size(); ++i) {
    IREE_EXPECT_OK(statuses[i]);
    EXPECT_EQ(iree_vm_environment_lookup_ref_type_table(
                  environment, providers[i].table.namespace_name),
              &providers[i].table);
  }
  EXPECT_GT(successful_lookups.load(std::memory_order_relaxed), 0u);

  iree_vm_environment_free(environment);
}

}  // namespace
