// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"

namespace iree::hal::amd::fabric {

TEST_F(MemoryBackendTest, RejectsAtomicPromisesAbsentFromNativeBuffers) {
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    for (uint32_t operation = 1;
         operation <= IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL; operation <<= 1) {
      iree_hal_atomic_operation_capabilities_t atomics = {};
      atomics.*cell = operation;
      iree_hal_pool_t* pool = nullptr;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_UNAVAILABLE,
          CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0,
                     IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_BUFFER_USAGE_STORAGE,
                     atomics));
      EXPECT_EQ(pool, nullptr);
    }
  }
  EXPECT_GT(native_.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, ZeroComputeRequirementsSelectNativeControlStorage) {
  native_.required_flags = 0;
  native_families_[0].program_atomics = true;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.atomics[0] = {
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE,
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM,
                             AMDF_ATOMIC_SCOPE_SYSTEM};
  const auto operations = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  iree_hal_pool_family_access_t family = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .atomics =
          {
              .queue_zero_compute = {operations, operations, operations,
                                     operations},
          },
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                           allocator_, &pool));
  EXPECT_EQ(native_.create_count, 0);
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  EXPECT_EQ(iree_hal_pool_query_zero_compute_atomic_operations(pool, queue)
                .system_scope_64,
            operations);
  EXPECT_EQ(iree_hal_pool_query_atomic_operations(pool, queue).system_scope_64,
            operations);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(8, 8),
      iree_infinite_timeout(), &buffer));
  ASSERT_EQ(native_.created_accesses.size(), 1u);
  EXPECT_TRUE(iree_all_bits_set(
      native_.created_accesses[0].requirements.flags,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();

  // The same guarantee is unavailable through a CPU worker.
  family.family = iree_hal_device_queue_family(cpu_device_, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                                  allocator_, &pool));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(MemoryBackendTest,
       ShaderOperationsCannotSatisfyZeroComputeRequirements) {
  native_.required_flags = 0;
  native_families_[0].program_atomics = true;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.program_atomics[0] = {
      AMDF_ATOMIC_OPERATION_WAIT, AMDF_ATOMIC_OPERATION_WAIT,
      AMDF_ATOMIC_OPERATION_WAIT, AMDF_ATOMIC_OPERATION_WAIT};
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    iree_hal_pool_family_access_t family = {
        .family = families_[0],
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    family.atomics.queue_zero_compute.*cell =
        IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
    };
    iree_hal_pool_t* pool = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_slab_pool_create(group_, scope, &pool_options_, allocator_,
                                  &pool));
    EXPECT_EQ(pool, nullptr);
  }
  EXPECT_EQ(native_.create_count, 0);
}

TEST_F(MemoryBackendTest, AtomicGuaranteesFollowSitesAndEveryBackingRoute) {
  native_.registration_enabled = true;
  native_.host_mapping_features |= AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.atomics[0] = {
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE |
          AMDF_ATOMIC_OPERATION_ADD,
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE};
  native_.registration_atomics[0] = {
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE,
      AMDF_ATOMIC_OPERATION_STORE};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM,
                             AMDF_ATOMIC_SCOPE_SYSTEM};
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = families_[0],
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
          .atomics =
              {
                  .queue_zero_compute =
                      {
                          .system_scope_32 =
                              IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
                          .system_scope_64 =
                              IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                      },
              },
      },
      {
          .family = families_[1],
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      },
      {
          .family = cpu_family,
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
          .atomics =
              {
                  .program =
                      {
                          .system_scope_32 =
                              IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
                      },
              },
      },
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = IREE_ARRAYSIZE(families),
      .families = families,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
              .atomics =
                  {
                      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                  },
          },
  };
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  pool_options_.imports = {1, &import_class};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                           allocator_, &pool));
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto npu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[1]);
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto cpu = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, cpu_family);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST, nullptr);
  const auto operations = iree_hal_pool_query_atomic_operations(pool, gpu);
  EXPECT_EQ(operations.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(operations.system_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  const auto zero_compute =
      iree_hal_pool_query_zero_compute_atomic_operations(pool, gpu);
  EXPECT_EQ(zero_compute.system_scope_32, operations.system_scope_32);
  EXPECT_EQ(zero_compute.system_scope_64, operations.system_scope_64);
  for (auto site : {npu, program, cpu, host}) {
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        iree_hal_pool_query_zero_compute_atomic_operations(pool, site)));
  }
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_pool_query_atomic_operations(pool, npu)));
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_pool_query_atomic_operations(pool, program)));
  EXPECT_EQ(iree_hal_pool_query_atomic_operations(pool, cpu).system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(iree_hal_pool_query_atomic_operations(pool, host).system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_pool_query_atomic_operations(
          pool, {gpu.domain, IREE_HAL_MEMORY_SCOPE_ANY})));

  iree_hal_buffer_t* owned = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(256, 64),
      iree_infinite_timeout(), &owned));
  alignas(4096) uint8_t bytes[4096] = {};
  const iree_hal_external_storage_t source = {
      .storage_class = import_class.storage_class,
      .payload = {.host_pointer = bytes},
      .byte_length = sizeof(bytes),
      .host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                          IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
      .maintenance_range = {0, sizeof(bytes)},
  };
  auto storage = source;
  iree_hal_buffer_t* imported = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(128, 256), &imported));
  const int query_count = native_.pair_query_count;
  for (auto* buffer : {owned, imported}) {
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 64, 64, allocator_, &view));
    EXPECT_EQ(
        iree_hal_buffer_query_atomic_operations(view, gpu).system_scope_64,
        operations.system_scope_64);
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        iree_hal_buffer_query_atomic_operations(view, npu)));
    EXPECT_EQ(iree_hal_buffer_query_zero_compute_atomic_operations(view, gpu)
                  .system_scope_64,
              zero_compute.system_scope_64);
    iree_hal_buffer_release(view);
    iree_hal_buffer_release(buffer);
  }
  EXPECT_EQ(native_.pair_query_count, query_count);
  native_.AwaitEmpty();
  native_.attachment_atomics_supported = false;
  storage = source;
  imported = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_import_buffer(
          pool, &storage, iree_hal_make_buffer_range(0, 64), &imported));
  EXPECT_EQ(imported, nullptr);
  EXPECT_EQ(storage.payload.host_pointer, bytes);
  native_.attachment_atomics_supported = true;
  native_.host_mapping_features &= ~AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
  storage = source;
  const int map_count = native_.map_count;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_import_buffer(
          pool, &storage, iree_hal_make_buffer_range(0, 64), &imported));
  EXPECT_EQ(imported, nullptr);
  EXPECT_EQ(storage.payload.host_pointer, bytes);
  EXPECT_EQ(native_.map_count, map_count + 1);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
}

TEST_F(MemoryBackendTest, AtomicReachIsQualifiedIndependentlyForEachWidth) {
  native_.required_flags = 0;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.atomics[0] = {AMDF_ATOMIC_OPERATION_WAIT,
                        AMDF_ATOMIC_OPERATION_STORE};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_DEVICE,
                             AMDF_ATOMIC_SCOPE_SYSTEM};
  iree_hal_pool_t* pool = nullptr;
  const iree_hal_atomic_operation_capabilities_t unsupported = {
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                 IREE_HAL_MEMORY_ACCESS_NONE, 0, unsupported));
  EXPECT_EQ(pool, nullptr);
  const iree_hal_atomic_operation_capabilities_t supported = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
  };
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                            IREE_HAL_MEMORY_ACCESS_NONE, 0, supported));
  const auto operations = iree_hal_pool_query_atomic_operations(
      pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]));
  EXPECT_EQ(operations.device_scope_32, IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(operations.system_scope_32, 0u);
  EXPECT_EQ(operations.device_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(operations.system_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, ModularSubtractUsesNativeAddOnControlStorage) {
  native_.required_flags = 0;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.atomics[0] = {AMDF_ATOMIC_OPERATION_ADD, AMDF_ATOMIC_OPERATION_ADD};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM,
                             AMDF_ATOMIC_SCOPE_SYSTEM};
  const iree_hal_atomic_operation_capabilities_t required = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
      .device_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                            IREE_HAL_MEMORY_ACCESS_NONE, 0, required));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_support(
      iree_hal_buffer_query_atomic_operations(
          buffer, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0])),
      required));
  ASSERT_EQ(native_.created_accesses.size(), 2u);
  EXPECT_EQ(native_.created_accesses[0].device, native_.devices[0]);
  EXPECT_EQ(native_.created_accesses[0].requirements.flags,
            AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT |
                AMDF_MEMORY_FLAG_QUEUE_STORAGE);
  EXPECT_EQ(native_.created_accesses[1].requirements.flags,
            AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();

  // A raw subtract instruction does not qualify our add-based implementation.
  native_.atomics[0] = {AMDF_ATOMIC_OPERATION_SUBTRACT,
                        AMDF_ATOMIC_OPERATION_SUBTRACT};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                                   IREE_HAL_MEMORY_ACCESS_NONE, 0, required));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(MemoryBackendTest, AtomicWaitAndStoreAcceptOneWayUsage) {
  native_.required_flags = 0;
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  native_.atomics[0] = {
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE, 0};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM, AMDF_ATOMIC_SCOPE_NONE};
  for (const auto usage : {IREE_HAL_BUFFER_USAGE_STORAGE_READ,
                           IREE_HAL_BUFFER_USAGE_STORAGE_WRITE}) {
    SCOPED_TRACE(usage);
    const iree_hal_atomic_operation_flags_t operation =
        usage == IREE_HAL_BUFFER_USAGE_STORAGE_READ
            ? IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT
            : IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    const iree_hal_atomic_operation_capabilities_t required = {
        .system_scope_32 = operation,
    };
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(
        CreatePool(usage, &pool, 0, IREE_HAL_MEMORY_ACCESS_NONE, 0, required));
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_support(
        iree_hal_pool_query_atomic_operations(
            pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0])),
        required));
    iree_hal_pool_release(pool);
  }
}

TEST_F(MemoryBackendTest, ShaderOperationsPreserveScopeWidthAndPhysicalRecipe) {
  native_.required_flags = 0;
  native_.program_atomics[0] = {
      .device_scope_32 =
          AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
      .device_scope_64 = AMDF_ATOMIC_OPERATION_STORE,
      .system_scope_32 = AMDF_ATOMIC_OPERATION_LOAD,
      .system_scope_64 = AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
  };
  // An engine that cannot issue queue atomics can still run atomic programs.
  native_.program_atomics[1].device_scope_64 = AMDF_ATOMIC_OPERATION_STORE;
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = families_[0],
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
          .atomics =
              {
                  .program =
                      {
                          .device_scope_32 =
                              IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND,
                      },
              },
      },
      {
          .family = families_[1],
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
          .atomics =
              {
                  .program =
                      {
                          .device_scope_64 =
                              IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                      },
              },
      },
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = IREE_ARRAYSIZE(families),
      .families = families,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                           allocator_, &pool));
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto operations = iree_hal_pool_query_atomic_operations(pool, program);
  EXPECT_EQ(operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL &
                ~IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(operations.device_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(operations.system_scope_32, IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  // CAS alone cannot implement our recipe: it also reads the initial value.
  EXPECT_EQ(operations.system_scope_64, 0u);
  for (const auto* family : families_) {
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        iree_hal_pool_query_atomic_operations(
            pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, family))));
  }
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(256, 64),
      iree_infinite_timeout(), &buffer));
  for (const auto& access : native_.created_accesses) {
    EXPECT_EQ(access.requirements.flags, AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  }
  const int query_count = native_.pair_query_count;
  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 64, 64, allocator_, &subspan));
  EXPECT_EQ(
      iree_hal_buffer_query_atomic_operations(subspan, program).device_scope_32,
      operations.device_scope_32);
  EXPECT_EQ(native_.pair_query_count, query_count);
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, QueueShadersPreserveOrdinaryBackingAndCpSubset) {
  native_.required_flags = 0;
  native_families_[0].program_atomics = true;
  native_.program_atomics[0] = {
      .device_scope_32 = AMDF_ATOMIC_OPERATION_LOAD |
                         AMDF_ATOMIC_OPERATION_STORE |
                         AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
      .device_scope_64 = AMDF_ATOMIC_OPERATION_STORE,
  };
  const iree_hal_atomic_operation_capabilities_t required = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      .device_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                            IREE_HAL_MEMORY_ACCESS_NONE, 0, required));
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_support(
      iree_hal_pool_query_atomic_operations(pool, queue), required));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(256, 64),
      iree_infinite_timeout(), &buffer));
  for (const auto& access : native_.created_accesses) {
    EXPECT_EQ(access.requirements.flags, AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  }
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_zero_compute_atomic_operations(buffer, queue)));
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest,
       ShaderGuaranteesCoverEveryImportAndAchievedAttachment) {
  native_.registration_enabled = true;
  native_.access_flags[0] = AMDF_MEMORY_FLAG_HOST_COHERENT;
  native_.program_atomics[0] = {
      .device_scope_32 = AMDF_ATOMIC_OPERATION_LOAD |
                         AMDF_ATOMIC_OPERATION_STORE |
                         AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
      .system_scope_64 = AMDF_ATOMIC_OPERATION_STORE,
  };
  native_.registration_program_atomics[0] = {
      .device_scope_32 =
          AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_STORE,
      .system_scope_64 = AMDF_ATOMIC_OPERATION_STORE,
  };
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  pool_options_.imports = {1, &import_class};
  iree_hal_pool_family_access_t family = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .atomics =
          {
              .program =
                  {
                      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                  },
          },
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                           allocator_, &pool));
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  EXPECT_EQ(
      iree_hal_pool_query_atomic_operations(pool, program).device_scope_32,
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  alignas(4096) uint8_t bytes[4096] = {};
  const iree_hal_external_storage_t source = {
      .storage_class = import_class.storage_class,
      .payload = {.host_pointer = bytes},
      .byte_length = sizeof(bytes),
      .host_properties = IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                         IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK,
      .maintenance_range = {0, sizeof(bytes)},
  };
  auto storage = source;
  iree_hal_buffer_t* imported = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(128, 256), &imported));
  EXPECT_EQ(iree_hal_buffer_query_atomic_operations(imported, program)
                .system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  iree_hal_buffer_release(imported);
  native_.AwaitEmpty();
  native_.attachment_atomics_supported = false;
  storage = source;
  imported = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_import_buffer(
          pool, &storage, iree_hal_make_buffer_range(128, 256), &imported));
  EXPECT_EQ(imported, nullptr);
  EXPECT_EQ(storage.payload.host_pointer, bytes);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  // Requesting an operation absent on one selected route rejects the pool.
  family.atomics.program.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                                  allocator_, &pool));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(MemoryBackendTest, OneWayShaderAccessNeedsNoReadWriteSelfPair) {
  native_.required_flags = 0;
  for (const iree_hal_buffer_usage_t usage :
       {IREE_HAL_BUFFER_USAGE_STORAGE_READ,
        IREE_HAL_BUFFER_USAGE_STORAGE_WRITE}) {
    const bool read = usage == IREE_HAL_BUFFER_USAGE_STORAGE_READ;
    const amdf_atomic_operations_t native_operation =
        read ? AMDF_ATOMIC_OPERATION_LOAD : AMDF_ATOMIC_OPERATION_STORE;
    const iree_hal_atomic_operation_flags_t operation =
        read ? IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT
             : IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    native_.program_atomics[0] = {
        .device_scope_32 = native_operation,
    };
    const iree_hal_pool_family_access_t family = {
        .family = families_[0],
        .usage = usage,
        .atomics =
            {
                .program =
                    {
                        .device_scope_32 = operation,
                    },
            },
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
    };
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                             allocator_, &pool));
    const int queries = native_.pair_query_count;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(64, 64),
        iree_infinite_timeout(), &buffer));
    EXPECT_EQ(native_.pair_query_count, queries);
    EXPECT_EQ(
        iree_hal_buffer_query_atomic_operations(
            buffer, ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]))
            .device_scope_32,
        family.atomics.program.device_scope_32);
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(pool);
    native_.AwaitEmpty();
  }
}

TEST_F(MemoryBackendTest, SystemAtomicRequestsQualifyCoherentAttachment) {
  native_.required_flags = 0;
  native_families_[0].program_atomics = true;
  native_.access_flags[0] = AMDF_MEMORY_FLAG_HOST_COHERENT;
  native_.program_atomics[0].system_scope_32 = AMDF_ATOMIC_OPERATION_STORE;
  for (bool program : {false, true}) {
    iree_hal_pool_family_access_t family = {
        .family = families_[0],
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    (program ? family.atomics.program : family.atomics.queue).system_scope_32 =
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, {1, &family, {}}, &pool_options_, allocator_, &pool));
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(64, 64),
        iree_infinite_timeout(), &buffer));
    ASSERT_EQ(native_.created_accesses.size(), 1u);
    EXPECT_EQ(native_.created_accesses[0].requirements.flags,
              AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT);
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(pool);
    native_.AwaitEmpty();
  }
}

TEST_F(MemoryBackendTest, ExactShaderPairsKeepCpuRmwAndNpuAccessSeparate) {
  native_families_[0].program_atomics = true;
  native_.host_mapping_features |= AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
  const auto transactions = AMDF_ATOMIC_OPERATION_LOAD |
                            AMDF_ATOMIC_OPERATION_STORE |
                            AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
  native_.program_atomics[0] = {
      .device_scope_32 = transactions,
      .device_scope_64 = transactions,
      .system_scope_32 = transactions,
      .system_scope_64 = transactions,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                            IREE_HAL_MEMORY_ACCESS_ALL,
                            IREE_HAL_BUFFER_USAGE_STORAGE));
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto npu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[1]);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto cpu = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM,
                                iree_hal_device_queue_family(cpu_device_, 0));
  for (auto producer : {queue, program}) {
    const auto pair = PairInfo(pool->memory_contract, producer, program);
    for (auto cell :
         {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
          &iree_hal_atomic_operation_capabilities_t::device_scope_64,
          &iree_hal_atomic_operation_capabilities_t::system_scope_32,
          &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
      EXPECT_EQ(pair.producer_atomics.*cell,
                IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
      EXPECT_EQ(pair.consumer_atomics.*cell,
                IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
    }
    EXPECT_EQ(pair.atomic_reach.scope_32, IREE_HAL_ATOMIC_REACH_SYSTEM);
    const auto npu_pair = PairInfo(pool->memory_contract, producer, npu);
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        npu_pair.producer_atomics));
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        npu_pair.consumer_atomics));
    for (auto host_site : {host, cpu}) {
      const auto to_gpu = PairInfo(pool->memory_contract, host_site, producer);
      const auto from_gpu =
          PairInfo(pool->memory_contract, producer, host_site);
      EXPECT_EQ(to_gpu.producer_atomics.system_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                    IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
      EXPECT_EQ(from_gpu.consumer_atomics.system_scope_64,
                IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                    IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
      EXPECT_EQ(to_gpu.consumer_atomics.system_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
      EXPECT_EQ(from_gpu.producer_atomics.system_scope_64,
                IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
      EXPECT_EQ(to_gpu.producer_atomics.device_scope_32, 0u);
      EXPECT_EQ(from_gpu.consumer_atomics.device_scope_64, 0u);
      // CPU RMW exists at this site, but not against a GPU transaction.
      EXPECT_EQ(iree_hal_pool_query_atomic_operations(pool, host_site)
                    .system_scope_32,
                IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
    }
  }
  const auto cpu_pair = PairInfo(pool->memory_contract, host, cpu);
  EXPECT_EQ(cpu_pair.producer_atomics.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(cpu_pair.consumer_atomics.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  const int queries = native_.pair_query_count;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  const auto buffer_pair =
      PairInfo(iree_hal_buffer_memory_view(buffer).contract, program, host);
  EXPECT_EQ(buffer_pair.consumer_atomics.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(native_.pair_query_count, queries);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, QueuePairsQualifyEverySelectedExecutorAndWidth) {
  native_families_[0].program_atomics = true;
  native_.atomics[0] = {AMDF_ATOMIC_OPERATION_STORE, 0};
  native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM, AMDF_ATOMIC_SCOPE_NONE};
  const auto transactions = AMDF_ATOMIC_OPERATION_LOAD |
                            AMDF_ATOMIC_OPERATION_STORE |
                            AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
  native_.program_atomics[0] = {
      .device_scope_32 = transactions,
      .device_scope_64 = transactions,
      .system_scope_32 = transactions,
      .system_scope_64 = transactions,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto pair = PairInfo(pool->memory_contract, queue, queue);
  EXPECT_EQ(pair.producer_atomics.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(pair.consumer_atomics.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  // Native SYSTEM/SYSTEM does not prove CP SYSTEM against shader DEVICE.
  EXPECT_EQ(pair.producer_atomics.device_scope_32, 0u);
  EXPECT_EQ(pair.consumer_atomics.device_scope_32, 0u);
  // The 32-bit CP selection cannot remove a 64-bit shader-only relationship.
  EXPECT_EQ(pair.producer_atomics.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(pair.consumer_atomics.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  const auto shader_pair = PairInfo(pool->memory_contract, program, program);
  EXPECT_EQ(shader_pair.producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(shader_pair.consumer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, ExactPairIntersectsRoutesWithoutChangingSiteSupport) {
  native_.registration_enabled = true;
  native_families_[0].program_atomics = true;
  const auto transactions = AMDF_ATOMIC_OPERATION_LOAD |
                            AMDF_ATOMIC_OPERATION_STORE |
                            AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
  native_.program_atomics[0] = native_.registration_program_atomics[0] = {
      .device_scope_32 = transactions,
      .system_scope_64 = transactions,
  };
  native_.api.memory_scope_query_pair_info =
      [](amdf_memory_scope_t* scope,
         const amdf_memory_profile_pair_query_t* query,
         amdf_memory_pair_info_t* info) -> amdf_status_t {
    auto status = FakeNative::MemoryScopeQueryPairInfo(scope, query, info);
    if (amdf_status_is_ok(status) && info->next &&
        query->memory_profile_ordinal == 1) {
      auto* pair = static_cast<amdf_memory_pair_atomics_t*>(info->next);
      pair->producer.device_scope_32 &= AMDF_ATOMIC_OPERATION_STORE;
      pair->consumer.device_scope_32 &= AMDF_ATOMIC_OPERATION_LOAD;
      pair->producer.system_scope_64 &= AMDF_ATOMIC_OPERATION_LOAD;
      pair->consumer.system_scope_64 &= AMDF_ATOMIC_OPERATION_STORE;
    }
    return status;
  };
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  pool_options_.imports = {1, &import_class};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  const auto program = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  EXPECT_EQ(
      iree_hal_pool_query_atomic_operations(pool, program).device_scope_32,
      IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  const auto pair = PairInfo(pool->memory_contract, program, program);
  EXPECT_EQ(pair.producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(pair.consumer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(pair.producer_atomics.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(pair.consumer_atomics.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  iree_hal_pool_release(pool);
}

}  // namespace iree::hal::amd::fabric
