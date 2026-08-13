// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/device_spec_builder.h"

#include "iree/hal/local/atomic.h"
#include "iree/hal/local/cpu_device_spec.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

typedef struct test_executable_loader_t {
  iree_hal_executable_loader_t base;
} test_executable_loader_t;

static void test_executable_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {}

static bool test_executable_loader_query_target_support(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_target_t* target) {
  return iree_string_view_equal(target->family, IREE_SV("cpu")) ||
         iree_string_view_equal(target->family, IREE_SV("test-family"));
}

static void test_executable_loader_query_spec(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_device_executable_spec_t* out_executable_spec) {
  static const iree_hal_executable_target_t executable_targets[] = {
      {
          /*.family=*/IREE_SVL("test-family"),
          /*.target_key=*/IREE_SVL("test-target"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/7,
          /*.physical_device_affinity=*/1ull,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  *out_executable_spec = {
      /*.target_count=*/IREE_ARRAYSIZE(executable_targets),
      /*.targets=*/executable_targets,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
}

static bool test_executable_loader_claims_executable(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params) {
  return false;
}

static iree_status_t test_executable_loader_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  return iree_make_status(IREE_STATUS_INTERNAL);
}

static const iree_hal_executable_loader_vtable_t test_executable_loader_vtable =
    {
        /*.destroy=*/test_executable_loader_destroy,
        /*.query_target_support=*/test_executable_loader_query_target_support,
        /*.query_spec=*/test_executable_loader_query_spec,
        /*.claims_executable=*/test_executable_loader_claims_executable,
        /*.load=*/test_executable_loader_load,
};

TEST(LocalDeviceSpecBuilderTest, CapturesCommonLocalFacts) {
  test_executable_loader_t loader;
  iree_hal_executable_loader_initialize(
      &test_executable_loader_vtable,
      iree_hal_executable_import_provider_null(), &loader.base);
  iree_hal_executable_loader_t* loader_ptr = &loader.base;

  iree_hal_local_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("local0"),
      /*.display_name=*/IREE_SV("Local Device"),
      /*.driver_id=*/IREE_SV("local-test"),
      /*.backend_id=*/IREE_SV("local"),
      /*.queue_count=*/2,
      /*.default_queue_worker_count=*/8,
      /*.atomic_capabilities=*/
      {
          /*.operations=*/
          {
              /*.device_scope_32=*/IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
          },
          /*.wait_conditions=*/{},
      },
      /*.zero_compute_atomic_capabilities=*/
      {
          /*.operations=*/
          {
              /*.device_scope_32=*/IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
          },
          /*.wait_conditions=*/{},
      },
      /*.loader_count=*/1,
      /*.loaders=*/&loader_ptr,
  };
  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(iree_hal_local_device_spec_create(
      &params, iree_allocator_system(), &spec));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(identity->logical_device_id, IREE_SV("local0")));
  ASSERT_EQ(identity->physical_device_count, 1);

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(spec);
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->heap_count, 1);
  EXPECT_TRUE(iree_all_bits_set(
      memory->heaps[0].flags,
      IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN |
          IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN));
  ASSERT_EQ(memory->memory_type_count, 1);
  EXPECT_EQ(memory->memory_types[0].memory_type,
            IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE);
  EXPECT_EQ(memory->memory_types[0].allowed_buffer_usage,
            IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE);
  const iree_hal_atomic_operation_capabilities_t expected_memory_operations =
      iree_hal_local_atomic_operation_capabilities(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(memory->memory_types[0].atomic_operations.device_scope_32,
            expected_memory_operations.device_scope_32);
  EXPECT_EQ(memory->memory_types[0].atomic_operations.device_scope_64,
            expected_memory_operations.device_scope_64);
  EXPECT_EQ(memory->memory_types[0].atomic_operations.system_scope_32,
            expected_memory_operations.system_scope_32);
  EXPECT_EQ(memory->memory_types[0].atomic_operations.system_scope_64,
            expected_memory_operations.system_scope_64);

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_count, 2);
  EXPECT_EQ(queues->families[0].queue_affinity, 3u);
  EXPECT_TRUE(iree_all_bits_set(queues->families[0].role_flags,
                                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC));
  EXPECT_EQ(queues->families[0].atomic_capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(queues->families[0]
                .zero_compute_atomic_capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->execution.unit_count, 8);
  EXPECT_EQ(dispatch->subgroup.default_size, 1);

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(spec);
  ASSERT_NE(executables, nullptr);
  iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV("test-family"),
      /*.target_key=*/IREE_SV("test-target"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t selection_result =
      iree_hal_device_spec_select_executable_target(spec, &target_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            selection_result.outcome);
  ASSERT_NE(selection_result.target, nullptr);

  target_selection = {
      /*.family=*/IREE_SV("cpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t cpu_selection_result =
      iree_hal_device_spec_select_executable_target(spec, &target_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            cpu_selection_result.outcome);
  ASSERT_NE(cpu_selection_result.target, nullptr);

  const iree_hal_device_spec_facet_t* cpu_facet =
      iree_hal_cpu_device_spec_find_facet(spec);
  ASSERT_NE(cpu_facet, nullptr);
  iree_hal_cpu_device_spec_t cpu_spec = {};
  IREE_ASSERT_OK(iree_hal_cpu_device_spec_decode_facet(cpu_facet, &cpu_spec));
  iree_cpu_data_t target_cpu_data = {};
  IREE_ASSERT_OK(iree_cpu_data_parse_target_key(
      cpu_selection_result.target->target_key, &target_cpu_data));
  EXPECT_TRUE(
      iree_cpu_data_satisfies_features(&cpu_spec.cpu_data, &target_cpu_data));

  iree_string_builder_t target_key_builder;
  iree_string_builder_initialize(iree_allocator_system(), &target_key_builder);
  IREE_ASSERT_OK(
      iree_cpu_data_append_target_key(&cpu_spec.cpu_data, &target_key_builder));
  EXPECT_TRUE(
      iree_string_view_equal(cpu_selection_result.target->target_key,
                             iree_string_builder_view(&target_key_builder)));
  iree_string_builder_deinitialize(&target_key_builder);

  iree_hal_device_spec_release(spec);
  iree_hal_executable_loader_release(&loader.base);
}

}  // namespace
}  // namespace iree::hal
