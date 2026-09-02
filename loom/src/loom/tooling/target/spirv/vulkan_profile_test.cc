// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/vulkan_profile.h"

#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"

namespace loom {
namespace {

typedef struct fake_hal_device_t {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;
  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} fake_hal_device_t;

static const iree_hal_device_spec_t* fake_hal_device_spec(
    iree_hal_device_t* base_device) {
  fake_hal_device_t* device = (fake_hal_device_t*)base_device;
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.device_spec = fake_hal_device_spec;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

static void InitializeFakeHalDevice(const iree_hal_device_spec_t* device_spec,
                                    fake_hal_device_t* out_device) {
  out_device->device_spec = device_spec;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &out_device->resource);
}

static iree_hal_vulkan_features_t BaselineVulkanFeatures() {
  return IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL |
         IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
         IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT |
         IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL;
}

static iree_hal_vulkan_features_t AtomicVulkanFeatures() {
  return IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_INT64_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_INT64_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT16_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT16_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT16_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT16_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT32_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT32_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT32_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT32_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT64_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_FLOAT64_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT64_ATOMICS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SHARED_FLOAT64_ATOMIC_ADD |
         IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;
}

static iree_status_t CreateDeviceSpec(
    iree_hal_vulkan_features_t enabled_features, bool include_target,
    iree_host_size_t cooperative_matrix_property_count,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties,
    iree_hal_device_spec_t** out_device_spec) {
  *out_device_spec = NULL;
  iree_hal_device_dispatch_spec_t dispatch = {
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/256,
          /*.maximum_workgroup_size=*/{256, 128, 64},
          /*.maximum_workgroup_count=*/{65535, 65535, 65535},
      },
      /*.subgroup=*/
      {
          /*.default_size=*/32,
          /*.minimum_size=*/32,
          /*.maximum_size=*/32,
          /*.supported_size_mask=*/1ull << 32,
      },
      /*.execution=*/
      {
          /*.unit_count=*/1,
          /*.group_count=*/1,
      },
      /*.addressing=*/
      {
          /*.pointer_size_bits=*/64,
          /*.address_space_bits=*/64,
      },
      /*.flags=*/IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  iree_hal_vulkan_device_spec_t vulkan_spec = {
      /*.api_version=*/LOOM_SPIRV_VULKAN_API_VERSION_1_3,
      /*.driver_version=*/1,
      /*.physical_device_type=*/2,
      /*.enabled_features=*/enabled_features,
      /*.flags=*/IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };
  iree_host_size_t vulkan_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      cooperative_matrix_property_count, &vulkan_payload_size));
  std::vector<uint8_t> vulkan_payload_storage(vulkan_payload_size);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_encode(
      &vulkan_spec, cooperative_matrix_property_count,
      cooperative_matrix_properties,
      iree_make_byte_span(vulkan_payload_storage.data(),
                          vulkan_payload_storage.size())));
  iree_hal_device_spec_facet_t vulkan_facet = {
      /*.schema_id=*/
      iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
      /*.schema_version=*/IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
      /*.payload=*/
      iree_make_const_byte_span(vulkan_payload_storage.data(),
                                vulkan_payload_storage.size()),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/include_target ? 1u : 0u,
      /*.targets=*/include_target ? &executable_target : nullptr,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status) && include_target) {
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_add_facet(&builder, &vulkan_facet);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

static loom_spirv_vulkan_hal_profile_facts_t BaselineFacts() {
  return loom_spirv_vulkan_hal_profile_facts_t{
      /*.api_version=*/LOOM_SPIRV_VULKAN_API_VERSION_1_3,
      /*.flags=*/LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE |
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS |
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
      /*.subgroup_size=*/32,
      /*.max_compute_workgroup_invocations=*/256,
      /*.max_compute_workgroup_size=*/
      {
          /*.x=*/256,
          /*.y=*/128,
          /*.z=*/64,
      },
      /*.max_compute_workgroup_count=*/
      {
          /*.x=*/65535,
          /*.y=*/65535,
          /*.z=*/65535,
      },
  };
}

static loom_spirv_cooperative_matrix_query_t F16MatrixQuery() {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/16,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_F16,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_F16,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/{},
      /*.policy=*/LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
}

static iree_hal_vulkan_cooperative_matrix_property_t F16DeviceMatrixRow() {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/16,
      /*.a_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      /*.b_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      /*.c_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      /*.result_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      /*.saturating_accumulation=*/{},
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
  };
}

static loom_spirv_cooperative_matrix_query_t U8MatrixQuery() {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/32,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_U8,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_U8,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_U32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_U32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/{},
      /*.policy=*/LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
}

static iree_hal_vulkan_cooperative_matrix_property_t U8DeviceMatrixRow() {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/32,
      /*.a_type=*/LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV,
      /*.b_type=*/LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV,
      /*.c_type=*/LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      /*.result_type=*/LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      /*.saturating_accumulation=*/{},
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
  };
}

TEST(VulkanProfileTest, QueryReadsHalDeviceFacts) {
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateDeviceSpec(BaselineVulkanFeatures(),
                                  /*include_target=*/true,
                                  /*cooperative_matrix_property_count=*/0,
                                  /*cooperative_matrix_properties=*/nullptr,
                                  &device_spec));
  fake_hal_device_t device = {};
  InitializeFakeHalDevice(device_spec, &device);

  loom_spirv_vulkan_hal_profile_facts_t facts = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_query((iree_hal_device_t*)&device, &facts));

  EXPECT_EQ(facts.api_version, LOOM_SPIRV_VULKAN_API_VERSION_1_3);
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SUBGROUP_SIZE_CONTROL));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_16BIT_ACCESS));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16));
  EXPECT_FALSE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT));
  EXPECT_FALSE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT16));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INTEGER_DOT_PRODUCT));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL));
  EXPECT_FALSE(iree_all_bits_set(
      facts.flags,
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
  EXPECT_EQ(facts.subgroup_size, 32u);
  EXPECT_EQ(facts.max_compute_workgroup_invocations, 256u);
  EXPECT_EQ(facts.max_compute_workgroup_size.x, 256u);
  EXPECT_EQ(facts.max_compute_workgroup_size.y, 128u);
  EXPECT_EQ(facts.max_compute_workgroup_size.z, 64u);
  EXPECT_EQ(facts.max_compute_workgroup_count.x, 65535u);
  EXPECT_EQ(facts.max_compute_workgroup_count.y, 65535u);
  EXPECT_EQ(facts.max_compute_workgroup_count.z, 65535u);
  iree_hal_device_spec_release(device_spec);
}

TEST(VulkanProfileTest, QueryKeepsExecutableTargetSupportSeparate) {
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateDeviceSpec(BaselineVulkanFeatures(),
                                  /*include_target=*/false,
                                  /*cooperative_matrix_property_count=*/0,
                                  /*cooperative_matrix_properties=*/nullptr,
                                  &device_spec));
  fake_hal_device_t device = {};
  InitializeFakeHalDevice(device_spec, &device);

  loom_spirv_vulkan_hal_profile_facts_t facts = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_query((iree_hal_device_t*)&device, &facts));

  EXPECT_FALSE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS));
  iree_hal_device_spec_release(device_spec);
}

TEST(VulkanProfileTest, QueryProjectsAtomicFeatures) {
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateDeviceSpec(
      BaselineVulkanFeatures() | IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64 |
          AtomicVulkanFeatures(),
      /*include_target=*/true,
      /*cooperative_matrix_property_count=*/0,
      /*cooperative_matrix_properties=*/nullptr, &device_spec));
  fake_hal_device_t device = {};
  InitializeFakeHalDevice(device_spec, &device);

  loom_spirv_vulkan_hal_profile_facts_t facts = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_query((iree_hal_device_t*)&device, &facts));

  const loom_spirv_vulkan_hal_profile_flags_t expected_flags =
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_INT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_INT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;
  EXPECT_TRUE(iree_all_bits_set(facts.flags, expected_flags));
  iree_hal_device_spec_release(device_spec);
}

TEST(VulkanProfileTest, CopiesCooperativeMatrixRowsFromDeviceSpec) {
  const iree_hal_vulkan_cooperative_matrix_property_t source_row =
      F16DeviceMatrixRow();
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateDeviceSpec(BaselineVulkanFeatures(),
                                  /*include_target=*/true,
                                  /*cooperative_matrix_property_count=*/1,
                                  &source_row, &device_spec));
  fake_hal_device_t device = {};
  InitializeFakeHalDevice(device_spec, &device);

  iree_hal_vulkan_cooperative_matrix_property_t* copied_rows = nullptr;
  iree_host_size_t copied_row_count = 0;
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
      (iree_hal_device_t*)&device, iree_allocator_system(), &copied_rows,
      &copied_row_count));

  ASSERT_EQ(copied_row_count, 1);
  ASSERT_NE(copied_rows, nullptr);
  EXPECT_EQ(copied_rows[0].m_size, source_row.m_size);
  EXPECT_EQ(copied_rows[0].k_size, source_row.k_size);
  EXPECT_EQ(copied_rows[0].result_type, source_row.result_type);
  iree_allocator_free(iree_allocator_system(), copied_rows);
  iree_hal_device_spec_release(device_spec);
}

TEST(VulkanProfileTest, MaterializesRawBdaHalKernelTarget) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  loom_target_bundle_storage_t storage = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));

  EXPECT_TRUE(iree_string_view_equal(storage.bundle.name,
                                     IREE_SV("spirv-vulkan1.3-bda-hal")));
  EXPECT_TRUE(iree_string_view_equal(storage.snapshot.name,
                                     IREE_SV("spirv-vulkan1.3-bda")));
  EXPECT_TRUE(storage.bundle.snapshot == &storage.snapshot);
  EXPECT_TRUE(storage.bundle.export_plan == &storage.export_plan);
  EXPECT_TRUE(storage.bundle.config == &storage.config);
  EXPECT_EQ(storage.snapshot.codegen_format, LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(storage.snapshot.default_pointer_bitwidth, 64u);
  EXPECT_EQ(storage.snapshot.index_bitwidth, 32u);
  EXPECT_EQ(storage.snapshot.offset_bitwidth, 64u);
  EXPECT_EQ(storage.snapshot.subgroup_size, 32u);
  EXPECT_EQ(storage.snapshot.max_flat_workgroup_size, 256u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.x, 256u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.y, 128u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.z, 64u);
  EXPECT_EQ(storage.snapshot.max_workgroup_count.x, 65535u);
  EXPECT_EQ(storage.snapshot.max_workgroup_count.y, 65535u);
  EXPECT_EQ(storage.snapshot.max_workgroup_count.z, 65535u);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(iree_string_view_equal(storage.export_plan.name,
                                     IREE_SV("spirv-hal-kernel")));
  EXPECT_TRUE(iree_string_view_equal(
      storage.config.name, IREE_SV("spirv.logical.core.vulkan1.3.bda")));
  EXPECT_EQ(storage.config.contract_feature_bits,
            LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA);
}

TEST(VulkanProfileTest, RejectsVulkan12) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.api_version = LOOM_SPIRV_VULKAN_API_VERSION(1, 2, 0);

  loom_target_bundle_storage_t storage = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));
}

TEST(VulkanProfileTest, RejectsMissingRawBdaExecutableFormat) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags &= ~LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE;

  loom_target_bundle_storage_t storage = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));
}

TEST(VulkanProfileTest, RejectsMissingBufferDeviceAddress) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags &= ~LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS;

  loom_target_bundle_storage_t storage = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));
}

TEST(VulkanProfileTest, RejectsMissingShaderInt64) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags &= ~LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64;

  loom_target_bundle_storage_t storage = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));
}

TEST(VulkanProfileTest, MaterializesCooperativeMatrixFeature) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_16BIT_ACCESS;

  loom_target_bundle_storage_t storage = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));

  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA));
  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_FLOAT16));
  EXPECT_TRUE(
      iree_all_bits_set(storage.config.contract_feature_bits,
                        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS));
}

TEST(VulkanProfileTest, MaterializesBfloat16Features) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT;
  facts.flags |=
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX;

  loom_target_bundle_storage_t storage = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));

  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR));
  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR));
  EXPECT_TRUE(
      iree_all_bits_set(storage.config.contract_feature_bits,
                        LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR));
}

TEST(VulkanProfileTest, MaterializesAtomicFeatures) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |=
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16 |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64 |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_INT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_INT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;

  loom_target_bundle_storage_t storage = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));

  const loom_spirv_feature_bits_t expected_features =
      LOOM_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD;
  EXPECT_TRUE(iree_all_bits_set(storage.config.contract_feature_bits,
                                expected_features));
}

TEST(VulkanProfileTest, KeepsDependentBfloat16FeaturesClosed) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT;
  facts.flags |=
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX;

  loom_target_bundle_storage_t storage = {};
  IREE_ASSERT_OK(
      loom_spirv_vulkan_hal_profile_initialize_target_bundle(&facts, &storage));

  EXPECT_FALSE(iree_any_bit_set(storage.config.contract_feature_bits,
                                LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR));
  EXPECT_FALSE(
      iree_any_bit_set(storage.config.contract_feature_bits,
                       LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR));
}

TEST(VulkanProfileTest, KeepsFloatingPointAtomicScalarDependenciesClosed) {
  const loom_spirv_vulkan_hal_profile_flags_t float16_atomic_flags =
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMIC_ADD;
  const loom_spirv_vulkan_hal_profile_flags_t float64_atomic_flags =
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMICS |
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMIC_ADD;
  const loom_spirv_feature_bits_t float16_atomic_features =
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD;
  const loom_spirv_feature_bits_t float64_atomic_features =
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS |
      LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD;

  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64 |
                 float16_atomic_flags | float64_atomic_flags;
  loom_target_bundle_storage_t float64_storage = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_profile_initialize_target_bundle(
      &facts, &float64_storage));
  EXPECT_FALSE(iree_any_bit_set(float64_storage.config.contract_feature_bits,
                                float16_atomic_features));
  EXPECT_TRUE(iree_all_bits_set(float64_storage.config.contract_feature_bits,
                                float64_atomic_features));

  facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16 |
                 float16_atomic_flags | float64_atomic_flags;
  loom_target_bundle_storage_t float16_storage = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_profile_initialize_target_bundle(
      &facts, &float16_storage));
  EXPECT_TRUE(iree_all_bits_set(float16_storage.config.contract_feature_bits,
                                float16_atomic_features));
  EXPECT_FALSE(iree_any_bit_set(float16_storage.config.contract_feature_bits,
                                float64_atomic_features));
}

TEST(VulkanProfileTest, ImportsExactCooperativeMatrixRows) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16;
  const iree_hal_vulkan_cooperative_matrix_property_t device_rows[] = {
      F16DeviceMatrixRow(),
  };

  loom_spirv_vulkan_hal_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_target_profile_storage_initialize(
      &facts, device_rows, IREE_ARRAYSIZE(device_rows), iree_allocator_system(),
      &storage));

  ASSERT_NE(storage.profile.cooperative_properties, nullptr);
  EXPECT_EQ(storage.profile.cooperative_properties->matrix_property_count, 1u);
  EXPECT_TRUE(
      iree_all_bits_set(storage.profile.cooperative_properties->feature_bits,
                        LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR));
  const loom_spirv_cooperative_matrix_query_t query = F16MatrixQuery();
  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(
          storage.profile.cooperative_properties, &query, &diagnostic);
  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.f16.16x16x16.f32.subgroup")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);

  loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
      &storage, iree_allocator_system());
}

TEST(VulkanProfileTest, ImportsUnsignedCooperativeMatrixRows) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8;
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS;
  const iree_hal_vulkan_cooperative_matrix_property_t device_rows[] = {
      U8DeviceMatrixRow(),
  };

  loom_spirv_vulkan_hal_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_target_profile_storage_initialize(
      &facts, device_rows, IREE_ARRAYSIZE(device_rows), iree_allocator_system(),
      &storage));

  ASSERT_NE(storage.profile.cooperative_properties, nullptr);
  EXPECT_EQ(storage.profile.cooperative_properties->matrix_property_count, 1u);
  EXPECT_TRUE(
      iree_all_bits_set(storage.profile.cooperative_properties->feature_bits,
                        LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(
      iree_all_bits_set(storage.profile.cooperative_properties->feature_bits,
                        LOOM_SPIRV_FEATURE_INT8));
  EXPECT_TRUE(
      iree_all_bits_set(storage.profile.cooperative_properties->feature_bits,
                        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS));
  const loom_spirv_cooperative_matrix_query_t query = U8MatrixQuery();
  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(
          storage.profile.cooperative_properties, &query, &diagnostic);
  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.u8.16x16x32.u32.subgroup")));
  EXPECT_EQ(property->lhs_type, LOOM_SPIRV_SCALAR_TYPE_U8);
  EXPECT_EQ(property->rhs_type, LOOM_SPIRV_SCALAR_TYPE_U8);
  EXPECT_EQ(property->accumulator_type, LOOM_SPIRV_SCALAR_TYPE_U32);
  EXPECT_EQ(property->result_type, LOOM_SPIRV_SCALAR_TYPE_U32);
  EXPECT_EQ(property->operand_flags, 0u);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);

  loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
      &storage, iree_allocator_system());
}

TEST(VulkanProfileTest, RejectsCooperativeMatrixRowsAbsentFromDevice) {
  loom_spirv_vulkan_hal_profile_facts_t facts = BaselineFacts();
  facts.flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR;
  iree_hal_vulkan_cooperative_matrix_property_t mismatched_row =
      F16DeviceMatrixRow();
  mismatched_row.k_size = 8;

  loom_spirv_vulkan_hal_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_target_profile_storage_initialize(
      &facts, &mismatched_row, 1, iree_allocator_system(), &storage));

  ASSERT_NE(storage.profile.cooperative_properties, nullptr);
  EXPECT_EQ(storage.profile.cooperative_properties->matrix_property_count, 0u);
  const loom_spirv_cooperative_matrix_query_t query = F16MatrixQuery();
  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(
                storage.profile.cooperative_properties, &query, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE));

  loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
      &storage, iree_allocator_system());
}

}  // namespace
}  // namespace loom
