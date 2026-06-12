// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/vulkan_profile.h"

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"

namespace loom {
namespace {

typedef struct fake_query_row_t {
  // HAL query category.
  iree_string_view_t category;
  // HAL query key within |category|.
  iree_string_view_t key;
  // Scalar query value returned to the caller.
  int64_t value;
} fake_query_row_t;

typedef struct fake_hal_device_t {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;
  // Query rows exposed by this fake device.
  const fake_query_row_t* rows;
  // Number of entries in |rows|.
  iree_host_size_t row_count;
} fake_hal_device_t;

typedef struct fake_executable_cache_t {
  // HAL resource header used by executable-cache vtable dispatch.
  iree_hal_resource_t resource;
  // Whether this cache accepts the raw Vulkan BDA executable format.
  bool raw_bda_supported;
} fake_executable_cache_t;

static iree_status_t fake_hal_device_query_i64(iree_hal_device_t* base_device,
                                               iree_string_view_t category,
                                               iree_string_view_t key,
                                               int64_t* out_value) {
  fake_hal_device_t* device = (fake_hal_device_t*)base_device;
  for (iree_host_size_t i = 0; i < device->row_count; ++i) {
    if (iree_string_view_equal(category, device->rows[i].category) &&
        iree_string_view_equal(key, device->rows[i].key)) {
      *out_value = device->rows[i].value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static bool fake_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)caching_mode;
  fake_executable_cache_t* executable_cache =
      (fake_executable_cache_t*)base_executable_cache;
  return executable_cache->raw_bda_supported &&
         iree_string_view_equal(executable_format, IREE_SV("vulkan-spirv-bda"));
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.query_i64 = fake_hal_device_query_i64;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

static iree_hal_executable_cache_vtable_t MakeFakeExecutableCacheVtable() {
  iree_hal_executable_cache_vtable_t vtable = {};
  vtable.can_prepare_format = fake_executable_cache_can_prepare_format;
  return vtable;
}

static const iree_hal_executable_cache_vtable_t kFakeExecutableCacheVtable =
    MakeFakeExecutableCacheVtable();

static fake_hal_device_t FakeDevice(const fake_query_row_t* rows,
                                    iree_host_size_t row_count) {
  fake_hal_device_t device = {};
  device.rows = rows;
  device.row_count = row_count;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device.resource);
  return device;
}

static fake_executable_cache_t FakeExecutableCache(bool raw_bda_supported) {
  fake_executable_cache_t executable_cache = {};
  executable_cache.raw_bda_supported = raw_bda_supported;
  iree_hal_resource_initialize(&kFakeExecutableCacheVtable,
                               &executable_cache.resource);
  return executable_cache;
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

static constexpr fake_query_row_t kBaselineQueryRows[] = {
    {IREE_SVL("vulkan.device"), IREE_SVL("api_version"),
     LOOM_SPIRV_VULKAN_API_VERSION_1_3},
    {IREE_SVL("vulkan.device"), IREE_SVL("subgroup_size"), 32},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_invocations"),
     256},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_x"), 256},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_y"), 128},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_z"), 64},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_x"),
     65535},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_y"),
     65535},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_z"),
     65535},
    {IREE_SVL("vulkan.feature"), IREE_SVL("buffer_device_address"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("subgroup_size_control"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("cooperative_matrix_khr"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("storage_buffer_8bit_access"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("storage_buffer_16bit_access"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_float16"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_float64"), 0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_type"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_dot_product"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_cooperative_matrix"),
     0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int8"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int16"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int64"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_integer_dot_product"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("vulkan_memory_model"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("vulkan_memory_model_device_scope"),
     0},
    {IREE_SVL("vulkan.cooperative_matrix"), IREE_SVL("property_count"), 4},
    {IREE_SVL("vulkan.cooperative_matrix"), IREE_SVL("supported_stages"), 0x20},
};

TEST(VulkanProfileTest, QueryReadsHalDeviceAndExecutableCacheFacts) {
  fake_hal_device_t device =
      FakeDevice(kBaselineQueryRows, IREE_ARRAYSIZE(kBaselineQueryRows));
  fake_executable_cache_t executable_cache = FakeExecutableCache(true);

  loom_spirv_vulkan_hal_profile_facts_t facts = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_profile_query(
      (iree_hal_device_t*)&device,
      (iree_hal_executable_cache_t*)&executable_cache, &facts));

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
  EXPECT_EQ(facts.cooperative_matrix_property_count, 4u);
  EXPECT_EQ(facts.cooperative_matrix_supported_stages, 0x20u);
}

TEST(VulkanProfileTest, QueryKeepsRawBdaExecutableSupportSeparate) {
  fake_hal_device_t device =
      FakeDevice(kBaselineQueryRows, IREE_ARRAYSIZE(kBaselineQueryRows));
  fake_executable_cache_t executable_cache = FakeExecutableCache(false);

  loom_spirv_vulkan_hal_profile_facts_t facts = {};
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_profile_query(
      (iree_hal_device_t*)&device,
      (iree_hal_executable_cache_t*)&executable_cache, &facts));

  EXPECT_FALSE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE));
  EXPECT_TRUE(iree_all_bits_set(
      facts.flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS));
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
  facts.cooperative_matrix_property_count = 7;
  facts.cooperative_matrix_supported_stages = 0x20;

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
