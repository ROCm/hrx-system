// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/reference.h"

#include <cstring>
#include <vector>

#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

template <typename T>
struct BufferContents {
  const T* values;
  iree_host_size_t count;
};

template <typename T>
static iree_status_t FillBufferView(iree_hal_buffer_mapping_t* mapping,
                                    void* user_data) {
  const BufferContents<T>* contents =
      static_cast<const BufferContents<T>*>(user_data);
  const iree_host_size_t byte_count =
      contents->count * sizeof(contents->values[0]);
  memcpy(mapping->contents.data, contents->values, byte_count);
  return iree_ok_status();
}

class ReferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, host_allocator_, &block_pool_);
    loom_context_initialize(host_allocator_, &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("reference_test"),
                                        &block_pool_, nullptr, host_allocator_,
                                        &module_));
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           host_allocator_, &vm_instance_));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(vm_instance_));
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("testbench"), host_allocator_,
                                       host_allocator_, &device_allocator_));
    reference_options_ = {
        /*.device=*/{},
        /*.device_allocator=*/device_allocator_,
        /*.result_buffer_params=*/BufferParams(),
        /*.host_allocator=*/host_allocator_,
    };
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_vm_instance_release(vm_instance_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_hal_buffer_params_t BufferParams() {
    return iree_hal_buffer_params_t{
        /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT |
            IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
    };
  }

  template <typename T>
  iree_vm_variant_t MakeBufferView(std::vector<iree_hal_dim_t> shape,
                                   iree_hal_element_type_t element_type,
                                   const std::vector<T>& values) {
    BufferContents<T> contents = {
        /*.values=*/values.data(),
        /*.count=*/values.size(),
    };
    iree_hal_buffer_view_t* buffer_view = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_generate_buffer(
        /*device=*/nullptr, device_allocator_, shape.size(), shape.data(),
        element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, BufferParams(),
        FillBufferView<T>, &contents, &buffer_view));
    iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
    return iree_vm_make_variant_ref_assign(buffer_view_ref);
  }

  void ExpectF32BufferView(const iree_vm_variant_t& variant,
                           const std::vector<iree_hal_dim_t>& expected_shape,
                           std::vector<float> expected_values) {
    ASSERT_TRUE(iree_vm_variant_is_ref(variant));
    iree_hal_buffer_view_t* buffer_view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_view_check_deref(variant.ref, &buffer_view));
    ASSERT_NE(buffer_view, nullptr);
    EXPECT_EQ(iree_hal_buffer_view_shape_rank(buffer_view),
              expected_shape.size());
    for (iree_host_size_t i = 0; i < expected_shape.size(); ++i) {
      EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, i),
                expected_shape[i]);
    }
    EXPECT_EQ(iree_hal_buffer_view_element_type(buffer_view),
              IREE_HAL_ELEMENT_TYPE_FLOAT_32);

    std::vector<float> actual_values(expected_values.size());
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        iree_hal_buffer_view_buffer(buffer_view), /*source_offset=*/0,
        actual_values.data(), actual_values.size() * sizeof(float)));
    EXPECT_EQ(actual_values, expected_values);
  }

  void ExpectS32BufferView(const iree_vm_variant_t& variant,
                           const std::vector<iree_hal_dim_t>& expected_shape,
                           std::vector<int32_t> expected_values) {
    ASSERT_TRUE(iree_vm_variant_is_ref(variant));
    iree_hal_buffer_view_t* buffer_view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_view_check_deref(variant.ref, &buffer_view));
    ASSERT_NE(buffer_view, nullptr);
    EXPECT_EQ(iree_hal_buffer_view_shape_rank(buffer_view),
              expected_shape.size());
    for (iree_host_size_t i = 0; i < expected_shape.size(); ++i) {
      EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, i),
                expected_shape[i]);
    }
    EXPECT_EQ(iree_hal_buffer_view_element_type(buffer_view),
              IREE_HAL_ELEMENT_TYPE_SINT_32);

    std::vector<int32_t> actual_values(expected_values.size());
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        iree_hal_buffer_view_buffer(buffer_view), /*source_offset=*/0,
        actual_values.data(), actual_values.size() * sizeof(int32_t)));
    EXPECT_EQ(actual_values, expected_values);
  }

  loom_named_attr_slice_t MakeMatmulContractAttrs(
      iree_string_view_t lhs, iree_string_view_t rhs,
      iree_string_view_t accumulator, iree_string_view_t result) {
    loom_string_id_t lhs_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t rhs_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t accumulator_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t result_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t lhs_value = LOOM_STRING_ID_INVALID;
    loom_string_id_t rhs_value = LOOM_STRING_ID_INVALID;
    loom_string_id_t accumulator_value = LOOM_STRING_ID_INVALID;
    loom_string_id_t result_value = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("lhs"), &lhs_name));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("rhs"), &rhs_name));
    IREE_CHECK_OK(loom_module_intern_string(module_, IREE_SV("accumulator"),
                                            &accumulator_name));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("result"), &result_name));
    IREE_CHECK_OK(loom_module_intern_string(module_, lhs, &lhs_value));
    IREE_CHECK_OK(loom_module_intern_string(module_, rhs, &rhs_value));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, accumulator, &accumulator_value));
    IREE_CHECK_OK(loom_module_intern_string(module_, result, &result_value));
    const loom_named_attr_t attrs[] = {
        {/*.name_id=*/lhs_name, /*.reserved=*/{},
         /*.value=*/loom_attr_string(lhs_value)},
        {/*.name_id=*/rhs_name, /*.reserved=*/{},
         /*.value=*/loom_attr_string(rhs_value)},
        {/*.name_id=*/accumulator_name,
         /*.reserved=*/{}, /*.value=*/loom_attr_string(accumulator_value)},
        {/*.name_id=*/result_name, /*.reserved=*/{},
         /*.value=*/loom_attr_string(result_value)},
    };
    loom_attribute_t attr = {};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module_, loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
        &attr));
    return loom_attr_as_dict(attr);
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  iree_vm_instance_t* vm_instance_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
  loom_testbench_reference_matmul_oracle_options_t reference_options_ = {};
};

TEST_F(ReferenceTest, ComputesF16MatmulWithF32Accumulator) {
  std::vector<uint16_t> lhs_values = {
      iree_math_f32_to_f16(1.0f), iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f), iree_math_f32_to_f16(4.0f),
      iree_math_f32_to_f16(5.0f), iree_math_f32_to_f16(6.0f),
  };
  std::vector<uint16_t> rhs_values = {
      iree_math_f32_to_f16(7.0f),  iree_math_f32_to_f16(8.0f),
      iree_math_f32_to_f16(9.0f),  iree_math_f32_to_f16(10.0f),
      iree_math_f32_to_f16(11.0f), iree_math_f32_to_f16(12.0f),
  };
  std::vector<float> init_values = {0.5f, 1.0f, 1.5f, 2.0f};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<uint16_t>({2, 3}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               lhs_values),
      MakeBufferView<uint16_t>({3, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               rhs_values),
      MakeBufferView<float>({2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                            init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
  };
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectF32BufferView(results[0], {2, 2}, {58.5f, 65.0f, 140.5f, 156.0f});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

TEST_F(ReferenceTest, ComputesU8MatmulWithF32Accumulator) {
  std::vector<int8_t> lhs_values = {-1, 2, -2, 3, 4, -3};
  std::vector<int8_t> rhs_values = {1, 2, 3, 4, 5, 6};
  std::vector<float> init_values = {0.5f, 1.0f, 1.5f, 2.0f};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<int8_t>({2, 3}, IREE_HAL_ELEMENT_TYPE_SINT_8, lhs_values),
      MakeBufferView<int8_t>({3, 2}, IREE_HAL_ELEMENT_TYPE_SINT_8, rhs_values),
      MakeBufferView<float>({2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                            init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
      /*.op=*/{},
      /*.callee_ref=*/{},
      /*.provider_id=*/{},
      /*.provider=*/{},
      /*.attrs=*/
      MakeMatmulContractAttrs(IREE_SV("u8"), IREE_SV("u8"), IREE_SV("f32"),
                              IREE_SV("f32")),
  };
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectF32BufferView(results[0], {2, 2}, {1531.5f, 2043.0f, 1281.5f, 1542.0f});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

TEST_F(ReferenceTest, ComputesTilePackedF16MatmulWithF32Accumulator) {
  std::vector<uint16_t> lhs_values = {
      iree_math_f32_to_f16(1.0f), iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f), iree_math_f32_to_f16(4.0f),
      iree_math_f32_to_f16(5.0f), iree_math_f32_to_f16(6.0f),
      iree_math_f32_to_f16(7.0f), iree_math_f32_to_f16(8.0f),
  };
  std::vector<uint16_t> rhs_values = {
      iree_math_f32_to_f16(9.0f),  iree_math_f32_to_f16(10.0f),
      iree_math_f32_to_f16(11.0f), iree_math_f32_to_f16(12.0f),
      iree_math_f32_to_f16(13.0f), iree_math_f32_to_f16(14.0f),
      iree_math_f32_to_f16(15.0f), iree_math_f32_to_f16(16.0f),
  };
  std::vector<float> init_values = {0.5f, 1.0f, 1.5f, 2.0f};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<uint16_t>({1, 2, 2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               lhs_values),
      MakeBufferView<uint16_t>({2, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               rhs_values),
      MakeBufferView<float>({1, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                            init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
  };
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectF32BufferView(results[0], {1, 1, 2, 2},
                      {186.5f, 201.0f, 283.5f, 306.0f});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

TEST_F(ReferenceTest, ComputesTilePackedBF16MatmulWithF32Accumulator) {
  std::vector<uint16_t> lhs_values = {
      iree_math_f32_to_bf16(1.0f),
      iree_math_f32_to_bf16(2.0f),
      iree_math_f32_to_bf16(3.0f),
      iree_math_f32_to_bf16(4.0f),
  };
  std::vector<uint16_t> rhs_values = {
      iree_math_f32_to_bf16(5.0f),
      iree_math_f32_to_bf16(6.0f),
      iree_math_f32_to_bf16(7.0f),
      iree_math_f32_to_bf16(8.0f),
  };
  std::vector<float> init_values = {0.5f, 1.0f, 1.5f, 2.0f};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<uint16_t>({1, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
                               lhs_values),
      MakeBufferView<uint16_t>({1, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
                               rhs_values),
      MakeBufferView<float>({1, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                            init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
  };
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectF32BufferView(results[0], {1, 1, 2, 2}, {19.5f, 23.0f, 44.5f, 52.0f});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

TEST_F(ReferenceTest, ComputesTilePackedU8MatmulWithI32Accumulator) {
  std::vector<int8_t> lhs_values = {-1, 2, -2, 3, 4, -3, 5, -4};
  std::vector<int8_t> rhs_values = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> init_values = {10, 20, 30, 40};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<int8_t>({1, 2, 2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                             lhs_values),
      MakeBufferView<int8_t>({2, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                             rhs_values),
      MakeBufferView<int32_t>({1, 1, 2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_32,
                              init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
      /*.op=*/{},
      /*.callee_ref=*/{},
      /*.provider_id=*/{},
      /*.provider=*/{},
      /*.attrs=*/
      MakeMatmulContractAttrs(IREE_SV("u8"), IREE_SV("u8"), IREE_SV("i32"),
                              IREE_SV("i32")),
  };
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectS32BufferView(results[0], {1, 1, 2, 2}, {2062, 2586, 2082, 2606});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

TEST_F(ReferenceTest, RejectsIntegerAccumulatorOverflow) {
  std::vector<int8_t> lhs_values = {-1};
  std::vector<int8_t> rhs_values = {2};
  std::vector<int8_t> init_values = {0};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<int8_t>({1, 1, 1, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                             lhs_values),
      MakeBufferView<int8_t>({1, 1, 1, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                             rhs_values),
      MakeBufferView<int8_t>({1, 1, 1, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                             init_values),
  };

  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &reference_options_, &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {
      /*.kind=*/{},
      /*.module=*/module_,
      /*.op=*/{},
      /*.callee_ref=*/{},
      /*.provider_id=*/{},
      /*.provider=*/{},
      /*.attrs=*/
      MakeMatmulContractAttrs(IREE_SV("u8"), IREE_SV("u8"), IREE_SV("i8"),
                              IREE_SV("i8")),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      provider.invoke.fn(provider.invoke.user_data, &invocation,
                         IREE_ARRAYSIZE(inputs), inputs,
                         IREE_ARRAYSIZE(results), results));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

}  // namespace
}  // namespace loom
