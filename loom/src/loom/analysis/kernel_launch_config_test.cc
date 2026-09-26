// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_launch_config.h"

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class KernelLaunchConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("kernel_launch_config_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  enum class StorageRequest { kAbsent, kConstant, kWorkload };

  iree_status_t BuildStorageKernel(StorageRequest request,
                                   int64_t constant_byte_length,
                                   ModulePtr* out_module) {
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_module_allocate(&context_, IREE_SV("storage"), &block_pool_,
                             nullptr, iree_allocator_system(), &module));
    out_module->reset(module);
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, IREE_SV("entry"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol));
    const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    const loom_symbol_ref_t reference = {/*.module_id=*/0,
                                         /*.symbol_id=*/symbol};
    loom_op_t* kernel = nullptr;
    IREE_RETURN_IF_ERROR(loom_kernel_def_build(
        &builder, /*build_flags=*/0, /*retain=*/0, loom_symbol_ref_null(),
        LOOM_STRING_ID_INVALID, /*export_linkage=*/0, reference, &offset_type,
        request == StorageRequest::kWorkload ? 1 : 0,
        /*arg_types=*/nullptr, /*arg_types_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &kernel));
    loom_builder_set_block(
        &builder, loom_region_entry_block(loom_kernel_def_body(kernel)));
    loom_op_t* terminator = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_kernel_return_build(&builder, LOOM_LOCATION_UNKNOWN, &terminator));

    loom_block_t* config =
        loom_region_entry_block(loom_kernel_def_config(kernel));
    loom_builder_set_block(&builder, config);
    loom_op_t* constant = nullptr;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &builder, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &constant));
    const loom_value_id_t unit = loom_index_constant_result(constant);
    loom_value_id_t byte_length = LOOM_VALUE_ID_INVALID;
    if (request == StorageRequest::kWorkload) {
      byte_length = loom_block_arg_id(config, 0);
    } else if (request == StorageRequest::kConstant) {
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &builder, loom_attr_i64(constant_byte_length), offset_type,
          LOOM_LOCATION_UNKNOWN, &constant));
      byte_length = loom_index_constant_result(constant);
    }
    return loom_kernel_launch_config_build(
        &builder,
        request == StorageRequest::kAbsent
            ? 0
            : LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE,
        unit, unit, unit, unit, unit, unit, LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, byte_length,
        LOOM_LOCATION_UNKNOWN, &terminator);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(KernelLaunchConfigTest, DirectlyEvaluatesTargetIndependentConstants) {
  ModulePtr module = Parse(R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %four = index.constant 4 : index
  %five = index.constant 5 : index
  %six = index.constant 6 : index
  kernel.launch.config workgroups(%two, %three, %four) workgroup_size(%five, %six, %one) : index
} launch() {
  kernel.return
}
)");

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("@entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
      /*.function_target_facts=*/nullptr,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  bool evaluated = false;
  IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
      module.get(), &block_pool_, &options, &config, &evaluated));

  EXPECT_TRUE(evaluated);
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 2u);
  EXPECT_EQ(config.workgroup_count.y, 3u);
  EXPECT_EQ(config.workgroup_count.z, 4u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 5u);
  EXPECT_EQ(config.workgroup_size.y, 6u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST_F(KernelLaunchConfigTest, DirectPathSkipsTargetBoundKernels) {
  ModulePtr module = Parse(R"(
target.generic<reference> @gpu {
  subgroup_size = 32
}

kernel.def target(@gpu) @entry() {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
      /*.function_target_facts=*/nullptr,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  bool evaluated = true;
  IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
      module.get(), &block_pool_, &options, &config, &evaluated));

  EXPECT_FALSE(evaluated);
  EXPECT_EQ(config.fields, 0u);
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
}

TEST_F(KernelLaunchConfigTest, DirectlyEvaluatesWithFunctionTargetFacts) {
  ModulePtr module = Parse(R"(
target.generic<reference> @authored_gpu {
  subgroup_size = 32
}

kernel.def target(@authored_gpu) @entry() {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loom_target_snapshot_t effective_snapshot = {};
  effective_snapshot.name = IREE_SVL("effective");
  effective_snapshot.subgroup_size = 64;
  const loom_target_fact_type_t fact_type = {
      /*.name=*/IREE_SVL("test"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  loom_target_facts_t function_target_facts = {};
  function_target_facts.fact_type = &fact_type;
  function_target_facts.storage.snapshot = effective_snapshot;
  function_target_facts.storage.bundle.name = IREE_SVL("effective");
  loom_target_bundle_storage_rebind(&function_target_facts.storage);

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
      /*.function_target_facts=*/&function_target_facts,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  bool evaluated = false;
  IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
      module.get(), &block_pool_, &options, &config, &evaluated));

  EXPECT_TRUE(evaluated);
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_EQ(config.workgroup_count.x, 1u);
  EXPECT_EQ(config.subgroup_size, 64u);

  config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_EQ(config.workgroup_count.x, 1u);
  EXPECT_EQ(config.subgroup_size, 64u);
}

TEST_F(KernelLaunchConfigTest,
       EvaluatesContextualSubgroupSizeFromFunctionTargetFacts) {
  ModulePtr module = Parse(R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  %subgroup_size = target.subgroup.size : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%subgroup_size, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loom_target_snapshot_t effective_snapshot = {};
  effective_snapshot.name = IREE_SVL("effective");
  const loom_target_fact_type_t fact_type = {
      /*.name=*/IREE_SVL("test"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  loom_target_facts_t function_target_facts = {};
  function_target_facts.fact_type = &fact_type;
  function_target_facts.storage.snapshot = effective_snapshot;
  function_target_facts.storage.bundle.name = IREE_SVL("effective");
  loom_target_bundle_storage_rebind(&function_target_facts.storage);

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
      /*.function_target_facts=*/&function_target_facts,
      /*.diagnostic_emitter=*/{},
  };

  for (uint32_t subgroup_size : {32u, 64u}) {
    function_target_facts.storage.snapshot.subgroup_size = subgroup_size;
    loom_kernel_launch_config_t config = {};
    IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(
        module.get(), &block_pool_, &options, &config));

    EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
    EXPECT_EQ(config.workgroup_count.x, 1u);
    EXPECT_EQ(config.workgroup_size.x, subgroup_size);
    EXPECT_EQ(config.subgroup_size, subgroup_size);
  }
}

TEST_F(KernelLaunchConfigTest, DoesNotInventContextualSubgroupSize) {
  ModulePtr module = Parse(R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  %subgroup_size = target.subgroup.size : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%subgroup_size, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
      /*.function_target_facts=*/nullptr,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));

  EXPECT_EQ(config.failure,
            LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_SIZE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_FALSE(config.fields &
               LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
}

TEST_F(KernelLaunchConfigTest, EvaluatesTargetAndWorkloadBackedFields) {
  ModulePtr module = Parse(R"(
target.generic<reference> @gpu {
  subgroup_size = 32
}

kernel.def target(@gpu) @entry(%rows: index) {
  %one = index.constant 1 : index
  %sixty_three = index.constant 63 : index
  %sixty_four = index.constant 64 : index
  %rounded_rows = index.add %rows, %sixty_three : index
  %row_groups = index.div %rounded_rows, %sixty_four : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%sixty_four, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const int64_t workload_arguments[] = {129};
  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/workload_arguments,
      /*.workload_argument_count=*/IREE_ARRAYSIZE(workload_arguments),
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
      /*.function_target_facts=*/nullptr,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));

  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 3u);
  EXPECT_EQ(config.workgroup_count.y, 1u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE);
  EXPECT_EQ(config.subgroup_size, 32u);
}

TEST_F(KernelLaunchConfigTest,
       AddsFixedPrefixToRepeatedRuntimeStorageRequests) {
  ModulePtr module;
  IREE_ASSERT_OK(BuildStorageKernel(StorageRequest::kWorkload, 0, &module));
  const uint64_t fixed_bytes = 64;
  int64_t dynamic_bytes = 0;
  loom_kernel_launch_config_options_t options = {};
  options.function_symbol = IREE_SV("entry");
  options.workload_arguments = &dynamic_bytes;
  options.workload_argument_count = 1;
  options.required_fields =
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  options.fixed_workgroup_storage_bytes = &fixed_bytes;

  for (int64_t request :
       {INT64_C(0), INT64_C(64), INT64_C(192), INT64_C(64), INT64_C(1) << 34}) {
    SCOPED_TRACE(request);
    dynamic_bytes = request;
    loom_kernel_launch_config_t config = {};
    IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(
        module.get(), &block_pool_, &options, &config));
    EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
    EXPECT_TRUE(config.fields &
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES);
    EXPECT_EQ(config.workgroup_storage_bytes,
              fixed_bytes + static_cast<uint64_t>(request));
  }
}

TEST_F(KernelLaunchConfigTest, DirectAndFullStorageTotalsUseKnownFixedPrefix) {
  for (StorageRequest request :
       {StorageRequest::kAbsent, StorageRequest::kConstant}) {
    ModulePtr module;
    IREE_ASSERT_OK(BuildStorageKernel(request, 192, &module));
    const uint64_t fixed_bytes = 64;
    loom_kernel_launch_config_options_t options = {};
    options.function_symbol = IREE_SV("entry");
    options.required_fields =
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;

    for (const uint64_t* prefix :
         {static_cast<const uint64_t*>(nullptr), &fixed_bytes}) {
      options.fixed_workgroup_storage_bytes = prefix;
      loom_kernel_launch_config_t direct_config = {};
      bool evaluated = false;
      IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
          module.get(), &block_pool_, &options, &direct_config, &evaluated));
      loom_kernel_launch_config_t config = {};
      IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(
          module.get(), &block_pool_, &options, &config));
      const auto expected_failure =
          prefix
              ? LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE
              : LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES;
      EXPECT_EQ(config.failure, expected_failure);
      EXPECT_EQ(direct_config.failure, expected_failure);
      EXPECT_EQ(evaluated, prefix != nullptr);
      EXPECT_EQ(config.fields, direct_config.fields);
      const uint64_t expected_total =
          prefix
              ? fixed_bytes + (request == StorageRequest::kConstant ? 192 : 0)
              : 0;
      EXPECT_EQ(config.workgroup_storage_bytes, expected_total);
      EXPECT_EQ(direct_config.workgroup_storage_bytes, expected_total);
    }
  }
}

TEST_F(KernelLaunchConfigTest, DistinguishesZeroRequestFromUnknownRequest) {
  ModulePtr module;
  IREE_ASSERT_OK(BuildStorageKernel(StorageRequest::kWorkload, 0, &module));
  const uint64_t fixed_bytes = 0;
  loom_kernel_launch_config_options_t options = {};
  options.function_symbol = IREE_SV("entry");
  options.required_fields =
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  options.fixed_workgroup_storage_bytes = &fixed_bytes;
  loom_kernel_launch_config_t config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));
  EXPECT_EQ(config.failure,
            LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES);
  EXPECT_FALSE(config.fields &
               LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES);

  const int64_t dynamic_bytes = 0;
  options.workload_arguments = &dynamic_bytes;
  options.workload_argument_count = 1;
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES);
  EXPECT_EQ(config.workgroup_storage_bytes, 0u);
}

TEST_F(KernelLaunchConfigTest, ChecksStorageTotalAtOffsetBoundary) {
  ModulePtr module;
  IREE_ASSERT_OK(BuildStorageKernel(StorageRequest::kWorkload, 0, &module));
  struct StorageCase {
    // Compiled fixed prefix supplied by the caller.
    uint64_t fixed_bytes;
    // Additional bytes supplied as a workload value.
    int64_t dynamic_bytes;
    // Whether the total is representable in the nonnegative offset domain.
    bool valid;
  };
  const StorageCase cases[] = {
      {0, INT64_MAX, true},
      {INT64_MAX, 0, true},
      {INT64_MAX, 1, false},
      {64, INT64_MAX, false},
      {uint64_t{INT64_MAX} + 1, 0, false},
      {0, -1, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "fixed=" << test_case.fixed_bytes
                 << ", dynamic=" << test_case.dynamic_bytes);
    loom_kernel_launch_config_options_t options = {};
    options.function_symbol = IREE_SV("entry");
    options.workload_arguments = &test_case.dynamic_bytes;
    options.workload_argument_count = 1;
    options.required_fields =
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
    options.fixed_workgroup_storage_bytes = &test_case.fixed_bytes;
    loom_kernel_launch_config_t config = {};
    IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(
        module.get(), &block_pool_, &options, &config));
    EXPECT_EQ(
        config.failure,
        test_case.valid
            ? LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE
            : LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES);
    EXPECT_EQ(
        (config.fields &
         LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES) != 0,
        test_case.valid);
    EXPECT_EQ(config.workgroup_storage_bytes,
              test_case.valid
                  ? test_case.fixed_bytes +
                        static_cast<uint64_t>(test_case.dynamic_bytes)
                  : 0);
  }
}

}  // namespace
}  // namespace loom
