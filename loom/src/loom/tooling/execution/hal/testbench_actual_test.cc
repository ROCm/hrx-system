// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/link/linker.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/facts.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/profile.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/testbench.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(
      RegisterDialect(context, LOOM_DIALECT_CHECK, loom_check_dialect_vtables));
  IREE_RETURN_IF_ERROR(
      RegisterDialect(context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables));
  IREE_RETURN_IF_ERROR(
      RegisterDialect(context, LOOM_DIALECT_INDEX, loom_index_dialect_vtables));
  return RegisterDialect(context, LOOM_DIALECT_KERNEL,
                         loom_kernel_dialect_vtables);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        /*.fn=*/RegisterContext,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            /*.fn=*/InitializeLowDescriptorRegistry,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override {
    loom_run_session_deinitialize(&session_);
    iree_arena_deinitialize(&plan_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ParseAndPlan(iree_string_view_t source, loom_run_module_t* out_module,
                    loom_testbench_module_plan_t* out_plan) {
    loom_run_module_parse_options_t parse_options = {};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = IREE_SV("hal_testbench_actual_test.loom");
    parse_options.source = source;
    IREE_ASSERT_OK(
        loom_run_module_parse(&session_, &parse_options, out_module));
    IREE_ASSERT_OK(loom_testbench_plan_module(out_module->module, nullptr,
                                              &plan_arena_, out_plan));
    ASSERT_EQ(out_plan->issue_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_run_session_t session_ = {};
};

static loom_testbench_value_t I32Value(int32_t value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
  result.scalar.storage.i32 = value;
  return result;
}

static loom_testbench_value_t I64Value(int64_t value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
  result.scalar.storage.i64 = value;
  return result;
}

static loom_testbench_value_t F64Value(double value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
  result.scalar.storage.f64 = value;
  return result;
}

static const loom_target_profile_type_t kFakeTargetProfileType = {
    /*.name=*/IREE_SVL("fake"),
};
static const loom_target_profile_t kFakeTargetProfile = {
    /*.type=*/&kFakeTargetProfileType,
    /*.target_bundle=*/nullptr,
};

static iree_status_t FakeHalSelectDeviceTarget(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  (void)provider;
  (void)runtime;
  (void)allocator;
  *out_target = (loom_run_hal_device_target_t){
      /*.hal_target=*/nullptr,
      /*.target_profile=*/&kFakeTargetProfile,
      /*.target_key=*/IREE_SVL("fake"),
  };
  return iree_ok_status();
}

static iree_status_t FakeHalSelectCompatibleDeviceTarget(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  if (target_requirement != nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fake HAL provider requires targetless input");
  }
  return FakeHalSelectDeviceTarget(provider, runtime, allocator, out_target);
}

static const loom_run_hal_artifact_provider_t kFakeHalArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.hal_driver_name=*/IREE_SVL("fake"),
    /*.target_family_name=*/IREE_SVL("fake-target"),
    /*.default_pipeline_options=*/{},
    /*.select_device_target=*/FakeHalSelectDeviceTarget,
    /*.select_compatible_device_target=*/FakeHalSelectCompatibleDeviceTarget,
};

static bool ModuleHasSymbol(const loom_module_t* module,
                            iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  return name_id != LOOM_STRING_ID_INVALID &&
         loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID;
}

static bool ModuleSymbolIsFuncDef(const loom_module_t* module,
                                  iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) return false;
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  return symbol_id != LOOM_SYMBOL_ID_INVALID &&
         loom_func_def_isa(module->symbols.entries[symbol_id].defining_op);
}

TEST_F(HalTestbenchActualTest, RequiresExplicitDeviceWhenHalProviderExists) {
  const loom_run_hal_artifact_provider_t* artifact_providers[] = {
      &kFakeHalArtifactProvider,
  };
  loom_run_hal_artifact_provider_registry_t registry = {};
  loom_run_hal_artifact_provider_registry_initialize_from_entries(
      artifact_providers, IREE_ARRAYSIZE(artifact_providers), &registry);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(&registry, iree_allocator_system(),
                                            &context);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_testbench_context_ensure_runtime(&context));

  loom_run_hal_testbench_context_deinitialize(&context);
}

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      I32Value(0x12345678),
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 0x12345678u);
  EXPECT_EQ(options.constants[1], 0x55667788u);
  EXPECT_EQ(options.constants[2], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      F64Value(1.0),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputPacksAsOneDispatchConstantWord) {
  loom_testbench_value_t inputs[] = {
      I64Value(3584),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], 3584u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputPacksAsTwoDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, RejectsNestedKernelLaunchSchedules) {
  static constexpr char kSource[] = R"(
kernel.decl @step() launch(%output: buffer)

check.case @nested_launch_schedule {
  %output = check.generate.fill value(0) : tensor<1xi32>
  kernel.launch.serial {
    kernel.launch.concurrent {
      kernel.launch @step(%output) : (tensor<1xi32>)
    }
  }
  check.return
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);

  iree_host_size_t kernel_launch_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_run_hal_testbench_count_kernel_launches(
                            &module_plan.cases[0], &kernel_launch_count));

  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalTestbenchActualTest, RetainsResolvedLaunchConfigForExactWorkload) {
  static constexpr char kSource[] = R"(
kernel.def @dynamic(%workgroup_count: index) {
  %unit = index.constant 1 : index
  %size = index.constant 64 : index
  kernel.launch.config workgroups(%workgroup_count, %unit, %unit) workgroup_size(%size, %unit, %unit) : index
} launch(%workgroup_count: index) {
  kernel.return
}

check.case @dynamic_case {
  %workgroup_count = check.param.choice values([1, 4]) name("workgroup_count") : index
  kernel.launch @dynamic[%workgroup_count](%workgroup_count) : [index](index)
  check.return
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);
  const loom_testbench_case_plan_t* case_plan = &module_plan.cases[0];
  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_ASSERT_OK(
      loom_run_hal_testbench_select_kernel_launch(case_plan, &kernel_launch));

  loom_target_facts_t target_facts = {};
  int64_t workload_arguments[1] = {};
  loom_run_hal_testbench_actual_provider_t provider = {};
  provider.session = &session_;
  provider.run_module = &run_module;
  provider.kernel_launch = kernel_launch;
  provider.launch_config_module = run_module.module;
  provider.launch_config_target_facts = &target_facts;
  provider.workload_arguments = workload_arguments;
  provider.invocation_options.function_name = IREE_SV("dynamic");

  loom_testbench_value_materializer_options_t materializer_options = {};
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  loom_testbench_value_table_t value_table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      run_module.module, case_plan, iree_allocator_system(), &value_table));
  for (iree_host_size_t sample_ordinal = 0; sample_ordinal < 2;
       ++sample_ordinal) {
    loom_testbench_value_table_reset(&value_table);
    IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
        &materializer_options, case_plan, sample_ordinal, &value_table));
    loom_run_hal_invocation_options_t invocation_options = {};
    loom_run_hal_binding_list_t bindings = {};
    IREE_ASSERT_OK(loom_run_hal_testbench_materialize_invocation_from_table(
        &value_table, &provider, iree_allocator_system(), &invocation_options,
        &bindings));

    const uint32_t expected_workgroup_count = sample_ordinal == 0 ? 1 : 4;
    EXPECT_EQ(provider.workload_arguments[0], expected_workgroup_count);
    EXPECT_EQ(invocation_options.workgroup_count[0], expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_count.x,
              expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_size.x, 64u);
    EXPECT_TRUE(iree_all_bits_set(
        provider.resolved_launch_config.fields,
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
            LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE));
    loom_run_hal_binding_list_deinitialize(&bindings);
  }

  loom_testbench_value_table_deinitialize(&value_table);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalTestbenchActualTest, CompileModuleClonesLinkedSelectedEntry) {
  static constexpr char kInputSource[] = R"(
func.decl @linked_identity(%value: index) -> (index)

kernel.def @selected() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%element_count: index) {
  %unused = func.call @linked_identity(%element_count) : (index) -> (index)
  kernel.return
}

kernel.def @uncalled() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}

check.case @selected_case {
  %element_count = check.param.choice values([31, 32]) name("element_count") : index
  kernel.launch @selected(%element_count) : (index)
  check.return
}
)";
  static constexpr char kLibrarySource[] = R"(
func.def inline @linked_identity(%value: index) -> (index) {
  func.return %value : index
}
)";
  loom_run_module_t input_module = {};
  loom_run_module_parse_options_t parse_options = {};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = IREE_SV("hal_testbench_actual_input.loom");
  parse_options.source = IREE_SV(kInputSource);
  IREE_ASSERT_OK(
      loom_run_module_parse(&session_, &parse_options, &input_module));

  loom_run_module_t library_module = {};
  parse_options.filename = IREE_SV("hal_testbench_actual_library.loom");
  parse_options.source = IREE_SV(kLibrarySource);
  IREE_ASSERT_OK(
      loom_run_module_parse(&session_, &parse_options, &library_module));

  const loom_module_t* source_modules[] = {
      input_module.module,
      library_module.module,
  };
  const loom_link_options_t link_options = {
      /*.module_name=*/IREE_SV("linked_test"),
  };
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules), &link_options,
      loom_run_session_block_pool(&session_), iree_allocator_system(),
      &run_module.module));
  loom_run_module_deinitialize(&library_module);
  loom_run_module_deinitialize(&input_module);

  loom_testbench_module_plan_t module_plan = {};
  IREE_ASSERT_OK(loom_testbench_plan_module(run_module.module, nullptr,
                                            &plan_arena_, &module_plan));
  ASSERT_EQ(module_plan.issue_count, 0u);
  ASSERT_EQ(module_plan.case_count, 1u);
  const loom_testbench_case_plan_t* case_plan = &module_plan.cases[0];
  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_ASSERT_OK(
      loom_run_hal_testbench_select_kernel_launch(case_plan, &kernel_launch));

  loom_run_hal_testbench_context_t context = {};
  context.artifact_provider = &kFakeHalArtifactProvider;
  // Provider compile only needs target selection before the fake artifact
  // provider rejects this source; avoid requiring a real HAL device for a
  // rooted-link contract test.
  context.runtime_initialized = true;
  context.host_allocator = iree_allocator_system();

  loom_run_hal_testbench_actual_provider_options_t options = {};
  options.context = &context;
  options.session = &session_;
  options.run_module = &run_module;
  options.kernel_launch = kernel_launch;

  loom_run_hal_testbench_actual_provider_t provider = {};
  loom_run_hal_testbench_actual_provider_initialize(&options, &provider);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_actual_provider_compile(&provider));
  EXPECT_TRUE(
      ModuleHasSymbol(provider.compile_module.module, IREE_SV("selected")));
  EXPECT_TRUE(ModuleSymbolIsFuncDef(provider.compile_module.module,
                                    IREE_SV("linked_identity")));
  EXPECT_FALSE(
      ModuleHasSymbol(provider.compile_module.module, IREE_SV("uncalled")));
  EXPECT_FALSE(ModuleHasSymbol(provider.compile_module.module,
                               IREE_SV("selected_case")));

  loom_run_hal_testbench_actual_provider_deinitialize(&provider);
  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
