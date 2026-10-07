// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/link/linker.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pass/ops.h"
#include "loom/target/facts.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/profile.h"
#include "loom/target/provider.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/transforms/cleanup/configured.h"

namespace loom {
namespace {

extern const loom_target_provider_t kFakeTargetProvider;

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    target_provider_.initialize_low_descriptor_registry =
        loom_target_core_test_low_descriptor_registry_initialize;
    loom_target_provider_set_storage_initialize(&target_provider_storage_);
    IREE_ASSERT_OK(loom_target_provider_set_storage_append(
        &target_provider_storage_, &target_provider_));
    IREE_ASSERT_OK(loom_target_provider_set_storage_append(
        &target_provider_storage_, &kFakeTargetProvider));
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &target_provider_storage_.provider_set, &target_environment_));

    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.target_environment = &target_environment_;
    options.cleanup_pattern_provider_set =
        loom_cleanup_configured_pattern_provider_set();
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override {
    loom_run_session_deinitialize(&session_);
    loom_target_environment_deinitialize(&target_environment_);
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

  void ExpectTargetSelection(iree_string_view_t target,
                             bool expects_explicit_selection);

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_run_session_t session_ = {};
  loom_target_provider_t target_provider_ = {};
  loom_target_provider_set_storage_t target_provider_storage_ = {};
  loom_target_environment_t target_environment_ = {};
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

static loom_testbench_value_t F32Value(float value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F32;
  result.scalar.storage.f32 = value;
  return result;
}

static loom_testbench_value_t F64Value(double value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
  result.scalar.storage.f64 = value;
  return result;
}

static loom_target_snapshot_t AddressTargetSnapshot(uint32_t index_bitwidth,
                                                    uint32_t offset_bitwidth) {
  loom_target_snapshot_t snapshot = {};
  snapshot.index_bitwidth = index_bitwidth;
  snapshot.offset_bitwidth = offset_bitwidth;
  return snapshot;
}

static loom_target_snapshot_t FakeTargetSnapshot() {
  loom_target_snapshot_t snapshot = AddressTargetSnapshot(32, 64);
  snapshot.name = IREE_SV("fake-snapshot");
  return snapshot;
}

static const loom_target_snapshot_t kFakeTargetSnapshot = FakeTargetSnapshot();
static const loom_target_snapshot_t kIndex32Offset64TargetSnapshot =
    AddressTargetSnapshot(32, 64);
static const loom_target_snapshot_t kIndex64Offset32TargetSnapshot =
    AddressTargetSnapshot(64, 32);

static const loom_target_export_plan_t kFakeTargetExportPlan = {
    /*.name=*/IREE_SVL("fake-export"),
    /*.export_symbol=*/{},
    /*.calling_convention=*/{},
    /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
};
static const loom_target_config_t kFakeTargetConfig = {
    /*.name=*/IREE_SVL("fake-config"),
};
static const loom_target_bundle_t kFakeTargetBundle = {
    /*.name=*/IREE_SVL("fake-bundle"),
    /*.snapshot=*/&kFakeTargetSnapshot,
    /*.export_plan=*/&kFakeTargetExportPlan,
    /*.config=*/&kFakeTargetConfig,
};
static const loom_target_fact_type_t kFakeTargetFactType = {
    /*.name=*/IREE_SVL("fake"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};

static const loom_target_profile_t* g_projected_target_profile = nullptr;
static iree_host_size_t g_compatible_target_selection_count = 0;
static iree_host_size_t g_profile_target_selection_count = 0;

static iree_status_t ProjectFakeTargetFacts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)arena;
  (void)out_facts;
  g_projected_target_profile = profile;
  return iree_ok_status();
}

static const loom_target_profile_type_t kFakeTargetProfileType = {
    /*.name=*/IREE_SVL("fake"),
    /*.fact_type=*/&kFakeTargetFactType,
    /*.project_facts=*/ProjectFakeTargetFacts,
};
static const loom_target_profile_t kFakeTargetProfile = {
    /*.type=*/&kFakeTargetProfileType,
    /*.target_bundle=*/&kFakeTargetBundle,
};

static iree_status_t SelectFakeTargetProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = nullptr;
  if (!iree_string_view_equal(selector, IREE_SV("forced"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown fake target selector");
  }
  *out_profile = &kFakeTargetProfile;
  return iree_ok_status();
}

static loom_target_provider_t MakeFakeTargetProvider() {
  loom_target_provider_t provider = {};
  provider.profile_type = &kFakeTargetProfileType;
  provider.select_profile = SelectFakeTargetProfile;
  return provider;
}

const loom_target_provider_t kFakeTargetProvider = MakeFakeTargetProvider();

static iree_status_t FakeHalSelectDeviceTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  (void)provider;
  (void)allocator;
  const iree_hal_device_spec_t* device_spec =
      runtime != nullptr && runtime->device != nullptr
          ? iree_hal_device_spec(runtime->device)
          : nullptr;
  const iree_hal_device_executable_spec_t* executable_spec =
      device_spec != nullptr ? iree_hal_device_spec_executables(device_spec)
                             : nullptr;
  const iree_hal_executable_target_t* executable_target =
      executable_spec != nullptr && executable_spec->target_count != 0
          ? &executable_spec->targets[0]
          : nullptr;
  *out_target = (loom_device_target_t){
      /*.executable_target=*/executable_target,
      /*.target_profile=*/&kFakeTargetProfile,
  };
  return iree_ok_status();
}

static iree_status_t FakeHalSelectCompatibleDeviceTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  if (target_requirement != nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fake HAL provider requires targetless input");
  }
  ++g_compatible_target_selection_count;
  return FakeHalSelectDeviceTarget(provider, runtime, allocator, out_target);
}

static iree_status_t FakeHalSelectProfileDeviceTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_profile_t* target_profile,
    loom_device_target_t* out_target) {
  ++g_profile_target_selection_count;
  if (target_profile != &kFakeTargetProfile) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake target profile");
  }
  return FakeHalSelectDeviceTarget(provider, runtime, iree_allocator_null(),
                                   out_target);
}

static iree_status_t EmitFakeTargetArtifact(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  (void)request;
  *out_emitted = false;
  *out_artifact = {};
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "fake target emitter rejects emission");
}

static const loom_target_emitter_t kFakeTargetEmitter = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.public_artifact_format=*/IREE_SVL("fake-hal"),
    /*.default_identifier=*/IREE_SVL("fake.bin"),
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    /*.default_pipeline_options=*/{},
    /*.emit=*/EmitFakeTargetArtifact,
};

static const loom_device_provider_t kFakeDeviceProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.target_profile_type=*/&kFakeTargetProfileType,
    /*.target_emitter=*/&kFakeTargetEmitter,
    /*.driver_name=*/IREE_SVL("fake"),
    /*.select_compatible_target=*/FakeHalSelectCompatibleDeviceTarget,
    /*.select_profile_target=*/FakeHalSelectProfileDeviceTarget,
};

static iree_status_t InitializeFakeHalContext(
    loom_run_hal_testbench_context_t* context,
    iree_hal_queue_t* out_dispatch_queue) {
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("fake"),
      /*.target_key=*/IREE_SV("fake-target"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/50,
      /*.physical_device_affinity=*/1,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/1,
      /*.targets=*/&executable_target,
  };
  const iree_hal_queue_priority_t normal_priority =
      IREE_HAL_QUEUE_PRIORITY_NORMAL;
  const iree_hal_queue_family_spec_t queue_family = {
      /*.name=*/IREE_SV("dispatch"),
      /*.provisioned_queue_count=*/0,
      /*.priority_count=*/1,
      /*.priorities=*/&normal_priority,
      /*.execution_unit_count=*/0,
      /*.execution_resource_group_count=*/0,
      /*.execution_resource_groups=*/nullptr,
      /*.execution_resource_count=*/0,
      /*.execution_resources=*/nullptr,
      /*.supported_queue_features=*/IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
      /*.timestamp_valid_bits=*/0,
      /*.timestamp_frequency_hz=*/0,
      /*.physical_device_affinity=*/1,
      /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
      /*.atomic_capabilities=*/{},
      /*.zero_compute_atomic_capabilities=*/{},
      /*.flags=*/IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  const iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/1,
      /*.families=*/&queue_family,
  };
  iree_hal_device_spec_params_t params = {};
  params.queues = &queues;
  params.executables = &executables;
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);
  mock_options.identifier = IREE_SV("fake");
  mock_options.device_spec = device_spec;
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_mock_device_create(
      &mock_options, iree_allocator_system(), &device);
  iree_hal_device_spec_release(device_spec);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  out_dispatch_queue->queue_family = iree_hal_device_queue_family(device, 0);
  context->device_provider = &kFakeDeviceProvider;
  context->runtime = (loom_run_hal_runtime_t){
      /*.device=*/device,
      /*.dispatch_queue=*/out_dispatch_queue,
  };
  context->runtime_initialized = true;
  return iree_ok_status();
}

void HalTestbenchActualTest::ExpectTargetSelection(
    iree_string_view_t target, bool expects_explicit_selection) {
  static constexpr char kSource[] = R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}

check.case @entry_case {
  kernel.launch @entry() : ()
  check.return
}

pass.pipeline<module> @debug pipeline {
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);
  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_ASSERT_OK(loom_run_hal_testbench_select_kernel_launch(
      &module_plan.cases[0], &kernel_launch));

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*device_provider_registry=*/nullptr, iree_allocator_system(), &context);
  iree_hal_queue_t dispatch_queue = {};
  IREE_ASSERT_OK(InitializeFakeHalContext(&context, &dispatch_queue));

  g_projected_target_profile = nullptr;
  g_compatible_target_selection_count = 0;
  g_profile_target_selection_count = 0;
  loom_run_hal_testbench_actual_provider_options_t options = {};
  options.context = &context;
  options.session = &session_;
  options.run_module = &run_module;
  options.pipeline = IREE_SV("@debug");
  options.target = target;
  options.kernel_launch = kernel_launch;
  loom_run_hal_testbench_actual_provider_t provider = {};
  loom_run_hal_testbench_actual_provider_initialize(&options, &provider);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_actual_provider_compile(&provider));

  EXPECT_EQ(g_profile_target_selection_count,
            expects_explicit_selection ? 1u : 0u);
  EXPECT_EQ(g_compatible_target_selection_count,
            expects_explicit_selection ? 0u : 1u);
  EXPECT_EQ(provider.compile_device_target.target_profile, &kFakeTargetProfile);
  EXPECT_EQ(provider.owns_compile_device_target, !expects_explicit_selection);
  EXPECT_EQ(g_projected_target_profile, &kFakeTargetProfile);

  loom_source_table_resolver_t* table = &provider.compile_module.sources.table;
  EXPECT_EQ(table->module, provider.compile_module.module);
  const auto* module = table->module;
  auto symbol = loom_module_find_symbol(
      module, loom_module_lookup_string(module, IREE_SV("entry")));
  ASSERT_NE(symbol, LOOM_SYMBOL_ID_INVALID);
  loom_source_range_t range = {};
  ASSERT_TRUE(loom_source_table_resolve(
      table, module, module->symbols.entries[symbol].defining_op->location,
      &range));
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_TRUE(iree_string_view_equal(range.source, IREE_SV(kSource)));

  loom_run_hal_testbench_actual_provider_deinitialize(&provider);
  loom_run_hal_testbench_context_deinitialize(&context);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalTestbenchActualTest, UsesAutomaticTargetWhenOverrideIsEmpty) {
  ExpectTargetSelection(iree_string_view_empty(),
                        /*expects_explicit_selection=*/false);
}

TEST_F(HalTestbenchActualTest, UsesExplicitTargetOverride) {
  ExpectTargetSelection(IREE_SV("fake:forced"),
                        /*expects_explicit_selection=*/true);
}

static bool ModuleHasSymbol(const loom_module_t* module,
                            iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  return name_id != LOOM_STRING_ID_INVALID &&
         loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID;
}

static bool ModuleSymbolIsFuncDef(const loom_module_t* module,
                                  iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return false;
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  return symbol_id != LOOM_SYMBOL_ID_INVALID &&
         loom_func_def_isa(module->symbols.entries[symbol_id].defining_op);
}

TEST_F(HalTestbenchActualTest, RequiresExplicitDeviceWhenHalProviderExists) {
  const loom_device_provider_t* device_providers[] = {
      &kFakeDeviceProvider,
  };
  loom_device_provider_registry_t registry = {};
  loom_device_provider_registry_initialize_from_entries(
      device_providers, IREE_ARRAYSIZE(device_providers), &registry);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(&registry, iree_allocator_system(),
                                            &context);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_testbench_context_ensure_runtime(&context));

  loom_run_hal_testbench_context_deinitialize(&context);
}

TEST_F(HalTestbenchActualTest,
       AddsAuthoredSanitizerRequirementsBeforeRuntimeInitialization) {
  static constexpr char kSource[] = R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%condition: i1) {
  kernel.assert %condition : i1
  kernel.return
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*device_provider_registry=*/nullptr, iree_allocator_system(), &context);
  const loom_sanitizer_options_t sanitizer = {};
  IREE_ASSERT_OK(loom_run_hal_testbench_context_add_module_runtime_requirements(
      &context, run_module.module, &sanitizer));
  EXPECT_EQ(context.runtime_features,
            IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK);

  loom_run_hal_testbench_context_deinitialize(&context);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalTestbenchActualTest,
       RejectsNewSanitizerRequirementsAfterRuntimeInitialization) {
  static constexpr char kSource[] = R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%condition: i1) {
  kernel.assert %condition : i1
  kernel.return
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*device_provider_registry=*/nullptr, iree_allocator_system(), &context);
  iree_hal_queue_t dispatch_queue = {};
  IREE_ASSERT_OK(InitializeFakeHalContext(&context, &dispatch_queue));
  const loom_sanitizer_options_t sanitizer = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_context_add_module_runtime_requirements(
          &context, run_module.module, &sanitizer));

  loom_run_hal_testbench_context_deinitialize(&context);
  loom_run_module_deinitialize(&run_module);
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
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
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
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSelected32BitTargetCarrier) {
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
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], 3584u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSelected64BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex64Offset32TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsAddressInputWithoutSelectedTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(1),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, /*target_snapshot=*/nullptr,
          /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT32_MIN),
      I64Value(INT32_MAX),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {
      {
          /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          /*.flags=*/{},
          /*.size=*/4,
          /*.offset=*/0,
      },
      {
          /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          /*.flags=*/{},
          /*.size=*/4,
          /*.offset=*/4,
      },
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x80000000u);
  EXPECT_EQ(options.constants[1], 0x7fffffffu);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsIndexOutsideSignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(2147483648)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      /*.flags=*/{},
      /*.size=*/4,
      /*.offset=*/0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          input_parameters, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, MixedInputsUseReflectedWidthsAndOffsets) {
  loom_testbench_value_t inputs[] = {
      I64Value(3584),
      F32Value(4.0f),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {
      {
          /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          /*.flags=*/{},
          /*.size=*/8,
          /*.offset=*/0,
      },
      {
          /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          /*.flags=*/{},
          /*.size=*/4,
          /*.offset=*/8,
      },
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 3584u);
  EXPECT_EQ(options.constants[1], 0u);
  EXPECT_EQ(options.constants[2], 0x40800000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesSelected64BitTargetCarrier) {
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
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesSelected32BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967295)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex64Offset32TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], UINT32_MAX);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesReflectedFourByteWidth) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967295)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      /*.flags=*/{},
      /*.size=*/4,
      /*.offset=*/0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], UINT32_MAX);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsOffsetOutsideUnsignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967296)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      /*.flags=*/{},
      /*.size=*/4,
      /*.offset=*/0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          input_parameters, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, RejectsNegativeOffsetInput) {
  loom_testbench_value_t inputs[] = {
      I64Value(-1),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
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

TEST_F(HalTestbenchActualTest, RejectsCaseWithoutKernelLaunch) {
  static constexpr char kSource[] = R"(
check.case @host_only {
  %value = check.literal value(1) : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}
)";
  loom_run_module_t run_module = {};
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &run_module, &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);

  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_run_hal_testbench_select_kernel_launch(
                            &module_plan.cases[0], &kernel_launch));
  EXPECT_EQ(kernel_launch, nullptr);

  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalTestbenchActualTest, EvaluatesCompiledLaunchConfigForExactWorkload) {
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
  static constexpr char kLaunchConfigSource[] = R"(
func.def public pure @device_dynamic(%workgroup_count: index) -> (index, index, index, index, index, index, index, index, index, index, index) {
  %one = index.constant 1 : index
  %size = index.constant 64 : index
  %subgroup_size = index.constant 32 : index
  %storage = index.constant 0 : index
  func.return %workgroup_count, %one, %one, %size, %one, %one, %one, %one, %one, %subgroup_size, %storage : index, index, index, index, index, index, index, index, index, index, index
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

  loom_run_module_parse_options_t parse_options = {};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = IREE_SV("hal_testbench_launch_config.loom");
  parse_options.source = IREE_SV(kLaunchConfigSource);
  loom_run_module_t launch_config_module = {};
  IREE_ASSERT_OK(
      loom_run_module_parse(&session_, &parse_options, &launch_config_module));
  const loom_string_id_t launch_name_id = loom_module_lookup_string(
      launch_config_module.module, IREE_SV("device_dynamic"));
  const loom_symbol_id_t launch_symbol_id =
      loom_module_find_symbol(launch_config_module.module, launch_name_id);
  ASSERT_NE(launch_symbol_id, LOOM_SYMBOL_ID_INVALID);

  uint64_t workload_argument_bits[1] = {};
  loom_run_hal_testbench_context_t context = {};
  context.host_allocator = iree_allocator_system();
  loom_run_hal_testbench_actual_provider_t provider = {};
  provider.context = &context;
  provider.session = &session_;
  provider.run_module = &run_module;
  provider.kernel_launch = kernel_launch;
  provider.launch_config_module = launch_config_module.module;
  provider.launch_config_function = loom_kernel_launch_config_function_bind(
      launch_config_module.module,
      loom_func_like_cast(
          launch_config_module.module,
          launch_config_module.module->symbols.entries[launch_symbol_id]
              .defining_op));
  loom_pass_value_fact_owner_initialize(loom_run_session_block_pool(&session_),
                                        &provider.launch_config_fact_owner);
  provider.workload_argument_bits = workload_argument_bits;
  provider.prepared_candidate.target_bundle = &kFakeTargetBundle;
  provider.invocation_options.function_name = IREE_SV("device_dynamic");

  loom_testbench_value_materializer_options_t materializer_options = {};
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  loom_testbench_value_table_t value_table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
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
    EXPECT_EQ(provider.workload_argument_bits[0], expected_workgroup_count);
    EXPECT_EQ(invocation_options.workgroup_count[0], expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_count.x,
              expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_size.x, 64u);
    EXPECT_TRUE(iree_all_bits_set(
        provider.resolved_launch_config.fields,
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
            LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE));

    const loom_testbench_value_t* workload = nullptr;
    const loom_testbench_value_t* input = nullptr;
    IREE_ASSERT_OK(loom_testbench_value_table_lookup_borrow(
        &value_table, kernel_launch->workload_value_ids[0], &workload));
    IREE_ASSERT_OK(loom_testbench_value_table_lookup_borrow(
        &value_table, kernel_launch->input_value_ids[0], &input));
    loom_run_hal_invocation_options_t prepared_options = {};
    loom_run_hal_binding_list_t prepared_bindings = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_run_hal_testbench_actual_provider_materialize_invocation(
            &provider, /*workload_count=*/1, workload, /*input_count=*/1, input,
            &prepared_options, &prepared_bindings));
    provider.prepared_candidate_initialized = true;
    IREE_ASSERT_OK(
        loom_run_hal_testbench_actual_provider_materialize_invocation(
            &provider, /*workload_count=*/1, workload, /*input_count=*/1, input,
            &prepared_options, &prepared_bindings));
    provider.prepared_candidate_initialized = false;
    EXPECT_EQ(prepared_options.workgroup_count[0], expected_workgroup_count);
    EXPECT_EQ(prepared_options.constant_count, 1u);
    EXPECT_EQ(prepared_options.constants[0], expected_workgroup_count);
    EXPECT_EQ(prepared_bindings.count, 0u);
    loom_run_hal_binding_list_deinitialize(&prepared_bindings);
    loom_run_hal_binding_list_deinitialize(&bindings);
  }

  loom_testbench_value_table_deinitialize(&value_table);
  loom_pass_value_fact_owner_deinitialize(&provider.launch_config_fact_owner);
  loom_run_module_deinitialize(&launch_config_module);
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
  context.device_provider = &kFakeDeviceProvider;
  // An empty pipeline cannot produce the required launch program. Disable
  // transforms so this test can inspect the rooted clone at that boundary.
  context.runtime_initialized = true;
  context.host_allocator = iree_allocator_system();

  loom_run_hal_testbench_actual_provider_options_t options = {};
  options.context = &context;
  options.session = &session_;
  options.run_module = &run_module;
  options.pipeline = IREE_SV("none");
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
