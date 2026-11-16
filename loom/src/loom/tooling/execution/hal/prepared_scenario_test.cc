// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/hal/memory/heap_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/target/vm/testbench.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tooling/testbench/value_materializer.h"
#include "loomc/interop.h"
#include "loomc/target/configured.h"
#include "loomc/target/cpu/iree_hal.h"
#include "test/util.h"

namespace loom {
namespace {

using loomc::testing::HandlePtr;

using ByteSequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static const loom_run_hal_target_route_t kTargetRoutes[] = {
    {
        .driver_name = IREE_SV("task"),
        .provider = loomc_cpu_iree_hal_target_provider(),
    },
};

static constexpr char kScenarioSource[] = R"(
kernel.def @increment() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%storage: buffer) {
  %origin = index.constant 0 : offset
  %aligned = buffer.assume.alignment %storage {minimum_alignment = 4} : buffer
  %view = buffer.view %aligned[%origin] : buffer -> view<1xi32>
  %value = view.load %view[0] : view<1xi32> -> i32
  %one = scalar.constant 1 : i32
  %result = scalar.addi %value, %one : i32
  view.store %result, %view[0] : i32, view<1xi32>
  kernel.return
}

func.def @increment_oracle(%storage: buffer) {
  %origin = index.constant 0 : offset
  %aligned = buffer.assume.alignment %storage {minimum_alignment = 4} : buffer
  %view = buffer.view %aligned[%origin] : buffer -> view<1xi32>
  %value = view.load %view[0] : view<1xi32> -> i32
  %one = scalar.constant 1 : i32
  %result = scalar.addi %value, %one : i32
  view.store %result, %view[0] : i32, view<1xi32>
  func.return
}

check.scenario public @prepared_actual {
  check.trial[3](%trial: index, %entropy: check.entropy) {
    %storage = check.generate.fill value(41) : tensor<1xi32>
    check.compare<@increment, @increment_oracle>(%storage) : (tensor<1xi32>) -> () {
      check.expect.equal actual(%storage) expected(%storage) : tensor<1xi32>
    }
  }
  check.return
}
)";

static bool ExpectSucceededResult(const loomc_result_t* result) {
  if (result == nullptr) {
    ADD_FAILURE() << "compiler operation returned no result";
    return false;
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic != nullptr &&
        diagnostic->severity == LOOMC_DIAGNOSTIC_SEVERITY_ERROR) {
      ADD_FAILURE() << std::string(diagnostic->message.data,
                                   diagnostic->message.size);
    }
  }
  const bool succeeded = loomc_result_succeeded(result);
  EXPECT_TRUE(succeeded);
  return succeeded;
}

static iree_status_t ObserveResult(void* user_data,
                                   const loomc_result_t* result) {
  (void)user_data;
  ExpectSucceededResult(result);
  return iree_ok_status();
}

static ContextPtr CreateContext(
    loomc_target_environment_t* target_environment) {
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = target_environment,
  };
  const loomc_context_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

static WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

static CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context, /*options=*/nullptr,
                                        loomc_allocator_system(), &compiler));
  return CompilerPtr(compiler);
}

static ModulePtr DeserializeTextModule(loomc_context_t* context,
                                       loomc_workspace_t* workspace) {
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("prepared_scenario_test.loom"),
      .contents =
          loomc_make_byte_span(kScenarioSource, sizeof(kScenarioSource) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source_ptr.get(), /*options=*/nullptr,
      loomc_allocator_system(), &module, &result));
  ResultPtr result_ptr(result);
  if (!ExpectSucceededResult(result_ptr.get())) {
    loomc_module_release(module);
    module = nullptr;
  }
  return ModulePtr(module);
}

static ModulePtr DeserializeModule(loomc_context_t* context,
                                   loomc_workspace_t* workspace,
                                   loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, /*options=*/nullptr, loomc_allocator_system(),
      &module, &result));
  ResultPtr result_ptr(result);
  if (!ExpectSucceededResult(result_ptr.get())) {
    loomc_module_release(module);
    module = nullptr;
  }
  return ModulePtr(module);
}

static const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                            loomc_artifact_kind_t kind,
                                            const char* format) {
  const loomc_string_view_t expected_format =
      format != nullptr ? loomc_make_cstring_view(format)
                        : loomc_string_view_empty();
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        (format == nullptr ||
         loomc_string_view_equal(artifact->format, expected_format))) {
      return artifact;
    }
  }
  return nullptr;
}

typedef struct PreparedActualResolverState {
  // Prepared module reopened in the consumer context.
  const loomc_module_t* module;
  // Retained launch-config artifact reopened from package bytes.
  const loomc_artifact_t* launch_config;
  // Exact profile selected while producing the prepared module.
  loomc_target_profile_t* target_profile;
  // Exact public emitter format selected by the producer.
  loomc_string_view_t artifact_format;
  // Number of concrete products resolved by the scenario executor.
  iree_host_size_t resolve_count;
} PreparedActualResolverState;

static iree_status_t ResolvePreparedActual(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    loom_run_hal_testbench_prepared_actual_t* out_prepared_actual) {
  (void)configuration;
  PreparedActualResolverState* state =
      static_cast<PreparedActualResolverState*>(user_data);
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared test package requires a kernel subject");
  }
  ++state->resolve_count;
  *out_prepared_actual = {
      .module = state->module,
      .launch_config = state->launch_config,
      .target_profile = state->target_profile,
      .artifact_format = state->artifact_format,
  };
  return iree_ok_status();
}

class PreparedScenarioTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_target_environment_t* target_environment = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_configured(
        loomc_allocator_system(), &target_environment));
    target_environment_.reset(target_environment);

    loom_run_hal_testbench_context_initialize(
        target_environment_.get(), kTargetRoutes, IREE_ARRAYSIZE(kTargetRoutes),
        iree_allocator_system(), &hal_context_);

    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    const iree_hal_pool_host_access_t host_access = {
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .modes =
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
    };
    IREE_ASSERT_OK(
        iree_hal_heap_pool_create(host_access, iree_allocator_system(),
                                  iree_allocator_system(), &host_pool_));
    IREE_ASSERT_OK(loom_vm_testbench_initialize(
        target_environment_.get(), iree_allocator_system(), &vm_testbench_));
  }

  void TearDown() override {
    loom_vm_testbench_deinitialize(&vm_testbench_);
    iree_hal_pool_release(host_pool_);
    iree_arena_deinitialize(&plan_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
    loom_run_hal_testbench_context_deinitialize(&hal_context_);
    target_environment_.reset();
  }

  TargetEnvironmentPtr target_environment_;
  loom_run_hal_testbench_context_t hal_context_ = {};
  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t plan_arena_ = {};
  iree_hal_pool_t* host_pool_ = nullptr;
  loom_vm_testbench_t vm_testbench_ = {};
};

TEST_F(PreparedScenarioTest, ExecutesPreparedActualThroughScenarioProfile) {
  TargetProfilePtr packaged_target_profile;
  SourcePtr packaged_module_source;
  ByteSequencePtr packaged_launch_config_contents;
  std::string packaged_artifact_format;
  {
    ContextPtr producer_context = CreateContext(target_environment_.get());
    WorkspacePtr producer_workspace = CreateWorkspace();
    CompilerPtr producer_compiler = CreateCompiler(producer_context.get());
    ModulePtr producer_module =
        DeserializeTextModule(producer_context.get(), producer_workspace.get());
    ASSERT_NE(producer_module, nullptr);

    loomc_iree_hal_target_selection_t target_selection = {};
    loomc_result_t* target_result = nullptr;
    IREE_ASSERT_OK(loom_run_hal_testbench_context_select_target(
        &hal_context_, loomc_make_cstring_view("prepared scenario producer"),
        /*target_profile=*/nullptr, &target_selection, &target_result));
    ResultPtr target_result_ptr(target_result);
    ASSERT_TRUE(loomc_result_succeeded(target_result_ptr.get()));
    ASSERT_NE(target_selection.target_profile, nullptr);
    packaged_target_profile.reset(target_selection.target_profile);

    const loomc_string_view_t root = loomc_make_cstring_view("increment");
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    const loomc_compile_artifact_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        .structure_size = sizeof(compile_options),
        .roots = &root,
        .root_count = 1,
        .target_profile = packaged_target_profile.get(),
        .emit_options = &emit_options,
        .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
                          LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
    };
    loomc_result_t* compile_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_artifact(
        producer_compiler.get(), producer_workspace.get(),
        /*pass_program=*/nullptr, producer_module.get(), &compile_options,
        loomc_allocator_system(), &compile_result));
    ResultPtr compile_result_ptr(compile_result);
    ASSERT_TRUE(loomc_result_succeeded(compile_result_ptr.get()));

    const loomc_artifact_t* prepared_module_artifact =
        FindArtifact(compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                     LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    const loomc_artifact_t* launch_config_artifact = FindArtifact(
        compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    const loomc_artifact_t* executable_artifact =
        FindArtifact(compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                     /*format=*/nullptr);
    ASSERT_NE(prepared_module_artifact, nullptr);
    ASSERT_NE(launch_config_artifact, nullptr);
    ASSERT_NE(executable_artifact, nullptr);
    packaged_artifact_format.assign(executable_artifact->format.data,
                                    executable_artifact->format.size);

    loomc_source_t* module_source = nullptr;
    LOOMC_ASSERT_OK(loomc_artifact_create_source(
        prepared_module_artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
        loomc_allocator_system(), &module_source));
    packaged_module_source.reset(module_source);
    loomc_byte_sequence_retain(launch_config_artifact->contents);
    packaged_launch_config_contents.reset(launch_config_artifact->contents);
  }

  const loomc_artifact_t packaged_launch_config = {
      .kind = LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      .format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      .identifier = loomc_make_cstring_view("increment-launch.loombc"),
      .contents = packaged_launch_config_contents.get(),
  };
  ContextPtr consumer_context = CreateContext(target_environment_.get());
  WorkspacePtr consumer_workspace = CreateWorkspace();
  CompilerPtr oracle_compiler = CreateCompiler(consumer_context.get());
  ModulePtr authored_module =
      DeserializeTextModule(consumer_context.get(), consumer_workspace.get());
  ModulePtr prepared_module =
      DeserializeModule(consumer_context.get(), consumer_workspace.get(),
                        packaged_module_source.get());
  ASSERT_NE(authored_module, nullptr);
  ASSERT_NE(prepared_module, nullptr);

  loomc_module_interop_view_t authored_view = {};
  loomc_result_t* interop_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_get_interop_view(
      authored_module.get(), loomc_allocator_system(), &authored_view,
      &interop_result));
  ResultPtr interop_result_ptr(interop_result);
  ASSERT_TRUE(loomc_result_succeeded(interop_result_ptr.get()));

  loom_testbench_module_plan_t module_plan = {};
  IREE_ASSERT_OK(loom_testbench_plan_module(
      authored_view.module, /*options=*/nullptr, &plan_arena_, &module_plan));
  ASSERT_EQ(module_plan.issue_count, 0u);
  ASSERT_EQ(module_plan.scenario_count, 1u);
  const loom_testbench_scenario_plan_t* scenario = &module_plan.scenarios[0];
  ASSERT_EQ(scenario->trial_count, 1u);

  const loom_testbench_compilation_t target_compilation = {
      .context = consumer_context.get(),
      .compiler = nullptr,
      .workspace = consumer_workspace.get(),
      .module = authored_module.get(),
  };
  const loom_run_hal_testbench_actual_provider_options_t provider_options = {
      .context = &hal_context_,
      .compilation = &target_compilation,
      .module = authored_module.get(),
      .native_module = authored_view.module,
      .result_callback =
          {
              .fn = ObserveResult,
          },
  };
  PreparedActualResolverState resolver_state = {
      .module = prepared_module.get(),
      .launch_config = &packaged_launch_config,
      .target_profile = packaged_target_profile.get(),
      .artifact_format = loomc_make_string_view(
          packaged_artifact_format.data(), packaged_artifact_format.size()),
  };

  loomc_target_profile_t* mismatched_target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment_.get(), loomc_make_cstring_view("vm:core"),
      loomc_allocator_system(), &mismatched_target_profile));
  TargetProfilePtr mismatched_target_profile_ptr(mismatched_target_profile);
  loom_run_hal_testbench_actual_provider_options_t mismatch_options =
      provider_options;
  mismatch_options.invocation = &scenario->trials[0].action.target;
  mismatch_options.requested_target_profile = mismatched_target_profile;
  loom_run_hal_testbench_actual_provider_t mismatch_provider = {};
  loom_run_hal_testbench_actual_provider_initialize(&mismatch_options,
                                                    &mismatch_provider);
  const loom_run_hal_testbench_prepared_actual_t prepared_actual = {
      .module = resolver_state.module,
      .launch_config = resolver_state.launch_config,
      .target_profile = resolver_state.target_profile,
      .artifact_format = resolver_state.artifact_format,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_run_hal_testbench_actual_provider_load_prepared(
                            &mismatch_provider, &prepared_actual));
  loom_run_hal_testbench_actual_provider_deinitialize(&mismatch_provider);

  loom_run_hal_testbench_scenario_profile_t target_profile = {};
  loom_run_hal_testbench_prepared_scenario_profile_initialize(
      IREE_SV("task:prepared"), &provider_options,
      {
          .fn = ResolvePreparedActual,
          .user_data = &resolver_state,
      },
      authored_view.source_table, /*diagnostic_sink=*/{}, &target_profile);

  const loom_testbench_compilation_t oracle_compilation = {
      .context = consumer_context.get(),
      .compiler = oracle_compiler.get(),
      .workspace = consumer_workspace.get(),
      .module = authored_module.get(),
  };
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target =
      loom_run_hal_testbench_scenario_execution_profile(&target_profile);
  execution_options.oracle = loom_vm_testbench_execution_profile(
      &vm_testbench_, &oracle_compilation, authored_view.source_table,
      /*diagnostic_sink=*/{},
      {
          .fn = ObserveResult,
      });

  loom_testbench_value_materializer_options_t materializer_options;
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  materializer_options.target_pool = host_pool_;
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      authored_view.module, scenario, iree_allocator_system(), &configuration));
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer_options, loom_testbench_entropy_root(0x1020304050607080ull),
      /*configuration_ordinal=*/0, &configuration));

  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
      &execution_options, &configuration, &prepared));
  loom_testbench_scenario_trial_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &materializer_options,
      scenario->trials[0].trial_count, &executor));
  loom_testbench_scenario_trial_result_list_t results = {};
  IREE_ASSERT_OK(loom_testbench_run_scenario_trial_batch(
      &executor, /*first_trial_ordinal=*/0, scenario->trials[0].trial_count,
      &results));

  ASSERT_EQ(results.count, 3u);
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    EXPECT_TRUE(results.values[i].passed);
  }
  EXPECT_EQ(resolver_state.resolve_count, 1u);

  loom_testbench_scenario_trial_executor_deinitialize(&executor);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
}

}  // namespace
}  // namespace loom
