// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;

using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->code) << ": "
                  << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

void ExpectFailedResultCode(const loomc_result_t* result, const char* code) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  bool found = false;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ASSERT_NE(diagnostic, nullptr);
    found |= ToString(diagnostic->code) == code;
  }
  EXPECT_TRUE(found);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace, const char* text) {
  SourcePtr source = CreateTextSource("launch_config.loom", text);
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source.get(), nullptr, loomc_allocator_system(),
      &module, &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

void Evaluate(loomc_module_t* module, loomc_workspace_t* workspace,
              const loomc_launch_config_eval_options_t* options,
              loomc_launch_config_t* out_config, ResultPtr* out_result) {
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_evaluate_launch_config(
      module, workspace, options, loomc_allocator_system(), out_config,
      &result));
  out_result->reset(result);
}

loomc_launch_config_eval_options_t EvalOptions(
    const char* function_symbol,
    loomc_launch_config_field_flags_t required_fields) {
  loomc_launch_config_eval_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS;
  options.structure_size = sizeof(options);
  options.function_symbol = loomc_make_cstring_view(function_symbol);
  options.required_fields = required_fields;
  return options;
}

loomc_launch_config_t EmptyResultConfig() {
  loomc_launch_config_t config = {};
  config.type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG;
  config.structure_size = sizeof(config);
  return config;
}

TEST(LaunchConfigTest, EvaluatesConstantLaunchConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c5 = index.constant 5 : index
  %c6 = index.constant 6 : index
  kernel.launch.config workgroups(%c2, %c3, %c4) workgroup_size(%c5, %c6, %c1) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("@entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                                LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_TRUE(config.fields & LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 2u);
  EXPECT_EQ(config.workgroup_count.y, 3u);
  EXPECT_EQ(config.workgroup_count.z, 4u);
  EXPECT_TRUE(config.fields & LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 5u);
  EXPECT_EQ(config.workgroup_size.y, 6u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, EvaluatesConfigBackedLaunchConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
config.decl @shape.rows : index
config.decl @shape.cols : index
config.decl @tuning.workgroup_x : index

kernel.def @entry() {
  %one = index.constant 1 : index
  %rows = config.get @shape.rows : index
  %cols = config.get @shape.cols : index
  %workgroup_x = config.get @tuning.workgroup_x : index
  kernel.launch.config workgroups(%rows, %cols, %one) workgroup_size(%workgroup_x, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("shape.rows"),
          /*.value=*/loomc_make_cstring_view("7"),
      },
      {
          /*.key=*/loomc_make_cstring_view("shape.cols"),
          /*.value=*/loomc_make_cstring_view("8"),
      },
      {
          /*.key=*/loomc_make_cstring_view("tuning.workgroup_x"),
          /*.value=*/loomc_make_cstring_view("10"),
      },
  };
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                               LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  options.config.bindings = bindings;
  options.config.binding_count = 3;
  options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                         LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.workgroup_count.x, 7u);
  EXPECT_EQ(config.workgroup_count.y, 8u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 10u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, EvaluatesJsonConfigBackedLaunchConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
config.decl @shape.rows : index
config.decl @shape.cols : index

kernel.def @entry() {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  %rows = config.get @shape.rows : index
  %cols = config.get @shape.cols : index
  kernel.launch.config workgroups(%rows, %cols, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                               LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  options.config.json_object = loomc_make_cstring_view(R"({
        "shape": {
          "rows": 13,
          "cols": 17
        }
      })");
  options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                         LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.workgroup_count.x, 13u);
  EXPECT_EQ(config.workgroup_count.y, 17u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, ReportsInvalidConfigBindingAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
config.decl @shape.rows : index

kernel.def @entry() {
  %one = index.constant 1 : index
  %rows = config.get @shape.rows : index
  kernel.launch.config workgroups(%rows, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_config_binding_t bindings[] = {{
      /*.key=*/loomc_make_cstring_view("shape.cols"),
      /*.value=*/loomc_make_cstring_view("7"),
  }};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  options.config.bindings = bindings;
  options.config.binding_count = 1;
  options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "CONFIG/INVALID");
}

TEST(LaunchConfigTest, EvaluatesWorkloadArguments) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index, %cols: index) {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%rows, %cols, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  int64_t workload_arguments[] = {11, 12};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                               LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 2;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.workgroup_count.x, 11u);
  EXPECT_EQ(config.workgroup_count.y, 12u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, EvaluatesWorkloadArgumentLaunchMath) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index) {
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

  int64_t workload_arguments[] = {129};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                               LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 1;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.workgroup_count.x, 3u);
  EXPECT_EQ(config.workgroup_count.y, 1u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, EvaluatesIntegerWorkloadArgumentsThroughIndexCast) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows_i32: i32, %cols_i64: i64) {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  %rows = index.cast %rows_i32 : i32 to index
  %cols = index.cast %cols_i64 : i64 to index
  kernel.launch.config workgroups(%rows, %cols, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  int64_t workload_arguments[] = {9, 10};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                               LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 2;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.workgroup_count.x, 9u);
  EXPECT_EQ(config.workgroup_count.y, 10u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, AllowsPartialLaunchConfigResults) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index) {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%rows, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_FALSE(config.fields & LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_TRUE(config.fields & LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST(LaunchConfigTest, ReportsWorkloadArgumentsWithoutSignature) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  int64_t workload_arguments[] = {11};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 1;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/WORKLOAD_ARGUMENT_COUNT");
}

TEST(LaunchConfigTest, ReportsWorkloadArgumentCountMismatch) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index, %cols: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %cols, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  int64_t workload_arguments[] = {11};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 1;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/WORKLOAD_ARGUMENT_COUNT");
}

TEST(LaunchConfigTest, ReportsUnsupportedWorkloadArgumentType) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%scale: f32) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  int64_t workload_arguments[] = {1};
  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  options.workload_arguments = workload_arguments;
  options.workload_argument_count = 1;
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/WORKLOAD_ARGUMENT_TYPE");
}

TEST(LaunchConfigTest, ReportsMissingFunctionSymbol) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("missing", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/NOT_FOUND");
}

TEST(LaunchConfigTest, ReportsNonKernelFunctionSymbol) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
func.def @helper() {
  func.return
}

kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("helper", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/NOT_KERNEL");
}

TEST(LaunchConfigTest, ReportsMissingRequiredWorkgroupCount) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/MISSING_WORKGROUP_COUNT");
}

TEST(LaunchConfigTest, ReportsMissingRequiredSubgroupSize) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/MISSING_SUBGROUP_SIZE");
}

TEST(LaunchConfigTest, ReportsMissingRequiredWorkgroupSize) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%threads: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(), "LAUNCH_CONFIG/MISSING_WORKGROUP_SIZE");
}

TEST(LaunchConfigTest, ReportsMissingRequiredWorkgroupStorageBytes) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options = EvalOptions(
      "entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES);
  loomc_launch_config_t config = EmptyResultConfig();
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectFailedResultCode(result.get(),
                         "LAUNCH_CONFIG/MISSING_WORKGROUP_STORAGE_BYTES");
}

TEST(LaunchConfigTest, AcceptsZeroInitializedApiStructs) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options = {};
  options.function_symbol = loomc_make_cstring_view("entry");
  options.required_fields = LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                            LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE;
  loomc_launch_config_t config = {};
  ResultPtr result;
  Evaluate(module.get(), workspace.get(), &options, &config, &result);

  ExpectSucceededResult(result.get());
  EXPECT_EQ(config.type, LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG);
  EXPECT_EQ(config.structure_size, sizeof(config));
  EXPECT_EQ(config.workgroup_count.x, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
}

TEST(LaunchConfigTest, RejectsUnknownRequiredFieldBits) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options = EvalOptions(
      "entry", static_cast<loomc_launch_config_field_flags_t>(1u << 31));
  loomc_launch_config_t config = EmptyResultConfig();
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status = loomc_module_evaluate_launch_config(
      module.get(), workspace.get(), &options, loomc_allocator_system(),
      &config, &raw_result);

  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_result, nullptr);
}

TEST(LaunchConfigTest, RejectsEmptyFunctionSymbolBeforeResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  loomc_launch_config_t config = EmptyResultConfig();
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status = loomc_module_evaluate_launch_config(
      module.get(), workspace.get(), &options, loomc_allocator_system(),
      &config, &raw_result);

  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_result, nullptr);
}

TEST(LaunchConfigTest, RejectsMissingWorkloadArgumentPointerBeforeResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = DeserializeModule(context.get(), workspace.get(), R"(
kernel.def @entry(%rows: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  loomc_launch_config_eval_options_t options =
      EvalOptions("entry", LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  options.workload_argument_count = 1;
  loomc_launch_config_t config = EmptyResultConfig();
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status = loomc_module_evaluate_launch_config(
      module.get(), workspace.get(), &options, loomc_allocator_system(),
      &config, &raw_result);

  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_result, nullptr);
}

}  // namespace
