// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/interop.h"

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "iree/testing/gtest.h"
#include "loom/error/parameter.h"
#include "loom/pass/registry.h"
#include "loom/target/math_policy.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/source.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

ContextPtr CreateTargetContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

SourcePtr CreateSource(const char* identifier, const char* contents) {
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, std::strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

ModulePtr Deserialize(loomc_context_t* context, loomc_workspace_t* workspace,
                      const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_text_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

constexpr char kReportInput[] = R"(
func.def public @identity(%value: i32) -> (i32) {
  %unused = scalar.addi %value, %value : i32
  func.return %value : i32
}
)";

PassProgramPtr PrepareText(loomc_context_t* context, const char* text) {
  loomc_pass_program_t* program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view(text), nullptr, loomc_allocator_system(),
      &program, &result));
  ResultPtr preparation(result);
  EXPECT_TRUE(loomc_result_succeeded(result));
  return PassProgramPtr(program);
}

struct PassReport {
  PassReport() { loom_pass_report_initialize(iree_allocator_system(), &value); }
  ~PassReport() { loom_pass_report_deinitialize(&value); }
  // Invocation records owned by this execution.
  loom_pass_report_t value = {};
};

struct CompileReport {
  CompileReport() {
    loom_target_compile_report_initialize(&value, iree_allocator_system());
    value.requested_detail_flags =
        LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS;
  }
  ~CompileReport() { loom_target_compile_report_deinitialize(&value); }
  // Target observations owned by this execution.
  loom_target_compile_report_t value = {};
};

iree_status_t ObserveModule(loom_pass_t* pass, loom_module_t* module) {
  const iree_string_view_t function_name = loom_string_table_get(
      &module->strings, module->symbols.entries[0].name_id);
  const loom_pass_report_detail_field_t field =
      loom_pass_report_detail_string_field(IREE_SV("function"), function_name);
  IREE_RETURN_IF_ERROR(
      loom_pass_report_append_detail(pass, IREE_SV("observation"), &field, 1));
  loom_target_compile_report_t* report =
      loom_target_math_pass_capability_compile_report(
          loom_target_math_pass_capability_from_pass(pass));
  if (report != nullptr) {
    const loom_target_compile_report_target_capability_row_t row = {
        .function_name = function_name,
        .target_family_name = IREE_SVL("test"),
        .namespace_name = IREE_SVL("test"),
        .key = IREE_SVL("observed"),
        .value_kind = LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL,
        .value_u64 = 1,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_target_capability_row(report, &row));
  }
  return iree_ok_status();
}

iree_status_t RejectModule(loom_pass_t* pass, loom_module_t* module) {
  IREE_RETURN_IF_ERROR(ObserveModule(pass, module));
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("test.reject")),
      loom_param_string(IREE_SV("module")),
      loom_param_string(IREE_SV("identity")),
      loom_param_string(IREE_SV("rejected by native provider")),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 28),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(pass->diagnostic_emitter, &emission));
  return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

TargetEnvironmentPtr CreateReportingEnvironment() {
  static const loom_pass_descriptor_t descriptors[] = {
      {
          .key = IREE_SVL("test.observe"),
          .info =
              []() {
                static const loom_pass_info_t info = {
                    .name = IREE_SVL("test.observe"),
                    .kind = LOOM_PASS_MODULE,
                };
                return &info;
              },
          .module_run = ObserveModule,
      },
      {
          .key = IREE_SVL("test.reject"),
          .info =
              []() {
                static const loom_pass_info_t info = {
                    .name = IREE_SVL("test.reject"),
                    .kind = LOOM_PASS_MODULE,
                };
                return &info;
              },
          .module_run = RejectModule,
      },
  };
  static const loom_pass_registry_t registry = {
      .descriptors = descriptors,
      .descriptor_count = IREE_ARRAYSIZE(descriptors),
  };
  static const loom_target_provider_t provider = {.pass_registry = &registry};
  static const loom_target_provider_t* const providers[] = {&provider};
  static const loom_target_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_target_environment_t* environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &environment));
  return TargetEnvironmentPtr(environment);
}

ContextPtr CreateReportingContext() {
  auto environment = CreateReportingEnvironment();
  return CreateTargetContext(environment.get());
}

TEST(InteropTest, PassRegistryDescribesBuiltinsAndTargetPasses) {
  auto environment = CreateReportingEnvironment();
  loom_pass_registry_storage_t storage = {};
  LOOMC_ASSERT_OK(loomc_target_environment_get_native_pass_registry(
      environment.get(), &storage));
  const auto* registry = loom_pass_registry_storage_registry(&storage);
  const auto* canonicalize =
      loom_pass_registry_lookup(registry, IREE_SV("canonicalize"));
  ASSERT_NE(canonicalize, nullptr);
  EXPECT_TRUE(loom_pass_descriptor_is_available(canonicalize));
  ASSERT_EQ(canonicalize->option_schema_count, 1);
  EXPECT_EQ(canonicalize->option_schema[0].kind,
            LOOM_PASS_OPTION_SCHEMA_UINT32);
  EXPECT_EQ(canonicalize->option_schema[0].minimum_uint32, 1u);
  const auto* observer =
      loom_pass_registry_lookup(registry, IREE_SV("test.observe"));
  ASSERT_NE(observer, nullptr);
  EXPECT_EQ(observer->info()->kind, LOOM_PASS_MODULE);

  auto context = CreateTargetContext(environment.get());
  auto program = PrepareText(context.get(), "canonicalize,test.observe");
  EXPECT_NE(program.get(), nullptr);

  loom_pass_registry_storage_t builtins = {};
  LOOMC_ASSERT_OK(
      loomc_target_environment_get_native_pass_registry(nullptr, &builtins));
  const auto* builtin_registry = loom_pass_registry_storage_registry(&builtins);
  EXPECT_NE(loom_pass_registry_lookup(builtin_registry, IREE_SV("dce")),
            nullptr);
  EXPECT_EQ(
      loom_pass_registry_lookup(builtin_registry, IREE_SV("test.observe")),
      nullptr);
  EXPECT_EQ(registry->descriptor_count, builtin_registry->descriptor_count + 2);
}

TEST(InteropTest, PassRegistryRequiresOutputStorage) {
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_environment_get_native_pass_registry(nullptr, nullptr));
}

TEST(InteropTest, ProjectsVerifiedModuleAndOwnedSources) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());
  source.reset();

  loomc_module_interop_view_t view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &view, &result));
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(view.module, nullptr);
  ASSERT_NE(view.source_table, nullptr);
  ASSERT_EQ(view.source_table->module, view.module);
  ASSERT_EQ(view.source_table->count, 1u);
  EXPECT_TRUE(iree_string_view_equal(view.source_table->entries[0].filename,
                                     IREE_SV("module.loom")));
  EXPECT_FALSE(iree_string_view_is_empty(view.source_table->entries[0].source));
}

TEST(InteropTest, RejectsStructurallyInvalidModuleBeforeProjection) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("invalid.loom", R"(
func.def @missing_result(%value: i32) -> (i32, i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_module_interop_view_t view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &view, &result));
  ResultPtr result_ptr(result);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(view.module, nullptr);
  EXPECT_EQ(view.source_table, nullptr);

  bool found_structural_error = false;
  for (loomc_host_size_t i = 0;
       i < loomc_result_diagnostic_count(result_ptr.get()); ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result_ptr.get(), i);
    found_structural_error |=
        diagnostic != nullptr &&
        loomc_string_view_equal(diagnostic->code,
                                loomc_make_cstring_view("STRUCTURE/008"));
  }
  EXPECT_TRUE(found_structural_error);
}

TEST(InteropTest, MutableProjectionReturnsToPublicVerification) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_module_mutable_interop_view_t mutable_view =
      loomc_module_get_mutable_interop_view(module.get());
  ASSERT_NE(mutable_view.module, nullptr);
  ASSERT_NE(mutable_view.source_table, nullptr);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(mutable_view.module,
                                           IREE_SV("adapted"), &name_id));
  mutable_view.module->name_id = name_id;

  loomc_module_interop_view_t verified_view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &verified_view, &result));
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(verified_view.module, mutable_view.module);
  EXPECT_EQ(verified_view.source_table, mutable_view.source_table);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&verified_view.module->strings,
                            verified_view.module->name_id),
      IREE_SV("adapted")));
}

TEST(InteropTest, CreatesOwnedNativeTargetEnvironment) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  const loom_target_environment_t* native_environment =
      loomc_target_environment_get_interop_view(target_environment_ptr.get());
  ASSERT_NE(native_environment, nullptr);
  EXPECT_EQ(native_environment->provider_set, &provider_set);
}

TEST(InteropTest, BuildsNativeSourceLowPassProgram) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  ContextPtr context = CreateTargetContext(target_environment_ptr.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_native_source_low_pipeline(
      context.get(), loom_target_pipeline_build_to_source_low_artifacts,
      loomc_string_view_empty(), /*options=*/nullptr, loomc_allocator_system(),
      &pass_program, &result));
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_NE(pass_program_ptr.get(), nullptr);

  result_ptr.reset();
  pass_program_ptr.reset();
  context.reset();
  target_environment_ptr.reset();
}

TEST(InteropTest, CompilesPublicModuleWithNativeReport) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  ContextPtr context = CreateTargetContext(target_environment_ptr.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context.get(), /*options=*/nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_ptr(compiler);
  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_OK(
      loomc_pass_program_create_empty(context.get(), /*options=*/nullptr,
                                      loomc_allocator_system(), &pass_program));
  PassProgramPtr pass_program_ptr(pass_program);

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  const loomc_native_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .report = &report,
  };
  const loomc_compile_options_t options = {.next = &report_options};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module(
      compiler_ptr.get(), workspace.get(), pass_program_ptr.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_NE(report.allocator.ctl, nullptr);
  EXPECT_EQ(report.status_code, IREE_STATUS_OK);
  loom_target_compile_report_deinitialize(&report);

  result_ptr.reset();
  pass_program_ptr.reset();
  compiler_ptr.reset();
  module.reset();
  source.reset();
  workspace.reset();
  context.reset();
  target_environment_ptr.reset();
}

TEST(InteropTest, ComposesReportsAndTraceForRunAndCompile) {
  auto context = CreateReportingContext();
  auto workspace = CreateWorkspace();
  auto source = CreateSource("input.loom", kReportInput);
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  CompilerPtr compiler(raw_compiler);
  for (bool compile : {false, true}) {
    for (bool fail : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "compile=" << compile << " fail=" << fail);
      std::array<ModulePtr, 2> modules = {
          Deserialize(context.get(), workspace.get(), source.get()),
          Deserialize(context.get(), workspace.get(), source.get()),
      };
      auto program =
          PrepareText(context.get(),
                      fail ? "test.observe,test.reject" : "test.observe,dce");
      PassReport passes;
      CompileReport compilation;
      std::string trace_output;
      const loomc_pass_trace_options_t trace = {
          .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
          .structure_size = sizeof(trace),
          .flags = LOOMC_PASS_TRACE_FLAG_AFTER_ALL,
          .sink =
              {
                  .write =
                      [](void* user_data, loomc_string_view_t fragment) {
                        static_cast<std::string*>(user_data)->append(
                            fragment.data, fragment.size);
                        return loomc_ok_status();
                      },
                  .user_data = &trace_output,
              },
      };
      const loomc_native_compile_report_options_t compile_report = {
          .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
          .structure_size = sizeof(compile_report),
          .next = &trace,
          .report = &compilation.value,
      };
      const loomc_native_pass_report_options_t pass_report = {
          .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
          .structure_size = sizeof(pass_report),
          .next = &compile_report,
          .report = &passes.value,
      };
      // The two calls append independently observed rows to the same
      // collectors.
      for (int invocation = 0; invocation < 2; ++invocation) {
        loomc_result_t* raw_result = nullptr;
        if (compile) {
          const loomc_compile_options_t options = {.next = &pass_report};
          LOOMC_ASSERT_OK(loomc_compile_module(
              compiler.get(), workspace.get(), program.get(),
              modules[invocation].get(), &options, loomc_allocator_system(),
              &raw_result));
        } else {
          const loomc_pass_run_options_t options = {.next = &pass_report};
          LOOMC_ASSERT_OK(loomc_pass_program_run(
              program.get(), workspace.get(), modules[invocation].get(),
              &options, loomc_allocator_system(), &raw_result));
        }
        ResultPtr result(raw_result);
        EXPECT_EQ(loomc_result_status_code(result.get()),
                  fail ? LOOMC_STATUS_OUT_OF_RANGE : LOOMC_STATUS_OK);
        EXPECT_EQ(loomc_result_succeeded(result.get()), !fail);
        EXPECT_EQ(loomc_result_diagnostic_count(result.get()), fail ? 1u : 0u);
      }
      ASSERT_EQ(passes.value.invocation_count, 4u);
      EXPECT_EQ(passes.value.invocations[1].status_code,
                fail ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_OK);
      ASSERT_EQ(compilation.value.target_capability_rows.count, fail ? 4u : 2u);
      const auto* rows = static_cast<
          const loom_target_compile_report_target_capability_row_t*>(
          loom_target_compile_report_vec_const_rows(
              compilation.value.target_capability_rows.head));
      EXPECT_TRUE(
          iree_string_view_equal(rows[0].function_name, IREE_SV("identity")));
      EXPECT_NE(trace_output.find("pipeline-text"), std::string::npos);
      // Pass records own mutable pipeline/module names and detail payloads.
      for (auto& module : modules) {
        module.reset();
      }
      program.reset();
      const auto& invocation = passes.value.invocations[0];
      ASSERT_EQ(invocation.detail_count, 1u);
      EXPECT_TRUE(iree_string_view_equal(
          invocation.details->fields[0].string_value, IREE_SV("identity")));
      EXPECT_FALSE(iree_string_view_is_empty(invocation.pipeline_symbol));
    }
  }
}

TEST(InteropTest, RetainsCompletedReportsWhenTraceOrReportAllocationFails) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateSource("input.loom", kReportInput);
  auto program = PrepareText(context.get(), "dce,cse");
  for (bool fail_trace : {false, true}) {
    auto module = Deserialize(context.get(), workspace.get(), source.get());
    PassReport passes;
    // Fail subsequent record allocation after the first completed invocation.
    if (!fail_trace) {
      passes.value.allocator = {
          .self = &passes.value,
          .ctl =
              [](void* self, iree_allocator_command_t command,
                 const void* parameters, void** pointer) {
                const auto* report =
                    static_cast<const loom_pass_report_t*>(self);
                if (command != IREE_ALLOCATOR_COMMAND_FREE &&
                    report->invocation_count != 0) {
                  return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
                const auto allocator = iree_allocator_system();
                return allocator.ctl(allocator.self, command, parameters,
                                     pointer);
              },
      };
    }
    const loomc_pass_trace_options_t trace = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
        .structure_size = sizeof(trace),
        .flags = LOOMC_PASS_TRACE_FLAG_AFTER_ALL,
        .sink =
            {
                .write =
                    [](void*, loomc_string_view_t) {
                      return loomc_status_from_code(
                          LOOMC_STATUS_INVALID_ARGUMENT);
                    },
            },
    };
    const loomc_native_pass_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
        .structure_size = sizeof(report),
        .next = fail_trace ? &trace : nullptr,
        .report = &passes.value,
    };
    const loomc_pass_run_options_t options = {.next = &report};
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(
        fail_trace ? LOOMC_STATUS_INVALID_ARGUMENT
                   : LOOMC_STATUS_RESOURCE_EXHAUSTED,
        loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                               &options, loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
    ASSERT_EQ(passes.value.invocation_count, 1u);
    EXPECT_TRUE(iree_string_view_equal(passes.value.invocations[0].pass_key,
                                       IREE_SV("dce")));
    EXPECT_EQ(passes.value.invocations[0].status_code, IREE_STATUS_OK);
  }
}

TEST(InteropTest, RejectsMalformedNativeReportDescriptorsBeforeExecution) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateSource("input.loom", kReportInput);
  auto module = Deserialize(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "dce");
  PassReport passes;
  CompileReport compilation;
  for (int invalid_case = 0; invalid_case < 6; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    loomc_native_pass_report_options_t pass_report = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
        .structure_size = sizeof(pass_report),
        .report = &passes.value,
    };
    loomc_native_compile_report_options_t compile_report = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
        .structure_size = sizeof(compile_report),
        .next = &pass_report,
        .report = &compilation.value,
    };
    switch (invalid_case) {
      case 0:
        pass_report.structure_size = sizeof(pass_report) - 1;
        break;
      case 1:
        pass_report.report = nullptr;
        break;
      case 2:
        pass_report.next = &pass_report;
        break;
      case 3:
        compile_report.structure_size = sizeof(compile_report) - 1;
        break;
      case 4:
        compile_report.report = nullptr;
        break;
      case 5:
        compile_report.next = &compile_report;
        break;
    }
    const loomc_pass_run_options_t options = {.next = &compile_report};
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_INVALID_ARGUMENT,
        loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                               &options, loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(passes.value.invocation_count, 0u);
    EXPECT_EQ(compilation.value.target_capability_rows.count, 0u);
  }
}

TEST(InteropTest, SharesPreparedProgramAcrossIndependentReportOwners) {
  auto context = CreateReportingContext();
  auto source = CreateSource("input.loom", kReportInput);
  auto program = PrepareText(context.get(), "test.observe,dce");
  std::array<std::thread, 4> workers;
  for (auto& worker : workers) {
    worker = std::thread([&] {
      auto workspace = CreateWorkspace();
      auto module = Deserialize(context.get(), workspace.get(), source.get());
      PassReport passes;
      CompileReport compilation;
      const loomc_native_compile_report_options_t compile_report = {
          .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
          .report = &compilation.value,
      };
      const loomc_native_pass_report_options_t pass_report = {
          .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
          .next = &compile_report,
          .report = &passes.value,
      };
      const loomc_pass_run_options_t options = {.next = &pass_report};
      loomc_result_t* raw_result = nullptr;
      LOOMC_ASSERT_OK(loomc_pass_program_run(
          program.get(), workspace.get(), module.get(), &options,
          loomc_allocator_system(), &raw_result));
      ResultPtr result(raw_result);
      EXPECT_TRUE(loomc_result_succeeded(result.get()));
      EXPECT_EQ(passes.value.invocation_count, 2u);
      EXPECT_EQ(compilation.value.target_capability_rows.count, 1u);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

TEST(InteropTest,
     RequestCompilationRetainsPassReportsAndRejectsBorrowedReports) {
  auto context = CreateReportingContext();
  auto workspace = CreateWorkspace();
  auto source = CreateSource("input.loom", kReportInput);
  auto module = Deserialize(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "test.observe,dce");
  loomc_source_t* raw_bytecode = nullptr;
  LOOMC_ASSERT_OK(loomc_module_serialize_bytecode_to_source(
      module.get(), nullptr, loomc_allocator_system(), &raw_bytecode));
  SourcePtr bytecode(raw_bytecode);
  module.reset();
  const loomc_request_root_t root = {.module_ordinal = 0, .symbol_ordinal = 0};
  loomc_request_t* raw_request = nullptr;
  LOOMC_ASSERT_OK(loomc_request_create(
      loomc_compiled_module_product_descriptor(), bytecode.get(), &root, 1,
      nullptr, 0, loomc_allocator_system(), &raw_request));
  HandlePtr<loomc_request_t, loomc_request_release> request(raw_request);
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  CompilerPtr compiler(raw_compiler);
  PassReport passes;
  int trace_writes = 0;
  const loomc_pass_trace_options_t trace = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .flags = LOOMC_PASS_TRACE_FLAG_AFTER_ALL,
      .sink =
          {
              .write =
                  [](void* user_data, loomc_string_view_t) {
                    ++*static_cast<int*>(user_data);
                    return loomc_ok_status();
                  },
              .user_data = &trace_writes,
          },
  };
  const loomc_native_pass_report_options_t pass_report = {
      .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
      .next = &trace,
      .report = &passes.value,
  };
  const loomc_compile_options_t options = {.next = &pass_report};
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), program.get(), request.get(), &options,
      loomc_allocator_system(), &raw_product, &raw_result));
  HandlePtr<loomc_product_t, loomc_product_release> product(raw_product);
  ResultPtr result(raw_result);
  EXPECT_TRUE(loomc_result_succeeded(result.get()));
  EXPECT_GT(trace_writes, 0);
  ASSERT_EQ(passes.value.invocation_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      passes.value.invocations[0].details->fields[0].string_value,
      IREE_SV("identity")));

  CompileReport compilation;
  const loomc_native_compile_report_options_t compile_report = {
      .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
      .report = &compilation.value,
  };
  const loomc_compile_options_t borrowed_options = {.next = &compile_report};
  raw_product = nullptr;
  raw_result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_compile_request(compiler.get(), workspace.get(), program.get(),
                            request.get(), &borrowed_options,
                            loomc_allocator_system(), &raw_product,
                            &raw_result));
  EXPECT_EQ(raw_product, nullptr);
  EXPECT_EQ(raw_result, nullptr);
  EXPECT_EQ(compilation.value.target_capability_rows.count, 0u);
}

}  // namespace
