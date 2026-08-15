// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/program_plan.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/target/provider.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/iree.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/source.h"
#include "src/module.h"
#include "src/program_provider.h"
#include "src/result.h"
#include "src/target.h"
#include "test/util.h"

namespace {

using CompilerPtr =
    loomc::testing::HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr =
    loomc::testing::HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr =
    loomc::testing::HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    loomc::testing::HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProgramPlanPtr =
    loomc::testing::HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ResultPtr =
    loomc::testing::HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr =
    loomc::testing::HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    loomc::testing::HandlePtr<loomc_target_environment_t,
                              loomc_target_environment_release>;
using WorkspacePtr =
    loomc::testing::HandlePtr<loomc_workspace_t, loomc_workspace_release>;

struct TestPlanStorage {
  // Provider storage header consumed by the program-plan implementation.
  loomc_program_plan_storage_t base;

  // Hermetic unit module retained for public inspection.
  loomc_module_t* module;

  // Optional provider-storage destruction counter.
  int* destroy_count;
};

const loomc_module_t* TestUnitModule(const loomc_program_plan_storage_t* base,
                                     loomc_host_size_t /*unit_index*/) {
  const TestPlanStorage* storage =
      reinterpret_cast<const TestPlanStorage*>(base);
  return storage->module;
}

loomc_status_t CompileTestUnit(
    const loomc_program_plan_storage_t* /*base*/,
    loomc_compiler_t* /*compiler*/, loomc_workspace_t* /*workspace*/,
    loomc_host_size_t /*unit_index*/,
    const loomc_pass_program_t* /*pass_program*/,
    const loomc_program_plan_unit_compile_options_t* /*options*/,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  return loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator,
                             out_result);
}

void DestroyTestPlanStorage(loomc_program_plan_storage_t* base,
                            loomc_allocator_t allocator) {
  TestPlanStorage* storage = reinterpret_cast<TestPlanStorage*>(base);
  if (storage->destroy_count != nullptr) ++*storage->destroy_count;
  loomc_module_release(storage->module);
  loomc_allocator_free(allocator, storage);
}

const loomc_program_plan_operations_t kTestPlanOperations = {
    /*.unit_module=*/TestUnitModule,
    /*.compile_unit=*/CompileTestUnit,
    /*.destroy=*/DestroyTestPlanStorage,
};

bool OwnsTestProgramRoot(const loom_module_t* /*module*/,
                         const loom_op_t* /*root_op*/) {
  return true;
}

loomc_string_view_t TestRootName(const loom_module_t* module,
                                 const loom_op_t* root_op) {
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->defining_op == root_op) {
      return loomc_string_view_from_iree(
          module->strings.entries[symbol->name_id]);
    }
  }
  return loomc_string_view_empty();
}

loomc_status_t PrepareTestProgramPlan(
    loomc_workspace_t* workspace, const loomc_module_t* sealed_module,
    const loom_op_t* const* root_ops, loomc_host_size_t root_count,
    const loomc_program_plan_options_t* /*options*/, loomc_result_t* /*result*/,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  loomc_module_t* unit_module = nullptr;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_clone(sealed_module, workspace, allocator, &unit_module));

  TestPlanStorage* storage = nullptr;
  loomc_status_t status =
      loomc_allocator_malloc(allocator, sizeof(*storage), (void**)&storage);
  if (!loomc_status_is_ok(status)) {
    loomc_module_release(unit_module);
    return status;
  }
  *storage = {
      /*.base=*/{&kTestPlanOperations},
      /*.module=*/unit_module,
      /*.destroy_count=*/nullptr,
  };

  const loomc_program_plan_unit_t required_unit = {0};
  loomc_program_plan_root_create_params_t* roots = nullptr;
  status = loomc_status_from_iree(
      iree_allocator_malloc_array(iree_allocator_from_loomc(allocator),
                                  root_count, sizeof(*roots), (void**)&roots));
  for (loomc_host_size_t i = 0; i < root_count && loomc_status_is_ok(status);
       ++i) {
    roots[i] = {
        /*.name=*/TestRootName(loomc_module_const_loom_module(sealed_module),
                               root_ops[i]),
        /*.required_units=*/&required_unit,
        /*.required_unit_count=*/1,
    };
  }
  if (loomc_status_is_ok(status)) {
    const loomc_program_plan_create_params_t params = {
        /*.roots=*/roots,
        /*.root_count=*/root_count,
        /*.unit_count=*/1,
        /*.storage=*/&storage->base,
    };
    status = loomc_program_plan_create(&params, allocator, out_program_plan);
  }
  loomc_allocator_free(allocator, roots);
  if (!loomc_status_is_ok(status)) {
    DestroyTestPlanStorage(&storage->base, allocator);
  }
  return status;
}

const loomc_program_provider_t kTestProgramProvider = {
    /*.owns_root=*/OwnsTestProgramRoot,
    /*.prepare=*/PrepareTestProgramPlan,
};

const loomc_program_provider_t* const kTestProgramProviders[] = {
    &kTestProgramProvider,
};

const loomc_program_provider_set_t kTestProgramProviderSet =
    loomc_program_provider_set_make(kTestProgramProviders,
                                    IREE_ARRAYSIZE(kTestProgramProviders));

const loom_target_provider_set_t kEmptyTargetProviderSet = {0};

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ArtifactContents(const loomc_artifact_t* artifact) {
  return std::string(reinterpret_cast<const char*>(artifact->contents.data),
                     artifact->contents.data_length);
}

struct TestEnvironment {
  // Target environment carrying the test program provider.
  TargetEnvironmentPtr target_environment;

  // Loom context configured with |target_environment|.
  ContextPtr context;

  // Scratch workspace shared by serial test operations.
  WorkspacePtr workspace;

  TestEnvironment() {
    loomc_target_environment_t* target_environment_raw = nullptr;
    LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_sets(
        &kEmptyTargetProviderSet, &kTestProgramProviderSet,
        loomc_allocator_system(), &target_environment_raw));
    target_environment.reset(target_environment_raw);
    const loomc_context_target_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_environment=*/target_environment.get(),
    };
    const loomc_context_options_t context_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /*.structure_size=*/sizeof(context_options),
        /*.next=*/&target_options,
    };
    loomc_context_t* context_raw = nullptr;
    LOOMC_EXPECT_OK(loomc_context_create(
        &context_options, loomc_allocator_system(), &context_raw));
    context.reset(context_raw);
    loomc_workspace_t* workspace_raw = nullptr;
    LOOMC_EXPECT_OK(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &workspace_raw));
    workspace.reset(workspace_raw);
  }

  ModulePtr Parse(const char* source_text) {
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view("program_plan.loom"),
        /*.contents=*/loomc_make_byte_span(source_text, strlen(source_text)),
        /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
    };
    loomc_source_t* source_raw = nullptr;
    LOOMC_EXPECT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source_raw));
    SourcePtr source(source_raw);
    loomc_module_t* module_raw = nullptr;
    loomc_result_t* result_raw = nullptr;
    LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), source.get(), /*options=*/nullptr,
        loomc_allocator_system(), &module_raw, &result_raw));
    ResultPtr result(result_raw);
    EXPECT_TRUE(loomc_result_succeeded(result.get()));
    return ModulePtr(module_raw);
  }
};

TEST(ProgramPlanTest, PreparesExplicitRootsInCallerOrder) {
  TestEnvironment environment;
  ModulePtr module = environment.Parse(R"(
func.def public @prefill() {
  func.return
}
func.def public @decode() {
  func.return
}
func.def public @unused() {
  func.return
}
)");
  const loomc_string_view_t root_names[] = {
      loomc_make_cstring_view("@decode"),
      loomc_make_cstring_view("prefill"),
  };
  const loomc_compile_report_options_t report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(report_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_COMPILE_REPORT_MODE_DETAILS,
  };
  const loomc_program_plan_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&report_options,
  };
  loomc_program_plan_t* program_plan_raw = nullptr;
  loomc_result_t* result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_prepare(
      environment.workspace.get(), module.get(), root_names,
      IREE_ARRAYSIZE(root_names), &options, loomc_allocator_system(),
      &program_plan_raw, &result_raw));
  ProgramPlanPtr program_plan(program_plan_raw);
  ResultPtr result(result_raw);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_EQ(loomc_program_plan_root_count(program_plan.get()), 2u);

  loomc_program_plan_root_info_t decode_info = {};
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(
      program_plan.get(), loomc_program_plan_root_at(program_plan.get(), 0),
      &decode_info));
  EXPECT_EQ(ToString(decode_info.name), "decode");
  loomc_program_plan_root_info_t prefill_info = {};
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(
      program_plan.get(), loomc_program_plan_root_at(program_plan.get(), 1),
      &prefill_info));
  EXPECT_EQ(ToString(prefill_info.name), "prefill");

  loomc_program_plan_unit_info_t unit_info = {};
  LOOMC_ASSERT_OK(loomc_program_plan_unit_info(
      program_plan.get(), loomc_program_plan_unit_at(program_plan.get(), 0),
      &unit_info));
  ASSERT_NE(unit_info.module, nullptr);
  EXPECT_NE(unit_info.module, module.get());
  module.reset();

  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* report = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(report->format),
            LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  const std::string report_json = ArtifactContents(report);
  EXPECT_THAT(report_json, testing::HasSubstr(R"("kind":"loom.program_plan")"));
  EXPECT_THAT(report_json, testing::HasSubstr(R"("name":"decode")"));
  EXPECT_THAT(report_json, testing::HasSubstr(R"("name":"prefill")"));
  EXPECT_THAT(report_json,
              testing::Not(testing::HasSubstr(R"("name":"unused")")));
}

TEST(ProgramPlanTest, OwnsMetadataAndDispatchesUnitCompilation) {
  char root_name[] = "entry";
  loomc_program_plan_unit_t required_units[] = {{0}, {1}};
  const loomc_program_plan_root_create_params_t roots[] = {
      {
          /*.name=*/loomc_make_cstring_view(root_name),
          /*.required_units=*/required_units,
          /*.required_unit_count=*/IREE_ARRAYSIZE(required_units),
      },
  };
  int destroy_count = 0;
  TestPlanStorage* storage = nullptr;
  LOOMC_ASSERT_OK(loomc_allocator_malloc(loomc_allocator_system(),
                                         sizeof(*storage), (void**)&storage));
  *storage = {
      /*.base=*/{&kTestPlanOperations},
      /*.module=*/nullptr,
      /*.destroy_count=*/&destroy_count,
  };
  const loomc_program_plan_create_params_t params = {
      /*.roots=*/roots,
      /*.root_count=*/IREE_ARRAYSIZE(roots),
      /*.unit_count=*/2,
      /*.storage=*/&storage->base,
  };
  loomc_program_plan_t* program_plan_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_create(&params, loomc_allocator_system(),
                                            &program_plan_raw));
  ProgramPlanPtr program_plan(program_plan_raw);

  root_name[0] = 'X';
  required_units[0] = loomc_program_plan_unit_invalid();
  loomc_program_plan_root_info_t info = {};
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(
      program_plan.get(), loomc_program_plan_root_at(program_plan.get(), 0),
      &info));
  EXPECT_EQ(ToString(info.name), "entry");
  ASSERT_EQ(info.required_unit_count, 2u);
  EXPECT_EQ(info.required_units[0].value, 0u);
  EXPECT_EQ(info.required_units[1].value, 1u);

  TestEnvironment environment;
  loomc_compiler_t* compiler_raw = nullptr;
  LOOMC_ASSERT_OK(
      loomc_compiler_create(environment.context.get(), /*options=*/nullptr,
                            loomc_allocator_system(), &compiler_raw));
  CompilerPtr compiler(compiler_raw);
  loomc_pass_program_t* pass_program_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      environment.context.get(), /*options=*/nullptr, loomc_allocator_system(),
      &pass_program_raw));
  PassProgramPtr pass_program(pass_program_raw);
  loomc_result_t* result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      program_plan.get(), compiler.get(), environment.workspace.get(),
      loomc_program_plan_unit_at(program_plan.get(), 1), pass_program.get(),
      /*options=*/nullptr, loomc_allocator_system(), &result_raw));
  ResultPtr result(result_raw);
  EXPECT_TRUE(loomc_result_succeeded(result.get()));

  program_plan.reset();
  EXPECT_EQ(destroy_count, 1);
}

}  // namespace
