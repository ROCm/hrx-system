// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/module.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

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
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
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
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
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
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

void ExpectFailedResultCode(const loomc_result_t* result, const char* code) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  const loomc_host_size_t diagnostic_count =
      loomc_result_diagnostic_count(result);
  ASSERT_NE(diagnostic_count, 0u);
  bool found = false;
  for (loomc_host_size_t i = 0; i < diagnostic_count; ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ASSERT_NE(diagnostic, nullptr);
    found |= ToString(diagnostic->code) == code;
  }
  EXPECT_TRUE(found);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreateFunctionModule(loomc_context_t* context,
                               loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("functions.loom", R"(
func.def public @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

kernel.def export("dispatch") @entry() {
  %workgroups_z = index.constant 4 : index
  %workgroups_y = index.constant 3 : index
  %workgroups_x = index.constant 2 : index
  %workgroup_size_z = index.constant 1 : index
  %workgroup_size_y = index.constant 6 : index
  %workgroup_size_x = index.constant 5 : index
  kernel.launch.config workgroups(%workgroups_x, %workgroups_y, %workgroups_z) workgroup_size(%workgroup_size_x, %workgroup_size_y, %workgroup_size_z) : index
} launch() {
  kernel.return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

ModulePtr CreateMixedSymbolModule(loomc_context_t* context,
                                  loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("mixed_symbols.loom", R"(
global.constant @answer : index = 40

func.def public @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

global.variable @state : index = 0

kernel.def export("dispatch") @entry() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

const loomc_module_function_t* FindFunction(
    const std::vector<loomc_module_function_t>& functions, const char* name) {
  for (const loomc_module_function_t& function : functions) {
    if (ToString(function.symbol_name) == name) {
      return &function;
    }
  }
  return nullptr;
}

const loomc_module_global_t* FindGlobal(
    const std::vector<loomc_module_global_t>& globals, const char* name) {
  for (const loomc_module_global_t& global : globals) {
    if (ToString(global.symbol_name) == name) {
      return &global;
    }
  }
  return nullptr;
}

TEST(ModuleTest, QueriesFunctionsAndKernelSidecars) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_host_size_t function_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 0, nullptr,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr count_result(result);
  ExpectSucceededResult(count_result.get());
  ASSERT_EQ(function_count, 2u);

  std::vector<loomc_module_function_t> functions(function_count);
  result = nullptr;
  status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), functions.size(),
      functions.data(), &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr query_result(result);
  ExpectSucceededResult(query_result.get());
  ASSERT_EQ(function_count, functions.size());

  const loomc_module_function_t* helper = FindFunction(functions, "helper");
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(helper->function_ordinal, 0u);
  EXPECT_EQ(helper->kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
  EXPECT_TRUE(helper->flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
  loomc_module_function_export_info_t export_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_export_info_at(
      module.get(), helper->function_ordinal, &export_info));
  status = loomc_module_function_get_export_info_at(
      module.get(), helper->function_ordinal, &export_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);

  const loomc_module_function_t* entry = FindFunction(functions, "entry");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->function_ordinal, 1u);
  EXPECT_EQ(entry->kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  EXPECT_TRUE(entry->flags & LOOMC_MODULE_FUNCTION_FLAG_HAS_EXPORT_INFO);

  export_info = {};
  ASSERT_TRUE(loomc_module_function_try_get_export_info(module.get(), entry,
                                                        &export_info));
  ASSERT_TRUE(loomc_module_function_try_get_export_info_at(
      module.get(), entry->function_ordinal, &export_info));
  EXPECT_TRUE(export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL);
  EXPECT_EQ(ToString(export_info.export_symbol), "dispatch");
}

TEST(ModuleTest, LooksUpFunctionNamesWithOrWithoutSigil) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_t by_plain_name = {};
  loomc_status_t status = loomc_module_lookup_function(
      module.get(), loomc_make_cstring_view("entry"), &by_plain_name);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(by_plain_name.symbol_name), "entry");

  loomc_module_function_t by_symbol_name = {};
  ASSERT_TRUE(loomc_module_try_lookup_function(
      module.get(), loomc_make_cstring_view("@entry"), &by_symbol_name));
  EXPECT_EQ(by_symbol_name.function_ordinal, by_plain_name.function_ordinal);
  EXPECT_EQ(by_symbol_name.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
}

TEST(ModuleTest, CloneCopiesModuleIntoTargetWorkspace) {
  ContextPtr context = CreateContext();
  WorkspacePtr source_workspace = CreateWorkspace();
  ModulePtr source =
      CreateMixedSymbolModule(context.get(), source_workspace.get());
  source_workspace.reset();

  WorkspacePtr clone_workspace = CreateWorkspace();
  loomc_module_t* raw_clone = nullptr;
  loomc_status_t status =
      loomc_module_clone(source.get(), clone_workspace.get(),
                         loomc_allocator_system(), &raw_clone);
  LOOMC_ASSERT_OK(status);
  ModulePtr clone(raw_clone);
  clone_workspace.reset();
  source.reset();

  loomc_module_function_t function = {};
  status = loomc_module_lookup_function(
      clone.get(), loomc_make_cstring_view("@entry"), &function);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);

  loomc_module_global_t global = {};
  status = loomc_module_lookup_global(
      clone.get(), loomc_make_cstring_view("@answer"), &global);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(global.kind, LOOMC_MODULE_GLOBAL_KIND_CONSTANT);
}

TEST(ModuleTest, GetsFunctionsByPublicOrdinal) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_t function = {};
  ASSERT_TRUE(loomc_module_try_get_function_at(module.get(), 0, &function));
  EXPECT_EQ(ToString(function.symbol_name), "helper");
  EXPECT_EQ(function.function_ordinal, 0u);

  loomc_status_t status =
      loomc_module_get_function_at(module.get(), 1, &function);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(function.symbol_name), "entry");
  EXPECT_EQ(function.function_ordinal, 1u);

  EXPECT_FALSE(loomc_module_try_get_function_at(module.get(), 2, &function));
  status = loomc_module_get_function_at(module.get(), 2, &function);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
}

TEST(ModuleTest, QueryReportsTotalFunctionCountForPartialStorage) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_module_function_t function = {};
  loomc_host_size_t function_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 1, &function,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(function_count, 2u);
  EXPECT_FALSE(ToString(function.symbol_name).empty());
}

TEST(ModuleTest, RejectsMalformedFunctionViewsWithoutCrashing) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_t lookup_function = {};
  loomc_string_view_t invalid_symbol_name = loomc_make_string_view(nullptr, 1);
  EXPECT_FALSE(loomc_module_try_lookup_function(
      module.get(), invalid_symbol_name, &lookup_function));
  loomc_status_t status = loomc_module_lookup_function(
      module.get(), invalid_symbol_name, &lookup_function);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);

  loomc_module_function_t function = {};
  status = loomc_module_lookup_function(
      module.get(), loomc_make_cstring_view("entry"), &function);
  LOOMC_ASSERT_OK(status);
  function.symbol_name =
      loomc_make_string_view(nullptr, function.symbol_name.size);

  loomc_module_function_export_info_t export_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_export_info(
      module.get(), &function, &export_info));
  status = loomc_module_function_get_export_info(module.get(), &function,
                                                 &export_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
}

TEST(ModuleTest, ReportsNamedFunctionQueryMissInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateFunctionModule(context.get(), workspace.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_make_cstring_view("@missing"),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_module_function_t function = {};
  loomc_host_size_t function_count = 1;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 1, &function,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_EQ(function_count, 0u);
  ExpectFailedResultCode(result_ptr.get(), "MODULE_FUNCTION/NOT_FOUND");
}

TEST(ModuleTest, QueriesGlobalsInMixedModule) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.global_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_GLOBAL_KIND_UNKNOWN,
  };
  loomc_host_size_t global_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_globals(
      module.get(), &options, loomc_allocator_system(), 0, nullptr,
      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr count_result(result);
  ExpectSucceededResult(count_result.get());
  ASSERT_EQ(global_count, 2u);

  std::vector<loomc_module_global_t> globals(global_count);
  result = nullptr;
  status = loomc_module_query_globals(module.get(), &options,
                                      loomc_allocator_system(), globals.size(),
                                      globals.data(), &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr query_result(result);
  ExpectSucceededResult(query_result.get());
  ASSERT_EQ(global_count, globals.size());

  const loomc_module_global_t* answer = FindGlobal(globals, "answer");
  ASSERT_NE(answer, nullptr);
  EXPECT_EQ(answer->global_ordinal, 0u);
  EXPECT_EQ(answer->kind, LOOMC_MODULE_GLOBAL_KIND_CONSTANT);

  const loomc_module_global_t* state = FindGlobal(globals, "state");
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->global_ordinal, 1u);
  EXPECT_EQ(state->kind, LOOMC_MODULE_GLOBAL_KIND_VARIABLE);

  loomc_module_function_query_options_t function_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(function_options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_host_size_t function_count = 0;
  result = nullptr;
  status = loomc_module_query_functions(module.get(), &function_options,
                                        loomc_allocator_system(), 0, nullptr,
                                        &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr function_result(result);
  ExpectSucceededResult(function_result.get());
  EXPECT_EQ(function_count, 2u);
}

TEST(ModuleTest, FiltersGlobalsByKind) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.global_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_GLOBAL_KIND_CONSTANT,
  };
  loomc_module_global_t global = {};
  loomc_host_size_t global_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_globals(
      module.get(), &options, loomc_allocator_system(), 1, &global,
      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr constant_result(result);
  ExpectSucceededResult(constant_result.get());
  ASSERT_EQ(global_count, 1u);
  EXPECT_EQ(ToString(global.symbol_name), "answer");
  EXPECT_EQ(global.kind, LOOMC_MODULE_GLOBAL_KIND_CONSTANT);

  options.kind = LOOMC_MODULE_GLOBAL_KIND_VARIABLE;
  global = {};
  result = nullptr;
  status = loomc_module_query_globals(module.get(), &options,
                                      loomc_allocator_system(), 1, &global,
                                      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr variable_result(result);
  ExpectSucceededResult(variable_result.get());
  ASSERT_EQ(global_count, 1u);
  EXPECT_EQ(ToString(global.symbol_name), "state");
  EXPECT_EQ(global.kind, LOOMC_MODULE_GLOBAL_KIND_VARIABLE);

  options.kind = static_cast<loomc_module_global_kind_t>(1234);
  result = nullptr;
  status = loomc_module_query_globals(module.get(), &options,
                                      loomc_allocator_system(), 0, nullptr,
                                      &global_count, &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(ModuleTest, LooksUpGlobalNamesWithOrWithoutSigil) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_t by_plain_name = {};
  loomc_status_t status = loomc_module_lookup_global(
      module.get(), loomc_make_cstring_view("answer"), &by_plain_name);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(by_plain_name.symbol_name), "answer");
  EXPECT_EQ(by_plain_name.kind, LOOMC_MODULE_GLOBAL_KIND_CONSTANT);

  loomc_module_global_t by_symbol_name = {};
  ASSERT_TRUE(loomc_module_try_lookup_global(
      module.get(), loomc_make_cstring_view("@answer"), &by_symbol_name));
  EXPECT_EQ(by_symbol_name.global_ordinal, by_plain_name.global_ordinal);
  EXPECT_EQ(by_symbol_name.kind, LOOMC_MODULE_GLOBAL_KIND_CONSTANT);
}

TEST(ModuleTest, GetsGlobalsByOrdinal) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_t global = {};
  ASSERT_TRUE(loomc_module_try_get_global_at(module.get(), 0, &global));
  EXPECT_EQ(ToString(global.symbol_name), "answer");
  EXPECT_EQ(global.global_ordinal, 0u);

  loomc_status_t status = loomc_module_get_global_at(module.get(), 1, &global);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(global.symbol_name), "state");
  EXPECT_EQ(global.global_ordinal, 1u);

  EXPECT_FALSE(loomc_module_try_get_global_at(module.get(), 2, &global));
  status = loomc_module_get_global_at(module.get(), 2, &global);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
}

TEST(ModuleTest, QueryReportsTotalGlobalCountForPartialStorage) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.global_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_GLOBAL_KIND_UNKNOWN,
  };
  loomc_module_global_t global = {};
  loomc_host_size_t global_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_globals(
      module.get(), &options, loomc_allocator_system(), 1, &global,
      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(global_count, 2u);
  EXPECT_FALSE(ToString(global.symbol_name).empty());
}

TEST(ModuleTest, HandlesMissingAndWrongKindGlobalProbesWithoutStatus) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_t global = {};
  EXPECT_FALSE(loomc_module_try_lookup_global(
      module.get(), loomc_make_cstring_view("missing"), &global));
  EXPECT_EQ(global.global_ordinal, 0u);
  EXPECT_TRUE(loomc_string_view_is_empty(global.symbol_name));

  EXPECT_FALSE(loomc_module_try_lookup_global(
      module.get(), loomc_make_cstring_view("entry"), &global));
  EXPECT_EQ(global.global_ordinal, 0u);
  EXPECT_TRUE(loomc_string_view_is_empty(global.symbol_name));

  loomc_status_t status = loomc_module_lookup_global(
      module.get(), loomc_make_cstring_view("entry"), &global);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
}

TEST(ModuleTest, RejectsMalformedGlobalQueriesWithoutCrashing) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_t global = {};
  loomc_string_view_t invalid_symbol_name = loomc_make_string_view(nullptr, 1);
  EXPECT_FALSE(loomc_module_try_lookup_global(module.get(), invalid_symbol_name,
                                              &global));
  loomc_status_t status =
      loomc_module_lookup_global(module.get(), invalid_symbol_name, &global);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);

  loomc_module_global_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.global_symbol=*/invalid_symbol_name,
      /*.kind=*/LOOMC_MODULE_GLOBAL_KIND_UNKNOWN,
  };
  loomc_host_size_t global_count = 0;
  loomc_result_t* result = nullptr;
  status = loomc_module_query_globals(module.get(), &options,
                                      loomc_allocator_system(), 0, nullptr,
                                      &global_count, &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(ModuleTest, ReportsNamedGlobalQueryMissInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateMixedSymbolModule(context.get(), workspace.get());

  loomc_module_global_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.global_symbol=*/loomc_make_cstring_view("@missing"),
      /*.kind=*/LOOMC_MODULE_GLOBAL_KIND_UNKNOWN,
  };
  loomc_module_global_t global = {};
  loomc_host_size_t global_count = 1;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_globals(
      module.get(), &options, loomc_allocator_system(), 1, &global,
      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr missing_result(result);
  EXPECT_EQ(global_count, 0u);
  ExpectFailedResultCode(missing_result.get(), "MODULE_GLOBAL/NOT_FOUND");

  options.global_symbol = loomc_make_cstring_view("@helper");
  result = nullptr;
  global_count = 1;
  status = loomc_module_query_globals(module.get(), &options,
                                      loomc_allocator_system(), 1, &global,
                                      &global_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr wrong_kind_result(result);
  EXPECT_EQ(global_count, 0u);
  ExpectFailedResultCode(wrong_kind_result.get(), "MODULE_GLOBAL/NOT_FOUND");
}

}  // namespace
