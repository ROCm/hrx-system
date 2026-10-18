// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/module_config.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/config/schema.h"
#include "loom/util/stream.h"
#include "loomc/pass.h"
#include "module.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ReadSchema(const loomc_result_t* result) {
  EXPECT_TRUE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_artifact_count(result), 1u);
  const auto* artifact = loomc_result_artifact_at(result, 0);
  if (!artifact) {
    return {};
  }
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_JSON);
  EXPECT_EQ(ToString(artifact->identifier), "config-schema.json");
  std::string json;
  const loomc_byte_sequence_callback_t callback = {
      .fn =
          [](void* user_data, loomc_byte_span_t segment) {
            static_cast<std::string*>(user_data)->append(
                reinterpret_cast<const char*>(segment.data),
                segment.data_length);
            return loomc_ok_status();
          },
      .user_data = &json,
  };
  LOOMC_EXPECT_OK(loomc_byte_sequence_enumerate(artifact->contents, callback));
  return json;
}

class ModuleConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(nullptr, loomc_allocator_system(), &context));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);
  }

  ModulePtr CreateModule(const char* text = R"(
config.decl @answer : %value: index where [range(%value, 0, 128), multiple_of(%value, 16)]
config.decl @limit : index
func.def public @read_config() -> (index) {
  %answer = config.get @answer : index
  func.return %answer : index
}
)") {
    const loomc_source_options_t options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("configuration.loom"),
        .contents = loomc_make_byte_span(text, strlen(text)),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_EXPECT_OK(
        loomc_source_create(&options, loomc_allocator_system(), &source));
    SourcePtr source_owner(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_deserialize_text_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result));
    ResultPtr result_owner(result);
    EXPECT_TRUE(loomc_result_succeeded(result));
    return ModulePtr(module);
  }

  ResultPtr Apply(loomc_module_t* module,
                  const loomc_module_config_options_t* options) {
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_apply_config(
        module, workspace_.get(), options, loomc_allocator_system(), &result));
    return ResultPtr(result);
  }

  ResultPtr Schema(loomc_module_t* module) {
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_query_config_schema(
        module, loomc_allocator_system(), &result));
    return ResultPtr(result);
  }

  // Context shared by the operations in one caller flow.
  ContextPtr context_;
  // Module and transient storage for that flow.
  WorkspacePtr workspace_;
};

TEST_F(ModuleConfigTest, AppliesOverridesBeforePreparedExecution) {
  auto module = CreateModule();
  std::string key = "@answer";
  std::string value = "64";
  std::string json = R"({"answer":16,"limit":7,"unused":1})";
  loomc_config_binding_t binding = {
      .key = loomc_make_cstring_view(key.c_str()),
      .value = loomc_make_cstring_view(value.c_str()),
  };
  const loomc_module_config_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS,
      .structure_size = sizeof(options),
      .config =
          {
              .bindings = &binding,
              .binding_count = 1,
              .json_object = loomc_make_cstring_view(json.c_str()),
          },
  };
  auto applied = Apply(module.get(), &options);
  ASSERT_TRUE(loomc_result_succeeded(applied.get()));
  key.assign(key.size(), '?');
  value.assign(value.size(), '?');
  json.assign(json.size(), '?');
  applied.reset();

  loomc_pass_program_t* program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
      context_.get(), loomc_make_cstring_view("canonicalize,dce"), nullptr,
      loomc_allocator_system(), &program, &result));
  PassProgramPtr program_owner(program);
  ResultPtr preparation(result);
  ASSERT_TRUE(loomc_result_succeeded(result));
  LOOMC_ASSERT_OK(loomc_pass_program_run(program, workspace_.get(),
                                         module.get(), nullptr,
                                         loomc_allocator_system(), &result));
  ResultPtr execution(result);
  ASSERT_TRUE(loomc_result_succeeded(result));
  const loomc_module_config_options_t final_options = {
      .config = {.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED},
  };
  auto final_result = Apply(module.get(), &final_options);
  EXPECT_TRUE(loomc_result_succeeded(final_result.get()));
  const auto* bindings = loomc_module_config_bindings(module.get());
  ASSERT_NE(bindings->head, nullptr);
  EXPECT_EQ(std::string(bindings->head->key.data, bindings->head->key.size),
            "limit");
  ASSERT_NE(bindings->head->next, nullptr);
  EXPECT_EQ(std::string(bindings->head->next->value.data,
                        bindings->head->next->value.size),
            "64");
  EXPECT_EQ(bindings->head->next, bindings->tail);
  auto schema = Schema(module.get());
  EXPECT_NE(ReadSchema(schema.get()).find(R"("default":"64")"),
            std::string::npos);
}

TEST_F(ModuleConfigTest, SchemaMatchesNativeAndOutlivesCompiler) {
  auto module = CreateModule();
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_config_format_schema_json(
      loomc_module_const_loom_module(module.get()), &stream));
  const auto native_view = iree_string_builder_view(&builder);
  const std::string expected(native_view.data, native_view.size);
  iree_string_builder_deinitialize(&builder);

  auto schema = Schema(module.get());
  module.reset();
  workspace_.reset();
  context_.reset();
  EXPECT_EQ(ReadSchema(schema.get()), expected);
}

TEST_F(ModuleConfigTest, FailedResolutionPreservesAppliedValues) {
  auto module = CreateModule();
  loomc_module_config_options_t options = {
      .config = {.json_object = loomc_make_cstring_view(R"({"answer":32})")},
  };
  auto applied = Apply(module.get(), &options);
  ASSERT_TRUE(loomc_result_succeeded(applied.get()));
  const auto* first = loomc_module_config_bindings(module.get())->head;
  options.config = {.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  auto unresolved = Apply(module.get(), &options);
  ASSERT_FALSE(loomc_result_succeeded(unresolved.get()));
  EXPECT_EQ(loomc_module_config_bindings(module.get())->head, first);
  ASSERT_NE(loomc_result_diagnostic_count(unresolved.get()), 0u);
  EXPECT_NE(ToString(loomc_result_diagnostic_at(unresolved.get(), 0)->message)
                .find("limit"),
            std::string::npos);
  auto schema = Schema(module.get());
  EXPECT_EQ(loomc_module_config_bindings(module.get())->head, first);
  EXPECT_NE(ReadSchema(schema.get()).find(R"("default":"32")"),
            std::string::npos);
  options.config = {
      .json_object = loomc_make_cstring_view(R"({"limit":8})"),
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  auto resolved = Apply(module.get(), &options);
  EXPECT_TRUE(loomc_result_succeeded(resolved.get()));
  EXPECT_EQ(loomc_module_config_bindings(module.get())->head, first);
  EXPECT_NE(first->next, nullptr);
}

TEST_F(ModuleConfigTest, ReportsInvalidConfigurationAndIrInResults) {
  for (const char* json :
       {R"({"answer":17})", R"({"answer":16,"answer":32})", "[16]"}) {
    SCOPED_TRACE(json);
    auto module = CreateModule();
    const loomc_module_config_options_t options = {
        .config = {.json_object = loomc_make_cstring_view(json)},
    };
    auto result = Apply(module.get(), &options);
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(loomc_result_succeeded(result.get()));
    ASSERT_NE(loomc_result_diagnostic_count(result.get()), 0u);
    EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
              "CONFIG/INVALID");
    EXPECT_EQ(loomc_module_config_bindings(module.get())->head, nullptr);
  }
  auto invalid = CreateModule("func.def @invalid() -> (i32) { func.return }\n");
  auto result = Apply(invalid.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  auto schema = Schema(invalid.get());
  EXPECT_FALSE(loomc_result_succeeded(schema.get()));
  EXPECT_EQ(loomc_result_artifact_count(schema.get()), 0u);
}

TEST_F(ModuleConfigTest, RejectsInvalidDescriptorsBeforeMutation) {
  auto module = CreateModule();
  const loomc_config_binding_t invalid_binding = {
      .key = {.data = nullptr, .size = 1},
  };
  const loomc_module_config_options_t invalid_options[] = {
      {.type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS},
      {.structure_size = sizeof(loomc_module_config_options_t) - 1},
      {.config = {.binding_count = 1}},
      {.config = {.bindings = &invalid_binding, .binding_count = 1}},
      {.config = {.json_object = {.data = nullptr, .size = 1}}},
      {.config = {.flags = UINT32_MAX}},
  };
  for (const auto& options : invalid_options) {
    loomc_result_t* result = reinterpret_cast<loomc_result_t*>(0x1);
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_INVALID_ARGUMENT,
        loomc_module_apply_config(module.get(), workspace_.get(), &options,
                                  loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
  }
  const loomc_module_config_options_t extension = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS,
      .structure_size = sizeof(extension),
  };
  const loomc_module_config_options_t options = {.next = &extension};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_module_apply_config(module.get(), workspace_.get(), &options,
                                loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_module_apply_config(nullptr, workspace_.get(), nullptr,
                                loomc_allocator_system(), &result));
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_module_apply_config(module.get(), nullptr, nullptr,
                                loomc_allocator_system(), &result));
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_module_apply_config(module.get(), workspace_.get(), nullptr,
                                loomc_allocator_system(), nullptr));
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_module_query_config_schema(
                             nullptr, loomc_allocator_system(), &result));
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_module_query_config_schema(
                             module.get(), loomc_allocator_system(), nullptr));
}

struct OperationAllocator {
  // Allocation ordinal to fail, or the maximum to allow all requests.
  size_t failure_at = LOOMC_HOST_SIZE_MAX;
  // Number of allocation requests made by the operation.
  size_t allocation_count = 0;
  // Live allocations owned by the operation or its result.
  size_t live_count = 0;

  static loomc_status_t Control(void* user_data,
                                loomc_allocator_command_t command,
                                const void* params, void** pointer) {
    auto& self = *static_cast<OperationAllocator*>(user_data);
    const bool was_live = *pointer != nullptr;
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE &&
        self.allocation_count++ == self.failure_at) {
      return loomc_status_from_code(LOOMC_STATUS_RESOURCE_EXHAUSTED);
    }
    auto system = loomc_allocator_system();
    auto status = system.ctl(system.self, command, params, pointer);
    if (loomc_status_is_ok(status)) {
      if (command == LOOMC_ALLOCATOR_COMMAND_FREE) {
        self.live_count -= was_live;
      } else if (!was_live && *pointer) {
        ++self.live_count;
      }
    }
    return status;
  }
};

TEST_F(ModuleConfigTest, AllocationFailuresReleasePartialResults) {
  for (int operation = 0; operation < 3; ++operation) {
    SCOPED_TRACE(operation);
    const loomc_module_config_options_t options = {
        .config =
            {
                .json_object = loomc_make_cstring_view(
                    operation == 0 ? R"({"answer":32,"limit":7})"
                                   : R"({"answer":17})"),
            },
    };
    size_t allocation_count = 0;
    // Establish the complete allocation path, then fail each request in turn.
    for (size_t attempt = 0; attempt <= allocation_count; ++attempt) {
      SCOPED_TRACE(attempt);
      auto module = CreateModule();
      OperationAllocator tracking;
      tracking.failure_at = attempt == 0 ? LOOMC_HOST_SIZE_MAX : attempt - 1;
      const loomc_allocator_t allocator = {&tracking,
                                           OperationAllocator::Control};
      loomc_result_t* result = nullptr;
      auto status = operation == 2 ? loomc_module_query_config_schema(
                                         module.get(), allocator, &result)
                                   : loomc_module_apply_config(
                                         module.get(), workspace_.get(),
                                         &options, allocator, &result);
      if (attempt == 0) {
        LOOMC_ASSERT_OK(status);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(loomc_result_succeeded(result), operation != 1);
        allocation_count = tracking.allocation_count;
      } else {
        LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_RESOURCE_EXHAUSTED, status);
        EXPECT_EQ(result, nullptr);
      }
      loomc_result_release(result);
      EXPECT_EQ(tracking.live_count, 0u);
    }
  }
}

}  // namespace
