// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <string_view>

#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/object_admission_testdata.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using EnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string Diagnostics(const loomc_result_t* result) {
  std::string text;
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    text += ToString(loomc_result_diagnostic_at(result, i)->message) + "\n";
  }
  return text;
}

class ObjectAdmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto allocator = loomc_allocator_system();
    loomc_target_environment_t* environment = nullptr;
    LOOMC_ASSERT_OK(
        loomc_target_environment_create_vm(allocator, &environment));
    environment_.reset(environment);
    const loomc_context_target_options_t target = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target),
        .target_environment = environment,
    };
    const loomc_context_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target,
    };
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(&options, allocator, &context));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, allocator, &workspace));
    workspace_.reset(workspace);
    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(
        loomc_compiler_create(context, nullptr, allocator, &compiler));
    compiler_.reset(compiler);
    loomc_pass_program_t* empty = nullptr;
    LOOMC_ASSERT_OK(
        loomc_pass_program_create_empty(context, nullptr, allocator, &empty));
    empty_.reset(empty);
  }

  ModulePtr Deserialize(loomc_source_t* source) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result));
    ResultPtr owner(result);
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      ADD_FAILURE() << ToString(loomc_result_diagnostic_at(result, i)->message);
    }
    EXPECT_TRUE(loomc_result_succeeded(result));
    return ModulePtr(module);
  }

  SourcePtr Serialize(loomc_module_t* module, loomc_source_format_t format) {
    const loomc_module_serialize_options_t options = {.format = format};
    loomc_source_t* source = nullptr;
    LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
        module, &options, loomc_allocator_system(), &source));
    return SourcePtr(source);
  }

  std::string Text(loomc_module_t* module) {
    auto source = Serialize(module, LOOMC_SOURCE_FORMAT_TEXT);
    const auto span = loomc_source_contents(source.get());
    return std::string(reinterpret_cast<const char*>(span.data),
                       span.data_length);
  }

  void CheckPaths(loomc_module_t* original, bool prepared, bool pointer_cleanup,
                  const char* invalid_code) {
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      SCOPED_TRACE(format);
      auto serialized = Serialize(original, format);
      for (int path = 0; path < 3; ++path) {
        SCOPED_TRACE(path);
        auto module = Deserialize(serialized.get());
        ASSERT_NE(module, nullptr);
        const auto before = Text(module.get());
        const loomc_emit_options_t emit = {
            .artifact_format = loomc_make_cstring_view("vm"),
        };
        const loomc_compile_artifact_options_t options = {
            .emit_options = &emit,
        };
        loomc_result_t* result = nullptr;
        if (path == 2) {
          LOOMC_ASSERT_OK(loomc_emit_module(
              environment_.get(), workspace_.get(), module.get(), &emit,
              loomc_allocator_system(), &result));
        } else {
          LOOMC_ASSERT_OK(loomc_compile_artifact(
              compiler_.get(), workspace_.get(),
              path == 1 ? empty_.get() : nullptr, module.get(), &options,
              loomc_allocator_system(), &result));
        }
        ResultPtr result_owner(result);
        ASSERT_NE(result, nullptr);
        const bool expected = prepared || (pointer_cleanup && path == 0);
        EXPECT_EQ(loomc_result_succeeded(result), expected)
            << Diagnostics(result);
        if (!expected) {
          EXPECT_EQ(loomc_result_diagnostic_count(result), 1u);
          if (loomc_result_diagnostic_count(result) == 1) {
            EXPECT_EQ(ToString(loomc_result_diagnostic_at(result, 0)->code),
                      pointer_cleanup ? "DOMINANCE/020" : invalid_code)
                << Diagnostics(result);
          }
          EXPECT_EQ(loomc_result_artifact_count(result), 0u);
        }
        if (path == 2) {
          EXPECT_EQ(Text(module.get()), before);
        }
      }
    }
  }

  ModulePtr ParseFile(std::string_view filename) {
    const auto* files = loomc_object_admission_testdata_create();
    for (size_t i = 0; i < loomc_object_admission_testdata_size(); ++i) {
      const auto& file = files[i];
      if (filename != file.name) {
        continue;
      }
      const loomc_source_options_t options = {
          .format = LOOMC_SOURCE_FORMAT_TEXT,
          .identifier = loomc_make_cstring_view(file.name),
          .contents = loomc_make_byte_span(file.data, file.size),
          .storage = LOOMC_SOURCE_STORAGE_COPY,
      };
      loomc_source_t* source = nullptr;
      LOOMC_EXPECT_OK(
          loomc_source_create(&options, loomc_allocator_system(), &source));
      SourcePtr owner(source);
      return Deserialize(source);
    }
    ADD_FAILURE() << "Missing fixture " << filename;
    return nullptr;
  }

  // Immutable target tables used by parsing, compilation, and direct emission.
  EnvironmentPtr environment_;
  // Public compiler context, with no application-specific registrations.
  ContextPtr context_;
  // Invocation-owned scratch and module storage.
  WorkspacePtr workspace_;
  // Compiler for default and empty-pipeline artifact production.
  CompilerPtr compiler_;
  // Deliberately empty user pipeline; executable admission remains mandatory.
  PassProgramPtr empty_;
};

TEST_F(ObjectAdmissionTest, EveryArtifactPathChecksPreparedLifetimes) {
  struct Case {
    // Authored Low program exercised through text and bytecode admission.
    const char* filename;
    // Already prepared and accepted through every artifact path.
    bool prepared;
    // Accepted only when the default pipeline supplies pointer cleanup.
    bool pointer_cleanup;
    // Diagnostic for invalid ownership, independent of cleanup preparation.
    const char* invalid_code;
  };
  const Case cases[] = {
      {"object_prepared.loom", true, false, ""},
      {"object_storage_prepared.loom", true, false, ""},
      {"object_borrow_prepared.loom", true, false, ""},
      {"object_pointer_cleanup.loom", false, true, ""},
      {"object_borrow_cleanup.loom", false, true, ""},
      {"object_owner_leak.loom", false, false, "DOMINANCE/014"},
      {"object_storage_leak.loom", false, false, "DOMINANCE/014"},
      {"object_selection_leak.loom", false, false, "DOMINANCE/014"},
      {"object_borrow_consumed.loom-test", false, false, "DOMINANCE/002"},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.filename);
    auto original = ParseFile(test.filename);
    ASSERT_NE(original, nullptr);
    CheckPaths(original.get(), test.prepared, test.pointer_cleanup,
               test.invalid_code);
  }
}

TEST_F(ObjectAdmissionTest, LinkedLibraryCannotBypassAdmission) {
  for (bool prepared : {false, true}) {
    SCOPED_TRACE(prepared);
    auto input = ParseFile("object_link_root.loom");
    auto library = ParseFile(prepared ? "object_library_prepared.loom"
                                      : "object_library_leak.loom");
    ASSERT_NE(input, nullptr);
    ASSERT_NE(library, nullptr);
    loomc_linker_t* linker = nullptr;
    LOOMC_ASSERT_OK(loomc_linker_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &linker));
    HandlePtr<loomc_linker_t, loomc_linker_release> linker_owner(linker);
    const loomc_link_module_provider_t providers[] = {
        {
            .module = input.get(),
            .provider_name = loomc_make_cstring_view("input"),
            .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
        },
        {
            .module = library.get(),
            .provider_name = loomc_make_cstring_view("library"),
            .role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
        },
    };
    const auto root = loomc_make_cstring_view("entry");
    const loomc_link_options_t options = {
        .mode = LOOMC_LINK_MODE_LINK,
        .root_symbols = &root,
        .root_symbol_count = 1,
        .module_providers = providers,
        .module_provider_count = 2,
    };
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_module(linker, workspace_.get(), &options,
                                      &module, &result));
    ResultPtr result_owner(result);
    ModulePtr module_owner(module);
    ASSERT_TRUE(loomc_result_succeeded(result)) << Diagnostics(result);
    ASSERT_NE(module, nullptr);
    CheckPaths(module, prepared, false, "DOMINANCE/014");
  }
}

}  // namespace
