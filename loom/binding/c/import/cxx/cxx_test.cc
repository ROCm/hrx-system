// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/import/cxx.h"

#include <array>
#include <atomic>
#include <string>
#include <thread>
#include <utility>

#include "iree/testing/gtest.h"
#include "loomc/link.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using PassPtr = HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using LinkBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;

std::string ToString(loomc_string_view_t value) {
  return value.size ? std::string(value.data, value.size) : std::string();
}

std::string Contents(const loomc_source_t* source) {
  auto value = loomc_source_contents(source);
  return value.data_length
             ? std::string(reinterpret_cast<const char*>(value.data),
                           value.data_length)
             : std::string();
}

SourcePtr Source(const char* name, const char* text) {
  loomc_source_options_t options = {};
  options.identifier = loomc_make_cstring_view(name);
  options.contents = loomc_make_byte_span(text, strlen(text));
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

void ExpectSuccess(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    EXPECT_NE(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR)
        << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

std::string Print(const loomc_module_t* module) {
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_text_to_source(
      module, nullptr, loomc_allocator_system(), &source));
  SourcePtr owner(source);
  return Contents(source);
}

class CxxTest : public ::testing::Test {
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

  loomc_status_t Import(
      const loomc_source_t* source,
      const loomc_cxx_import_options_t* options = nullptr,
      loomc_allocator_t allocator = loomc_allocator_system()) {
    module_.reset();
    result_.reset();
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    auto status =
        loomc_module_import_cxx(context_.get(), workspace_.get(), source,
                                options, allocator, &module, &result);
    module_.reset(module);
    result_.reset(result);
    return status;
  }

  // Immutable context shared by imports and subsequent compilation.
  ContextPtr context_;
  // Source invocation workspace retained by live modules.
  WorkspacePtr workspace_;
  // Most recent imported module.
  ModulePtr module_;
  // Most recent import result.
  ResultPtr result_;
};

TEST_F(CxxTest, SourceAndWorkspaceCanBeReleasedBeforeCompilation) {
  auto source = Source("unit.cpp",
                       "static int helper(int x) { return x * 3; }"
                       "int entry(int x) { return helper(x) + 1; }");
  LOOMC_ASSERT_OK(Import(source.get()));
  ExpectSuccess(result_.get());
  ASSERT_NE(module_, nullptr);
  source.reset();
  result_.reset();
  workspace_.reset();

  loomc_compiler_t* compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_owner(compiler);
  loomc_pass_program_t* passes = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
      context_.get(), loomc_make_cstring_view("canonicalize,cse,dce"), nullptr,
      loomc_allocator_system(), &passes, &result));
  PassPtr pass_owner(passes);
  ResultPtr pass_result(result);
  ExpectSuccess(result);
  context_.reset();

  loomc_workspace_t* workspace = nullptr;
  LOOMC_ASSERT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  WorkspacePtr compile_workspace(workspace);
  LOOMC_ASSERT_OK(loomc_compile_module(compiler, workspace, passes,
                                       module_.get(), nullptr,
                                       loomc_allocator_system(), &result));
  ResultPtr compile_result(result);
  ExpectSuccess(result);
  EXPECT_NE(Print(module_.get()).find("@entry"), std::string::npos);
  loomc_module_t* clone = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module_.get(), workspace,
                                     loomc_allocator_system(), &clone));
  ModulePtr cloned_module(clone);
  module_.reset();
  EXPECT_NE(Print(clone).find("@entry"), std::string::npos);
}

TEST_F(CxxTest, RejectedSourceRetainsBorrowedContentsAndAllowsReuse) {
  std::string text = "int entry() { return absent_name; }";
  const std::string expected = text;
  loomc_source_options_t source_options = {};
  source_options.identifier = loomc_make_cstring_view("broken.cpp");
  source_options.contents = loomc_make_byte_span(text.data(), text.size());
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  LOOMC_ASSERT_OK(Import(source));
  loomc_source_release(source);
  text.assign(text.size(), '?');
  EXPECT_EQ(module_, nullptr);
  ASSERT_FALSE(loomc_result_succeeded(result_.get()));
  ASSERT_GT(loomc_result_diagnostic_count(result_.get()), 0u);
  const auto* diagnostic = loomc_result_diagnostic_at(result_.get(), 0);
  EXPECT_EQ(Contents(diagnostic->range.source), expected);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "broken.cpp");
  EXPECT_EQ(diagnostic->range.start_line, 1u);
  EXPECT_GT(diagnostic->range.end, diagnostic->range.start);
  auto good = Source("good.cpp", "int answer() { return 42; }");
  LOOMC_ASSERT_OK(Import(good.get()));
  ExpectSuccess(result_.get());
}

TEST_F(CxxTest, RenamedExportsLinkWithTheirTypesAfterSourceRelease) {
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    SCOPED_TRACE(format);
    auto source = Source("library.cpp", R"(
      [[loom::symbol("library.increment")]] unsigned increment(unsigned);
      unsigned increment(unsigned value) { return value + 1u; }
    )");
    LOOMC_ASSERT_OK(Import(source.get()));
    ExpectSuccess(result_.get());
    loomc_module_serialize_options_t serialization = {};
    serialization.format = format;
    loomc_source_t* library = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
        module_.get(), &serialization, loomc_allocator_system(), &library));
    SourcePtr library_owner(library);
    source.reset();
    module_.reset();
    result_.reset();

    for (const auto& [argument_type, result_type] :
         {std::pair{"i32", "i32"}, std::pair{"f32", "i32"},
          std::pair{"i32", "f32"}}) {
      const std::string signature =
          std::string(argument_type) + ") -> (" + result_type;
      SCOPED_TRACE(signature);
      const std::string caller_text =
          "func.decl @library.increment(" + signature + ")\n" +
          "func.def public @entry(%value: " + signature + ") {\n" +
          "  %result = func.call @library.increment(%value) : (" + signature +
          ")\n  func.return %result : " + result_type + "\n}\n";
      auto caller = Source("caller.loom", caller_text.c_str());
      loomc_link_index_builder_t* builder = nullptr;
      LOOMC_ASSERT_OK(loomc_link_index_builder_create(
          context_.get(), nullptr, loomc_allocator_system(), &builder));
      LinkBuilderPtr builder_owner(builder);
      loomc_link_index_source_options_t provider = {};
      provider.role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
      LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(builder, library,
                                                          &provider, nullptr));
      provider.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
      LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(builder, caller.get(),
                                                          &provider, nullptr));
      loomc_link_index_t* index = nullptr;
      loomc_result_t* result = nullptr;
      LOOMC_ASSERT_OK(
          loomc_link_index_builder_finish(builder, &index, &result));
      LinkIndexPtr index_owner(index);
      ResultPtr index_result(result);
      ExpectSuccess(result);
      builder_owner.reset();
      caller.reset();

      loomc_linker_t* linker = nullptr;
      LOOMC_ASSERT_OK(loomc_linker_create(context_.get(), nullptr,
                                          loomc_allocator_system(), &linker));
      LinkerPtr linker_owner(linker);
      loomc_link_options_t options = {};
      options.link_index = index;
      options.mode = LOOMC_LINK_MODE_LINK;
      options.flags = LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS;
      loomc_module_t* linked = nullptr;
      LOOMC_ASSERT_OK(loomc_link_module(linker, workspace_.get(), &options,
                                        &linked, &result));
      ModulePtr linked_owner(linked);
      ResultPtr link_result(result);
      if (signature == "i32) -> (i32") {
        ExpectSuccess(result);
        ASSERT_NE(linked, nullptr);
        auto text = Print(linked);
        EXPECT_NE(text.find("@library.increment"), std::string::npos);
        EXPECT_NE(text.find("scalar.addi"), std::string::npos);
        EXPECT_EQ(text.find("func.decl"), std::string::npos);
      } else {
        EXPECT_FALSE(loomc_result_succeeded(result));
        EXPECT_EQ(linked, nullptr);
        ASSERT_GT(loomc_result_diagnostic_count(result), 0u);
        const auto* diagnostic = loomc_result_diagnostic_at(result, 0);
        EXPECT_NE(ToString(diagnostic->message).find("incompatible"),
                  std::string::npos);
      }
    }
  }
}

// An immutable cache shared by independently owned frontend invocations.
struct HeaderCache {
  // Shared immutable source handle owned outside each import.
  SourcePtr header = Source(
      "/headers/helper.h",
      "template<int Factor> static int helper(int x) { return x * Factor; }");
  // Counts callback hits without serializing the immutable cache.
  std::atomic<unsigned> hits = 0;

  static loomc_status_t Resolve(void* user_data, loomc_string_view_t path,
                                loomc_source_t** out_source) {
    auto& self = *static_cast<HeaderCache*>(user_data);
    *out_source = nullptr;
    if (ToString(path) == "/headers/helper.h") {
      ++self.hits;
      loomc_source_retain(self.header.get());
      *out_source = self.header.get();
    }
    return loomc_ok_status();
  }
};

TEST_F(CxxTest, OptionsAndSharedIncludeProviderAcrossWorkers) {
  HeaderCache cache;
  auto source = Source("unit.cpp",
                       "#include <helper.h>\n"
                       "int first(int x) { return helper<FACTOR>(x); }"
                       "int second(int x) { return x + 2; }");
  const auto include_path = loomc_make_cstring_view("/headers");
  const auto root = loomc_make_cstring_view("first");
  const loomc_cxx_define_t define = {loomc_make_cstring_view("FACTOR"),
                                     loomc_make_cstring_view("7")};
  loomc_cxx_import_options_t options = {};
  options.source_provider = {HeaderCache::Resolve, &cache};
  options.system_include_paths = &include_path;
  options.system_include_path_count = 1;
  options.roots = &root;
  options.root_count = 1;
  options.defines = &define;
  options.define_count = 1;
  options.flags = LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  std::array<ModulePtr, 2> modules;
  std::array<std::thread, 2> workers;
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i]() {
      loomc_workspace_t* workspace = nullptr;
      LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                             &workspace));
      WorkspacePtr owner(workspace);
      loomc_module_t* module = nullptr;
      loomc_result_t* result = nullptr;
      LOOMC_ASSERT_OK(loomc_module_import_cxx(
          context_.get(), workspace, source.get(), &options,
          loomc_allocator_system(), &module, &result));
      ResultPtr result_owner(result);
      ExpectSuccess(result);
      modules[i].reset(module);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  source.reset();
  cache.header.reset();
  context_.reset();
  EXPECT_EQ(cache.hits, 2u);
  for (const auto& module : modules) {
    ASSERT_NE(module, nullptr);
    auto text = Print(module.get());
    EXPECT_NE(text.find("@first"), std::string::npos);
    EXPECT_NE(text.find("scalar.constant 7"), std::string::npos);
    EXPECT_EQ(text.find("@second"), std::string::npos);
  }
}

TEST_F(CxxTest, HeaderDiagnosticSurvivesProviderDestruction) {
  HeaderCache cache;
  const std::string expected = "static int broken() { return missing; }";
  cache.header = Source("/headers/helper.h", expected.c_str());
  auto source = Source("unit.cpp",
                       "#include <helper.h>\n"
                       "int entry() { return broken(); }");
  auto path = loomc_make_cstring_view("/headers");
  loomc_cxx_import_options_t options = {};
  options.source_provider = {HeaderCache::Resolve, &cache};
  options.include_paths = &path;
  options.include_path_count = 1;
  LOOMC_ASSERT_OK(Import(source.get(), &options));
  source.reset();
  cache.header.reset();
  ASSERT_FALSE(loomc_result_succeeded(result_.get()));
  ASSERT_GT(loomc_result_diagnostic_count(result_.get()), 0u);
  const auto* diagnostic = loomc_result_diagnostic_at(result_.get(), 0);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "/headers/helper.h");
  EXPECT_EQ(Contents(diagnostic->range.source), expected);
}

TEST_F(CxxTest, InvalidOptionsClearOutputs) {
  auto source = Source("unit.cpp", "int value() { return 42; }");
  loomc_cxx_import_options_t options = {};
  options.standard = {nullptr, 1};
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         Import(source.get(), &options));
  options = {};
  options.root_count = 1;
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         Import(source.get(), &options));
  auto path = loomc_make_cstring_view("include");
  options = {};
  options.include_paths = &path;
  options.include_path_count = LOOMC_HOST_SIZE_MAX;
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE,
                         Import(source.get(), &options));
  options = {};
  options.next = &options;
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_UNIMPLEMENTED,
                         Import(source.get(), &options));
  options = {};
  options.flags = 1u << 31;
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         Import(source.get(), &options));
  EXPECT_EQ(result_, nullptr);
  EXPECT_EQ(module_, nullptr);
}

TEST_F(CxxTest, LaunchRangesUseNormalCompilationConfigValidation) {
  auto source = Source("bounded.cpp", R"(
    [[loom::kernel, loom::workgroup_count_range(1, 4, 1, 1, 1, 1),
      loom::workgroup_size(64, 1, 1)]] void bounded() {}
  )");
  loomc_compiler_t* compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_owner(compiler);
  loomc_pass_program_t* passes = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
      context_.get(), loomc_make_cstring_view("canonicalize,cse,dce"), nullptr,
      loomc_allocator_system(), &passes, &result));
  PassPtr pass_owner(passes);
  ResultPtr pass_result(result);
  ExpectSuccess(result);
  for (int count : {1, 4, 0, 5}) {
    SCOPED_TRACE(count);
    LOOMC_ASSERT_OK(Import(source.get()));
    ExpectSuccess(result_.get());
    const std::string text =
        "config.def @bounded.workgroup_count.x = " + std::to_string(count) +
        " : index\nconfig.def @bounded.workgroup_count.y = 1 : index\n"
        "config.def @bounded.workgroup_count.z = 1 : index\n";
    auto config_source = Source("config.loom", text.c_str());
    loomc_module_t* config = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_text_from_source(
        context_.get(), workspace_.get(), config_source.get(), nullptr,
        loomc_allocator_system(), &config, &result));
    ModulePtr config_owner(config);
    ResultPtr config_result(result);
    ExpectSuccess(result);
    loomc_compile_options_t options = {};
    options.config_module = config;
    options.config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    LOOMC_ASSERT_OK(loomc_compile_module(compiler, workspace_.get(), passes,
                                         module_.get(), &options,
                                         loomc_allocator_system(), &result));
    ResultPtr compile_result(result);
    if (count == 1 || count == 4) {
      ExpectSuccess(result);
    } else {
      EXPECT_FALSE(loomc_result_succeeded(result));
      ASSERT_GT(loomc_result_diagnostic_count(result), 0u);
      EXPECT_EQ(ToString(loomc_result_diagnostic_at(result, 0)->code),
                "CONFIG/INVALID");
    }
  }
}

TEST_F(CxxTest, ConfigProvidersSpecializeFreshBytecodeAfterImportRelease) {
  auto source = Source("reusable.cpp", R"(
    [[loom::config("tuning.factor")]] extern const unsigned factor;
    unsigned entry() { return factor * 3; }
  )");
  LOOMC_ASSERT_OK(Import(source.get()));
  ExpectSuccess(result_.get());

  loomc_compiler_t* compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_owner(compiler);
  loomc_pass_program_t* passes = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
      context_.get(), loomc_make_cstring_view("canonicalize,cse,dce"), nullptr,
      loomc_allocator_system(), &passes, &result));
  PassPtr pass_owner(passes);
  ResultPtr pass_result(result);
  ExpectSuccess(result);
  LOOMC_ASSERT_OK(loomc_compile_module(compiler, workspace_.get(), passes,
                                       module_.get(), nullptr,
                                       loomc_allocator_system(), &result));
  ResultPtr cleanup_result(result);
  ExpectSuccess(result);
  const std::string unresolved = Print(module_.get());
  EXPECT_NE(unresolved.find("config.get @tuning.factor"), std::string::npos);

  loomc_source_t* bytecode = nullptr;
  LOOMC_ASSERT_OK(loomc_module_serialize_bytecode_to_source(
      module_.get(), nullptr, loomc_allocator_system(), &bytecode));
  SourcePtr bytecode_owner(bytecode);
  const std::string original_bytecode = Contents(bytecode);
  source.reset();
  result_.reset();
  cleanup_result.reset();
  module_.reset();
  workspace_.reset();

  for (unsigned factor : {5u, 9u}) {
    SCOPED_TRACE(factor);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    WorkspacePtr workspace_owner(workspace);
    loomc_module_t* module = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_bytecode_from_source(
        context_.get(), workspace, bytecode, nullptr, loomc_allocator_system(),
        &module, &result));
    ModulePtr module_owner(module);
    ResultPtr deserialize_result(result);
    ExpectSuccess(result);
    EXPECT_EQ(Print(module), unresolved);

    const std::string provider_text =
        "[[loom::config(\"tuning.factor\")]] const unsigned factor = " +
        std::to_string(factor) + ";";
    auto provider_source = Source("provider.cpp", provider_text.c_str());
    loomc_module_t* provider = nullptr;
    LOOMC_ASSERT_OK(loomc_module_import_cxx(
        context_.get(), workspace, provider_source.get(), nullptr,
        loomc_allocator_system(), &provider, &result));
    ModulePtr provider_owner(provider);
    ResultPtr provider_result(result);
    ExpectSuccess(result);
    provider_source.reset();
    provider_result.reset();
    const std::string provider_before = Print(provider);
    EXPECT_NE(provider_before.find("config.def @tuning.factor = " +
                                   std::to_string(factor)),
              std::string::npos);

    loomc_compile_options_t options = {};
    options.config_module = provider;
    options.config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    LOOMC_ASSERT_OK(loomc_compile_module(compiler, workspace, passes, module,
                                         &options, loomc_allocator_system(),
                                         &result));
    ResultPtr compile_result(result);
    ExpectSuccess(result);
    const std::string specialized = Print(module);
    EXPECT_EQ(specialized.find("config.get"), std::string::npos);
    EXPECT_NE(specialized.find("scalar.constant " + std::to_string(factor * 3) +
                               " : i32"),
              std::string::npos);
    EXPECT_EQ(Print(provider), provider_before);
    EXPECT_EQ(Contents(bytecode), original_bytecode);
  }
}

TEST_F(CxxTest, ProviderFailureIsInfrastructureStatus) {
  auto source = Source("unit.cpp", "#include \"header.h\"\n");
  loomc_cxx_import_options_t options = {};
  options.source_provider.fn = [](void*, loomc_string_view_t,
                                  loomc_source_t** out_source) {
    *out_source = nullptr;
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE, "provider failed");
  };
  LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_UNAVAILABLE,
                         Import(source.get(), &options));
  EXPECT_EQ(result_, nullptr);
  EXPECT_EQ(module_, nullptr);
}

struct FailingAllocator {
  // Allocation ordinal at which the allocator fails exactly once.
  size_t failure_at;
  // Number of allocation requests observed.
  size_t count = 0;

  static loomc_status_t Control(void* user_data,
                                loomc_allocator_command_t command,
                                const void* params, void** pointer) {
    auto& self = *static_cast<FailingAllocator*>(user_data);
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE &&
        self.count++ == self.failure_at) {
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "injected allocation failure");
    }
    auto system = loomc_allocator_system();
    return system.ctl(system.self, command, params, pointer);
  }
};

TEST_F(CxxTest, AllocationFailuresLeaveNoPartialOutputs) {
  // Exercise both normal ownership transfer and diagnostic construction. The
  // allocator remains alive until all handles created through it are released.
  for (const char* text :
       {"int entry() { return 42; }", "int entry() { return unknown; }"}) {
    auto source = Source("unit.cpp", text);
    auto path = loomc_make_cstring_view("/headers");
    loomc_cxx_import_options_t options = {};
    options.include_paths = &path;
    options.include_path_count = 1;
    FailingAllocator baseline = {LOOMC_HOST_SIZE_MAX};
    LOOMC_ASSERT_OK(
        Import(source.get(), &options, {&baseline, FailingAllocator::Control}));
    module_.reset();
    result_.reset();
    for (size_t i = 0; i < baseline.count; ++i) {
      SCOPED_TRACE(i);
      FailingAllocator failing = {i};
      LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             Import(source.get(), &options,
                                    {&failing, FailingAllocator::Control}));
      EXPECT_EQ(module_, nullptr);
      EXPECT_EQ(result_, nullptr);
    }
  }
}

}  // namespace
