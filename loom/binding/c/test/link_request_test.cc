// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/module_bytecode.h"
#include "loom/binding/c/src/product.h"
#include "loom/ir/module.h"
#include "loomc/compile.h"
#include "loomc/link.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
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

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* linker = nullptr;
  LOOMC_EXPECT_OK(
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker));
  return LinkerPtr(linker);
}

SourcePtr CreateSource(loomc_source_format_t format, const char* identifier,
                       const void* contents, loomc_host_size_t length) {
  const loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, length),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, identifier, contents,
                      strlen(contents));
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

std::string SerializeModuleToText(const loomc_module_t* module) {
  const loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("linked.loom"),
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);
  return ToString(loomc_source_contents(source_ptr.get()));
}

RequestPtr CreateRequest(
    loomc_context_t* context, loomc_module_t* module,
    std::initializer_list<const char*> root_names,
    const std::vector<loomc_request_binding_t>& bindings = {}) {
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  EXPECT_NE(internal_module, nullptr);
  if (internal_module == nullptr) return RequestPtr();

  std::vector<loom_symbol_id_t> module_symbol_ids;
  module_symbol_ids.reserve(root_names.size());
  for (const char* root_name : root_names) {
    const loom_string_id_t name_id = loom_module_lookup_string(
        internal_module, iree_make_cstring_view(root_name));
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    if (name_id == LOOM_STRING_ID_INVALID) return RequestPtr();
    const loom_symbol_id_t symbol_id =
        loom_module_find_symbol(internal_module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) return RequestPtr();
    module_symbol_ids.push_back(symbol_id);
  }

  std::vector<loom_symbol_id_t> wire_symbol_ordinals(module_symbol_ids.size(),
                                                     LOOM_SYMBOL_ID_INVALID);
  const loomc_module_symbol_projection_t projection = {
      /*.module_symbol_ids=*/module_symbol_ids.data(),
      /*.bytecode_symbol_ordinals=*/wire_symbol_ordinals.data(),
      /*.count=*/wire_symbol_ordinals.size(),
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_internal_bytecode_to_source(
      context, internal_module, loomc_make_cstring_view("request.loombc"),
      &projection, loomc_allocator_system(), &source));

  std::vector<loomc_request_root_t> roots;
  roots.reserve(wire_symbol_ordinals.size());
  for (loom_symbol_id_t symbol_ordinal : wire_symbol_ordinals) {
    roots.push_back({
        /*.module_ordinal=*/0,
        /*.symbol_ordinal=*/symbol_ordinal,
    });
  }
  loomc_request_t* request = nullptr;
  loomc_status_t status = loomc_request_create_take_source(
      loomc_compiled_module_product_descriptor(), &source, roots.data(),
      roots.size(), bindings.data(), bindings.size(), loomc_allocator_system(),
      &request);
  loomc_source_release(source);
  LOOMC_EXPECT_OK(status);
  return RequestPtr(request);
}

RequestPtr CreateRawRequest(SourcePtr source, loomc_request_root_t root) {
  loomc_source_t* transferred_source = source.release();
  loomc_request_t* request = nullptr;
  loomc_status_t status = loomc_request_create_take_source(
      loomc_compiled_module_product_descriptor(), &transferred_source, &root, 1,
      nullptr, 0, loomc_allocator_system(), &request);
  loomc_source_release(transferred_source);
  LOOMC_EXPECT_OK(status);
  return RequestPtr(request);
}

LinkIndexPtr CreateIndex(loomc_context_t* context, loomc_source_t* source,
                         loomc_link_provider_role_t role) {
  loomc_link_index_builder_t* builder = nullptr;
  LOOMC_EXPECT_OK(loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder));
  BuilderPtr builder_ptr(builder);
  const loomc_link_index_source_options_t source_options = {
      /*.provider_name=*/loomc_string_view_empty(),
      /*.role=*/role,
  };
  LOOMC_EXPECT_OK(loomc_link_index_builder_add_source(
      builder_ptr.get(), source, &source_options, nullptr));
  loomc_link_index_t* index = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(
      loomc_link_index_builder_finish(builder_ptr.get(), &index, &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return LinkIndexPtr(index);
}

std::vector<std::string> ResolveRequestRootNames(
    loomc_context_t* context, const loomc_request_t* request) {
  LinkIndexPtr index = CreateIndex(context, loomc_request_source(request),
                                   LOOMC_LINK_PROVIDER_ROLE_INPUT);
  loomc_link_index_provider_t provider = {};
  EXPECT_TRUE(loomc_link_index_provider_at(index.get(), 0, &provider));

  std::vector<std::string> names;
  for (loomc_host_size_t i = 0; i < loomc_request_root_count(request); ++i) {
    loomc_request_root_t root = {};
    EXPECT_TRUE(loomc_request_root_at(request, i, &root));
    loomc_link_index_module_t module = {};
    EXPECT_TRUE(loomc_link_index_module_at(
        index.get(), provider.module_start_ordinal + root.module_ordinal,
        &module));
    loomc_link_index_symbol_t symbol = {};
    EXPECT_TRUE(loomc_link_index_symbol_at(
        index.get(), module.symbol_start_ordinal + root.symbol_ordinal,
        &symbol));
    names.push_back(ToString(symbol.name));
  }
  return names;
}

PassProgramPtr CreateEmptyPassProgram(loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_empty(
      context, nullptr, loomc_allocator_system(), &pass_program));
  return PassProgramPtr(pass_program);
}

class LinkRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = CreateContext();
    workspace_ = CreateWorkspace();
    linker_ = CreateLinker(context_.get());
  }

  RequestPtr CreateRequestFromText(
      const char* text, std::initializer_list<const char*> root_names,
      const std::vector<loomc_request_binding_t>& bindings = {}) {
    SourcePtr source = CreateTextSource("request.loom", text);
    ModulePtr module =
        DeserializeModule(context_.get(), workspace_.get(), source.get());
    return CreateRequest(context_.get(), module.get(), root_names, bindings);
  }

  RequestPtr LinkRequest(const loomc_request_t* input_request,
                         const loomc_link_request_options_t* options,
                         ResultPtr* out_result) {
    loomc_request_t* output_request = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_link_request(
        linker_.get(), workspace_.get(), input_request, options,
        loomc_allocator_system(), &output_request, &result));
    out_result->reset(result);
    return RequestPtr(output_request);
  }

  std::string SerializeRequestToText(const loomc_request_t* request) {
    ModulePtr module = DeserializeModule(context_.get(), workspace_.get(),
                                         loomc_request_source(request));
    return SerializeModuleToText(module.get());
  }

  ContextPtr context_;
  WorkspacePtr workspace_;
  LinkerPtr linker_;
};

TEST_F(LinkRequestTest, SealsReachableSourceAndOwnsItsResult) {
  RequestPtr input_request = CreateRequestFromText(R"(
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @unused(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @entry(%x: i32) -> (i32) {
  %result = func.call @helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
)",
                                                   {"entry"});
  ASSERT_NE(input_request, nullptr);

  loomc_link_request_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.library_index=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("sealed"),
  };
  ResultPtr result;
  RequestPtr output_request =
      LinkRequest(input_request.get(), &options, &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(output_request, nullptr);
  EXPECT_EQ(ToString(loomc_source_identifier(
                loomc_request_source(output_request.get()))),
            "sealed.loombc");

  input_request.reset();
  workspace_.reset();
  workspace_ = CreateWorkspace();
  const std::string text = SerializeRequestToText(output_request.get());
  EXPECT_THAT(text, ::testing::HasSubstr("@entry"));
  EXPECT_THAT(text, ::testing::HasSubstr("@helper"));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("@unused")));

  loomc_compiler_t* compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_ptr(compiler);
  PassProgramPtr pass_program = CreateEmptyPassProgram(context_.get());
  loomc_product_t* product = nullptr;
  loomc_result_t* compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler_ptr.get(), workspace_.get(), pass_program.get(),
      output_request.get(), nullptr, loomc_allocator_system(), &product,
      &compile_result));
  ProductPtr product_ptr(product);
  ResultPtr compile_result_ptr(compile_result);
  ExpectSucceededResult(compile_result_ptr.get());
  ASSERT_NE(product_ptr, nullptr);
  EXPECT_EQ(loomc_product_export_count(product_ptr.get()), 1u);
}

TEST_F(LinkRequestTest, ResolvesLibraryAndPreservesRootBindings) {
  SourcePtr library_source = CreateTextSource("library.loom", R"(
func.def public @identity(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @unused_library(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  LinkIndexPtr library_index = CreateIndex(context_.get(), library_source.get(),
                                           LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  const std::vector<loomc_request_binding_t> bindings = {
      {/*.requirement_ordinal=*/3, /*.root_ordinal=*/2},
      {/*.requirement_ordinal=*/9, /*.root_ordinal=*/0},
  };
  RequestPtr input_request =
      CreateRequestFromText(R"(
func.decl @identity(%x: i32) -> (i32)

func.def public @second(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @caller(%x: i32) -> (i32) {
  %result = func.call @identity(%x) : (i32) -> (i32)
  func.return %result : i32
}
)",
                            {"second", "caller", "caller"}, bindings);

  const loomc_link_request_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.library_index=*/library_index.get(),
  };
  ResultPtr result;
  RequestPtr output_request =
      LinkRequest(input_request.get(), &options, &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(output_request, nullptr);

  input_request.reset();
  library_index.reset();
  library_source.reset();
  EXPECT_EQ(ResolveRequestRootNames(context_.get(), output_request.get()),
            (std::vector<std::string>{"second", "caller", "caller"}));
  ASSERT_EQ(loomc_request_binding_count(output_request.get()), 2u);
  for (loomc_host_size_t i = 0; i < bindings.size(); ++i) {
    loomc_request_binding_t actual = {};
    ASSERT_TRUE(loomc_request_binding_at(output_request.get(), i, &actual));
    EXPECT_EQ(actual.requirement_ordinal, bindings[i].requirement_ordinal);
    EXPECT_EQ(actual.root_ordinal, bindings[i].root_ordinal);
  }

  const std::string text = SerializeRequestToText(output_request.get());
  EXPECT_THAT(text, ::testing::HasSubstr("func.def @identity"));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("public @identity")));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("@unused_library")));
}

TEST_F(LinkRequestTest, SelectsPrivateRootByIndexedIdentity) {
  SourcePtr library_source = CreateTextSource("library.loom", R"(
func.def @library_helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @entry(%x: i32) -> (i32) {
  %result = func.call @library_helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
)");
  LinkIndexPtr library_index = CreateIndex(context_.get(), library_source.get(),
                                           LOOMC_LINK_PROVIDER_ROLE_LIBRARY);
  RequestPtr input_request = CreateRequestFromText(R"(
func.def @request_helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @entry(%x: i32) -> (i32) {
  %result = func.call @request_helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
)",
                                                   {"entry"});

  const loomc_link_request_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.library_index=*/library_index.get(),
  };
  ResultPtr result;
  RequestPtr output_request =
      LinkRequest(input_request.get(), &options, &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(output_request, nullptr);
  const std::string text = SerializeRequestToText(output_request.get());
  EXPECT_THAT(text, ::testing::HasSubstr("@request_helper"));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("@library_helper")));
}

TEST_F(LinkRequestTest, AppliesInvocationConfig) {
  RequestPtr input_request = CreateRequestFromText(R"(
config.decl @model.hidden_size : %value: index where [range(%value, 0, 8192)]

func.def public @entry() -> (index) {
  %value = config.get @model.hidden_size : index
  func.return %value : index
}
)",
                                                   {"entry"});
  const loomc_config_binding_t bindings[] = {{
      /*.key=*/loomc_make_cstring_view("@model.hidden_size"),
      /*.value=*/loomc_make_cstring_view("4096"),
  }};
  const loomc_link_request_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.library_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };

  ResultPtr result;
  RequestPtr output_request =
      LinkRequest(input_request.get(), &options, &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(output_request, nullptr);
  const std::string text = SerializeRequestToText(output_request.get());
  EXPECT_THAT(text, ::testing::HasSubstr(
                        "config.def @model.hidden_size = 4096 : index"));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("config.decl @model")));
}

TEST_F(LinkRequestTest, MalformedBytecodeProducesFailedResult) {
  static const char malformed_bytecode[] = "not Loom bytecode";
  RequestPtr input_request = CreateRawRequest(
      CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "malformed.loombc",
                   malformed_bytecode, sizeof(malformed_bytecode) - 1),
      {/*.module_ordinal=*/0, /*.symbol_ordinal=*/0});

  ResultPtr result;
  RequestPtr output_request =
      LinkRequest(input_request.get(), nullptr, &result);
  EXPECT_EQ(output_request, nullptr);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_NE(loomc_result_diagnostic_count(result.get()), 0u);
}

TEST_F(LinkRequestTest, RejectsLibraryIndexContainingInputProviders) {
  SourcePtr input_source = CreateTextSource("input.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  LinkIndexPtr input_index = CreateIndex(context_.get(), input_source.get(),
                                         LOOMC_LINK_PROVIDER_ROLE_INPUT);
  RequestPtr input_request = CreateRequestFromText(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)",
                                                   {"entry"});
  const loomc_link_request_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.library_index=*/input_index.get(),
  };
  loomc_request_t* output_request = reinterpret_cast<loomc_request_t*>(0x1);
  loomc_result_t* result = reinterpret_cast<loomc_result_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_link_request(linker_.get(), workspace_.get(), input_request.get(),
                         &options, loomc_allocator_system(), &output_request,
                         &result));
  EXPECT_EQ(output_request, nullptr);
  EXPECT_EQ(result, nullptr);
}

}  // namespace
