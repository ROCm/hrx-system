// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target/cmd.h"
#include "loomc/target/kernel.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
}

void ExpectSucceededResult(const loomc_result_t* result) {
  if (result && !loomc_result_succeeded(result)) {
    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
         ++i) {
      const loomc_diagnostic_t* diagnostic =
          loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << ToString(diagnostic->code) << ": "
                    << ToString(diagnostic->message);
    }
  }
  EXPECT_TRUE(result && loomc_result_succeeded(result));
}

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
      /*.contents=*/loomc_make_byte_span(contents, std::strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
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
  EXPECT_TRUE(result_ptr && loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

SourcePtr SerializeModuleToBytecode(const loomc_module_t* module,
                                    const char* identifier) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view(identifier),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

BuilderPtr CreateIndexBuilder(loomc_context_t* context) {
  loomc_link_index_builder_t* builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder);
  LOOMC_EXPECT_OK(status);
  return BuilderPtr(builder);
}

LinkIndexPtr FinishIndex(loomc_link_index_builder_t* builder) {
  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_index_builder_finish(builder, &link_index, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return LinkIndexPtr(link_index);
}

struct RequestCapture {
  std::vector<RequestPtr> requests;
};

loomc_status_t CaptureRequest(void* user_data, loomc_request_t* request) {
  RequestCapture* capture = static_cast<RequestCapture*>(user_data);
  capture->requests.push_back(RequestPtr(request));
  return loomc_ok_status();
}

struct RejectRequestState {
  loomc_host_size_t publish_count = 0;
};

loomc_status_t RejectRequest(void* user_data, loomc_request_t* request) {
  RejectRequestState* state = static_cast<RejectRequestState*>(user_data);
  ++state->publish_count;
  loomc_request_release(request);
  return loomc_make_status(LOOMC_STATUS_ABORTED,
                           "embedding rejected kernel request");
}

TEST(TargetCmdProgramTest,
     ComposesMixedProvidersAndPublishesDurableKernelRequests) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();

  SourcePtr requester_source = CreateTextSource("requester.loom", R"(
template.decl @request.schedule(%size: index)

kernel.def @classified() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%size: index, %storage: buffer) {
  template.apply<@request.schedule>(%size) : (index)
  kernel.return
}

kernel.entry.decl @external(%storage: buffer)

command.program.def public @public_root() launch(%storage: buffer) {
  %small = index.constant 64 : index
  %large = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch @classified(%small, %storage) : (index, buffer)
  kernel.launch @classified(%large, %storage) : (index, buffer)
  kernel.dispatch @external[%unit](%storage) : [index](buffer)
  command.return
}

command.program.def @private_root() launch(%storage: buffer) {
  %small = index.constant 64 : index
  kernel.launch @classified(%small, %storage) : (index, buffer)
  command.return
}
)");
  SourcePtr kernel_text_source = CreateTextSource("large_schedule.loom", R"(
template.decl @request.schedule(%size: index)

template.def<@request.schedule> priority(10) @large(%size: index) where [ge(%size, 128)] {
  template.return
}
)");
  ModulePtr kernel_module = DeserializeModule(context.get(), workspace.get(),
                                              kernel_text_source.get());
  SourcePtr kernel_bytecode =
      SerializeModuleToBytecode(kernel_module.get(), "large_schedule.loombc");

  SourcePtr schedule_source = CreateTextSource("schedule.loom", R"(
template.decl @request.schedule(%size: index)

template.def<@request.schedule> priority(1) @small(%size: index) {
  template.return
}
)");

  BuilderPtr builder = CreateIndexBuilder(context.get());
  loomc_link_index_source_options_t requester_options = {
      /*.provider_name=*/loomc_make_cstring_view("requester"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  loomc_link_index_source_slot_t requester_slot = {};
  LOOMC_ASSERT_OK(
      loomc_link_index_builder_add_source(builder.get(), requester_source.get(),
                                          &requester_options, &requester_slot));
  loomc_link_index_source_options_t schedule_options = {
      /*.provider_name=*/loomc_make_cstring_view("schedule"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), schedule_source.get(), &schedule_options, nullptr));
  loomc_link_index_source_options_t kernel_options = {
      /*.provider_name=*/loomc_make_cstring_view("kernel"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
  };
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
      builder.get(), kernel_bytecode.get(), &kernel_options, nullptr));
  LinkIndexPtr link_index = FinishIndex(builder.get());
  builder.reset();

  loomc_link_index_provider_t requester_provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(
      link_index.get(), requester_slot.ordinal, &requester_provider));
  ASSERT_EQ(requester_provider.module_count, 1u);
  loomc_link_index_module_t requester_index_module = {};
  ASSERT_TRUE(loomc_link_index_module_at(
      link_index.get(), requester_provider.module_start_ordinal,
      &requester_index_module));
  loomc_link_index_symbol_t private_root = {};
  ASSERT_TRUE(loomc_link_index_lookup_private(
      link_index.get(), &requester_index_module,
      loomc_make_cstring_view("private_root"), &private_root));
  loomc_link_index_symbol_t public_root = {};
  ASSERT_TRUE(loomc_link_index_lookup_global(
      link_index.get(), loomc_make_cstring_view("public_root"), &public_root));

  RequestCapture capture;
  const loomc_host_size_t explicit_roots[] = {private_root.ordinal};
  loomc_cmd_program_product_options_t product_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      /*.structure_size=*/sizeof(product_options),
      /*.next=*/nullptr,
      /*.link_index=*/link_index.get(),
      /*.root_symbol_ordinals=*/explicit_roots,
      /*.root_symbol_count=*/std::size(explicit_roots),
      /*.flags=*/LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      /*.config=*/{},
      /*.request_sink=*/
      {
          /*.publish=*/CaptureRequest,
          /*.user_data=*/&capture,
      },
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_cmd_program_product_build(
      workspace.get(), &product_options, loomc_allocator_system(), &product,
      &result);
  LOOMC_ASSERT_OK(status);
  ProductPtr product_ptr(product);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_TRUE(result_ptr && loomc_result_succeeded(result_ptr.get()));

  EXPECT_EQ(loomc_product_descriptor(product_ptr.get()),
            loomc_cmd_program_product_descriptor());
  EXPECT_EQ(loomc_product_artifact_count(product_ptr.get()), 2u);
  EXPECT_EQ(loomc_product_export_count(product_ptr.get()), 2u);
  EXPECT_EQ(loomc_product_requirement_count(product_ptr.get()), 3u);
  ASSERT_EQ(loomc_cmd_program_product_program_count(product_ptr.get()), 2u);
  loomc_cmd_program_t private_program = {};
  ASSERT_TRUE(loomc_cmd_program_product_program_at(product_ptr.get(), 0,
                                                   &private_program));
  EXPECT_EQ(ToString(private_program.symbol), "private_root");
  EXPECT_EQ(private_program.artifact.kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(private_program.artifact.format),
            LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM);
  EXPECT_EQ(ToString(private_program.artifact.identifier), "private_root");
  EXPECT_GT(loomc_byte_sequence_length(private_program.artifact.contents), 0u);
  EXPECT_EQ(private_program.entry_requirement_count, 1u);
  const loomc_artifact_t* private_artifact =
      loomc_product_artifact_at(product_ptr.get(), 0);
  ASSERT_NE(private_artifact, nullptr);
  EXPECT_EQ(private_artifact->contents, private_program.artifact.contents);

  loomc_cmd_program_t public_program = {};
  ASSERT_TRUE(loomc_cmd_program_product_program_at(product_ptr.get(), 1,
                                                   &public_program));
  EXPECT_EQ(ToString(public_program.symbol), "public_root");
  EXPECT_GT(loomc_byte_sequence_length(public_program.artifact.contents), 0u);
  EXPECT_EQ(public_program.entry_requirement_count, 3u);

  ASSERT_EQ(capture.requests.size(), 2u);
  std::vector<loomc_requirement_ordinal_t> request_requirement_ordinals;
  for (const RequestPtr& request : capture.requests) {
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(loomc_request_product_descriptor(request.get()),
              loomc_kernel_product_descriptor());
    EXPECT_EQ(loomc_request_root_count(request.get()), 1u);
    EXPECT_EQ(loomc_request_binding_count(request.get()), 1u);
    loomc_request_binding_t binding = {};
    ASSERT_TRUE(loomc_request_binding_at(request.get(), 0, &binding));
    EXPECT_EQ(binding.root_ordinal, 0u);
    request_requirement_ordinals.push_back(binding.requirement_ordinal);
    loomc_source_t* request_source = loomc_request_source(request.get());
    ASSERT_NE(request_source, nullptr);
    EXPECT_EQ(loomc_source_format(request_source),
              LOOMC_SOURCE_FORMAT_BYTECODE);
  }

  ASSERT_EQ(
      loomc_cmd_program_product_entry_requirement_count(product_ptr.get()), 3u);
  loomc_host_size_t classified_requirement_count = 0;
  loomc_host_size_t external_requirement_count = 0;
  for (loomc_host_size_t i = 0; i < 3; ++i) {
    loomc_cmd_entry_requirement_t requirement = {};
    ASSERT_TRUE(loomc_cmd_program_product_entry_requirement_at(
        product_ptr.get(), i, &requirement));
    if (ToString(requirement.symbol) == "classified") {
      ++classified_requirement_count;
    } else {
      ++external_requirement_count;
      EXPECT_EQ(ToString(requirement.symbol), "external");
    }
  }
  EXPECT_EQ(classified_requirement_count, 2u);
  EXPECT_EQ(external_requirement_count, 1u);
  std::sort(request_requirement_ordinals.begin(),
            request_requirement_ordinals.end());
  EXPECT_EQ(request_requirement_ordinals,
            (std::vector<loomc_requirement_ordinal_t>{0, 1}));

  const loomc_host_size_t repeated_roots[] = {
      public_root.ordinal,
      public_root.ordinal,
  };
  product_options.root_symbol_ordinals = repeated_roots;
  product_options.root_symbol_count = std::size(repeated_roots);
  product_options.flags = 0;
  product_options.request_sink = {};
  loomc_product_t* repeated_product = nullptr;
  loomc_result_t* repeated_result = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_product_build(
      workspace.get(), &product_options, loomc_allocator_system(),
      &repeated_product, &repeated_result));
  ProductPtr repeated_product_ptr(repeated_product);
  ResultPtr repeated_result_ptr(repeated_result);
  ExpectSucceededResult(repeated_result_ptr.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(repeated_product_ptr.get()),
            2u);
  loomc_cmd_program_t repeated_programs[2] = {};
  ASSERT_TRUE(loomc_cmd_program_product_program_at(repeated_product_ptr.get(),
                                                   0, &repeated_programs[0]));
  ASSERT_TRUE(loomc_cmd_program_product_program_at(repeated_product_ptr.get(),
                                                   1, &repeated_programs[1]));
  EXPECT_EQ(ToString(repeated_programs[0].symbol), "public_root");
  EXPECT_EQ(ToString(repeated_programs[1].symbol), "public_root");
  EXPECT_EQ(ToString(repeated_programs[0].artifact.contents),
            ToString(repeated_programs[1].artifact.contents));

  RejectRequestState reject_state;
  const loomc_host_size_t public_roots[] = {public_root.ordinal};
  product_options.root_symbol_ordinals = public_roots;
  product_options.root_symbol_count = std::size(public_roots);
  product_options.flags = 0;
  product_options.request_sink = {
      /*.publish=*/RejectRequest,
      /*.user_data=*/&reject_state,
  };
  product = reinterpret_cast<loomc_product_t*>(0x1);
  result = reinterpret_cast<loomc_result_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_ABORTED,
                         loomc_cmd_program_product_build(
                             workspace.get(), &product_options,
                             loomc_allocator_system(), &product, &result));
  EXPECT_EQ(product, nullptr);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(reject_state.publish_count, 1u);

  link_index.reset();
  kernel_module.reset();
  requester_source.reset();
  kernel_text_source.reset();
  kernel_bytecode.reset();
  schedule_source.reset();
  workspace.reset();
  context.reset();

  ContextPtr restored_context = CreateContext();
  for (const RequestPtr& request : capture.requests) {
    loomc_request_root_t root = {};
    ASSERT_TRUE(loomc_request_root_at(request.get(), 0, &root));
    EXPECT_EQ(root.goal, LOOMC_KERNEL_ROOT_GOAL_EXECUTABLE_ENTRY);
    BuilderPtr restored_builder = CreateIndexBuilder(restored_context.get());
    loomc_link_index_source_options_t options = {
        /*.provider_name=*/loomc_make_cstring_view("restored_request"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        restored_builder.get(), loomc_request_source(request.get()), &options,
        nullptr));
    LinkIndexPtr restored_index = FinishIndex(restored_builder.get());
    loomc_link_index_provider_t provider = {};
    ASSERT_TRUE(
        loomc_link_index_provider_at(restored_index.get(), 0, &provider));
    ASSERT_LT(root.module_ordinal, provider.module_count);
    loomc_link_index_module_t module = {};
    ASSERT_TRUE(loomc_link_index_module_at(
        restored_index.get(),
        provider.module_start_ordinal + root.module_ordinal, &module));
    ASSERT_LT(root.symbol_ordinal, module.symbol_count);
    loomc_link_index_symbol_t symbol = {};
    ASSERT_TRUE(loomc_link_index_symbol_at(
        restored_index.get(), module.symbol_start_ordinal + root.symbol_ordinal,
        &symbol));
    EXPECT_EQ(ToString(symbol.name), "classified");
  }

  EXPECT_EQ(ToString(private_program.symbol), "private_root");
  EXPECT_GT(loomc_byte_sequence_length(private_program.artifact.contents), 0u);
}

}  // namespace
