// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/target/provider.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"
#include "loomc/compile.h"
#include "loomc/pass.h"
#include "module.h"
#include "target.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;

static const loom_target_snapshot_t kFakeTargetSnapshot = {
    /*.name=*/IREE_SVL("fake-link-target"),
    /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
};

static const loom_target_export_plan_t kFakeTargetExportPlan = {
    /*.name=*/IREE_SVL("fake-link-target"),
    /*.export_symbol=*/IREE_SVL(""),
    /*.abi_kind=*/LOOM_TARGET_ABI_OBJECT_FUNCTION,
};

static const loom_target_config_t kFakeTargetConfig = {
    /*.name=*/IREE_SVL("fake-link-target"),
};

static const loom_target_bundle_t kFakeTargetBundle = {
    /*.name=*/IREE_SVL("fake-link-target"),
    /*.snapshot=*/&kFakeTargetSnapshot,
    /*.export_plan=*/&kFakeTargetExportPlan,
    /*.config=*/&kFakeTargetConfig,
};

static iree_status_t RegisterFakeTargetContext(loom_context_t* context) {
  return loom_test_dialect_register(context);
}

static iree_status_t MaterializeFakeTargetSelection(
    const loom_target_provider_t* provider,
    const loom_target_selection_materialization_request_t* request,
    bool* out_materialized, loom_symbol_ref_t* out_target_ref) {
  (void)provider;
  *out_materialized = false;
  *out_target_ref = loom_symbol_ref_null();
  if (request->target_selection.bundle != &kFakeTargetBundle) {
    return iree_ok_status();
  }

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      request->module, IREE_SV("selected_link_target"), &symbol_name_id));
  loom_symbol_id_t symbol_id =
      loom_module_find_symbol(request->module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(request->module, symbol_name_id, &symbol_id));
  }
  *out_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };
  if (request->module->symbols.entries[symbol_id].defining_op != nullptr) {
    *out_materialized = true;
    return iree_ok_status();
  }

  loom_block_t* module_block = loom_module_block(request->module);
  loom_builder_t builder = {};
  loom_builder_initialize(request->module, &request->module->arena,
                          module_block, &builder);
  if (module_block->first_op != nullptr) {
    loom_builder_set_before(&builder, module_block->first_op);
  }
  loom_op_t* target_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_test_target_build(
      &builder, /*build_flags=*/0, LOOM_TEST_TARGET_KIND_LOW_CORE,
      *out_target_ref, /*codegen_format=*/0, /*artifact_format=*/0,
      /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
      /*offset_bitwidth=*/0, /*max_workgroup_size_x=*/0,
      /*max_workgroup_size_y=*/0, /*max_workgroup_size_z=*/0,
      /*max_flat_workgroup_size=*/0, /*subgroup_size=*/0,
      /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
      /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
      /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
      /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
      /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
      /*memory_space_constant=*/0, /*memory_space_private=*/0,
      /*memory_space_host=*/0, /*memory_space_descriptor=*/0, /*abi=*/0,
      /*export_symbol=*/LOOM_STRING_ID_INVALID, /*linkage=*/0,
      /*hal_buffer_resource_flags=*/0,
      /*contract_set_key=*/LOOM_STRING_ID_INVALID,
      /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
  *out_materialized = true;
  return iree_ok_status();
}

static const loom_target_provider_t kFakeTargetProvider = {
    /*.register_context=*/RegisterFakeTargetContext,
    /*.initialize_low_descriptor_registry=*/nullptr,
    /*.initialize_low_lower_policy_registry=*/nullptr,
    /*.initialize_math_policy_registry=*/nullptr,
    /*.low_legality_provider_list=*/{},
    /*.legalizer_provider_list=*/{},
    /*.low_packet_diagnostic_provider_list=*/{},
    /*.low_asm_diagnostic_provider_list=*/{},
    /*.low_verify_provider_list=*/{},
    /*.emitter_list=*/{},
    /*.pass_registry=*/nullptr,
    /*.contribute_pipeline=*/nullptr,
    /*.materialize_selection=*/MaterializeFakeTargetSelection,
};

static const loom_target_provider_t* const kFakeTargetProviders[] = {
    &kFakeTargetProvider,
};

static const loom_target_provider_set_t kFakeTargetProviderSet = {
    /*.providers=*/kFakeTargetProviders,
    /*.provider_count=*/IREE_ARRAYSIZE(kFakeTargetProviders),
};

std::string ToString(loomc_string_view_t value) {
  if (value.size == 0) {
    return "";
  }
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t contents) {
  if (contents.data_length == 0) {
    return "";
  }
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

std::string ReadPathToString(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.good());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadOpenFileBytes(FILE* file) {
  EXPECT_EQ(fflush(file), 0);
  EXPECT_EQ(fseek(file, 0, SEEK_END), 0);
  long file_length = ftell(file);
  EXPECT_GE(file_length, 0);
  if (file_length < 0) {
    return {};
  }
  EXPECT_EQ(fseek(file, 0, SEEK_SET), 0);
  std::vector<uint8_t> bytes((size_t)file_length);
  if (!bytes.empty()) {
    EXPECT_EQ(fread(bytes.data(), 1, bytes.size(), file), bytes.size());
  }
  return bytes;
}

SourcePtr SerializeModuleToSource(const loomc_module_t* module,
                                  loomc_source_format_t format,
                                  const char* identifier) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/loomc_make_cstring_view(identifier),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

std::string SerializeModuleToText(const loomc_module_t* module) {
  SourcePtr source =
      SerializeModuleToSource(module, LOOMC_SOURCE_FORMAT_TEXT, "module.loom");
  return ToString(loomc_source_contents(source.get()));
}

ModulePtr DeserializeModuleFromSource(loomc_context_t* context,
                                      loomc_workspace_t* workspace,
                                      const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

ModulePtr DeserializeModuleFromFile(loomc_context_t* context,
                                    loomc_workspace_t* workspace, FILE* file) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_file(
      context, workspace, file, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

ModulePtr DeserializeModuleFromPath(loomc_context_t* context,
                                    loomc_workspace_t* workspace,
                                    const std::string& path) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_path(
      context, workspace, loomc_make_string_view(path.data(), path.size()),
      nullptr, loomc_allocator_system(), &module, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

ContextPtr CreateContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  loomc_context_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(&options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

TargetEnvironmentPtr CreateFakeTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_from_provider_set(
      &kFakeTargetProviderSet, loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

TargetProfilePtr CreateFakeTargetProfile(
    loomc_target_environment_t* target_environment) {
  loomc_target_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-link-profile"),
  };
  loom_target_selection_t selection = {
      /*.bundle=*/&kFakeTargetBundle,
      /*.data=*/nullptr,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_from_selection(
      target_environment, &options, selection, /*payload_type=*/nullptr,
      /*payload=*/nullptr, /*deinitialize=*/nullptr, loomc_allocator_system(),
      &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateTargetSelection(loomc_target_profile_t* profile) {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile, loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

SourcePtr CreateSource(loomc_source_format_t format, const char* identifier,
                       const void* contents, size_t contents_length) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, contents_length),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

SourcePtr CreateTextSource(const char* identifier, const char* source_text) {
  return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, identifier, source_text,
                      strlen(source_text));
}

BuilderPtr CreateBuilder(loomc_context_t* context) {
  loomc_link_index_builder_t* builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder);
  LOOMC_EXPECT_OK(status);
  return BuilderPtr(builder);
}

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* linker = nullptr;
  loomc_status_t status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  LOOMC_EXPECT_OK(status);
  return LinkerPtr(linker);
}

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
  };
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, &options, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
}

PassProgramPtr CreatePassProgramFromPipelineText(loomc_context_t* context,
                                                 const char* pipeline_text) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view(pipeline_text), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return PassProgramPtr(pass_program);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

void FinishIndex(loomc_link_index_builder_t* builder,
                 LinkIndexPtr* out_link_index) {
  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_index_builder_finish(builder, &link_index, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(link_index, nullptr);
  *out_link_index = LinkIndexPtr(link_index);
}

void AddSource(loomc_link_index_builder_t* builder, loomc_source_t* source,
               const char* provider_name,
               loomc_link_provider_role_t provider_role) {
  loomc_link_index_source_options_t options = {
      /*.provider_name=*/loomc_make_cstring_view(provider_name),
      /*.role=*/provider_role,
  };
  LOOMC_ASSERT_OK(
      loomc_link_index_builder_add_source(builder, source, &options, nullptr));
}

ModulePtr LinkIndex(loomc_linker_t* linker, loomc_workspace_t* workspace,
                    loomc_link_index_t* link_index,
                    const loomc_link_options_t* options,
                    ResultPtr* out_result) {
  loomc_link_options_t local_options =
      options ? *options : loomc_link_options_t{};
  local_options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  local_options.structure_size = sizeof(local_options);
  local_options.link_index = link_index;

  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_link_module(linker, workspace, &local_options, &module, &result);
  LOOMC_EXPECT_OK(status);
  *out_result = ResultPtr(result);
  return ModulePtr(module);
}

bool ModuleHasSymbol(const loom_module_t* module, const char* name) {
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->name_id >= module->strings.count) {
      continue;
    }
    iree_string_view_t symbol_name = module->strings.entries[symbol->name_id];
    if (symbol_name.size == strlen(name) &&
        memcmp(symbol_name.data, name, symbol_name.size) == 0) {
      return true;
    }
  }
  return false;
}

std::string PrintModule(const loom_module_t* module) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                   LOOM_TEXT_PRINT_DEFAULT));
  std::string text(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return text;
}

void VerifyModule(const loom_module_t* module) {
  loom_verify_options_t options = {
      /*.sink=*/{},
      /*.max_errors=*/20,
      /*.source_resolver=*/{},
  };
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    ADD_FAILURE() << PrintModule(module);
  }
  EXPECT_EQ(result.error_count, 0u);
}

void VerifyLinkedCallerModule(const loomc_module_t* module) {
  ASSERT_NE(module, nullptr);
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  ASSERT_NE(internal_module, nullptr);
  VerifyModule(internal_module);
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "caller"));
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "identity"));
  EXPECT_FALSE(ModuleHasSymbol(internal_module, "unused_harness"));
  EXPECT_FALSE(ModuleHasSymbol(internal_module, "unused_library"));
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

std::vector<uint8_t> WriteBytecodeModule(const char* source_text) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  loom_context_t context = {};
  loom_context_initialize(allocator, &context);
  IREE_CHECK_OK(loom_op_registry_register_all_dialects(&context));
  IREE_CHECK_OK(loom_context_finalize(&context));

  loom_text_parse_options_t parse_options = {
      /*.diagnostic_sink=*/{},
      /*.max_errors=*/20,
      /*.low_asm_environment=*/{},
  };
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source_text),
                                IREE_SV("bytecode-input.loom"), &context,
                                &block_pool, &parse_options, &module));
  if (module == nullptr) {
    ADD_FAILURE() << "bytecode fixture source did not parse";
    loom_context_deinitialize(&context);
    iree_arena_block_pool_deinitialize(&block_pool);
    return {};
  }

  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, allocator, &stream));
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, &block_pool));

  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  std::vector<uint8_t> bytes(length);
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  iree_io_stream_release(stream);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
  return bytes;
}

TEST(LinkTest, LinksSelectiveTextRoots) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr harness = CreateTextSource("harness.loom", R"(
func.decl @identity(%x: i32) -> (i32)

func.def public @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def public @unused_harness(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr library = CreateTextSource("library.loom", R"(
func.def public @identity(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @unused_library(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  AddSource(builder.get(), harness.get(), "harness",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);
  AddSource(builder.get(), library.get(), "library",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@caller"),
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  VerifyLinkedCallerModule(module.get());

  SourcePtr serialized_text = SerializeModuleToSource(
      module.get(), LOOMC_SOURCE_FORMAT_TEXT, "linked.loom");
  EXPECT_EQ(loomc_source_format(serialized_text.get()),
            LOOMC_SOURCE_FORMAT_TEXT);
  EXPECT_EQ(ToString(loomc_source_identifier(serialized_text.get())),
            "linked.loom");
  std::string linked_text =
      ToString(loomc_source_contents(serialized_text.get()));
  EXPECT_NE(linked_text.find("func.def public @caller"), std::string::npos);
  EXPECT_NE(linked_text.find("func.def public @identity"), std::string::npos);
  EXPECT_EQ(linked_text.find("unused_harness"), std::string::npos);
  EXPECT_EQ(linked_text.find("unused_library"), std::string::npos);
  ModulePtr text_source_module = DeserializeModuleFromSource(
      context.get(), workspace.get(), serialized_text.get());
  VerifyLinkedCallerModule(text_source_module.get());

  loomc_module_serialize_options_t text_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(text_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
  };
  FILE* text_file = tmpfile();
  ASSERT_NE(text_file, nullptr);
  LOOMC_EXPECT_OK(
      loomc_module_serialize_to_file(module.get(), &text_options, text_file));
  std::vector<uint8_t> text_file_bytes = ReadOpenFileBytes(text_file);
  EXPECT_EQ(ToString(loomc_make_byte_span(text_file_bytes.data(),
                                          text_file_bytes.size())),
            linked_text);
  ASSERT_EQ(fseek(text_file, 0, SEEK_SET), 0);
  ModulePtr text_file_module =
      DeserializeModuleFromFile(context.get(), workspace.get(), text_file);
  VerifyLinkedCallerModule(text_file_module.get());
  fclose(text_file);

  iree::testing::TempFilePath text_path("loomc-linked", ".loom");
  LOOMC_ASSERT_OK(loomc_module_serialize_to_path(
      module.get(), &text_options,
      loomc_make_string_view(text_path.path().data(), text_path.path().size()),
      loomc_allocator_system()));
  EXPECT_EQ(ReadPathToString(text_path.path()), linked_text);
  ModulePtr text_path_module = DeserializeModuleFromPath(
      context.get(), workspace.get(), text_path.path());
  VerifyLinkedCallerModule(text_path_module.get());
  EXPECT_TRUE(text_path.Remove());

  SourcePtr serialized_bytecode = SerializeModuleToSource(
      module.get(), LOOMC_SOURCE_FORMAT_BYTECODE, "linked.loombc");
  EXPECT_EQ(loomc_source_format(serialized_bytecode.get()),
            LOOMC_SOURCE_FORMAT_BYTECODE);
  EXPECT_GT(loomc_source_contents(serialized_bytecode.get()).data_length, 0u);
  ModulePtr bytecode_source_module = DeserializeModuleFromSource(
      context.get(), workspace.get(), serialized_bytecode.get());
  VerifyLinkedCallerModule(bytecode_source_module.get());

  loomc_byte_span_t bytecode_contents =
      loomc_source_contents(serialized_bytecode.get());
  SourcePtr inferred_bytecode_source =
      CreateSource(LOOMC_SOURCE_FORMAT_UNKNOWN, "inferred.loombc",
                   bytecode_contents.data, bytecode_contents.data_length);
  ModulePtr inferred_bytecode_source_module = DeserializeModuleFromSource(
      context.get(), workspace.get(), inferred_bytecode_source.get());
  VerifyLinkedCallerModule(inferred_bytecode_source_module.get());

  BuilderPtr serialized_builder = CreateBuilder(context.get());
  AddSource(serialized_builder.get(), serialized_bytecode.get(), "serialized",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);
  LinkIndexPtr serialized_index;
  FinishIndex(serialized_builder.get(), &serialized_index);

  loomc_module_serialize_options_t bytecode_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(bytecode_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  FILE* bytecode_file = tmpfile();
  ASSERT_NE(bytecode_file, nullptr);
  LOOMC_EXPECT_OK(loomc_module_serialize_to_file(
      module.get(), &bytecode_options, bytecode_file));
  EXPECT_GT(ReadOpenFileBytes(bytecode_file).size(), 0u);
  ASSERT_EQ(fseek(bytecode_file, 0, SEEK_SET), 0);
  ModulePtr bytecode_file_module =
      DeserializeModuleFromFile(context.get(), workspace.get(), bytecode_file);
  VerifyLinkedCallerModule(bytecode_file_module.get());
  fclose(bytecode_file);

  iree::testing::TempFilePath bytecode_path("loomc-linked", ".loombc");
  LOOMC_ASSERT_OK(loomc_module_serialize_to_path(
      module.get(), &bytecode_options,
      loomc_make_string_view(bytecode_path.path().data(),
                             bytecode_path.path().size()),
      loomc_allocator_system()));
  EXPECT_GT(ReadPathToString(bytecode_path.path()).size(), 0u);
  ModulePtr bytecode_path_module = DeserializeModuleFromPath(
      context.get(), workspace.get(), bytecode_path.path());
  VerifyLinkedCallerModule(bytecode_path_module.get());
  EXPECT_TRUE(bytecode_path.Remove());
}

TEST(LinkTest, SelectiveRootPullsBytecodeApplyProviders) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr harness = CreateTextSource("harness.loom", R"(
func.def public @caller(%x: i32) -> (i32) {
  %y = func.apply<demo.targeted>(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def public @unused_harness(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  std::vector<uint8_t> library_bytes = WriteBytecodeModule(R"(
func.template<demo.targeted> priority(20) @fast_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.targeted> priority(1) @fallback_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.unused> @unused_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr library =
      CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "library.loombc",
                   library_bytes.data(), library_bytes.size());
  AddSource(builder.get(), harness.get(), "harness",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);
  AddSource(builder.get(), library.get(), "library",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@caller"),
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  const loom_module_t* internal_module =
      loomc_module_const_loom_module(module.get());
  ASSERT_NE(internal_module, nullptr);
  VerifyModule(internal_module);
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "caller"));
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "fast_provider"));
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "fallback_provider"));
  EXPECT_FALSE(ModuleHasSymbol(internal_module, "unused_harness"));
  EXPECT_FALSE(ModuleHasSymbol(internal_module, "unused_provider"));
}

TEST(LinkTest, LinkModuleMaterializesInvocationConfigOnLinkedOutput) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("config.loom", R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]

func.def public @entry() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}
)");
  AddSource(builder.get(), source.get(), "config",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@entry"),
  };
  loomc_config_binding_t first_bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@model36.model.hidden_size"),
          /*.value=*/loomc_make_cstring_view("4096"),
      },
  };
  loomc_link_options_t first_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/first_bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_make_cstring_view(R"({
                "model36": {
                  "model": {"hidden_size": 2048}
                }
              })"),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
              LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  ResultPtr first_result;
  ModulePtr first_module =
      LinkIndex(linker.get(), workspace.get(), link_index.get(), &first_options,
                &first_result);

  ASSERT_TRUE(loomc_result_succeeded(first_result.get()));
  ASSERT_NE(first_module.get(), nullptr);
  std::string first_text = SerializeModuleToText(first_module.get());
  EXPECT_NE(
      first_text.find("config.def @model36.model.hidden_size = 4096 : index"),
      std::string::npos);
  EXPECT_EQ(first_text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
  EXPECT_EQ(first_text.find("2048"), std::string::npos);

  loomc_config_binding_t second_bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@model36.model.hidden_size"),
          /*.value=*/loomc_make_cstring_view("1024"),
      },
  };
  loomc_link_options_t second_options = first_options;
  second_options.config.bindings = second_bindings;
  second_options.config.binding_count = 1;
  second_options.config.json_object = loomc_string_view_empty();
  ResultPtr second_result;
  ModulePtr second_module =
      LinkIndex(linker.get(), workspace.get(), link_index.get(),
                &second_options, &second_result);

  ASSERT_TRUE(loomc_result_succeeded(second_result.get()));
  ASSERT_NE(second_module.get(), nullptr);
  std::string second_text = SerializeModuleToText(second_module.get());
  EXPECT_NE(
      second_text.find("config.def @model36.model.hidden_size = 1024 : index"),
      std::string::npos);
  EXPECT_EQ(second_text.find("4096"), std::string::npos);
}

TEST(LinkTest, LinkModuleMaterializesInvocationTargetOnLinkedOutput) {
  TargetEnvironmentPtr target_environment = CreateFakeTargetEnvironment();
  TargetProfilePtr profile = CreateFakeTargetProfile(target_environment.get());
  TargetSelectionPtr target_selection = CreateTargetSelection(profile.get());
  ContextPtr context = CreateContext(target_environment.get());
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("multi_root.loom", R"(
func.def public @first() {
  func.return
}

func.def public @second() {
  func.return
}
)");
  AddSource(builder.get(), source.get(), "multi_root",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection.get(),
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/&target_options,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  const loom_module_t* internal_module =
      loomc_module_const_loom_module(module.get());
  ASSERT_NE(internal_module, nullptr);
  VerifyModule(internal_module);
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "first"));
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "second"));
  EXPECT_TRUE(ModuleHasSymbol(internal_module, "selected_link_target"));

  std::string linked_text = SerializeModuleToText(module.get());
  EXPECT_NE(linked_text.find("test.target<low_core> @selected_link_target"),
            std::string::npos);
}

TEST(LinkTest, LinkAndCompileSelectsTargetApplicableProviderThroughCOptions) {
  TargetEnvironmentPtr target_environment = CreateFakeTargetEnvironment();
  TargetProfilePtr profile = CreateFakeTargetProfile(target_environment.get());
  TargetSelectionPtr target_selection = CreateTargetSelection(profile.get());
  ContextPtr context = CreateContext(target_environment.get());
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr root_source = CreateTextSource("root.loom", R"(
func.def public @entry(%arg: i32) -> (i32) {
  %result = func.apply<demo.capi_selected>(%arg) : (i32) -> (i32)
  func.return %result : i32
}
)");
  SourcePtr provider_source = CreateTextSource("providers.loom", R"(
test.target<low_core> @selected_link_target
test.target<low_core> @other_target

func.template<demo.capi_selected> target(@selected_link_target) priority(20) @selected_provider(%value: i32) -> (i32) {
  %selected = test.addi %value, %value : i32
  func.return %selected : i32
}

func.template<demo.capi_selected> target(@other_target) priority(30) @other_provider(%value: i32) -> (i32) {
  func.return %value : i32
}

func.template<demo.capi_selected> priority(1) @fallback_provider(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  AddSource(builder.get(), root_source.get(), "root",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);
  AddSource(builder.get(), provider_source.get(), "providers",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection.get(),
  };
  loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@entry"),
  };
  loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/&target_options,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
  };
  ResultPtr link_result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &link_options, &link_result);

  ASSERT_TRUE(loomc_result_succeeded(link_result.get()));
  ASSERT_NE(module.get(), nullptr);
  const loom_module_t* linked_module =
      loomc_module_const_loom_module(module.get());
  ASSERT_NE(linked_module, nullptr);
  VerifyModule(linked_module);
  EXPECT_TRUE(ModuleHasSymbol(linked_module, "selected_provider"));
  EXPECT_TRUE(ModuleHasSymbol(linked_module, "other_provider"));
  EXPECT_TRUE(ModuleHasSymbol(linked_module, "fallback_provider"));

  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePassProgramFromPipelineText(
      context.get(), "select-templates,inline-callables,symbol-dce");
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("selected_provider_module"),
      /*.artifact_flags=*/0,
      /*.config=*/{},
  };
  loomc_result_t* raw_compile_result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &raw_compile_result);
  LOOMC_ASSERT_OK(status);
  ResultPtr compile_result(raw_compile_result);
  ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));

  const loom_module_t* compiled_module =
      loomc_module_const_loom_module(module.get());
  ASSERT_NE(compiled_module, nullptr);
  VerifyModule(compiled_module);
  EXPECT_TRUE(ModuleHasSymbol(compiled_module, "entry"));
  EXPECT_TRUE(ModuleHasSymbol(compiled_module, "selected_link_target"));

  const std::string compiled_text = SerializeModuleToText(module.get());
  EXPECT_NE(compiled_text.find("test.target<low_core> @selected_link_target"),
            std::string::npos);
  EXPECT_NE(compiled_text.find("test.addi %arg, %arg : i32"),
            std::string::npos);
  EXPECT_EQ(compiled_text.find("func.apply<demo.capi_selected>"),
            std::string::npos);
  EXPECT_EQ(compiled_text.find("@selected_provider"), std::string::npos);
  EXPECT_EQ(compiled_text.find("@other_provider"), std::string::npos);
  EXPECT_EQ(compiled_text.find("@fallback_provider"), std::string::npos);
}

TEST(LinkTest, DeserializeParseErrorsProduceFailedResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("broken.loom", R"(
func.def public @broken(%x: i32) -> (i32) {
  func.return %missing : i32
}
)");

  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), nullptr,
      loomc_allocator_system(), &module, &result);
  LOOMC_EXPECT_OK(status);
  ModulePtr module_ptr(module);
  ResultPtr result_ptr(result);

  EXPECT_EQ(module_ptr.get(), nullptr);
  ASSERT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GT(loomc_result_diagnostic_count(result_ptr.get()), 0u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  ASSERT_NE(diagnostic->range.source, nullptr);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "broken.loom");
  source.reset();
  diagnostic = loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  ASSERT_NE(diagnostic->range.source, nullptr);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "broken.loom");
}

TEST(LinkTest, DuplicatePublicDefinitionsProduceFailedResult) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr first = CreateTextSource("first.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr second = CreateTextSource("second.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  AddSource(builder.get(), first.get(), "first",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);
  AddSource(builder.get(), second.get(), "second",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               nullptr, &result);

  EXPECT_EQ(module.get(), nullptr);
  ASSERT_FALSE(loomc_result_succeeded(result.get()));
  ASSERT_GT(loomc_result_diagnostic_count(result.get()), 0u);
}

TEST(LinkTest, LinksBytecodeAfterSourceRelease) {
  std::vector<uint8_t> bytecode = WriteBytecodeModule(R"(
func.def public @from_bytecode(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "module.loombc",
                                  bytecode.data(), bytecode.size());
  AddSource(builder.get(), source.get(), "bytecode",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  source.reset();
  bytecode.clear();

  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               nullptr, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  const loom_module_t* linked_module =
      loomc_module_const_loom_module(module.get());
  ASSERT_NE(linked_module, nullptr);
  VerifyModule(linked_module);
  EXPECT_TRUE(ModuleHasSymbol(linked_module, "from_bytecode"));
}

TEST(LinkTest, LinkModuleMaterializesConfigFromBytecodeIndex) {
  std::vector<uint8_t> bytecode = WriteBytecodeModule(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]

func.def public @from_bytecode() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}
)");
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "module.loombc",
                                  bytecode.data(), bytecode.size());
  AddSource(builder.get(), source.get(), "bytecode",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  source.reset();
  bytecode.clear();

  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@model36.model.hidden_size"),
          /*.value=*/loomc_make_cstring_view("4096"),
      },
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
              LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  std::string text = SerializeModuleToText(module.get());
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
}

TEST(LinkTest, LinkModuleAcceptsDeclaredUnusedConfigFromBytecodeIndex) {
  std::vector<uint8_t> bytecode = WriteBytecodeModule(R"(
config.decl @id4.vae.conv3x3_bias.output_channel_count : %value: index where [range(%value, 1, 8192)]

func.def public @generic() -> (index) {
  %channels = config.get @id4.vae.conv3x3_bias.output_channel_count : index
  func.return %channels : index
}

func.def public @specialized_rgb() -> (index) {
  %channels = index.constant 3 : index
  func.return %channels : index
}
)");
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "module.loombc",
                                  bytecode.data(), bytecode.size());
  AddSource(builder.get(), source.get(), "bytecode",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  source.reset();
  bytecode.clear();

  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@specialized_rgb"),
  };
  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/
          loomc_make_cstring_view("@id4.vae.conv3x3_bias.output_channel_count"),
          /*.value=*/loomc_make_cstring_view("3"),
      },
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
              LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module.get(), nullptr);
  std::string text = SerializeModuleToText(module.get());
  EXPECT_NE(text.find("func.def public @specialized_rgb"), std::string::npos);
  EXPECT_EQ(text.find("func.def public @generic"), std::string::npos);
  EXPECT_EQ(text.find("id4.vae.conv3x3_bias.output_channel_count"),
            std::string::npos);
}

TEST(LinkTest, LinkModuleRejectsConfigDeclaredOnlyByUnselectedProvider) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr selected_source = CreateTextSource("selected.loom", R"(
func.def public @selected() -> (index) {
  %value = index.constant 1 : index
  func.return %value : index
}
)");
  AddSource(builder.get(), selected_source.get(), "selected",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);
  SourcePtr unused_source = CreateTextSource("unused.loom", R"(
config.decl @unused.operation.tile_count : index

func.def public @unused() -> (index) {
  %tiles = config.get @unused.operation.tile_count : index
  func.return %tiles : index
}
)");
  AddSource(builder.get(), unused_source.get(), "unused",
            LOOMC_LINK_PROVIDER_ROLE_LIBRARY);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@selected"),
  };
  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@unused.operation.tile_count"),
          /*.value=*/loomc_make_cstring_view("4"),
      },
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/roots,
      /*.root_symbol_count=*/1,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN,
      },
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  EXPECT_EQ(module.get(), nullptr);
  ExpectFailedResultCode(result.get(), "CONFIG/INVALID");
}

TEST(LinkTest, LinkModuleReportsUnknownConfigAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("entry.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  AddSource(builder.get(), source.get(), "entry",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("tile_m"),
          /*.value=*/loomc_make_cstring_view("128"),
      },
  };
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN,
      },
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  EXPECT_EQ(module.get(), nullptr);
  ExpectFailedResultCode(result.get(), "CONFIG/INVALID");
}

TEST(LinkTest, LinkModuleReportsUnresolvedConfigAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  BuilderPtr builder = CreateBuilder(context.get());
  SourcePtr source = CreateTextSource("config.loom", R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]

func.def public @entry() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}
)");
  AddSource(builder.get(), source.get(), "config",
            LOOMC_LINK_PROVIDER_ROLE_INPUT);

  LinkIndexPtr link_index;
  FinishIndex(builder.get(), &link_index);
  LinkerPtr linker = CreateLinker(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  loomc_link_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.link_index=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/0,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  ResultPtr result;
  ModulePtr module = LinkIndex(linker.get(), workspace.get(), link_index.get(),
                               &options, &result);

  EXPECT_EQ(module.get(), nullptr);
  ExpectFailedResultCode(result.get(), "CONFIG/INVALID");
}

}  // namespace
