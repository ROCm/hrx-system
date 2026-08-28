// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include <algorithm>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/descriptors/descriptors.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

typedef struct KernelRequestCapture {
  std::vector<loom_cmd_program_kernel_request_t> requests;
} KernelRequestCapture;

static iree_status_t CaptureKernelRequest(
    void* user_data, loom_cmd_program_kernel_request_t request) {
  static_cast<KernelRequestCapture*>(user_data)->requests.push_back(request);
  return iree_ok_status();
}

typedef struct RejectKernelRequestState {
  iree_host_size_t publish_count;
} RejectKernelRequestState;

static iree_status_t RejectKernelRequest(
    void* user_data, loom_cmd_program_kernel_request_t request) {
  RejectKernelRequestState* state =
      static_cast<RejectKernelRequestState*>(user_data);
  ++state->publish_count;
  loom_kernel_class_product_deinitialize(&request.source.product);
  return iree_make_status(IREE_STATUS_ABORTED, "kernel request sink stopped");
}

class CmdProgramPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerify(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    const loom_low_descriptor_set_provider_t descriptor_set_providers[] = {
        loom_cmd_core_descriptor_set,
    };
    const loom_low_descriptor_registry_t descriptor_registry = {
        /*.descriptor_sets=*/{},
        /*.descriptor_set_count=*/{},
        /*.descriptor_set_providers=*/descriptor_set_providers,
        /*.descriptor_set_provider_count=*/
        IREE_ARRAYSIZE(descriptor_set_providers),
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry, &parse_options.low_asm_environment);
    DiagnosticCapture capture;
    parse_options.diagnostic_sink = capture.sink();
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("cmd_program_plan_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);
    if (!module) {
      for (const auto& diagnostic : capture.diagnostics) {
        ADD_FAILURE() << diagnostic.error->summary << " at line "
                      << diagnostic.origin_line << ", column "
                      << diagnostic.origin_column;
      }
      return module_ptr;
    }

    loom_verify_options_t verify_options = {};
    verify_options.max_errors = 20;
    verify_options.sink = capture.sink();
    loom_verify_result_t result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &result));
    for (const auto& diagnostic : capture.diagnostics) {
      ADD_FAILURE() << diagnostic.error->summary << " at line "
                    << diagnostic.origin_line << ", column "
                    << diagnostic.origin_column;
    }
    EXPECT_EQ(result.error_count, 0u);
    return module_ptr;
  }

  std::vector<uint8_t> WriteBytecode(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  loom_symbol_ref_t FindSymbolRef(loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  // Shared arena block pool backing source and prepared modules.
  iree_arena_block_pool_t block_pool_;

  // Source dialect context used by parsing, linking, and lowering.
  loom_context_t context_;
};

TEST_F(CmdProgramPlanTest, OwnsMultipleRootsAndDeduplicatesEntries) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @increment(%source: buffer, %target: buffer)
kernel.entry.decl @double(%source: buffer, %target: buffer)

command.program.def public @increment_then_double() launch(%source: buffer, %scratch: buffer, %target: buffer) {
  %count = index.constant 1 : index
  kernel.dispatch @increment[%count](%source, %scratch) : [index](buffer, buffer)
  kernel.dispatch @double[%count](%scratch, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public @increment_twice() launch(%source: buffer, %scratch: buffer, %target: buffer) {
  %count = index.constant 1 : index
  kernel.dispatch @increment[%count](%source, %scratch) : [index](buffer, buffer)
  kernel.dispatch @increment[%count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("increment_twice")),
      FindSymbolRef(source_module.get(), IREE_SV("increment_then_double")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_materialization(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      /*kernel_source=*/nullptr, loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 2u);
  ASSERT_NE(plan.roots, nullptr);
  ASSERT_NE(plan.root_module, nullptr);
  const loom_cmd_program_root_t& twice = plan.roots[0];
  const loom_cmd_program_root_t& mixed = plan.roots[1];
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_twice")),
            twice.function_op);
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_then_double")),
            mixed.function_op);
  EXPECT_TRUE(loom_low_func_def_isa(twice.function_op));
  EXPECT_TRUE(loom_low_func_def_isa(mixed.function_op));
  EXPECT_EQ(twice.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(mixed.launch_counts.binding_index, UINT32_MAX);

  ASSERT_EQ(plan.entry_requirement_count, 2u);
  ASSERT_NE(plan.entry_requirements, nullptr);
  EXPECT_EQ(plan.entry_requirements[0].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("increment")));
  EXPECT_EQ(plan.entry_requirements[1].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("double")));
  ASSERT_EQ(twice.entry_requirement_count, 1u);
  ASSERT_NE(twice.entry_requirement_indices, nullptr);
  EXPECT_EQ(twice.entry_requirement_indices[0], 0u);
  ASSERT_EQ(mixed.entry_requirement_count, 2u);
  ASSERT_NE(mixed.entry_requirement_indices, nullptr);
  EXPECT_EQ(mixed.entry_requirement_indices[0], 0u);
  EXPECT_EQ(mixed.entry_requirement_indices[1], 1u);

  loom_cmd_program_artifact_set_t artifact_set = {};
  IREE_ASSERT_OK(loom_cmd_program_artifact_set_build(&plan, &artifact_set,
                                                     iree_allocator_system()));
  ASSERT_EQ(artifact_set.programs.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(artifact_set.programs.values[0].symbol,
                                     IREE_SV("increment_twice")));
  EXPECT_TRUE(iree_string_view_equal(artifact_set.programs.values[1].symbol,
                                     IREE_SV("increment_then_double")));
  ASSERT_EQ(artifact_set.entries.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(artifact_set.entries.values[0].symbol,
                                     IREE_SV("increment")));
  EXPECT_TRUE(iree_string_view_equal(artifact_set.entries.values[1].symbol,
                                     IREE_SV("double")));
  ASSERT_EQ(artifact_set.programs.values[0].entry_requirement_count, 1u);
  EXPECT_EQ(artifact_set.programs.values[0].entry_requirement_indices[0], 0u);
  ASSERT_EQ(artifact_set.programs.values[1].entry_requirement_count, 2u);
  EXPECT_EQ(artifact_set.programs.values[1].entry_requirement_indices[0], 0u);
  EXPECT_EQ(artifact_set.programs.values[1].entry_requirement_indices[1], 1u);
  loom_cmd_program_artifact_set_deinitialize(&artifact_set);

  loom_cmd_program_plan_deinitialize(&plan);
  EXPECT_EQ(plan.root_module, nullptr);
  EXPECT_EQ(plan.roots, nullptr);
  EXPECT_EQ(plan.entry_requirements, nullptr);
}

TEST_F(CmdProgramPlanTest, IndexedPreparationExpandsNestedCommandTemplates) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @one_group() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%storage: buffer) {
  kernel.return
}

kernel.def @eight_groups() {
  %unit = index.constant 1 : index
  %c8 = index.constant 8 : index
  kernel.launch.config workgroups(%c8, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%storage: buffer) {
  kernel.return
}

template.decl @test.command_schedule(%token_count: index, %storage: buffer)
template.decl @test.dispatch_schedule(%token_count: index, %storage: buffer)

template.def<@test.command_schedule> @decode(%token_count: index, %storage: buffer) {
  template.apply<@test.dispatch_schedule>(%token_count, %storage) : (index, buffer)
  template.return
}

template.def<@test.dispatch_schedule> @dispatch_one(%token_count: index, %storage: buffer) where [eq(%token_count, 1)] {
  kernel.launch @one_group(%storage) : (buffer)
  template.return
}

template.def<@test.dispatch_schedule> @dispatch_many(%token_count: index, %storage: buffer) where [range(%token_count, 2, 8)] {
  kernel.launch @eight_groups(%storage) : (buffer)
  template.return
}

command.program.def public @selected_schedule() launch(%storage: buffer) {
  %token_count = index.constant 1 : index
  template.apply<@test.command_schedule>(%token_count, %storage) : (index, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);

  loom_link_module_index_t* index = nullptr;
  IREE_ASSERT_OK(loom_link_module_index_allocate(
      &context_, &block_pool_, iree_allocator_system(), &index));
  const loom_link_module_index_add_options_t add_options = {
      /*.provider_name=*/IREE_SV("indexed_command_test"),
  };
  iree_host_size_t provider_ordinal = 0;
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index, source_module.get(), &add_options, &provider_ordinal));
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index, provider_ordinal);
  ASSERT_NE(provider, nullptr);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index, provider->module_start_ordinal);
  ASSERT_NE(indexed_module, nullptr);
  const loom_symbol_ref_t source_root =
      FindSymbolRef(source_module.get(), IREE_SV("selected_schedule"));
  const iree_host_size_t root_symbol_ordinal =
      indexed_module->symbol_start_ordinal + source_root.symbol_id;

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_cmd_program_plan_t plan = {};
  DiagnosticEmissionCapture diagnostic_capture;
  bool valid = false;
  const loom_link_plan_materialization_environment_t environment = {
      /*.context=*/&context_,
      /*.block_pool=*/&block_pool_,
      /*.low_repr_environment=*/{},
      /*.diagnostic_sink=*/nullptr,
      /*.prepare_module=*/nullptr,
      /*.user_data=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  iree_status_t status = loom_cmd_program_plan_prepare_index(
      index, &root_symbol_ordinal, 1, /*options=*/nullptr,
      loom_pass_builtin_registry(), diagnostic_capture.emitter(), &environment,
      &scratch_arena, &valid, &plan);
  loom_link_module_index_free(index);
  source_module.reset();
  iree_arena_deinitialize(&scratch_arena);
  if (!iree_status_is_ok(status)) {
    loom_cmd_program_plan_deinitialize(&plan);
    IREE_ASSERT_OK(status);
  }
  if (!valid) {
    for (const auto& emission : diagnostic_capture.emissions) {
      std::string detail = emission.error->summary;
      for (const std::string& parameter : emission.string_params) {
        detail.append(" [").append(parameter).append("]");
      }
      ADD_FAILURE() << detail;
    }
    loom_cmd_program_plan_deinitialize(&plan);
    ASSERT_TRUE(valid);
  }

  ASSERT_EQ(plan.root_count, 1u);
  ASSERT_EQ(plan.entry_requirement_count, 1u);
  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.entry_requirement_count, 1u);
  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &data,
                                                      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  ASSERT_EQ(program.commands.count, 1u);
  const loom_cmd_program_command_t dispatch =
      loom_cmd_program_command_at(&program, 0);
  ASSERT_EQ(dispatch.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_x, 1u);

  iree_allocator_free(iree_allocator_system(), data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest,
       BodyBlindPreparationDoesNotReadKernelImplementationBytes) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @child() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}

command.program.def public @root() launch() {
  kernel.launch @child() : ()
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  std::vector<uint8_t> bytecode = WriteBytecode(source_module.get());

  loom_link_module_index_t* index = nullptr;
  IREE_ASSERT_OK(loom_link_module_index_allocate(
      &context_, &block_pool_, iree_allocator_system(), &index));
  loom_bytecode_index_options_t index_options = {};
  index_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("provider.loombc"), &index_options, /*options=*/nullptr,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_symbol_t* root =
      loom_link_module_index_lookup_name(index, IREE_SV("root"));
  const loom_link_module_index_symbol_t* child =
      loom_link_module_index_lookup_name(index, IREE_SV("child"));
  ASSERT_NE(root, nullptr);
  ASSERT_NE(child, nullptr);

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(index, child);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_symbol_module(index, child);
  ASSERT_NE(provider, nullptr);
  ASSERT_NE(indexed_module, nullptr);
  const loom_bytecode_module_metadata_t* metadata =
      &provider->bytecode.metadata
           .modules[indexed_module->provider_module_ordinal];
  const loom_bytecode_symbol_metadata_t* symbol =
      &metadata->symbols[child->module_symbol_ordinal];
  ASSERT_NE(symbol->kernel_workload_region_payload_ordinal_plus_one, 0u);
  ASSERT_NE(symbol->body_region_payload_ordinal_plus_one, 0u);
  const loom_bytecode_region_payload_metadata_t* body_payload =
      &metadata
           ->region_payloads[symbol->first_region_payload_index +
                             symbol->body_region_payload_ordinal_plus_one - 1];
  ASSERT_GT(body_payload->length, 0u);
  std::fill_n(bytecode.data() + body_payload->absolute_offset,
              body_payload->length, UINT8_C(0xFF));

  const loom_link_plan_materialization_environment_t environment = {
      /*.context=*/&context_,
      /*.block_pool=*/&block_pool_,
      /*.low_repr_environment=*/{},
      /*.diagnostic_sink=*/nullptr,
      /*.prepare_module=*/nullptr,
      /*.user_data=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  iree_arena_allocator_t body_blind_arena;
  iree_arena_initialize(&block_pool_, &body_blind_arena);
  loom_cmd_program_plan_t body_blind_plan = {};
  bool body_blind_valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_index(
      index, &root->ordinal, /*program_count=*/1, /*options=*/nullptr,
      loom_pass_builtin_registry(), /*diagnostic_emitter=*/{}, &environment,
      &body_blind_arena, &body_blind_valid, &body_blind_plan));
  ASSERT_TRUE(body_blind_valid);
  ASSERT_EQ(body_blind_plan.root_count, 1u);
  ASSERT_EQ(body_blind_plan.entry_requirement_count, 1u);
  EXPECT_FALSE(body_blind_plan.entry_requirements[0].has_source_request);
  loom_cmd_program_plan_deinitialize(&body_blind_plan);
  iree_arena_deinitialize(&body_blind_arena);

  KernelRequestCapture request_capture;
  loom_cmd_program_plan_index_options_t request_options;
  loom_cmd_program_plan_index_options_initialize(&request_options);
  request_options.kernel_request_sink = {
      /*.publish=*/CaptureKernelRequest,
      /*.user_data=*/&request_capture,
  };
  iree_arena_allocator_t request_arena;
  iree_arena_initialize(&block_pool_, &request_arena);
  loom_cmd_program_plan_t request_plan = {};
  bool request_valid = false;
  IREE_EXPECT_NOT_OK(loom_cmd_program_plan_prepare_index(
      index, &root->ordinal, /*program_count=*/1, &request_options,
      loom_pass_builtin_registry(), /*diagnostic_emitter=*/{}, &environment,
      &request_arena, &request_valid, &request_plan));
  EXPECT_FALSE(request_valid);
  EXPECT_EQ(request_plan.root_module, nullptr);
  EXPECT_TRUE(request_capture.requests.empty());
  loom_cmd_program_plan_deinitialize(&request_plan);
  iree_arena_deinitialize(&request_arena);
  loom_link_module_index_free(index);
}

TEST_F(CmdProgramPlanTest,
       IndexedPreparationStreamsIndependentKernelsAndSharedClasses) {
  ModulePtr source_module = ParseAndVerify(R"(
template.decl @request.schedule(%size: index)

template.def<@request.schedule> priority(10) @large(%size: index) where [ge(%size, 128)] {
  template.return
}

template.def<@request.schedule> priority(1) @small(%size: index) {
  template.return
}

kernel.def @classified() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%size: index, %storage: buffer) {
  template.apply<@request.schedule>(%size) : (index)
  kernel.return
}

kernel.def @fixed() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%storage: buffer) {
  kernel.return
}

kernel.entry.decl @external(%storage: buffer)

command.program.def public @root_a() launch(%storage: buffer) {
  %small = index.constant 64 : index
  %large = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch @classified(%small, %storage) : (index, buffer)
  kernel.launch @classified(%large, %storage) : (index, buffer)
  kernel.launch @fixed(%storage) : (buffer)
  kernel.dispatch @external[%unit](%storage) : [index](buffer)
  command.return
}

command.program.def public @root_b() launch(%storage: buffer) {
  %small = index.constant 64 : index
  %large = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch @classified(%large, %storage) : (index, buffer)
  kernel.launch @classified(%small, %storage) : (index, buffer)
  kernel.launch @fixed(%storage) : (buffer)
  kernel.launch @fixed(%storage) : (buffer)
  kernel.dispatch @external[%unit](%storage) : [index](buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);

  loom_link_module_index_t* index = nullptr;
  IREE_ASSERT_OK(loom_link_module_index_allocate(
      &context_, &block_pool_, iree_allocator_system(), &index));
  const loom_link_module_index_add_options_t add_options = {
      /*.provider_name=*/IREE_SV("indexed_kernel_request_test"),
  };
  iree_host_size_t provider_ordinal = 0;
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      index, source_module.get(), &add_options, &provider_ordinal));
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index, provider_ordinal);
  ASSERT_NE(provider, nullptr);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index, provider->module_start_ordinal);
  ASSERT_NE(indexed_module, nullptr);
  const loom_symbol_ref_t root_a_ref =
      FindSymbolRef(source_module.get(), IREE_SV("root_a"));
  const loom_symbol_ref_t root_b_ref =
      FindSymbolRef(source_module.get(), IREE_SV("root_b"));
  const iree_host_size_t root_symbol_ordinals[] = {
      indexed_module->symbol_start_ordinal + root_a_ref.symbol_id,
      indexed_module->symbol_start_ordinal + root_b_ref.symbol_id,
  };

  KernelRequestCapture request_capture;
  loom_cmd_program_plan_index_options_t plan_options;
  loom_cmd_program_plan_index_options_initialize(&plan_options);
  plan_options.kernel_request_sink = {
      /*.publish=*/CaptureKernelRequest,
      /*.user_data=*/&request_capture,
  };
  const loom_link_plan_materialization_environment_t environment = {
      /*.context=*/&context_,
      /*.block_pool=*/&block_pool_,
      /*.low_repr_environment=*/{},
      /*.diagnostic_sink=*/nullptr,
      /*.prepare_module=*/nullptr,
      /*.user_data=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_cmd_program_plan_t plan = {};
  DiagnosticEmissionCapture diagnostic_capture;
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_index(
      index, root_symbol_ordinals, IREE_ARRAYSIZE(root_symbol_ordinals),
      &plan_options, loom_pass_builtin_registry(), diagnostic_capture.emitter(),
      &environment, &scratch_arena, &valid, &plan));
  ASSERT_TRUE(valid);

  RejectKernelRequestState reject_state = {};
  plan_options.kernel_request_sink = {
      /*.publish=*/RejectKernelRequest,
      /*.user_data=*/&reject_state,
  };
  iree_arena_allocator_t reject_scratch_arena;
  iree_arena_initialize(&block_pool_, &reject_scratch_arena);
  loom_cmd_program_plan_t rejected_plan = {};
  bool rejected_valid = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_cmd_program_plan_prepare_index(
          index, root_symbol_ordinals, IREE_ARRAYSIZE(root_symbol_ordinals),
          &plan_options, loom_pass_builtin_registry(),
          diagnostic_capture.emitter(), &environment, &reject_scratch_arena,
          &rejected_valid, &rejected_plan));
  EXPECT_FALSE(rejected_valid);
  EXPECT_EQ(rejected_plan.root_module, nullptr);
  EXPECT_EQ(rejected_plan.roots, nullptr);
  EXPECT_EQ(reject_state.publish_count, 1u);

  loom_link_module_index_free(index);
  source_module.reset();
  iree_arena_deinitialize(&reject_scratch_arena);
  iree_arena_deinitialize(&scratch_arena);

  ASSERT_EQ(request_capture.requests.size(), 3u);
  EXPECT_EQ(request_capture.requests[0].source.member_count, 2u);
  EXPECT_EQ(request_capture.requests[1].source.member_count, 2u);
  EXPECT_EQ(request_capture.requests[2].source.member_count, 3u);
  EXPECT_NE(request_capture.requests[0].source.class_ordinal,
            request_capture.requests[1].source.class_ordinal);

  ASSERT_EQ(plan.entry_requirement_count, 4u);
  iree_host_size_t source_requirement_count = 0;
  iree_host_size_t external_requirement_count = 0;
  for (iree_host_size_t i = 0; i < plan.entry_requirement_count; ++i) {
    if (plan.entry_requirements[i].has_source_request) {
      ++source_requirement_count;
    } else {
      ++external_requirement_count;
    }
  }
  EXPECT_EQ(source_requirement_count, 3u);
  EXPECT_EQ(external_requirement_count, 1u);

  ASSERT_EQ(plan.root_count, 2u);
  const loom_cmd_program_root_t& root_a = plan.roots[0];
  const loom_cmd_program_root_t& root_b = plan.roots[1];
  ASSERT_EQ(root_a.entry_requirement_count, 4u);
  ASSERT_EQ(root_b.entry_requirement_count, 4u);
  EXPECT_EQ(root_a.entry_requirement_indices[0],
            root_b.entry_requirement_indices[1]);
  EXPECT_EQ(root_a.entry_requirement_indices[1],
            root_b.entry_requirement_indices[0]);
  EXPECT_EQ(root_a.entry_requirement_indices[2],
            root_b.entry_requirement_indices[2]);
  EXPECT_EQ(root_a.entry_requirement_indices[3],
            root_b.entry_requirement_indices[3]);
  EXPECT_FALSE(plan.entry_requirements[root_a.entry_requirement_indices[3]]
                   .has_source_request);

  for (loom_cmd_program_kernel_request_t& request : request_capture.requests) {
    ASSERT_NE(request.source.product.module, nullptr);
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t verify_result = {};
    IREE_ASSERT_OK(loom_verify_module(request.source.product.module,
                                      &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
    loom_kernel_class_product_deinitialize(&request.source.product);
  }
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, OwnsParameterRequirementTables) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @combine(%lhs: view<3xi32>, %rhs: view<4xi32>, %target: buffer)

command.program.def public @parameterized() launch(%parameters: buffer, %target: buffer) {
  %layer = index.constant 3 : index
  %count = index.constant 1 : index
  %lhs = command.parameter %parameters, "blk.{}.lhs"[%layer] : view<3xi32>
  %rhs = command.parameter %parameters, "shared.rhs" : view<4xi32>
  kernel.dispatch @combine[%count](%lhs, %rhs, %target) : [index](view<3xi32>, view<4xi32>, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("parameterized")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_materialization(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      /*kernel_source=*/nullptr, loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 1u);
  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.parameters.root_count, 1u);
  ASSERT_NE(root.parameters.roots, nullptr);
  EXPECT_EQ(root.parameters.roots[0].source_binding_ordinal, 0u);
  EXPECT_EQ(root.parameters.roots[0].fixed_buffer_index, 0u);
  EXPECT_EQ(root.parameters.roots[0].required_byte_length, 272u);
  EXPECT_EQ(root.parameters.roots[0].minimum_alignment, 256u);

  ASSERT_EQ(root.parameters.count, 2u);
  ASSERT_NE(root.parameters.entries, nullptr);
  const loom_cmd_parameter_requirement_t& lhs = root.parameters.entries[0];
  EXPECT_TRUE(iree_string_view_equal(lhs.key, IREE_SV("blk.3.lhs")));
  EXPECT_EQ(lhs.source_binding_ordinal, 0u);
  EXPECT_EQ(lhs.fixed_buffer_index, 0u);
  EXPECT_EQ(lhs.byte_offset, 0u);
  EXPECT_EQ(lhs.byte_length, 12u);
  EXPECT_EQ(lhs.minimum_alignment, 256u);
  const loom_cmd_parameter_requirement_t& rhs = root.parameters.entries[1];
  EXPECT_TRUE(iree_string_view_equal(rhs.key, IREE_SV("shared.rhs")));
  EXPECT_EQ(rhs.source_binding_ordinal, 0u);
  EXPECT_EQ(rhs.fixed_buffer_index, 0u);
  EXPECT_EQ(rhs.byte_offset, 256u);
  EXPECT_EQ(rhs.byte_length, 16u);
  EXPECT_EQ(rhs.minimum_alignment, 256u);
  EXPECT_EQ(root.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(root.transient.required_byte_length, 0u);
  EXPECT_EQ(root.transient.minimum_alignment, 0u);

  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, OwnsBodylessEntryRequirementsAndArtifact) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @configured_a(%scale: i8, %output: buffer)
kernel.entry.decl @configured_b(%output: buffer)

command.program.def public @bodyless() launch(%output: buffer) {
  %count_x = index.constant 7 : index
  %count_y = index.constant 5 : index
  %scale = scalar.constant -1 : i8
  command.concurrent {
    kernel.dispatch @configured_a[%count_x](%scale, %output) : [index](i8, buffer)
    kernel.dispatch @configured_a[%count_x](%scale, %output) : [index](i8, buffer)
  }
  kernel.dispatch @configured_b[%count_x, %count_y](%output) : [index, index](buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("bodyless")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_materialization(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      /*kernel_source=*/nullptr, loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 1u);
  ASSERT_NE(plan.root_module, nullptr);
  ASSERT_EQ(plan.entry_requirement_count, 2u);
  ASSERT_NE(plan.entry_requirements, nullptr);
  EXPECT_EQ(plan.entry_requirements[0].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("configured_a")));
  EXPECT_EQ(plan.entry_requirements[1].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("configured_b")));

  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.entry_requirement_count, 2u);
  ASSERT_NE(root.entry_requirement_indices, nullptr);
  EXPECT_EQ(root.entry_requirement_indices[0], 0u);
  EXPECT_EQ(root.entry_requirement_indices[1], 1u);
  EXPECT_EQ(root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(root.launch_counts.required_byte_length, 0u);
  EXPECT_EQ(root.launch_counts.minimum_alignment, 0u);

  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &data,
                                                      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  EXPECT_EQ(program.requirements.executable_count, 2u);
  EXPECT_EQ(program.requirements.entry_count, 2u);
  ASSERT_EQ(program.commands.count, 3u);
  const loom_cmd_program_command_t first =
      loom_cmd_program_command_at(&program, 0);
  const loom_cmd_program_command_t second =
      loom_cmd_program_command_at(&program, 1);
  const loom_cmd_program_command_t third =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(first.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(second.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(third.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  EXPECT_EQ(first.payload.dispatch_direct.executable_index, 0u);
  EXPECT_EQ(first.payload.dispatch_direct.entry_index, 0u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_x, 7u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_y, 1u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_z, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.executable_index, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.entry_index, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_x, 7u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_y, 5u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_z, 1u);

  iree_allocator_free(iree_allocator_system(), data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, SerializesIndirectCountOrigins) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @stable_entry()
kernel.entry.decl @produce_counts(%counts: view<3xi32>)
kernel.entry.decl @dynamic_entry()

command.program.def public @stable_root() launch(%count_storage: buffer) {
  %c0 = index.constant 0 : offset
  %counts = buffer.view %count_storage[%c0] : buffer -> view<3xi32>
  kernel.dispatch @stable_entry[%counts]() : [view<3xi32>]()
  command.return
}

command.program.def public @dynamic_root() launch() {
  %c0 = index.constant 0 : offset
  %c1 = index.constant 1 : index
  %count_bytes = index.constant 12 : offset
  %count_storage = buffer.alloca<global> align(4) %count_bytes : buffer
  %counts = buffer.view %count_storage[%c0] : buffer -> view<3xi32>
  kernel.dispatch @produce_counts[%c1](%counts) : [index](view<3xi32>)
  kernel.dispatch @dynamic_entry[%counts]() : [view<3xi32>]()
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("stable_root")),
      FindSymbolRef(source_module.get(), IREE_SV("dynamic_root")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare_materialization(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      /*kernel_source=*/nullptr, loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 2u);
  const loom_cmd_program_root_t& stable_root = plan.roots[0];
  EXPECT_EQ(stable_root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(stable_root.launch_counts.required_byte_length, 0u);
  EXPECT_EQ(stable_root.launch_counts.minimum_alignment, 0u);
  iree_byte_span_t stable_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &stable_data,
                                                      iree_allocator_system()));
  loom_cmd_program_t stable_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(stable_data.data, stable_data.data_length),
      &stable_program));
  ASSERT_EQ(stable_program.commands.count, 1u);
  const loom_cmd_program_command_t stable_dispatch =
      loom_cmd_program_command_at(&stable_program, 0);
  EXPECT_EQ(stable_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(
          &stable_program,
          stable_dispatch.payload.dispatch_indirect.workgroup_count_buffer_ref)
          .root_index,
      0u);

  const loom_cmd_program_root_t& dynamic_root = plan.roots[1];
  EXPECT_EQ(dynamic_root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(dynamic_root.transient.binding_index, 0u);
  iree_byte_span_t dynamic_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 1, &dynamic_data,
                                                      iree_allocator_system()));
  loom_cmd_program_t dynamic_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(dynamic_data.data, dynamic_data.data_length),
      &dynamic_program));
  ASSERT_EQ(dynamic_program.commands.count, 2u);
  const loom_cmd_program_command_t dynamic_dispatch =
      loom_cmd_program_command_at(&dynamic_program, 1);
  EXPECT_EQ(dynamic_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC_BARRIER);
  const loom_cmd_program_buffer_ref_t dynamic_count_ref =
      loom_cmd_program_buffer_ref_at(&dynamic_program,
                                     dynamic_dispatch.payload.dispatch_indirect
                                         .workgroup_count_buffer_ref);
  EXPECT_EQ(dynamic_count_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(dynamic_count_ref.root_index,
            dynamic_program.requirements.transient.binding_index);

  iree_allocator_free(iree_allocator_system(), dynamic_data.data);
  iree_allocator_free(iree_allocator_system(), stable_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, SerializesMovedReferenceValues) {
  ModulePtr module = ParseAndVerify(R"(
low.func.def target<cmd.core> abi(command_program) @moved_binding() {
  %binding = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %moved = low.move %binding : reg<cmd.binding> -> reg<cmd.binding>
  %zero_u32 = low.const<cmd.constant.u32> {value = 0} : reg<cmd.u32>
  %one_u32 = low.const<cmd.constant.u32> {value = 1} : reg<cmd.u32>
  %zero_u64 = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %length = low.const<cmd.constant.u64> {value = 16} : reg<cmd.u64>
  %target_ref = low.op<cmd.buffer.ref.binding>(%moved, %zero_u64, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  low.op<cmd.fill>(%target_ref, %zero_u32, %one_u32) : (reg<cmd.buffer_ref>, reg<cmd.u32>, reg<cmd.u32>)
  low.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_cmd_program_root_t root = {};
  root.function_op = FindSymbol(module.get(), IREE_SV("moved_binding"));
  root.abi_layout.rebindable_binding_count = 1;
  root.transient.binding_index = UINT32_MAX;
  root.launch_counts.binding_index = UINT32_MAX;
  loom_cmd_program_plan_t plan = {};
  plan.root_module = module.get();
  plan.roots = &root;
  plan.root_count = 1;

  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &data,
                                                      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  EXPECT_EQ(program.requirements.rebindable_binding_count, 1u);
  EXPECT_EQ(program.commands.count, 1u);
  iree_allocator_free(iree_allocator_system(), data.data);
}

}  // namespace
}  // namespace loom
