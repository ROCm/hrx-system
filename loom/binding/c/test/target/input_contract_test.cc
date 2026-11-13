// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/environment.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"
#include "iree/vm/sync.h"
#include "loom/binding/c/test/target/testdata/input_contracts/input_contracts.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"

namespace {

static void Check(loomc_status_t status) {
  IREE_CHECK_OK(iree_status_from_loomc(status));
}
static void Require(bool value, const char* message) {
  if (!value) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}
static void Done(loomc_result_t* result) {
  if (!loomc_result_succeeded(result)) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto* d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s: %.*s\n", int(d->code.size), d->code.data,
                   int(d->message.size), d->message.data);
    }
    std::abort();
  }
  loomc_result_release(result);
}
static int Execute(loomc_byte_span_t contents) {
  auto allocator = iree_allocator_system();
  iree_vm_environment_t* environment = nullptr;
  IREE_CHECK_OK(iree_vm_environment_allocate(allocator, &environment));
  iree_vm_module_t* module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("control"),
      {iree_make_const_byte_span(contents.data, contents.data_length),
       iree_allocator_null()},
      allocator, &module));
  iree_vm_program_t* program = nullptr;
  IREE_CHECK_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                       allocator, &program));
  alignas(iree_max_align_t) uint8_t storage[16384] = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_CHECK_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage, sizeof(storage)), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_CHECK_OK(iree_vm_process_create(
      program, invocation, iree_vm_variant_span_empty(), allocator, &process));
  iree_vm_function_t function = iree_vm_function_null();
  IREE_CHECK_OK(iree_vm_process_lookup_function(process, IREE_SV("control"),
                                                IREE_SV("entry"), &function));
  uint8_t values[96];
  for (size_t i = 0; i < sizeof(values); ++i) {
    values[i] = uint8_t(i);
  }
  iree_vm_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(values, sizeof(values)), 0,
                           allocator, &buffer));
  iree_vm_ref_types_t types = {};
  IREE_CHECK_OK(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_ptr_borrowed(buffer, types.buffer)};
  iree_vm_variant_t results[1] = {};
  IREE_CHECK_OK(iree_vm_invoke(invocation, function,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results)));
  int32_t value = 0;
  IREE_CHECK_OK(iree_vm_i32_from_variant(results[0], &value));
  iree_vm_buffer_release(buffer);
  iree_vm_process_release(process);
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  iree_vm_environment_free(environment);
  return value;
}

static int CompileAndExecute(const std::vector<uint8_t>& bytes) {
  auto allocator = loomc_allocator_system();
  loomc_target_environment_t* environment = nullptr;
  Check(loomc_target_environment_create_vm(allocator, &environment));
  loomc_target_profile_t* profile = nullptr;
  Check(loomc_target_profile_select(
      environment, loomc_make_cstring_view("vm:core"), allocator, &profile));
  const loomc_context_target_options_t targets = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(targets),
      .target_environment = environment,
  };
  const loomc_context_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &targets,
  };
  loomc_context_t* context = nullptr;
  Check(loomc_context_create(&options, allocator, &context));
  loomc_workspace_t* workspace = nullptr;
  Check(loomc_workspace_create(nullptr, allocator, &workspace));
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
      .identifier = loomc_make_cstring_view("imported-inputs.loombc"),
      .contents = loomc_make_byte_span(bytes.data(), bytes.size()),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  Check(loomc_source_create(&source_options, allocator, &source));
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  Check(loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, allocator, &module, &result));
  Done(result);
  loomc_source_release(source);
  Check(loomc_module_verify(module, nullptr, allocator, &result));
  Done(result);
  loomc_compiler_t* compiler = nullptr;
  Check(loomc_compiler_create(context, nullptr, allocator, &compiler));
  loomc_pass_program_t* pipeline = nullptr;
  Check(loomc_pass_program_create_from_target_pipeline(
      context, nullptr, allocator, &pipeline, &result));
  Done(result);
  const loomc_string_view_t root = loomc_make_cstring_view("entry");
  const loomc_target_specialization_t specialization = {
      .function_symbol = root,
      .target_profile = profile,
  };
  const loomc_target_specialization_options_t target = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target),
      .specializations = &specialization,
      .specialization_count = 1,
  };
  const loomc_compile_options_t compile = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile),
      .next = &target,
  };
  Check(loomc_compile_module(compiler, workspace, pipeline, module, &compile,
                             allocator, &result));
  Done(result);
  Check(loomc_emit_module(environment, workspace, module, nullptr, allocator,
                          &result));
  Require(loomc_result_succeeded(result), "VM emission failed");
  Require(loomc_result_artifact_count(result) == 1, "expected one VM artifact");
  loomc_byte_span_t contents = {};
  Check(loomc_byte_sequence_clone(loomc_result_artifact_at(result, 0)->contents,
                                  allocator, &contents));
  Done(result);
  loomc_module_release(module);
  loomc_pass_program_release(pipeline);
  loomc_compiler_release(compiler);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_profile_release(profile);
  loomc_target_environment_release(environment);
  const int value = Execute(contents);
  loomc_allocator_free(allocator, const_cast<uint8_t*>(contents.data));
  return value;
}
static loomc_source_t* Source(const char* filename) {
  const auto* files = loomc_input_contracts_testdata_create();
  for (size_t i = 0; i < loomc_input_contracts_testdata_size(); ++i) {
    const auto& file = files[i];
    if (std::strcmp(filename, file.name) != 0) {
      continue;
    }
    const loomc_source_options_t options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view(file.name),
        .contents = loomc_make_byte_span(file.data, file.size),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    Check(loomc_source_create(&options, loomc_allocator_system(), &source));
    return source;
  }
  Require(false, "missing input contract fixture");
  return nullptr;
}
static loomc_module_t* Parse(loomc_context_t* context,
                             loomc_workspace_t* workspace,
                             loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  Check(loomc_module_deserialize_from_source(context, workspace, source,
                                             nullptr, loomc_allocator_system(),
                                             &module, &result));
  Done(result);
  return module;
}
static loomc_source_t* Serialize(loomc_module_t* module) {
  const loomc_module_serialize_options_t options = {
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  loomc_source_t* source = nullptr;
  Check(loomc_module_serialize_to_source(module, &options,
                                         loomc_allocator_system(), &source));
  return source;
}
struct Case {
  // Caller source with its own complete storage identity.
  const char* filename;
  // Row-one byte offset distinguishes the selected storage implementation.
  int expected;
};
static constexpr Case cases[] = {
    {.filename = "dense_q4.loom", .expected = 18},
    {.filename = "dense_q8.loom", .expected = 34},
    {.filename = "padded_q4.loom", .expected = 26},
    {.filename = "strided_q4.loom", .expected = 18},
};

// One frozen source index is reused by independent caller modules. Each caller
// interns its encoding first, so imported provider IDs differ across calls.
// Execution happens only after every library and link owner has retired.
TEST(InputContractTest,
     ReusedLibrariesKeepCallerEncodingsAfterSourceRetirement) {
  for (bool bytecode : {false, true}) {
    for (bool reversed : {false, true}) {
      loomc_context_t* context = nullptr;
      loomc_workspace_t* workspace = nullptr;
      Check(loomc_context_create(nullptr, loomc_allocator_system(), &context));
      Check(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                   &workspace));
      loomc_source_t* source =
          Source(reversed ? "providers_reversed.loom" : "providers.loom");
      loomc_module_t* native = Parse(context, workspace, source);
      loomc_source_release(source);
      loomc_link_index_builder_t* builder = nullptr;
      Check(loomc_link_index_builder_allocate(
          context, nullptr, loomc_allocator_system(), &builder));
      const loomc_link_index_source_options_t library_options = {
          .role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
      };
      if (bytecode) {
        auto* bytes = Serialize(native);
        Check(loomc_link_index_builder_add_source(builder, bytes,
                                                  &library_options, nullptr));
        loomc_source_release(bytes);
      } else {
        Check(loomc_link_index_builder_add_module(builder, native,
                                                  &library_options, nullptr));
      }
      loomc_module_release(native);
      loomc_link_index_t* index = nullptr;
      loomc_result_t* result = nullptr;
      Check(loomc_link_index_builder_finish(builder, &index, &result));
      Done(result);
      loomc_link_index_builder_free(builder);
      loomc_linker_t* linker = nullptr;
      Check(loomc_linker_create(context, nullptr, loomc_allocator_system(),
                                &linker));
      std::vector<std::vector<uint8_t>> programs;
      for (int repetition = 0; repetition < 2; ++repetition) {
        for (const auto& c : cases) {
          auto* input_source = Source(c.filename);
          auto* input = Parse(context, workspace, input_source);
          loomc_source_release(input_source);
          const loomc_link_module_provider_t provider = {
              .module = input,
              .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
          };
          const auto symbol = loomc_make_cstring_view("entry");
          const loomc_link_options_t options = {
              .link_index = index,
              .module_name = loomc_make_cstring_view("control"),
              .mode = LOOMC_LINK_MODE_LINK,
              .root_symbols = &symbol,
              .root_symbol_count = 1,
              .module_providers = &provider,
              .module_provider_count = 1,
          };
          loomc_module_t* linked = nullptr;
          Check(
              loomc_link_module(linker, workspace, &options, &linked, &result));
          Done(result);
          loomc_module_release(input);
          auto* bytes = Serialize(linked);
          const auto contents = loomc_source_contents(bytes);
          programs.emplace_back(contents.data,
                                contents.data + contents.data_length);
          loomc_source_release(bytes);
          loomc_module_release(linked);
        }
      }
      loomc_linker_release(linker);
      loomc_link_index_release(index);
      loomc_workspace_release(workspace);
      loomc_context_release(context);
      for (size_t i = 0; i < programs.size(); ++i) {
        const int value = CompileAndExecute(programs[i]);
        EXPECT_EQ(value, cases[i % 4].expected)
            << "bytecode=" << bytecode << " reversed=" << reversed
            << " repetition=" << i / 4 << " caller=" << cases[i % 4].filename;
      }
    }
  }
}

}  // namespace
