// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/sync.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

::testing::AssertionResult Succeeded(const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  auto failure = ::testing::AssertionFailure();
  if (!result) {
    return failure << "missing compiler result";
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const auto message = loomc_result_diagnostic_at(result, i)->message;
    failure << std::string(message.data, message.size);
  }
  return failure;
}

// The native export handles any static cardinality through the same entry.
const iree_vm_native_call_field_t kInputFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_VALUE, 0, nullptr},
    {IREE_VM_NATIVE_CALL_FIELD_REF, 1, nullptr}};
const iree_vm_native_call_schema_t kInput = {2, 2, kInputFields, 0};
const iree_vm_native_call_field_t kOutputFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_RESULT_VALUE, 0, nullptr},
    {IREE_VM_NATIVE_CALL_FIELD_RESULT_REF, 1, nullptr}};
const iree_vm_native_call_schema_t kOutput = {2, 2, kOutputFields, 0};
const iree_vm_native_call_field_t kRootFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0, &kInput},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 2, &kOutput}};
const iree_vm_native_call_schema_t kRoot = {4, 2, kRootFields, 0};

struct Observations {
  // Resolved core reference types borrowed by the provider.
  iree_vm_ref_types_t types = {};
  // Number of completed native entries.
  int calls = 0;
  // Number of scalar selectors naming frame storage.
  int stored_values = 0;
  // Largest selected reference slot across inputs and outputs.
  uint32_t maximum_ref = 0;
  // Number of backing owners destroyed.
  int releases = 0;
};

iree_status_t Transform(const void* user_data,
                        const iree_vm_native_call_t* call) {
  auto& observations =
      *static_cast<Observations*>(const_cast<void*>(user_data));
  const uint32_t* record = call->record;
  if (!record[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty input range");
  }
  uint64_t sum = 0;
  for (uint32_t i = 0; i < record[0]; ++i) {
    const uint32_t* input = record + record[1] + i * 2;
    observations.stored_values +=
        !!(input[0] & IREE_VM_NATIVE_CALL_LOCAL_VALUE);
    observations.maximum_ref = iree_max(observations.maximum_ref, input[1]);
    iree_vm_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_ptr_from_ref_borrowed(
        &observations.types, call->refs[input[1]], &buffer));
    if (!buffer || iree_vm_buffer_length(buffer) != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one byte");
    }
    sum += (i + 1) * iree_vm_native_call_load_value(call, input[0]) +
           static_cast<const uint8_t*>(iree_vm_buffer_const_data(buffer))[0];
  }
  // All input reads precede publication. One temporary owner anchors aliased
  // outputs while the first publication may replace the last input owner.
  iree_vm_ref_t owner = iree_vm_ref_retain(call->refs[record[record[1] + 1]]);
  for (uint32_t i = 0; i < record[2]; ++i) {
    const uint32_t* output = record + record[3] + i * 2;
    observations.stored_values +=
        !!(output[0] & IREE_VM_NATIVE_CALL_LOCAL_VALUE);
    observations.maximum_ref = iree_max(observations.maximum_ref, output[1]);
    iree_vm_native_call_store_value(call, output[0], sum + i);
    iree_vm_ref_reset(&call->refs[output[1]]);
    call->refs[output[1]] = iree_vm_ref_retain(owner);
  }
  iree_vm_ref_reset(&owner);
  ++observations.calls;
  return iree_ok_status();
}

// Produces a real Low caller whose wide intermediate must use both value and
// reference storage. All cardinalities resolve to the one provider above.
std::string Source(uint32_t count) {
  const std::string value = "reg<vm.value : i64>";
  const std::string ref = "reg<vm.ref : buffer>";
  auto signature = [&](uint32_t count, bool named) {
    std::string result;
    for (uint32_t i = 0; i < count; ++i) {
      if (i) {
        result += ", ";
      }
      result += (named ? "%v" + std::to_string(i) + ": " : "") + value;
      result += ", " + (named ? "%r" + std::to_string(i) + ": " : "") + ref;
    }
    return result;
  };
  auto declaration = [&](const char* name, uint32_t inputs, uint32_t outputs) {
    auto offsets = [](uint32_t start, uint32_t count) {
      std::string result;
      for (uint32_t i = 0; i < count; ++i) {
        if (i) {
          result += ", ";
        }
        result += std::to_string(start + i);
      }
      return result;
    };
    std::string words = std::to_string(inputs) + ", 4, " +
                        std::to_string(outputs) + ", " +
                        std::to_string(4 + inputs * 2);
    for (uint32_t i = 0; i < (inputs + outputs) * 2; ++i) {
      words += ", 0";
    }
    return std::string(
               "low.func.decl import(native, \"transform\", module = "
               "\"host\") ") +
           "target<vm.core>(@vm) abi(vm_native) abi_layout({arguments = [" +
           offsets(4, inputs * 2) + "], record = [" + words + "], results = [" +
           offsets(4 + inputs * 2, outputs * 2) + "], version = 0}) @" + name +
           "(" + signature(inputs, true) + ") -> (" +
           signature(outputs, false) + ")\n";
  };
  std::string source = "vm.target<core> @vm\n" +
                       declaration("expand", 1, count) +
                       declaration("reduce", count, 1);
  source += "low.func.def public target<vm.core>(@vm) @run(" +
            signature(1, true) + ") -> (" + signature(1, false) + ") {\n";
  std::string outputs, inputs;
  for (uint32_t i = 0; i < count; ++i) {
    if (i) {
      outputs += ", ";
      inputs += ", ";
    }
    outputs += "%a" + std::to_string(i) + ", %b" + std::to_string(i);
    inputs += "%a" + std::to_string(count - i - 1) + ", %b" +
              std::to_string(count - i - 1);
  }
  source += outputs + " = low.func.call @expand(%v0, %r0) : (" +
            signature(1, false) + ") -> (" + signature(count, false) + ")\n";
  source += "%sum, %owner = low.func.call @reduce(" + inputs + ") : (" +
            signature(count, false) + ") -> (" + signature(1, false) + ")\n";
  return source + "low.return %sum, %owner : " + signature(1, false) + "\n}\n";
}

class VmNativeCallTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &observations_.types));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    export_ = {IREE_SV("transform"), 0, &kRoot, nullptr, Transform,
               &observations_};
    native_ = {&export_, 1, nullptr, 0};
    iree::vm::testing::TableModuleDefinition definition(IREE_SV("host"));
    definition.descriptor.native = &native_;
    IREE_ASSERT_OK(provider_.Publish(definition));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    provider_.ReleaseOwner();
    iree_vm_environment_free(environment_);
  }

  void Compile(const std::string& text) {
    const auto allocator = loomc_allocator_system();
    loomc_target_environment_t* raw_environment = nullptr;
    LOOMC_ASSERT_OK(
        loomc_target_environment_create_vm(allocator, &raw_environment));
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>
        environment(raw_environment);
    loomc_context_target_options_t target_options = {};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size = sizeof(target_options);
    target_options.target_environment = environment.get();
    loomc_context_options_t context_options = {};
    context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size = sizeof(context_options);
    context_options.next = &target_options;
    loomc_context_t* raw_context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(&context_options, allocator, &raw_context));
    HandlePtr<loomc_context_t, loomc_context_release> context(raw_context);
    loomc_workspace_t* raw_workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, allocator, &raw_workspace));
    HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace(
        raw_workspace);
    loomc_source_options_t source_options = {};
    source_options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    source_options.structure_size = sizeof(source_options);
    source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    source_options.identifier = loomc_make_cstring_view("native.loom");
    source_options.contents = loomc_make_byte_span(text.data(), text.size());
    source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t* raw_source = nullptr;
    LOOMC_ASSERT_OK(
        loomc_source_create(&source_options, allocator, &raw_source));
    HandlePtr<loomc_source_t, loomc_source_release> source(raw_source);
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), source.get(), nullptr, allocator,
        &raw_module, &raw_result));
    HandlePtr<loomc_module_t, loomc_module_release> module(raw_module);
    HandlePtr<loomc_result_t, loomc_result_release> result(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    loomc_pass_program_t* raw_pass_program = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
        context.get(), nullptr, allocator, &raw_pass_program, &raw_result));
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release> pass_program(
        raw_pass_program);
    result.reset(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    loomc_compiler_t* raw_compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context.get(), nullptr, allocator,
                                          &raw_compiler));
    HandlePtr<loomc_compiler_t, loomc_compiler_release> compiler(raw_compiler);
    LOOMC_ASSERT_OK(loomc_compile_module(compiler.get(), workspace.get(),
                                         pass_program.get(), module.get(),
                                         nullptr, allocator, &raw_result));
    result.reset(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    LOOMC_ASSERT_OK(loomc_emit_module(environment.get(), workspace.get(),
                                      module.get(), nullptr, allocator,
                                      &raw_result));
    result.reset(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    ASSERT_EQ(loomc_result_artifact_count(result.get()), 1);
    loomc_byte_span_t bytes;
    LOOMC_ASSERT_OK(loomc_byte_sequence_clone(
        loomc_result_artifact_at(result.get(), 0)->contents, allocator,
        &bytes));
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("caller"),
        {iree_make_const_byte_span(bytes.data, bytes.data_length),
         iree_allocator_system()},
        iree_allocator_system(), &module_));
    iree_vm_module_t* libraries[] = {provider_.module()};
    IREE_ASSERT_OK(iree_vm_program_create(
        {module_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  // Provider observations, live until every output owner is released.
  Observations observations_;
  // Published synchronous entry and its immutable directory.
  iree_vm_native_export_t export_ = {};
  // Native directory borrowed by the provider module.
  iree_vm_native_module_t native_ = {};
  // Linked provider owner.
  iree::vm::testing::TableModule provider_;
  // Canonical runtime type environment.
  iree_vm_environment_t* environment_ = nullptr;
  // Owned compiled bytecode module.
  iree_vm_module_t* module_ = nullptr;
  // Linked program retaining both modules.
  iree_vm_program_t* program_ = nullptr;
  // Mutable process instance.
  iree_vm_process_t* process_ = nullptr;
  // Invocation arena sized for the wide test's real spilled owners.
  alignas(iree_max_align_t) std::array<uint8_t, 65536> storage_ = {};
  // Invocation placed in |storage_|.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_P(VmNativeCallTest, CompilesLinksAndExecutesBorrowedRegistersAndSpills) {
  const uint32_t count = GetParam();
  ASSERT_NO_FATAL_FAILURE(Compile(Source(count)));
  provider_.ReleaseOwner();
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process_, IREE_SV("caller"),
                                                 IREE_SV("run"), &function));
  uint8_t contents = 11;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_byte_span(&contents, 1),
      {[](void* context, iree_byte_span_t) {
         ++static_cast<Observations*>(context)->releases;
       },
       &observations_},
      iree_allocator_system(), &buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64(5),
      iree_vm_buffer_variant_from_ptr_move(&observations_.types, &buffer)};
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, function,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  int64_t expected = 0;
  for (uint32_t i = 0; i < count; ++i) {
    expected += (i + 1) * (16 + count - i - 1) + 11;
  }
  int64_t actual = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[0], &actual));
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(observations_.calls, 2);
  EXPECT_EQ(observations_.releases, 0);
  if (count > 256) {
    EXPECT_GT(observations_.stored_values, 0);
    EXPECT_GE(observations_.maximum_ref, 256);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_EQ(observations_.releases, 1);
}

INSTANTIATE_TEST_SUITE_P(Cardinality, VmNativeCallTest,
                         ::testing::Values(1u, 17u, 257u, 300u));

}  // namespace
