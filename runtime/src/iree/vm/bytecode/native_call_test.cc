// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/disassembler.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/native_call_testdata.h"
#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/sync.h"

namespace {

struct Counters {
  // Successfully published native operations.
  int accepted = 0;
  // Native entry attempts.
  int entered = 0;
  // Released native result backing allocations.
  int released = 0;
};

struct Context {
  // Borrowed core type descriptors.
  iree_vm_ref_types_t types = {};
  // Invocation-observation counters owned by the fixture.
  Counters* counters = nullptr;
};

void ReleaseOutput(void* user_data, iree_byte_span_t storage) {
  ++static_cast<Counters*>(user_data)->released;
  iree_allocator_free(iree_allocator_system(), storage.data);
}

iree_status_t Replace(const void* user_data,
                      const iree_vm_native_call_t* call) {
  const auto& context = *static_cast<const Context*>(user_data);
  ++context.counters->entered;
  iree_vm_buffer_t* input = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_ptr_from_ref_borrowed(
      &context.types, call->refs[call->record[1]], &input));
  if (!input) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null input");
  }
  iree_const_byte_span_t input_bytes;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(input, 0, 8, &input_bytes));
  uint64_t sum = 0;
  memcpy(&sum, input_bytes.data, sizeof(sum));
  sum += iree_vm_native_call_load_value(call, call->record[0]);
  auto* scratch = reinterpret_cast<uint64_t*>(call->scratch);
  for (uint32_t i = 0; i < call->record[4]; ++i) {
    scratch[i] =
        iree_vm_native_call_load_value(call, call->record[call->record[5] + i]);
    sum += scratch[i];
  }
  uint8_t* data = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(), 8,
                                             reinterpret_cast<void**>(&data)));
  memcpy(data, &sum, 8);
  iree_vm_buffer_t* output = nullptr;
  iree_status_t status = iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_byte_span(data, 8),
      {ReleaseOutput, context.counters}, iree_allocator_system(), &output);
  if (iree_status_is_ok(status)) {
    ++context.counters->accepted;
    // The input may be the last owner in the same slot. All native input use
    // is finished before its owner is replaced by the already-owned output.
    iree_vm_ref_reset(&call->refs[call->record[3]]);
    call->refs[call->record[3]] =
        iree_vm_buffer_ref_from_ptr_move(&context.types, &output);
    iree_vm_native_call_store_value(call, call->record[2], sum);
  } else {
    iree_allocator_free(iree_allocator_system(), data);
  }
  return status;
}

iree_status_t Fail(const void* user_data, const iree_vm_native_call_t* call) {
  ++static_cast<const Context*>(user_data)->counters->entered;
  return iree_make_status(IREE_STATUS_ABORTED, "native failure");
}

const iree_vm_native_call_field_t kFields[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_NATIVE_CALL_FIELD_REF, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_NATIVE_CALL_FIELD_REF, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
};
const iree_vm_native_call_layout_t kLayouts[] = {
    {0, 1, 1, 0}, {1, 6, 5, 1}, {6, 0, 0, 0}};
const iree_vm_native_call_schema_t kReplace = {kLayouts, kFields, 3, 6, 1};
const iree_vm_native_call_schema_t kEmpty = {kLayouts, kFields, 3, 6, 2};

class BytecodeNativeCallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &context_.types));
    context_.counters = &counters_;
    exports_[0] = {
        .name = IREE_SV("fail"),
        .schema = &kEmpty,
        .entry = Fail,
        .user_data = &context_,
    };
    exports_[1] = {
        .name = IREE_SV("replace"),
        .schema = &kReplace,
        .scratch_sizes = scratch_sizes_,
        .entry = Replace,
        .user_data = &context_,
    };
    native_ = {exports_.data(), exports_.size(), nullptr, 0};
    const auto* source = iree_vm_bytecode_native_call_testdata_create();
    IREE_ASSERT_OK(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(reinterpret_cast<const char*>(source[0].data),
                              source[0].size),
        iree_allocator_system(), &fixture_));
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("caller"),
        {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_bytecode_module_fixture_deinitialize(&fixture_);
    iree_vm_environment_free(environment_);
  }

  iree_status_t Link() {
    iree::vm::testing::TableModuleDefinition definition(IREE_SV("native.test"));
    definition.descriptor.ref_types = {&context_.types.buffer, 1};
    definition.descriptor.native = &native_;
    IREE_RETURN_IF_ERROR(provider_.Publish(definition));
    iree_vm_module_t* libraries[] = {provider_.module()};
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {module_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    return iree_vm_process_create(program_, invocation_,
                                  iree_vm_variant_span_empty(),
                                  iree_allocator_system(), &process_);
  }

  iree_status_t Invoke(iree_string_view_t name, iree_vm_variant_t* results) {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("caller"), name, &function));
    const uint64_t payload = 100;
    iree_vm_buffer_t* input = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(&payload, sizeof(payload)), 0,
        iree_allocator_system(), &input));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i64(7),
        iree_vm_buffer_variant_from_ptr_move(&context_.types, &input)};
    return iree_vm_invoke(invocation_, function,
                          iree_vm_variant_span_from_array(arguments),
                          {results, 2});
  }

  // Observations and immutable provider declarations outlive linked objects.
  Counters counters_;
  // Resolved canonical ref descriptors and test observations.
  Context context_;
  // Scratch contributions, immutable throughout linked lifetime.
  uint32_t scratch_sizes_[3] = {8, 0, 0};
  // Native export rows, finalized before module publication.
  std::array<iree_vm_native_export_t, 2> exports_;
  // Native declaration directory.
  iree_vm_native_module_t native_ = {};
  // Provider module anchoring the native declarations.
  iree::vm::testing::TableModule provider_;
  // Ref-type environment used during caller creation.
  iree_vm_environment_t* environment_ = nullptr;
  // Owned bytecode storage.
  iree_vm_bytecode_module_fixture_t fixture_ = {};
  // Owned immutable bytecode module.
  iree_vm_module_t* module_ = nullptr;
  // Owned linked program.
  iree_vm_program_t* program_ = nullptr;
  // Owned process instance.
  iree_vm_process_t* process_ = nullptr;
  // Fixed invocation arena including the linked native scratch tail.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> storage_ = {};
  // Invocation object placed in the arena.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(BytecodeNativeCallTest, BorrowsSpillsAndPublishesAliasedResults) {
  IREE_ASSERT_OK(Link());
  provider_.ReleaseOwner();
  EXPECT_EQ(provider_.destruction_count(), 0);
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("run"), results));
  int64_t sum = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[0], &sum));
  EXPECT_EQ(sum, 125);
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(&context_.types,
                                                          results[1], &buffer));
  uint64_t contents = 0;
  memcpy(&contents, iree_vm_buffer_const_data(buffer), sizeof(contents));
  EXPECT_EQ(contents, 125u);
  EXPECT_EQ(counters_.accepted, 1);
  EXPECT_EQ(counters_.released, 0);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_EQ(counters_.released, 1);
}

TEST_F(BytecodeNativeCallTest, FailureAfterAcceptanceUnwindsOwners) {
  IREE_ASSERT_OK(Link());
  iree_vm_variant_t results[2] = {iree_vm_variant_from_i64(999), {}};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        Invoke(IREE_SV("failure"), results));
  int64_t sentinel = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[0], &sentinel));
  EXPECT_EQ(sentinel, 999);
  EXPECT_EQ(counters_.accepted, 1);
  EXPECT_EQ(counters_.entered, 2);
  EXPECT_EQ(counters_.released, 1);
}

TEST_F(BytecodeNativeCallTest, RejectsVersionBeforeEnteringProvider) {
  exports_[1].version = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, Link());
  EXPECT_EQ(counters_.entered, 0);
}

TEST_F(BytecodeNativeCallTest, ScratchExhaustionPrecedesEntry) {
  scratch_sizes_[1] = 65536;
  IREE_ASSERT_OK(Link());
  iree_vm_variant_t results[2] = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Invoke(IREE_SV("run"), results));
  EXPECT_EQ(counters_.entered, 0);
}

TEST_F(BytecodeNativeCallTest, RejectsMissingNativeExport) {
  exports_[1].name = IREE_SV("renamed");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Link());
  EXPECT_EQ(counters_.entered, 0);
}

TEST_F(BytecodeNativeCallTest, LinkingRejectsMalformedProviderRecord) {
  // External image bytes can be structurally valid without satisfying a
  // provider's record schema. Their selectors are admitted by program linking.
  iree_vm_module_release(module_);
  module_ = nullptr;
  iree_vm_bytecode_module_plan_t plan;
  IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
      iree_const_cast_byte_span(fixture_.contents), &plan));
  auto* words = const_cast<uint32_t*>(plan.layout.native_calls.words);
  words[5] = 0;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment_, IREE_SV("caller"),
      {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
      iree_allocator_system(), &module_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Link());
  EXPECT_EQ(counters_.entered, 0);
}

TEST_F(BytecodeNativeCallTest, CallsiteCannotBorrowAnotherFunctionsExtent) {
  iree_vm_module_release(module_);
  module_ = nullptr;
  iree_vm_bytecode_module_plan_t plan;
  IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
      iree_const_cast_byte_span(fixture_.contents), &plan));
  auto* rows = const_cast<iree_vm_bytecode_v0_native_callsite_row_t*>(
      plan.layout.native_calls.rows);
  rows[0].function_ordinal_u16 = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_bytecode_module_create(
          environment_, IREE_SV("caller"),
          {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
          iree_allocator_system(), &module_));
}

TEST_F(BytecodeNativeCallTest, NativeSectionRoundtripsWithoutLosingWords) {
  iree_string_builder_t text;
  iree_string_builder_initialize(iree_allocator_system(), &text);
  const iree_vm_bytecode_disassembler_write_callback_t callback = {
      [](void* user_data, iree_string_view_t fragment) {
        return iree_string_builder_append_string(
            static_cast<iree_string_builder_t*>(user_data), fragment);
      },
      &text};
  IREE_ASSERT_OK(iree_vm_bytecode_disassemble_module(
      iree_const_cast_byte_span(fixture_.contents), callback,
      iree_allocator_system()));
  iree_vm_bytecode_module_fixture_t roundtrip = {};
  IREE_ASSERT_OK(iree_vm_bytecode_module_fixture_initialize(
      iree_string_builder_view(&text), iree_allocator_system(), &roundtrip));
  ASSERT_EQ(roundtrip.contents.data_length, fixture_.contents.data_length);
  EXPECT_EQ(memcmp(roundtrip.contents.data, fixture_.contents.data,
                   fixture_.contents.data_length),
            0);
  iree_vm_bytecode_module_fixture_deinitialize(&roundtrip);
  iree_string_builder_deinitialize(&text);
}

}  // namespace
