// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/sync.h"

namespace {

extern const iree_vm_ref_type_table_t kBorrowedTypes;
const iree_vm_ref_type_descriptor_t kByteType = {nullptr, &kBorrowedTypes,
                                                 IREE_SV("byte")};
const iree_vm_ref_type_t kBorrowedRefTypes[] = {&kByteType};
const iree_vm_ref_type_table_t kBorrowedTypes = {
    sizeof(kBorrowedTypes),
    IREE_VM_REF_TYPE_TABLE_FLAG_BORROWED_ONLY,
    IREE_SV("borrowed"),
    {kBorrowedRefTypes, 1}};

// Exercises public invocation admission, module verification, and execution.
// Host descriptors borrow actual arrays; no native adapter substitutes for the
// bytecode accesses under test.
class IncomingSpanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
        environment_, &kBorrowedTypes));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_bytecode_module_fixture_deinitialize(&fixture_);
    iree_vm_environment_free(environment_);
  }

  iree_status_t Load(const std::string& arguments, const std::string& body,
                     const std::string& child = "") {
    const std::string frame =
        " frame(local_bytes = 0, value_registers = 32, "
        "ref_registers = 4, function_registers = 0, local_refs = 0, "
        "local_functions = 0, call_values = 32) {\n^bb0:\ncontrol.block\n";
    std::string source = R"(vm.module core(0.0) {
section strings alignment(8) {
  string @namespace = "vm"
  string @buffer = "buffer"
  string @borrowed = "borrowed"
  string @byte = "byte"
  string @run = "run"
}
section ref_types alignment(8) {
  namespace @borrowed_family name(@borrowed) { ref_type @byte_type name(@byte) }
  namespace @family name(@namespace) { ref_type @buffer_type name(@buffer) }
}
section signatures alignment(8) {
  signature @signature0 ()" +
                         arguments + R"() -> (i64)
}
section callable_types alignment(8) {
  callable @callable0 flags([]) depth(0) : @signature0
}
section exports alignment(8) {
  export @export0 name(@run) : @callable0 = @function0
}
section functions alignment(8) {
  func @function0 : @callable0)" +
                         (child.empty() ? " flags([])" : " flags([has_call])") +
                         frame + body + "\ncontrol.return\n}\n";
    if (!child.empty()) {
      source += "func @function1 : @callable0 flags([])" + frame + child +
                "\ncontrol.return\n}\n";
    }
    source += "}\n}\n";
    return LoadSource(source);
  }

  iree_status_t LoadBufferCall(const std::string& body,
                               const std::string& child,
                               const std::string& access = "read") {
    return LoadSource(R"(vm.module core(0.0) {
section strings alignment(8) {
  string @namespace = "vm"
  string @buffer = "buffer"
  string @run = "run"
}
section ref_types alignment(8) {
  namespace @family name(@namespace) { ref_type @buffer_type name(@buffer) }
}
section signatures alignment(8) {
  signature @parent (ref(@buffer_type)) -> (i64)
  signature @child (span<i64, )" +
                      access + R"(>) -> (i64)
}
section callable_types alignment(8) {
  callable @parent flags([]) depth(0) : @parent
  callable @child flags([]) depth(0) : @child
}
section exports alignment(8) {
  export @export0 name(@run) : @parent = @function0
}
section functions alignment(8) {
  func @function0 : @parent flags([has_call]) frame(local_bytes = 0,
      value_registers = 4, ref_registers = 2, function_registers = 0,
      local_refs = 0, local_functions = 0, call_values = 2) {
  ^bb0:
    control.block
)" + body + R"(
    control.return
  }
  func @function1 : @child flags([]) frame(local_bytes = 0,
      value_registers = 4, ref_registers = 1, function_registers = 0,
      local_refs = 0, local_functions = 0, call_values = 0) {
  ^bb0:
    control.block
)" + child + R"(
    control.return
  }
}
})");
  }

  iree_status_t LoadSource(const std::string& source) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(source.data(), source.size()),
        iree_allocator_system(), &fixture_));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        environment_, IREE_SV("arrays"),
        {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
        iree_allocator_system(), &module_));
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {module_, {}}, iree_allocator_system(), &program_));
    IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_RETURN_IF_ERROR(iree_vm_process_create(
        program_, invocation_, {}, iree_allocator_system(), &process_));
    return iree_vm_process_lookup_function(process_, IREE_SV("arrays"),
                                           IREE_SV("run"), &function_);
  }

  iree_status_t Invoke(iree_vm_variant_span_t arguments, int64_t* result) {
    iree_vm_variant_t value = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(invocation_, function_, arguments, {&value, 1}));
    return iree_vm_i64_from_variant(value, result);
  }

  // Registered built-in buffer identity used by reference span cases.
  iree_vm_ref_types_t types_ = {};
  // Type registry outliving the linked bytecode.
  iree_vm_environment_t* environment_ = nullptr;
  // Assembled immutable bytes borrowed by module_.
  iree_vm_bytecode_module_fixture_t fixture_ = {};
  // Bytecode execution implementation.
  iree_vm_module_t* module_ = nullptr;
  // Linked callable and public ABI plans.
  iree_vm_program_t* program_ = nullptr;
  // Process outliving each synchronous invocation.
  iree_vm_process_t* process_ = nullptr;
  // Storage enclosing every frame and private incoming region descriptor.
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage_{};
  // Invocation initialized within storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Exported bytecode entry.
  iree_vm_function_t function_ = {};
};

TEST_F(IncomingSpanTest, ReadsAndSlicesThroughBytecodeCall) {
  IREE_ASSERT_OK(Load("span<i64, read>", R"(
    %v2 = constant.s16 8
    %v0 = integer.add.i64 %v0, %v2
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
  )",
                      "%v0.x1 = span.load.i64 %v0"));
  const uint64_t values[] = {11, 37, 59};
  const iree_vm_typed_span_t span = {values, 3, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ, nullptr};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_ASSERT_OK(Invoke({&argument, 1}, &result));
  EXPECT_EQ(result, 37);
}

TEST_F(IncomingSpanTest, SanitizesOverflowArgumentAddresses) {
  std::string signature;
  for (int i = 0; i < 16; ++i) {
    signature += "i64, ";
  }
  signature += "span<i64, read>";
  IREE_ASSERT_OK(Load(signature, "%v0 = value.abi.argument.load 0"));
  const uint64_t values[] = {37};
  const iree_vm_typed_span_t span = {values, 1, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ, nullptr};
  std::array<iree_vm_variant_t, 17> arguments;
  arguments.fill(iree_vm_variant_from_i64(0));
  arguments.back() = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_ASSERT_OK(Invoke({arguments.data(), arguments.size()}, &result));
  EXPECT_EQ(result, INT64_C(9) << 48);
  EXPECT_NE(static_cast<uint64_t>(result), reinterpret_cast<uintptr_t>(values));
}

TEST_F(IncomingSpanTest, ChecksRegionDomainAndCompleteRange) {
  IREE_ASSERT_OK(Load("span<i64, read>, i64", R"(
    %v0 = integer.add.i64 %v0, %v2
    %v0.x1 = span.load.i64 %v0
  )"));
  const uint64_t values[] = {37};
  const iree_vm_typed_span_t span = {values, 1, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ, nullptr};
  for (const int64_t offset :
       {INT64_C(1), INT64_C(8), INT64_C(-1), INT64_C(1) << 48}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_span_borrowed(&span),
                                     iree_vm_variant_from_i64(offset)};
    int64_t result = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        Invoke(iree_vm_variant_span_from_array(arguments), &result));
  }
}

TEST_F(IncomingSpanTest, RejectsScalarStorageAsReferenceCells) {
  IREE_ASSERT_OK(Load("span<i64, read>", R"(
    %r0 = span.ref.load %v0
    %v0 = constant.zero
  )"));
  const uint64_t values[2] = {};
  const iree_vm_typed_span_t span = {values, 2, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ, nullptr};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke({&argument, 1}, &result));
}

TEST_F(IncomingSpanTest, RejectsWritesThroughReadOnlySignature) {
  IREE_ASSERT_OK(Load("span<i64, read>", R"(
    %v2 = constant.s16 42
    span.store.i64 %v2.x1, %v0
    %v0 = constant.zero
  )"));
  uint64_t value = 37;
  const iree_vm_typed_span_t span = {&value, 1, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ_WRITE, nullptr};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Invoke({&argument, 1}, &result));
  EXPECT_EQ(value, 37);
}

TEST_F(IncomingSpanTest, ValidatesHostStorageAndPermissions) {
  IREE_ASSERT_OK(Load("span<i64>", "%v0 = constant.zero"));
  uint64_t value = 37;
  iree_vm_typed_span_t span = {&value, 1, IREE_VM_SCALAR_TYPE_I64,
                               IREE_VM_SPAN_ACCESS_READ, nullptr};
  auto invoke = [&]() {
    auto argument = iree_vm_variant_from_span_borrowed(&span);
    int64_t result = 0;
    return Invoke({&argument, 1}, &result);
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, invoke());
  span.access = IREE_VM_SPAN_ACCESS_READ_WRITE;
  span.scalar_type = IREE_VM_SCALAR_TYPE_I32;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, invoke());
  span.scalar_type = IREE_VM_SCALAR_TYPE_I64;
  span.data = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, invoke());
  span.count = 0;
  IREE_ASSERT_OK(invoke());
  span.data = &value;
  span.count = UINT64_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke());
  span.data = reinterpret_cast<const void*>(UINTPTR_MAX - 3);
  span.count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke());
  span.data = &value;
  span.access = IREE_VM_SPAN_ACCESS_READ_WRITE | 4u;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, invoke());
}

TEST_F(IncomingSpanTest, RejectsPermissionEscalationThroughBytecodeCall) {
  IREE_ASSERT_OK(LoadSource(R"(vm.module core(0.0) {
section strings alignment(8) { string @run = "run" }
section signatures alignment(8) {
  signature @mutable (span<i64>) -> (i64)
  signature @read_only (span<i64, read>) -> (i64)
}
section callable_types alignment(8) {
  callable @mutable flags([]) depth(0) : @mutable
  callable @read_only flags([]) depth(0) : @read_only
}
section exports alignment(8) {
  export @export0 name(@run) : @read_only = @function0
}
section functions alignment(8) {
  func @function0 : @read_only flags([has_call]) frame(local_bytes = 0, value_registers = 2, ref_registers = 0, function_registers = 0, local_refs = 0, local_functions = 0, call_values = 2) {
  ^bb0:
    control.block
    control.call @function1 {direct_ref_move_mask = 0x0000}
    control.return
  }
  func @function1 : @mutable flags([]) frame(local_bytes = 0, value_registers = 3, ref_registers = 0, function_registers = 0, local_refs = 0, local_functions = 0, call_values = 0) {
  ^bb0:
    control.block
    %v2 = constant.s16 42
    span.store.i64 %v2.x1, %v0
    %v0 = constant.zero
    control.return
  }
}
})"));
  uint64_t value = 37;
  const iree_vm_typed_span_t span = {&value, 1, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ_WRITE, nullptr};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Invoke({&argument, 1}, &result));
  EXPECT_EQ(value, 37);
}

TEST_F(IncomingSpanTest, RejectsReferenceTypeMismatchBeforeReplacingOwner) {
  IREE_ASSERT_OK(Load("ptr(@byte_type), span<ref(@buffer_type)>",
                      "span.ref.store %r0, %v0\n%v0 = constant.zero"));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(37, 0, iree_allocator_system(), &buffer));
  iree_vm_ref_t cell = iree_vm_buffer_ref_from_ptr_move(&types_, &buffer);
  const iree_vm_typed_span_t span = {&cell, 1, IREE_VM_SCALAR_TYPE_NONE,
                                     IREE_VM_SPAN_ACCESS_READ_WRITE,
                                     types_.buffer};
  uint8_t byte = 42;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_ptr_borrowed(&byte, &kByteType),
      iree_vm_variant_from_span_borrowed(&span)};
  const iree_vm_ref_t original = cell;
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Invoke(iree_vm_variant_span_from_array(arguments), &result));
  EXPECT_EQ(cell.object, original.object);
  EXPECT_EQ(cell.type_and_state, original.type_and_state);
  iree_vm_ref_reset(&cell);
}

TEST_F(IncomingSpanTest, RejectsCapturingBorrowedOnlyObjectInArray) {
  IREE_ASSERT_OK(Load("ptr(@byte_type), span<ptr(@byte_type)>",
                      "span.ref.store %r0, %v0\n%v0 = constant.zero"));
  iree_vm_ref_t cell = iree_vm_ref_null();
  const iree_vm_typed_span_t span = {&cell, 1, IREE_VM_SCALAR_TYPE_NONE,
                                     IREE_VM_SPAN_ACCESS_READ_WRITE,
                                     &kByteType};
  uint8_t byte = 42;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_ptr_borrowed(&byte, &kByteType),
      iree_vm_variant_from_span_borrowed(&span)};
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      Invoke(iree_vm_variant_span_from_array(arguments), &result));
  EXPECT_TRUE(iree_vm_ref_is_null(cell));
}

TEST_F(IncomingSpanTest, WritableReferenceLoadPinsReplacedOwner) {
  IREE_ASSERT_OK(Load("span<ref(@buffer_type)>", R"(
    %r0 = span.ref.load %v0
    %r1 = ref.null
    span.ref.store %r1, %v0
    %v0 = buffer.length %r0
  )"));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(37, 0, iree_allocator_system(), &buffer));
  iree_vm_ref_t cell = iree_vm_buffer_ref_from_ptr_move(&types_, &buffer);
  EXPECT_EQ(buffer, nullptr);
  const iree_vm_typed_span_t span = {&cell, 1, IREE_VM_SCALAR_TYPE_NONE,
                                     IREE_VM_SPAN_ACCESS_READ_WRITE,
                                     types_.buffer};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_EXPECT_OK(Invoke({&argument, 1}, &result));
  EXPECT_EQ(result, 37);
  EXPECT_TRUE(iree_vm_ref_is_null(cell));
  iree_vm_ref_reset(&cell);
}

TEST_F(IncomingSpanTest, PinsBufferAcrossRegisterReuse) {
  IREE_ASSERT_OK(LoadBufferCall(R"(
    %v0 = span.from.buffer %r0
    ref.discard %r0
    %v2 = constant.s16 8
    %v0 = integer.add.i64 %v0, %v2
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
  )",
                                "%v0.x1 = span.load.i64 %v0"));
  uint64_t values[] = {11, 37};
  unsigned releases = 0;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(values, sizeof(values)),
                          {[](void* user_data, iree_byte_span_t) {
                             ++*static_cast<unsigned*>(user_data);
                           },
                           &releases},
                          iree_allocator_system(), &buffer));
  auto argument = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  int64_t result = 0;
  IREE_ASSERT_OK(Invoke({&argument, 1}, &result));
  EXPECT_EQ(result, 37);
  EXPECT_EQ(releases, 1u);
}

TEST_F(IncomingSpanTest, ReusesBufferPinsAfterChildCompletion) {
  IREE_ASSERT_OK(LoadBufferCall(R"(
    %v0 = span.from.buffer %r0
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
    %v0 = span.from.buffer %r0
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
  )",
                                "%v0.x1 = span.load.i64 %v0"));
  const uint64_t value = 37;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(&value, sizeof(value)), 0,
                           iree_allocator_system(), &buffer));
  auto argument = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  int64_t result = 0;
  IREE_ASSERT_OK(Invoke({&argument, 1}, &result));
  EXPECT_EQ(result, 37);
}

TEST_F(IncomingSpanTest, RejectsExpiredBufferCoordinate) {
  IREE_ASSERT_OK(LoadBufferCall(R"(
    %v0 = span.from.buffer %r0
    ref.discard %r0
    %v2 = value.copy %v0
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
    %v0.x1 = span.load.i64 %v2
  )",
                                "%v0.x1 = span.load.i64 %v0"));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, iree_allocator_system(), &buffer));
  auto argument = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Invoke({&argument, 1}, &result));
}

TEST_F(IncomingSpanTest, RejectsReferenceAccessToBufferBytes) {
  IREE_ASSERT_OK(LoadBufferCall(R"(
    %v0 = span.from.buffer %r0
    %r1 = span.ref.load %v0
    control.call @function1 {direct_ref_move_mask = 0x0000}
  )",
                                "%v0 = constant.zero"));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(16, 0, iree_allocator_system(), &buffer));
  auto argument = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke({&argument, 1}, &result));
}

TEST_F(IncomingSpanTest, RejectsWriteThroughReadOnlyBuffer) {
  IREE_ASSERT_OK(LoadBufferCall(R"(
    %v0 = span.from.buffer %r0
    %v1 = constant.s16 1
    control.call @function1 {direct_ref_move_mask = 0x0000}
  )",
                                R"(
    %v2 = constant.s16 99
    span.store.i64 %v2.x1, %v0
    %v0 = constant.zero
  )",
                                "write"));
  uint64_t value = 37;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                     iree_make_byte_span(&value, sizeof(value)),
                                     iree_vm_buffer_release_callback_null(),
                                     iree_allocator_system(), &buffer));
  auto argument = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Invoke({&argument, 1}, &result));
  EXPECT_EQ(value, 37u);
}

class IncomingSpanFormatTest : public IncomingSpanTest,
                               public ::testing::WithParamInterface<unsigned> {
};

TEST_P(IncomingSpanFormatTest, TransfersUnalignedScalarLanes) {
  const unsigned width = 8u << (GetParam() / 4);
  const unsigned lanes = 1u << (GetParam() % 4);
  const std::string suffix = ".i" + std::to_string(width);
  const std::string registers = "%v4.x" + std::to_string(lanes);
  IREE_ASSERT_OK(Load("span<i8, read>, span<i8, write>",
                      registers + " = span.load" + suffix + " %v0\n" +
                          "span.store" + suffix + " " + registers + ", %v2\n" +
                          "%v0 = constant.zero"));
  std::array<uint8_t, 66> input;
  std::array<uint8_t, 66> output;
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = i * 7 + 11;
  }
  output.fill(0xCD);
  const size_t byte_length = width / 8 * lanes;
  const iree_vm_typed_span_t spans[] = {
      {input.data() + 1, byte_length, IREE_VM_SCALAR_TYPE_I8,
       IREE_VM_SPAN_ACCESS_READ, nullptr},
      {output.data() + 1, byte_length, IREE_VM_SCALAR_TYPE_I8,
       IREE_VM_SPAN_ACCESS_WRITE, nullptr}};
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_span_borrowed(&spans[0]),
      iree_vm_variant_from_span_borrowed(&spans[1])};
  int64_t result = 0;
  IREE_ASSERT_OK(Invoke(iree_vm_variant_span_from_array(arguments), &result));
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], i > 0 && i <= byte_length ? input[i] : 0xCD);
  }
}

INSTANTIATE_TEST_SUITE_P(AllFormats, IncomingSpanFormatTest,
                         ::testing::Range(0u, 16u));

TEST_F(IncomingSpanTest, RejectsYieldingDescendantBeforeStartingIt) {
  IREE_ASSERT_OK(LoadSource(R"(vm.module core(0.0) {
section strings alignment(8) { string @run = "run" }
section signatures alignment(8) {
  signature @empty () -> (i64)
  signature @borrow (span<i64, read>) -> (i64)
}
section callable_types alignment(8) {
  callable @yieldable flags([yieldable]) depth(0) : @empty
  callable @bounded flags([]) depth(0) : @borrow
}
section exports alignment(8) {
  export @export0 name(@run) : @bounded = @function0
}
section functions alignment(8) {
  func @function0 : @bounded flags([has_call]) frame(local_bytes = 0, value_registers = 2, ref_registers = 0, function_registers = 0, local_refs = 0, local_functions = 0, call_values = 0) {
  ^bb0:
    control.block
    control.call @function1 {direct_ref_move_mask = 0x0000}
    control.return
  }
  func @function1 : @yieldable flags([yieldable]) frame(local_bytes = 0, value_registers = 1, ref_registers = 0, function_registers = 0, local_refs = 0, local_functions = 0, call_values = 0) {
  ^bb0:
    control.block
    control.yield.s32 ^bb1
  ^bb1:
    control.block
    %v0 = constant.s16 42
    control.return
  }
}
})"));
  const uint64_t value = 37;
  const iree_vm_typed_span_t span = {&value, 1, IREE_VM_SCALAR_TYPE_I64,
                                     IREE_VM_SPAN_ACCESS_READ, nullptr};
  auto argument = iree_vm_variant_from_span_borrowed(&span);
  int64_t result = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        Invoke({&argument, 1}, &result));
}

}  // namespace
