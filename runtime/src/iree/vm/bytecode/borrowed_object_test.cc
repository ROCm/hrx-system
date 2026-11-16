// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/sync.h"

namespace {

extern const iree_vm_ref_type_table_t kTypes;
const iree_vm_ref_type_descriptor_t kType = {nullptr, &kTypes, IREE_SV("byte")};
const iree_vm_ref_type_t kRefTypes[] = {&kType};
const iree_vm_ref_type_table_t kTypes = {
    sizeof(kTypes),
    IREE_VM_REF_TYPE_TABLE_FLAG_BORROWED_ONLY,
    IREE_SV("borrowed"),
    {kRefTypes, 1}};
const iree_vm_module_signature_type_t kArguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0}};
const iree_vm_module_signature_type_t kResults[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
const iree_vm_module_callable_type_declaration_t kCallables[] = {
    {{{kArguments, 1, 0, 1, 0}, {kResults, 1, 1, 0, 0}}, 0, 0, 0}};
const iree_vm_module_export_declaration_t kExports[] = {
    {IREE_SV("inspect"), 0, 0, 0}};

// One read-only byte cannot accidentally satisfy an intrusive-prefix contract.
const uint8_t kObject = 42;

iree_vm_variant_t Borrow() {
  return iree_vm_variant_from_ptr_borrowed(const_cast<uint8_t*>(&kObject),
                                           &kType);
}

iree_status_t Inspect(iree_vm_module_t* module,
                      const iree_vm_module_function_start_params_t* params,
                      iree_vm_execution_outcome_t* outcome) {
  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &ref);
  void* pointer = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_ptr_from_ref_borrowed(ref, &kType, &pointer));
  iree_vm_call_value_result_store(
      &params->call, 0, pointer ? *static_cast<const uint8_t*>(pointer) : 0);
  iree_vm_ref_reset(&ref);
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

std::string Source(const std::string& body, size_t argument_count = 1,
                   const std::string& globals = "") {
  std::string source = R"(vm.module core(0.0) {
  section strings alignment(8) {
    string @namespace = "borrowed"
    string @type = "byte"
    string @inspect = "inspect"
    string @module = "native"
    string @entry = "run"
  }
  section ref_types alignment(8) {
    namespace @family name(@namespace) { ref_type @type0 name(@type) }
  }
  section signatures alignment(8) {
    signature @signature0 (ptr(@type0)) -> (i32)
)";
  if (argument_count != 1) {
    source += "signature @signature1 (";
    for (size_t i = 0; i < argument_count; ++i) {
      source += i ? ", ptr(@type0)" : "ptr(@type0)";
    }
    source += ") -> (i32)\n";
  }
  source += R"( }
  section callable_types alignment(8) {
    callable @callable0 flags([]) depth(0) : @signature0
)";
  if (argument_count != 1) {
    source += "callable @callable1 flags([]) depth(0) : @signature1\n";
  }
  const std::string callable =
      argument_count == 1 ? "@callable0" : "@callable1";
  source +=
      R"( }
  section imports alignment(8) {
    group @group module_name(@module) {
      import @import0 name(@inspect) flags([]) : @callable0
    }
  }
  section exports alignment(8) {
    export @export0 name(@entry) : )" +
      callable + R"( = @function0
  }
  section functions alignment(8) {
    func @function0 : )" +
      callable +
      R"( flags([has_call]) frame(local_bytes = 0, value_registers = 1, ref_registers = 16, function_registers = 0, local_refs = 3, local_functions = 0) {
    ^bb0:
      control.block
)" + body +
      R"(
      control.call @import0 {direct_ref_move_mask = 0x0000}
      control.return
    }
  }
)" + globals +
      "}\n";
  return source;
}

class BorrowedObjectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_.vtable().function_start = Inspect;
    IREE_ASSERT_OK(native_.Publish(
        iree::vm::testing::TableModuleDefinition(IREE_SV("native"), 1)
            .WithRefTypes(kRefTypes)
            .WithCallableTypes(kCallables)
            .WithExports(kExports)));
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(
        iree_vm_environment_register_ref_type_table(environment_, &kTypes));
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

  iree_status_t Load(const std::string& source) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(source.data(), source.size()),
        iree_allocator_system(), &fixture_));
    return iree_vm_bytecode_module_create(
        environment_, IREE_SV("borrow"),
        {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
        iree_allocator_system(), &module_);
  }

  void Link() {
    iree_vm_module_t* libraries[] = {native_.module()};
    IREE_ASSERT_OK(iree_vm_program_create({module_, {libraries, 1}},
                                          iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_, {},
                                          iree_allocator_system(), &process_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(process_, IREE_SV("borrow"),
                                                   IREE_SV("run"), &function_));
  }

  // Native implementation reads the stack data through the checked borrow API.
  iree::vm::testing::TableModule native_;
  // Canonical type registration, borrowed by module construction.
  iree_vm_environment_t* environment_ = nullptr;
  // Owned assembled source bytes kept live for the module.
  iree_vm_bytecode_module_fixture_t fixture_ = {};
  // Bytecode module owner.
  iree_vm_module_t* module_ = nullptr;
  // Linked module composition.
  iree_vm_program_t* program_ = nullptr;
  // Process used by all invocations in the case.
  iree_vm_process_t* process_ = nullptr;
  // Fixed caller-owned invocation capacity.
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage_{};
  // Invocation borrowing storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Public entry selected from the bytecode module.
  iree_vm_function_t function_ = {};
};

TEST_F(BorrowedObjectTest, PreservesBorrowThroughRegistersLocalsAndNativeCall) {
  IREE_ASSERT_OK(Load(Source(R"(
    %r1 = ref.copy %r0
    %r0 = ref.move %r1
    ref.stack.store.copy %r0, #r2
    %r1 = ref.stack.load.copy #r2
    ref.stack.store.borrow %r1, #r2
    %r0 = ref.stack.load.borrow #r2
    %r1 = ref.borrow %r0
    ref.stack.store.move %r1, #r2
    %r0 = ref.stack.load.move #r2
  )")));
  Link();
  for (int i = 0; i < 2; ++i) {
    auto argument = Borrow();
    iree_vm_variant_t result{};
    IREE_ASSERT_OK(
        iree_vm_invoke(invocation_, function_, {&argument, 1}, {&result, 1}));
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
    EXPECT_EQ(value, 42);
  }
}

TEST_F(BorrowedObjectTest, SelectionPreservesBorrowedOnlyStateAndAliases) {
  IREE_ASSERT_OK(Load(Source(R"(
    %r1 = ref.null
    %v0 = constant.s16 1
    %r0 = ref.select %v0, %r0, %r1
    %v0 = constant.s16 0
    %r0 = ref.select %v0, %r1, %r0
  )")));
  Link();
  for (bool present : {true, false}) {
    auto argument =
        present ? Borrow() : iree_vm_variant_from_ptr_borrowed(nullptr, &kType);
    iree_vm_variant_t result{};
    IREE_ASSERT_OK(
        iree_vm_invoke(invocation_, function_, {&argument, 1}, {&result, 1}));
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
    EXPECT_EQ(value, present ? 42 : 0);
  }
}

TEST_F(BorrowedObjectTest, PreservesBorrowThroughOverflowArguments) {
  IREE_ASSERT_OK(
      Load(Source("%r1 = ref.abi.argument.load.borrow 0\n%r0 = "
                  "ref.abi.argument.load.move 1",
                  18)));
  Link();
  std::array<iree_vm_variant_t, 18> arguments;
  arguments.fill(Borrow());
  iree_vm_variant_t result{};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, function_,
                                {arguments.data(), arguments.size()},
                                {&result, 1}));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
  EXPECT_EQ(value, 42);
}

class BorrowedRetainTest : public BorrowedObjectTest,
                           public ::testing::WithParamInterface<const char*> {};
TEST_P(BorrowedRetainTest,
       RejectsBeforeTouchingObjectAndLeavesInvocationReusable) {
  IREE_ASSERT_OK(Load(Source(GetParam())));
  Link();
  auto argument = Borrow();
  iree_vm_variant_t result = iree_vm_variant_from_i32(-1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_invoke(invocation_, function_, {&argument, 1}, {&result, 1}));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
  EXPECT_EQ(value, -1);
  argument = iree_vm_variant_from_ptr_borrowed(nullptr, &kType);
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation_, function_, {&argument, 1}, {&result, 1}));
  IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
  EXPECT_EQ(value, 0);
}
INSTANTIATE_TEST_SUITE_P(
    OwnershipOpcodes, BorrowedRetainTest,
    ::testing::Values(
        "%r1 = ref.retain %r0", "ref.stack.store.retain %r0, #r2",
        "ref.stack.store.copy %r0, #r2\n%r1 = ref.stack.load.retain #r2"));

TEST_F(BorrowedObjectTest, ConsumesBorrowedArgumentsOnAdmissionFailure) {
  IREE_ASSERT_OK(Load(Source("")));
  Link();
  iree_vm_variant_t arguments[] = {Borrow(), Borrow()};
  iree_vm_variant_t result{};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(invocation_, function_, {arguments, 2}, {&result, 1}));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
}

TEST_F(BorrowedObjectTest, RejectsBorrowedTypeInGlobalsAtModuleLoad) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Load(Source("", 1, R"(
    section globals alignment(8) {
      values count(0) immutable(0)
      refs immutable(0) { global.ref @global0 : @type0 flags([nullable]) }
      functions immutable(0) {}
    }
  )")));
}

TEST(BorrowedObject, RejectsOwningArgumentsAndAllResultSignatures) {
  for (bool is_result : {false, true}) {
    for (auto kind : {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF,
                      IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR}) {
      if (!is_result && kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR) {
        continue;
      }
      const iree_vm_module_signature_type_t field[] = {
          {static_cast<uint16_t>(kind), 0}};
      // A one-field signature exercises the same publication validation used by
      // native providers and resolved bytecode modules.
      iree_vm_module_callable_type_declaration_t declarations[1] = {};
      auto& side = is_result ? declarations[0].signature.results
                             : declarations[0].signature.arguments;
      side = {field, 1, 0, 1, 0};
      iree::vm::testing::TableModule module;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          module.Publish(
              iree::vm::testing::TableModuleDefinition(IREE_SV("invalid"), 1)
                  .WithRefTypes(kRefTypes)
                  .WithCallableTypes(declarations)));
    }
  }
}

TEST(BorrowedObject,
     CheckedOwnershipConversionsFailWithoutChangingSourceOrOutput) {
  auto variant = Borrow();
  iree_vm_ref_t ref{};
  IREE_ASSERT_OK(iree_vm_ref_from_variant_borrowed(variant, &ref));
  EXPECT_TRUE(iree_vm_ref_is_borrowed_only(ref));
  const auto original = ref;
  void* pointer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_vm_ptr_from_ref_retained(ref, &kType, &pointer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_vm_ptr_from_ref_move(&ref, &kType, &pointer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_ptr_from_variant_retained(variant, &kType, &pointer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_ptr_from_variant_move(&variant, &kType, &pointer));
  EXPECT_EQ(pointer, nullptr);
  iree_vm_ref_t output{};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_vm_ref_from_variant_retained(variant, &output));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_vm_ref_from_variant_move(&variant, &output));
  EXPECT_TRUE(iree_vm_ref_is_null(output));
  EXPECT_EQ(ref.object, original.object);
  EXPECT_EQ(ref.type_and_state, original.type_and_state);
  EXPECT_EQ(variant.payload, reinterpret_cast<uintptr_t>(&kObject));
  auto round_trip = iree_vm_variant_from_ref_borrowed(ref);
  EXPECT_EQ(round_trip.metadata, variant.metadata);
  iree_vm_variant_reset(&round_trip);
  iree_vm_variant_reset(&variant);
  iree_vm_ref_reset(&ref);
}

enum class Publication { kResult, kOverflowResult, kGlobal };
class BorrowedPublicationTest
    : public BorrowedObjectTest,
      public ::testing::WithParamInterface<Publication> {};
TEST_P(BorrowedPublicationTest, RejectsBorrowMasqueradingAsManagedStorage) {
  // The declared output/global has a legal managed type. A forged guest body
  // routes its borrowed-only input there without an ownership-producing opcode.
  const bool overflow = GetParam() == Publication::kOverflowResult;
  const size_t result_count = overflow ? 17 : 1;
  std::string result_types = "ptr(@type1)";
  for (size_t i = 1; i < result_count; ++i) {
    result_types += ", ptr(@type1)";
  }
  const char* body = GetParam() == Publication::kGlobal
                         ? "global.ref.mutable.store.move %r0, @global0\n"
                     : overflow ? "ref.abi.result.store.move %r0, 0\n"
                                : "";
  const std::string source = std::string(R"(vm.module core(0.0) {
    section strings alignment(8) {
      string @borrowed = "borrowed"
      string @byte = "byte"
      string @vm = "vm"
      string @buffer = "buffer"
      string @run = "run"
    }
    section ref_types alignment(8) {
      namespace @family0 name(@borrowed) { ref_type @type0 name(@byte) }
      namespace @family1 name(@vm) { ref_type @type1 name(@buffer) }
    }
    section signatures alignment(8) {
      signature @signature0 (ptr(@type0)) -> ()") +
                             result_types + R"()
    }
    section callable_types alignment(8) {
      callable @callable0 flags([]) depth(0) : @signature0
    }
    section exports alignment(8) {
      export @export0 name(@run) : @callable0 = @function0
    }
    section functions alignment(8) {
      func @function0 : @callable0 flags([]) frame(local_bytes = 0, value_registers = 0, ref_registers = 16, function_registers = 0, local_refs = 0, local_functions = 0) {
      ^bb0:
        control.block
  )" + std::string(body) +
                             R"(
        control.return
      }
    }
    section globals alignment(8) {
      values count(0) immutable(0)
      refs immutable(0) { global.ref @global0 : @type1 flags([nullable]) }
      functions immutable(0) {}
    }
  })";
  IREE_ASSERT_OK(Load(source));
  Link();
  auto argument = Borrow();
  std::array<iree_vm_variant_t, 17> results{};
  results[0] = iree_vm_variant_from_i32(-1);
  const iree_vm_variant_span_t result_span = {results.data(), result_count};
  IREE_EXPECT_STATUS_IS(
      overflow ? IREE_STATUS_FAILED_PRECONDITION : IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(invocation_, function_, {&argument, 1}, result_span));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, -1);
  argument = iree_vm_variant_from_ptr_borrowed(nullptr, &kType);
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation_, function_, {&argument, 1}, result_span));
  for (size_t i = 0; i < result_count; ++i) {
    EXPECT_TRUE(iree_vm_variant_is_null(results[i]));
  }
}
INSTANTIATE_TEST_SUITE_P(ResultAndGlobal, BorrowedPublicationTest,
                         ::testing::Values(Publication::kResult,
                                           Publication::kOverflowResult,
                                           Publication::kGlobal));

// A native callee lying about suspension is still subject to the invocation's
// borrowed-root rule, even after it has allocated durable frame storage.
iree_status_t SuspendBorrow(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* outcome) {
  iree_vm_ref_t ref{};
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &ref);
  if (!ref.object) {
    return Inspect(module, params, outcome);
  }
  iree_vm_frame_t* frame = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_invocation_push_frame(params, {8, 8}, nullptr, &frame));
  *outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

TEST(BorrowedObject, RejectsDeclaredAndUnexpectedSuspensionAndUnwinds) {
  for (bool declared : {false, true}) {
    iree_vm_module_callable_type_declaration_t callables[] = {kCallables[0]};
    callables[0].flags = declared ? IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD : 0;
    iree::vm::testing::TableModule module;
    module.vtable().function_start = SuspendBorrow;
    IREE_ASSERT_OK(module.Publish(
        iree::vm::testing::TableModuleDefinition(IREE_SV("native"), 1)
            .WithRefTypes(kRefTypes)
            .WithCallableTypes(callables)
            .WithExports(kExports)));
    iree_vm_program_t* program = nullptr;
    IREE_ASSERT_OK(iree_vm_program_create({module.module(), {}},
                                          iree_allocator_system(), &program));
    alignas(iree_max_align_t) uint8_t storage[4096] = {};
    iree_vm_invocation_t* invocation = nullptr;
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage, sizeof(storage)), &invocation));
    iree_vm_process_t* process = nullptr;
    IREE_ASSERT_OK(iree_vm_process_create(program, invocation, {},
                                          iree_allocator_system(), &process));
    iree_vm_function_t function{};
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        process, IREE_SV("native"), IREE_SV("inspect"), &function));
    auto argument = Borrow();
    iree_vm_variant_t result = iree_vm_variant_from_i32(-1);
    IREE_EXPECT_STATUS_IS(
        declared ? IREE_STATUS_INVALID_ARGUMENT
                 : IREE_STATUS_FAILED_PRECONDITION,
        iree_vm_invoke(invocation, function, {&argument, 1}, {&result, 1}));
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
    EXPECT_EQ(value, -1);
    argument = iree_vm_variant_from_ptr_borrowed(nullptr, &kType);
    IREE_ASSERT_OK(
        iree_vm_invoke(invocation, function, {&argument, 1}, {&result, 1}));
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
    EXPECT_EQ(value, 0);
    iree_vm_process_release(process);
    iree_vm_invocation_deinitialize(invocation);
    iree_vm_program_release(program);
  }
}

TEST(BorrowedObject, RejectsDestructorOnBorrowedOnlyType) {
  iree_vm_ref_type_table_t table = kTypes;
  const iree_vm_ref_type_descriptor_t descriptor = {+[](void*) {}, &table,
                                                    IREE_SV("byte")};
  const iree_vm_ref_type_t types[] = {&descriptor};
  table.types = {types, 1};
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_environment_register_ref_type_table(environment, &table));
  iree_vm_environment_free(environment);
}

}  // namespace
