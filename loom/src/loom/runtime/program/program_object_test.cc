// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/invocation_test_module.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/program.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, object_source, "",
          "Source-defined object state transitions.");

namespace {

class ProgramObjectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator_), &target_environment_)));
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
        environment_, iree_vm_invocation_test_object_type()->table));
    IREE_ASSERT_OK(
        iree_vm_invocation_test_module_initialize(&counters_, &provider_));
  }

  void TearDown() override {
    loom_runtime_program_free(program_);
    if (provider_.base.descriptor) {
      iree_vm_module_release(&provider_.base);
      EXPECT_EQ(counters_.destroy_count, 1);
    }
    iree_vm_environment_free(environment_);
    loomc_target_environment_release(target_environment_);
  }

  iree_status_t Compile(std::initializer_list<iree_string_view_t> roots) {
    const loom_runtime_sources_t sources = {.base_directory = IREE_SV(".")};
    iree_vm_module_t* library = &provider_.base;
    return loom_runtime_program_allocate(
        target_environment_, environment_, &sources,
        iree_make_cstring_view(FLAG_object_source), roots.size(), roots.begin(),
        {&library, 1}, &compilation_, host_allocator_, &program_);
  }

  iree_status_t Invoke(iree_string_view_t name,
                       iree_vm_variant_span_t arguments,
                       iree_vm_variant_span_t results) {
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(
        loom_runtime_program_lookup(program_, name, &function));
    return iree_vm_invoke(loom_runtime_program_invocation(program_), function,
                          arguments, results);
  }

  static iree_vm_variant_t CreateObject(
      int* destroy_count, iree_vm_invocation_test_object_t* object) {
    iree_vm_invocation_test_object_initialize(destroy_count, object);
    void* pointer = object;
    return iree_vm_variant_from_ptr_move(&pointer,
                                         iree_vm_invocation_test_object_type());
  }

  static void ExpectObject(iree_vm_variant_t value,
                           iree_vm_invocation_test_object_t* object) {
    void* pointer = nullptr;
    IREE_EXPECT_OK(iree_vm_ptr_from_variant_borrowed(
        value, iree_vm_invocation_test_object_type(), &pointer));
    EXPECT_EQ(pointer, object);
  }

  // Host allocation for the compiler and VM environment.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Ordinary VM target; no application-specific grammar or compiler context.
  loomc_target_environment_t* target_environment_ = nullptr;
  // Canonical type lookup outliving the program and returned references.
  iree_vm_environment_t* environment_ = nullptr;
  // Native call and module lifetime observations.
  iree_vm_invocation_test_counters_t counters_ = {};
  // Native provider whose payload type is registered through the public API.
  iree_vm_invocation_test_module_t provider_ = {};
  // Compiler diagnostics copied independently of source/compiler storage.
  loom_runtime_compilation_report_t compilation_ = {};
  // Source-JIT code and serialized invocation shared by request states.
  loom_runtime_program_t* program_ = nullptr;
};

TEST_F(ProgramObjectTest, ReturnedOwnerOutlivesCompiledProgram) {
  IREE_ASSERT_OK(Compile({IREE_SV("echo")}));
  int destroy_count = 0;
  iree_vm_invocation_test_object_t object = {};
  iree_vm_variant_t argument = CreateObject(&destroy_count, &object);
  iree_vm_variant_t result = {};
  IREE_EXPECT_OK(Invoke(IREE_SV("echo"), {&argument, 1}, {&result, 1}));
  ExpectObject(result, &object);
  EXPECT_EQ(destroy_count, 0);
  loom_runtime_program_free(program_);
  program_ = nullptr;
  EXPECT_EQ(destroy_count, 0);
  iree_vm_variant_reset(&argument);
  iree_vm_variant_reset(&result);
  EXPECT_EQ(destroy_count, 1);
}

TEST_F(ProgramObjectTest, InterleavedStatesKeepIndependentOwnersAndSnapshots) {
  IREE_ASSERT_OK(Compile({IREE_SV("advance")}));
  int destroyed[4] = {};
  iree_vm_invocation_test_object_t objects[4] = {};
  iree_vm_variant_t states[] = {
      CreateObject(&destroyed[0], &objects[0]),
      CreateObject(&destroyed[1], &objects[1]),
  };
  iree_vm_variant_t snapshots[2] = {};
  for (size_t request = 0; request < 2; ++request) {
    SCOPED_TRACE(request);
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_move(&states[request]),
        CreateObject(&destroyed[request + 2], &objects[request + 2]),
    };
    iree_vm_variant_t results[2] = {};
    IREE_EXPECT_OK(Invoke(IREE_SV("advance"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    states[request] = iree_vm_variant_move(&results[0]);
    snapshots[request] = iree_vm_variant_move(&results[1]);
    ExpectObject(states[request], &objects[request + 2]);
    ExpectObject(snapshots[request], &objects[request]);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
  EXPECT_THAT(destroyed, ::testing::ElementsAre(0, 0, 0, 0));

  // Releasing one reader's snapshot retires only its replaced payload.
  iree_vm_variant_reset(&snapshots[0]);
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0, 0, 0));
  ExpectObject(states[0], &objects[2]);
  ExpectObject(states[1], &objects[3]);
  ExpectObject(snapshots[1], &objects[1]);

  // A later turn can end one request while the other retains its state and
  // earlier snapshot. Null uses the same typed source ownership contract.
  void* empty = nullptr;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_move(&states[0]),
      iree_vm_variant_from_ptr_move(&empty,
                                    iree_vm_invocation_test_object_type()),
  };
  iree_vm_variant_t results[2] = {};
  IREE_EXPECT_OK(Invoke(IREE_SV("advance"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  ExpectObject(results[0], nullptr);
  ExpectObject(results[1], &objects[2]);
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0, 0, 0));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0, 1, 0));

  loom_runtime_program_free(program_);
  program_ = nullptr;
  ExpectObject(states[1], &objects[3]);
  ExpectObject(snapshots[1], &objects[1]);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(states));
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0, 1, 1));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(snapshots));
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 1, 1, 1));
  EXPECT_EQ(counters_.start_count, 3);
}

TEST_F(ProgramObjectTest, NativeFailureRetiresInputAndPreservesHostResult) {
  IREE_ASSERT_OK(Compile({IREE_SV("echo"), IREE_SV("fail")}));
  int destroyed[2] = {};
  iree_vm_invocation_test_object_t objects[2] = {};
  iree_vm_variant_t argument = CreateObject(&destroyed[0], &objects[0]);
  iree_vm_variant_t result = CreateObject(&destroyed[1], &objects[1]);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        Invoke(IREE_SV("fail"), {&argument, 1}, {&result, 1}));
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0));
  ExpectObject(result, &objects[1]);
  iree_vm_variant_reset(&argument);

  // Terminal cleanup leaves the same invocation reusable for another request.
  argument = iree_vm_variant_move(&result);
  IREE_EXPECT_OK(Invoke(IREE_SV("echo"), {&argument, 1}, {&result, 1}));
  ExpectObject(result, &objects[1]);
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 0));
  iree_vm_variant_reset(&argument);
  iree_vm_variant_reset(&result);
  EXPECT_THAT(destroyed, ::testing::ElementsAre(1, 1));
}

TEST_F(ProgramObjectTest, BindingPreservesAuthoredOwnerAndPointerResults) {
  IREE_ASSERT_OK(Compile({IREE_SV("advance")}));
  const iree_vm_signature_type_t owner = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value = {.ref = iree_vm_invocation_test_object_type()},
  };
  const iree_vm_signature_type_t arguments[] = {owner, owner};
  iree_vm_signature_type_t results[] = {owner, owner};
  results[1].kind = IREE_VM_SIGNATURE_TYPE_KIND_PTR;
  iree_vm_function_t function = {};
  const loom_runtime_program_binding_t binding = {
      IREE_SV("advance"), {arguments, 2}, {results, 2}, &function};
  IREE_ASSERT_OK(loom_runtime_program_bind(program_, 1, &binding));
  results[1].kind = IREE_VM_SIGNATURE_TYPE_KIND_REF;
  const iree::Status status = loom_runtime_program_bind(program_, 1, &binding);
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("source declares ptr<invocation.test.object>"));
  EXPECT_EQ(counters_.start_count, 0);
}

TEST_F(ProgramObjectTest, SynchronousSourceCannotBindSuspendingProvider) {
  const iree::Status status = Compile({IREE_SV("yield")});
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("forbids target suspension"));
  EXPECT_EQ(program_, nullptr);
  EXPECT_EQ(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  EXPECT_EQ(counters_.start_count, 0);
}

}  // namespace
