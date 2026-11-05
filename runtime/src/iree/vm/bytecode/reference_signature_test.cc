// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/reference_signature_testdata.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"

namespace {

class ReferenceSignatureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    const iree_file_toc_t* files =
        iree_vm_bytecode_reference_signature_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_reference_signature_testdata_size(), 1u);
    IREE_ASSERT_OK(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                              files[0].size),
        iree_allocator_system(), &module_fixture_));
    const iree_vm_bytecode_module_storage_t storage = {
        iree_const_cast_byte_span(module_fixture_.contents),
        iree_allocator_null(),
    };
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("references"), storage, iree_allocator_system(),
        &module_));
    IREE_ASSERT_OK(
        iree_vm_program_create({module_, iree_vm_module_span_empty()},
                               iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_bytecode_module_fixture_deinitialize(&module_fixture_);
    iree_vm_environment_free(environment_);
  }

  iree_vm_function_t Lookup(iree_string_view_t name) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_EXPECT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("references"), name, &function));
    return function;
  }

  auto MakeBuffer() {
    iree_vm_buffer_t* buffer = nullptr;
    const iree_vm_buffer_release_callback_t callback = {
        [](void* user_data, iree_byte_span_t) {
          ++*static_cast<int*>(user_data);
        },
        &release_count_,
    };
    IREE_EXPECT_OK(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_byte_span(&payload_, 1),
        callback, iree_allocator_system(), &buffer));
    return std::unique_ptr<iree_vm_buffer_t, decltype(&iree_vm_buffer_release)>(
        buffer, iree_vm_buffer_release);
  }

  void ExpectLiveBuffer(iree_vm_variant_t value) {
    EXPECT_EQ(release_count_, 0);
    iree_vm_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        iree_vm_buffer_ptr_from_variant_borrowed(&types_, value, &buffer));
    ASSERT_NE(buffer, nullptr);
    iree_const_byte_span_t contents = iree_const_byte_span_empty();
    IREE_ASSERT_OK(iree_vm_buffer_map_read(buffer, 0, 1, &contents));
    EXPECT_EQ(contents.data[0], 42);
  }

  // Payload storage outlives buffers even when an assertion aborts a test.
  uint8_t payload_ = 42;
  // Finalizer count survives invocation teardown on failure.
  int release_count_ = 0;
  // Environment owns the canonical runtime type descriptors.
  iree_vm_environment_t* environment_ = nullptr;
  // Resolved VM buffer identity.
  iree_vm_ref_types_t types_ = {};
  // Assembled bytes borrowed by the bytecode provider.
  iree_vm_bytecode_module_fixture_t module_fixture_ = {};
  // Verified bytecode provider.
  iree_vm_module_t* module_ = nullptr;
  // Linked callable identities and provider ownership.
  iree_vm_program_t* program_ = nullptr;
  // Process state bound to the linked program.
  iree_vm_process_t* process_ = nullptr;
  // Caller-owned invocation storage, including suspended frames.
  alignas(
      iree_max_align_t) std::array<uint8_t, 16 * 1024> invocation_storage_ = {};
  // Initialized execution state within invocation_storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(ReferenceSignatureTest, ReflectionPreservesDeclaredMode) {
  for (const auto name : {IREE_SV("owned"), IREE_SV("borrow")}) {
    iree_vm_export_t exported = {};
    IREE_ASSERT_OK(iree_vm_module_lookup_export(module_, name, &exported));
    iree_host_size_t required_size = 0;
    IREE_ASSERT_OK(iree_vm_export_query_description(
        exported, iree_byte_span_empty(), &required_size, nullptr));
    std::vector<std::max_align_t> storage(
        (required_size + sizeof(std::max_align_t) - 1) /
        sizeof(std::max_align_t));
    iree_vm_export_description_t description = {};
    IREE_ASSERT_OK(iree_vm_export_query_description(
        exported,
        iree_make_byte_span(reinterpret_cast<uint8_t*>(storage.data()),
                            required_size),
        &required_size, &description));
    ASSERT_EQ(description.arguments.count, 1u);
    ASSERT_EQ(description.results.count, 1u);
    const auto expected_kind = iree_string_view_equal(name, IREE_SV("owned"))
                                   ? IREE_VM_SIGNATURE_TYPE_KIND_REF
                                   : IREE_VM_SIGNATURE_TYPE_KIND_PTR;
    for (const auto field :
         {description.arguments.data[0], description.results.data[0]}) {
      EXPECT_EQ(field.type.kind, expected_kind);
      EXPECT_EQ(field.type.value.ref, types_.buffer);
    }
  }
}

TEST_F(ReferenceSignatureTest, ProtectsBorrowedResultsAfterHostOwnerRelease) {
  for (const auto name : {IREE_SV("borrow"), IREE_SV("local")}) {
    release_count_ = 0;
    auto buffer = MakeBuffer();
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_borrowed(&types_, buffer.get()),
    };
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_, Lookup(name),
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
    buffer.reset();
    ExpectLiveBuffer(results[0]);
    iree_vm_variant_reset(&results[0]);
    EXPECT_EQ(release_count_, 1);
  }
}

TEST_F(ReferenceSignatureTest, TransfersProtectionThroughBorrowedParameter) {
  auto owner = MakeBuffer();
  auto* buffer = owner.release();
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
  };
  iree_vm_variant_t results[1] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, Lookup(IREE_SV("local")),
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  EXPECT_EQ(buffer, nullptr);
  ExpectLiveBuffer(results[0]);
  iree_vm_variant_reset(&results[0]);
  EXPECT_EQ(release_count_, 1);
}

TEST_F(ReferenceSignatureTest, AcceptsNullInEitherDeclaredMode) {
  for (const auto name : {IREE_SV("owned"), IREE_SV("borrow")}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_null()};
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_, Lookup(name),
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    EXPECT_TRUE(iree_vm_variant_is_null(results[0]));
    iree_vm_variant_reset(&results[0]);
  }
}

TEST_F(ReferenceSignatureTest, IndirectCallsRequireExactDeclaredMode) {
  for (const auto name : {IREE_SV("owned"), IREE_SV("borrow")}) {
    iree_vm_export_t exported = {};
    IREE_ASSERT_OK(iree_vm_module_lookup_export(module_, name, &exported));
    iree_vm_function_ref_t function_ref = {};
    IREE_ASSERT_OK(
        iree_vm_function_ref_from_export(program_, exported, &function_ref));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_function_ref(function_ref),
        iree_vm_variant_null(),
    };
    iree_vm_variant_t results[1] = {};
    iree_status_t status =
        iree_vm_invoke(invocation_, Lookup(IREE_SV("indirect")),
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    if (iree_string_view_equal(name, IREE_SV("borrow"))) {
      IREE_EXPECT_OK(status);
      EXPECT_TRUE(iree_vm_variant_is_null(results[0]));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_reset(&results[0]);
  }
}

TEST_F(ReferenceSignatureTest, KeepsProtectionAcrossSuspensionAndCancellation) {
  for (bool cancel : {false, true}) {
    release_count_ = 0;
    auto owner = MakeBuffer();
    auto* buffer = owner.release();
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
    };
    iree_vm_variant_t results[] = {iree_vm_variant_empty()};
    int wake_count = 0;
    const iree_vm_invocation_wake_callback_t wake_callback = {
        [](void* user_data) { ++*static_cast<int*>(user_data); }, &wake_count};
    iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    IREE_ASSERT_OK(iree_vm_invocation_start(
        invocation_, Lookup(IREE_SV("yield")),
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results), wake_callback, &outcome));
    ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
    EXPECT_EQ(wake_count, 1);
    EXPECT_EQ(release_count_, 0);
    EXPECT_TRUE(iree_vm_variant_is_empty(results[0]));
    if (cancel) {
      EXPECT_TRUE(iree_vm_invocation_request_cancel(
          invocation_, IREE_VM_CANCEL_REASON_CANCELLED));
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          iree_vm_invocation_resume(
              invocation_, iree_vm_variant_span_from_array(results), &outcome));
      EXPECT_TRUE(iree_vm_variant_is_empty(results[0]));
    } else {
      IREE_ASSERT_OK(iree_vm_invocation_resume(
          invocation_, iree_vm_variant_span_from_array(results), &outcome));
      EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
      ExpectLiveBuffer(results[0]);
    }
    iree_vm_variant_reset(&results[0]);
    EXPECT_EQ(release_count_, 1);
  }
}

TEST_F(ReferenceSignatureTest, RejectsUnprotectedExternalBorrowAcrossYield) {
  auto buffer = MakeBuffer();
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&types_, buffer.get()),
  };
  iree_vm_variant_t results[] = {iree_vm_variant_empty()};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(invocation_, Lookup(IREE_SV("yield")),
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results), {},
                               &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(results[0]));
  EXPECT_EQ(release_count_, 0);
  buffer.reset();
  EXPECT_EQ(release_count_, 1);
}

TEST_F(ReferenceSignatureTest, FailureReleasesBorrowedParameterProtection) {
  auto owner = MakeBuffer();
  auto* buffer = owner.release();
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
  };
  iree_vm_variant_t results[] = {iree_vm_variant_empty()};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_vm_invoke(invocation_, Lookup(IREE_SV("fail")),
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(results[0]));
  EXPECT_EQ(release_count_, 1);
}

}  // namespace
