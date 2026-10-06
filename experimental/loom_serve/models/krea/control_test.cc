// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, control_source, "", "Production image control source.");

namespace {

class Krea2ControlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    IREE_ASSERT_OK(loom_serve_input_module_create(environment, nullptr, &input,
                                                  allocator));
    const iree_string_view_t root = IREE_SVL("prepare_request");
    IREE_ASSERT_OK(loom_serve_program_create(
        environment, iree_make_cstring_view(FLAG_control_source), 1, &root,
        (iree_vm_module_span_t){&input, 1}, allocator, &program));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"), root, &prepare));
  }

  void TearDown() override {
    loom_serve_program_destroy(program);
    iree_vm_module_release(input);
    iree_vm_environment_free(environment);
  }

  // Allocator owning the process and temporary argument wrappers.
  const iree_allocator_t allocator = iree_allocator_system();
  // Owner of the VM reference type provider.
  iree_vm_environment_t* environment = nullptr;
  // Canonical buffer reference type, borrowed from environment.
  iree_vm_ref_types_t types = {};
  // One source-JIT request process reused across input failures.
  loom_serve_program_t* program = nullptr;
  // Actual source request export; no tokenizer is installed in these failures.
  iree_vm_function_t prepare = {};
  // Synchronous source validation capability, retained by the program.
  iree_vm_module_t* input = nullptr;
};

TEST_F(Krea2ControlTest, SourceRejectsStrengthBeforeTokenization) {
  for (float strength : {0.0f, std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN(), 1.0f}) {
    iree_vm_buffer_t* state = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_create(8, 8, allocator, &state));
    uint8_t tag_bytes[8];
    iree_unaligned_store_le_u64(tag_bytes, 512);
    iree_vm_buffer_t* tags = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span(tag_bytes, sizeof(tag_bytes)),
        iree_vm_buffer_release_callback_null(), allocator, &tags));
    iree_vm_buffer_t* prompt = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_create(0, 1, allocator, &prompt));
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_move(&types, &state),
        iree_vm_buffer_variant_from_ptr_move(&types, &tags),
        iree_vm_variant_from_i32(1),
        iree_vm_buffer_variant_from_ptr_move(&types, &prompt),
        iree_vm_variant_from_i64(0),
        iree_vm_variant_from_f32(strength)};
    iree_vm_variant_t results[2] = {};
    IREE_EXPECT_STATUS_IS(
        strength == 1.0f ? IREE_STATUS_FAILED_PRECONDITION
                         : IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_invoke(loom_serve_program_invocation(program), prepare,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results)));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

}  // namespace
