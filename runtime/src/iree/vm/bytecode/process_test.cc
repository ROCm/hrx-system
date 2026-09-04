// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/process.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer_provider.h"
#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/bytecode/verifier_testdata.h"

namespace {

class BytecodeProcessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_verifier_testdata_size(), 1u);
    const iree_const_byte_span_t contents =
        iree_make_const_byte_span(files[0].data, files[0].size);
    IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(contents, &plan_));
    image_.layout = plan_.layout;
    image_.process_layout = plan_.process_layout;
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         plan_.process_layout.total_size,
                                         (void**)&process_storage_));
    std::memset(process_storage_, 0, plan_.process_layout.total_size);
  }

  void TearDown() override {
    iree_vm_bytecode_process_detach_state(
        &image_.base_module,
        iree_make_byte_span(process_storage_, plan_.process_layout.total_size));
    iree_allocator_free(iree_allocator_system(), process_storage_);
  }

  iree_vm_bytecode_module_plan_t plan_ = {};
  iree_vm_bytecode_image_t image_ = {};
  uint8_t* process_storage_ = nullptr;
};

TEST_F(BytecodeProcessTest, SealsOnlyAfterRequiredImmutableInitialization) {
  const iree_byte_span_t storage =
      iree_make_byte_span(process_storage_, plan_.process_layout.total_size);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_bytecode_process_seal_state(&image_.base_module, storage));

  iree_vm_bytecode_process_bit_set(
      iree_vm_bytecode_process_value_set_bits(&image_, process_storage_), 0);
  IREE_EXPECT_OK(
      iree_vm_bytecode_process_seal_state(&image_.base_module, storage));
  EXPECT_EQ(
      iree_vm_bytecode_process_header(process_storage_)->construction_state,
      IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_SEALED);
}

static void RecordBufferRelease(void* user_data, iree_byte_span_t storage) {
  (void)storage;
  ++*(int*)user_data;
}

TEST_F(BytecodeProcessTest, DetachReleasesRefGlobals) {
  int release_count = 0;
  iree_vm_buffer_t* buffer = nullptr;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  IREE_ASSERT_OK(iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                     iree_byte_span_empty(), release_callback,
                                     iree_allocator_system(), &buffer));
  void* buffer_ptr = buffer;
  const iree_vm_ref_type_t buffer_type = iree_vm_ref_type_storage_at(
      iree_vm_buffer_provider_table()->types, IREE_VM_REF_TYPE_BUFFER);
  iree_vm_bytecode_process_refs(&image_, process_storage_)[0] =
      iree_vm_ref_from_ptr_move(&buffer_ptr, buffer_type);

  iree_vm_bytecode_process_detach_state(
      &image_.base_module,
      iree_make_byte_span(process_storage_, plan_.process_layout.total_size));
  EXPECT_EQ(release_count, 1);
}

}  // namespace
