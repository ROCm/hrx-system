// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

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
    const iree_string_view_t entry = IREE_SVL("select_stage");
    IREE_ASSERT_OK(loom_serve_program_create(
        environment, iree_make_cstring_view(FLAG_control_source), 1, &entry,
        iree_vm_module_span_empty(), allocator, &program));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"), entry, &select));
  }

  void TearDown() override {
    loom_serve_program_destroy(program);
    iree_vm_environment_free(environment);
  }

  void ExpectSelection(const std::vector<int64_t>& tags, int32_t token_count,
                       int32_t expected) {
    std::vector<uint8_t> storage(tags.size() * sizeof(int64_t));
    for (size_t i = 0; i < tags.size(); ++i) {
      iree_unaligned_store_le_u64(storage.data() + i * sizeof(int64_t),
                                  tags[i]);
    }
    iree_vm_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span(storage.data(), storage.size()),
        iree_vm_buffer_release_callback_null(), allocator, &buffer));
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_move(&types, &buffer),
        iree_vm_variant_from_i32((int32_t)tags.size()),
        iree_vm_variant_from_i32(token_count),
    };
    iree_vm_variant_t results[1] = {};
    iree_status_t status =
        iree_vm_invoke(loom_serve_program_invocation(program), select,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    int32_t selected = -1;
    if (iree_status_is_ok(status)) {
      IREE_EXPECT_OK(iree_vm_i32_from_variant(results[0], &selected));
      EXPECT_EQ(selected, expected);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_EXPECT_OK(status);
  }

  // Allocator owning the process and temporary argument wrappers.
  const iree_allocator_t allocator = iree_allocator_system();
  // Owner of the VM reference type provider.
  iree_vm_environment_t* environment = nullptr;
  // Canonical buffer reference type, borrowed from environment.
  iree_vm_ref_types_t types = {};
  // One source-JIT selector reused across all prompt counts.
  loom_serve_program_t* program = nullptr;
  // Selector export in that process.
  iree_vm_function_t select = {};
};

TEST_F(Krea2ControlTest, RetainedShapesRespectLiveKeyBoundary) {
  for (const int64_t maximum : {16, 80, 128, 144, 192, 512}) {
    SCOPED_TRACE(maximum);
    std::vector<int64_t> tags = {maximum};
    const bool compact = maximum > 128 && maximum % 64 == 0;
    if (compact) {
      tags.push_back(128);
    }
    for (int32_t tokens : {0, 34, 88, 98, 99, 128, 429, 541}) {
      SCOPED_TRACE(tokens);
      if (tokens > maximum + 29) {
        continue;
      }
      ExpectSelection(tags, tokens, compact && tokens <= 98 ? 1 : 0);
    }
  }
}

TEST_F(Krea2ControlTest, SelectionFollowsTagsNotFixedSlotNumbers) {
  ExpectSelection({512, 256, 128}, 98, 2);
  ExpectSelection({512, 256, 128}, 99, 0);
  ExpectSelection({512, 256}, 34, 0);
}

}  // namespace
