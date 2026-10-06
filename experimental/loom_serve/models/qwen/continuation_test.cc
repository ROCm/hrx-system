// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, continuation_source, "", "Production continuation source.");

namespace {

class ContinuationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    const iree_string_view_t roots[] = {IREE_SVL("qwen38_continue_spans")};
    IREE_ASSERT_OK(loom_serve_program_create(
        environment, iree_make_cstring_view(FLAG_continuation_source), 1, roots,
        iree_vm_module_span_empty(), allocator, &program));
    IREE_ASSERT_OK(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), roots[0], &function));
  }

  void TearDown() override {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    loom_serve_program_destroy(program);
    iree_vm_environment_free(environment);
  }

  void Run(int count, int trial) {
    constexpr int kContext = 1024;
    constexpr int kEos = 42;
    std::array<std::vector<int32_t>, 3> actual = {
        std::vector<int32_t>(516, -99), std::vector<int32_t>(512, -77),
        std::vector<int32_t>(208, -55)};
    auto& metadata = actual[0];
    auto& results = actual[2];
    metadata[0] = count * 4;
    metadata[1] = count;
    metadata[2] = count * 4;
    metadata[515] = kEos;
    std::vector<int> continuing;
    for (int span = 0; span < count; ++span) {
      const int kind = trial >= 10 ? trial - 10 : (span + trial) % 5;
      const int consumed = 1 + (span + trial) % 4;
      const int position = kind == 4   ? kContext - consumed - 3
                           : trial % 2 ? kContext - consumed - 4
                                       : 100 + span;
      const int credit = kind == 3 ? consumed : trial % 2 ? consumed + 1 : 8;
      const int fields[] = {4, position, span * 4, 15 - span, span * 4};
      std::memcpy(metadata.data() + 3 + span * 5, fields, sizeof(fields));
      metadata[387 + span * 2] = kind == 1 ? 0 : 2;
      metadata[388 + span * 2] = credit;
      results[span * 6] = consumed;
      results[span * 6 + 1] = consumed;
      for (int token = 0; token < 4; ++token) {
        results[span * 6 + 2 + token] =
            kind == 2 && token == consumed - 1 ? kEos : 1000 + span * 4 + token;
      }
      if (kind == 0) {
        continuing.push_back(span);
      }
    }
    auto expected = actual;
    std::fill(expected[1].begin(), expected[1].end(), 0);
    std::fill(expected[2].begin() + 96, expected[2].begin() + 192, 0);
    std::fill(expected[2].begin() + 192, expected[2].end(), -1);
    for (size_t slot = 0; slot < continuing.size(); ++slot) {
      const int span = continuing[slot];
      const int consumed = results[span * 6];
      const int fields[] = {4, metadata[3 + span * 5 + 1] + consumed,
                            static_cast<int>(slot * 4), 15 - span,
                            static_cast<int>(slot * 4)};
      std::memcpy(expected[0].data() + 3 + slot * 5, fields, sizeof(fields));
      expected[0][387 + slot * 2] = 2;
      expected[0][388 + slot * 2] = metadata[388 + span * 2] - consumed;
      for (int token = 0; token < 4; ++token) {
        expected[0][323 + slot * 4 + token] = slot * 4 + token;
      }
      expected[1][slot * 4] = results[span * 6 + 1 + consumed];
      expected[2][192 + slot] = span;
    }
    expected[0][0] = continuing.size() * 4;
    expected[0][1] = continuing.size();
    expected[0][2] = continuing.size() * 4;
    arguments[0] = iree_vm_variant_from_i64(kContext);
    for (size_t i = 0; i < actual.size(); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_vm_buffer_wrap(
          IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
          iree_make_byte_span(actual[i].data(), actual[i].size() * 4),
          iree_vm_buffer_release_callback_null(), allocator, &buffer));
      arguments[1 + i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
    iree_status_t status =
        iree_vm_invoke(loom_serve_program_invocation(program), function,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_empty());
    // Wrapped host storage stays live until all VM references are released.
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    IREE_ASSERT_OK(status);
    EXPECT_EQ(actual, expected);
  }

  // Host policy shared by the source program and borrowed buffer wrappers.
  const iree_allocator_t allocator = iree_allocator_system();
  // Environment outliving the VM and argument references.
  iree_vm_environment_t* environment = nullptr;
  // Canonical buffer reference types borrowed from environment.
  iree_vm_ref_types_t types = {};
  // Shared source function lowered to the actual VM oracle.
  loom_serve_program_t* program = nullptr;
  // Resolved production function accepting context and three mutable buffers.
  iree_vm_function_t function = {};
  // Arguments own wrappers while the test retains their backing vectors.
  iree_vm_variant_t arguments[4] = {};
};

TEST_F(ContinuationTest, CompactsLiveSpansAndPreservesFeedback) {
  for (int count = 0; count <= 16; ++count) {
    for (int trial = 0; trial < 15; ++trial) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(trial);
      ASSERT_NO_FATAL_FAILURE(Run(count, trial));
    }
  }
}

}  // namespace
