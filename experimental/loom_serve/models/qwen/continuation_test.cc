// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"

IREE_FLAG(string, continuation_source, "", "Production continuation source.");

namespace {

class ContinuationTest : public ::testing::Test {
 protected:
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation = {};

  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    const iree_string_view_t roots[] = {IREE_SVL("qwen38_continue_spans")};
    IREE_ASSERT_OK(loom_serve_program_create(
        environment, iree_make_cstring_view(FLAG_continuation_source), 1, roots,
        iree_vm_module_span_empty(), &compilation, allocator, &program));
    IREE_ASSERT_OK(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), roots[0], &function));
  }

  void TearDown() override {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    loom_serve_program_destroy(program);
    iree_vm_environment_free(environment);
  }

  void Run(int count, int trial, int width) {
    constexpr int kContext = 1024;
    constexpr int kEos = 42;
    std::array<std::vector<int32_t>, 3> actual = {
        std::vector<int32_t>(1176, -99), std::vector<int32_t>(1024, -77),
        std::vector<int32_t>(336, -55)};
    auto& metadata = actual[0];
    auto& results = actual[2];
    metadata[0] = count * width;
    metadata[1] = count;
    metadata[2] = count * width;
    metadata[579] = kEos;
    metadata[581] = count;
    int queued_count = 0;
    for (int slot = 0; slot < count; ++slot) {
      // Rotation and reversal ensure source records cannot be recovered from
      // the compact slot or from descriptors overwritten earlier in the loop.
      const int first = trial % 2 ? count - 1 - slot : (slot + trial) % count;
      const int kind = trial >= 14 ? (trial - 14) % 7 : (first + trial) % 7;
      const bool known = kind == 1 || kind == 6;
      const bool promoted = kind == 5;
      const int length =
          known ? (count == 1 ? 512 : 1 + (slot * 11 + trial) % 24) : width;
      const int consumed =
          known || promoted ? 13 + first : 1 + (first + trial) % width;
      const int emitted = known ? 0 : promoted ? 1 : consumed;
      const int position = kind == 4   ? kContext - consumed - (width - 1)
                           : trial % 2 ? kContext - consumed - length
                                       : 100 + first;
      const int credit = known       ? 0
                         : kind == 3 ? emitted
                         : trial % 2 ? emitted + 1
                                     : 2 * width;
      const int fields[] = {length, position, queued_count, 15 - first,
                            kind == 1 ? -1 : 0};
      std::memcpy(metadata.data() + 583 + slot * 5, fields, sizeof(fields));
      metadata[1031 + slot * 2] = known ? 0 : 2;
      metadata[1032 + slot * 2] = credit;
      metadata[1160 + slot] = first;
      results[first * 10] = consumed;
      results[first * 10 + 1] = emitted;
      for (int token = 0; token < 8; ++token) {
        results[first * 10 + 2 + token] =
            kind == 2 && token == emitted - 1 ? kEos : 1000 + first * 8 + token;
      }
      for (int token = 0; token < length; ++token) {
        actual[1][512 + queued_count + token] = 10000 + queued_count + token;
      }
      queued_count += length;
    }
    auto expected = actual;
    std::fill(expected[1].begin(), expected[1].begin() + 512, 0);
    std::fill(expected[2].begin() + 160, expected[2].begin() + 320, 0);
    std::fill(expected[2].begin() + 320, expected[2].end(), -1);
    int live_count = 0;
    int input_count = 0;
    int output_count = 0;
    for (int plan = 0; plan < count; ++plan) {
      const int first = metadata[1160 + plan];
      const int* descriptor = metadata.data() + 583 + plan * 5;
      const int consumed = results[first * 10];
      const int emitted = results[first * 10 + 1];
      const bool speculative = metadata[1031 + plan * 2] != 0;
      const int prediction =
          speculative ? results[first * 10 + emitted + 1] : 0;
      const int credit = metadata[1032 + plan * 2] - emitted;
      const int position = descriptor[1] + consumed;
      if (speculative &&
          (prediction == kEos || credit <= 0 || position + width > kContext)) {
        continue;
      }
      const int selected = descriptor[4] >= 0 ? (speculative ? width : 1) : 0;
      const int fields[] = {descriptor[0], position, input_count, descriptor[3],
                            selected ? output_count : -1};
      std::memcpy(expected[0].data() + 3 + live_count * 5, fields,
                  sizeof(fields));
      expected[0][451 + live_count * 2] = speculative ? 2 : 0;
      expected[0][452 + live_count * 2] = credit;
      if (speculative) {
        expected[1][input_count] = prediction;
      } else {
        std::copy_n(actual[1].begin() + 512 + descriptor[2], descriptor[0],
                    expected[1].begin() + input_count);
      }
      for (int token = 0; token < selected; ++token) {
        expected[0][323 + output_count + token] =
            input_count + descriptor[0] - selected + token;
      }
      expected[2][320 + live_count] = first;
      ++live_count;
      input_count += descriptor[0];
      output_count += selected;
    }
    expected[0][0] = input_count;
    expected[0][1] = live_count;
    expected[0][2] = output_count;
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
    for (int trial = 0; trial < 28; ++trial) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(trial);
      for (int width : {4, 8}) {
        SCOPED_TRACE(width);
        ASSERT_NO_FATAL_FAILURE(Run(count, trial, width));
      }
    }
  }
}

}  // namespace
