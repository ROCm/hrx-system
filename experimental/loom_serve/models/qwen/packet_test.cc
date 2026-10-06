// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "iree/vm/test_allocator.h"

IREE_FLAG(string, control_source, "", "Production token control source.");

namespace {

class PacketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    const iree_string_view_t roots[] = {IREE_SVL("encode_epoch"),
                                        IREE_SVL("publish_epoch")};
    IREE_ASSERT_OK(loom_serve_program_create(
        environment, iree_make_cstring_view(FLAG_control_source), 2, roots,
        iree_vm_module_span_empty(), allocator, &program));
    IREE_ASSERT_OK(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), roots[0], &encode));
    IREE_ASSERT_OK(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), roots[1], &publish));
  }

  void ReleaseBuffers() {
    for (auto*& buffer : buffers) {
      iree_vm_buffer_release(buffer);
      buffer = nullptr;
    }
  }

  void TearDown() override {
    ReleaseBuffers();
    loom_serve_program_destroy(program);
    iree_vm_environment_free(environment);
  }

  iree_vm_variant_t Buffer(size_t index) {
    return iree_vm_buffer_variant_from_ptr_borrowed(&types, buffers[index]);
  }

  void Run(int count, int trial) {
    ReleaseBuffers();
    const bool verifies = trial >= 12 || trial % 3 != 0;
    const int next_count =
        verifies && trial >= 3 ? (trial % 2 ? count : (count + 1) / 2) : 0;
    storage = {std::vector<int32_t>(256, -1),
               std::vector<int32_t>(1024, -2),
               std::vector<int32_t>(1048, -3),
               std::vector<int32_t>(1024, -4),
               std::vector<int32_t>(16, -5),
               std::vector<int32_t>(verifies ? 96 : 0, -6),
               std::vector<int32_t>(next_count ? 112 : 0, -7),
               std::vector<int32_t>(192, -8)};
    auto& plans = storage[0];
    int input_count = 0;
    for (int i = 0; i < count; ++i) {
      const int kind =
          trial >= 12 ? trial - 12 : (i + trial) % (verifies ? 4 : 2);
      const bool speculative = kind >= 2;
      const int length = speculative ? 4 : count == 1 ? 512 : 1 + i * 7 % 24;
      const int known = kind == 3 ? 1 : length;
      const int flags = kind == 0 ? 0 : kind == 3 ? 3 : 1;
      const int values[] = {length,
                            100 + i * 41,
                            15 - i,
                            input_count,
                            known,
                            flags,
                            speculative ? (next_count ? 8 : 4) : 0,
                            i};
      std::copy_n(values, 8, plans.data() + i * 8);
      for (int j = 0; j < known; ++j) {
        storage[1][input_count++] = 1000 + i * 100 + j;
      }
    }
    uint32_t next_speculative_mask = 0;
    for (int i = 0; i < next_count; ++i) {
      const int first = trial % 2 ? next_count - 1 - i : (i + trial) % count;
      const int* prior = plans.data() + first * 8;
      const bool speculative = (prior[5] & 1) != 0;
      const int length = speculative ? 4 : count == 1 ? 512 : 1 + i * 11 % 24;
      const int values[] = {length,
                            prior[1],
                            prior[2],
                            input_count,
                            speculative ? 0 : length,
                            speculative ? 3 : i % 2,
                            speculative ? (prior[6] ? 8 : 5) : 0,
                            first};
      std::copy_n(values, 8, plans.data() + (16 + i) * 8);
      if (speculative) {
        next_speculative_mask |= 1u << first;
      } else {
        for (int j = 0; j < length; ++j) {
          storage[1][input_count++] = 20000 + i * 100 + j;
        }
      }
    }

    auto expected = storage;
    std::fill(expected[3].begin(), expected[3].end(), 0);
    int first_outputs = 0;
    for (int bank = 0; bank < (next_count ? 2 : 1); ++bank) {
      int tokens = 0;
      int outputs = 0;
      const int bank_count = bank ? next_count : count;
      int32_t* metadata = expected[2].data() + bank * 516;
      for (int i = 0; i < bank_count; ++i) {
        const int* request = plans.data() + (bank * 16 + i) * 8;
        const int selected = request[5] & 1 ? (request[6] ? 4 : 1) : 0;
        const int values[] = {request[0], request[1], tokens, request[2],
                              selected ? outputs : -1};
        std::copy_n(values, 5, metadata + 3 + i * 5);
        std::copy_n(storage[1].data() + request[3], request[4],
                    expected[3].data() + bank * 512 + tokens);
        tokens += request[0];
        for (int j = 0; j < selected; ++j) {
          metadata[323 + outputs++] = tokens - selected + j;
        }
        metadata[387 + i * 2] = request[5] & 2 ? 2 : request[6] != 0;
        metadata[388 + i * 2] = request[6];
        if (bank) {
          expected[2][1032 + i] = request[7];
        }
      }
      metadata[0] = tokens;
      metadata[1] = bank_count;
      metadata[2] = outputs;
      metadata[515] = 42;
      if (!bank) {
        first_outputs = outputs;
      }
    }

    std::fill(expected[7].begin(), expected[7].end(), 0);
    int ordinary_index = 0;
    for (int i = 0; i < count; ++i) {
      const int* request = plans.data() + i * 8;
      const bool speculative = request[6] != 0;
      const int consumed = speculative ? 1 + (i + trial) % 4 : request[0];
      const int outputs = speculative ? consumed : request[5] & 1;
      int32_t* progress = expected[7].data() + i * 12;
      progress[0] = consumed;
      progress[1] = speculative ? 0 : consumed;
      progress[2] = outputs;
      progress[3] = speculative;
      if (verifies) {
        storage[5][i * 6] = consumed;
        storage[5][i * 6 + 1] = outputs;
        for (int j = 0; j < outputs; ++j) {
          progress[4 + j] = storage[5][i * 6 + 2 + j] = 3000 + i * 10 + j;
        }
      } else if (outputs) {
        progress[4] = storage[4][ordinary_index++] = 4000 + i;
      }
    }
    int live = 0;
    if (next_count) {
      std::fill(storage[6].begin() + 96, storage[6].end(), -1);
    }
    for (int i = 0; i < next_count; ++i) {
      if ((i + trial) % 4 == 0) {
        continue;
      }
      const int* request = plans.data() + (16 + i) * 8;
      const bool speculative = request[6] != 0;
      const int consumed = speculative ? 1 + (i + trial) % 4 : request[0];
      const int outputs = speculative ? consumed : request[5] & 1;
      int32_t* progress = expected[7].data() + request[7] * 12;
      storage[6][live * 6] = consumed;
      storage[6][live * 6 + 1] = outputs;
      storage[6][96 + live] = request[7];
      for (int j = 0; j < outputs; ++j) {
        progress[4 + progress[2] + j] = storage[6][live * 6 + 2 + j] =
            5000 + i * 10 + j;
      }
      progress[0] += consumed;
      progress[1] += speculative ? 0 : consumed;
      progress[2] += outputs;
      progress[3] += speculative;
      ++live;
    }
    for (size_t i : {4, 5, 6}) {
      expected[i] = storage[i];
    }
    for (size_t i = 0; i < storage.size(); ++i) {
      IREE_ASSERT_OK(iree_vm_buffer_wrap(
          IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
          iree_make_byte_span(storage[i].data(), storage[i].size() * 4),
          iree_vm_buffer_release_callback_null(), allocator, &buffers[i]));
    }
    iree_vm_variant_t arguments[] = {Buffer(0),
                                     Buffer(1),
                                     iree_vm_variant_from_i32(count),
                                     iree_vm_variant_from_i32(next_count),
                                     iree_vm_variant_from_i32(42),
                                     Buffer(2),
                                     Buffer(3)};
    iree_vm_variant_t results[1] = {};
    const auto allocation_count = counting_allocator.allocation_count();
    const auto free_count = counting_allocator.free_count();
    iree_status_t status =
        iree_vm_invoke(loom_serve_program_invocation(program), encode,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    int32_t outputs = 0;
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(results[0], &outputs);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    EXPECT_EQ(outputs, first_outputs);
    iree_vm_variant_t publication[] = {
        Buffer(0),
        iree_vm_variant_from_i32(count),
        iree_vm_variant_from_i32(verifies),
        iree_vm_variant_from_i32(next_count != 0),
        iree_vm_variant_from_i32(next_speculative_mask),
        Buffer(4),
        Buffer(5),
        Buffer(6),
        Buffer(7)};
    status = iree_vm_invoke(loom_serve_program_invocation(program), publish,
                            iree_vm_variant_span_from_array(publication),
                            iree_vm_variant_span_empty());
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(publication));
    IREE_ASSERT_OK(status);
    EXPECT_EQ(counting_allocator.allocation_count(), allocation_count);
    EXPECT_EQ(counting_allocator.free_count(), free_count);
    EXPECT_EQ(storage, expected);
  }

  // Tracks allocator activity across the actual warm VM invocations.
  iree::vm::testing::CountingAllocator counting_allocator;
  // Host allocator shared by the VM and cold borrowed-storage wrappers.
  const iree_allocator_t allocator = counting_allocator.allocator();
  // Environment outliving every buffer reference and the source program.
  iree_vm_environment_t* environment = nullptr;
  // Canonical buffer reference identity from environment.
  iree_vm_ref_types_t types = {};
  // Shared source transforms without native execution imports.
  loom_serve_program_t* program = nullptr;
  // Production packet encoder.
  iree_vm_function_t encode = {};
  // Production completed-feedback publisher.
  iree_vm_function_t publish = {};
  // Host plans, known IDs, opaque upload/feedback banks and semantic progress.
  std::array<std::vector<int32_t>, 8> storage;
  // Wrappers borrowing storage until the next cold case initialization.
  std::array<iree_vm_buffer_t*, 8> buffers = {};
};

TEST_F(PacketTest, PreservesPacketLayoutAndPublishesSemanticProgress) {
  for (int count = 1; count <= 16; ++count) {
    for (int trial = 0; trial < 16; ++trial) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(trial);
      ASSERT_NO_FATAL_FAILURE(Run(count, trial));
    }
  }
}

}  // namespace
