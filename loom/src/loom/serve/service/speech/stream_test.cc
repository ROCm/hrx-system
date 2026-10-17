// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/stream.h"

#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct loom_serve_speech_test_allocator_t {
  // Successful new allocations permitted before reporting memory pressure.
  size_t remaining = 0;
  // Live allocations owned by this allocator, excluding request text storage.
  size_t live = 0;

  iree_allocator_t allocator() {
    return {this, +[](void* self, iree_allocator_command_t command,
                      const void* parameters, void** pointer) {
              auto* state =
                  static_cast<loom_serve_speech_test_allocator_t*>(self);
              if (command != IREE_ALLOCATOR_COMMAND_FREE && !state->remaining) {
                return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              const bool freeing =
                  command == IREE_ALLOCATOR_COMMAND_FREE && *pointer;
              const auto system = iree_allocator_system();
              const auto status =
                  system.ctl(system.self, command, parameters, pointer);
              if (iree_status_is_ok(status)) {
                if (freeing) {
                  --state->live;
                } else if (command != IREE_ALLOCATOR_COMMAND_FREE) {
                  --state->remaining;
                  ++state->live;
                }
              }
              return status;
            }};
  }
};

TEST(SpeechStreamTest, InputAllocationFailurePreservesTheCallerRequest) {
  for (size_t permitted : {0u, 1u}) {
    loom_serve_speech_request_t request;
    IREE_ASSERT_OK(loom_serve_speech_request_initialize_text(
        IREE_SV("retained"), IREE_SV("positive"), {}, 13, 1.0f,
        iree_allocator_system(), &request));
    const char* bytes = iree_string_builder_buffer(&request.input);
    loom_serve_speech_test_allocator_t allocator = {.remaining = permitted};
    loom_serve_speech_stream_t* stream = nullptr;
    loom_serve_speech_input_t* input = nullptr;
    loom_serve_speech_consumer_t consumer = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_speech_stream_allocate({24000, 1}, 5, 7, 1024, &request,
                                          allocator.allocator(), &stream,
                                          &input, &consumer));
    EXPECT_EQ(stream, nullptr);
    EXPECT_EQ(input, nullptr);
    EXPECT_EQ(consumer.self, nullptr);
    EXPECT_EQ(iree_string_builder_buffer(&request.input), bytes);
    EXPECT_EQ(iree_string_builder_size(&request.input), 8u);
    EXPECT_EQ(request.seed, 13u);
    EXPECT_EQ(allocator.live, 0u);
    loom_serve_speech_request_deinitialize(&request);
  }
}

TEST(SpeechStreamTest, FiniteTextKeepsItsOriginalAllocatorAcrossDetachment) {
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize_text(
      IREE_SV("retained"), IREE_SV("positive"), {}, 13, 1.0f,
      iree_allocator_system(), &request));
  const char* bytes = iree_string_builder_buffer(&request.input);
  loom_serve_speech_test_allocator_t allocator = {.remaining = 2};
  loom_serve_speech_stream_t* stream = nullptr;
  loom_serve_speech_input_t* input = nullptr;
  loom_serve_speech_consumer_t consumer = {};
  IREE_ASSERT_OK(loom_serve_speech_stream_allocate(
      {24000, 1}, 5, 7, 1024, &request, allocator.allocator(), &stream, &input,
      &consumer));
  EXPECT_EQ(iree_string_builder_size(&request.input), 0u);
  EXPECT_EQ(loom_serve_speech_input_peek(input).spans[0].data,
            reinterpret_cast<const uint8_t*>(bytes));
  EXPECT_EQ(allocator.live, 2u);
  IREE_EXPECT_OK(consumer.end(consumer.self, LOOM_SERVE_TELEMETRY_COMPLETED,
                              iree_ok_status()));
  loom_serve_speech_input_release(input);
  EXPECT_EQ(allocator.live, 1u);
  EXPECT_TRUE(loom_serve_speech_stream_is_complete(stream));
  EXPECT_EQ(loom_serve_speech_stream_write_credit(stream), 0u);
  loom_serve_speech_stream_close(stream);
  EXPECT_EQ(allocator.live, 0u);
  loom_serve_speech_request_deinitialize(&request);
}

}  // namespace
