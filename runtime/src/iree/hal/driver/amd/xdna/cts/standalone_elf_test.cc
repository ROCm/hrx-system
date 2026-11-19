// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/xdna/cts/native_test_base.h"
#include "iree/hal/driver/amd/xdna/image/testdata/aie/full_elf_42.h"
#include "iree/hal/driver/amd/xdna/image/testdata/aie/vector_scalar_add.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::xdna::cts {
namespace {

TEST_F(XdnaNativeTest, RunsStandaloneHelloWorld) {
  const auto* file = iree_hal_amd_xdna_test_full_elf_42_create();
  iree::Status status(
      LoadElf(*file, IREE_SV("main:sequence"), &executable_, &function_));
  if (status.code() == iree::StatusCode::kIncompatible) {
    ASSERT_EQ(executable_, nullptr);
    GTEST_SKIP() << status.ToString();
  }
  IREE_ASSERT_OK(status);
  iree_hal_buffer_t* output = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&output));
  iree_hal_semaphore_t* done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  uint64_t value = 1;
  iree_hal_buffer_ref_t binding =
      iree_hal_make_buffer_ref(output, 0, sizeof(uint32_t));
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue_, {}, {1, &done, &value}, executable_, function_,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), {1, &binding}, nullptr,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  std::array<uint32_t, 3 * kBytes / sizeof(uint32_t)> words;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(iree_hal_buffer_allocated_buffer(output), 0,
                               words.data(), sizeof(words)));
  for (size_t i = 0; i < words.size(); ++i) {
    EXPECT_EQ(words[i], i == kBytes / sizeof(uint32_t) ? 42u : 0xA5A5A5A5u)
        << i;
  }
}

TEST_F(XdnaNativeTest, RunsStandaloneVectorScalarAddAcrossNativeDispatch) {
  const auto* file = iree_hal_amd_xdna_test_vector_scalar_add_create();
  iree::Status status(
      LoadElf(*file, IREE_SV("main:sequence"), &executable_, &function_));
  if (status.code() == iree::StatusCode::kIncompatible) {
    ASSERT_EQ(executable_, nullptr);
    GTEST_SKIP() << status.ToString();
  }
  IREE_ASSERT_OK(status);
  iree_hal_semaphore_t* done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  constexpr size_t kElementCount = 1024;
  std::array<uint32_t, kElementCount> input;
  for (uint64_t pass = 1; pass <= 2; ++pass) {
    uint64_t value = 2 * pass - 1;
    iree_hal_buffer_t *source = nullptr, *destination = nullptr;
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(sizeof(input), &source));
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(sizeof(input), &destination));
    // The first dispatch uses FastFlow's published 1..1024 input. Change both
    // addresses and the data on reuse so a stale command cannot pass.
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = (uint32_t)i + (pass == 1 ? 1u : 0x98760000u);
    }
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(source, 0, input.data(), sizeof(input)));
    iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(source, 0, sizeof(input)),
        iree_hal_make_buffer_ref(destination, 0, sizeof(input))};
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue_, {}, {1, &done, &value}, executable_, function_,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
        nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    std::array<uint32_t, 3 * kElementCount> readback;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(iree_hal_buffer_allocated_buffer(destination),
                                 0, readback.data(), sizeof(readback)));
    for (size_t i = 0; i < input.size(); ++i) {
      ASSERT_EQ(readback[kElementCount + i], input[i] + 1) << i;
      ASSERT_EQ(readback[i], 0xA5A5A5A5u) << i;
      ASSERT_EQ(readback[2 * kElementCount + i], 0xA5A5A5A5u) << i;
    }
    if (pass == 1) {
      // Reconfigure the same placement with a native Loom program between
      // standalone submissions. Both formats must establish their own state.
      ASSERT_NO_FATAL_FAILURE(LoadProgram(
          iree_hal_amd_xdna_test_mul_i32_create(),
          iree_hal_amd_xdna_test_mul_i32_npu4_create(), IREE_SV("mul_i32"),
          &alternate_executable_, &alternate_function_));
      iree_hal_buffer_ref_t native_bindings[] = {
          iree_hal_make_buffer_ref(source, 0, kBytes),
          iree_hal_make_buffer_ref(source, 0, kBytes),
          iree_hal_make_buffer_ref(destination, 0, kBytes)};
      uint64_t native_value = value + 1;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue_, {}, {1, &done, &native_value}, alternate_executable_,
          alternate_function_, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(),
          {IREE_ARRAYSIZE(native_bindings), native_bindings}, nullptr,
          IREE_HAL_DISPATCH_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(done, native_value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      std::array<uint32_t, kBytes / sizeof(uint32_t)> squares;
      IREE_ASSERT_OK(iree_hal_buffer_map_read(destination, 0, squares.data(),
                                              sizeof(squares)));
      for (size_t i = 0; i < squares.size(); ++i) {
        EXPECT_EQ(squares[i], input[i] * input[i]) << i;
      }
    }
  }
}

TEST_F(XdnaNativeTest, ReplaysPendingStandaloneBatchesAcrossExecutableReload) {
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  const auto& image = *iree_hal_amd_xdna_test_vector_scalar_add_create();
  constexpr size_t kWordCount = 1024;
  constexpr size_t kVectorBytes = kWordCount * sizeof(uint32_t);
  constexpr uint32_t kAdds = 8;
  for (uint32_t generation = 0; generation < 2; ++generation) {
    iree::Status status(LoadElf(image, IREE_SV("main:sequence"),
                                &alternate_executable_, &alternate_function_));
    if (generation == 0 && status.code() == iree::StatusCode::kIncompatible) {
      ASSERT_EQ(alternate_executable_, nullptr);
      GTEST_SKIP() << status.ToString();
    }
    IREE_ASSERT_OK(status);
    iree::hal::cts::Ref<iree_hal_command_buffer_t> recorded;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 2, recorded.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recorded));
    for (uint32_t i = 0; i < kAdds; ++i) {
      if (i) {
        IREE_ASSERT_OK(BarrierBetweenDispatches(recorded));
      }
      const iree_hal_buffer_ref_t arguments[] = {
          iree_hal_make_indirect_buffer_ref(i % 2, 0, kVectorBytes),
          iree_hal_make_indirect_buffer_ref((i + 1) % 2, 0, kVectorBytes)};
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recorded, alternate_executable_, alternate_function_,
          iree_hal_make_static_dispatch_config(1, 1, 1), {},
          {IREE_ARRAYSIZE(arguments), arguments}, 0));
    }
    // Loom squares the first 16 words produced by the final FastFlow dispatch.
    IREE_ASSERT_OK(BarrierBetweenDispatches(recorded));
    const auto result = iree_hal_make_indirect_buffer_ref(kAdds % 2, 0, kBytes);
    const iree_hal_buffer_ref_t square[] = {result, result, result};
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recorded, executable_, function_,
        iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {IREE_ARRAYSIZE(square), square}, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recorded));

    iree_hal_semaphore_t* start = nullptr;
    ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&start));
    uint64_t one = 1;
    std::array<std::array<iree_hal_buffer_t*, 2>, 2> buffers = {};
    std::array<iree_hal_semaphore_t*, 2> done = {};
    const auto input_word = [generation](size_t batch, size_t word) {
      return static_cast<uint32_t>((generation * 2 + batch) * kWordCount +
                                   word);
    };
    for (size_t batch = 0; batch < buffers.size(); ++batch) {
      for (auto& buffer : buffers[batch]) {
        ASSERT_NO_FATAL_FAILURE(MakeBuffer(kVectorBytes, &buffer));
      }
      ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done[batch]));
      std::array<uint32_t, kWordCount> input;
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = input_word(batch, i);
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[batch][0], 0,
                                               input.data(), kVectorBytes));
    }
    iree_status_t submit_status = iree_ok_status();
    for (size_t batch = 0;
         batch < buffers.size() && iree_status_is_ok(submit_status); ++batch) {
      const iree_hal_buffer_binding_t table[] = {
          {buffers[batch][0], 0, kVectorBytes},
          {buffers[batch][1], 0, kVectorBytes}};
      submit_status = iree_hal_queue_execute(queue_, {1, &start, &one},
                                             {1, &done[batch], &one}, recorded,
                                             {IREE_ARRAYSIZE(table), table}, 0);
    }
    // Release the recording and image while both binding tables are pending.
    // The next generation reloads the image after these executions retire.
    recorded.reset();
    iree_hal_executable_release(alternate_executable_);
    alternate_executable_ = nullptr;
    IREE_ASSERT_OK(iree_status_join(
        submit_status, iree_hal_semaphore_signal(start, one, nullptr)));
    for (size_t batch = 0; batch < buffers.size(); ++batch) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(done[batch], one,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      std::array<uint32_t, 3 * kWordCount> words;
      for (size_t slot = 0; slot < buffers[batch].size(); ++slot) {
        IREE_ASSERT_OK(iree_hal_buffer_map_read(
            iree_hal_buffer_allocated_buffer(buffers[batch][slot]), 0,
            words.data(), sizeof(words)));
        for (size_t i = 0; i < kWordCount; ++i) {
          uint32_t expected = input_word(batch, i) + kAdds;
          if (slot != kAdds % 2) {
            --expected;
          } else if (i < kBytes / sizeof(uint32_t)) {
            expected *= expected;
          }
          ASSERT_EQ(words[kWordCount + i], expected)
              << generation << "/" << batch << "/" << slot << "/" << i;
          ASSERT_EQ(words[i], 0xA5A5A5A5u);
          ASSERT_EQ(words[2 * kWordCount + i], 0xA5A5A5A5u);
        }
      }
    }
  }
}

}  // namespace
}  // namespace iree::hal::amd::xdna::cts
