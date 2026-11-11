// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_CROSS_PROCESS_PROGRESS_H_
#define AMDF_CTS_GPU_KERNELS_CROSS_PROCESS_PROGRESS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernels::cross_process_progress {

inline constexpr uint32_t kMultiplier = 1664525u;
inline constexpr uint32_t kIncrement = 1013904223u;

// Host-visible coordination and the shader's exact terminal result. The
// surrounding guards make every unintended neighboring write observable.
struct alignas(64) Control {
  // Words preserved before the live protocol fields.
  uint32_t leading_guard[4];
  // Shader release publication after it begins the finite arithmetic loop.
  uint32_t started;
  // Host release request acquired by the shader on each iteration.
  uint32_t release;
  // Number of completed recurrence iterations.
  uint32_t iteration_count;
  // Exact recurrence value after iteration_count updates.
  uint32_t result;
  // Words preserved after the live protocol fields.
  uint32_t trailing_guard[8];
};
static_assert(sizeof(Control) == 64);

struct Arguments {
  // Device address of the complete guarded Control record.
  uint64_t control;
  // Device address of the first word in the bulk-readback payload.
  uint64_t payload;
  // Maximum recurrence iterations when the host cannot release the shader.
  uint32_t iteration_limit;
  // Initial recurrence state.
  uint32_t seed;
};
static_assert(sizeof(Arguments) == 24);

inline constexpr std::array<uint32_t, 4> kArgumentByteOffsets = {
    offsetof(Arguments, control), offsetof(Arguments, payload),
    offsetof(Arguments, iteration_limit), offsetof(Arguments, seed)};
inline constexpr std::array<uint32_t, 4> kArgumentByteLengths = {8, 8, 4, 4};
inline constexpr std::array<std::string_view, 4> kArgumentValueKinds = {
    "global_buffer", "global_buffer", "by_value", "by_value"};

}  // namespace kernels::cross_process_progress

#endif  // AMDF_CTS_GPU_KERNELS_CROSS_PROCESS_PROGRESS_H_
