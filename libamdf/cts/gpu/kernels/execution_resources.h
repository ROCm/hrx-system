// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_EXECUTION_RESOURCES_H_
#define AMDF_CTS_GPU_KERNELS_EXECUTION_RESOURCES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernels::execution_resources {

// Each workgroup writes its physical scheduling domain and numerical witness.
struct Record {
  // Physical CU/WGP identity including shader engine and XCC, excluding slots.
  uint32_t identity;
  // Workgroup index XOR the supplied epoch.
  uint32_t value;
};
static_assert(sizeof(Record) == 8);

struct alignas(16) Arguments {
  // Device address of one Record per launched workgroup.
  uint64_t output;
  // Changes the numerical witness between completed dispatches.
  uint32_t epoch;
};
static_assert(sizeof(Arguments) == 16);

inline constexpr std::array<uint32_t, 2> kArgumentByteOffsets = {
    offsetof(Arguments, output), offsetof(Arguments, epoch)};
inline constexpr std::array<uint32_t, 2> kArgumentByteLengths = {
    sizeof(Arguments::output), sizeof(Arguments::epoch)};
inline constexpr std::array<std::string_view, 2> kArgumentValueKinds = {
    "global_buffer", "by_value"};

}  // namespace kernels::execution_resources

#endif  // AMDF_CTS_GPU_KERNELS_EXECUTION_RESOURCES_H_
