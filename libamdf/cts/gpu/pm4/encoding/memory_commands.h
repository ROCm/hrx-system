// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_ENCODING_MEMORY_COMMANDS_H_
#define AMDF_CTS_GPU_PM4_ENCODING_MEMORY_COMMANDS_H_

#include <cstddef>
#include <cstdint>

namespace pm4 {

// COPY_DATA count selection. A transfer width does not imply atomicity.
enum class CopyDataWidth : uint32_t {
  k32Bit = 0,
  k64Bit = 1,
};

// COPY_DATA locality hint in cache-policy and temporal-policy encodings.
enum class CopyDataPolicy : uint32_t {
  kDefault = 0,
  kStreaming = 1,
};

// RELEASE_MEM immediate-data selection for one naturally aligned update.
enum class ReleaseDataWidth : uint32_t {
  k32Bit = 1,
  k64Bit = 2,
};

// Encodes a confirmed bottom-of-pipe RELEASE_MEM write into eight caller-owned
// DWORDs and returns eight. It performs no cache action or read-modify-write.
// The address is naturally aligned for width; k32Bit consumes a zero-extended
// uint32_t value. Atomicity and participant reach belong to the target mapping,
// independently of this packet's width. Other payload visibility is explicit.
size_t ReleaseImmediate(uint32_t* words, uint64_t target_address,
                        uint64_t value, ReleaseDataWidth width);

// Encodes one confirmed TC/L2-to-TC/L2 copy into six caller-owned DWORDs and
// returns six. Both addresses are aligned to four bytes for k32Bit and eight
// bytes for k64Bit. The k32Bit result width does not bound native source reads;
// readable trailing backing belongs to the source's lifetime. Queue admission
// and execution visibility belong to the caller.
size_t CopyData(uint32_t* words, uint64_t source_address,
                uint64_t target_address, CopyDataWidth width);

// Encodes a confirmed 64-bit GPU-clock COPY_DATA with MEMORY destination into
// six caller-owned DWORDs and returns six. The target is eight-byte aligned.
// Admission, XCC selection, visibility and completion belong to the caller;
// sampling CP progress is not a shader fence and supplies no clock-frequency
// or correlation information.
size_t CopyGpuClock64(uint32_t* words, uint64_t target_address,
                      CopyDataPolicy policy = CopyDataPolicy::kDefault);

// Encodes a confirmed, incrementing TC/L2 write and returns 4 + value_count.
// The target is four-byte aligned. The caller supplies 1..16381 payload DWORDs
// and sufficient output storage that does not overlap values. Queue admission
// and execution visibility belong to the caller.
size_t WriteData(uint32_t* words, uint64_t target_address,
                 const uint32_t* values, size_t value_count);

}  // namespace pm4

#endif  // AMDF_CTS_GPU_PM4_ENCODING_MEMORY_COMMANDS_H_
