// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_ENCODING_MEMORY_COMMANDS_H_
#define AMDF_CTS_GPU_PM4_ENCODING_MEMORY_COMMANDS_H_

#include <cstddef>
#include <cstdint>

// Ordinary memory comparisons used by the CTS. These are the MEC
// WAIT_REG_MEM/WAIT_REG_MEM64 and COND_INDIRECT_BUFFER function values, not
// host comparison opcodes.
enum class Pm4MemoryComparison : uint32_t {
  kAlways = 0,
  kLess = 1,
  kLessOrEqual = 2,
  kEqual = 3,
  kNotEqual = 4,
  kGreaterOrEqual = 5,
  kGreater = 6,
};

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

// Encodes an ordinary MEC memory wait into seven or nine caller-owned DWORDs
// and returns that count. The naturally aligned target has separately qualified
// WAIT semantics. The comparison uses (loaded_value & mask) versus value.
// No ACE offload or payload-cache operation is performed.
size_t WaitMemory32(uint32_t* words, uint64_t address, uint32_t value,
                    Pm4MemoryComparison comparison, uint32_t mask);
size_t WaitMemory64(uint32_t* words, uint64_t address, uint64_t value,
                    Pm4MemoryComparison comparison, uint64_t mask);

// Encodes an immediate-to-TC/L2 COPY_DATA into six caller-owned DWORDs and
// returns six. Confirmation orders this CP write before following CP commands;
// it performs no cache action, shader wait, or read-modify-write.
// The address is naturally aligned for width; k32Bit consumes a zero-extended
// uint32_t value. Atomicity and participant reach belong to the target mapping,
// independently of this packet's width. Other payload visibility is explicit.
size_t StoreImmediate(uint32_t* words, uint64_t target_address, uint64_t value,
                      CopyDataWidth width);

// Encodes one single-pass ATOMIC_MEM ADD_RTN into nine caller-owned DWORDs
// and returns nine. The naturally aligned target has separately qualified ADD
// semantics and participant reach. RTN orders the operation before following CP
// commands; the returned value is discarded. No payload cache work is
// performed.
size_t AtomicAdd32(uint32_t* words, uint64_t target_address, uint32_t value);
size_t AtomicAdd64(uint32_t* words, uint64_t target_address, uint64_t value);

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
