// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_
#define AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_

#include <cstddef>
#include <cstdint>

#include "libamdf/cts/gpu/pm4/encoding/profile.h"

// Ordinary memory comparisons used by the CTS. These are the MEC
// WAIT_REG_MEM/WAIT_REG_MEM64 function values, not host comparison opcodes.
enum class Pm4MemoryComparison : uint32_t {
  kLess = 1,
  kEqual = 3,
  kNotEqual = 4,
  kGreaterOrEqual = 5,
};

// Compiled RDNA wave32 program with no scratch or hidden runtime inputs. The
// caller supplies only a kernarg pointer; hardware supplies group/local IDs.
struct Pm4ComputeProgram {
  // GPU entry address, aligned to 256 bytes and below the 48-bit program limit.
  uint64_t entry_address;
  // Compiler COMPUTE_PGM_RSRC1, without runtime instrumentation overrides.
  uint32_t resource1;
  // Immutable compiler COMPUTE_PGM_RSRC2 with its user SGPR and ID enables.
  // Binding replaces its descriptor-zero LDS field with the allocation below.
  uint32_t resource2;
  // Compiler COMPUTE_PGM_RSRC3, including its backed instruction-prefetch
  // extent.
  uint32_t resource3;
  // Total group-segment allocation in bytes, within the target's group limit.
  uint32_t group_segment_byte_length;
  // Complete workgroup dimensions in workitems, matching the compiled program.
  uint32_t workgroup_size[3];
};

// Encodes the selected RDNA compute recipe using PM4 format version 1.
// Native callers admit the target and corresponding packet, transfer and
// cache-control requirements before constructing a stream. Callers supply
// sufficient storage and addresses aligned to four bytes for 32-bit operations
// and eight bytes for 64-bit operations.
class Pm4CommandWriter {
 public:
  Pm4CommandWriter(uint32_t* words, const Pm4CommandProfile& profile)
      : words_(words), profile_(profile) {}

  // Joins preceding compute waves and acquires code and data from SYSTEM.
  void SystemBarrier();
  // Acquires SYSTEM data while retaining instruction-cache contents. The
  // caller has already published immutable code and joined preceding waves
  // through an independent completion dependency. This emits no shader wait.
  void AcquireFromSystem();
  // Releases preceding ordinary compute-buffer stores to coherent SYSTEM
  // backing and writes a known 32-bit value at bottom-of-pipe. The confirmed
  // TC/L2 write has no interrupt, scalar-store or CP-DMA completion request.
  void ReleaseSystem32(uint64_t target_address, uint32_t value);
  // Samples the GPU clock at bottom-of-pipe with a confirmed TC/L2 write.
  // This cacheless release requires an explicit visibility/ownership join.
  void ReleaseGpuClock64(uint64_t target_address);
  // Writes a confirmed known DWORD at bottom-of-pipe without cache actions.
  // The marker alone does not publish other addresses to the host.
  void Release32(uint64_t target_address, uint32_t value);
  // Joins bottom-of-pipe through a private known-value fence and PAL's compute
  // equality wait, then writes back GL2. The caller owns the fence through a
  // subsequent completion; this fixed recipe performs no invalidation.
  void WaitEndOfPipeAndWriteback(uint64_t fence_address, uint32_t value);
  // Binds ordinary shader inputs without touching profiling, dispatch-pointer,
  // scratch or scheduler context. The caller separately publishes code/data.
  void BindCompute(const Pm4ComputeProgram& program, uint64_t kernarg_address);
  // Direct wave32 launch in thread units, starting at zero with complete
  // groups. Shader completion and memory visibility require an explicit
  // subsequent completion/cache operation.
  void DispatchWave32(uint32_t x, uint32_t y, uint32_t z);
  // MEC wave32 launch from three uint32 workgroup counts at a four-byte-aligned
  // GPU byte address. The caller publishes the complete tuple before fetch,
  // retains it unchanged through its last consumer, supplies the shader ABI,
  // and separately joins shader completion.
  void DispatchIndirectWave32(uint64_t argument_address);
  // Calls one immutable first-level MEC command IB and returns to the ring.
  // The caller publishes 1..0xfffff DWORDs in four-byte-aligned owned
  // executable backing wholly below 2^48 before ring publication. Complete
  // backing stays immutable and retained through final use and checked queue
  // removal; return alone does not join shader completion.
  void CallIndirectBuffer(uint64_t buffer_address, uint32_t word_count);
  // Confirmed TC/L2 memory transfers; width does not imply atomicity. A 32-bit
  // source retains readable trailing backing for native reads beyond its DWORD.
  void CopyData32(uint64_t source_address, uint64_t target_address);
  void CopyData64(uint64_t source_address, uint64_t target_address);
  // Atomic STORE via single-pass TC/L2 swap, without exposing a prior value.
  // Target operation support and pair reach are separate from queue encoding;
  // a subsequent barrier and completion establish visibility and retirement.
  void AtomicStore32(uint64_t target_address, uint32_t value);
  void AtomicStore64(uint64_t target_address, uint64_t value);
  // MEC incrementing L2 copy with RAW_WAIT and write confirmation.
  // The caller supplies a nonzero byte count within the native 26-bit field
  // and the selected transfer policy, then joins final use with WaitDma and
  // explicit cache/marker work. The CTS currently selects 1024 bytes.
  void DmaCopyL2(uint64_t source_address, uint64_t target_address,
                 uint32_t byte_length);
  // MEC zero-byte DMA drain, with all reserved fields clear.
  // This does not perform cache maintenance or publish a host marker.
  void WaitDma();
  // Confirmed, incrementing TC/L2 writes. The payload has 1..16381 DWORDs.
  void WriteData(uint64_t target_address, const uint32_t* values,
                 size_t value_count);
  void WriteData32(uint64_t target_address, uint32_t value);
  // Explicit memory dependencies with ordinary MEC execution, without ACE
  // offload. These operations do not acquire payload caches.
  void WaitMemory32(
      uint64_t address, uint32_t value,
      Pm4MemoryComparison comparison = Pm4MemoryComparison::kEqual,
      uint32_t mask = UINT32_MAX);
  void WaitMemory64(
      uint64_t address, uint64_t value,
      Pm4MemoryComparison comparison = Pm4MemoryComparison::kEqual,
      uint64_t mask = UINT64_MAX);
  // Samples the GPU clock at the command processor using confirmed COPY_DATA.
  // This is not shader completion, cache release, or a host-correlated time.
  void CopyGpuClock64(uint64_t target_address);
  void PadToEightWords();
  size_t word_count() const { return word_count_; }

 private:
  // Emits whole-cache acquisition with the selected native GCR fields.
  void AcquireMemory(uint32_t gcr);
  // Writes a known ordinary compute register interval relative to SH space.
  void SetComputeRegisters(uint32_t first_register, const uint32_t* values,
                           size_t value_count);
  // Emits one type-3 NOP of at least two words, including its header.
  void Noop(size_t word_count);

  // Caller-owned command storage, large enough for the known test sequence.
  uint32_t* words_;
  // Borrowed immutable encoding facts selected before stream construction.
  const Pm4CommandProfile& profile_;
  // Number of complete command words emitted into words_.
  size_t word_count_ = 0;
};

#endif  // AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_
