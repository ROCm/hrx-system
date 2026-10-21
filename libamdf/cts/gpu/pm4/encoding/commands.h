// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_
#define AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_

#include <cstddef>
#include <cstdint>

#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"
#include "libamdf/cts/gpu/pm4/encoding/profile.h"

// Compiled PM4 compute program with no hidden runtime inputs. The caller
// supplies a kernarg pointer and any required scratch; hardware supplies IDs.
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
  // Compiled workitems per wave (32 or 64), shared by binding and dispatch.
  uint32_t wavefront_size;
  // Complete workgroup dimensions in workitems, matching the compiled program.
  uint32_t workgroup_size[3];
};

// One complete command block retained by the enclosing graph owner.
struct Pm4IndirectBuffer {
  // Four-byte-aligned GPU byte address of immutable executable backing.
  uint64_t address;
  // Positive direct DWORD count, at most 0xfffff, wholly within that backing.
  uint32_t word_count;
};

// Encodes the selected RDNA compute recipe using PM4 format version 1.
// Native callers admit the target and corresponding packet, transfer and
// cache-control requirements before constructing a stream. Callers supply
// sufficient storage and addresses aligned to four bytes for 32-bit operations
// and eight bytes for 64-bit operations, except the explicitly documented
// conditional-branch comparison address.
class Pm4CommandWriter {
 public:
  Pm4CommandWriter(uint32_t* words, const Pm4CommandProfile& profile)
      : words_(words), profile_(profile) {}

  // Joins preceding compute waves and acquires code and data from SYSTEM.
  void SystemBarrier();
  // Joins preceding compute waves without performing cache operations.
  void WaitCompute();
  // Acquires SYSTEM data while retaining instruction-cache contents. The
  // caller has already published immutable code and joined preceding waves
  // through an independent completion dependency. This emits no shader wait.
  void AcquireFromSystem();
  // Releases preceding ordinary compute-buffer stores to coherent SYSTEM
  // backing and writes a known 32-bit value at bottom-of-pipe. The confirmed
  // TC/L2 write has no interrupt, scalar-store or CP-DMA completion request.
  void ReleaseSystem32(uint64_t target_address, uint32_t value);
  // Releases to SYSTEM, writes the event mailbox DWORD, then generates an
  // interrupt with the event's context. The caller supplies already-qualified
  // native producer operands and retains the event through this packet's tail.
  void ReleaseSystemInterrupt32(uint64_t target_address, uint32_t value,
                                uint32_t interrupt_context);
  // Samples the GPU clock at bottom-of-pipe with a confirmed TC/L2 write.
  // This cacheless release requires an explicit visibility/ownership join.
  void ReleaseGpuClock64(uint64_t target_address);
  // Writes a confirmed known value at bottom-of-pipe without cache actions or
  // read-modify-write. The marker alone does not publish other addresses to
  // the host. Atomicity and reach require the target mapping's STORE contract.
  void Release32(uint64_t target_address, uint32_t value);
  // Writes one naturally aligned value at the CP stage with confirmation.
  // Atomicity and reach require the target mapping's STORE contract; shader
  // completion and other payload visibility are separate operations.
  void Store32(uint64_t target_address, uint32_t value);
  void Store64(uint64_t target_address, uint64_t value);
  // Joins bottom-of-pipe through a private known-value fence and PAL's compute
  // equality wait, then writes back GL2. The caller owns the fence through a
  // subsequent completion; this fixed recipe performs no invalidation.
  void WaitEndOfPipeAndWriteback(uint64_t fence_address, uint32_t value);
  // Combines immutable shader inputs with the selected queue family's native
  // register requirements. The caller separately publishes code and data.
  void BindCompute(const Pm4ComputeProgram& program, uint64_t kernarg_address,
                   amdf_queue_format_features_t format_features);
  // Programs admitted command-owned scratch after prior users have joined.
  // Base is 256-byte aligned; each engine owns waves_per_engine slots of
  // wave_byte_length bytes, a multiple of 256 covering every dispatched lane.
  void BindScratch(uint64_t base, uint32_t waves_per_engine,
                   uint32_t wave_byte_length);
  // Launches the bound program in thread units, starting at zero with complete
  // groups. The program supplies the compiled wave size; shader completion and
  // memory visibility require an explicit subsequent completion/cache
  // operation.
  void Dispatch(const Pm4ComputeProgram& program, uint32_t x, uint32_t y,
                uint32_t z);
  // MEC launch of the bound program from three uint32 workgroup counts at a
  // four-byte-aligned GPU byte address. The caller publishes the complete tuple
  // before fetch, retains it unchanged through its last consumer, supplies the
  // shader ABI, and separately joins shader completion.
  void DispatchIndirect(const Pm4ComputeProgram& program,
                        uint64_t argument_address);
  // Calls one immutable first-level MEC command IB and returns to the ring.
  // The caller publishes 1..0xfffff DWORDs in four-byte-aligned owned
  // executable backing wholly below 2^48 before ring publication. Complete
  // backing stays immutable and retained through final use and checked queue
  // removal; return alone does not join shader completion.
  void CallIndirectBuffer(uint64_t buffer_address, uint32_t word_count);
  // Continues at the same IB level, without saving a return address. Padding
  // precedes this terminal packet. The enclosing owner retains the complete
  // graph through native retirement, including its explicit continuations.
  void ChainIndirectBuffer(const Pm4IndirectBuffer& successor);
  // Terminal if-then-else branch comparing (memory & mask) with reference.
  // The caller pre-masks reference: reference & ~mask is zero. This keeps the
  // comparison equivalent on engines that also mask the reference. The writer
  // preserves both fields unchanged. PAL accepts a four-byte-aligned, stable
  // readable QWORD. Each selected block continues at the same IB level; no
  // shader wait or cache action is implicit. Both blocks and their
  // continuations remain owned.
  void BranchIndirectBuffer(uint64_t operand_address, uint64_t reference,
                            uint64_t mask, Pm4MemoryComparison comparison,
                            const Pm4IndirectBuffer& pass,
                            const Pm4IndirectBuffer& fail);
  // Executes the next complete packet range only if the addressed DWORD is
  // nonzero. The four-byte-aligned predicate is published and stable through
  // sampling; word_count is the following range's direct DWORD count, at most
  // 0x3fff. Sampling is neither a payload acquire nor shader completion.
  void ExecuteIfNonzero(uint64_t predicate_address, uint32_t word_count);
  // Confirmed TC/L2 memory transfers; width does not imply atomicity. A 32-bit
  // source retains readable trailing backing for native reads beyond its DWORD.
  void CopyData32(uint64_t source_address, uint64_t target_address);
  void CopyData64(uint64_t source_address, uint64_t target_address);
  // MEC incrementing copy through the native TC_L2 selectors, with RAW_WAIT
  // and write confirmation. Those selectors route through MALL on GFX12.
  // The caller supplies a nonzero byte count within the native 26-bit field
  // and the selected transfer policy, then joins final use with WaitDma and
  // explicit cache/marker work. Source and destination need only byte
  // alignment.
  void DmaCopy(uint64_t source_address, uint64_t target_address,
               uint32_t byte_length);
  // Repeats an immediate DWORD through the native TC_L2 destination selector.
  // The target and positive byte count are DWORD aligned; the count fits the
  // native 26-bit field and selected transfer policy. RAW_WAIT and enabled
  // write confirmation match DmaCopy; final use still needs WaitDma and
  // explicit cache/marker work.
  void DmaFill32(uint64_t target_address, uint32_t pattern,
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
  // Emits one type-3 NOP of at least two words, including its header. A primary
  // ring producer uses this to fill the tail before wrapping a complete packet.
  void Noop(size_t word_count);
  // Emits a complete NOP so this stream plus a following terminal packet is
  // eight-DWORD aligned. The caller emits exactly trailing_word_count words
  // afterward; zero pads the current stream normally.
  void PadToEightWords(size_t trailing_word_count = 0);
  size_t word_count() const { return word_count_; }

 private:
  // Emits whole-cache acquisition with the selected native GCR fields.
  void AcquireMemory(uint32_t gcr);
  // Writes a known ordinary compute register interval relative to SH space.
  void SetComputeRegisters(uint32_t first_register, const uint32_t* values,
                           size_t value_count);
  // Caller-owned command storage, large enough for the known test sequence.
  uint32_t* words_;
  // Borrowed immutable encoding facts selected before stream construction.
  const Pm4CommandProfile& profile_;
  // Number of complete command words emitted into words_.
  size_t word_count_ = 0;
};

#endif  // AMDF_CTS_GPU_PM4_ENCODING_COMMANDS_H_
