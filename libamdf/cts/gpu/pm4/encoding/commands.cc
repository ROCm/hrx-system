// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/encoding/commands.h"

#include <cstring>

#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"

namespace {

uint32_t MakeHeader(uint32_t opcode, size_t word_count) {
  return (UINT32_C(3) << 30) | (opcode << 8) |
         (static_cast<uint32_t>(word_count - 2) << 16);
}

}  // namespace

void Pm4CommandWriter::SetComputeRegisters(uint32_t first_register,
                                           const uint32_t* values,
                                           size_t value_count) {
  // Ordinary SET_SH_REG with compute shader type, no indexed register mode.
  words_[word_count_++] = MakeHeader(0x76, value_count + 2) | (1 << 1);
  words_[word_count_++] = first_register - 0x2c00;
  std::memcpy(words_ + word_count_, values, value_count * sizeof(*values));
  word_count_ += value_count;
}

void Pm4CommandWriter::BindCompute(
    const Pm4ComputeProgram& program, uint64_t kernarg_address,
    amdf_queue_format_features_t format_features) {
  const uint32_t entry[] = {
      static_cast<uint32_t>(program.entry_address >> 8),
      static_cast<uint32_t>(program.entry_address >> 40),
  };
  SetComputeRegisters(0x2e0c, entry, 2);
  // HSA descriptors leave LDS_SIZE zero. The PM4 caller realizes the total
  // group allocation in the target's units without changing the compiler
  // image.
  const uint32_t lds_units =
      profile_.EncodeGroupSegmentByteLength(program.group_segment_byte_length);
  const uint32_t resources[] = {
      program.resource1 |
          ((format_features & AMDF_GPU_PM4_FORMAT_FEATURE_COMPUTE_PRIV)
               ? UINT32_C(1) << 20
               : 0),
      (program.resource2 & ~UINT32_C(0x00ff8000)) | (lds_units << 15) |
          ((format_features & AMDF_GPU_PM4_FORMAT_FEATURE_COMPUTE_TRAP_PRESENT)
               ? UINT32_C(1) << 6
               : 0),
  };
  SetComputeRegisters(0x2e12, resources, 2);
  SetComputeRegisters(profile_.resource3_register, &program.resource3, 1);
  // PAL and Mesa's ordinary compute policy selects SIMD_DEST_CNTL when the
  // complete workgroup contains a multiple of four waves.
  const uint32_t workitem_count = program.workgroup_size[0] *
                                  program.workgroup_size[1] *
                                  program.workgroup_size[2];
  const uint32_t wave_count =
      (workitem_count + program.wavefront_size - 1u) / program.wavefront_size;
  const uint32_t resource_limits = wave_count % 4 == 0 ? (1u << 22) : 0;
  SetComputeRegisters(0x2e15, &resource_limits, 1);
  // The interval ends before native PIPELINESTAT_ENABLE/PERFCOUNT_ENABLE.
  const uint32_t geometry[] = {0,
                               0,
                               0,
                               program.workgroup_size[0],
                               program.workgroup_size[1],
                               program.workgroup_size[2]};
  SetComputeRegisters(0x2e04, geometry, 6);
  const uint32_t arguments[] = {static_cast<uint32_t>(kernarg_address),
                                static_cast<uint32_t>(kernarg_address >> 32)};
  SetComputeRegisters(0x2e40, arguments, 2);
}

void Pm4CommandWriter::Dispatch(const Pm4ComputeProgram& program, uint32_t x,
                                uint32_t y, uint32_t z) {
  words_[word_count_++] = MakeHeader(0x15, 5) | (1 << 1);
  words_[word_count_++] = x;
  words_[word_count_++] = y;
  words_[word_count_++] = z;
  // COMPUTE_SHADER_EN, FORCE_START_AT_000 and USE_THREAD_DIMENSIONS. CS_W32_EN
  // follows the compiled program rather than the preceding dispatch.
  words_[word_count_++] = 0x25 | (program.wavefront_size == 32 ? 0x8000 : 0);
}

void Pm4CommandWriter::DispatchIndirect(const Pm4ComputeProgram& program,
                                        uint64_t argument_address) {
  // The MEC form takes an absolute byte address, not a SET_BASE offset.
  words_[word_count_++] = MakeHeader(0x16, 4) | (1 << 1);
  words_[word_count_++] = static_cast<uint32_t>(argument_address);
  words_[word_count_++] = static_cast<uint32_t>(argument_address >> 32);
  // COMPUTE_SHADER_EN and FORCE_START_AT_000; dimensions are groups. CS_W32_EN
  // uses the same compiled wave size as the complete binding.
  words_[word_count_++] = 0x5 | (program.wavefront_size == 32 ? 0x8000 : 0);
}

void Pm4CommandWriter::CallIndirectBuffer(uint64_t buffer_address,
                                          uint32_t word_count) {
  words_[word_count_++] = MakeHeader(0x3f, 4) | (1u << 1);
  words_[word_count_++] = static_cast<uint32_t>(buffer_address);
  words_[word_count_++] = static_cast<uint32_t>(buffer_address >> 32);
  // PAL's ordinary MEC call sets VALID, with VMID and cache policy zero.
  // Policy zero is LRU on GFX11 and the regular temporal hint on GFX12+.
  words_[word_count_++] = word_count | (1u << 23);
}

void Pm4CommandWriter::ChainIndirectBuffer(const Pm4IndirectBuffer& successor) {
  words_[word_count_++] = MakeHeader(0x3f, 4);
  words_[word_count_++] = static_cast<uint32_t>(successor.address);
  words_[word_count_++] = static_cast<uint32_t>(successor.address >> 32);
  // MEC CHAIN and VALID, with zero VMID, policy and all reserved bits.
  words_[word_count_++] = successor.word_count | (1u << 20) | (1u << 23);
}

void Pm4CommandWriter::BranchIndirectBuffer(uint64_t operand_address,
                                            uint64_t reference, uint64_t mask,
                                            Pm4MemoryComparison comparison,
                                            const Pm4IndirectBuffer& pass,
                                            const Pm4IndirectBuffer& fail) {
  words_[word_count_++] = MakeHeader(0x3f, 14);
  words_[word_count_++] = 2u | (static_cast<uint32_t>(comparison) << 8);
  words_[word_count_++] = static_cast<uint32_t>(operand_address);
  words_[word_count_++] = static_cast<uint32_t>(operand_address >> 32);
  words_[word_count_++] = static_cast<uint32_t>(mask);
  words_[word_count_++] = static_cast<uint32_t>(mask >> 32);
  words_[word_count_++] = static_cast<uint32_t>(reference);
  words_[word_count_++] = static_cast<uint32_t>(reference >> 32);
  words_[word_count_++] = static_cast<uint32_t>(pass.address);
  words_[word_count_++] = static_cast<uint32_t>(pass.address >> 32);
  // These branch words have counts and default policies, not CHAIN/VALID bits.
  words_[word_count_++] = pass.word_count;
  words_[word_count_++] = static_cast<uint32_t>(fail.address);
  words_[word_count_++] = static_cast<uint32_t>(fail.address >> 32);
  words_[word_count_++] = fail.word_count;
}

void Pm4CommandWriter::ExecuteIfNonzero(uint64_t predicate_address,
                                        uint32_t word_count) {
  words_[word_count_++] = MakeHeader(0x22, 5);
  words_[word_count_++] = static_cast<uint32_t>(predicate_address);
  words_[word_count_++] = static_cast<uint32_t>(predicate_address >> 32);
  // PAL's ordinary policy is zero: LRU on GFX11, regular temporal on GFX12+.
  words_[word_count_++] = 0;
  words_[word_count_++] = word_count;
}

void Pm4CommandWriter::AcquireMemory(uint32_t gcr) {
  words_[word_count_++] = MakeHeader(0x58, 8);
  words_[word_count_++] = 0;
  words_[word_count_++] = UINT32_MAX;
  words_[word_count_++] = 0xff;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0x0a;
  words_[word_count_++] = gcr;
}

void Pm4CommandWriter::SystemBarrier() {
  WaitCompute();
  AcquireMemory(profile_.system_acquire_gcr);
}

void Pm4CommandWriter::WaitCompute() {
  words_[word_count_++] = MakeHeader(0x46, 2);
  words_[word_count_++] = 7 | (4 << 8);  // CS_PARTIAL_FLUSH.
}

void Pm4CommandWriter::AcquireFromSystem() {
  // GLI_INV occupies bits 1:0 in every admitted profile. Preserve all data
  // cache operations, writebacks, scope and sequencing selected for the target.
  AcquireMemory(profile_.system_acquire_gcr & ~UINT32_C(3));
}

void Pm4CommandWriter::ReleaseSystem32(uint64_t target_address,
                                       uint32_t value) {
  // RELEASE_MEM has its own GCR layout, distinct from ACQUIRE_MEM. The
  // selected profile supplies the cache actions before the completion write.
  words_[word_count_++] = MakeHeader(0x49, 8);
  words_[word_count_++] = 0x28 | (5 << 8) | (profile_.system_release_gcr << 12);
  // Immediate DWORD, write confirmation without interrupt, TC/L2 destination.
  words_[word_count_++] = (1 << 29) | (3 << 24) | (1 << 16);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = value;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
}

void Pm4CommandWriter::ReleaseSystemInterrupt32(uint64_t target_address,
                                                uint32_t value,
                                                uint32_t interrupt_context) {
  words_[word_count_++] = MakeHeader(0x49, 8);
  words_[word_count_++] = 0x28 | (5 << 8) | (profile_.system_release_gcr << 12);
  // Immediate DWORD, interrupt after write confirmation, TC/L2 destination.
  words_[word_count_++] = (1 << 29) | (2 << 24) | (1 << 16);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = value;
  words_[word_count_++] = 0;
  words_[word_count_++] = interrupt_context;
}

void Pm4CommandWriter::ReleaseGpuClock64(uint64_t target_address) {
  // PAL's bottom-of-pipe timestamp requests no release cache action or CP DMA
  // wait. GPU-clock data, confirmation without interrupt and TC/L2 destination
  // are independent fields from the event/index and release GCR word.
  words_[word_count_++] = MakeHeader(0x49, 8);
  words_[word_count_++] = 0x28 | (5 << 8);
  words_[word_count_++] = (3 << 29) | (3 << 24) | (1 << 16);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
}

void Pm4CommandWriter::Release32(uint64_t target_address, uint32_t value) {
  words_[word_count_++] = MakeHeader(0x49, 8);
  words_[word_count_++] = 0x28 | (5 << 8);
  words_[word_count_++] = (1 << 29) | (3 << 24) | (1 << 16);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = value;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
}

void Pm4CommandWriter::Store32(uint64_t target_address, uint32_t value) {
  word_count_ += pm4::StoreImmediate(words_ + word_count_, target_address,
                                     value, pm4::CopyDataWidth::k32Bit);
}

void Pm4CommandWriter::Store64(uint64_t target_address, uint64_t value) {
  word_count_ += pm4::StoreImmediate(words_ + word_count_, target_address,
                                     value, pm4::CopyDataWidth::k64Bit);
}

void Pm4CommandWriter::WaitEndOfPipeAndWriteback(uint64_t fence_address,
                                                 uint32_t value) {
  Release32(fence_address, value);
  // PAL compute waits on the exact private-fence value with ACE offload and
  // poll interval 10. This GPU wait joins the release before cache work.
  words_[word_count_++] = MakeHeader(0x3c, 7);
  words_[word_count_++] = 3 | (1 << 4);
  words_[word_count_++] = static_cast<uint32_t>(fence_address);
  words_[word_count_++] = static_cast<uint32_t>(fence_address >> 32);
  words_[word_count_++] = value;
  words_[word_count_++] = UINT32_MAX;
  words_[word_count_++] = UINT32_C(0x8000000a);
  // ACE ACQUIRE does immediate cache work, not shader-idle waiting. Whole-cache
  // GL2_WB is bit 15; the selected profile also supplies its scope. The high
  // size preserves only defined MEC bits rather than the wider ME layout.
  AcquireMemory(profile_.gl2_writeback_gcr);
}

void Pm4CommandWriter::BindScratch(uint64_t base, uint32_t waves_per_engine,
                                   uint32_t wave_byte_length) {
  const uint32_t address[] = {static_cast<uint32_t>(base >> 8),
                              static_cast<uint32_t>(base >> 40)};
  SetComputeRegisters(0x2e10, address, 2);
  const uint32_t size = waves_per_engine | ((wave_byte_length >> 8) << 12);
  SetComputeRegisters(profile_.scratch_size_register, &size, 1);
}

void Pm4CommandWriter::CopyData32(uint64_t source_address,
                                  uint64_t target_address) {
  word_count_ += pm4::CopyData(words_ + word_count_, source_address,
                               target_address, pm4::CopyDataWidth::k32Bit);
}

void Pm4CommandWriter::CopyData64(uint64_t source_address,
                                  uint64_t target_address) {
  word_count_ += pm4::CopyData(words_ + word_count_, source_address,
                               target_address, pm4::CopyDataWidth::k64Bit);
}

void Pm4CommandWriter::DmaCopy(uint64_t source_address, uint64_t target_address,
                               uint32_t byte_length) {
  words_[word_count_++] = MakeHeader(0x50, 7);
  // TC_L2 source/destination, default policies and no PFP-layout CP_SYNC bit.
  words_[word_count_++] = (3u << 29) | (3u << 20);
  words_[word_count_++] = static_cast<uint32_t>(source_address);
  words_[word_count_++] = static_cast<uint32_t>(source_address >> 32);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  // Direct byte count and RAW_WAIT; incrementing addresses and DIS_WC=0.
  words_[word_count_++] = byte_length | (1u << 30);
}

void Pm4CommandWriter::DmaFill32(uint64_t target_address, uint32_t pattern,
                                 uint32_t byte_length) {
  words_[word_count_++] = MakeHeader(0x50, 7);
  // Immediate DWORD source and TC_L2 destination; reserved MEC fields clear.
  words_[word_count_++] = (2u << 29) | (3u << 20);
  words_[word_count_++] = pattern;
  words_[word_count_++] = 0;
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = byte_length | (1u << 30);
}

void Pm4CommandWriter::WaitDma() {
  // RADV's compute emitter leaves CP_SYNC clear even for its logical drain.
  // Zero length performs no source or destination memory access.
  words_[word_count_++] = MakeHeader(0x50, 7);
  std::memset(words_ + word_count_, 0, 6 * sizeof(*words_));
  word_count_ += 6;
}

void Pm4CommandWriter::WriteData(uint64_t target_address,
                                 const uint32_t* values, size_t value_count) {
  word_count_ +=
      pm4::WriteData(words_ + word_count_, target_address, values, value_count);
}

void Pm4CommandWriter::WriteData32(uint64_t target_address, uint32_t value) {
  WriteData(target_address, &value, 1);
}

void Pm4CommandWriter::Noop(size_t word_count) {
  words_[word_count_++] = MakeHeader(0x10, word_count);
  std::memset(words_ + word_count_, 0, (word_count - 1) * sizeof(*words_));
  word_count_ += word_count - 1;
}

void Pm4CommandWriter::WaitMemory32(uint64_t address, uint32_t value,
                                    Pm4MemoryComparison comparison,
                                    uint32_t mask) {
  word_count_ +=
      pm4::WaitMemory32(words_ + word_count_, address, value, comparison, mask);
}

void Pm4CommandWriter::WaitMemory64(uint64_t address, uint64_t value,
                                    Pm4MemoryComparison comparison,
                                    uint64_t mask) {
  word_count_ +=
      pm4::WaitMemory64(words_ + word_count_, address, value, comparison, mask);
}

void Pm4CommandWriter::CopyGpuClock64(uint64_t target_address) {
  word_count_ += pm4::CopyGpuClock64(words_ + word_count_, target_address);
}

void Pm4CommandWriter::PadToEightWords(size_t trailing_word_count) {
  size_t padding = 8 - (word_count_ + trailing_word_count) % 8;
  if (padding == 1) {
    padding += 8;
  }
  Noop(padding);
}
