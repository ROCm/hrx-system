// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_MAPPING_FIXTURE_H_
#define AMDF_CTS_GPU_PM4_MAPPING_FIXTURE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/sparse_mapping_layout.h"

// Owns one mapping scenario's reservations, backing, compute copy stream, and
// checked queue-first teardown. Derived tests select the mapping transitions.
class Pm4MappingFixture : public Pm4DispatchTest {
 protected:
  static constexpr uint64_t kSlabByteLength =
      amdf::testing::kSparseSlabByteLength;
  static constexpr uint64_t kPayloadByteOffset = 512;
  static constexpr uint64_t kPlacementByteOffset = UINT64_C(18) << 20;
  static constexpr uint64_t kLargePayloadByteLength = UINT64_C(16) << 20;
  static constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;

  enum class SparseReadRoute { kReservation, kOriginalBacking };

  void TearDown() override;
  void ReleaseReservation(amdf_memory_reservation_t** owner);
  void ReleaseBacking(GpuMemory* backing);
  void Prepare(amdf_memory_scope_t* scope, uint64_t command_byte_length = 65536,
               uint64_t secondary_mapping_granularity = 0);
  void CreateBacking(uint64_t byte_length, amdf_device_t* consumer,
                     GpuMemory** out_memory);
  void CreateBacking(uint64_t byte_length, GpuMemory** out_memory) {
    CreateBacking(byte_length, nullptr, out_memory);
  }
  void Copy(uint64_t source, uint64_t target, uint64_t byte_length);
  void Verify(uint64_t address, uint64_t byte_length, uint32_t owner);
  void WriteAndVerify(uint64_t offset, uint64_t byte_length, uint32_t owner,
                      const GpuMemory& backing);
  void WriteSparseGeneration(std::span<const amdf::testing::SparseRange> ranges,
                             const std::array<uint64_t, 2>& addresses,
                             uint32_t generation);
  void ReadSparseGeneration(
      std::span<const amdf::testing::SparseRange> ranges,
      std::span<const amdf::testing::SparseMapping> mappings,
      std::span<GpuMemory* const> slabs,
      const std::array<uint64_t, 2>& addresses, uint32_t generation,
      SparseReadRoute route, std::span<uint32_t> capture);
  void RunRetainedSparse(
      amdf_memory_scope_t* scope,
      std::span<const amdf::testing::SparseRange> original_ranges,
      std::span<const amdf::testing::SparseRange> replacement_ranges);

  static uint32_t TaggedWord(uint32_t owner, uint64_t word);

  // Case-owned VA, explicitly cleared before its ordinary borrowed backing.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Second independent address interval in the retained-slab scenario.
  amdf_memory_reservation_t* second_reservation_ = nullptr;
  // Immutable consumer projection queried before physical backing exists.
  amdf_memory_reservation_access_info_t reservation_info_ = {};
  // Borrowed storage placement chosen by the scenario.
  amdf_memory_scope_t* scope_ = nullptr;
  // Exact permissions supported by this native mapping domain.
  amdf_memory_access_t access_ = 0;
  // Permissions that native hardware cannot deny on an attached alias.
  amdf_memory_access_t required_access_ = 0;
  // Stable physical owners; early release consumes handles without erasing
  // slots.
  std::deque<GpuMemory> backing_;
  // Caller-owned pages retained through every registered source's teardown.
  std::vector<void*> caller_storage_;
  // Case-owned logical device used to exercise shared native address domains.
  amdf_device_t* second_device_ = nullptr;
  // Failed consuming teardown preserves the remaining native dependencies.
  bool cleanup_failed_ = false;
  // Persistent stream, owned by the command fixture and never recreated.
  GpuCommandQueue* queue_ = nullptr;
  // Immutable compiled program selected by endpoint capabilities.
  Pm4ComputeProgram program_ = {};
  // Fixed CPU-authored source outside the remapped reservation.
  GpuMemory* upload_ = nullptr;
  // Fixed output observing the first GPU access after each change.
  GpuMemory* readback_ = nullptr;
  // Reusable argument storage modified only after execution completion.
  GpuMemory* arguments_ = nullptr;
  // Program-ordered completion marker, distinct from ring consumption.
  GpuMemory* completion_ = nullptr;
  // Cumulative finite command extent; the stream does not wrap.
  size_t command_word_count_ = 0;
  // Monotonic completion value for the finite dispatch sequence.
  uint32_t completed_ = 0;
};

#endif  // AMDF_CTS_GPU_PM4_MAPPING_FIXTURE_H_
