// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_XDNA_UTIL_EXECUTABLE_H_
#define AMDF_CTS_XDNA_UTIL_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "amdf/xdna.h"
#include "gtest/gtest.h"

// CTS-only reader of contiguous native XDNA ELF fixtures. The format and load,
// binding and invocation rules are derived from runtime/src/iree/schemas/
// xdna_executable.{h,c}, runtime/src/iree/hal/drivers/amd/xdna/image/ and
// runtime/src/iree/hal/drivers/amd/xdna/executable_storage.c. This reader has
// no dependency on those implementations.
//
// The admitted subset is ELF32LE EM_AIE, AIE2P flags 3, metadata version 2 and
// transaction encoding 0.1: one mutable COMMAND allocation, one allocation
// use, one entry, no static relocations and caller-declared GLOBAL buffers.
// Each buffer has a 64-byte minimum extent, four-byte alignment and
// DEVICE_VISIBLE | COHERENT usage; access matches the caller's binding list.
// The caller uses logical offset zero, which the declared range must admit.
// Dynamic relocations are declared eight-byte SHIM_ADDRESS
// fields. Load ranges may alias source bytes and have explicit zero-fill tails.
// Only invocation zero, which establishes the program's state, is exposed.
//
// Profile revision and firmware ABI IDs come from the compiler's NPU2 target
// contract keyed by exact endpoint architecture and target ID. They are not
// firmware-version facts reported by device_query_info. Admission checks the
// available native geometry and instruction constraints separately. Neither
// ELF admission nor command completion certifies worker termination; the
// fixture's compiled program must satisfy that native submission contract.
class XdnaExecutable {
 public:
  static constexpr uint64_t kBindingByteLength = 64;

  // Admits the fixture once and borrows its immutable bytes. They must remain
  // live through every Load call. Native objects are neither created nor
  // retained. Failure leaves this object unchanged.
  ::testing::AssertionResult Initialize(
      std::span<const uint8_t> elf,
      const amdf_xdna_endpoint_info_t& endpoint_info,
      const amdf_xdna_device_info_t& device_info, uint32_t logical_column_count,
      std::span<const amdf_memory_access_t> binding_accesses);

  uint64_t allocation_byte_length() const { return allocation_byte_length_; }
  uint64_t allocation_alignment() const { return allocation_alignment_; }

  // Requires successful admission and exclusive writable storage of at least
  // allocation_byte_length() bytes, disjoint from the borrowed fixture. Copies
  // declared loads and clears only their explicit tails; gaps retain the
  // caller's initialization.
  void Load(std::span<uint8_t> storage) const;

  // Requires loaded storage with the same extent as Load and no pending users.
  // The caller supplies addresses of the complete logical buffers satisfying
  // the admitted permissions, extent and memory contract. Validates every
  // address and relocation before changing any byte; failure leaves storage
  // unchanged. Cache publication remains the caller's responsibility.
  ::testing::AssertionResult Bind(std::span<uint8_t> storage,
                                  std::span<const uint64_t> addresses) const;

  // Borrows invocation zero from successfully loaded and bound storage. The
  // caller may copy its complete native transaction into EXECUTE backing or
  // compose its operation body with additional case-owned native setup/drain.
  // This operation does not publish or submit the command.
  std::span<const uint8_t> ResolveInvocation(
      std::span<const uint8_t> storage) const;

 private:
  struct LoadRange {
    // Beginning of the borrowed file payload, in bytes.
    uint32_t source_byte_offset;
    // Beginning of the allocation destination, in bytes.
    uint32_t target_byte_offset;
    // Number of file bytes copied into the destination.
    uint32_t file_byte_length;
    // Initialized destination extent, including its explicit zero-fill tail.
    uint32_t memory_byte_length;
  };

  struct Relocation {
    // External buffer selected by this dynamic address field.
    uint32_t binding_ordinal;
    // Beginning of the eight-byte field in command backing.
    uint32_t byte_offset;
    // Signed displacement from the supplied buffer address.
    int64_t addend;
    // Inclusive lower bound for the relocated address.
    uint64_t minimum_value;
    // Inclusive upper bound for the relocated address.
    uint64_t maximum_value;
    // Required power-of-two relocated-address alignment.
    uint64_t alignment;
  };

  // Immutable fixture bytes borrowed from the caller.
  std::span<const uint8_t> elf_;
  // Decoded destination-ordered load records admitted once at initialization.
  std::vector<LoadRange> loads_;
  // Decoded nonoverlapping dynamic fields admitted once at initialization.
  std::vector<Relocation> relocations_;
  // Number of caller-declared logical buffers admitted with the image.
  size_t binding_count_ = 0;
  // Required command-backing capacity in bytes.
  uint64_t allocation_byte_length_ = 0;
  // Required command-backing address alignment in bytes.
  uint64_t allocation_alignment_ = 0;
  // Beginning of the establishing invocation within command backing.
  uint32_t invocation_byte_offset_ = 0;
  // Complete byte extent of the establishing invocation.
  uint32_t invocation_byte_length_ = 0;
};

#endif  // AMDF_CTS_XDNA_UTIL_EXECUTABLE_H_
