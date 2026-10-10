// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/xdna/util/executable.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

namespace {

// ELF32LE and XDNA metadata version 2 use explicit little-endian fields,
// independent of host structure padding. These sizes and identities are the
// fixed wire declarations from iree/schemas/xdna_executable.h.
constexpr uint32_t kElfHeaderSize = 52;
constexpr uint32_t kProgramHeaderSize = 32;
constexpr uint32_t kSectionHeaderSize = 40;
constexpr uint32_t kMaximumHeaderCount = 4096;
constexpr uint32_t kMaximumRecordCount = 65535;
constexpr uint32_t kMaximumMetadataSize = 16 * 1024 * 1024;
constexpr uint32_t kMetadataType = 0x6C584408;
constexpr uint32_t kMetadataHeaderSize = 64;
constexpr uint32_t kAllocationSize = 32;
constexpr uint32_t kAllocationUseSize = 4;
constexpr uint32_t kEntrySize = 48;
constexpr uint32_t kBindingSize = 40;
constexpr uint32_t kRelocationSize = 48;
constexpr uint32_t kInvocationSize = 16;
constexpr uint64_t kMaximumShimAddress = 0xFFFFFFFFFFFF;
constexpr uint32_t kProfileInstructionAlignment = 32768;

uint16_t Read16(const uint8_t* bytes) {
  return uint16_t(bytes[0]) | (uint16_t(bytes[1]) << 8);
}

uint32_t Read32(const uint8_t* bytes) {
  return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
         (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

uint64_t Read64(const uint8_t* bytes) {
  return uint64_t(Read32(bytes)) | (uint64_t(Read32(bytes + 4)) << 32);
}

void Write32(uint8_t* bytes, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    bytes[i] = uint8_t(value >> (i * 8));
  }
}

bool Fits(uint64_t offset, uint64_t length, uint64_t capacity) {
  return offset <= capacity && length <= capacity - offset;
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

struct FileRange {
  // Beginning of an already bounded file range, in bytes.
  uint64_t offset;
  // Number of file bytes in the range.
  uint64_t length;
};

bool Overlaps(FileRange lhs, FileRange rhs) {
  return lhs.length != 0 && rhs.length != 0 &&
         lhs.offset < rhs.offset + rhs.length &&
         rhs.offset < lhs.offset + lhs.length;
}

// Compiler execution identities from the NPU2 target predecessor. The public
// device API does not report these profile or firmware ABI identifiers.
uint64_t DeviceProfile(const amdf_xdna_endpoint_info_t& endpoint_info) {
  if (endpoint_info.architecture != AMDF_XDNA_ARCHITECTURE_AIE2P) {
    return 0;
  }
  const char* end = std::find(std::begin(endpoint_info.target_id),
                              std::end(endpoint_info.target_id), '\0');
  const std::string_view target(endpoint_info.target_id,
                                end - endpoint_info.target_id);
  if (target == "amd.xdna.strix.17f0_10" ||
      target == "amd.xdna.krackan.17f0_20") {
    return UINT64_C(0x5354524958000001);
  }
  if (target == "amd.xdna.strix_halo.17f0_11") {
    return UINT64_C(0x535848414C4F0001);
  }
  return 0;
}

}  // namespace

::testing::AssertionResult XdnaExecutable::Initialize(
    std::span<const uint8_t> elf,
    const amdf_xdna_endpoint_info_t& endpoint_info,
    const amdf_xdna_device_info_t& device_info, uint32_t logical_column_count,
    std::span<const amdf_memory_access_t> binding_accesses) {
  const uint64_t profile = DeviceProfile(endpoint_info);
  if (profile == 0 || logical_column_count == 0 || logical_column_count > 8 ||
      logical_column_count > device_info.array.column_count ||
      device_info.array.row_count != 6 ||
      !IsPowerOfTwo(device_info.instruction.address_alignment) ||
      device_info.instruction.byte_length_granularity == 0 ||
      device_info.instruction.maximum_byte_length == 0 ||
      device_info.instruction.format.format !=
          AMDF_XDNA_BINARY_FORMAT_TRANSACTION ||
      device_info.instruction.format.version !=
          AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1) {
    return ::testing::AssertionFailure()
           << "fixture requires an admitted AIE2P NPU2 transaction target";
  }
  constexpr std::array<uint8_t, 16> kIdentity = {
      0x7F, 'E', 'L', 'F', 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  if (elf.size() < kElfHeaderSize ||
      !std::equal(kIdentity.begin(), kIdentity.end(), elf.begin()) ||
      Read16(elf.data() + 16) != 2 || Read16(elf.data() + 18) != 264 ||
      Read32(elf.data() + 20) != 1 || Read32(elf.data() + 24) != 0 ||
      Read32(elf.data() + 28) != kElfHeaderSize ||
      Read32(elf.data() + 36) != 3 ||
      Read16(elf.data() + 40) != kElfHeaderSize ||
      Read16(elf.data() + 42) != kProgramHeaderSize) {
    return ::testing::AssertionFailure()
           << "fixture is not canonical AIE2P ELF32LE";
  }
  const uint32_t header_count = Read16(elf.data() + 44);
  const FileRange header_range = {0, kElfHeaderSize};
  const FileRange program_directory = {
      kElfHeaderSize, uint64_t(header_count) * kProgramHeaderSize};
  if (header_count == 0 || header_count > kMaximumHeaderCount ||
      !Fits(program_directory.offset, program_directory.length, elf.size())) {
    return ::testing::AssertionFailure() << "invalid ELF program directory";
  }
  const uint32_t section_offset = Read32(elf.data() + 32);
  const uint32_t section_size = Read16(elf.data() + 46);
  const uint32_t section_count = Read16(elf.data() + 48);
  const uint32_t section_names = Read16(elf.data() + 50);
  const FileRange section_directory = {section_offset,
                                       uint64_t(section_count) * section_size};
  if ((section_offset != 0 || section_size != 0 || section_count != 0 ||
       section_names != 0) &&
      (section_offset == 0 || section_offset % 4 != 0 ||
       section_size != kSectionHeaderSize || section_count == 0 ||
       section_count > kMaximumHeaderCount || section_names >= section_count ||
       !Fits(section_directory.offset, section_directory.length, elf.size()) ||
       Overlaps(section_directory, header_range) ||
       Overlaps(section_directory, program_directory))) {
    return ::testing::AssertionFailure() << "invalid ELF section directory";
  }

  const uint8_t* metadata_header = elf.data() + kElfHeaderSize;
  const uint32_t metadata_offset = Read32(metadata_header + 4);
  const uint32_t metadata_size = Read32(metadata_header + 16);
  if (Read32(metadata_header) != kMetadataType ||
      Read32(metadata_header + 8) != 0 || Read32(metadata_header + 12) != 0 ||
      Read32(metadata_header + 20) != metadata_size ||
      metadata_size < kMetadataHeaderSize ||
      metadata_size > kMaximumMetadataSize ||
      !Fits(metadata_offset, metadata_size, elf.size())) {
    return ::testing::AssertionFailure() << "invalid XDNA metadata header";
  }
  const uint8_t* metadata = elf.data() + metadata_offset;
  if (Read32(metadata) != 0x414E4458 || Read16(metadata + 4) != 2 ||
      Read16(metadata + 6) != 1 || Read32(metadata + 8) != 3 ||
      Read32(metadata + 12) != 1 || Read64(metadata + 16) != profile ||
      Read64(metadata + 24) != UINT64_C(0x4E5055320006000C) ||
      Read16(metadata + 32) != logical_column_count ||
      Read16(metadata + 34) != 6) {
    return ::testing::AssertionFailure()
           << "XDNA metadata does not match the exact compiler/native target";
  }
  const uint32_t relocation_count = Read32(metadata + 52);
  const uint32_t invocation_count = Read32(metadata + 56);
  const uint32_t name_length = Read32(metadata + 60);
  if (Read32(metadata + 36) != 1 || Read32(metadata + 40) != 1 ||
      Read32(metadata + 44) != 1 ||
      Read32(metadata + 48) != binding_accesses.size() ||
      binding_accesses.size() > kMaximumRecordCount ||
      relocation_count > kMaximumRecordCount || invocation_count == 0 ||
      invocation_count > kMaximumRecordCount) {
    return ::testing::AssertionFailure()
           << "fixture requires one allocation/use/entry and the declared "
              "bindings";
  }
  constexpr uint32_t kAllocationOffset = kMetadataHeaderSize;
  constexpr uint32_t kUseOffset = kAllocationOffset + kAllocationSize;
  constexpr uint32_t kEntryOffset = kUseOffset + kAllocationUseSize;
  constexpr uint32_t kBindingOffset = kEntryOffset + kEntrySize;
  const uint32_t relocation_offset =
      kBindingOffset + binding_accesses.size() * kBindingSize;
  const uint64_t invocation_offset =
      relocation_offset + uint64_t(relocation_count) * kRelocationSize;
  const uint64_t name_offset =
      invocation_offset + uint64_t(invocation_count) * kInvocationSize;
  if (!Fits(name_offset, name_length, metadata_size) ||
      name_offset + name_length != metadata_size) {
    return ::testing::AssertionFailure()
           << "XDNA table extents do not match metadata";
  }

  XdnaExecutable candidate;
  candidate.elf_ = elf;
  candidate.binding_count_ = binding_accesses.size();
  const uint8_t* allocation = metadata + kAllocationOffset;
  candidate.allocation_byte_length_ = Read64(allocation + 8);
  candidate.allocation_alignment_ = Read64(allocation + 16);
  const uint32_t instruction_alignment = std::max(
      kProfileInstructionAlignment, device_info.instruction.address_alignment);
  if (Read32(allocation) != 1 || Read32(allocation + 4) != 0 ||
      candidate.allocation_byte_length_ == 0 ||
      candidate.allocation_byte_length_ > std::numeric_limits<size_t>::max() ||
      !IsPowerOfTwo(candidate.allocation_alignment_) ||
      candidate.allocation_alignment_ < instruction_alignment ||
      Read32(allocation + 24) != 1 ||
      Read32(allocation + 28) != header_count - 1 ||
      Read32(metadata + kUseOffset) != 0) {
    return ::testing::AssertionFailure()
           << "fixture requires one aligned mutable COMMAND allocation";
  }
  const uint8_t* entry = metadata + kEntryOffset;
  if (Read32(entry + 4) == 0 || Read32(entry + 4) > 4096 ||
      !Fits(Read32(entry), Read32(entry + 4), name_length) ||
      Read32(entry + 8) != 0 || Read32(entry + 12) != 1 ||
      Read32(entry + 16) != 0 ||
      Read32(entry + 20) != binding_accesses.size() ||
      Read32(entry + 24) != 0 || Read32(entry + 28) != 0 ||
      Read32(entry + 32) != 0 || Read32(entry + 36) != relocation_count ||
      Read32(entry + 40) != 0 || Read32(entry + 44) != invocation_count) {
    return ::testing::AssertionFailure()
           << "entry must own all uses/bindings/dynamic "
              "relocations/invocations";
  }
  // Bind supplies complete logical buffers at offset zero. A zero minimum
  // admits that offset for every unsigned maximum, including an unbounded one.
  for (size_t i = 0; i < binding_accesses.size(); ++i) {
    const uint8_t* binding = metadata + kBindingOffset + i * kBindingSize;
    // Recipes bind complete logical buffers at offset zero. A zero minimum
    // admits that use regardless of the unsigned maximum logical offset.
    if (Read16(binding) != 1 || Read16(binding + 2) != 1 ||
        Read16(binding + 4) != binding_accesses[i] ||
        Read16(binding + 6) != 5 || Read64(binding + 8) != kBindingByteLength ||
        Read64(binding + 16) != 4 || Read64(binding + 24) != 0) {
      return ::testing::AssertionFailure()
             << "binding " << i << " does not match the fixed buffer contract";
    }
  }

  std::vector<FileRange> source_ranges;
  source_ranges.reserve(header_count);
  candidate.loads_.reserve(header_count - 1);
  uint64_t previous_end = 0;
  for (uint32_t i = 0; i < header_count; ++i) {
    const uint8_t* header =
        elf.data() + kElfHeaderSize + i * kProgramHeaderSize;
    const FileRange file = {Read32(header + 4), Read32(header + 16)};
    const uint32_t destination = Read32(header + 8);
    const uint32_t memory_size = Read32(header + 20);
    const uint32_t alignment = Read32(header + 28);
    if (!IsPowerOfTwo(alignment) ||
        file.offset % alignment != destination % alignment ||
        file.length > memory_size || Read32(header + 24) != 4 ||
        !Fits(file.offset, file.length, elf.size()) ||
        Overlaps(file, header_range) || Overlaps(file, program_directory) ||
        Overlaps(file, section_directory)) {
      return ::testing::AssertionFailure() << "invalid load header " << i;
    }
    source_ranges.push_back(file);
    if (i == 0) {
      continue;
    }
    if (Read32(header) != 1 || Read32(header + 12) != 0 || memory_size == 0 ||
        destination < previous_end ||
        !Fits(destination, memory_size, candidate.allocation_byte_length_)) {
      return ::testing::AssertionFailure()
             << "load " << i << " escapes or overlaps command storage";
    }
    candidate.loads_.push_back({uint32_t(file.offset), destination,
                                uint32_t(file.length), memory_size});
    previous_end = uint64_t(destination) + memory_size;
  }
  std::sort(source_ranges.begin(), source_ranges.end(),
            [](FileRange lhs, FileRange rhs) {
              if (lhs.offset != rhs.offset) {
                return lhs.offset < rhs.offset;
              }
              return lhs.length > rhs.length;
            });
  FileRange previous = {};
  for (FileRange range : source_ranges) {
    if (range.length == 0) {
      continue;
    }
    if (Overlaps(previous, range) &&
        (previous.offset != range.offset || previous.length != range.length)) {
      return ::testing::AssertionFailure()
             << "ELF payload ranges partially overlap";
    }
    previous = range;
  }
  const auto initialized = [&candidate](uint64_t offset, uint64_t length) {
    if (!Fits(offset, length, candidate.allocation_byte_length_)) {
      return false;
    }
    const uint64_t end = offset + length;
    for (const LoadRange& load : candidate.loads_) {
      const uint64_t load_end =
          uint64_t(load.target_byte_offset) + load.memory_byte_length;
      if (load_end <= offset) {
        continue;
      }
      if (load.target_byte_offset > offset) {
        return false;
      }
      offset = load_end;
      if (offset >= end) {
        return true;
      }
    }
    return false;
  };
  candidate.relocations_.reserve(relocation_count);
  previous_end = 0;
  for (uint32_t i = 0; i < relocation_count; ++i) {
    const uint8_t* row = metadata + relocation_offset + i * kRelocationSize;
    const Relocation relocation = {Read32(row + 4),
                                   Read32(row + 8),
                                   std::bit_cast<int64_t>(Read64(row + 16)),
                                   Read64(row + 24),
                                   Read64(row + 32),
                                   Read64(row + 40)};
    if (Read32(row) != 0 ||
        relocation.binding_ordinal >= binding_accesses.size() ||
        Read32(row + 12) != 1 || relocation.byte_offset % 4 != 0 ||
        relocation.byte_offset < previous_end ||
        relocation.minimum_value > relocation.maximum_value ||
        relocation.maximum_value > kMaximumShimAddress ||
        relocation.alignment < 4 || !IsPowerOfTwo(relocation.alignment) ||
        !initialized(relocation.byte_offset, 8)) {
      return ::testing::AssertionFailure()
             << "invalid dynamic SHIM_ADDRESS relocation " << i;
    }
    candidate.relocations_.push_back(relocation);
    previous_end = uint64_t(relocation.byte_offset) + 8;
  }
  for (uint32_t i = 0; i < invocation_count; ++i) {
    const uint8_t* row = metadata + invocation_offset + i * kInvocationSize;
    const uint32_t offset = Read32(row + 4);
    const uint32_t length = Read32(row + 8);
    if (Read32(row) != 0 || Read32(row + 12) >= invocation_count ||
        offset % instruction_alignment != 0 || length == 0 || length % 4 != 0 ||
        length % device_info.instruction.byte_length_granularity != 0 ||
        length > device_info.instruction.maximum_byte_length ||
        !initialized(offset, length)) {
      return ::testing::AssertionFailure() << "invalid native invocation " << i;
    }
    if (i == 0) {
      candidate.invocation_byte_offset_ = offset;
      candidate.invocation_byte_length_ = length;
    }
  }
  *this = std::move(candidate);
  return ::testing::AssertionSuccess();
}

void XdnaExecutable::Load(std::span<uint8_t> storage) const {
  for (const LoadRange& load : loads_) {
    uint8_t* destination = storage.data() + load.target_byte_offset;
    std::memcpy(destination, elf_.data() + load.source_byte_offset,
                load.file_byte_length);
    std::memset(destination + load.file_byte_length, 0,
                load.memory_byte_length - load.file_byte_length);
  }
}

::testing::AssertionResult XdnaExecutable::Bind(
    std::span<uint8_t> storage, std::span<const uint64_t> addresses) const {
  if (addresses.size() != binding_count_) {
    return ::testing::AssertionFailure()
           << "binding address count does not match image";
  }
  for (size_t i = 0; i < addresses.size(); ++i) {
    if (addresses[i] % 4 != 0 ||
        addresses[i] > kMaximumShimAddress - (kBindingByteLength - 1)) {
      return ::testing::AssertionFailure()
             << "binding " << i << " violates the aligned 48-bit buffer range";
    }
  }
  for (const Relocation& relocation : relocations_) {
    const uint64_t base = addresses[relocation.binding_ordinal];
    const uint64_t displacement =
        relocation.addend < 0 ? uint64_t(0) - uint64_t(relocation.addend)
                              : uint64_t(relocation.addend);
    if ((relocation.addend < 0 && base < displacement) ||
        (relocation.addend >= 0 &&
         base > std::numeric_limits<uint64_t>::max() - displacement)) {
      return ::testing::AssertionFailure() << "relocated DMA address overflows";
    }
    const uint64_t address = base + uint64_t(relocation.addend);
    if (address < relocation.minimum_value ||
        address > relocation.maximum_value ||
        address % relocation.alignment != 0) {
      return ::testing::AssertionFailure()
             << "binding " << relocation.binding_ordinal
             << " violates its declared relocation constraints";
    }
  }
  for (const Relocation& relocation : relocations_) {
    const uint64_t address =
        addresses[relocation.binding_ordinal] + uint64_t(relocation.addend);
    uint8_t* field = storage.data() + relocation.byte_offset;
    Write32(field, (Read32(field) & 3u) | uint32_t(address));
    Write32(field + 4,
            (Read32(field + 4) & 0xFFFF0000u) | uint32_t(address >> 32));
  }
  return ::testing::AssertionSuccess();
}

std::span<const uint8_t> XdnaExecutable::ResolveInvocation(
    std::span<const uint8_t> storage) const {
  return storage.subspan(invocation_byte_offset_, invocation_byte_length_);
}
