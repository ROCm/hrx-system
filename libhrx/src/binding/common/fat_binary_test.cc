// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/fat_binary.h"

#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

struct BundleEntry {
  uint64_t offset;
  uint64_t size;
  uint64_t triple_size;
};

struct Elf64Header {
  uint8_t magic[4];
  uint8_t elf_class;
  uint8_t elf_data;
  uint8_t elf_version;
  uint8_t osabi;
  uint8_t abiversion;
  uint8_t padding[7];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t phoff;
  uint64_t shoff;
  uint32_t flags;
  uint16_t ehsize;
  uint16_t phentsize;
  uint16_t phnum;
  uint16_t shentsize;
  uint16_t shnum;
  uint16_t shstrndx;
};

static_assert(sizeof(Elf64Header) == 64, "ELF64 header must be 64 bytes");

std::vector<uint8_t> MakeMinimalAmdgpuElf(uint32_t machine = 0x041,
                                          uint32_t generic_version = 0) {
  Elf64Header header = {};
  header.magic[0] = 0x7f;
  header.magic[1] = 'E';
  header.magic[2] = 'L';
  header.magic[3] = 'F';
  header.elf_class = 2;
  header.elf_data = 1;
  header.elf_version = 1;
  header.osabi = 64;
  header.abiversion = 4;
  header.machine = 224;
  header.version = 1;
  header.shoff = sizeof(Elf64Header);
  header.flags = machine | (generic_version << 24);
  std::vector<uint8_t> elf(sizeof(header), 0);
  memcpy(elf.data(), &header, sizeof(header));
  return elf;
}

std::vector<uint8_t> MakeMinimalGenericAmdgpuElf(uint32_t machine) {
  return MakeMinimalAmdgpuElf(machine, /*generic_version=*/1);
}

void AppendBytes(std::vector<uint8_t>& buffer, const void* data,
                 size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  buffer.insert(buffer.end(), bytes, bytes + length);
}

std::vector<uint8_t> MakeBundle(
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries) {
  constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  static_assert(sizeof(kMagic) - 1 == 24, "bundle magic length changed");

  uint64_t payload_offset = sizeof(kMagic) - 1 + sizeof(uint64_t);
  for (const auto& entry : entries) {
    payload_offset += sizeof(BundleEntry) + entry.first.size();
  }

  std::vector<uint8_t> bundle;
  bundle.reserve(payload_offset + entries.size() * sizeof(Elf64Header));
  AppendBytes(bundle, kMagic, sizeof(kMagic) - 1);

  const uint64_t entry_count = entries.size();
  AppendBytes(bundle, &entry_count, sizeof(entry_count));

  uint64_t next_payload_offset = payload_offset;
  for (const auto& entry : entries) {
    const BundleEntry bundle_entry = {
        /*.offset=*/next_payload_offset,
        /*.size=*/entry.second.size(),
        /*.triple_size=*/entry.first.size(),
    };
    AppendBytes(bundle, &bundle_entry, sizeof(bundle_entry));
    AppendBytes(bundle, entry.first.data(), entry.first.size());
    next_payload_offset += entry.second.size();
  }

  for (const auto& entry : entries) {
    AppendBytes(bundle, entry.second.data(), entry.second.size());
  }
  return bundle;
}

std::string TripleString(const iree_hal_streaming_fat_binary_elf_t& match) {
  return std::string(match.triple.data, match.triple.size);
}

TEST(FatBinaryTest, SelectsExactTargetBeforeGeneric) {
  const auto generic_elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  const auto exact_elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);
  std::vector<uint8_t> bundle = MakeBundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx11-generic", generic_elf},
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", exact_elf},
  });

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx1100:sramecc-:xnack-")},
      {/*.value=*/IREE_SV("gfx11-generic")},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]),
            "hipv4-amdgcn-amd-amdhsa--gfx1100");
  EXPECT_STREQ(extract.matches[0].executable_format, "gfx1100");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, FallsBackToGenericTarget) {
  const auto elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  std::vector<uint8_t> bundle =
      MakeBundle({{"hipv4-amdgcn-amd-amdhsa--gfx11-generic", elf}});

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx1100:sramecc-:xnack-")},
      {/*.value=*/IREE_SV("gfx11-generic")},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]),
            "hipv4-amdgcn-amd-amdhsa--gfx11-generic");
  EXPECT_STREQ(extract.matches[0].executable_format, "gfx11-generic");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, MatchesBareGenericTarget) {
  const auto elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  std::vector<uint8_t> bundle = MakeBundle({{"gfx11-generic", elf}});

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx1100")},
      {/*.value=*/IREE_SV("gfx11-generic")},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]), "gfx11-generic");
  EXPECT_STREQ(extract.matches[0].executable_format, "gfx11-generic");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, ReportsMissingRankedTargets) {
  const auto elf = MakeMinimalAmdgpuElf();
  std::vector<uint8_t> bundle =
      MakeBundle({{"hipv4-amdgcn-amd-amdhsa--gfx942", elf}});

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx1100")},
      {/*.value=*/IREE_SV("gfx11-generic")},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_extract_for_targets(
          iree_make_const_byte_span(bundle.data(), bundle.size()),
          IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract)),
      StatusIs(iree::StatusCode::kNotFound));
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

}  // namespace
