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

struct Elf64SectionHeader {
  uint32_t name;               // Section name offset.
  uint32_t type;               // Section type.
  uint64_t flags;              // Section flags.
  uint64_t address;            // Virtual address.
  uint64_t offset;             // File offset.
  uint64_t size;               // Section byte length.
  uint32_t link;               // Linked section index.
  uint32_t info;               // Section-specific information.
  uint64_t address_alignment;  // Required address alignment.
  uint64_t entry_size;         // Fixed entry byte length.
};

static_assert(sizeof(Elf64SectionHeader) == 64,
              "ELF64 section header must be 64 bytes");

struct Elf64Symbol {
  uint32_t name;           // Symbol name offset.
  uint8_t info;            // Symbol binding and type.
  uint8_t other;           // Symbol visibility.
  uint16_t section_index;  // Defining section index.
  uint64_t value;          // Symbol value.
  uint64_t size;           // Symbol byte length.
};

static_assert(sizeof(Elf64Symbol) == 24, "ELF64 symbol must be 24 bytes");

std::vector<uint8_t> MakeMinimalAmdgpuElf(uint32_t machine = 0x041,
                                          uint32_t generic_version = 0,
                                          uint32_t feature_flags = 0) {
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
  header.flags = machine | (generic_version << 24) | feature_flags;
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

std::vector<uint8_t> MakeAmdgpuElfWithGlobalSymbols() {
  std::vector<uint8_t> elf = MakeMinimalAmdgpuElf();

  const size_t string_offset = elf.size();
  constexpr char kStrings[] =
      "\0managed_value.managed\0ordinary_global\0ignored_descriptor.kd\0"
      "local_object\0undefined_object\0function\0";
  AppendBytes(elf, kStrings, sizeof(kStrings));
  const uint32_t managed_name = 1;
  const uint32_t ordinary_name = managed_name + sizeof("managed_value.managed");
  const uint32_t descriptor_name = ordinary_name + sizeof("ordinary_global");
  const uint32_t local_name = descriptor_name + sizeof("ignored_descriptor.kd");
  const uint32_t undefined_name = local_name + sizeof("local_object");
  const uint32_t function_name = undefined_name + sizeof("undefined_object");

  while (elf.size() % 8 != 0) elf.push_back(0);
  const size_t symbol_offset = elf.size();
  const Elf64Symbol symbols[] = {
      {},
      {/*.name=*/managed_name, /*.info=*/0x11, /*.other=*/0,
       /*.section_index=*/1, /*.value=*/0, /*.size=*/64},
      {/*.name=*/ordinary_name, /*.info=*/0x21, /*.other=*/0,
       /*.section_index=*/1, /*.value=*/0, /*.size=*/32},
      {/*.name=*/descriptor_name, /*.info=*/0x11, /*.other=*/0,
       /*.section_index=*/1, /*.value=*/0, /*.size=*/16},
      {/*.name=*/local_name, /*.info=*/0x01, /*.other=*/0,
       /*.section_index=*/1, /*.value=*/0, /*.size=*/8},
      {/*.name=*/undefined_name, /*.info=*/0x11, /*.other=*/0,
       /*.section_index=*/0, /*.value=*/0, /*.size=*/8},
      {/*.name=*/function_name, /*.info=*/0x12, /*.other=*/0,
       /*.section_index=*/1, /*.value=*/0, /*.size=*/8},
  };
  AppendBytes(elf, symbols, sizeof(symbols));

  while (elf.size() % 8 != 0) elf.push_back(0);
  const size_t section_offset = elf.size();
  const Elf64SectionHeader sections[] = {
      {},
      {/*.name=*/0, /*.type=*/3, /*.flags=*/0, /*.address=*/0,
       /*.offset=*/string_offset, /*.size=*/sizeof(kStrings), /*.link=*/0,
       /*.info=*/0, /*.address_alignment=*/1, /*.entry_size=*/0},
      {/*.name=*/0, /*.type=*/2, /*.flags=*/0, /*.address=*/0,
       /*.offset=*/symbol_offset, /*.size=*/sizeof(symbols), /*.link=*/1,
       /*.info=*/0, /*.address_alignment=*/8,
       /*.entry_size=*/sizeof(Elf64Symbol)},
  };
  AppendBytes(elf, sections, sizeof(sections));

  Elf64Header header;
  memcpy(&header, elf.data(), sizeof(header));
  header.shoff = section_offset;
  header.shentsize = sizeof(Elf64SectionHeader);
  header.shnum = IREE_ARRAYSIZE(sections);
  memcpy(elf.data(), &header, sizeof(header));
  return elf;
}

iree_status_t CollectGlobalName(void* user_data, iree_string_view_t name) {
  auto* names = static_cast<std::vector<std::string>*>(user_data);
  names->emplace_back(name.data, name.size);
  return iree_ok_status();
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

iree_hal_executable_target_t MakeExecutableTarget(
    iree_string_view_t target_key, iree_hal_executable_target_kind_t kind,
    uint32_t priority) {
  return {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/target_key,
      /*.kind=*/kind,
      /*.priority=*/priority,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
}

TEST(FatBinaryTest, SelectsExactTargetBeforeGeneric) {
  const auto generic_elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  const auto exact_elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);
  std::vector<uint8_t> bundle = MakeBundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx11-generic", generic_elf},
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", exact_elf},
  });

  const iree_hal_executable_target_t executable_targets[] = {
      MakeExecutableTarget(IREE_SV("gfx1100:sramecc-:xnack-"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100),
      MakeExecutableTarget(IREE_SV("gfx11-generic"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, 50),
  };
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_targets[0]},
      {/*.executable_target=*/&executable_targets[1]},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]),
            "hipv4-amdgcn-amd-amdhsa--gfx1100");
  EXPECT_EQ(extract.matches[0].executable_target, &executable_targets[0]);
  EXPECT_STREQ(extract.matches[0].code_object_target_key, "gfx1100");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, VisitsDefinedGlobalObjects) {
  const std::vector<uint8_t> elf = MakeAmdgpuElfWithGlobalSymbols();
  std::vector<std::string> names;
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_visit_elf_global_objects(
      iree_make_const_byte_span(elf.data(), elf.size()), CollectGlobalName,
      &names));

  EXPECT_EQ(names, (std::vector<std::string>{"managed_value.managed",
                                             "ordinary_global"}));
}

TEST(FatBinaryTest, RejectsTruncatedGlobalSymbolTable) {
  std::vector<uint8_t> elf = MakeAmdgpuElfWithGlobalSymbols();
  Elf64Header header;
  memcpy(&header, elf.data(), sizeof(header));
  Elf64SectionHeader symbol_section;
  memcpy(&symbol_section, elf.data() + header.shoff + 2 * header.shentsize,
         sizeof(symbol_section));
  symbol_section.size = elf.size();
  memcpy(elf.data() + header.shoff + 2 * header.shentsize, &symbol_section,
         sizeof(symbol_section));

  std::vector<std::string> names;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_visit_elf_global_objects(
          iree_make_const_byte_span(elf.data(), elf.size()), CollectGlobalName,
          &names)),
      StatusIs(iree::StatusCode::kOutOfRange));
}

TEST(FatBinaryTest, RejectsOutOfRangeGlobalName) {
  std::vector<uint8_t> elf = MakeAmdgpuElfWithGlobalSymbols();
  Elf64Header header;
  memcpy(&header, elf.data(), sizeof(header));
  Elf64SectionHeader symbol_section;
  memcpy(&symbol_section, elf.data() + header.shoff + 2 * header.shentsize,
         sizeof(symbol_section));
  Elf64Symbol symbol;
  memcpy(&symbol, elf.data() + symbol_section.offset + sizeof(Elf64Symbol),
         sizeof(symbol));
  symbol.name = UINT32_MAX;
  memcpy(elf.data() + symbol_section.offset + sizeof(Elf64Symbol), &symbol,
         sizeof(symbol));

  std::vector<std::string> names;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_visit_elf_global_objects(
          iree_make_const_byte_span(elf.data(), elf.size()), CollectGlobalName,
          &names)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(FatBinaryTest, FallsBackToGenericTarget) {
  const auto elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  std::vector<uint8_t> bundle =
      MakeBundle({{"hipv4-amdgcn-amd-amdhsa--gfx11-generic", elf}});

  const iree_hal_executable_target_t executable_targets[] = {
      MakeExecutableTarget(IREE_SV("gfx1100:sramecc-:xnack-"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100),
      MakeExecutableTarget(IREE_SV("gfx11-generic"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, 50),
  };
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_targets[0]},
      {/*.executable_target=*/&executable_targets[1]},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]),
            "hipv4-amdgcn-amd-amdhsa--gfx11-generic");
  EXPECT_EQ(extract.matches[0].executable_target, &executable_targets[1]);
  EXPECT_STREQ(extract.matches[0].code_object_target_key, "gfx11-generic");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, MatchesBareGenericTarget) {
  const auto elf = MakeMinimalGenericAmdgpuElf(/*machine=*/0x054);
  std::vector<uint8_t> bundle = MakeBundle({{"gfx11-generic", elf}});

  const iree_hal_executable_target_t executable_targets[] = {
      MakeExecutableTarget(IREE_SV("gfx1100"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100),
      MakeExecutableTarget(IREE_SV("gfx11-generic"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, 50),
  };
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_targets[0]},
      {/*.executable_target=*/&executable_targets[1]},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(bundle.data(), bundle.size()),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));

  EXPECT_EQ(extract.match_count, 1);
  EXPECT_EQ(TripleString(extract.matches[0]), "gfx11-generic");
  EXPECT_EQ(extract.matches[0].executable_target, &executable_targets[1]);
  EXPECT_STREQ(extract.matches[0].code_object_target_key, "gfx11-generic");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, ReportsMissingRankedTargets) {
  const auto elf = MakeMinimalAmdgpuElf(/*machine=*/0x04c);
  std::vector<uint8_t> bundle =
      MakeBundle({{"hipv4-amdgcn-amd-amdhsa--gfx942", elf}});

  const iree_hal_executable_target_t executable_targets[] = {
      MakeExecutableTarget(IREE_SV("gfx1100"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100),
      MakeExecutableTarget(IREE_SV("gfx11-generic"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, 50),
  };
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_targets[0]},
      {/*.executable_target=*/&executable_targets[1]},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_extract_for_targets(
          iree_make_const_byte_span(bundle.data(), bundle.size()),
          IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract)),
      StatusIs(iree::StatusCode::kNotFound));
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, SelectsRawElfWithCompatibleFeatures) {
  constexpr uint32_t kSrameccOn = 0xC00;
  constexpr uint32_t kXnackOff = 0x200;
  const auto elf = MakeMinimalAmdgpuElf(
      /*machine=*/0x04c, /*generic_version=*/0, kSrameccOn | kXnackOff);
  const iree_hal_executable_target_t executable_target =
      MakeExecutableTarget(IREE_SV("gfx942:sramecc+:xnack-"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100);
  const iree_hal_streaming_fat_binary_target_t target = {
      /*.executable_target=*/&executable_target,
  };

  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(elf.data(), elf.size()), 1, &target,
      iree_allocator_system(), &extract));

  ASSERT_EQ(extract.match_count, 1);
  EXPECT_EQ(extract.matches[0].executable_target, &executable_target);
  EXPECT_STREQ(extract.matches[0].code_object_target_key,
               "gfx942:sramecc+:xnack-");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(FatBinaryTest, FiltersIncompatibleConcatenatedElfFeatures) {
  constexpr uint32_t kSrameccOn = 0xC00;
  constexpr uint32_t kXnackOff = 0x200;
  constexpr uint32_t kXnackOn = 0x300;
  const auto xnack_on_elf = MakeMinimalAmdgpuElf(
      /*machine=*/0x04c, /*generic_version=*/0, kSrameccOn | kXnackOn);
  const auto xnack_off_elf = MakeMinimalAmdgpuElf(
      /*machine=*/0x04c, /*generic_version=*/0, kSrameccOn | kXnackOff);
  std::vector<uint8_t> concatenated = xnack_on_elf;
  concatenated.insert(concatenated.end(), xnack_off_elf.begin(),
                      xnack_off_elf.end());

  const iree_hal_executable_target_t executable_target =
      MakeExecutableTarget(IREE_SV("gfx942:sramecc+:xnack-"),
                           IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100);
  const iree_hal_streaming_fat_binary_target_t target = {
      /*.executable_target=*/&executable_target,
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_EXPECT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(concatenated.data(), concatenated.size()), 1,
      &target, iree_allocator_system(), &extract));

  ASSERT_EQ(extract.match_count, 1);
  EXPECT_EQ(extract.matches[0].data.data,
            concatenated.data() + xnack_on_elf.size());
  EXPECT_EQ(extract.matches[0].executable_target, &executable_target);
  EXPECT_STREQ(extract.matches[0].code_object_target_key,
               "gfx942:sramecc+:xnack-");
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

}  // namespace
