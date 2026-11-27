// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr uint32_t kShtNull = 0;
constexpr uint32_t kShtProgbits = 1;
constexpr uint32_t kShtSymtab = 2;
constexpr uint32_t kShtStrtab = 3;
constexpr uint32_t kShtRela = 4;
constexpr uint32_t kShtNote = 7;
constexpr uint32_t kShtDynsym = 11;
constexpr uint32_t kShtGroup = 17;
constexpr uint32_t kElf32EhdrSize = 52;
constexpr uint32_t kElf32ShdrSize = 40;
constexpr uint32_t kElf32SymSize = 16;
constexpr uint32_t kElf32RelaSize = 12;
constexpr uint32_t kPatchShimDmaAie4 = 6;

struct RelocDef {
  uint32_t offset = 0;
  std::string symbol;
  uint32_t type = kPatchShimDmaAie4;
  int32_t addend = 0;
  std::string section;
};

struct CtrltextSec {
  std::string name;
  std::vector<uint8_t> bytes;
};

// One kernel of a multi-kernel ELF: an SHT_GROUP named by `symbol` that owns
// the sections named in `members`.
struct GroupDef {
  std::string symbol;
  std::vector<std::string> members;
};

static void WriteU16(std::vector<uint8_t>* b, size_t off, uint16_t v) {
  std::memcpy(b->data() + off, &v, sizeof(v));
}

static void WriteU32(std::vector<uint8_t>* b, size_t off, uint32_t v) {
  std::memcpy(b->data() + off, &v, sizeof(v));
}

static void AppendU16(std::vector<uint8_t>* b, uint16_t v) {
  uint8_t bytes[2];
  std::memcpy(bytes, &v, sizeof(v));
  b->insert(b->end(), bytes, bytes + 2);
}

static void AppendU32(std::vector<uint8_t>* b, uint32_t v) {
  uint8_t bytes[4];
  std::memcpy(bytes, &v, sizeof(v));
  b->insert(b->end(), bytes, bytes + 4);
}

static void AppendI32(std::vector<uint8_t>* b, int32_t v) {
  uint8_t bytes[4];
  std::memcpy(bytes, &v, sizeof(v));
  b->insert(b->end(), bytes, bytes + 4);
}

static uint16_t CtrltextShndx(const std::vector<CtrltextSec>& texts,
                              const std::string& section) {
  if (section.empty()) return 1;
  for (size_t i = 0; i < texts.size(); ++i) {
    if (texts[i].name == section) {
      return static_cast<uint16_t>(i + 1);
    }
  }
  return 1;
}

static std::vector<uint8_t> BuildPagedCtrlcodeElf(
    uint8_t os_abi, uint8_t abi_version, const std::vector<CtrltextSec>& texts,
    const std::vector<RelocDef>& relocs, uint32_t partition_cols = 0,
    const std::vector<GroupDef>& groups = {}) {
  struct DynsymEntry {
    std::string name;
    std::string section;
  };
  std::vector<DynsymEntry> symbols;
  for (const auto& reloc : relocs) {
    bool seen = false;
    for (const auto& existing : symbols) {
      if (existing.name == reloc.symbol && existing.section == reloc.section) {
        seen = true;
        break;
      }
    }
    if (!seen) symbols.push_back({reloc.symbol, reloc.section});
  }

  std::string dynstr(1, '\0');
  std::vector<uint32_t> dynstr_offs(symbols.size(), 0);
  for (size_t i = 0; i < symbols.size(); ++i) {
    dynstr_offs[i] = static_cast<uint32_t>(dynstr.size());
    dynstr.append(symbols[i].name);
    dynstr.push_back('\0');
  }

  std::vector<uint8_t> dynsym(kElf32SymSize, 0);  // NULL symbol.
  for (size_t i = 0; i < symbols.size(); ++i) {
    AppendU32(&dynsym, dynstr_offs[i]);
    AppendU32(&dynsym, 0);
    AppendU32(&dynsym, 0);
    dynsym.push_back(0x10);  // STB_GLOBAL, STT_NOTYPE
    dynsym.push_back(0);
    AppendU16(&dynsym, CtrltextShndx(texts, symbols[i].section));
  }

  std::vector<uint8_t> rela;
  for (const auto& reloc : relocs) {
    uint32_t symidx = 0;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i].name == reloc.symbol &&
          symbols[i].section == reloc.section) {
        symidx = static_cast<uint32_t>(i + 1);
        break;
      }
    }
    AppendU32(&rela, reloc.offset);
    AppendU32(&rela, (symidx << 8) | (reloc.type & 0xFFu));
    AppendI32(&rela, reloc.addend);
  }

  std::string shstrtab("\0.shstrtab\0", 11);
  std::vector<uint32_t> text_name_offs(texts.size(), 0);
  for (size_t i = 0; i < texts.size(); ++i) {
    text_name_offs[i] = static_cast<uint32_t>(shstrtab.size());
    shstrtab.append(texts[i].name);
    shstrtab.push_back('\0');
  }
  const uint32_t dynstr_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".dynstr");
  shstrtab.push_back('\0');
  const uint32_t dynsym_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".dynsym");
  shstrtab.push_back('\0');
  const uint32_t rela_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".rela.dyn");
  shstrtab.push_back('\0');
  uint32_t note_name_off = 0;
  if (partition_cols != 0) {
    note_name_off = static_cast<uint32_t>(shstrtab.size());
    shstrtab.append(".note.xrt.configuration");
    shstrtab.push_back('\0');
  }
  const uint32_t strtab_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".strtab");
  shstrtab.push_back('\0');
  const uint32_t symtab_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".symtab");
  shstrtab.push_back('\0');
  const uint32_t group_name_off = static_cast<uint32_t>(shstrtab.size());
  shstrtab.append(".group");
  shstrtab.push_back('\0');
  const uint32_t shstrtab_name = 1;

  // Group symbols live in .symtab (via .strtab); each group lists its member
  // section indices after a GRP_COMDAT flags word.
  std::string strtab(1, '\0');
  std::vector<uint8_t> symtab(kElf32SymSize, 0);  // NULL symbol.
  std::vector<std::vector<uint8_t>> group_blobs;
  for (const auto& group : groups) {
    AppendU32(&symtab, static_cast<uint32_t>(strtab.size()));
    AppendU32(&symtab, 0);
    AppendU32(&symtab, 0);
    symtab.push_back(0x01);  // STB_LOCAL, STT_OBJECT
    symtab.push_back(0);
    AppendU16(&symtab, 0);
    strtab.append(group.symbol);
    strtab.push_back('\0');
    std::vector<uint8_t> blob;
    AppendU32(&blob, 1);  // GRP_COMDAT
    for (const auto& member : group.members) {
      AppendU32(&blob, CtrltextShndx(texts, member));
    }
    group_blobs.push_back(std::move(blob));
  }

  std::vector<uint8_t> note;
  if (partition_cols != 0) {
    AppendU32(&note, 4);  // namesz "XRT"
    AppendU32(&note, 4);  // descsz
    AppendU32(&note, 1);  // type
    note.push_back('X');
    note.push_back('R');
    note.push_back('T');
    note.push_back('\0');
    AppendU32(&note, partition_cols);
  }

  std::vector<uint8_t> elf(kElf32EhdrSize, 0);
  auto append_blob = [&](const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    elf.insert(elf.end(), b, b + n);
  };

  std::vector<uint32_t> text_offs(texts.size(), 0);
  for (size_t i = 0; i < texts.size(); ++i) {
    text_offs[i] = static_cast<uint32_t>(elf.size());
    append_blob(texts[i].bytes.data(), texts[i].bytes.size());
  }
  const uint32_t dynstr_off = static_cast<uint32_t>(elf.size());
  append_blob(dynstr.data(), dynstr.size());
  const uint32_t dynsym_off = static_cast<uint32_t>(elf.size());
  append_blob(dynsym.data(), dynsym.size());
  const uint32_t rela_off = static_cast<uint32_t>(elf.size());
  append_blob(rela.data(), rela.size());
  const uint32_t note_off = static_cast<uint32_t>(elf.size());
  if (!note.empty()) append_blob(note.data(), note.size());
  const uint32_t strtab_off = static_cast<uint32_t>(elf.size());
  append_blob(strtab.data(), strtab.size());
  const uint32_t symtab_off = static_cast<uint32_t>(elf.size());
  append_blob(symtab.data(), symtab.size());
  std::vector<uint32_t> group_offs;
  for (const auto& blob : group_blobs) {
    group_offs.push_back(static_cast<uint32_t>(elf.size()));
    append_blob(blob.data(), blob.size());
  }
  const uint32_t shstr_off = static_cast<uint32_t>(elf.size());
  append_blob(shstrtab.data(), shstrtab.size());
  const uint32_t shoff = static_cast<uint32_t>(elf.size());

  auto write_shdr = [&](uint32_t name, uint32_t type, uint32_t offset,
                        uint32_t size, uint32_t link, uint32_t info,
                        uint32_t entsize) {
    size_t at = elf.size();
    elf.resize(at + kElf32ShdrSize, 0);
    WriteU32(&elf, at + 0, name);
    WriteU32(&elf, at + 4, type);
    WriteU32(&elf, at + 16, offset);
    WriteU32(&elf, at + 20, size);
    WriteU32(&elf, at + 24, link);
    WriteU32(&elf, at + 28, info);
    WriteU32(&elf, at + 36, entsize);
  };

  const uint16_t dynstr_ndx = static_cast<uint16_t>(texts.size() + 1);
  const uint16_t dynsym_ndx = static_cast<uint16_t>(dynstr_ndx + 1);
  write_shdr(0, kShtNull, 0, 0, 0, 0, 0);
  for (size_t i = 0; i < texts.size(); ++i) {
    write_shdr(text_name_offs[i], kShtProgbits, text_offs[i],
               static_cast<uint32_t>(texts[i].bytes.size()), 0, 0, 0);
  }
  write_shdr(dynstr_name_off, kShtStrtab, dynstr_off,
             static_cast<uint32_t>(dynstr.size()), 0, 0, 0);
  write_shdr(dynsym_name_off, kShtDynsym, dynsym_off,
             static_cast<uint32_t>(dynsym.size()), dynstr_ndx, /*info=*/1,
             kElf32SymSize);
  write_shdr(rela_name_off, kShtRela, rela_off, static_cast<uint32_t>(rela.size()),
             dynsym_ndx, 0, kElf32RelaSize);
  if (!note.empty()) {
    write_shdr(note_name_off, kShtNote, note_off,
               static_cast<uint32_t>(note.size()), 0, 0, 0);
  }
  const uint16_t strtab_ndx =
      static_cast<uint16_t>(1 + texts.size() + 3 + (note.empty() ? 0 : 1));
  const uint16_t symtab_ndx = static_cast<uint16_t>(strtab_ndx + 1);
  if (!groups.empty()) {
    write_shdr(strtab_name_off, kShtStrtab, strtab_off,
               static_cast<uint32_t>(strtab.size()), 0, 0, 0);
    write_shdr(symtab_name_off, kShtSymtab, symtab_off,
               static_cast<uint32_t>(symtab.size()), strtab_ndx, /*info=*/1,
               kElf32SymSize);
    for (size_t i = 0; i < groups.size(); ++i) {
      write_shdr(group_name_off, kShtGroup, group_offs[i],
                 static_cast<uint32_t>(group_blobs[i].size()), symtab_ndx,
                 /*info=*/static_cast<uint32_t>(i + 1), 4);
    }
  }
  write_shdr(shstrtab_name, kShtStrtab, shstr_off,
             static_cast<uint32_t>(shstrtab.size()), 0, 0, 0);

  const uint16_t shnum = static_cast<uint16_t>(
      strtab_ndx + (groups.empty() ? 0 : 2 + groups.size()) + 1);
  const uint16_t shstrndx = static_cast<uint16_t>(shnum - 1);

  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 1;
  elf[5] = 1;
  elf[6] = 1;
  elf[7] = os_abi;
  elf[8] = abi_version;
  WriteU16(&elf, 16, 1);  // ET_REL
  WriteU16(&elf, 18, 0);
  WriteU32(&elf, 20, 1);
  WriteU32(&elf, 32, shoff);
  WriteU16(&elf, 40, kElf32EhdrSize);
  WriteU16(&elf, 46, kElf32ShdrSize);
  WriteU16(&elf, 48, shnum);
  WriteU16(&elf, 50, shstrndx);
  return elf;
}

static std::vector<uint8_t> BuildCtrlcodeElf(uint8_t os_abi, uint8_t abi_version,
                                             const std::string& ctrltext_name,
                                             const std::vector<uint8_t>& ctrltext,
                                             const std::vector<RelocDef>& relocs) {
  return BuildPagedCtrlcodeElf(os_abi, abi_version,
                               {{ctrltext_name, ctrltext}}, relocs);
}

TEST(CtrlcodeElfTest, ParsesAie2pCtrltextAndNumericReloc) {
  const uint32_t words[] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P, 0,
                              ".ctrltext", ctrltext,
                              {{/*offset=*/0, "0", kPatchShimDmaAie4, 16}});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.os_abi, IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P);
  ASSERT_EQ(parsed.ctrl_word_count, 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0x11111111u);
  EXPECT_EQ(parsed.ctrl_words[3], 0x44444444u);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 0u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.patches[2], 16u);
  EXPECT_EQ(parsed.skipped_non_host_relocs, 0u);
  EXPECT_EQ(parsed.dpu_slice_count, 0u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, ParsesArgvSymbolAndKeepsControlCodeSelfPatch) {
  const uint32_t words[] = {0, 1, 2, 3, 4, 5};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P, 0, ".ctrltext", ctrltext,
      {{0, "argv1", kPatchShimDmaAie4, 8},
       {8, "control-code-0", kPatchShimDmaAie4, 0}});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  ASSERT_EQ(parsed.patch_count, 6u);
  EXPECT_EQ(parsed.patches[0], 0u);
  EXPECT_EQ(parsed.patches[1], 1u);
  EXPECT_EQ(parsed.patches[2], 8u);
  EXPECT_EQ(parsed.patches[3], 8u);
  EXPECT_EQ(parsed.patches[4], 0xFFFFFFFFu);
  EXPECT_EQ(parsed.patches[5], 0u);
  EXPECT_EQ(parsed.skipped_non_host_relocs, 0u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, SkipsScratchPadAndControlPacketRelocs) {
  const uint32_t words[] = {0, 1, 2, 3, 4, 5};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P, 0, ".ctrltext", ctrltext,
      {{0, "argv0", kPatchShimDmaAie4, 4},
       {4, "scratch-pad-mem", kPatchShimDmaAie4, 0},
       {8, "control-packet", kPatchShimDmaAie4, 0}});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 0u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.patches[2], 4u);
  EXPECT_EQ(parsed.skipped_non_host_relocs, 2u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, PadsAie2psPageAndAppliesPageRelocBias) {
  const uint32_t words[] = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS, 0,
                              ".ctrltext.0.0", ctrltext,
                              {{/*offset=*/0, "2", kPatchShimDmaAie4, 4}});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.os_abi, IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS);
  EXPECT_EQ(parsed.ctrl_word_count, 8192u / 4u);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 16u);
  EXPECT_EQ(parsed.patches[1], 2u);
  EXPECT_EQ(parsed.patches[2], 4u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, UnpacksAbiVersion1AddendScheme) {
  const uint32_t words[] = {1, 2, 3, 4};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  const int32_t packed_addend =
      (24 << 4) | IREE_HAL_AMDXDNA_CTRLCODE_ELF_PATCH_SHIM_DMA_AIE4;
  auto elf = BuildCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P, 1,
                              ".ctrltext", ctrltext,
                              {{4, "0", /*type unused when abi=1*/0,
                                packed_addend}});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 4u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.patches[2], 24u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, PadsAie4PagedCtrltext) {
  const uint32_t words[] = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21,
                              ".ctrltext.0.0", ctrltext, {});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.os_abi, IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4);
  EXPECT_EQ(parsed.ctrl_word_count, 8192u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0xAAu);
  EXPECT_EQ(parsed.patch_count, 0u);
  ASSERT_EQ(parsed.dpu_slice_count, 1u);
  EXPECT_EQ(parsed.dpu_slices[0].uc_index, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_offset, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_size, 8192u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, PadsAie4TwoColumnsAndBiasesRelocToColumnBase) {
  std::vector<uint8_t> col0(4, 0);
  WriteU32(&col0, 0, 0xAAu);
  std::vector<uint8_t> col1(4, 0);
  WriteU32(&col1, 0, 0xBBu);
  RelocDef reloc;
  reloc.offset = 0;
  reloc.symbol = "0";
  reloc.section = ".ctrltext.1.0";
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21,
      {{".ctrltext.0.0", col0}, {".ctrltext.1.0", col1}}, {reloc});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.ctrl_word_count, 16384u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0xAAu);
  EXPECT_EQ(parsed.ctrl_words[8192u / 4u], 0xBBu);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 8192u + 16u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.patches[2], 0u);
  ASSERT_EQ(parsed.dpu_slice_count, 2u);
  EXPECT_EQ(parsed.dpu_slices[0].uc_index, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_offset, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_size, 8192u);
  EXPECT_EQ(parsed.dpu_slices[1].uc_index, 1u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_offset, 8192u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_size, 8192u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, PacksSparseAie4ColumnsWithoutEmptyPages) {
  std::vector<uint8_t> col0(4, 0);
  WriteU32(&col0, 0, 0x11u);
  std::vector<uint8_t> col2(4, 0);
  WriteU32(&col2, 0, 0x22u);
  RelocDef reloc;
  reloc.offset = 0;
  reloc.symbol = "1";
  reloc.section = ".ctrltext.2.0";
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21,
      {{".ctrltext.0.0", col0}, {".ctrltext.2.0", col2}}, {reloc});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.ctrl_word_count, 16384u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0x11u);
  EXPECT_EQ(parsed.ctrl_words[8192u / 4u], 0x22u);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 8192u + 16u);
  EXPECT_EQ(parsed.patches[1], 1u);
  ASSERT_EQ(parsed.dpu_slice_count, 2u);
  EXPECT_EQ(parsed.dpu_slices[0].uc_index, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_size, 8192u);
  EXPECT_EQ(parsed.dpu_slices[1].uc_index, 2u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_offset, 8192u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_size, 8192u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, DeduplicatesAndSortsAie4Patches) {
  std::vector<uint8_t> page(32, 0);
  auto reloc = [](uint32_t offset, const char* symbol) {
    RelocDef r;
    r.offset = offset;
    r.symbol = symbol;
    r.section = ".ctrltext.0.0";
    return r;
  };
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21, {{".ctrltext.0.0", page}},
      {reloc(8, "1"), reloc(0, "0"), reloc(8, "1")});
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()),
      iree_allocator_system(), &parsed));
  ASSERT_EQ(parsed.patch_count, 6u);
  EXPECT_EQ(parsed.patches[0], 16u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.patches[3], 16u + 8u);
  EXPECT_EQ(parsed.patches[4], 1u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, AppendsPadInsideOwningColumn) {
  std::vector<uint8_t> col0(4, 0);
  WriteU32(&col0, 0, 0x11u);
  std::vector<uint8_t> col2(4, 0);
  WriteU32(&col2, 0, 0x22u);
  std::vector<uint8_t> pad(4, 0);
  WriteU32(&pad, 0, 0xABu);
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21,
      {{".ctrltext.0.0", col0}, {".pad.0.0", pad}, {".ctrltext.2.0", col2}},
      {});

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.ctrl_word_count, (8192u + 4u + 8192u) / 4u);
  EXPECT_EQ(parsed.ctrl_words[8192u / 4u], 0xABu);
  EXPECT_EQ(parsed.ctrl_words[(8192u + 4u) / 4u], 0x22u);
  ASSERT_EQ(parsed.dpu_slice_count, 2u);
  EXPECT_EQ(parsed.dpu_slices[0].uc_index, 0u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_size, 8192u + 4u);
  EXPECT_EQ(parsed.dpu_slices[1].uc_index, 2u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_offset, 8192u + 4u);
  EXPECT_EQ(parsed.dpu_slices[1].byte_size, 8192u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, ReadsPartitionColsFromXrtNote) {
  const uint32_t words[] = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4, 0x21,
      {{".ctrltext.0.0", ctrltext}}, {}, /*partition_cols=*/3);

  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed));
  EXPECT_EQ(parsed.partition_cols, 3u);
  EXPECT_EQ(parsed.ctrl_word_count, 8192u / 4u);
  ASSERT_EQ(parsed.dpu_slice_count, 1u);
  EXPECT_EQ(parsed.dpu_slices[0].byte_size, 8192u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, RejectsSingleNonZeroColumn) {
  std::vector<uint8_t> col2(4, 0);
  WriteU32(&col2, 0, 0x22u);
  auto elf = BuildPagedCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4,
                                   0x21, {{".ctrltext.2.0", col2}}, {});
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  iree_status_t status = iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
}

TEST(CtrlcodeElfTest, RejectsUnsupportedOsAbi) {
  const uint32_t words[] = {1, 2, 3, 4};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  auto elf = BuildCtrlcodeElf(1, 0, ".ctrltext", ctrltext, {});
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  iree_status_t status = iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()), iree_allocator_system(),
      &parsed);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
}

static std::vector<uint8_t> Word(uint32_t value) {
  std::vector<uint8_t> bytes(4, 0);
  WriteU32(&bytes, 0, value);
  return bytes;
}

// Two kernels shaped like a multi-kernel config ELF: pages on uC 0 and 2,
// `.<col>.<page>.<kernel>` names, one shared .rela.dyn.
static std::vector<uint8_t> BuildTwoKernelElf() {
  RelocDef a_reloc;
  a_reloc.offset = 0;
  a_reloc.symbol = "0";
  a_reloc.section = ".ctrltext.2.0.0";
  RelocDef b_reloc;
  b_reloc.offset = 4;
  b_reloc.symbol = "1";
  b_reloc.section = ".ctrltext.0.0.1";
  return BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG, 3,
      {{".ctrltext.0.0.0", Word(0xA0u)},
       {".ctrltext.2.0.0", Word(0xA2u)},
       {".ctrltext.0.0.1", Word(0xB0u)},
       {".ctrltext.2.0.1", Word(0xB2u)}},
      {a_reloc, b_reloc}, /*partition_cols=*/3,
      {{"k_a", {".ctrltext.0.0.0", ".ctrltext.2.0.0"}},
       {"k_b", {".ctrltext.0.0.1", ".ctrltext.2.0.1"}}});
}

TEST(CtrlcodeElfTest, ParsesKernelGroupByName) {
  auto elf = BuildTwoKernelElf();
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      iree_make_const_byte_span(elf.data(), elf.size()), IREE_SV("k_a"),
      iree_allocator_system(), &parsed));
  EXPECT_EQ(parsed.partition_cols, 3u);
  ASSERT_EQ(parsed.ctrl_word_count, 16384u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0xA0u);
  EXPECT_EQ(parsed.ctrl_words[8192u / 4u], 0xA2u);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 8192u + 16u);
  EXPECT_EQ(parsed.patches[1], 0u);
  EXPECT_EQ(parsed.skipped_non_host_relocs, 0u);
  ASSERT_EQ(parsed.dpu_slice_count, 2u);
  EXPECT_EQ(parsed.dpu_slices[0].uc_index, 0u);
  EXPECT_EQ(parsed.dpu_slices[1].uc_index, 2u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, ParsesKernelGroupByXrtQualifiedName) {
  auto elf = BuildTwoKernelElf();
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      iree_make_const_byte_span(elf.data(), elf.size()), IREE_SV("main:k_b"),
      iree_allocator_system(), &parsed));
  ASSERT_EQ(parsed.ctrl_word_count, 16384u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0xB0u);
  EXPECT_EQ(parsed.ctrl_words[8192u / 4u], 0xB2u);
  ASSERT_EQ(parsed.patch_count, 3u);
  EXPECT_EQ(parsed.patches[0], 16u + 4u);
  EXPECT_EQ(parsed.patches[1], 1u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, RejectsUnnamedOrUnknownKernelInMultiKernelElf) {
  auto elf = BuildTwoKernelElf();
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  iree_status_t status = iree_hal_amdxdna_ctrlcode_elf_parse(
      iree_make_const_byte_span(elf.data(), elf.size()),
      iree_allocator_system(), &parsed);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  status = iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      iree_make_const_byte_span(elf.data(), elf.size()), IREE_SV("main:k_c"),
      iree_allocator_system(), &parsed);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
}

TEST(CtrlcodeElfTest, ParsesEveryKernelFromOneIndex) {
  auto elf = BuildTwoKernelElf();
  iree_hal_amdxdna_ctrlcode_elf_index_t* index = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_index_create(
      iree_make_const_byte_span(elf.data(), elf.size()),
      iree_allocator_system(), &index));
  ASSERT_EQ(iree_hal_amdxdna_ctrlcode_elf_index_kernel_count(index), 2u);
  const uint32_t want_word[] = {0xA0u, 0xB0u};
  const uint32_t want_patch[] = {8192u + 16u, 16u + 4u};
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
    IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_index_parse_kernel(
        index, iree_hal_amdxdna_ctrlcode_elf_index_kernel_name(index, i),
        iree_allocator_system(), &parsed));
    EXPECT_EQ(parsed.ctrl_words[0], want_word[i]);
    ASSERT_EQ(parsed.patch_count, 3u);
    EXPECT_EQ(parsed.patches[0], want_patch[i]);
    iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(),
                                               &parsed);
  }
  iree_hal_amdxdna_ctrlcode_elf_index_free(index);
}

TEST(CtrlcodeElfTest, EnumeratesKernelGroupsInSectionOrder) {
  auto elf = BuildTwoKernelElf();
  std::vector<std::string> names;
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_enumerate_kernels(
      iree_make_const_byte_span(elf.data(), elf.size()),
      iree_allocator_system(),
      [](void* user_data, iree_string_view_t name) {
        static_cast<std::vector<std::string>*>(user_data)->emplace_back(
            name.data, name.size);
        return iree_ok_status();
      },
      &names));
  EXPECT_EQ(names, (std::vector<std::string>{"k_a", "k_b"}));
}

TEST(CtrlcodeElfTest, ParsesKernelFromElfWithManySections) {
  // Config ELFs can have thousands of sections; more than one page table's
  // worth of sections must not be a limit.
  constexpr int kKernels = 600;
  std::vector<CtrltextSec> texts;
  std::vector<GroupDef> groups;
  for (int i = 0; i < kKernels; ++i) {
    const std::string section = ".ctrltext.0.0." + std::to_string(i);
    texts.push_back({section, Word(0x1000u + i)});
    groups.push_back({"k" + std::to_string(i), {section}});
  }
  auto elf = BuildPagedCtrlcodeElf(
      IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG, 3, texts, {},
      /*partition_cols=*/3, groups);
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      iree_make_const_byte_span(elf.data(), elf.size()), IREE_SV("main:k599"),
      iree_allocator_system(), &parsed));
  ASSERT_EQ(parsed.ctrl_word_count, 8192u / 4u);
  EXPECT_EQ(parsed.ctrl_words[0], 0x1000u + 599u);
  ASSERT_EQ(parsed.dpu_slice_count, 1u);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

TEST(CtrlcodeElfTest, IgnoresKernelNameWithoutGroups) {
  auto elf = BuildPagedCtrlcodeElf(IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4,
                                   0x21, {{".ctrltext.0.0", Word(0xAAu)}}, {});
  iree_hal_amdxdna_ctrlcode_elf_t parsed = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      iree_make_const_byte_span(elf.data(), elf.size()), IREE_SV("vadd"),
      iree_allocator_system(), &parsed));
  EXPECT_EQ(parsed.ctrl_words[0], 0xAAu);
  iree_hal_amdxdna_ctrlcode_elf_deinitialize(iree_allocator_system(), &parsed);
}

}  // namespace
