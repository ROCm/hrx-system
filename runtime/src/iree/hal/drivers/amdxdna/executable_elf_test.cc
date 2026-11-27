// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/schemas/amdxdna_elf_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr uint32_t kShtNull = 0;
constexpr uint32_t kShtProgbits = 1;
constexpr uint32_t kShtStrtab = 3;
constexpr uint32_t kShtRela = 4;
constexpr uint32_t kShtDynsym = 11;
constexpr uint32_t kElf32EhdrSize = 52;
constexpr uint32_t kElf32ShdrSize = 40;
constexpr uint32_t kElf32SymSize = 16;
constexpr uint32_t kElf32RelaSize = 12;

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

static std::vector<uint8_t> BuildVaddLikeElf() {
  const uint32_t words[] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));
  const char* symbols[] = {"0", "1", "2"};
  std::string dynstr(1, '\0');
  uint32_t dynstr_offs[3];
  for (int i = 0; i < 3; ++i) {
    dynstr_offs[i] = static_cast<uint32_t>(dynstr.size());
    dynstr.append(symbols[i]);
    dynstr.push_back('\0');
  }
  std::vector<uint8_t> dynsym(kElf32SymSize, 0);
  for (int i = 0; i < 3; ++i) {
    AppendU32(&dynsym, dynstr_offs[i]);
    AppendU32(&dynsym, 0);
    AppendU32(&dynsym, 0);
    dynsym.push_back(0x10);
    dynsym.push_back(0);
    AppendU16(&dynsym, 1);
  }
  std::vector<uint8_t> rela;
  const uint32_t reloc_offs[] = {0, 8, 16};
  for (int i = 0; i < 3; ++i) {
    AppendU32(&rela, reloc_offs[i]);
    AppendU32(&rela, ((uint32_t)(i + 1) << 8) |
                         IREE_HAL_AMDXDNA_CTRLCODE_ELF_PATCH_SHIM_DMA_AIE4);
    AppendI32(&rela, i * 4);
  }
  const std::string shstrtab(
      std::string("\0.shstrtab\0.ctrltext\0.dynstr\0.dynsym\0.rela.dyn\0", 47));

  std::vector<uint8_t> elf(kElf32EhdrSize, 0);
  auto append_blob = [&](const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    elf.insert(elf.end(), b, b + n);
  };
  const uint32_t ctrl_off = static_cast<uint32_t>(elf.size());
  append_blob(ctrltext.data(), ctrltext.size());
  const uint32_t dynstr_off = static_cast<uint32_t>(elf.size());
  append_blob(dynstr.data(), dynstr.size());
  const uint32_t dynsym_off = static_cast<uint32_t>(elf.size());
  append_blob(dynsym.data(), dynsym.size());
  const uint32_t rela_off = static_cast<uint32_t>(elf.size());
  append_blob(rela.data(), rela.size());
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
  write_shdr(0, kShtNull, 0, 0, 0, 0, 0);
  write_shdr(11, kShtProgbits, ctrl_off, static_cast<uint32_t>(ctrltext.size()),
             0, 0, 0);
  write_shdr(21, kShtStrtab, dynstr_off, static_cast<uint32_t>(dynstr.size()),
             0, 0, 0);
  write_shdr(29, kShtDynsym, dynsym_off, static_cast<uint32_t>(dynsym.size()),
             2, 1, kElf32SymSize);
  write_shdr(37, kShtRela, rela_off, static_cast<uint32_t>(rela.size()), 3, 0,
             kElf32RelaSize);
  write_shdr(1, kShtStrtab, shstr_off, static_cast<uint32_t>(shstrtab.size()),
             0, 0, 0);

  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 1;
  elf[5] = 1;
  elf[6] = 1;
  elf[7] = IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P;
  elf[8] = 0;
  WriteU16(&elf, 16, 1);
  WriteU32(&elf, 20, 1);
  WriteU32(&elf, 32, shoff);
  WriteU16(&elf, 40, kElf32EhdrSize);
  WriteU16(&elf, 46, kElf32ShdrSize);
  WriteU16(&elf, 48, 6);
  WriteU16(&elf, 50, 5);
  return elf;
}

static iree_status_t MakeAelfExecutableWithExports(
    const std::vector<uint8_t>& elf,
    const std::vector<std::string>& export_names, uint32_t binding_count,
    std::vector<uint8_t>* out_data) {
  out_data->clear();
  flatbuffers_builder_t builder;
  if (flatcc_builder_init(&builder) != 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to initialize flatbuffer builder");
  }

  iree_status_t status = iree_ok_status();
  if (flatbuffers_failed(
          iree_hal_amdxdna_elf_ExecutableDef_start_as_root(&builder))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start AELF executable");
  }

  iree_hal_amdxdna_elf_ModuleDef_ref_t module_ref = 0;
  if (iree_status_is_ok(status)) {
    flatbuffers_uint8_vec_ref_t elf_ref =
        flatbuffers_uint8_vec_create(&builder, elf.data(), elf.size());
    module_ref = iree_hal_amdxdna_elf_ModuleDef_create(&builder, elf_ref);
    if (!elf_ref || !module_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create AELF module");
    }
  }

  std::vector<iree_hal_amdxdna_elf_ExportDef_ref_t> export_refs;
  for (const auto& export_name : export_names) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_amdxdna_elf_ExportDef_ref_t export_ref = 0;
    flatbuffers_string_ref_t name_ref =
        flatbuffers_string_create_str(&builder, export_name.c_str());
    if (name_ref &&
        !flatbuffers_failed(iree_hal_amdxdna_elf_ExportDef_start(&builder)) &&
        !flatbuffers_failed(
            iree_hal_amdxdna_elf_ExportDef_module_ordinal_add(&builder, 0)) &&
        !flatbuffers_failed(
            iree_hal_amdxdna_elf_ExportDef_name_add(&builder, name_ref)) &&
        !flatbuffers_failed(iree_hal_amdxdna_elf_ExportDef_binding_count_add(
            &builder, binding_count))) {
      export_ref = iree_hal_amdxdna_elf_ExportDef_end(&builder);
    }
    if (!export_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create AELF export");
    }
    export_refs.push_back(export_ref);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_elf_ModuleDef_vec_ref_t modules_ref =
        iree_hal_amdxdna_elf_ModuleDef_vec_create(&builder, &module_ref, 1);
    iree_hal_amdxdna_elf_ExportDef_vec_ref_t exports_ref =
        iree_hal_amdxdna_elf_ExportDef_vec_create(&builder, export_refs.data(),
                                                  export_refs.size());
    if (!modules_ref || !exports_ref ||
        flatbuffers_failed(iree_hal_amdxdna_elf_ExecutableDef_exports_add(
            &builder, exports_ref)) ||
        flatbuffers_failed(iree_hal_amdxdna_elf_ExecutableDef_modules_add(
            &builder, modules_ref)) ||
        !iree_hal_amdxdna_elf_ExecutableDef_end_as_root(&builder)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to finish AELF executable");
    }
  }

  size_t flatbuffer_size = 0;
  void* flatbuffer_data = nullptr;
  if (iree_status_is_ok(status)) {
    flatbuffer_data =
        flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
    if (!flatbuffer_data || flatbuffer_size == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to finalize AELF executable");
    }
  }
  if (iree_status_is_ok(status)) {
    out_data->assign(
        static_cast<const uint8_t*>(flatbuffer_data),
        static_cast<const uint8_t*>(flatbuffer_data) + flatbuffer_size);
  }
  flatcc_builder_aligned_free(flatbuffer_data);
  flatcc_builder_clear(&builder);
  return status;
}

static iree_status_t MakeAelfExecutable(const std::vector<uint8_t>& elf,
                                        const char* export_name,
                                        uint32_t binding_count,
                                        std::vector<uint8_t>* out_data) {
  return MakeAelfExecutableWithExports(elf, {export_name}, binding_count,
                                       out_data);
}

// Two kernel groups (`k_a`, `k_b`) in an OSABI 70 config ELF, one page each:
// [null] [.ctrltext.0.0.0] [.ctrltext.0.0.1] [.strtab] [.symtab] [.group]
// [.group] [.shstrtab].
static std::vector<uint8_t> BuildTwoKernelGroupElf() {
  const std::string names(
      "\0.shstrtab\0.ctrltext.0.0.0\0.ctrltext.0.0.1\0"
      ".strtab\0.symtab\0.group\0",
      66);
  const std::string strtab("\0k_a\0k_b\0", 9);
  std::vector<uint8_t> symtab(kElf32SymSize, 0);
  for (uint32_t name_off : {1u, 5u}) {
    AppendU32(&symtab, name_off);
    AppendU32(&symtab, 0);
    AppendU32(&symtab, 0);
    symtab.push_back(0x01);
    symtab.push_back(0);
    AppendU16(&symtab, 0);
  }

  std::vector<uint8_t> elf(kElf32EhdrSize, 0);
  auto append_u32_blob = [&](uint32_t value) {
    const uint32_t offset = static_cast<uint32_t>(elf.size());
    AppendU32(&elf, value);
    return offset;
  };
  auto append_blob = [&](const void* p, size_t n) {
    const uint32_t offset = static_cast<uint32_t>(elf.size());
    const auto* b = static_cast<const uint8_t*>(p);
    elf.insert(elf.end(), b, b + n);
    return offset;
  };
  const uint32_t a_off = append_u32_blob(0xA0u);
  const uint32_t b_off = append_u32_blob(0xB0u);
  const uint32_t strtab_off = append_blob(strtab.data(), strtab.size());
  const uint32_t symtab_off = append_blob(symtab.data(), symtab.size());
  const uint32_t group_a_off = append_u32_blob(1);  // GRP_COMDAT
  append_u32_blob(1);                               // .ctrltext.0.0.0
  const uint32_t group_b_off = append_u32_blob(1);
  append_u32_blob(2);  // .ctrltext.0.0.1
  const uint32_t shstr_off = append_blob(names.data(), names.size());
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
  write_shdr(0, kShtNull, 0, 0, 0, 0, 0);
  write_shdr(11, kShtProgbits, a_off, 4, 0, 0, 0);
  write_shdr(27, kShtProgbits, b_off, 4, 0, 0, 0);
  write_shdr(43, kShtStrtab, strtab_off, static_cast<uint32_t>(strtab.size()),
             0, 0, 0);
  write_shdr(51, /*SHT_SYMTAB=*/2, symtab_off,
             static_cast<uint32_t>(symtab.size()), 3, 1, kElf32SymSize);
  write_shdr(59, /*SHT_GROUP=*/17, group_a_off, 8, 4, 1, 4);
  write_shdr(59, /*SHT_GROUP=*/17, group_b_off, 8, 4, 2, 4);
  write_shdr(1, kShtStrtab, shstr_off, static_cast<uint32_t>(names.size()), 0,
             0, 0);

  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 1;
  elf[5] = 1;
  elf[6] = 1;
  elf[7] = IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG;
  elf[8] = 3;
  WriteU16(&elf, 16, 2);  // ET_EXEC
  WriteU32(&elf, 20, 1);
  WriteU32(&elf, 32, shoff);
  WriteU16(&elf, 40, kElf32EhdrSize);
  WriteU16(&elf, 46, kElf32ShdrSize);
  WriteU16(&elf, 48, 8);
  WriteU16(&elf, 50, 7);
  return elf;
}

TEST(ExecutableElfTest, InfersFormatAndLoadsCtrlcode) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(
      MakeAelfExecutable(BuildVaddLikeElf(), "vadd", 3, &executable_data));

  char format[64] = {};
  iree_host_size_t inferred_size = 0;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_infer_format(
      iree_make_const_byte_span(executable_data.data(), executable_data.size()),
      sizeof(format), format, &inferred_size));
  EXPECT_STREQ(format, "amdxdna-elf-fb");
  EXPECT_EQ(inferred_size, executable_data.size());

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  ASSERT_EQ(executable->entry_point_count, 1u);
  const auto& entry = executable->entry_points[0];
  EXPECT_EQ(std::string(entry.kernel_name.data, entry.kernel_name.size),
            "vadd");
  EXPECT_EQ(entry.pdi_image, nullptr);
  EXPECT_EQ(entry.xclbin_image, nullptr);
  ASSERT_EQ(entry.asm_inst_runlist_count, 1u);
  ASSERT_EQ(entry.asm_inst_runlist[0].count, 6u);
  EXPECT_EQ(entry.asm_inst_runlist[0].data[0], 0x10u);
  ASSERT_EQ(entry.patch_runlist_count, 1u);
  ASSERT_EQ(entry.patch_runlist[0].count, 9u);
  EXPECT_EQ(entry.patch_runlist[0].data[0], 0u);
  EXPECT_EQ(entry.patch_runlist[0].data[1], 0u);
  EXPECT_EQ(entry.patch_runlist[0].data[3], 8u);
  EXPECT_EQ(entry.patch_runlist[0].data[4], 1u);
  EXPECT_EQ(entry.patch_runlist[0].data[6], 16u);
  EXPECT_EQ(entry.patch_runlist[0].data[7], 2u);
  iree_hal_executable_release(base_executable);
}

TEST(ExecutableElfTest, LoadsAie4PagedCtrlcodeAndPartitionCols) {
  const uint32_t words[] = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
  std::vector<uint8_t> ctrltext(sizeof(words));
  std::memcpy(ctrltext.data(), words, sizeof(words));

  std::string names;
  names.push_back('\0');
  names.append(".shstrtab");
  names.push_back('\0');
  names.append(".ctrltext.0.0");
  names.push_back('\0');
  names.append(".dynstr");
  names.push_back('\0');
  names.append(".dynsym");
  names.push_back('\0');
  names.append(".rela.dyn");
  names.push_back('\0');
  names.append(".note.xrt.configuration");
  names.push_back('\0');

  std::vector<uint8_t> dynstr(1, 0);
  std::vector<uint8_t> dynsym(kElf32SymSize, 0);
  std::vector<uint8_t> rela;
  std::vector<uint8_t> note;
  AppendU32(&note, 4);
  AppendU32(&note, 4);
  AppendU32(&note, 1);
  note.push_back('X');
  note.push_back('R');
  note.push_back('T');
  note.push_back('\0');
  AppendU32(&note, 3);

  std::vector<uint8_t> elf(kElf32EhdrSize, 0);
  auto append_blob = [&](const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    elf.insert(elf.end(), b, b + n);
  };
  const uint32_t ctrl_off = static_cast<uint32_t>(elf.size());
  append_blob(ctrltext.data(), ctrltext.size());
  const uint32_t dynstr_off = static_cast<uint32_t>(elf.size());
  append_blob(dynstr.data(), dynstr.size());
  const uint32_t dynsym_off = static_cast<uint32_t>(elf.size());
  append_blob(dynsym.data(), dynsym.size());
  const uint32_t rela_off = static_cast<uint32_t>(elf.size());
  append_blob(rela.data(), rela.size());
  const uint32_t note_off = static_cast<uint32_t>(elf.size());
  append_blob(note.data(), note.size());
  const uint32_t shstr_off = static_cast<uint32_t>(elf.size());
  append_blob(names.data(), names.size());
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
  write_shdr(0, kShtNull, 0, 0, 0, 0, 0);
  write_shdr(11, kShtProgbits, ctrl_off, static_cast<uint32_t>(ctrltext.size()),
             0, 0, 0);
  write_shdr(25, kShtStrtab, dynstr_off, static_cast<uint32_t>(dynstr.size()),
             0, 0, 0);
  write_shdr(33, kShtDynsym, dynsym_off, static_cast<uint32_t>(dynsym.size()),
             2, 1, kElf32SymSize);
  write_shdr(41, kShtRela, rela_off, static_cast<uint32_t>(rela.size()), 3, 0,
             kElf32RelaSize);
  write_shdr(51, 7, note_off, static_cast<uint32_t>(note.size()), 0, 0, 0);
  write_shdr(1, kShtStrtab, shstr_off, static_cast<uint32_t>(names.size()), 0,
             0, 0);

  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 1;
  elf[5] = 1;
  elf[6] = 1;
  elf[7] = IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4;
  elf[8] = 0x21;
  WriteU16(&elf, 16, 1);
  WriteU32(&elf, 20, 1);
  WriteU32(&elf, 32, shoff);
  WriteU16(&elf, 40, kElf32EhdrSize);
  WriteU16(&elf, 46, kElf32ShdrSize);
  WriteU16(&elf, 48, 7);
  WriteU16(&elf, 50, 6);

  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeAelfExecutable(elf, "vadd", 3, &executable_data));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  ASSERT_EQ(executable->entry_point_count, 1u);
  const auto& entry = executable->entry_points[0];
  EXPECT_EQ(entry.partition_cols, 3u);
  EXPECT_EQ(entry.pdi_image, nullptr);
  ASSERT_EQ(entry.asm_inst_runlist_count, 1u);
  EXPECT_EQ(entry.asm_inst_runlist[0].count, 8192u / 4u);
  EXPECT_EQ(entry.asm_inst_runlist[0].data[0], 0xAAu);
  iree_hal_executable_release(base_executable);
}

TEST(ExecutableElfTest, LoadsOneEntryPointPerKernelGroup) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeAelfExecutableWithExports(
      BuildTwoKernelGroupElf(), {"main:k_b", "main:k_a"}, 6, &executable_data));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  ASSERT_EQ(executable->entry_point_count, 2u);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const auto& entry = executable->entry_points[i];
    ASSERT_EQ(entry.asm_inst_runlist_count, 1u);
    ASSERT_EQ(entry.asm_inst_runlist[0].count, 8192u / 4u);
    EXPECT_EQ(entry.asm_inst_runlist[0].data[0], i == 0 ? 0xB0u : 0xA0u);
    EXPECT_EQ(entry.dpu_slice_count, 1u);
  }
  iree_hal_executable_release(base_executable);
}

TEST(ExecutableElfTest, RejectsUnknownKernelGroupExport) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeAelfExecutable(BuildTwoKernelGroupElf(), "main:k_c", 6,
                                    &executable_data));
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* base_executable = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
}

TEST(ExecutableElfTest, RejectsEmptyModule) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeAelfExecutable({}, "vadd", 3, &executable_data));
  char format[64] = {};
  iree_host_size_t inferred_size = 0;
  iree_status_t status = iree_hal_amdxdna_native_executable_infer_format(
      iree_make_const_byte_span(executable_data.data(), executable_data.size()),
      sizeof(format), format, &inferred_size);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

}  // namespace
