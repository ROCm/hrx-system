// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  IREE_HAL_AMDXDNA_ELF32_CLASS = 1,
  IREE_HAL_AMDXDNA_ELF32_DATA_LE = 1,
  IREE_HAL_AMDXDNA_ELF32_VERSION = 1,
  IREE_HAL_AMDXDNA_ELF32_EHDR_SIZE = 52,
  IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE = 40,
  IREE_HAL_AMDXDNA_ELF32_SYM_SIZE = 16,
  IREE_HAL_AMDXDNA_ELF32_RELA_SIZE = 12,
  IREE_HAL_AMDXDNA_ELF_EI_CLASS = 4,
  IREE_HAL_AMDXDNA_ELF_EI_DATA = 5,
  IREE_HAL_AMDXDNA_ELF_EI_VERSION = 6,
  IREE_HAL_AMDXDNA_ELF_EI_OSABI = 7,
  IREE_HAL_AMDXDNA_ELF_EI_ABIVERSION = 8,
  IREE_HAL_AMDXDNA_ELF32_E_SHOFF = 0x20,
  IREE_HAL_AMDXDNA_ELF32_E_SHENTSIZE = 0x2E,
  IREE_HAL_AMDXDNA_ELF32_E_SHNUM = 0x30,
  IREE_HAL_AMDXDNA_ELF32_E_SHSTRNDX = 0x32,
  IREE_HAL_AMDXDNA_ELF32_SH_NAME = 0,
  IREE_HAL_AMDXDNA_ELF32_SH_TYPE = 0x04,
  IREE_HAL_AMDXDNA_ELF32_SH_OFFSET = 0x10,
  IREE_HAL_AMDXDNA_ELF32_SH_SIZE = 0x14,
  IREE_HAL_AMDXDNA_ELF32_SH_LINK = 0x18,
  IREE_HAL_AMDXDNA_ELF32_SH_INFO = 0x1C,
  IREE_HAL_AMDXDNA_ELF32_SH_ENTSIZE = 0x24,
  IREE_HAL_AMDXDNA_ELF_SHT_GROUP = 17,
  IREE_HAL_AMDXDNA_ELF32_ST_NAME = 0,
  IREE_HAL_AMDXDNA_ELF32_ST_SHNDX = 14,
  IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE = 8192,
  IREE_HAL_AMDXDNA_CTRLCODE_PAGE_HDR = 16,
  IREE_HAL_AMDXDNA_CTRLCODE_ADDEND_SHIFT = 4,
  IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS = 16,
  // KMD fill_indirect_pkt rejects uc_index or chained >= this
  // (HSA_MAX_LEVEL1_INDIRECT_ENTRIES). chained on the first entry is count-1,
  // so six occupied columns is the maximum indirect packet.
  IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES = 6,
};

typedef struct iree_hal_amdxdna_ctrl_page_t {
  uint32_t col;
  uint32_t page;
  uint32_t ctrltext_offset;
  uint32_t ctrltext_size;
  uint32_t ctrldata_offset;
  uint32_t ctrldata_size;
  bool has_ctrltext;
  bool has_ctrldata;
} iree_hal_amdxdna_ctrl_page_t;

typedef struct iree_hal_amdxdna_elf_section_t {
  uint32_t name_off;
  uint32_t type;
  uint32_t file_offset;
  uint32_t size;
  uint32_t link;
  uint32_t info;
  uint32_t entsize;
  uint32_t index;
} iree_hal_amdxdna_elf_section_t;

void iree_hal_amdxdna_ctrlcode_elf_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_ctrlcode_elf_t* elf) {
  if (!elf) return;
  iree_allocator_free(host_allocator, elf->ctrl_words);
  iree_allocator_free(host_allocator, elf->patches);
  iree_allocator_free(host_allocator, elf->dpu_slices);
  memset(elf, 0, sizeof(*elf));
}

static bool iree_hal_amdxdna_elf_range_in_bounds(size_t data_size,
                                                 uint32_t offset,
                                                 uint32_t size) {
  return (size_t)offset <= data_size &&
         (size_t)size <= data_size - (size_t)offset;
}

static uint16_t iree_hal_amdxdna_read_u16(const uint8_t* data, size_t offset) {
  uint16_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static uint32_t iree_hal_amdxdna_read_u32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static int32_t iree_hal_amdxdna_read_i32(const uint8_t* data, size_t offset) {
  int32_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static iree_status_t iree_hal_amdxdna_elf_section_name(
    iree_const_byte_span_t elf, const iree_hal_amdxdna_elf_section_t* shstr,
    const iree_hal_amdxdna_elf_section_t* section, iree_string_view_t* out) {
  *out = iree_string_view_empty();
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, shstr->file_offset,
                                            shstr->size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF shstrtab is out of range");
  }
  if (section->name_off >= shstr->size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section name is out of range");
  }
  const char* start =
      (const char*)elf.data + shstr->file_offset + section->name_off;
  size_t max_len = shstr->size - section->name_off;
  size_t len = 0;
  while (len < max_len && start[len] != '\0') ++len;
  if (len == max_len) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section name is not "
                            "NUL-terminated");
  }
  *out = iree_make_string_view(start, len);
  return iree_ok_status();
}

static bool iree_hal_amdxdna_name_is_ctrltext(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".ctrltext")) ||
         iree_string_view_starts_with(name, IREE_SV(".ctrltext."));
}

static bool iree_hal_amdxdna_name_is_ctrldata(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".ctrldata")) ||
         iree_string_view_starts_with(name, IREE_SV(".ctrldata."));
}

static bool iree_hal_amdxdna_name_is_rela_dyn(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".rela.dyn")) ||
         iree_string_view_starts_with(name, IREE_SV(".rela.dyn."));
}

static bool iree_hal_amdxdna_name_is_pad(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".pad")) ||
         iree_string_view_starts_with(name, IREE_SV(".pad."));
}

static bool iree_hal_amdxdna_name_is_xrt_configuration(
    iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".note.xrt.configuration"));
}

// ELF32 note: namesz/descsz/type, name padded to 4, then desc. XRT stores the
// partition column count as the first little-endian u32 of the descriptor.
static uint32_t iree_hal_amdxdna_parse_partition_cols(
    iree_const_byte_span_t elf, const iree_hal_amdxdna_elf_section_t* note) {
  if (!note || note->size < 16) return 0;
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, note->file_offset,
                                            note->size)) {
    return 0;
  }
  const uint8_t* blob = elf.data + note->file_offset;
  uint32_t namesz = iree_hal_amdxdna_read_u32(blob, 0);
  uint32_t descsz = iree_hal_amdxdna_read_u32(blob, 4);
  uint32_t name_pad = (namesz + 3u) & ~3u;
  if (descsz < 4 || (uint64_t)12 + name_pad + 4 > note->size) return 0;
  return iree_hal_amdxdna_read_u32(blob, 12 + name_pad);
}

static bool iree_hal_amdxdna_parse_u32_token(const char* start, const char* end,
                                             uint32_t* out_value,
                                             const char** out_next) {
  if (start >= end || *start < '0' || *start > '9') return false;
  uint32_t value = 0;
  const char* p = start;
  while (p < end && *p >= '0' && *p <= '9') {
    uint32_t digit = (uint32_t)(*p - '0');
    if (value > (UINT32_MAX - digit) / 10) return false;
    value = value * 10 + digit;
    ++p;
  }
  *out_value = value;
  *out_next = p;
  return true;
}

static iree_status_t iree_hal_amdxdna_parse_col_page(iree_string_view_t name,
                                                     uint32_t* out_col,
                                                     uint32_t* out_page) {
  *out_col = 0;
  *out_page = 0;
  // ".ctrltext" and ".ctrldata" are both 9 characters; extra ".col.page"
  // components follow that prefix.
  if (name.size <= 9) return iree_ok_status();
  if (name.data[9] != '.') return iree_ok_status();
  const char* p = name.data + 10;
  const char* end = name.data + name.size;
  uint32_t col = 0;
  if (!iree_hal_amdxdna_parse_u32_token(p, end, &col, &p)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' has an "
                            "invalid column index",
                            (int)name.size, name.data);
  }
  uint32_t page = 0;
  if (p < end) {
    if (*p != '.') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page suffix",
                              (int)name.size, name.data);
    }
    ++p;
    if (!iree_hal_amdxdna_parse_u32_token(p, end, &page, &p)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
  }
  // OSABI 70 config ELFs use `.ctrltext.<col>.<page>.<n>` (and the same for
  // `.ctrldata`), where <n> is the kernel group; group membership, not this
  // suffix, decides which kernel a page belongs to.
  if (p < end) {
    if (*p != '.') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
    ++p;
    uint32_t ignored = 0;
    if (!iree_hal_amdxdna_parse_u32_token(p, end, &ignored, &p) || p != end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
  }
  *out_col = col;
  *out_page = page;
  return iree_ok_status();
}

static iree_hal_amdxdna_ctrl_page_t* iree_hal_amdxdna_find_or_add_page(
    iree_hal_amdxdna_ctrl_page_t* pages, iree_host_size_t* page_count,
    iree_host_size_t page_capacity, uint32_t col, uint32_t page) {
  for (iree_host_size_t i = 0; i < *page_count; ++i) {
    if (pages[i].col == col && pages[i].page == page) return &pages[i];
  }
  if (*page_count >= page_capacity) return NULL;
  iree_hal_amdxdna_ctrl_page_t* slot = &pages[*page_count];
  memset(slot, 0, sizeof(*slot));
  slot->col = col;
  slot->page = page;
  ++(*page_count);
  return slot;
}

static int iree_hal_amdxdna_ctrl_page_cmp(const void* lhs, const void* rhs) {
  const iree_hal_amdxdna_ctrl_page_t* a =
      (const iree_hal_amdxdna_ctrl_page_t*)lhs;
  const iree_hal_amdxdna_ctrl_page_t* b =
      (const iree_hal_amdxdna_ctrl_page_t*)rhs;
  if (a->col != b->col) return a->col < b->col ? -1 : 1;
  if (a->page != b->page) return a->page < b->page ? -1 : 1;
  return 0;
}

static iree_status_t iree_hal_amdxdna_append_bytes(
    iree_allocator_t host_allocator, uint8_t** io_data, size_t* io_size,
    size_t* io_capacity, const uint8_t* src, size_t src_size) {
  if (src_size == 0) return iree_ok_status();
  size_t needed = *io_size + src_size;
  if (needed < *io_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna ctrlcode ELF buffer overflow");
  }
  if (needed > *io_capacity) {
    size_t new_capacity = *io_capacity ? *io_capacity : 256;
    while (new_capacity < needed) {
      if (new_capacity > (SIZE_MAX / 2)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "amdxdna ctrlcode ELF buffer overflow");
      }
      new_capacity *= 2;
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(host_allocator, new_capacity,
                                                (void**)io_data));
    *io_capacity = new_capacity;
  }
  memcpy(*io_data + *io_size, src, src_size);
  *io_size = needed;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pad_to_page(iree_allocator_t host_allocator,
                                                  uint8_t** io_data,
                                                  size_t* io_size,
                                                  size_t* io_capacity,
                                                  size_t col_base,
                                                  uint32_t page) {
  // Page indices restart at 0 in each column. Pad relative to that column's
  // concatenated start, not the whole instruction blob (AIE4 nop.elf is
  // `.ctrltext.{0,2,4}.0`, each an 8KiB page 0).
  const size_t pad =
      col_base + ((size_t)page + 1) * IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE;
  if (*io_size > pad) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF page %u exceeds the %zu-byte "
                            "page size",
                            page, pad);
  }
  if (pad == *io_size) return iree_ok_status();
  size_t old_size = *io_size;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_append_bytes(
      host_allocator, io_data, io_size, io_capacity, NULL, 0));
  if (pad > *io_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_realloc(host_allocator, pad, (void**)io_data));
    *io_capacity = pad;
  }
  memset(*io_data + old_size, 0, pad - old_size);
  *io_size = pad;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_section_bytes(
    iree_const_byte_span_t elf, uint32_t file_offset, uint32_t size,
    iree_allocator_t host_allocator, uint8_t** io_data, size_t* io_size,
    size_t* io_capacity) {
  if (size == 0) return iree_ok_status();
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, file_offset,
                                            size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section bytes are out of "
                            "range");
  }
  return iree_hal_amdxdna_append_bytes(host_allocator, io_data, io_size,
                                       io_capacity, elf.data + file_offset,
                                       size);
}

static bool iree_hal_amdxdna_is_skipped_symbol(iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) return true;
  if (iree_string_view_equal(name, IREE_SV("scratch-pad-mem"))) return true;
  if (iree_string_view_equal(name, IREE_SV("scratch-pad-ctrl"))) return true;
  if (iree_string_view_equal(name, IREE_SV("control-packet"))) return true;
  if (iree_string_view_starts_with(name, IREE_SV("ctrlpkt-pm"))) return true;
  if (iree_string_view_find(name, IREE_SV("pdi"), 0) != IREE_STRING_VIEW_NPOS) {
    return true;
  }
  return false;
}

static bool iree_hal_amdxdna_parse_arg_index(iree_string_view_t name,
                                             uint32_t* out_index) {
  iree_string_view_t token = name;
  if (iree_string_view_starts_with(name, IREE_SV("argv"))) {
    token = iree_string_view_remove_prefix(name, 4);
  }
  if (iree_string_view_is_empty(token)) return false;
  for (iree_host_size_t i = 0; i < token.size; ++i) {
    if (token.data[i] < '0' || token.data[i] > '9') return false;
  }
  uint32_t value = 0;
  const char* next = NULL;
  if (!iree_hal_amdxdna_parse_u32_token(token.data, token.data + token.size,
                                        &value, &next) ||
      next != token.data + token.size) {
    return false;
  }
  *out_index = value;
  return true;
}

static iree_status_t iree_hal_amdxdna_append_patch(
    iree_allocator_t host_allocator, uint32_t** io_patches, size_t* io_count,
    size_t* io_capacity, uint32_t offset, uint32_t arg_idx, uint32_t arg_plus) {
  size_t needed = *io_count + 3;
  if (needed > *io_capacity) {
    size_t new_capacity = *io_capacity ? *io_capacity * 2 : 24;
    while (new_capacity < needed) new_capacity *= 2;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        host_allocator, new_capacity * sizeof(uint32_t), (void**)io_patches));
    *io_capacity = new_capacity;
  }
  (*io_patches)[*io_count + 0] = offset;
  (*io_patches)[*io_count + 1] = arg_idx;
  (*io_patches)[*io_count + 2] = arg_plus;
  *io_count = needed;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_load_sections(
    iree_const_byte_span_t elf, iree_hal_amdxdna_elf_section_t* sections,
    uint16_t shnum, uint32_t shoff, uint16_t shentsize) {
  for (uint16_t i = 0; i < shnum; ++i) {
    size_t sh_off = (size_t)shoff + (size_t)i * shentsize;
    if (sh_off + IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE > elf.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section header %u is out "
                              "of range",
                              i);
    }
    sections[i].name_off = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_NAME);
    sections[i].type = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_TYPE);
    sections[i].file_offset = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_OFFSET);
    sections[i].size = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_SIZE);
    sections[i].link = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_LINK);
    sections[i].info = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_INFO);
    sections[i].entsize = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_ENTSIZE);
    sections[i].index = i;
  }
  return iree_ok_status();
}

typedef struct iree_hal_amdxdna_pad_span_t {
  uint32_t col;
  uint32_t file_offset;
  uint32_t size;
} iree_hal_amdxdna_pad_span_t;

// `.pad`, `.pad.<col>`, or `.pad.<col>.<page>`. Bare `.pad` is column 0.
static iree_status_t iree_hal_amdxdna_parse_pad_col(iree_string_view_t name,
                                                    uint32_t* out_col) {
  *out_col = 0;
  if (iree_string_view_equal(name, IREE_SV(".pad"))) return iree_ok_status();
  if (!iree_string_view_starts_with(name, IREE_SV(".pad."))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' is not a pad",
                            (int)name.size, name.data);
  }
  const char* p = name.data + 5;
  const char* end = name.data + name.size;
  if (!iree_hal_amdxdna_parse_u32_token(p, end, out_col, &p)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' has an "
                            "invalid column index",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_build_aie2ps_ctrl(
    iree_const_byte_span_t elf, iree_allocator_t host_allocator,
    const iree_hal_amdxdna_ctrl_page_t* pages, iree_host_size_t page_count,
    const iree_hal_amdxdna_pad_span_t* pads, iree_host_size_t pad_count,
    uint8_t** out_data, size_t* out_size, uint32_t* out_col_bases,
    uint32_t* out_col_count) {
  *out_data = NULL;
  *out_size = 0;
  *out_col_count = 0;
  memset(out_col_bases, 0,
         IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS * sizeof(uint32_t));
  if (page_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF has no .ctrltext section");
  }

  uint32_t max_col = 0;
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    if (pages[i].col >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF column %u exceeds %u",
                              pages[i].col,
                              IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS);
    }
    if (pages[i].col > max_col) max_col = pages[i].col;
  }
  for (iree_host_size_t i = 0; i < pad_count; ++i) {
    if (pads[i].col > max_col) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF pad column %u is outside "
                              "ctrltext columns 0..%u",
                              pads[i].col, max_col);
    }
  }
  *out_col_count = max_col + 1;

  uint8_t* data = NULL;
  size_t size = 0;
  size_t capacity = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t col = 0; col <= max_col && iree_status_is_ok(status); ++col) {
    out_col_bases[col] = (uint32_t)size;
    for (iree_host_size_t i = 0; i < page_count && iree_status_is_ok(status);
         ++i) {
      if (pages[i].col != col) continue;
      if (pages[i].has_ctrltext) {
        status = iree_hal_amdxdna_copy_section_bytes(
            elf, pages[i].ctrltext_offset, pages[i].ctrltext_size,
            host_allocator, &data, &size, &capacity);
      }
      if (iree_status_is_ok(status) && pages[i].has_ctrldata) {
        status = iree_hal_amdxdna_copy_section_bytes(
            elf, pages[i].ctrldata_offset, pages[i].ctrldata_size,
            host_allocator, &data, &size, &capacity);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_pad_to_page(
            host_allocator, &data, &size, &capacity, out_col_bases[col],
            pages[i].page);
      }
    }
    // aiebu appends this column's `.pad.*` before the next column starts.
    for (iree_host_size_t p = 0; p < pad_count && iree_status_is_ok(status);
         ++p) {
      if (pads[p].col != col) continue;
      status = iree_hal_amdxdna_copy_section_bytes(
          elf, pads[p].file_offset, pads[p].size, host_allocator, &data, &size,
          &capacity);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, data);
    return status;
  }
  if (size % sizeof(uint32_t) != 0) {
    iree_allocator_free(host_allocator, data);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF instruction size %zu is not "
                            "a multiple of 4",
                            size);
  }
  *out_data = data;
  *out_size = size;
  return iree_ok_status();
}

// State for turning relocations into host patches for one kernel.
typedef struct iree_hal_amdxdna_reloc_ctx_t {
  iree_const_byte_span_t elf;
  uint8_t abi_version;
  bool paged;
  uint16_t shnum;
  const iree_string_view_t* section_names;
  const iree_hal_amdxdna_elf_section_t* dynsym;
  const iree_hal_amdxdna_elf_section_t* dynstr;
  uint32_t symentsz;
  const uint32_t* col_bases;
  uint32_t col_count;
  size_t ctrl_size;
  iree_allocator_t host_allocator;
  uint32_t* patches;
  size_t patch_count;
  size_t patch_capacity;
  iree_host_size_t skipped;
} iree_hal_amdxdna_reloc_ctx_t;

// Appends the patch for the Elf32_Rela at |rela_off|, or counts it as skipped
// (non-host symbols, other schemes, columns outside the instruction buffer).
static iree_status_t iree_hal_amdxdna_process_reloc(
    iree_hal_amdxdna_reloc_ctx_t* ctx, size_t rela_off) {
  const uint8_t* data = ctx->elf.data;
  uint32_t r_offset = iree_hal_amdxdna_read_u32(data, rela_off);
  uint32_t r_info = iree_hal_amdxdna_read_u32(data, rela_off + 4);
  int32_t r_addend = iree_hal_amdxdna_read_i32(data, rela_off + 8);
  uint32_t symidx = r_info >> 8;
  uint32_t r_type = r_info & 0xFFu;
  // The symbol index was range-checked when the index was built.
  const size_t sym_off =
      ctx->dynsym->file_offset + (size_t)symidx * ctx->symentsz;
  uint32_t st_name =
      iree_hal_amdxdna_read_u32(data, sym_off + IREE_HAL_AMDXDNA_ELF32_ST_NAME);
  uint16_t st_shndx = iree_hal_amdxdna_read_u16(
      data, sym_off + IREE_HAL_AMDXDNA_ELF32_ST_SHNDX);
  if (st_name >= ctx->dynstr->size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF symbol name is out of range");
  }
  const char* sym_start =
      (const char*)data + ctx->dynstr->file_offset + st_name;
  size_t max_len = ctx->dynstr->size - st_name;
  size_t name_len = 0;
  while (name_len < max_len && sym_start[name_len] != '\0') ++name_len;
  iree_string_view_t symname = iree_make_string_view(sym_start, name_len);

  uint32_t scheme = r_type;
  uint32_t add_end_addr = (uint32_t)r_addend;
  if (ctx->abi_version == 1) {
    add_end_addr =
        ((uint32_t)r_addend) >> IREE_HAL_AMDXDNA_CTRLCODE_ADDEND_SHIFT;
    scheme = ((uint32_t)r_addend) & 0xFu;
  }

  const bool is_control_code =
      iree_string_view_starts_with(symname, IREE_SV("control-code"));
  if ((!is_control_code && iree_hal_amdxdna_is_skipped_symbol(symname)) ||
      scheme != IREE_HAL_AMDXDNA_CTRLCODE_ELF_PATCH_SHIM_DMA_AIE4) {
    ++ctx->skipped;
    return iree_ok_status();
  }
  uint32_t arg_idx = 0;
  if (is_control_code) {
    arg_idx = UINT32_MAX;
  } else if (!iree_hal_amdxdna_parse_arg_index(symname, &arg_idx)) {
    ++ctx->skipped;
    return iree_ok_status();
  }

  uint32_t abs_offset = r_offset;
  if (ctx->paged) {
    if (st_shndx >= ctx->shnum) {
      ++ctx->skipped;
      return iree_ok_status();
    }
    uint32_t col = 0;
    uint32_t page = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_col_page(
        ctx->section_names[st_shndx], &col, &page));
    if (col >= ctx->col_count) {
      ++ctx->skipped;
      return iree_ok_status();
    }
    // st_shndx names the page; r_offset is relative to the 16-byte page
    // header (same AIE2PS/OSABI-70 convention XRT/aiebu uses). Packed
    // layout is [ctrltext][ctrldata] zero-padded to 8KiB.
    uint64_t sec_offset = (uint64_t)page * IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE +
                          r_offset + IREE_HAL_AMDXDNA_CTRLCODE_PAGE_HDR;
    uint64_t total = (uint64_t)ctx->col_bases[col] + sec_offset;
    if (total > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF patch offset overflows");
    }
    abs_offset = (uint32_t)total;
  }
  if ((size_t)abs_offset + 8 > ctx->ctrl_size || (abs_offset & 3u) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna ctrlcode ELF patch offset %u is outside the %zu-byte "
        "instruction buffer",
        abs_offset, ctx->ctrl_size);
  }
  return iree_hal_amdxdna_append_patch(ctx->host_allocator, &ctx->patches,
                                       &ctx->patch_count, &ctx->patch_capacity,
                                       abs_offset, arg_idx, add_end_addr);
}

static iree_status_t iree_hal_amdxdna_build_dpu_slices(
    iree_allocator_t host_allocator, const uint32_t* col_bases,
    uint32_t col_count, size_t ctrl_size,
    iree_hal_amdxdna_ctrlcode_dpu_slice_t** out_slices,
    iree_host_size_t* out_count) {
  *out_slices = NULL;
  *out_count = 0;
  if (col_count == 0 || ctrl_size == 0) return iree_ok_status();
  if (ctrl_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna ctrlcode instruction buffer is too large");
  }
  iree_hal_amdxdna_ctrlcode_dpu_slice_t* slices = NULL;
  iree_host_size_t count = 0;
  size_t covered = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t col = 0; col < col_count && iree_status_is_ok(status); ++col) {
    const uint32_t begin = col_bases[col];
    const uint32_t end = (col + 1 < col_count) ? col_bases[col + 1]
                                               : (uint32_t)ctrl_size;
    if (end < begin) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "amdxdna ctrlcode column %u offset went "
                                "backwards",
                                col);
      break;
    }
    if (end == begin) continue;
    if (count >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna ctrlcode ELF has more than %u occupied columns; START_DPU "
          "indirect packets stop at HSA_MAX_LEVEL1_INDIRECT_ENTRIES",
          IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES);
      break;
    }
    if (count == 0) {
      status = iree_allocator_malloc(
          host_allocator,
          IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES * sizeof(*slices),
          (void**)&slices);
      if (!iree_status_is_ok(status)) break;
    }
    slices[count].uc_index = (uint16_t)col;
    slices[count].reserved = 0;
    slices[count].byte_offset = begin;
    slices[count].byte_size = end - begin;
    covered += (size_t)(end - begin);
    ++count;
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, slices);
    return status;
  }
  if (covered != ctrl_size) {
    iree_allocator_free(host_allocator, slices);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "amdxdna ctrlcode column slices cover %zu of %zu "
                            "instruction bytes",
                            covered, ctrl_size);
  }
  if (count == 1 && slices[0].uc_index != 0) {
    // KMD fill_direct_pkt ignores uc_index. One occupied column is a direct
    // packet, so a hole at column 0 cannot be named. More than one occupied
    // column uses fill_indirect_pkt, which does honor uc_index.
    const uint16_t uc = slices[0].uc_index;
    iree_allocator_free(host_allocator, slices);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna ctrlcode ELF occupies only column %u; a START_DPU direct "
        "packet cannot name a non-zero microcontroller",
        uc);
  }
  if (count > 1) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (slices[i].uc_index >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES) {
        iree_allocator_free(host_allocator, slices);
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "amdxdna ctrlcode column %u is outside the %u microcontrollers "
            "START_DPU indirect packets can name",
            slices[i].uc_index, IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES);
      }
    }
  }
  *out_slices = slices;
  *out_count = count;
  return iree_ok_status();
}

static int iree_hal_amdxdna_patch_offset_cmp(const void* lhs, const void* rhs) {
  const uint32_t a = *(const uint32_t*)lhs;
  const uint32_t b = *(const uint32_t*)rhs;
  return a < b ? -1 : (a > b ? 1 : 0);
}

// AIE4 tables repeat a relocation once per microcontroller that shares a page,
// but the page is packed once and must be patched once. Keeps the first entry
// per offset and sorts by offset, so the patcher needs no duplicate search.
static iree_status_t iree_hal_amdxdna_canonicalize_aie4_patches(
    iree_allocator_t host_allocator, size_t ctrl_size, uint32_t* patches,
    size_t* io_patch_count) {
  const size_t word_count = ctrl_size / sizeof(uint32_t);
  uint8_t* seen = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, (word_count + 7) / 8 + 1, (void**)&seen));
  memset(seen, 0, (word_count + 7) / 8 + 1);
  size_t kept = 0;
  for (size_t i = 0; i < *io_patch_count; i += 3) {
    const uint32_t word = patches[i] / sizeof(uint32_t);
    const uint8_t bit = (uint8_t)(1u << (word % 8));
    if (seen[word / 8] & bit) continue;
    seen[word / 8] |= bit;
    if (kept != i) memmove(&patches[kept], &patches[i], 3 * sizeof(uint32_t));
    kept += 3;
  }
  iree_allocator_free(host_allocator, seen);
  qsort(patches, kept / 3, 3 * sizeof(uint32_t),
        iree_hal_amdxdna_patch_offset_cmp);
  *io_patch_count = kept;
  return iree_ok_status();
}

// A validated ELF header plus a heap copy of its section table.
typedef struct iree_hal_amdxdna_elf_view_t {
  iree_const_byte_span_t bytes;
  uint8_t os_abi;
  uint8_t abi_version;
  bool paged;
  uint16_t shnum;
  iree_hal_amdxdna_elf_section_t* sections;
  const iree_hal_amdxdna_elf_section_t* shstr;
} iree_hal_amdxdna_elf_view_t;

static void iree_hal_amdxdna_elf_view_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_elf_view_t* view) {
  iree_allocator_free(host_allocator, view->sections);
  memset(view, 0, sizeof(*view));
}

static iree_status_t iree_hal_amdxdna_elf_view_initialize(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_elf_view_t* out_view) {
  memset(out_view, 0, sizeof(*out_view));
  if (!elf_bytes.data ||
      elf_bytes.data_length < IREE_HAL_AMDXDNA_ELF32_EHDR_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF is missing or truncated");
  }
  const uint8_t* ident = elf_bytes.data;
  if (ident[0] != 0x7F || ident[1] != 'E' || ident[2] != 'L' ||
      ident[3] != 'F') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode image is not an ELF object");
  }
  if (ident[IREE_HAL_AMDXDNA_ELF_EI_CLASS] != IREE_HAL_AMDXDNA_ELF32_CLASS ||
      ident[IREE_HAL_AMDXDNA_ELF_EI_DATA] != IREE_HAL_AMDXDNA_ELF32_DATA_LE ||
      ident[IREE_HAL_AMDXDNA_ELF_EI_VERSION] !=
          IREE_HAL_AMDXDNA_ELF32_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF must be ELF32 little-endian");
  }
  const uint8_t os_abi = ident[IREE_HAL_AMDXDNA_ELF_EI_OSABI];
  // OSABI 64 is the original paged AIE2PS layout (`.ctrltext.<col>.<page>`).
  // OSABI 70 (aie2ps_group) uses the same pages plus a trailing kernel group.
  // OSABI 75 (aiebu AIE4) is the same paged layout; VTD npu3 latency nop.elf
  // is this ABI with `.ctrltext.{0,2,4}.0`.
  if (os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna ctrlcode ELF OSABI %u is not supported",
                            os_abi);
  }

  uint32_t shoff =
      iree_hal_amdxdna_read_u32(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHOFF);
  uint16_t shentsize = iree_hal_amdxdna_read_u16(
      elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHENTSIZE);
  uint16_t shnum =
      iree_hal_amdxdna_read_u16(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHNUM);
  uint16_t shstrndx = iree_hal_amdxdna_read_u16(
      elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHSTRNDX);
  if (shentsize < IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE || shnum == 0 ||
      shstrndx >= shnum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section table is invalid");
  }

  iree_hal_amdxdna_elf_section_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, (iree_host_size_t)shnum * sizeof(*sections),
      (void**)&sections));
  iree_status_t status = iree_hal_amdxdna_load_sections(
      elf_bytes, sections, shnum, shoff, shentsize);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, sections);
    return status;
  }
  out_view->bytes = elf_bytes;
  out_view->os_abi = os_abi;
  out_view->abi_version = ident[IREE_HAL_AMDXDNA_ELF_EI_ABIVERSION];
  out_view->paged =
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS ||
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG ||
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4;
  out_view->shnum = shnum;
  out_view->sections = sections;
  out_view->shstr = &sections[shstrndx];
  return iree_ok_status();
}

// Name of symbol `symbol_index` in the symbol table section `symtab_index`
// (whose sh_link is its string table).
static iree_status_t iree_hal_amdxdna_elf_symbol_name(
    const iree_hal_amdxdna_elf_view_t* view, uint32_t symtab_index,
    uint32_t symbol_index, iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (symtab_index >= view->shnum ||
      view->sections[symtab_index].link >= view->shnum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF group symbol table %u is "
                            "invalid",
                            symtab_index);
  }
  const iree_hal_amdxdna_elf_section_t* symtab = &view->sections[symtab_index];
  const iree_hal_amdxdna_elf_section_t* strtab = &view->sections[symtab->link];
  const uint32_t entsize =
      symtab->entsize ? symtab->entsize : IREE_HAL_AMDXDNA_ELF32_SYM_SIZE;
  const uint64_t sym_off = (uint64_t)symbol_index * entsize;
  if (entsize < IREE_HAL_AMDXDNA_ELF32_SYM_SIZE ||
      !iree_hal_amdxdna_elf_range_in_bounds(
          view->bytes.data_length, symtab->file_offset, symtab->size) ||
      !iree_hal_amdxdna_elf_range_in_bounds(
          view->bytes.data_length, strtab->file_offset, strtab->size) ||
      sym_off + IREE_HAL_AMDXDNA_ELF32_SYM_SIZE > symtab->size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF group symbol %u is out of "
                            "range",
                            symbol_index);
  }
  const uint32_t st_name = iree_hal_amdxdna_read_u32(
      view->bytes.data,
      symtab->file_offset + (size_t)sym_off + IREE_HAL_AMDXDNA_ELF32_ST_NAME);
  if (st_name >= strtab->size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF symbol name is out of range");
  }
  const char* start =
      (const char*)view->bytes.data + strtab->file_offset + st_name;
  const size_t max_len = strtab->size - st_name;
  size_t len = 0;
  while (len < max_len && start[len] != '\0') ++len;
  if (len == max_len) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF symbol name is not "
                            "NUL-terminated");
  }
  *out_name = iree_make_string_view(start, len);
  return iree_ok_status();
}

typedef struct iree_hal_amdxdna_elf_group_t {
  iree_string_view_t name;
  uint32_t member_begin;  // into iree_hal_amdxdna_ctrlcode_elf_index_t::members
  uint32_t member_count;
} iree_hal_amdxdna_elf_group_t;

typedef struct iree_hal_amdxdna_elf_named_group_t {
  iree_string_view_t name;
  uint32_t group;
} iree_hal_amdxdna_elf_named_group_t;

struct iree_hal_amdxdna_ctrlcode_elf_index_t {
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_elf_view_t view;
  iree_string_view_t* section_names;
  const iree_hal_amdxdna_elf_section_t* dynsym;
  const iree_hal_amdxdna_elf_section_t* dynstr;
  const iree_hal_amdxdna_elf_section_t* xrt_configuration;
  uint32_t symentsz;
  // File offset of every .rela.dyn entry, in file order.
  size_t* relas;
  iree_host_size_t rela_count;
  // Relocations whose symbol names section s, in file order:
  // section_relas[section_rela_begin[s] .. section_rela_begin[s + 1]).
  uint32_t* section_rela_begin;
  size_t* section_relas;
  // Kernel groups in section order, their member sections, and the groups
  // sorted by name (ties by section order) for lookup.
  iree_hal_amdxdna_elf_group_t* groups;
  iree_host_size_t group_count;
  uint32_t* members;
  iree_hal_amdxdna_elf_named_group_t* named_groups;
};

void iree_hal_amdxdna_ctrlcode_elf_index_free(
    iree_hal_amdxdna_ctrlcode_elf_index_t* index) {
  if (!index) return;
  iree_allocator_t host_allocator = index->host_allocator;
  iree_allocator_free(host_allocator, index->named_groups);
  iree_allocator_free(host_allocator, index->members);
  iree_allocator_free(host_allocator, index->groups);
  iree_allocator_free(host_allocator, index->section_relas);
  iree_allocator_free(host_allocator, index->section_rela_begin);
  iree_allocator_free(host_allocator, index->relas);
  iree_allocator_free(host_allocator, index->section_names);
  iree_hal_amdxdna_elf_view_deinitialize(host_allocator, &index->view);
  iree_allocator_free(host_allocator, index);
}

static int iree_hal_amdxdna_named_group_cmp(const void* lhs, const void* rhs) {
  const iree_hal_amdxdna_elf_named_group_t* a =
      (const iree_hal_amdxdna_elf_named_group_t*)lhs;
  const iree_hal_amdxdna_elf_named_group_t* b =
      (const iree_hal_amdxdna_elf_named_group_t*)rhs;
  const int c = iree_string_view_compare(a->name, b->name);
  if (c != 0) return c;
  return a->group < b->group ? -1 : (a->group > b->group ? 1 : 0);
}

// Section names and the dynamic symbol/string tables.
static iree_status_t iree_hal_amdxdna_elf_index_sections(
    iree_hal_amdxdna_ctrlcode_elf_index_t* index) {
  const iree_hal_amdxdna_elf_view_t* view = &index->view;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator,
      (iree_host_size_t)view->shnum * sizeof(*index->section_names),
      (void**)&index->section_names));
  for (uint16_t i = 0; i < view->shnum; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_elf_section_name(
        view->bytes, view->shstr, &view->sections[i],
        &index->section_names[i]));
    const iree_string_view_t name = index->section_names[i];
    if (iree_string_view_equal(name, IREE_SV(".dynsym"))) {
      index->dynsym = &view->sections[i];
    }
    if (iree_string_view_equal(name, IREE_SV(".dynstr"))) {
      index->dynstr = &view->sections[i];
    }
    if (iree_hal_amdxdna_name_is_xrt_configuration(name)) {
      index->xrt_configuration = &view->sections[i];
    }
  }
  if (!index->dynsym || !index->dynstr) return iree_ok_status();
  index->symentsz = index->dynsym->entsize ? index->dynsym->entsize
                                           : IREE_HAL_AMDXDNA_ELF32_SYM_SIZE;
  if (index->symentsz < IREE_HAL_AMDXDNA_ELF32_SYM_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF dynsym entsize %u is invalid",
                            index->symentsz);
  }
  if (!iree_hal_amdxdna_elf_range_in_bounds(view->bytes.data_length,
                                            index->dynsym->file_offset,
                                            index->dynsym->size) ||
      !iree_hal_amdxdna_elf_range_in_bounds(view->bytes.data_length,
                                            index->dynstr->file_offset,
                                            index->dynstr->size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna ctrlcode ELF dynsym/dynstr is out of range");
  }
  return iree_ok_status();
}

// Every .rela.dyn entry, bucketed by the section its symbol names. Without a
// dynamic symbol table no relocation is applied.
static iree_status_t iree_hal_amdxdna_elf_index_relocs(
    iree_hal_amdxdna_ctrlcode_elf_index_t* index) {
  if (!index->dynsym || !index->dynstr) return iree_ok_status();
  const iree_hal_amdxdna_elf_view_t* view = &index->view;
  iree_host_size_t count = 0;
  for (uint16_t i = 0; i < view->shnum; ++i) {
    const iree_hal_amdxdna_elf_section_t* section = &view->sections[i];
    if (!iree_hal_amdxdna_name_is_rela_dyn(index->section_names[i])) continue;
    if (section->size % IREE_HAL_AMDXDNA_ELF32_RELA_SIZE != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF .rela.dyn size is not a "
                              "multiple of 12");
    }
    if (!iree_hal_amdxdna_elf_range_in_bounds(
            view->bytes.data_length, section->file_offset, section->size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF .rela.dyn is out of range");
    }
    count += section->size / IREE_HAL_AMDXDNA_ELF32_RELA_SIZE;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator, (count ? count : 1) * sizeof(*index->relas),
      (void**)&index->relas));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator,
      ((iree_host_size_t)view->shnum + 1) * sizeof(*index->section_rela_begin),
      (void**)&index->section_rela_begin));
  memset(
      index->section_rela_begin, 0,
      ((iree_host_size_t)view->shnum + 1) * sizeof(*index->section_rela_begin));
  for (uint16_t i = 0; i < view->shnum; ++i) {
    const iree_hal_amdxdna_elf_section_t* section = &view->sections[i];
    if (!iree_hal_amdxdna_name_is_rela_dyn(index->section_names[i])) continue;
    const uint32_t entries = section->size / IREE_HAL_AMDXDNA_ELF32_RELA_SIZE;
    for (uint32_t r = 0; r < entries; ++r) {
      const size_t rela_off = (size_t)section->file_offset +
                              (size_t)r * IREE_HAL_AMDXDNA_ELF32_RELA_SIZE;
      const uint32_t symidx =
          iree_hal_amdxdna_read_u32(view->bytes.data, rela_off + 4) >> 8;
      const uint64_t sym_off = (uint64_t)symidx * index->symentsz;
      if (sym_off + IREE_HAL_AMDXDNA_ELF32_SYM_SIZE > index->dynsym->size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "amdxdna ctrlcode ELF reloc symbol index %u "
                                "is out of range",
                                symidx);
      }
      const uint16_t st_shndx = iree_hal_amdxdna_read_u16(
          view->bytes.data, index->dynsym->file_offset + (size_t)sym_off +
                                IREE_HAL_AMDXDNA_ELF32_ST_SHNDX);
      if (st_shndx < view->shnum) ++index->section_rela_begin[st_shndx + 1];
      index->relas[index->rela_count++] = rela_off;
    }
  }
  for (uint16_t s = 0; s < view->shnum; ++s) {
    index->section_rela_begin[s + 1] += index->section_rela_begin[s];
  }
  const uint32_t bucketed = index->section_rela_begin[view->shnum];
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator,
      (bucketed ? bucketed : 1) * sizeof(*index->section_relas),
      (void**)&index->section_relas));
  uint32_t* cursor = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator, (iree_host_size_t)view->shnum * sizeof(*cursor),
      (void**)&cursor));
  memcpy(cursor, index->section_rela_begin,
         (iree_host_size_t)view->shnum * sizeof(*cursor));
  for (iree_host_size_t r = 0; r < index->rela_count; ++r) {
    const uint32_t symidx =
        iree_hal_amdxdna_read_u32(view->bytes.data, index->relas[r] + 4) >> 8;
    const uint16_t st_shndx = iree_hal_amdxdna_read_u16(
        view->bytes.data, index->dynsym->file_offset +
                              (size_t)symidx * index->symentsz +
                              IREE_HAL_AMDXDNA_ELF32_ST_SHNDX);
    if (st_shndx < view->shnum) {
      index->section_relas[cursor[st_shndx]++] = index->relas[r];
    }
  }
  iree_allocator_free(index->host_allocator, cursor);
  return iree_ok_status();
}

// Kernel groups: SHT_GROUP sections (a flags word, then member section
// indices) named by their symbol.
static iree_status_t iree_hal_amdxdna_elf_index_groups(
    iree_hal_amdxdna_ctrlcode_elf_index_t* index) {
  const iree_hal_amdxdna_elf_view_t* view = &index->view;
  iree_host_size_t member_total = 0;
  for (uint16_t i = 0; i < view->shnum; ++i) {
    const iree_hal_amdxdna_elf_section_t* group = &view->sections[i];
    if (group->type != IREE_HAL_AMDXDNA_ELF_SHT_GROUP) continue;
    if (group->size < sizeof(uint32_t) || group->size % sizeof(uint32_t) != 0 ||
        !iree_hal_amdxdna_elf_range_in_bounds(
            view->bytes.data_length, group->file_offset, group->size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna ctrlcode ELF group section %u is invalid", group->index);
    }
    ++index->group_count;
    member_total += group->size / sizeof(uint32_t) - 1;
  }
  if (index->group_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator, index->group_count * sizeof(*index->groups),
      (void**)&index->groups));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator, index->group_count * sizeof(*index->named_groups),
      (void**)&index->named_groups));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      index->host_allocator,
      (member_total ? member_total : 1) * sizeof(*index->members),
      (void**)&index->members));
  uint32_t g = 0;
  uint32_t m = 0;
  for (uint16_t i = 0; i < view->shnum; ++i) {
    const iree_hal_amdxdna_elf_section_t* section = &view->sections[i];
    if (section->type != IREE_HAL_AMDXDNA_ELF_SHT_GROUP) continue;
    iree_hal_amdxdna_elf_group_t* group = &index->groups[g];
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_elf_symbol_name(
        view, section->link, section->info, &group->name));
    group->member_begin = m;
    group->member_count = section->size / sizeof(uint32_t) - 1;
    for (uint32_t w = 0; w < group->member_count; ++w) {
      const uint32_t member = iree_hal_amdxdna_read_u32(
          view->bytes.data,
          section->file_offset + (size_t)(w + 1) * sizeof(uint32_t));
      if (member >= view->shnum) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "amdxdna ctrlcode ELF group section %u names "
                                "section %u of %u",
                                section->index, member, view->shnum);
      }
      index->members[m++] = member;
    }
    index->named_groups[g].name = group->name;
    index->named_groups[g].group = g;
    ++g;
  }
  qsort(index->named_groups, index->group_count, sizeof(*index->named_groups),
        iree_hal_amdxdna_named_group_cmp);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_index_create(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_index_t** out_index) {
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = NULL;
  iree_hal_amdxdna_ctrlcode_elf_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*index), (void**)&index));
  memset(index, 0, sizeof(*index));
  index->host_allocator = host_allocator;
  iree_status_t status = iree_hal_amdxdna_elf_view_initialize(
      elf_bytes, host_allocator, &index->view);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_elf_index_sections(index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_elf_index_relocs(index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_elf_index_groups(index);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_ctrlcode_elf_index_free(index);
    return status;
  }
  *out_index = index;
  return iree_ok_status();
}

// Finds the group a kernel name selects; NULL for ELFs without groups.
static iree_status_t iree_hal_amdxdna_elf_index_find_kernel(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index,
    iree_string_view_t kernel_name,
    const iree_hal_amdxdna_elf_group_t** out_group) {
  *out_group = NULL;
  if (index->group_count == 0) return iree_ok_status();
  const iree_host_size_t colon =
      iree_string_view_find_char(kernel_name, ':', 0);
  if (colon != IREE_STRING_VIEW_NPOS) {
    kernel_name = iree_string_view_remove_prefix(kernel_name, colon + 1);
  }
  if (iree_string_view_is_empty(kernel_name)) {
    if (index->group_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF has %" PRIhsz
                              " kernels; select one by name",
                              index->group_count);
    }
    *out_group = &index->groups[0];
    return iree_ok_status();
  }
  // Lower bound, so equal names resolve to the first group in section order.
  iree_host_size_t lo = 0, hi = index->group_count;
  while (lo < hi) {
    const iree_host_size_t mid = lo + (hi - lo) / 2;
    if (iree_string_view_compare(index->named_groups[mid].name, kernel_name) <
        0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == index->group_count ||
      !iree_string_view_equal(index->named_groups[lo].name, kernel_name)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "amdxdna ctrlcode ELF has no kernel '%.*s'",
                            (int)kernel_name.size, kernel_name.data);
  }
  *out_group = &index->groups[index->named_groups[lo].group];
  return iree_ok_status();
}

// Sections one kernel is built from. `pages` and `pads` hold up to
// `capacity` entries.
typedef struct iree_hal_amdxdna_elf_layout_t {
  iree_hal_amdxdna_ctrl_page_t* pages;
  iree_host_size_t page_count;
  iree_hal_amdxdna_pad_span_t* pads;
  iree_host_size_t pad_count;
  iree_host_size_t capacity;
  const iree_hal_amdxdna_elf_section_t* aie2p_ctrltext;
  iree_host_size_t aie2p_ctrltext_count;
} iree_hal_amdxdna_elf_layout_t;

static iree_status_t iree_hal_amdxdna_elf_collect_section(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index, uint32_t i,
    iree_hal_amdxdna_elf_layout_t* layout) {
  const iree_hal_amdxdna_elf_section_t* section = &index->view.sections[i];
  const iree_string_view_t name = index->section_names[i];
  if (!index->view.paged) {
    if (iree_hal_amdxdna_name_is_ctrltext(name)) {
      ++layout->aie2p_ctrltext_count;
      layout->aie2p_ctrltext = section;
    }
    return iree_ok_status();
  }
  if (iree_hal_amdxdna_name_is_ctrltext(name) ||
      iree_hal_amdxdna_name_is_ctrldata(name)) {
    uint32_t col = 0;
    uint32_t page = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_col_page(name, &col, &page));
    iree_hal_amdxdna_ctrl_page_t* slot = iree_hal_amdxdna_find_or_add_page(
        layout->pages, &layout->page_count, layout->capacity, col, page);
    if (!slot) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "amdxdna ctrlcode ELF has too many paged "
                              "sections");
    }
    if (iree_hal_amdxdna_name_is_ctrltext(name)) {
      slot->ctrltext_offset = section->file_offset;
      slot->ctrltext_size = section->size;
      slot->has_ctrltext = true;
    } else {
      slot->ctrldata_offset = section->file_offset;
      slot->ctrldata_size = section->size;
      slot->has_ctrldata = true;
    }
  } else if (iree_hal_amdxdna_name_is_pad(name)) {
    uint32_t col = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_pad_col(name, &col));
    iree_hal_amdxdna_pad_span_t* pad = &layout->pads[layout->pad_count++];
    pad->col = col;
    pad->file_offset = section->file_offset;
    pad->size = section->size;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_build_aie2p_ctrl(
    const iree_hal_amdxdna_elf_view_t* view,
    const iree_hal_amdxdna_elf_layout_t* layout,
    iree_allocator_t host_allocator, uint8_t** out_data, size_t* out_size) {
  if (layout->aie2p_ctrltext_count != 1 || !layout->aie2p_ctrltext) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna ctrlcode ELF OSABI %u requires exactly one .ctrltext "
        "section; found %" PRIhsz,
        view->os_abi, layout->aie2p_ctrltext_count);
  }
  if (layout->aie2p_ctrltext->size % sizeof(uint32_t) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF .ctrltext size %u is not a "
                            "multiple of 4",
                            layout->aie2p_ctrltext->size);
  }
  size_t capacity = 0;
  return iree_hal_amdxdna_copy_section_bytes(
      view->bytes, layout->aie2p_ctrltext->file_offset,
      layout->aie2p_ctrltext->size, host_allocator, out_data, out_size,
      &capacity);
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_index_parse_kernel(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index,
    iree_string_view_t kernel_name, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_t* out_elf) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_elf);
  memset(out_elf, 0, sizeof(*out_elf));
  const iree_hal_amdxdna_elf_view_t* view = &index->view;
  const iree_hal_amdxdna_elf_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_elf_index_find_kernel(index, kernel_name, &group));
  // A grouped ELF builds the kernel from its group's members only.
  const uint32_t* members = group ? &index->members[group->member_begin] : NULL;
  const iree_host_size_t section_count =
      group ? group->member_count : view->shnum;

  iree_hal_amdxdna_elf_layout_t layout;
  memset(&layout, 0, sizeof(layout));
  layout.capacity = section_count ? section_count : 1;
  uint8_t* ctrl_bytes = NULL;
  size_t ctrl_size = 0;
  uint32_t col_bases[IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS] = {0};
  uint32_t col_count = 0;
  iree_hal_amdxdna_reloc_ctx_t relocs;
  memset(&relocs, 0, sizeof(relocs));
  iree_hal_amdxdna_ctrlcode_dpu_slice_t* dpu_slices = NULL;
  iree_host_size_t dpu_slice_count = 0;

  iree_status_t status = iree_allocator_malloc(
      host_allocator, layout.capacity * sizeof(*layout.pages),
      (void**)&layout.pages);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   layout.capacity * sizeof(*layout.pads),
                                   (void**)&layout.pads);
  }
  for (iree_host_size_t i = 0; i < section_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdxdna_elf_collect_section(
        index, members ? members[i] : (uint32_t)i, &layout);
  }
  if (iree_status_is_ok(status)) {
    if (view->paged) {
      qsort(layout.pages, layout.page_count, sizeof(layout.pages[0]),
            iree_hal_amdxdna_ctrl_page_cmp);
      status = iree_hal_amdxdna_build_aie2ps_ctrl(
          view->bytes, host_allocator, layout.pages, layout.page_count,
          layout.pads, layout.pad_count, &ctrl_bytes, &ctrl_size, col_bases,
          &col_count);
    } else {
      status = iree_hal_amdxdna_build_aie2p_ctrl(view, &layout, host_allocator,
                                                 &ctrl_bytes, &ctrl_size);
      col_count = 1;
    }
  }
  if (iree_status_is_ok(status) && index->dynsym && index->dynstr) {
    relocs.elf = view->bytes;
    relocs.abi_version = view->abi_version;
    relocs.paged = view->paged;
    relocs.shnum = view->shnum;
    relocs.section_names = index->section_names;
    relocs.dynsym = index->dynsym;
    relocs.dynstr = index->dynstr;
    relocs.symentsz = index->symentsz;
    relocs.col_bases = col_bases;
    relocs.col_count = col_count;
    relocs.ctrl_size = ctrl_size;
    relocs.host_allocator = host_allocator;
    if (members) {
      // Relocations into another kernel's pages belong to that kernel.
      for (iree_host_size_t i = 0;
           i < section_count && iree_status_is_ok(status); ++i) {
        const uint32_t s = members[i];
        for (uint32_t r = index->section_rela_begin[s];
             r < index->section_rela_begin[s + 1] && iree_status_is_ok(status);
             ++r) {
          status =
              iree_hal_amdxdna_process_reloc(&relocs, index->section_relas[r]);
        }
      }
    } else {
      for (iree_host_size_t r = 0;
           r < index->rela_count && iree_status_is_ok(status); ++r) {
        status = iree_hal_amdxdna_process_reloc(&relocs, index->relas[r]);
      }
    }
  }
  if (iree_status_is_ok(status) && relocs.patch_count != 0 &&
      (view->os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG ||
       view->os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4)) {
    status = iree_hal_amdxdna_canonicalize_aie4_patches(
        host_allocator, ctrl_size, relocs.patches, &relocs.patch_count);
  }
  if (iree_status_is_ok(status) && view->paged) {
    status = iree_hal_amdxdna_build_dpu_slices(host_allocator, col_bases,
                                               col_count, ctrl_size,
                                               &dpu_slices, &dpu_slice_count);
  }

  if (iree_status_is_ok(status)) {
    out_elf->os_abi = view->os_abi;
    out_elf->ctrl_words = (uint32_t*)ctrl_bytes;
    out_elf->ctrl_word_count = ctrl_size / sizeof(uint32_t);
    out_elf->patches = relocs.patches;
    out_elf->patch_count = relocs.patch_count;
    out_elf->skipped_non_host_relocs = relocs.skipped;
    out_elf->dpu_slices = dpu_slices;
    out_elf->dpu_slice_count = dpu_slice_count;
    out_elf->partition_cols = iree_hal_amdxdna_parse_partition_cols(
        view->bytes, index->xrt_configuration);
  } else {
    iree_allocator_free(host_allocator, ctrl_bytes);
    iree_allocator_free(host_allocator, relocs.patches);
    iree_allocator_free(host_allocator, dpu_slices);
  }
  iree_allocator_free(host_allocator, layout.pads);
  iree_allocator_free(host_allocator, layout.pages);
  return status;
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
    iree_const_byte_span_t elf_bytes, iree_string_view_t kernel_name,
    iree_allocator_t host_allocator, iree_hal_amdxdna_ctrlcode_elf_t* out_elf) {
  IREE_ASSERT_ARGUMENT(out_elf);
  memset(out_elf, 0, sizeof(*out_elf));
  iree_hal_amdxdna_ctrlcode_elf_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_ctrlcode_elf_index_create(
      elf_bytes, host_allocator, &index));
  iree_status_t status = iree_hal_amdxdna_ctrlcode_elf_index_parse_kernel(
      index, kernel_name, host_allocator, out_elf);
  iree_hal_amdxdna_ctrlcode_elf_index_free(index);
  return status;
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_parse(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_t* out_elf) {
  return iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
      elf_bytes, iree_string_view_empty(), host_allocator, out_elf);
}

iree_host_size_t iree_hal_amdxdna_ctrlcode_elf_index_kernel_count(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index) {
  return index->group_count;
}

iree_string_view_t iree_hal_amdxdna_ctrlcode_elf_index_kernel_name(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index, iree_host_size_t i) {
  return i < index->group_count ? index->groups[i].name
                                : iree_string_view_empty();
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_enumerate_kernels(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_kernel_callback_fn_t callback,
    void* user_data) {
  IREE_ASSERT_ARGUMENT(callback);
  iree_hal_amdxdna_ctrlcode_elf_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_ctrlcode_elf_index_create(
      elf_bytes, host_allocator, &index));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < index->group_count && iree_status_is_ok(status); ++i) {
    status = callback(user_data, index->groups[i].name);
  }
  iree_hal_amdxdna_ctrlcode_elf_index_free(index);
  return status;
}
