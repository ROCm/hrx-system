// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_CTRLCODE_ELF_H_
#define IREE_HAL_DRIVERS_AMDXDNA_CTRLCODE_ELF_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ELF EI_OSABI values used by AIE ctrlcode objects (architecture IDs in the
// ELF ident, not a runtime vendor name).
enum {
  IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS = 64,
  IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P = 69,
  IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG = 70,
  // aiebu osabi_aie4. Paged `.ctrltext.<col>.<page>` like OSABI 64.
  IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4 = 75,
};

// Reloc scheme stored in r_type (abi_version != 1) or r_addend low 4 bits
// (abi_version == 1). Scheme 6 is the AIE4 57-bit shim-DMA BD (bd[0]/bd[1]).
enum { IREE_HAL_AMDXDNA_CTRLCODE_ELF_PATCH_SHIM_DMA_AIE4 = 6 };

// One occupied column inside the packed instruction BO. `uc_index` is the ELF
// column, not the dense slot. `byte_offset`/`byte_size` are that column's
// ctrltext/ctrldata pages plus its `.pad.*`, matching aiebu's per-column buf.
typedef struct iree_hal_amdxdna_ctrlcode_dpu_slice_t {
  uint16_t uc_index;
  uint16_t reserved;
  uint32_t byte_offset;
  uint32_t byte_size;
} iree_hal_amdxdna_ctrlcode_dpu_slice_t;

// Parsed ctrlcode ELF: concatenated instruction words plus a host patch table
// of (offset, arg_idx, arg_plus) triples for
// iree_hal_amdxdna_apply_patch_table_aie4. `control-code-*` self-patches use
// arg_idx UINT32_MAX (instruction BO address). Other non-host symbols are
// counted in `skipped_non_host_relocs` and omitted. Kernel args are identified
// by numeric dynsym names (`0`, `argv0`).
typedef struct iree_hal_amdxdna_ctrlcode_elf_t {
  uint8_t os_abi;
  uint32_t* ctrl_words;
  iree_host_size_t ctrl_word_count;
  uint32_t* patches;
  iree_host_size_t patch_count;
  iree_host_size_t skipped_non_host_relocs;
  // Column count from `.note.xrt.configuration` (XRT `elf.get_partition_size`).
  // 0 if the note was absent. CREATE_HWCTX uses this value, not the number of
  // `.ctrltext.<col>.*` sections (a 1-col hwctx around a 3-col vadd note hangs
  // even the first job).
  uint32_t partition_cols;
  // Occupied columns in the packed instruction buffer (XRT
  // fill_column_bo_address). Zero-size holes are omitted. One entry is a
  // direct START_DPU packet; more than one is chained fill_indirect_pkt.
  // Empty for non-paged ctrltext.
  iree_hal_amdxdna_ctrlcode_dpu_slice_t* dpu_slices;
  iree_host_size_t dpu_slice_count;
} iree_hal_amdxdna_ctrlcode_elf_t;

void iree_hal_amdxdna_ctrlcode_elf_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_ctrlcode_elf_t* elf);

// Parses the whole ELF. An ELF with more than one kernel group (see
// iree_hal_amdxdna_ctrlcode_elf_parse_kernel) is rejected.
iree_status_t iree_hal_amdxdna_ctrlcode_elf_parse(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_t* out_elf);

// Parses one kernel. Multi-kernel (OSABI 70 config) ELFs hold one SHT_GROUP
// per kernel, named by its symbol and owning that kernel's
// `.ctrltext.<col>.<page>.<n>` / `.ctrldata.<col>.<page>.<n>` sections; only
// that group's pages and the relocations into them are used. `kernel_name` is
// the group symbol, optionally prefixed by `<function>:` as XRT names kernels
// (`main:k0_...`). It is ignored for ELFs without groups, and may be empty for
// an ELF with exactly one group.
iree_status_t iree_hal_amdxdna_ctrlcode_elf_parse_kernel(
    iree_const_byte_span_t elf_bytes, iree_string_view_t kernel_name,
    iree_allocator_t host_allocator, iree_hal_amdxdna_ctrlcode_elf_t* out_elf);

// Section table, relocations bucketed by target section, and kernel groups of
// one ELF, built once so each kernel is parsed from its own sections only
// (config ELFs can hold hundreds of kernels). References |elf_bytes|,
// which must outlive the index.
typedef struct iree_hal_amdxdna_ctrlcode_elf_index_t
    iree_hal_amdxdna_ctrlcode_elf_index_t;

iree_status_t iree_hal_amdxdna_ctrlcode_elf_index_create(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_index_t** out_index);

void iree_hal_amdxdna_ctrlcode_elf_index_free(
    iree_hal_amdxdna_ctrlcode_elf_index_t* index);

// iree_hal_amdxdna_ctrlcode_elf_parse_kernel on an index.
iree_status_t iree_hal_amdxdna_ctrlcode_elf_index_parse_kernel(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index,
    iree_string_view_t kernel_name, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_t* out_elf);

// Kernel groups in section order; zero for ELFs without groups.
iree_host_size_t iree_hal_amdxdna_ctrlcode_elf_index_kernel_count(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index);
iree_string_view_t iree_hal_amdxdna_ctrlcode_elf_index_kernel_name(
    const iree_hal_amdxdna_ctrlcode_elf_index_t* index, iree_host_size_t i);

typedef iree_status_t (*iree_hal_amdxdna_ctrlcode_elf_kernel_callback_fn_t)(
    void* user_data, iree_string_view_t kernel_name);

// Calls `callback` with each kernel group symbol in section order. ELFs
// without groups have no named kernels and produce no calls.
iree_status_t iree_hal_amdxdna_ctrlcode_elf_enumerate_kernels(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_kernel_callback_fn_t callback,
    void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_CTRLCODE_ELF_H_
