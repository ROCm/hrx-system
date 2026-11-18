// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// AIE-visible DDR address offset added to every shim-DMA buffer address.
// AIE2P uses the 48-bit bd[1]/bd[2] split; AIE4 patch57 writes bd[1]/bd[0]
// and adds the same 2GB window. Linux PASID/SVA uses mmap VA + window as a
// 48-bit address. Windows KMD rewrites GpuVa+window to SPA+window.
static const uint64_t iree_hal_amdxdna_ddr_aie_addr_offset = 0x80000000ULL;

// AIEC RTP lowering emits amdaie.npu.write32 values tagged with this sentinel;
// the HAL replaces the low bits with the corresponding dispatch constant before
// handing the transaction to firmware.
static const uint32_t iree_hal_amdxdna_write32_constant_sentinel = 0xA1EC0000u;
static const uint32_t iree_hal_amdxdna_write32_constant_mask = 0xFFFF0000u;

// Misalignment-safe little-endian 32-bit read: a malformed op size can leave
// `p` non-word-aligned, so read through memcpy rather than an unaligned cast.
static uint32_t iree_hal_amdxdna_read_u32(const uint8_t* p) {
  uint32_t value;
  memcpy(&value, p, sizeof(value));
  return value;
}

static void iree_hal_amdxdna_write_u32(uint8_t* p, uint32_t value) {
  memcpy(p, &value, sizeof(value));
}

// Shim-DMA BD addresses are 48-bit and 4-byte aligned. Reject overflow and
// unaligned arg+arg_plus sums instead of masking, so a miscompiled TXN fails
// closed rather than silently pointing at a neighboring descriptor.
static bool iree_hal_amdxdna_bd_base_address(uint64_t arg, uint32_t arg_plus,
                                             uint64_t* out_base) {
  const uint64_t addend = (uint64_t)arg_plus;
  if (arg > UINT64_MAX - addend) return false;
  const uint64_t sum = arg + addend;
  if (sum > UINT64_MAX - iree_hal_amdxdna_ddr_aie_addr_offset) return false;
  const uint64_t base = sum + iree_hal_amdxdna_ddr_aie_addr_offset;
  if ((base & 0x3u) != 0) return false;
  if (base > 0x0000FFFFFFFFFFFFULL) return false;
  *out_base = base;
  return true;
}

typedef struct iree_hal_amdxdna_bd_patch_site_t {
  uint32_t key;
  uint32_t byte_offset;
} iree_hal_amdxdna_bd_patch_site_t;

static uint32_t iree_hal_amdxdna_bd_patch_key(uint32_t register_address) {
  return register_address & ((0x7Fu << 25) | (0x1Fu << 20) | (0x1Fu << 5));
}

void iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_write32_constant_patch_list_t* list) {
  if (!list) return;
  iree_allocator_free(host_allocator, list->data);
  list->data = NULL;
  list->count = 0;
}

uint32_t iree_hal_amdxdna_txn_op_size(const uint8_t* b, size_t total,
                                      size_t p) {
  if (p >= total) return 0;
  uint8_t op = b[p];
  if (op == 0) {  // WRITE32.
    if (p + 24 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 20);
  }
  if (op == 1) {  // BLOCKWRITE.
    if (p + 16 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 12);
  }
  if (op == 3 || op == 4) {
    if (p + 28 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 24);
  }
  if (op == 8) {  // LOAD_PDI.
    return p + 16 <= total ? 16 : 0;
  }
  if (op == 10) {  // CREATE_SCRATCHPAD.
    return p + 16 <= total ? 16 : 0;
  }
  if (op == 11) {  // UPDATE_STATE_TABLE.
    return p + 8 <= total ? 8 : 0;
  }
  if (op == 12) {  // UPDATE_REG from scratchpad state.
    return p + 12 <= total ? 12 : 0;
  }
  if (op == 13) {  // UPDATE_SCRATCH.
    return p + 4 <= total ? 4 : 0;
  }
  if (op >= 128) {  // Custom op.
    if (p + 8 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 4);
  }
  return 4;
}

void iree_hal_amdxdna_host_patch_table_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_host_patch_table_t* table) {
  if (!table) return;
  iree_allocator_free(host_allocator, table->data);
  table->data = NULL;
  table->count = 0;
}

iree_status_t iree_hal_amdxdna_build_host_patch_table(
    iree_allocator_t host_allocator, iree_const_byte_span_t transaction,
    iree_hal_amdxdna_host_patch_table_t* out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna host patch table output is NULL");
  }
  memset(out_table, 0, sizeof(*out_table));
  if (!transaction.data || transaction.data_length < 16 ||
      transaction.data_length % sizeof(uint32_t) != 0 ||
      transaction.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna XAie transaction length is invalid");
  }
  const uint8_t* bytes = transaction.data;
  const size_t total = transaction.data_length;
  const uint32_t op_count = iree_hal_amdxdna_read_u32(bytes + 8);
  if (op_count > (total - 16) / sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna XAie operation count exceeds transaction");
  }
  if (op_count == 0) {
    return total == 16 ? iree_ok_status()
                       : iree_make_status(
                             IREE_STATUS_INVALID_ARGUMENT,
                             "amdxdna XAie transaction has trailing bytes");
  }

  iree_hal_amdxdna_bd_patch_site_t* sites = NULL;
  uint32_t* patches = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, op_count, sizeof(*sites), (void**)&sites));
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, op_count, 3 * sizeof(*patches), (void**)&patches);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, sites);
    return status;
  }

  iree_host_size_t site_count = 0;
  iree_host_size_t patch_count = 0;
  size_t offset = 16;
  for (uint32_t i = 0; i < op_count; ++i) {
    const uint32_t op_size = iree_hal_amdxdna_txn_op_size(bytes, total, offset);
    if (op_size == 0 || offset + op_size > total) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "amdxdna XAie operation %u is malformed", i);
      goto cleanup;
    }
    const uint8_t opcode = bytes[offset];
    if (opcode == 1) {
      if (op_size < 28) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "amdxdna BLOCKWRITE operation is truncated");
        goto cleanup;
      }
      sites[site_count++] = (iree_hal_amdxdna_bd_patch_site_t){
          .key = iree_hal_amdxdna_bd_patch_key(
              iree_hal_amdxdna_read_u32(bytes + offset + 8)),
          .byte_offset = (uint32_t)(offset + 16),
      };
    } else if (opcode == 0x81) {
      if (op_size < 44) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "amdxdna DDR_PATCH operation is truncated");
        goto cleanup;
      }
      const uint32_t key = iree_hal_amdxdna_bd_patch_key(
          iree_hal_amdxdna_read_u32(bytes + offset + 24));
      iree_host_size_t site_index = site_count;
      while (site_index > 0 && sites[site_index - 1].key != key) --site_index;
      if (site_index == 0) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna DDR_PATCH has no preceding matching BLOCKWRITE");
        goto cleanup;
      }
      patches[patch_count++] = sites[site_index - 1].byte_offset;
      patches[patch_count++] = iree_hal_amdxdna_read_u32(bytes + offset + 32);
      patches[patch_count++] = iree_hal_amdxdna_read_u32(bytes + offset + 40);
    }
    offset += op_size;
  }
  if (offset != total) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna XAie transaction has trailing bytes");
    goto cleanup;
  }
  out_table->data = patches;
  out_table->count = patch_count;
  patches = NULL;

cleanup:
  iree_allocator_free(host_allocator, patches);
  iree_allocator_free(host_allocator, sites);
  return status;
}

static iree_status_t iree_hal_amdxdna_visit_write32_constant_sites(
    const uint32_t* txn, size_t txn_words,
    iree_hal_amdxdna_write32_constant_patch_site_t* sites,
    iree_host_size_t site_capacity, iree_host_size_t* out_site_count) {
  *out_site_count = 0;
  if (txn_words < 4) return iree_ok_status();
  const uint8_t* b = (const uint8_t*)txn;
  size_t total = txn_words * sizeof(uint32_t);
  uint32_t num_ops = txn[2];  // TXN header word 2 = NumOps.
  size_t p = 16;              // Past the 16-byte XAie_TxnHeader.
  for (uint32_t i = 0; i < num_ops; ++i) {
    uint32_t sz = iree_hal_amdxdna_txn_op_size(b, total, p);
    if (sz == 0 || p + sz > total) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna write32 RTP patch saw malformed transaction op %u at byte "
          "offset %zu",
          i, p);
    }
    if (b[p] == 0) {  // WRITE32: patch sentinel values from HAL constants.
      uint32_t value = iree_hal_amdxdna_read_u32(b + p + 16);
      if ((value & iree_hal_amdxdna_write32_constant_mask) ==
          iree_hal_amdxdna_write32_constant_sentinel) {
        if (sites) {
          if (*out_site_count >= site_capacity) {
            return iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "amdxdna write32 RTP constant patch site list overflow");
          }
          sites[*out_site_count] =
              (iree_hal_amdxdna_write32_constant_patch_site_t){
                  .byte_offset = (uint32_t)(p + 16),
                  .constant_index =
                      value & ~iree_hal_amdxdna_write32_constant_mask,
              };
        }
        ++*out_site_count;
      }
    }
    p += sz;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_build_write32_constant_patch_list(
    iree_allocator_t host_allocator, const uint32_t* txn, size_t txn_words,
    iree_hal_amdxdna_write32_constant_patch_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  iree_host_size_t site_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_visit_write32_constant_sites(
      txn, txn_words, NULL, 0, &site_count));
  if (site_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, site_count,
                                                   sizeof(*out_list->data),
                                                   (void**)&out_list->data));
  out_list->count = site_count;
  iree_host_size_t filled_count = 0;
  iree_status_t status = iree_hal_amdxdna_visit_write32_constant_sites(
      txn, txn_words, out_list->data, out_list->count, &filled_count);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_write32_constant_patch_list_deinitialize(host_allocator,
                                                              out_list);
    return status;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_patch_write32_constants(
    uint32_t* txn, size_t txn_words, iree_const_byte_span_t constants) {
  if (txn_words < 4) return iree_ok_status();
  uint8_t* b = (uint8_t*)txn;
  size_t total = txn_words * sizeof(uint32_t);
  uint32_t num_ops = txn[2];  // TXN header word 2 = NumOps.
  size_t p = 16;              // Past the 16-byte XAie_TxnHeader.
  for (uint32_t i = 0; i < num_ops; ++i) {
    uint32_t sz = iree_hal_amdxdna_txn_op_size(b, total, p);
    if (sz == 0 || p + sz > total) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna write32 RTP patch saw malformed transaction op %u at byte "
          "offset %zu",
          i, p);
    }
    if (b[p] == 0) {  // WRITE32: patch sentinel values from HAL constants.
      uint32_t value = iree_hal_amdxdna_read_u32(b + p + 16);
      if ((value & iree_hal_amdxdna_write32_constant_mask) ==
          iree_hal_amdxdna_write32_constant_sentinel) {
        uint32_t constant_index =
            value & ~iree_hal_amdxdna_write32_constant_mask;
        iree_host_size_t byte_offset =
            (iree_host_size_t)constant_index * sizeof(uint32_t);
        if (byte_offset + sizeof(uint32_t) > constants.data_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "amdxdna write32 RTP constant index %u out of bounds for "
              "%zu-byte constants block",
              constant_index, constants.data_length);
        }
        memcpy(&value, constants.data + byte_offset, sizeof(uint32_t));
        iree_hal_amdxdna_write_u32(b + p + 16, value);
      }
    }
    p += sz;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_patch_write32_constants_with_list(
    uint32_t* txn, size_t txn_words,
    const iree_hal_amdxdna_write32_constant_patch_list_t* patch_list,
    iree_const_byte_span_t constants) {
  if (!patch_list || patch_list->count == 0) return iree_ok_status();
  uint8_t* b = (uint8_t*)txn;
  size_t total = txn_words * sizeof(uint32_t);
  for (iree_host_size_t i = 0; i < patch_list->count; ++i) {
    const iree_hal_amdxdna_write32_constant_patch_site_t* site =
        &patch_list->data[i];
    if ((size_t)site->byte_offset + sizeof(uint32_t) > total ||
        (site->byte_offset & 0x3u) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna cached write32 RTP constant patch site offset %u out of "
          "bounds for %zu-byte transaction",
          site->byte_offset, total);
    }
    const iree_host_size_t constant_byte_offset =
        (iree_host_size_t)site->constant_index * sizeof(uint32_t);
    if (constant_byte_offset + sizeof(uint32_t) > constants.data_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna cached write32 RTP constant index %u out of bounds for "
          "%zu-byte constants block",
          site->constant_index, constants.data_length);
    }
    uint32_t value = 0;
    memcpy(&value, constants.data + constant_byte_offset, sizeof(uint32_t));
    iree_hal_amdxdna_write_u32(b + site->byte_offset, value);
  }
  return iree_ok_status();
}

bool iree_hal_amdxdna_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
                                        const uint32_t* patches,
                                        size_t patch_count,
                                        const uint64_t* args,
                                        size_t arg_count) {
  if (patch_count % 3 != 0) return false;
  if (patch_count && !patches) return false;
  uint8_t* b = (uint8_t*)ctrl_code;
  size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patch_count; i += 3) {
    uint32_t offset = patches[i];        // byte offset of the BD base word
    uint32_t arg_idx = patches[i + 1];   // index into `args`
    uint32_t arg_plus = patches[i + 2];  // byte addend into that buffer
    if (arg_idx >= arg_count) return false;
    // We touch bd[1] at offset+4 and bd[2] at offset+8 (4 bytes each).
    if ((size_t)offset + 12 > total || (offset & 0x3u) != 0) {
      return false;
    }
    uint32_t bd2 = iree_hal_amdxdna_read_u32(b + offset + 8);
    // DDR_PATCH arg_plus already carries the intra-buffer byte offset. Compute
    // the final AIE-visible BD address from scratch instead of accumulating
    // onto a compiler-baked address word, which would double-count sub-buffer
    // BDs. bd[2] bits [31:16] carry BD control state and must be preserved.
    uint64_t base = 0;
    if (!iree_hal_amdxdna_bd_base_address(args[arg_idx], arg_plus, &base)) {
      return false;
    }
    uint32_t bd1 = (uint32_t)base;
    bd2 = (bd2 & 0xFFFF0000) | (uint32_t)(base >> 32);
    iree_hal_amdxdna_write_u32(b + offset + 4, bd1);
    iree_hal_amdxdna_write_u32(b + offset + 8, bd2);
  }
  return true;
}

bool iree_hal_amdxdna_apply_patch_table_aie4(
    uint32_t* ctrl_code, size_t ctrl_words, const uint32_t* patches,
    size_t patch_count, const uint64_t* args, size_t arg_count,
    uint64_t control_code_addr) {
  if (patch_count % 3 != 0) return false;
  if (patch_count && !patches) return false;
  uint8_t* b = (uint8_t*)ctrl_code;
  size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patch_count; i += 3) {
    uint32_t offset = patches[i];
    uint32_t arg_idx = patches[i + 1];
    uint32_t arg_plus = patches[i + 2];
    uint64_t patch_va = 0;
    if (arg_idx == IREE_HAL_AMDXDNA_PATCH_ARG_CONTROL_CODE) {
      patch_va = control_code_addr;
    } else if (arg_idx >= arg_count) {
      return false;
    } else {
      patch_va = args[arg_idx];
    }
    if ((size_t)offset + 8 > total || (offset & 0x3u) != 0) {
      return false;
    }
    // Each microcontroller gets its own copy of the same page. The loader
    // keeps both relocation entries and packs one page, so applying the
    // second entry would add the buffer address on top of the first result.
    bool already_applied = false;
    for (size_t prev = 0; prev < i; prev += 3) {
      if (patches[prev] == offset) {
        already_applied = true;
        break;
      }
    }
    if (already_applied) continue;
    uint32_t bd0 = iree_hal_amdxdna_read_u32(b + offset);
    uint32_t bd1 = iree_hal_amdxdna_read_u32(b + offset + 4);
    // Keep the compiler-baked offset. Multi-column DMAs store the chunk
    // offset here (0, 4KB, 8KB, ...); control-code patches store the payload
    // offset inside the instruction image.
    uint64_t baked = ((uint64_t)(bd0 & 0x1FFFFFFu) << 32) | (uint64_t)bd1;
    uint64_t base =
        baked + patch_va + arg_plus + iree_hal_amdxdna_ddr_aie_addr_offset;
    bd1 = (uint32_t)base;
    bd0 = (bd0 & 0xFE000000u) | (uint32_t)((base >> 32) & 0x1FFFFFFu);
    iree_hal_amdxdna_write_u32(b + offset, bd0);
    iree_hal_amdxdna_write_u32(b + offset + 4, bd1);
  }
  return true;
}

bool iree_hal_amdxdna_apply_pdi_relocations(
    uint32_t* ctrl_code, size_t ctrl_words,
    const iree_hal_amdxdna_pdi_relocation_t* relocations,
    size_t relocation_count, const uint64_t* pdi_addresses, size_t pdi_count) {
  if (relocation_count == 0) return true;
  if (!ctrl_code || !relocations || !pdi_addresses) return false;
  if (ctrl_words > SIZE_MAX / sizeof(uint32_t)) return false;
  const size_t total = ctrl_words * sizeof(uint32_t);
  uint8_t* bytes = (uint8_t*)ctrl_code;
  for (size_t i = 0; i < relocation_count; ++i) {
    const iree_hal_amdxdna_pdi_relocation_t* relocation = &relocations[i];
    if (relocation->pdi_ordinal >= pdi_count ||
        (relocation->transaction_offset % sizeof(uint32_t)) != 0 ||
        total < sizeof(uint64_t) ||
        relocation->transaction_offset > total - sizeof(uint64_t)) {
      return false;
    }
    const uint64_t base = pdi_addresses[relocation->pdi_ordinal];
    uint64_t address = 0;
    if (relocation->addend >= 0) {
      const uint64_t addend = (uint64_t)relocation->addend;
      if (addend > UINT64_MAX - base) return false;
      address = base + addend;
    } else {
      const uint64_t magnitude = (uint64_t)(-(relocation->addend + 1)) + 1;
      if (magnitude > base) return false;
      address = base - magnitude;
    }
    iree_hal_amdxdna_write_u32(bytes + relocation->transaction_offset,
                               (uint32_t)address);
    iree_hal_amdxdna_write_u32(bytes + relocation->transaction_offset + 4,
                               (uint32_t)(address >> 32));
  }
  return true;
}

bool iree_hal_amdxdna_apply_control_parameter_relocations(
    uint32_t* ctrl_code, size_t ctrl_words,
    const iree_hal_amdxdna_control_parameter_relocation_t* relocations,
    size_t relocation_count, uint64_t control_parameter_address) {
  if (relocation_count == 0) return true;
  if (!ctrl_code || !relocations || ctrl_words > SIZE_MAX / sizeof(uint32_t)) {
    return false;
  }
  const size_t total = ctrl_words * sizeof(uint32_t);
  uint8_t* bytes = (uint8_t*)ctrl_code;
  for (size_t i = 0; i < relocation_count; ++i) {
    const iree_hal_amdxdna_control_parameter_relocation_t* relocation =
        &relocations[i];
    if ((relocation->transaction_offset % sizeof(uint32_t)) != 0 ||
        total < sizeof(uint64_t) ||
        relocation->transaction_offset > total - sizeof(uint64_t)) {
      return false;
    }
    uint64_t address = 0;
    if (relocation->addend >= 0) {
      const uint64_t addend = (uint64_t)relocation->addend;
      if (addend > UINT64_MAX - control_parameter_address) return false;
      address = control_parameter_address + addend;
    } else {
      const uint64_t magnitude = (uint64_t)(-(relocation->addend + 1)) + 1;
      if (magnitude > control_parameter_address) return false;
      address = control_parameter_address - magnitude;
    }
    iree_hal_amdxdna_write_u32(bytes + relocation->transaction_offset,
                               (uint32_t)address);
    iree_hal_amdxdna_write_u32(bytes + relocation->transaction_offset + 4,
                               (uint32_t)(address >> 32));
  }
  return true;
}

iree_status_t iree_hal_amdxdna_stage_control_parameters(
    const iree_hal_executable_function_parameter_t* parameters,
    iree_host_size_t parameter_count, iree_const_byte_span_t constants,
    iree_byte_span_t state_table) {
  if (state_table.data_length != 32 * sizeof(uint32_t) || !state_table.data ||
      (parameter_count != 0 && !parameters)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna control parameter state table is invalid");
  }
  memset(state_table.data, 0, state_table.data_length);
  for (iree_host_size_t i = 0; i < parameter_count; ++i) {
    const iree_hal_executable_function_parameter_t* parameter = &parameters[i];
    if (parameter->type !=
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT ||
        (parameter->flags &
         IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET) == 0 ||
        parameter->size == 0 || parameter->offset > constants.data_length ||
        parameter->size > constants.data_length - parameter->offset ||
        parameter->native_abi_offset > state_table.data_length ||
        parameter->size >
            state_table.data_length - parameter->native_abi_offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna control parameter %" PRIhsz " is out of bounds", i);
    }
    memcpy(state_table.data + parameter->native_abi_offset,
           constants.data + parameter->offset, parameter->size);
  }
  return iree_ok_status();
}

bool iree_hal_amdxdna_mark_referenced_pdis(
    const iree_hal_amdxdna_pdi_relocation_t* relocations,
    size_t relocation_count, size_t pdi_count, uint8_t* referenced_pdis) {
  if (pdi_count != 0 && !referenced_pdis) return false;
  if (pdi_count != 0) memset(referenced_pdis, 0, pdi_count);
  if (relocation_count != 0 && !relocations) return false;
  for (size_t i = 0; i < relocation_count; ++i) {
    if (relocations[i].pdi_ordinal >= pdi_count) return false;
    referenced_pdis[relocations[i].pdi_ordinal] = 1;
  }
  return true;
}

iree_status_t iree_hal_amdxdna_patch_dynamic_fields_from_template(
    uint32_t* ctrl_code, const uint32_t* template_code, size_t ctrl_words,
    const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patches,
    iree_const_byte_span_t constants, const uint32_t* patches,
    size_t patch_count, const uint64_t* args, size_t arg_count) {
  if (!ctrl_code || !template_code) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna control-code patch requires non-NULL "
                            "destination and template buffers");
  }

  if (constant_patches) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_patch_write32_constants_with_list(
        ctrl_code, ctrl_words, constant_patches, constants));
  } else if (constants.data_length != 0 && ctrl_words >= 4) {
    const uint8_t* template_bytes = (const uint8_t*)template_code;
    uint8_t* dst_bytes = (uint8_t*)ctrl_code;
    const size_t total = ctrl_words * sizeof(uint32_t);
    const uint32_t num_ops = template_code[2];  // TXN header NumOps.
    size_t p = 16;                              // Past XAie_TxnHeader.
    for (uint32_t i = 0; i < num_ops; ++i) {
      const uint32_t sz =
          iree_hal_amdxdna_txn_op_size(template_bytes, total, p);
      if (sz == 0 || p + sz > total) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna cached write32 RTP patch saw malformed transaction op %u "
            "at byte offset %zu",
            i, p);
      }
      if (template_bytes[p] == 0) {  // WRITE32.
        uint32_t value = iree_hal_amdxdna_read_u32(template_bytes + p + 16);
        if ((value & iree_hal_amdxdna_write32_constant_mask) ==
            iree_hal_amdxdna_write32_constant_sentinel) {
          const uint32_t constant_index =
              value & ~iree_hal_amdxdna_write32_constant_mask;
          const iree_host_size_t byte_offset =
              (iree_host_size_t)constant_index * sizeof(uint32_t);
          if (byte_offset + sizeof(uint32_t) > constants.data_length) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "amdxdna cached write32 RTP constant index %u out of bounds "
                "for %zu-byte constants block",
                constant_index, constants.data_length);
          }
          memcpy(&value, constants.data + byte_offset, sizeof(uint32_t));
          iree_hal_amdxdna_write_u32(dst_bytes + p + 16, value);
        }
      }
      p += sz;
    }
  }

  if (patch_count % 3 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna host patch table must contain triples");
  }
  if (patch_count && !patches) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna host patch table pointer is NULL");
  }
  const uint8_t* template_bytes = (const uint8_t*)template_code;
  uint8_t* dst_bytes = (uint8_t*)ctrl_code;
  const size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patch_count; i += 3) {
    const uint32_t offset = patches[i];
    const uint32_t arg_idx = patches[i + 1];
    const uint32_t arg_plus = patches[i + 2];
    if (arg_idx >= arg_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna host patch arg index %u out of range",
                              arg_idx);
    }
    if ((size_t)offset + 12 > total || (offset & 0x3u) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna host patch offset %u is out of bounds or misaligned",
          offset);
    }
    const uint32_t template_bd2 =
        iree_hal_amdxdna_read_u32(template_bytes + offset + 8);
    // Overwrite (do not accumulate onto) the compiler-baked BD address. See
    // iree_hal_amdxdna_apply_patch_table for details.
    uint64_t base = 0;
    if (!iree_hal_amdxdna_bd_base_address(args[arg_idx], arg_plus, &base)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna host patch address for arg %u overflowed or is not 4-byte "
          "aligned",
          arg_idx);
    }
    const uint32_t bd1 = (uint32_t)base;
    const uint32_t bd2 = (template_bd2 & 0xFFFF0000) | (uint32_t)(base >> 32);
    iree_hal_amdxdna_write_u32(dst_bytes + offset + 4, bd1);
    iree_hal_amdxdna_write_u32(dst_bytes + offset + 8, bd2);
  }
  return iree_ok_status();
}
