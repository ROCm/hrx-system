// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"

#include <string.h>

// AIE-visible DDR address offset added to every shim-DMA buffer address.
// Validated for npu4 / AIE2P_STRIX_B0 (the only target the chained path is
// enabled for); other AIE generations may use a different offset / BD address
// layout. This is an AIE DMA address ABI, not a native queue command window.
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
  if (op >= 128) {  // Custom op.
    if (p + 8 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 4);
  }
  return 4;
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
    uint32_t bd1 = iree_hal_amdxdna_read_u32(b + offset + 4);
    uint32_t bd2 = iree_hal_amdxdna_read_u32(b + offset + 8);
    uint64_t base = ((uint64_t)(bd2 & 0xFFFF) << 32) | bd1;
    base += args[arg_idx] + arg_plus + iree_hal_amdxdna_ddr_aie_addr_offset;
    bd1 = (uint32_t)(base & 0xFFFFFFFC);
    bd2 = (bd2 & 0xFFFF0000) | (uint32_t)(base >> 32);
    iree_hal_amdxdna_write_u32(b + offset + 4, bd1);
    iree_hal_amdxdna_write_u32(b + offset + 8, bd2);
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
    const uint32_t template_bd1 =
        iree_hal_amdxdna_read_u32(template_bytes + offset + 4);
    const uint32_t template_bd2 =
        iree_hal_amdxdna_read_u32(template_bytes + offset + 8);
    uint64_t base = ((uint64_t)(template_bd2 & 0xFFFF) << 32) | template_bd1;
    base += args[arg_idx] + arg_plus + iree_hal_amdxdna_ddr_aie_addr_offset;
    const uint32_t bd1 = (uint32_t)(base & 0xFFFFFFFC);
    const uint32_t bd2 = (template_bd2 & 0xFFFF0000) | (uint32_t)(base >> 32);
    iree_hal_amdxdna_write_u32(dst_bytes + offset + 4, bd1);
    iree_hal_amdxdna_write_u32(dst_bytes + offset + 8, bd2);
  }
  return iree_ok_status();
}
