// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_data_layout.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_elf.h"

#define IREE_HAL_AMDGPU_ELF_SHT_DYNSYM 11
#define IREE_HAL_AMDGPU_ELF_STT_OBJECT 1
#define IREE_HAL_AMDGPU_ELF_STB_GLOBAL 1
#define IREE_HAL_AMDGPU_ELF_STB_WEAK 2
#define IREE_HAL_AMDGPU_ELF_SHN_UNDEF 0
#define IREE_HAL_AMDGPU_ELF_SHF_ALLOC 0x2

// Sorted borrowed kernel-descriptor names used while decoding a marked layout.
typedef struct iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t {
  // Pointers into decoded kernel metadata and ELF-only kernel records.
  const iree_string_view_t** names;
  // Number of entries in |names|.
  iree_host_size_t count;
} iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t;

static int iree_hal_amdgpu_hsaco_data_layout_compare_descriptor_names(
    const void* lhs_ptr, const void* rhs_ptr) {
  const iree_string_view_t* lhs = *(const iree_string_view_t* const*)lhs_ptr;
  const iree_string_view_t* rhs = *(const iree_string_view_t* const*)rhs_ptr;
  return iree_string_view_compare(*lhs, *rhs);
}

static iree_status_t
iree_hal_amdgpu_hsaco_data_layout_descriptor_index_initialize(
    const iree_hal_amdgpu_hsaco_metadata_t* metadata,
    iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  if (!iree_host_size_checked_add(metadata->kernel_count,
                                  metadata->elf_kernel_symbol_count,
                                  &out_index->count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU ASAN kernel descriptor index count overflows");
  }
  if (out_index->count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      metadata->host_allocator, out_index->count, sizeof(out_index->names[0]),
      (void**)&out_index->names));
  iree_host_size_t index = 0;
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    out_index->names[index++] = &metadata->kernels[i].symbol_name;
  }
  for (iree_host_size_t i = 0; i < metadata->elf_kernel_symbol_count; ++i) {
    out_index->names[index++] = &metadata->elf_kernel_symbols[i].symbol_name;
  }
  qsort(out_index->names, out_index->count, sizeof(out_index->names[0]),
        iree_hal_amdgpu_hsaco_data_layout_compare_descriptor_names);
  return iree_ok_status();
}

static void iree_hal_amdgpu_hsaco_data_layout_descriptor_index_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t* index) {
  iree_allocator_free(host_allocator, index->names);
  memset(index, 0, sizeof(*index));
}

static bool iree_hal_amdgpu_hsaco_data_layout_descriptor_index_contains(
    const iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t* index,
    iree_string_view_t name) {
  iree_host_size_t lower_bound = 0;
  iree_host_size_t upper_bound = index->count;
  while (lower_bound < upper_bound) {
    const iree_host_size_t middle =
        lower_bound + (upper_bound - lower_bound) / 2;
    const int comparison =
        iree_string_view_compare(name, *index->names[middle]);
    if (comparison < 0) {
      upper_bound = middle;
    } else if (comparison > 0) {
      lower_bound = middle + 1;
    } else {
      return true;
    }
  }
  return false;
}

static bool iree_hal_amdgpu_hsaco_data_layout_is_global_object(
    const uint8_t* symbol) {
  const uint8_t binding = symbol[4] >> 4;
  const uint8_t type = symbol[4] & 0x0F;
  const uint16_t section_index = iree_unaligned_load_le_u16(symbol + 6);
  return type == IREE_HAL_AMDGPU_ELF_STT_OBJECT &&
         section_index != IREE_HAL_AMDGPU_ELF_SHN_UNDEF &&
         (binding == IREE_HAL_AMDGPU_ELF_STB_GLOBAL ||
          binding == IREE_HAL_AMDGPU_ELF_STB_WEAK);
}

static bool iree_hal_amdgpu_hsaco_data_layout_calculate_redzone_end(
    uint64_t address, uint64_t byte_length, uint64_t* out_redzone_end) {
  const uint64_t granule =
      UINT64_C(1) << IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_SHADOW_SCALE_SHIFT;
  uint64_t object_end = 0;
  if (!iree_checked_add_u64(address, byte_length, &object_end) ||
      object_end > UINT64_MAX - (granule - 1)) {
    return false;
  }
  const uint64_t redzone_address =
      (object_end + (granule - 1)) & ~(granule - 1);
  return iree_checked_add_u64(redzone_address, granule, out_redzone_end);
}

// Returns the redzone end for a range previously accepted by
// iree_hal_amdgpu_hsaco_data_layout_calculate_redzone_end.
static uint64_t iree_hal_amdgpu_hsaco_data_layout_redzone_end(
    uint64_t address, uint64_t byte_length) {
  const uint64_t granule =
      UINT64_C(1) << IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_SHADOW_SCALE_SHIFT;
  const uint64_t object_end = address + byte_length;
  const uint64_t redzone_address =
      (object_end + (granule - 1)) & ~(granule - 1);
  return redzone_address + granule;
}

static iree_status_t iree_hal_amdgpu_hsaco_data_layout_decode_object(
    const iree_hal_amdgpu_hsaco_metadata_t* metadata,
    const iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t*
        descriptor_index,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbols,
    iree_host_size_t symbol_index, bool* out_is_data_object,
    iree_hal_amdgpu_hsaco_metadata_data_object_t* out_data_object) {
  *out_is_data_object = false;
  memset(out_data_object, 0, sizeof(*out_data_object));

  const uint8_t* symbol = NULL;
  if (!iree_hal_amdgpu_hsaco_elf_symbol(metadata->elf_data, symbols,
                                        symbol_index, &symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN dynamic symbol exceeds ELF bounds");
  }
  if (!iree_hal_amdgpu_hsaco_data_layout_is_global_object(symbol)) {
    return iree_ok_status();
  }

  const iree_string_view_t name = iree_hal_amdgpu_hsaco_elf_symbol_name(
      metadata->elf_data, symbols, symbol);
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN dynamic data symbol has no valid name");
  }
  if (iree_hal_amdgpu_hsaco_data_layout_descriptor_index_contains(
          descriptor_index, name)) {
    return iree_ok_status();
  }

  const uint16_t section_index = iree_unaligned_load_le_u16(symbol + 6);
  const uint8_t* data_section = NULL;
  if (!iree_hal_amdgpu_hsaco_elf_section(metadata->elf_data, section_index,
                                         &data_section)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN data symbol `%.*s` has invalid section index %u",
        (int)name.size, name.data, section_index);
  }
  const uint64_t section_flags = iree_unaligned_load_le_u64(data_section + 8);
  const uint64_t section_address =
      iree_unaligned_load_le_u64(data_section + 16);
  const uint64_t section_size = iree_unaligned_load_le_u64(data_section + 32);
  uint64_t section_end = 0;
  if ((section_flags & IREE_HAL_AMDGPU_ELF_SHF_ALLOC) == 0 ||
      !iree_checked_add_u64(section_address, section_size, &section_end)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN data symbol `%.*s` is not in a valid allocated section",
        (int)name.size, name.data);
  }

  const uint64_t virtual_address = iree_unaligned_load_le_u64(symbol + 8);
  const uint64_t byte_length = iree_unaligned_load_le_u64(symbol + 16);
  const uint64_t granule =
      UINT64_C(1) << IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_SHADOW_SCALE_SHIFT;
  uint64_t redzone_end = 0;
  if ((virtual_address & (granule - 1)) != 0 ||
      !iree_hal_amdgpu_hsaco_data_layout_calculate_redzone_end(
          virtual_address, byte_length, &redzone_end) ||
      virtual_address < section_address || redzone_end > section_end) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN data symbol `%.*s` range [0x%016" PRIx64 ", +%" PRIu64
        ") lacks its aligned trailing redzone",
        (int)name.size, name.data, virtual_address, byte_length);
  }

  *out_is_data_object = true;
  *out_data_object = (iree_hal_amdgpu_hsaco_metadata_data_object_t){
      .name = name,
      .virtual_address = virtual_address,
      .byte_length = byte_length,
  };
  return iree_ok_status();
}

static int iree_hal_amdgpu_hsaco_data_layout_compare_objects(
    const void* lhs_ptr, const void* rhs_ptr) {
  const iree_hal_amdgpu_hsaco_metadata_data_object_t* lhs =
      (const iree_hal_amdgpu_hsaco_metadata_data_object_t*)lhs_ptr;
  const iree_hal_amdgpu_hsaco_metadata_data_object_t* rhs =
      (const iree_hal_amdgpu_hsaco_metadata_data_object_t*)rhs_ptr;
  return lhs->virtual_address < rhs->virtual_address
             ? -1
             : lhs->virtual_address > rhs->virtual_address;
}

static iree_status_t iree_hal_amdgpu_hsaco_data_layout_populate_marked(
    iree_hal_amdgpu_hsaco_metadata_t* metadata,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbols) {
  iree_hal_amdgpu_hsaco_data_layout_descriptor_index_t descriptor_index;
  iree_status_t status =
      iree_hal_amdgpu_hsaco_data_layout_descriptor_index_initialize(
          metadata, &descriptor_index);

  iree_host_size_t data_object_count = 0;
  for (iree_host_size_t i = 0; i < symbols->count && iree_status_is_ok(status);
       ++i) {
    const uint8_t* symbol = NULL;
    if (!iree_hal_amdgpu_hsaco_elf_symbol(metadata->elf_data, symbols, i,
                                          &symbol) ||
        !iree_hal_amdgpu_hsaco_data_layout_is_global_object(symbol)) {
      continue;
    }
    const iree_string_view_t name = iree_hal_amdgpu_hsaco_elf_symbol_name(
        metadata->elf_data, symbols, symbol);
    if (!iree_hal_amdgpu_hsaco_data_layout_descriptor_index_contains(
            &descriptor_index, name)) {
      ++data_object_count;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        metadata->host_allocator, data_object_count,
        sizeof(metadata->data_objects[0]), (void**)&metadata->data_objects);
  }

  iree_host_size_t data_object_index = 0;
  iree_host_size_t marker_count = 0;
  for (iree_host_size_t i = 0; i < symbols->count && iree_status_is_ok(status);
       ++i) {
    bool is_data_object = false;
    iree_hal_amdgpu_hsaco_metadata_data_object_t data_object;
    status = iree_hal_amdgpu_hsaco_data_layout_decode_object(
        metadata, &descriptor_index, symbols, i, &is_data_object, &data_object);
    if (!iree_status_is_ok(status) || !is_data_object) {
      continue;
    }
    if (iree_string_view_equal(
            data_object.name,
            IREE_SV(IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME))) {
      ++marker_count;
      if (data_object.byte_length !=
          IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_BYTE_LENGTH) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU ASAN global layout marker has invalid length %" PRIu64,
            data_object.byte_length);
        continue;
      }
    }
    metadata->data_objects[data_object_index++] = data_object;
  }
  IREE_ASSERT(!iree_status_is_ok(status) ||
              data_object_index == data_object_count);
  if (iree_status_is_ok(status) && IREE_UNLIKELY(marker_count != 1)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN global layout requires exactly one marker object");
  }

  if (iree_status_is_ok(status)) {
    qsort(metadata->data_objects, data_object_count,
          sizeof(metadata->data_objects[0]),
          iree_hal_amdgpu_hsaco_data_layout_compare_objects);
    for (iree_host_size_t i = 1;
         i < data_object_count && iree_status_is_ok(status); ++i) {
      const uint64_t previous_redzone_end =
          iree_hal_amdgpu_hsaco_data_layout_redzone_end(
              metadata->data_objects[i - 1].virtual_address,
              metadata->data_objects[i - 1].byte_length);
      if (IREE_UNLIKELY(metadata->data_objects[i].virtual_address <
                        previous_redzone_end)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU ASAN data objects `%.*s` and `%.*s` overlap a redzone",
            (int)metadata->data_objects[i - 1].name.size,
            metadata->data_objects[i - 1].name.data,
            (int)metadata->data_objects[i].name.size,
            metadata->data_objects[i].name.data);
      }
    }
  }

  iree_hal_amdgpu_hsaco_data_layout_descriptor_index_deinitialize(
      metadata->host_allocator, &descriptor_index);
  if (iree_status_is_ok(status)) {
    metadata->data_object_count = data_object_count;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_hsaco_data_layout_populate(
    iree_hal_amdgpu_hsaco_metadata_t* metadata) {
  iree_hal_amdgpu_hsaco_elf_symbol_table_t marked_symbols = {0};
  const uint8_t* header = metadata->elf_data.data;
  const uint16_t section_count = iree_unaligned_load_le_u16(header + 60);
  for (uint16_t section_index = 0; section_index < section_count;
       ++section_index) {
    const uint8_t* section = NULL;
    if (!iree_hal_amdgpu_hsaco_elf_section(metadata->elf_data, section_index,
                                           &section) ||
        iree_unaligned_load_le_u32(section + 4) !=
            IREE_HAL_AMDGPU_ELF_SHT_DYNSYM) {
      continue;
    }
    iree_hal_amdgpu_hsaco_elf_symbol_table_t symbols = {0};
    if (!iree_hal_amdgpu_hsaco_elf_symbol_table_initialize(metadata->elf_data,
                                                           section, &symbols)) {
      continue;
    }
    for (iree_host_size_t i = 0; i < symbols.count; ++i) {
      const uint8_t* symbol = NULL;
      if (!iree_hal_amdgpu_hsaco_elf_symbol(metadata->elf_data, &symbols, i,
                                            &symbol) ||
          !iree_hal_amdgpu_hsaco_data_layout_is_global_object(symbol)) {
        continue;
      }
      const iree_string_view_t name = iree_hal_amdgpu_hsaco_elf_symbol_name(
          metadata->elf_data, &symbols, symbol);
      if (!iree_string_view_equal(
              name,
              IREE_SV(IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME))) {
        continue;
      }
      if (marked_symbols.section) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU ASAN global layout marker is defined more than once");
      }
      marked_symbols = symbols;
    }
  }
  if (!marked_symbols.section) {
    return iree_ok_status();
  }
  return iree_hal_amdgpu_hsaco_data_layout_populate_marked(metadata,
                                                           &marked_symbols);
}
