// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/executable/data.h"

#include <string.h>

#include "iree/hal/utils/elf_format.h"

static bool iree_hal_task_executable_data_starts_with(
    iree_const_byte_span_t executable_data, iree_host_size_t prefix_length,
    const uint8_t* prefix) {
  return executable_data.data && executable_data.data_length >= prefix_length &&
         memcmp(executable_data.data, prefix, prefix_length) == 0;
}

typedef struct iree_hal_task_elf32_header_t {
  uint8_t ident[16];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint32_t entry;
  uint32_t program_header_offset;
  uint32_t section_header_offset;
  uint32_t flags;
  uint16_t header_size;
  uint16_t program_header_entry_size;
  uint16_t program_header_count;
  uint16_t section_header_entry_size;
  uint16_t section_header_count;
  uint16_t section_name_index;
} iree_hal_task_elf32_header_t;

typedef struct iree_hal_task_elf64_header_t {
  uint8_t ident[16];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t program_header_offset;
  uint64_t section_header_offset;
  uint32_t flags;
  uint16_t header_size;
  uint16_t program_header_entry_size;
  uint16_t program_header_count;
  uint16_t section_header_entry_size;
  uint16_t section_header_count;
  uint16_t section_name_index;
} iree_hal_task_elf64_header_t;

typedef struct iree_hal_task_elf32_program_header_t {
  uint32_t type;
  uint32_t offset;
  uint32_t virtual_address;
  uint32_t physical_address;
  uint32_t file_size;
  uint32_t memory_size;
  uint32_t flags;
  uint32_t alignment;
} iree_hal_task_elf32_program_header_t;

typedef struct iree_hal_task_elf64_program_header_t {
  uint32_t type;
  uint32_t flags;
  uint64_t offset;
  uint64_t virtual_address;
  uint64_t physical_address;
  uint64_t file_size;
  uint64_t memory_size;
  uint64_t alignment;
} iree_hal_task_elf64_program_header_t;

typedef struct iree_hal_task_elf32_dynamic_t {
  int32_t tag;
  uint32_t value;
} iree_hal_task_elf32_dynamic_t;

typedef struct iree_hal_task_elf64_dynamic_t {
  int64_t tag;
  uint64_t value;
} iree_hal_task_elf64_dynamic_t;

enum {
  IREE_HAL_TASK_ELF_CLASS_32 = 1,
  IREE_HAL_TASK_ELF_CLASS_64 = 2,
  IREE_HAL_TASK_ELF_PROGRAM_HEADER_DYNAMIC = 2,
  IREE_HAL_TASK_ELF_PROGRAM_HEADER_INTERPRETER = 3,
  IREE_HAL_TASK_ELF_PROGRAM_HEADER_TLS = 7,
  IREE_HAL_TASK_ELF_DYNAMIC_NULL = 0,
  IREE_HAL_TASK_ELF_DYNAMIC_NEEDED = 1,
  IREE_HAL_TASK_ELF_DYNAMIC_HASH = 4,
  IREE_HAL_TASK_ELF_DYNAMIC_GNU_HASH = 0x6FFFFEF5,
};

static bool iree_hal_task_executable_data_has_range(
    iree_const_byte_span_t executable_data, uint64_t offset, uint64_t length) {
  return offset <= executable_data.data_length &&
         length <= executable_data.data_length - offset;
}

static bool iree_hal_task_elf_dynamic_requires_system_loader_32(
    iree_const_byte_span_t executable_data, uint32_t offset,
    uint32_t file_size) {
  if (!iree_hal_task_executable_data_has_range(executable_data, offset,
                                               file_size)) {
    return false;
  }
  bool has_hash = false;
  bool has_gnu_hash = false;
  const iree_host_size_t count =
      file_size / sizeof(iree_hal_task_elf32_dynamic_t);
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_task_elf32_dynamic_t entry;
    memcpy(&entry, executable_data.data + offset + i * sizeof(entry),
           sizeof(entry));
    if (entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_NULL) break;
    if (entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_NEEDED) return true;
    has_hash |= entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_HASH;
    has_gnu_hash |= entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_GNU_HASH;
  }
  return has_gnu_hash && !has_hash;
}

static bool iree_hal_task_elf_dynamic_requires_system_loader_64(
    iree_const_byte_span_t executable_data, uint64_t offset,
    uint64_t file_size) {
  if (!iree_hal_task_executable_data_has_range(executable_data, offset,
                                               file_size)) {
    return false;
  }
  bool has_hash = false;
  bool has_gnu_hash = false;
  const uint64_t count = file_size / sizeof(iree_hal_task_elf64_dynamic_t);
  for (uint64_t i = 0; i < count; ++i) {
    iree_hal_task_elf64_dynamic_t entry;
    memcpy(&entry, executable_data.data + offset + i * sizeof(entry),
           sizeof(entry));
    if (entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_NULL) break;
    if (entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_NEEDED) return true;
    has_hash |= entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_HASH;
    has_gnu_hash |= entry.tag == IREE_HAL_TASK_ELF_DYNAMIC_GNU_HASH;
  }
  return has_gnu_hash && !has_hash;
}

static bool iree_hal_task_elf_data_requires_system_loader_32(
    iree_const_byte_span_t executable_data) {
  if (executable_data.data_length < sizeof(iree_hal_task_elf32_header_t)) {
    return false;
  }
  iree_hal_task_elf32_header_t header;
  memcpy(&header, executable_data.data, sizeof(header));
  if (header.program_header_entry_size <
      sizeof(iree_hal_task_elf32_program_header_t)) {
    return false;
  }
  for (uint16_t i = 0; i < header.program_header_count; ++i) {
    const uint64_t offset = (uint64_t)header.program_header_offset +
                            (uint64_t)i * header.program_header_entry_size;
    if (!iree_hal_task_executable_data_has_range(
            executable_data, offset,
            sizeof(iree_hal_task_elf32_program_header_t))) {
      return false;
    }
    iree_hal_task_elf32_program_header_t program_header;
    memcpy(&program_header, executable_data.data + offset,
           sizeof(program_header));
    if (program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_INTERPRETER ||
        program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_TLS) {
      return true;
    }
    if (program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_DYNAMIC &&
        iree_hal_task_elf_dynamic_requires_system_loader_32(
            executable_data, program_header.offset, program_header.file_size)) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_task_elf_data_requires_system_loader_64(
    iree_const_byte_span_t executable_data) {
  if (executable_data.data_length < sizeof(iree_hal_task_elf64_header_t)) {
    return false;
  }
  iree_hal_task_elf64_header_t header;
  memcpy(&header, executable_data.data, sizeof(header));
  if (header.program_header_entry_size <
      sizeof(iree_hal_task_elf64_program_header_t)) {
    return false;
  }
  for (uint16_t i = 0; i < header.program_header_count; ++i) {
    const uint64_t offset = header.program_header_offset +
                            (uint64_t)i * header.program_header_entry_size;
    if (!iree_hal_task_executable_data_has_range(
            executable_data, offset,
            sizeof(iree_hal_task_elf64_program_header_t))) {
      return false;
    }
    iree_hal_task_elf64_program_header_t program_header;
    memcpy(&program_header, executable_data.data + offset,
           sizeof(program_header));
    if (program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_INTERPRETER ||
        program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_TLS) {
      return true;
    }
    if (program_header.type == IREE_HAL_TASK_ELF_PROGRAM_HEADER_DYNAMIC &&
        iree_hal_task_elf_dynamic_requires_system_loader_64(
            executable_data, program_header.offset, program_header.file_size)) {
      return true;
    }
  }
  return false;
}

bool iree_hal_task_elf_data_requires_system_loader(
    iree_const_byte_span_t executable_data) {
  if (!iree_hal_elf_data_starts_with_magic(executable_data) ||
      executable_data.data_length < 16) {
    return false;
  }
  switch (executable_data.data[4]) {
    case IREE_HAL_TASK_ELF_CLASS_32:
      return iree_hal_task_elf_data_requires_system_loader_32(executable_data);
    case IREE_HAL_TASK_ELF_CLASS_64:
      return iree_hal_task_elf_data_requires_system_loader_64(executable_data);
    default:
      return false;
  }
}

bool iree_hal_task_executable_data_is_system_library(
    iree_const_byte_span_t executable_data) {
#if defined(IREE_PLATFORM_WINDOWS)
  static const uint8_t magic[] = {'M', 'Z'};
#elif defined(IREE_PLATFORM_APPLE)
  static const uint8_t magic[][4] = {
      {0xCE, 0xFA, 0xED, 0xFE},
      {0xCF, 0xFA, 0xED, 0xFE},
      {0xCA, 0xFE, 0xBA, 0xBE},
      {0xCA, 0xFE, 0xBA, 0xBF},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(magic); ++i) {
    if (iree_hal_task_executable_data_starts_with(executable_data,
                                                  sizeof(magic[i]), magic[i])) {
      return true;
    }
  }
  return false;
#elif defined(IREE_PLATFORM_WASM)
  static const uint8_t magic[] = {0x00, 0x61, 0x73, 0x6D};
#else
  return iree_hal_elf_data_starts_with_magic(executable_data);
#endif
#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_WASM)
  return iree_hal_task_executable_data_starts_with(executable_data,
                                                   sizeof(magic), magic);
#endif
}
