// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/image/hsaco_metadata.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// MessagePack reader
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_msgpack_reader_t {
  // First unread byte within the immutable payload.
  const uint8_t* current;
  // One-past the final byte of the payload.
  const uint8_t* end;
} iree_hal_amd_gpu_msgpack_reader_t;

static iree_host_size_t iree_hal_amd_gpu_msgpack_remaining(
    const iree_hal_amd_gpu_msgpack_reader_t* reader) {
  return (iree_host_size_t)(reader->end - reader->current);
}

static iree_status_t iree_hal_amd_gpu_msgpack_require(
    const iree_hal_amd_gpu_msgpack_reader_t* reader, iree_host_size_t length) {
  if (iree_hal_amd_gpu_msgpack_remaining(reader) < length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "truncated AMDGPU MessagePack metadata");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_u8(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint8_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, 1));
  *out_value = *reader->current++;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_be_u16(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint16_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, 2));
  *out_value = ((uint16_t)reader->current[0] << 8) | reader->current[1];
  reader->current += 2;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_be_u32(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, 4));
  *out_value = ((uint32_t)reader->current[0] << 24) |
               ((uint32_t)reader->current[1] << 16) |
               ((uint32_t)reader->current[2] << 8) | reader->current[3];
  reader->current += 4;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_be_u64(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint64_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, 8));
  *out_value = ((uint64_t)reader->current[0] << 56) |
               ((uint64_t)reader->current[1] << 48) |
               ((uint64_t)reader->current[2] << 40) |
               ((uint64_t)reader->current[3] << 32) |
               ((uint64_t)reader->current[4] << 24) |
               ((uint64_t)reader->current[5] << 16) |
               ((uint64_t)reader->current[6] << 8) | reader->current[7];
  reader->current += 8;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_skip_bytes(
    iree_hal_amd_gpu_msgpack_reader_t* reader, iree_host_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, length));
  reader->current += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_count_after_tag(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint8_t tag, uint8_t fix_base,
    uint8_t type16, uint8_t type32, uint32_t* out_count) {
  if ((tag & 0xF0u) == fix_base) {
    *out_count = tag & 0x0Fu;
    return iree_ok_status();
  }
  if (tag == type16) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_be_u16(reader, &value));
    *out_count = value;
    return iree_ok_status();
  }
  if (tag == type32) {
    return iree_hal_amd_gpu_msgpack_read_be_u32(reader, out_count);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected AMDGPU MessagePack container tag 0x%02X",
                          tag);
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_map_count(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t* out_count) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &tag));
  return iree_hal_amd_gpu_msgpack_read_count_after_tag(reader, tag, 0x80, 0xDE,
                                                       0xDF, out_count);
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_array_count(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t* out_count) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &tag));
  if ((tag & 0xF0u) == 0x90u) {
    *out_count = tag & 0x0Fu;
    return iree_ok_status();
  }
  if (tag == 0xDC) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_be_u16(reader, &value));
    *out_count = value;
    return iree_ok_status();
  }
  if (tag == 0xDD) {
    return iree_hal_amd_gpu_msgpack_read_be_u32(reader, out_count);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected AMDGPU MessagePack array tag, got 0x%02X",
                          tag);
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_string_after_tag(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint8_t tag,
    iree_string_view_t* out_value) {
  uint32_t length = 0;
  if ((tag & 0xE0u) == 0xA0u) {
    length = tag & 0x1Fu;
  } else if (tag == 0xD9) {
    uint8_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &value));
    length = value;
  } else if (tag == 0xDA) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_be_u16(reader, &value));
    length = value;
  } else if (tag == 0xDB) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_be_u32(reader, &length));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected AMDGPU MessagePack string tag, got "
                            "0x%02X",
                            tag);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, length));
  *out_value = iree_make_string_view((const char*)reader->current, length);
  reader->current += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_string(
    iree_hal_amd_gpu_msgpack_reader_t* reader, iree_string_view_t* out_value) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &tag));
  return iree_hal_amd_gpu_msgpack_read_string_after_tag(reader, tag, out_value);
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_uint64(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint64_t* out_value) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &tag));
  if (tag <= 0x7F) {
    *out_value = tag;
    return iree_ok_status();
  }
  switch (tag) {
    case 0xCC: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCD: {
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u16(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCE: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u32(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCF:
      return iree_hal_amd_gpu_msgpack_read_be_u64(reader, out_value);
    case 0xD0: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &value));
      if (value & 0x80u) {
        break;
      }
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD1: {
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u16(reader, &value));
      if (value & 0x8000u) {
        break;
      }
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD2: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u32(reader, &value));
      if (value & 0x80000000u) {
        break;
      }
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD3: {
      uint64_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u64(reader, &value));
      if (value & 0x8000000000000000ull) {
        break;
      }
      *out_value = value;
      return iree_ok_status();
    }
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected non-negative AMDGPU MessagePack integer");
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_uint32(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t* out_value) {
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_uint64(reader, &value));
  if (value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata integer exceeds uint32_t");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_skip(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t depth) {
  if (depth > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU MessagePack metadata is nested too deeply");
  }
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &tag));
  if (tag <= 0x7F || tag >= 0xE0 || tag == 0xC0 || tag == 0xC2 || tag == 0xC3) {
    return iree_ok_status();
  }
  if ((tag & 0xE0u) == 0xA0u) {
    return iree_hal_amd_gpu_msgpack_skip_bytes(reader, tag & 0x1Fu);
  }
  if ((tag & 0xF0u) == 0x90u) {
    const uint32_t count = tag & 0x0Fu;
    for (uint32_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
    }
    return iree_ok_status();
  }
  if ((tag & 0xF0u) == 0x80u) {
    const uint32_t count = tag & 0x0Fu;
    for (uint32_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
    }
    return iree_ok_status();
  }

  switch (tag) {
    case 0xC4: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC5: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u16(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC6: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u32(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC7: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC8: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u16(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC9: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u32(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xCA:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 4);
    case 0xCB:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 8);
    case 0xCC:
    case 0xD0:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 1);
    case 0xCD:
    case 0xD1:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 2);
    case 0xCE:
    case 0xD2:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 4);
    case 0xCF:
    case 0xD3:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 8);
    case 0xD4:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 2);
    case 0xD5:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 3);
    case 0xD6:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 5);
    case 0xD7:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 9);
    case 0xD8:
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, 17);
    case 0xD9: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_u8(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDA: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u16(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDB: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_be_u32(reader, &length));
      return iree_hal_amd_gpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDC:
    case 0xDD: {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_count_after_tag(
          reader, tag, 0x90, 0xDC, 0xDD, &count));
      for (uint32_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
      }
      return iree_ok_status();
    }
    case 0xDE:
    case 0xDF: {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_count_after_tag(
          reader, tag, 0x80, 0xDE, 0xDF, &count));
      for (uint32_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, depth + 1));
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU MessagePack tag 0x%02X", tag);
  }
}

//===----------------------------------------------------------------------===//
// Metadata scalar fields
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amd_gpu_msgpack_read_bool(
    iree_hal_amd_gpu_msgpack_reader_t* reader, bool* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_require(reader, 1));
  if (*reader->current == 0xC2 || *reader->current == 0xC3) {
    *out_value = *reader->current++ == 0xC3;
    return iree_ok_status();
  }
  // LLVM emits integer 0/1 for some Boolean metadata, including WGP mode.
  // Preserve that producer contract alongside MessagePack's Boolean tags.
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_uint32(reader, &value));
  if (value > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU Boolean integer must be zero or one");
  }
  *out_value = value != 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_msgpack_read_vector(
    iree_hal_amd_gpu_msgpack_reader_t* reader, uint32_t length,
    uint32_t* out_values) {
  uint32_t count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_array_count(reader, &count));
  if (count != length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata vector requires %u elements",
                            length);
  }
  for (uint32_t i = 0; i < length; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_msgpack_read_uint32(reader, &out_values[i]));
  }
  return iree_ok_status();
}

// Dense argument kinds deliberately match this spelling table. The normalized
// enum is consumed by dispatch planning; only this frontend reads ABI strings.
static const iree_string_view_t iree_hal_amd_gpu_metadata_argument_names[] = {
    IREE_SVL(""),
    IREE_SVL("by_value"),
    IREE_SVL("global_buffer"),
    IREE_SVL("dynamic_shared_pointer"),
    IREE_SVL("image"),
    IREE_SVL("sampler"),
    IREE_SVL("pipe"),
    IREE_SVL("queue"),
    IREE_SVL("hidden_none"),
    IREE_SVL("hidden_global_offset_x"),
    IREE_SVL("hidden_global_offset_y"),
    IREE_SVL("hidden_global_offset_z"),
    IREE_SVL("hidden_printf_buffer"),
    IREE_SVL("hidden_hostcall_buffer"),
    IREE_SVL("hidden_default_queue"),
    IREE_SVL("hidden_completion_action"),
    IREE_SVL("hidden_multigrid_sync_arg"),
    IREE_SVL("hidden_block_count_x"),
    IREE_SVL("hidden_block_count_y"),
    IREE_SVL("hidden_block_count_z"),
    IREE_SVL("hidden_group_size_x"),
    IREE_SVL("hidden_group_size_y"),
    IREE_SVL("hidden_group_size_z"),
    IREE_SVL("hidden_remainder_x"),
    IREE_SVL("hidden_remainder_y"),
    IREE_SVL("hidden_remainder_z"),
    IREE_SVL("hidden_grid_dims"),
    IREE_SVL("hidden_heap_v1"),
    IREE_SVL("hidden_dynamic_lds_size"),
    IREE_SVL("hidden_private_base"),
    IREE_SVL("hidden_shared_base"),
    IREE_SVL("hidden_queue_ptr"),
};
static_assert(IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_argument_names) ==
                  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_QUEUE_PTR + 1,
              "argument spelling table must cover every kind");

static uint32_t iree_hal_amd_gpu_metadata_lookup_name(
    iree_string_view_t value, iree_host_size_t count,
    const iree_string_view_t* names) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(value, names[i])) {
      return (uint32_t)i;
    }
  }
  return UINT32_MAX;
}

typedef enum iree_hal_amd_gpu_metadata_field_kind_e {
  IREE_HAL_AMD_GPU_METADATA_FIELD_STRING,
  IREE_HAL_AMD_GPU_METADATA_FIELD_UINT32,
  IREE_HAL_AMD_GPU_METADATA_FIELD_POSITIVE_UINT32,
  IREE_HAL_AMD_GPU_METADATA_FIELD_ALIGNMENT,
  IREE_HAL_AMD_GPU_METADATA_FIELD_BOOL,
  IREE_HAL_AMD_GPU_METADATA_FIELD_UINT_BOOL,
  IREE_HAL_AMD_GPU_METADATA_FIELD_VECTOR2,
  IREE_HAL_AMD_GPU_METADATA_FIELD_VECTOR3,
  IREE_HAL_AMD_GPU_METADATA_FIELD_CLUSTER,
  IREE_HAL_AMD_GPU_METADATA_FIELD_ARGUMENT_KIND,
  IREE_HAL_AMD_GPU_METADATA_FIELD_ADDRESS_SPACE,
  IREE_HAL_AMD_GPU_METADATA_FIELD_ACCESS,
  IREE_HAL_AMD_GPU_METADATA_FIELD_KERNEL_KIND,
} iree_hal_amd_gpu_metadata_field_kind_t;

typedef struct iree_hal_amd_gpu_metadata_field_t {
  // MessagePack key naming this field.
  iree_string_view_t name;
  // Byte offset into the normalized record.
  iree_host_size_t offset;
  // Scalar or fixed-width array decoding rule.
  iree_hal_amd_gpu_metadata_field_kind_t kind;
  // True when omission is an invalid ABI description.
  bool required;
} iree_hal_amd_gpu_metadata_field_t;

#define IREE_HAL_AMD_GPU_METADATA_FIELD(record, key, member, kind, required) \
  {IREE_SVL(key), offsetof(record, member),                                  \
   IREE_HAL_AMD_GPU_METADATA_FIELD_##kind, required}

static const iree_hal_amd_gpu_metadata_field_t
    iree_hal_amd_gpu_metadata_argument_fields[] = {
#define FIELD(key, member, kind, required)                                   \
  IREE_HAL_AMD_GPU_METADATA_FIELD(iree_hal_amd_gpu_metadata_argument_t, key, \
                                  member, kind, required)
        FIELD(".name", name, STRING, false),
        FIELD(".type_name", type_name, STRING, false),
        FIELD(".value_type", value_type, STRING, false),
        FIELD(".offset", offset, UINT32, true),
        FIELD(".size", size, UINT32, true),
        FIELD(".align", alignment, UINT32, false),
        FIELD(".value_kind", kind, ARGUMENT_KIND, true),
        FIELD(".pointee_align", pointee_alignment, ALIGNMENT, false),
        FIELD(".address_space", address_space, ADDRESS_SPACE, false),
        FIELD(".access", access, ACCESS, false),
        FIELD(".actual_access", actual_access, ACCESS, false),
        FIELD(".is_const", qualifiers.is_const, BOOL, false),
        FIELD(".is_restrict", qualifiers.is_restrict, BOOL, false),
        FIELD(".is_volatile", qualifiers.is_volatile, BOOL, false),
        FIELD(".is_pipe", qualifiers.is_pipe, BOOL, false),
#undef FIELD
};

static const iree_hal_amd_gpu_metadata_field_t
    iree_hal_amd_gpu_metadata_kernel_fields[] = {
#define FIELD(key, member, kind, required)                                 \
  IREE_HAL_AMD_GPU_METADATA_FIELD(iree_hal_amd_gpu_metadata_kernel_t, key, \
                                  member, kind, required)
        FIELD(".name", name, STRING, true),
        FIELD(".symbol", symbol_name, STRING, true),
        FIELD(".language", source.language, STRING, false),
        FIELD(".language_version", source.language_version, VECTOR2, false),
        FIELD(".vec_type_hint", source.vector_type_hint, STRING, false),
        FIELD(".device_enqueue_symbol", device_enqueue_symbol, STRING, false),
        FIELD(".kind", kind, KERNEL_KIND, false),
        FIELD(".kernarg_segment_size", kernarg_segment_size, UINT32, true),
        FIELD(".kernarg_segment_align", kernarg_segment_alignment, ALIGNMENT,
              true),
        FIELD(".group_segment_fixed_size", group_segment_fixed_size, UINT32,
              true),
        FIELD(".private_segment_fixed_size", private_segment_fixed_size, UINT32,
              true),
        FIELD(".wavefront_size", wavefront_size, ALIGNMENT, true),
        FIELD(".sgpr_count", sgpr_count, UINT32, true),
        FIELD(".vgpr_count", vgpr_count, UINT32, true),
        FIELD(".agpr_count", agpr_count, UINT32, false),
        FIELD(".sgpr_spill_count", sgpr_spill_count, UINT32, false),
        FIELD(".vgpr_spill_count", vgpr_spill_count, UINT32, false),
        FIELD(".max_flat_workgroup_size", max_flat_workgroup_size,
              POSITIVE_UINT32, true),
        FIELD(".reqd_workgroup_size", required_workgroup_size, VECTOR3, false),
        FIELD(".workgroup_size_hint", workgroup_size_hint, VECTOR3, false),
        FIELD(".max_num_work_groups_x", max_workgroup_count[0], POSITIVE_UINT32,
              false),
        FIELD(".max_num_work_groups_y", max_workgroup_count[1], POSITIVE_UINT32,
              false),
        FIELD(".max_num_work_groups_z", max_workgroup_count[2], POSITIVE_UINT32,
              false),
        FIELD(".uniform_work_group_size", uniform_workgroup_size, UINT_BOOL,
              false),
        FIELD(".uses_dynamic_stack", uses_dynamic_stack, BOOL, false),
        FIELD(".workgroup_processor_mode", workgroup_processor_mode, BOOL,
              false),
        FIELD(".cluster_dims", workgroup_cluster_size, CLUSTER, false),
        FIELD(".gfx1250_revision", gfx1250_revision, STRING, false),
#undef FIELD
};
#undef IREE_HAL_AMD_GPU_METADATA_FIELD

static iree_status_t iree_hal_amd_gpu_metadata_read_field(
    iree_hal_amd_gpu_msgpack_reader_t* reader,
    const iree_hal_amd_gpu_metadata_field_t* field, void* record) {
  void* target = (uint8_t*)record + field->offset;
  switch (field->kind) {
    case IREE_HAL_AMD_GPU_METADATA_FIELD_STRING:
      return iree_hal_amd_gpu_msgpack_read_string(reader, target);
    case IREE_HAL_AMD_GPU_METADATA_FIELD_UINT32:
      return iree_hal_amd_gpu_msgpack_read_uint32(reader, target);
    case IREE_HAL_AMD_GPU_METADATA_FIELD_POSITIVE_UINT32:
    case IREE_HAL_AMD_GPU_METADATA_FIELD_ALIGNMENT: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_uint32(reader, &value));
      if (value == 0 ||
          (field->kind == IREE_HAL_AMD_GPU_METADATA_FIELD_ALIGNMENT &&
           !iree_host_size_is_power_of_two(value))) {
        return iree_make_status(
            field->kind == IREE_HAL_AMD_GPU_METADATA_FIELD_ALIGNMENT
                ? IREE_STATUS_INVALID_ARGUMENT
                : IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU metadata `%.*s` requires %s", (int)field->name.size,
            field->name.data,
            field->kind == IREE_HAL_AMD_GPU_METADATA_FIELD_ALIGNMENT
                ? "a power-of-two alignment"
                : "a positive value");
      }
      *(uint32_t*)target = value;
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_BOOL:
      return iree_hal_amd_gpu_msgpack_read_bool(reader, target);
    case IREE_HAL_AMD_GPU_METADATA_FIELD_UINT_BOOL: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_uint32(reader, &value));
      if (value > 1) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU boolean integer must be zero or one");
      }
      *(bool*)target = value != 0;
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_VECTOR2:
      return iree_hal_amd_gpu_msgpack_read_vector(reader, 2, target);
    case IREE_HAL_AMD_GPU_METADATA_FIELD_VECTOR3:
      return iree_hal_amd_gpu_msgpack_read_vector(reader, 3, target);
    case IREE_HAL_AMD_GPU_METADATA_FIELD_CLUSTER: {
      uint32_t dimensions[3] = {0};
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_vector(reader, 3, dimensions));
      for (iree_host_size_t i = 0; i < 3; ++i) {
        if (dimensions[i] == 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "AMDGPU cluster dimension must be positive");
        }
        ((uint32_t*)target)[i] = dimensions[i];
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_ARGUMENT_KIND: {
      iree_hal_amd_gpu_metadata_argument_t* argument = record;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_string(reader, &argument->value_kind));
      uint32_t kind = iree_hal_amd_gpu_metadata_lookup_name(
          argument->value_kind,
          IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_argument_names),
          iree_hal_amd_gpu_metadata_argument_names);
      argument->kind = kind == UINT32_MAX
                           ? IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_UNKNOWN
                           : (iree_hal_amd_gpu_metadata_argument_kind_t)kind;
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_ADDRESS_SPACE: {
      static const iree_string_view_t names[] = {
          IREE_SVL(""),         IREE_SVL("private"), IREE_SVL("global"),
          IREE_SVL("constant"), IREE_SVL("local"),   IREE_SVL("generic"),
          IREE_SVL("region"),
      };
      iree_string_view_t value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_string(reader, &value));
      uint32_t ordinal = iree_hal_amd_gpu_metadata_lookup_name(
          value, IREE_ARRAYSIZE(names), names);
      if (ordinal == UINT32_MAX || ordinal == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown AMDGPU address space `%.*s`",
                                (int)value.size, value.data);
      }
      *(iree_hal_amd_gpu_metadata_address_space_t*)target =
          (iree_hal_amd_gpu_metadata_address_space_t)ordinal;
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_ACCESS: {
      static const iree_string_view_t names[] = {
          IREE_SVL(""),
          IREE_SVL("read_only"),
          IREE_SVL("write_only"),
          IREE_SVL("read_write"),
      };
      iree_string_view_t value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_string(reader, &value));
      uint32_t ordinal = iree_hal_amd_gpu_metadata_lookup_name(
          value, IREE_ARRAYSIZE(names), names);
      if (ordinal == UINT32_MAX || ordinal == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown AMDGPU access qualifier `%.*s`",
                                (int)value.size, value.data);
      }
      *(iree_hal_amd_gpu_metadata_access_t*)target =
          (iree_hal_amd_gpu_metadata_access_t)ordinal;
      return iree_ok_status();
    }
    case IREE_HAL_AMD_GPU_METADATA_FIELD_KERNEL_KIND: {
      static const iree_string_view_t names[] = {
          IREE_SVL("normal"), IREE_SVL("init"), IREE_SVL("fini")};
      iree_string_view_t value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_string(reader, &value));
      uint32_t ordinal = iree_hal_amd_gpu_metadata_lookup_name(
          value, IREE_ARRAYSIZE(names), names);
      if (ordinal == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported AMDGPU kernel kind `%.*s`",
                                (int)value.size, value.data);
      }
      *(iree_hal_amd_gpu_metadata_kernel_kind_t*)target =
          (iree_hal_amd_gpu_metadata_kernel_kind_t)ordinal;
      return iree_ok_status();
    }
    default:
      IREE_CHECK_UNREACHABLE("unknown metadata scalar field kind");
  }
}

static iree_status_t iree_hal_amd_gpu_metadata_mark_field(
    iree_string_view_t name, uint32_t ordinal, uint64_t* inout_seen) {
  const uint64_t bit = UINT64_C(1) << ordinal;
  if (*inout_seen & bit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata repeats field `%.*s`",
                            (int)name.size, name.data);
  }
  *inout_seen |= bit;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_parse_field(
    iree_hal_amd_gpu_msgpack_reader_t* reader, iree_string_view_t name,
    iree_host_size_t field_count,
    const iree_hal_amd_gpu_metadata_field_t* fields, uint64_t* inout_seen,
    void* record) {
  // The LLVM producer and ABI documentation use different workgroup-limit
  // spellings. Both spellings name the same field, including duplicate checks.
  static const iree_string_view_t aliases[][2] = {
      {IREE_SVL(".alignment"), IREE_SVL(".align")},
      {IREE_SVL(".max_num_workgroups_x"), IREE_SVL(".max_num_work_groups_x")},
      {IREE_SVL(".max_num_workgroups_y"), IREE_SVL(".max_num_work_groups_y")},
      {IREE_SVL(".max_num_workgroups_z"), IREE_SVL(".max_num_work_groups_z")},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(aliases); ++i) {
    if (iree_string_view_equal(name, aliases[i][0])) {
      name = aliases[i][1];
      break;
    }
  }
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (iree_string_view_equal(name, fields[i].name)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_metadata_mark_field(name, (uint32_t)i, inout_seen));
      return iree_hal_amd_gpu_metadata_read_field(reader, &fields[i], record);
    }
  }
  return iree_hal_amd_gpu_msgpack_skip(reader, 0);
}

static iree_status_t iree_hal_amd_gpu_metadata_check_required_fields(
    uint64_t seen, iree_host_size_t count,
    const iree_hal_amd_gpu_metadata_field_t* fields) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (fields[i].required && !(seen & (UINT64_C(1) << i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU metadata missing required field `%.*s`",
                              (int)fields[i].name.size, fields[i].name.data);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Printf record decoding and storage measurement
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_metadata_counts_t {
  // Number of kernel records across all payloads.
  iree_host_size_t kernels;
  // Number of kernel argument records across all payloads.
  iree_host_size_t arguments;
  // Number of printf records across all payloads.
  iree_host_size_t printfs;
  // Number of argument sizes referenced by printf records.
  iree_host_size_t printf_arguments;
} iree_hal_amd_gpu_metadata_counts_t;

static iree_status_t iree_hal_amd_gpu_metadata_add_count(
    iree_host_size_t increment, iree_host_size_t* inout_count) {
  if (!iree_host_size_checked_add(*inout_count, increment, inout_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata record count overflow");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_printf_read_integer(
    iree_string_view_t* inout_value, uint32_t* out_integer) {
  iree_string_view_t field = iree_string_view_empty();
  iree_string_view_t remaining = iree_string_view_empty();
  if (iree_string_view_split(*inout_value, ':', &field, &remaining) < 0 ||
      field.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU printf record missing an integer field");
  }
  uint32_t value = 0;
  for (iree_host_size_t i = 0; i < field.size; ++i) {
    if (field.data[i] < '0' || field.data[i] > '9' ||
        value > (UINT32_MAX - (uint32_t)(field.data[i] - '0')) / 10) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid AMDGPU printf integer");
    }
    value = value * 10 + (uint32_t)(field.data[i] - '0');
  }
  *out_integer = value;
  *inout_value = remaining;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_parse_printf(
    iree_string_view_t value, iree_hal_amd_gpu_metadata_printf_t* out_record,
    uint32_t* out_argument_sizes) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_metadata_printf_read_integer(&value, &out_record->id));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_printf_read_integer(
      &value, &out_record->argument_count));
  for (uint32_t i = 0; i < out_record->argument_count; ++i) {
    uint32_t size = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_metadata_printf_read_integer(&value, &size));
    if (size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU printf argument has zero size");
    }
    if (out_argument_sizes) {
      out_argument_sizes[i] = size;
    }
  }
  out_record->format = value;
  out_record->argument_sizes = out_argument_sizes;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_count_kernel(
    iree_hal_amd_gpu_msgpack_reader_t* reader,
    iree_hal_amd_gpu_metadata_counts_t* counts) {
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_map_count(reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_string(reader, &key));
    if (iree_string_view_equal(key, IREE_SV(".args"))) {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(reader, &count));
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_metadata_add_count(count, &counts->arguments));
      for (uint32_t j = 0; j < count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, 0));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(reader, 0));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_count_payload(
    iree_const_byte_span_t payload,
    iree_hal_amd_gpu_metadata_counts_t* counts) {
  iree_hal_amd_gpu_msgpack_reader_t reader = {
      .current = payload.data,
      .end = payload.data + payload.data_length,
  };
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_map_count(&reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_string(&reader, &key));
    if (iree_string_view_equal(key, IREE_SV("amdhsa.kernels"))) {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(&reader, &count));
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_metadata_add_count(count, &counts->kernels));
      for (uint32_t j = 0; j < count; ++j) {
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_gpu_metadata_count_kernel(&reader, counts));
      }
    } else if (iree_string_view_equal(key, IREE_SV("amdhsa.printf"))) {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(&reader, &count));
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_metadata_add_count(count, &counts->printfs));
      for (uint32_t j = 0; j < count; ++j) {
        iree_string_view_t value = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_gpu_msgpack_read_string(&reader, &value));
        iree_hal_amd_gpu_metadata_printf_t record = {0};
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_gpu_metadata_parse_printf(value, &record, NULL));
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_add_count(
            record.argument_count, &counts->printf_arguments));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(&reader, 0));
    }
  }
  if (reader.current != reader.end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata has trailing MessagePack bytes");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Kernel and module admission
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_metadata_parse_state_t {
  // Records owned by the module being constructed.
  iree_hal_amd_gpu_metadata_t* metadata;
  // Next output index in each array sized by the measurement pass.
  iree_hal_amd_gpu_metadata_counts_t next;
  // Printf argument sizes within the same owned record allocation.
  uint32_t* printf_argument_sizes;
} iree_hal_amd_gpu_metadata_parse_state_t;

static iree_status_t iree_hal_amd_gpu_metadata_parse_argument(
    iree_hal_amd_gpu_msgpack_reader_t* reader,
    iree_hal_amd_gpu_metadata_argument_t* argument) {
  uint64_t seen = 0;
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_map_count(reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_string(reader, &key));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_parse_field(
        reader, key, IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_argument_fields),
        iree_hal_amd_gpu_metadata_argument_fields, &seen, argument));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_check_required_fields(
      seen, IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_argument_fields),
      iree_hal_amd_gpu_metadata_argument_fields));
  if (argument->alignment != 0 &&
      (!iree_host_size_is_power_of_two(argument->alignment) ||
       !iree_host_size_has_alignment(argument->offset, argument->alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU argument storage alignment or offset is invalid");
  }
  argument->effective_access = argument->actual_access;
  if (argument->effective_access ==
      IREE_HAL_AMD_GPU_METADATA_ACCESS_UNSPECIFIED) {
    argument->effective_access = argument->access;
    if (argument->effective_access ==
        IREE_HAL_AMD_GPU_METADATA_ACCESS_UNSPECIFIED) {
      argument->effective_access =
          argument->qualifiers.is_const
              ? IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_ONLY
              : IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_WRITE;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_parse_kernel(
    iree_hal_amd_gpu_msgpack_reader_t* reader,
    iree_hal_amd_gpu_metadata_parse_state_t* state,
    iree_hal_amd_gpu_metadata_kernel_t* kernel) {
  uint64_t seen = 0;
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_map_count(reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_string(reader, &key));
    if (iree_string_view_equal(key, IREE_SV(".args"))) {
      // Reserve one bit outside the scalar table for the argument array.
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_metadata_mark_field(key, 63, &seen));
      kernel->has_argument_metadata = true;
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(reader, &count));
      kernel->argument_count = count;
      kernel->arguments =
          count ? &state->metadata->arguments[state->next.arguments] : NULL;
      for (uint32_t j = 0; j < count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_parse_argument(
            reader, &state->metadata->arguments[state->next.arguments++]));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_parse_field(
          reader, key, IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_kernel_fields),
          iree_hal_amd_gpu_metadata_kernel_fields, &seen, kernel));
      if (iree_string_view_equal(key, IREE_SV(".workgroup_processor_mode"))) {
        kernel->has_workgroup_processor_mode = true;
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_check_required_fields(
      seen, IREE_ARRAYSIZE(iree_hal_amd_gpu_metadata_kernel_fields),
      iree_hal_amd_gpu_metadata_kernel_fields));
  if (iree_string_view_is_empty(kernel->symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel descriptor symbol is empty");
  }
  uint32_t previous_end = 0;
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    const iree_hal_amd_gpu_metadata_argument_t* argument =
        &kernel->arguments[i];
    if (argument->offset > kernel->kernarg_segment_size ||
        argument->size > kernel->kernarg_segment_size - argument->offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU kernel `%.*s` argument %" PRIhsz
                              " exceeds kernarg segment size",
                              (int)kernel->symbol_name.size,
                              kernel->symbol_name.data, i);
    }
    if (argument->offset < previous_end) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel arguments overlap or are not sorted by offset");
    }
    previous_end = argument->offset + argument->size;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_add_count(
        argument->name.size, &kernel->argument_name_storage_size));
  }
  kernel->has_required_workgroup_size =
      kernel->required_workgroup_size[0] != 0 ||
      kernel->required_workgroup_size[1] != 0 ||
      kernel->required_workgroup_size[2] != 0;
  if (kernel->has_required_workgroup_size) {
    uint32_t product = 1;
    for (iree_host_size_t i = 0; i < 3; ++i) {
      const uint32_t dimension = kernel->required_workgroup_size[i];
      if (dimension == 0 ||
          product > kernel->max_flat_workgroup_size / dimension) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU required workgroup shape is zero or "
                                "exceeds the flat limit");
      }
      product *= dimension;
    }
  }
  kernel->has_workgroup_cluster_size = kernel->workgroup_cluster_size[0] > 1 ||
                                       kernel->workgroup_cluster_size[1] > 1 ||
                                       kernel->workgroup_cluster_size[2] > 1;
  kernel->reflection_name =
      iree_string_view_is_empty(kernel->name)
          ? iree_string_view_strip_suffix(kernel->symbol_name, IREE_SV(".kd"))
          : kernel->name;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_add_count(
      kernel->reflection_name.size,
      &state->metadata->reflection_name_storage_size));
  return iree_hal_amd_gpu_metadata_add_count(
      kernel->argument_name_storage_size,
      &state->metadata->argument_name_storage_size);
}

static iree_status_t iree_hal_amd_gpu_metadata_parse_payload(
    iree_const_byte_span_t payload,
    iree_hal_amd_gpu_metadata_parse_state_t* state) {
  iree_hal_amd_gpu_metadata_t* metadata = state->metadata;
  iree_hal_amd_gpu_msgpack_reader_t reader = {
      .current = payload.data,
      .end = payload.data + payload.data_length,
  };
  uint64_t seen = 0;
  uint32_t version[2] = {0};
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_msgpack_read_map_count(&reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_read_string(&reader, &key));
    if (iree_string_view_equal(key, IREE_SV("amdhsa.version"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_mark_field(key, 0, &seen));
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_vector(&reader, 2, version));
      if (version[0] != 1 || version[1] > 2) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported AMDHSA metadata version %u.%u",
                                version[0], version[1]);
      }
      if (metadata->version[0] && (metadata->version[0] != version[0] ||
                                   metadata->version[1] != version[1])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU metadata notes have conflicting versions");
      }
      memcpy(metadata->version, version, sizeof(version));
    } else if (iree_string_view_equal(key, IREE_SV("amdhsa.target"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_mark_field(key, 1, &seen));
      iree_string_view_t target = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_string(&reader, &target));
      if (iree_string_view_is_empty(target)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU metadata target is empty");
      }
      if (!iree_string_view_is_empty(metadata->target) &&
          !iree_string_view_equal(metadata->target, target)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU metadata notes have conflicting `amdhsa.target` values");
      }
      metadata->target = target;
    } else if (iree_string_view_equal(key, IREE_SV("amdhsa.kernels"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_mark_field(key, 2, &seen));
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(&reader, &count));
      for (uint32_t j = 0; j < count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_parse_kernel(
            &reader, state, &metadata->kernels[state->next.kernels++]));
      }
    } else if (iree_string_view_equal(key, IREE_SV("amdhsa.printf"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_mark_field(key, 3, &seen));
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_msgpack_read_array_count(&reader, &count));
      for (uint32_t j = 0; j < count; ++j) {
        iree_string_view_t value = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_gpu_msgpack_read_string(&reader, &value));
        iree_hal_amd_gpu_metadata_printf_t* record =
            &metadata->printfs[state->next.printfs++];
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_metadata_parse_printf(
            value, record,
            state->printf_argument_sizes
                ? state->printf_argument_sizes + state->next.printf_arguments
                : NULL));
        state->next.printf_arguments += record->argument_count;
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_msgpack_skip(&reader, 0));
    }
  }
  if (!(seen & (UINT64_C(1) << 0)) || !(seen & (UINT64_C(1) << 2))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata requires `amdhsa.version` and `amdhsa.kernels`");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_metadata_allocate_storage(
    const iree_hal_amd_gpu_metadata_counts_t* counts,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_metadata_t* metadata,
    uint32_t** out_printf_argument_sizes) {
  const iree_host_size_t sizes[] = {
      sizeof(iree_hal_amd_gpu_metadata_kernel_t),
      sizeof(iree_hal_amd_gpu_metadata_argument_t),
      sizeof(iree_hal_amd_gpu_metadata_printf_t),
      sizeof(uint32_t),
  };
  const iree_host_size_t alignments[] = {
      iree_alignof(iree_hal_amd_gpu_metadata_kernel_t),
      iree_alignof(iree_hal_amd_gpu_metadata_argument_t),
      iree_alignof(iree_hal_amd_gpu_metadata_printf_t),
      iree_alignof(uint32_t),
  };
  const iree_host_size_t lengths[] = {counts->kernels, counts->arguments,
                                      counts->printfs,
                                      counts->printf_arguments};
  iree_host_size_t offsets[IREE_ARRAYSIZE(sizes)] = {0};
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(sizes); ++i) {
    iree_host_size_t byte_length = 0;
    if (!iree_host_size_checked_align(total_size, alignments[i], &offsets[i]) ||
        !iree_host_size_checked_mul(lengths[i], sizes[i], &byte_length) ||
        !iree_host_size_checked_add(offsets[i], byte_length, &total_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU metadata storage size overflow");
    }
  }
  metadata->host_allocator = host_allocator;
  if (total_size == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, &metadata->storage));
  uint8_t* storage = metadata->storage;
  metadata->kernel_count = counts->kernels;
  metadata->kernels =
      counts->kernels
          ? (iree_hal_amd_gpu_metadata_kernel_t*)(storage + offsets[0])
          : NULL;
  metadata->argument_count = counts->arguments;
  metadata->arguments =
      counts->arguments
          ? (iree_hal_amd_gpu_metadata_argument_t*)(storage + offsets[1])
          : NULL;
  metadata->printf_count = counts->printfs;
  metadata->printfs =
      counts->printfs
          ? (iree_hal_amd_gpu_metadata_printf_t*)(storage + offsets[2])
          : NULL;
  *out_printf_argument_sizes =
      counts->printf_arguments ? (uint32_t*)(storage + offsets[3]) : NULL;
  return iree_ok_status();
}

static int iree_hal_amd_gpu_metadata_compare_printf_ids(const void* lhs,
                                                        const void* rhs) {
  uint32_t lhs_id = ((const iree_hal_amd_gpu_metadata_printf_t*)lhs)->id;
  uint32_t rhs_id = ((const iree_hal_amd_gpu_metadata_printf_t*)rhs)->id;
  return lhs_id < rhs_id ? -1 : lhs_id > rhs_id ? 1 : 0;
}

iree_status_t iree_hal_amd_gpu_hsaco_metadata_initialize(
    iree_host_size_t payload_count, const iree_const_byte_span_t* payloads,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_metadata_t* out_metadata) {
  memset(out_metadata, 0, sizeof(*out_metadata));
  if (payload_count == 0 || !payloads) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata requires at least one payload");
  }
  iree_hal_amd_gpu_metadata_counts_t counts = {0};
  for (iree_host_size_t i = 0; i < payload_count; ++i) {
    if (!payloads[i].data || payloads[i].data_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU metadata payload is empty");
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_metadata_count_payload(payloads[i], &counts));
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_metadata_t metadata = {0};
  iree_hal_amd_gpu_metadata_parse_state_t state = {.metadata = &metadata};
  iree_status_t status = iree_hal_amd_gpu_metadata_allocate_storage(
      &counts, host_allocator, &metadata, &state.printf_argument_sizes);
  for (iree_host_size_t i = 0; i < payload_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_metadata_parse_payload(payloads[i], &state);
  }
  if (iree_status_is_ok(status) && metadata.printf_count > 1) {
    qsort(metadata.printfs, metadata.printf_count, sizeof(metadata.printfs[0]),
          iree_hal_amd_gpu_metadata_compare_printf_ids);
    for (iree_host_size_t i = 1;
         i < metadata.printf_count && iree_status_is_ok(status); ++i) {
      if (metadata.printfs[i - 1].id == metadata.printfs[i].id) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "AMDGPU metadata repeats printf ID %u",
                                  metadata.printfs[i].id);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_metadata = metadata;
  } else {
    iree_hal_amd_gpu_metadata_deinitialize(&metadata);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
