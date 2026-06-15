// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/metadata.h"

#include <inttypes.h>

//===----------------------------------------------------------------------===//
// Validation and shared fields
//===----------------------------------------------------------------------===//

#define LOOM_AMDGPU_METADATA_VERSION_MAJOR 1u
#define LOOM_AMDGPU_METADATA_VERSION_MINOR 2u
#define LOOM_AMDGPU_METADATA_ELF_NOTE_NT_AMDGPU_METADATA 32u

static iree_status_t loom_amdgpu_metadata_validate_string(
    iree_string_view_t value, iree_string_view_t field_name, bool required) {
  if (iree_string_view_is_empty(value)) {
    if (!required) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata field `%.*s` is required",
                            (int)field_name.size, field_name.data);
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char character = (unsigned char)value.data[i];
    if (character <= ' ' || character == '\'' || character == '\\') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU metadata field `%.*s` contains an unsupported character",
          (int)field_name.size, field_name.data);
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_metadata_argument_value_kind(
    loom_amdgpu_metadata_argument_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE:
      return IREE_SV("by_value");
    case LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER:
      return IREE_SV("global_buffer");
    case LOOM_AMDGPU_METADATA_ARGUMENT_HIDDEN_NONE:
      return IREE_SV("hidden_none");
  }
  return iree_string_view_empty();
}

static iree_status_t loom_amdgpu_metadata_validate_argument(
    const loom_amdgpu_metadata_argument_t* argument) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      argument->name, IREE_SV(".args[].name"), /*required=*/false));
  iree_string_view_t value_kind =
      loom_amdgpu_metadata_argument_value_kind(argument->kind);
  if (iree_string_view_is_empty(value_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU metadata argument kind %d",
                            (int)argument->kind);
  }
  if (argument->alignment != 0 &&
      !iree_host_size_is_power_of_two(argument->alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata argument alignment must be a power of two");
  }
  if (argument->alignment != 0 &&
      !iree_host_size_has_alignment(argument->offset, argument->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata argument offset %" PRIu32
                            " is not aligned to %" PRIu32,
                            argument->offset, argument->alignment);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      argument->address_space, IREE_SV(".args[].address_space"),
      /*required=*/false));
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      argument->access, IREE_SV(".args[].access"), /*required=*/false));
  return loom_amdgpu_metadata_validate_string(argument->actual_access,
                                              IREE_SV(".args[].actual_access"),
                                              /*required=*/false);
}

static iree_status_t loom_amdgpu_metadata_validate_kernel(
    const loom_amdgpu_metadata_kernel_t* kernel) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      kernel->name, IREE_SV(".name"), /*required=*/true));
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      kernel->descriptor_symbol, IREE_SV(".symbol"), /*required=*/true));
  if (kernel->kernarg_segment_alignment == 0 ||
      !iree_host_size_is_power_of_two(kernel->kernarg_segment_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata kernarg segment alignment must be a power of two");
  }
  if (kernel->wavefront_size != 32 && kernel->wavefront_size != 64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata wavefront size must be either 32 or 64");
  }
  iree_host_size_t previous_argument_end = 0;
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    const loom_amdgpu_metadata_argument_t* argument = &kernel->arguments[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_argument(argument));
    iree_host_size_t argument_end = 0;
    if (!iree_host_size_checked_add(argument->offset, argument->size,
                                    &argument_end) ||
        argument_end > kernel->kernarg_segment_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU metadata argument %" PRIhsz
                              " exceeds kernarg segment size %" PRIu32,
                              i, kernel->kernarg_segment_size);
    }
    if (i > 0 && argument->offset < previous_argument_end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU metadata argument %" PRIhsz
                              " overlaps a previous argument",
                              i);
    }
    previous_argument_end = argument_end;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_metadata_validate(
    const loom_amdgpu_code_object_metadata_t* metadata) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate_string(
      metadata->target, IREE_SV("amdhsa.target"), /*required=*/true));
  if (metadata->kernel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata requires at least one kernel");
  }
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_metadata_validate_kernel(&metadata->kernels[i]));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Assembly metadata block writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_amdgpu_metadata_append_quoted_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  return iree_string_builder_append_format(builder, "'%.*s'", (int)value.size,
                                           value.data);
}

static iree_status_t loom_amdgpu_metadata_append_argument_assembly(
    const loom_amdgpu_metadata_argument_t* argument,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "        - "));
  if (!iree_string_view_is_empty(argument->name)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ".name: "));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_metadata_append_quoted_string(builder, argument->name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "          "));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(
          builder,
          ".offset: %" PRIu32 "\n"
          "          .size: %" PRIu32 "\n"
          "          .value_kind: %.*s\n",
          argument->offset, argument->size,
          (int)loom_amdgpu_metadata_argument_value_kind(argument->kind).size,
          loom_amdgpu_metadata_argument_value_kind(argument->kind).data));
  if (argument->alignment != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "          .align: %" PRIu32 "\n", argument->alignment));
  }
  if (!iree_string_view_is_empty(argument->address_space)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "          .address_space: %.*s\n",
        (int)argument->address_space.size, argument->address_space.data));
  }
  if (!iree_string_view_is_empty(argument->access)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "          .access: %.*s\n", (int)argument->access.size,
        argument->access.data));
  }
  if (!iree_string_view_is_empty(argument->actual_access)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "          .actual_access: %.*s\n",
        (int)argument->actual_access.size, argument->actual_access.data));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_metadata_append_kernel_assembly(
    const loom_amdgpu_metadata_kernel_t* kernel,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "    - .name: "));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_quoted_string(builder, kernel->name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n      .symbol: "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_quoted_string(
      builder, kernel->descriptor_symbol));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(
          builder,
          "\n"
          "      .kernarg_segment_size: %" PRIu32 "\n"
          "      .group_segment_fixed_size: %" PRIu32 "\n"
          "      .private_segment_fixed_size: %" PRIu32 "\n"
          "      .kernarg_segment_align: %" PRIu32 "\n"
          "      .wavefront_size: %" PRIu32 "\n"
          "      .sgpr_count: %" PRIu32 "\n"
          "      .vgpr_count: %" PRIu32 "\n"
          "      .max_flat_workgroup_size: %" PRIu32 "\n",
          kernel->kernarg_segment_size, kernel->group_segment_fixed_size,
          kernel->private_segment_fixed_size, kernel->kernarg_segment_alignment,
          kernel->wavefront_size, kernel->sgpr_count, kernel->vgpr_count,
          kernel->max_flat_workgroup_size));
  if (kernel->has_required_workgroup_size) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder,
                                          "      .reqd_workgroup_size:\n"
                                          "        - %" PRIu32 "\n"
                                          "        - %" PRIu32 "\n"
                                          "        - %" PRIu32 "\n",
                                          kernel->required_workgroup_size.x,
                                          kernel->required_workgroup_size.y,
                                          kernel->required_workgroup_size.z));
  }
  if (kernel->argument_count == 0) {
    return iree_string_builder_append_cstring(builder, "      .args: []\n");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "      .args:\n"));
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_argument_assembly(
        &kernel->arguments[i], builder));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_metadata_append_assembly(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate(metadata));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n.amdgpu_metadata\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  amdhsa.version:\n"
      "    - %u\n"
      "    - %u\n"
      "  amdhsa.target: ",
      LOOM_AMDGPU_METADATA_VERSION_MAJOR, LOOM_AMDGPU_METADATA_VERSION_MINOR));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_quoted_string(builder, metadata->target));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n  amdhsa.kernels:\n"));
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_kernel_assembly(
        &metadata->kernels[i], builder));
  }
  return iree_string_builder_append_cstring(builder, ".end_amdgpu_metadata\n");
}

//===----------------------------------------------------------------------===//
// MessagePack writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_amdgpu_msgpack_append_u8(
    iree_string_builder_t* builder, uint8_t value) {
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)&value, 1));
}

static iree_status_t loom_amdgpu_msgpack_append_be_u16(
    iree_string_builder_t* builder, uint16_t value) {
  uint8_t bytes[2] = {
      (uint8_t)(value >> 8),
      (uint8_t)value,
  };
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)bytes, sizeof(bytes)));
}

static iree_status_t loom_amdgpu_msgpack_append_be_u32(
    iree_string_builder_t* builder, uint32_t value) {
  uint8_t bytes[4] = {
      (uint8_t)(value >> 24),
      (uint8_t)(value >> 16),
      (uint8_t)(value >> 8),
      (uint8_t)value,
  };
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)bytes, sizeof(bytes)));
}

static iree_status_t loom_amdgpu_msgpack_append_map(
    iree_string_builder_t* builder, iree_host_size_t count) {
  if (count < 16) {
    return loom_amdgpu_msgpack_append_u8(builder, (uint8_t)(0x80u | count));
  }
  if (count <= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDE));
    return loom_amdgpu_msgpack_append_be_u16(builder, (uint16_t)count);
  }
  if (count <= UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDF));
    return loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)count);
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU MessagePack map is too large");
}

static iree_status_t loom_amdgpu_msgpack_append_array(
    iree_string_builder_t* builder, iree_host_size_t count) {
  if (count < 16) {
    return loom_amdgpu_msgpack_append_u8(builder, (uint8_t)(0x90u | count));
  }
  if (count <= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDC));
    return loom_amdgpu_msgpack_append_be_u16(builder, (uint16_t)count);
  }
  if (count <= UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDD));
    return loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)count);
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU MessagePack array is too large");
}

static iree_status_t loom_amdgpu_msgpack_append_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  if (value.size < 32) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_u8(builder, (uint8_t)(0xA0u | value.size)));
  } else if (value.size <= UINT8_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xD9));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_u8(builder, (uint8_t)value.size));
  } else if (value.size <= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDA));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_be_u16(builder, (uint16_t)value.size));
  } else if (value.size <= UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDB));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)value.size));
  } else {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU MessagePack string is too large");
  }
  return iree_string_builder_append_string(builder, value);
}

static iree_status_t loom_amdgpu_msgpack_append_uint32(
    iree_string_builder_t* builder, uint32_t value) {
  if (value <= 0x7Fu) {
    return loom_amdgpu_msgpack_append_u8(builder, (uint8_t)value);
  }
  if (value <= UINT8_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xCC));
    return loom_amdgpu_msgpack_append_u8(builder, (uint8_t)value);
  }
  if (value <= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xCD));
    return loom_amdgpu_msgpack_append_be_u16(builder, (uint16_t)value);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xCE));
  return loom_amdgpu_msgpack_append_be_u32(builder, value);
}

static iree_status_t loom_amdgpu_msgpack_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t key,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string(builder, key));
  return loom_amdgpu_msgpack_append_string(builder, value);
}

static iree_status_t loom_amdgpu_msgpack_append_uint32_field(
    iree_string_builder_t* builder, iree_string_view_t key, uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string(builder, key));
  return loom_amdgpu_msgpack_append_uint32(builder, value);
}

static iree_status_t loom_amdgpu_metadata_append_argument_msgpack(
    const loom_amdgpu_metadata_argument_t* argument,
    iree_string_builder_t* builder) {
  iree_host_size_t field_count = 3;
  if (!iree_string_view_is_empty(argument->name)) {
    ++field_count;
  }
  if (argument->alignment != 0) {
    ++field_count;
  }
  if (!iree_string_view_is_empty(argument->address_space)) {
    ++field_count;
  }
  if (!iree_string_view_is_empty(argument->access)) {
    ++field_count;
  }
  if (!iree_string_view_is_empty(argument->actual_access)) {
    ++field_count;
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_map(builder, field_count));
  if (!iree_string_view_is_empty(argument->name)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".name"), argument->name));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".offset"), argument->offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".size"), argument->size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV(".value_kind"),
      loom_amdgpu_metadata_argument_value_kind(argument->kind)));
  if (argument->alignment != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
        builder, IREE_SV(".align"), argument->alignment));
  }
  if (!iree_string_view_is_empty(argument->address_space)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".address_space"), argument->address_space));
  }
  if (!iree_string_view_is_empty(argument->access)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".access"), argument->access));
  }
  if (!iree_string_view_is_empty(argument->actual_access)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".actual_access"), argument->actual_access));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_metadata_append_kernel_msgpack(
    const loom_amdgpu_metadata_kernel_t* kernel,
    iree_string_builder_t* builder) {
  const iree_host_size_t field_count =
      11 + (kernel->has_required_workgroup_size ? 1 : 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_map(builder, field_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV(".name"), kernel->name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV(".symbol"), kernel->descriptor_symbol));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".kernarg_segment_size"), kernel->kernarg_segment_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".group_segment_fixed_size"),
      kernel->group_segment_fixed_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".private_segment_fixed_size"),
      kernel->private_segment_fixed_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".kernarg_segment_align"),
      kernel->kernarg_segment_alignment));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".wavefront_size"), kernel->wavefront_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".sgpr_count"), kernel->sgpr_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".vgpr_count"), kernel->vgpr_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".max_flat_workgroup_size"),
      kernel->max_flat_workgroup_size));
  if (kernel->has_required_workgroup_size) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string(
        builder, IREE_SV(".reqd_workgroup_size")));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_array(builder, 3));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->required_workgroup_size.x));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->required_workgroup_size.y));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->required_workgroup_size.z));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV(".args")));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_array(builder, kernel->argument_count));
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_argument_msgpack(
        &kernel->arguments[i], builder));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_metadata_append_msgpack(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_validate(metadata));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_map(builder, 3));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV("amdhsa.version")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_array(builder, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
      builder, LOOM_AMDGPU_METADATA_VERSION_MAJOR));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
      builder, LOOM_AMDGPU_METADATA_VERSION_MINOR));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV("amdhsa.target"), metadata->target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV("amdhsa.kernels")));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_array(builder, metadata->kernel_count));
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_kernel_msgpack(
        &metadata->kernels[i], builder));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ELF note writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_amdgpu_metadata_append_le_u32(
    iree_string_builder_t* builder, uint32_t value) {
  const uint8_t bytes[4] = {
      (uint8_t)value,
      (uint8_t)(value >> 8),
      (uint8_t)(value >> 16),
      (uint8_t)(value >> 24),
  };
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)bytes, sizeof(bytes)));
}

static iree_status_t loom_amdgpu_metadata_append_align4_padding(
    iree_string_builder_t* builder, iree_host_size_t field_length) {
  iree_host_size_t aligned_length = 0;
  if (!iree_host_size_checked_align(field_length, 4, &aligned_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ELF note alignment overflows");
  }
  const iree_host_size_t padding_length = aligned_length - field_length;
  if (padding_length == 0) {
    return iree_ok_status();
  }
  const uint8_t padding[3] = {0};
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)padding, padding_length));
}

iree_status_t loom_amdgpu_metadata_append_elf_note(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder) {
  iree_string_builder_t payload_builder;
  iree_string_builder_initialize(iree_allocator_system(), &payload_builder);
  iree_status_t status =
      loom_amdgpu_metadata_append_msgpack(metadata, &payload_builder);
  iree_string_view_t payload = iree_string_builder_view(&payload_builder);
  if (iree_status_is_ok(status) && payload.size > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU metadata payload is too large for an "
                              "ELF note descriptor");
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t note_name = IREE_SV("AMDGPU");
    const uint32_t note_name_size = (uint32_t)note_name.size + 1u;
    status = loom_amdgpu_metadata_append_le_u32(builder, note_name_size);
    if (iree_status_is_ok(status)) {
      status =
          loom_amdgpu_metadata_append_le_u32(builder, (uint32_t)payload.size);
    }
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_metadata_append_le_u32(
          builder, LOOM_AMDGPU_METADATA_ELF_NOTE_NT_AMDGPU_METADATA);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(builder, note_name);
    }
    if (iree_status_is_ok(status)) {
      const uint8_t null_terminator = 0;
      status = iree_string_builder_append_string(
          builder, iree_make_string_view((const char*)&null_terminator, 1));
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_amdgpu_metadata_append_align4_padding(builder, note_name_size);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(builder, payload);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_amdgpu_metadata_append_align4_padding(builder, payload.size);
    }
  }
  iree_string_builder_deinitialize(&payload_builder);
  return status;
}
