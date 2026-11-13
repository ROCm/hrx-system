// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/metadata.h"

#include <inttypes.h>

#include "iree/base/internal/unicode.h"
#include "loom/target/artifact_diagnostics.h"

//===----------------------------------------------------------------------===//
// Shared fields
//===----------------------------------------------------------------------===//

#define LOOM_AMDGPU_METADATA_VERSION_MAJOR 1u
#define LOOM_AMDGPU_METADATA_VERSION_MINOR 2u
#define LOOM_AMDGPU_METADATA_ELF_NOTE_NT_AMDGPU_METADATA 32u

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

//===----------------------------------------------------------------------===//
// Assembly metadata block writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_amdgpu_metadata_append_quoted_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  // Keep ordinary names compact. YAML double quotes additionally preserve
  // control characters and Unicode line breaks without scalar folding.
  bool needs_escapes = false;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const uint8_t character = (uint8_t)value.data[i];
    if (character < 0x20 || character >= 0x7F || character == '\'') {
      needs_escapes = true;
      break;
    }
  }
  const iree_string_view_t quote = needs_escapes ? IREE_SV("\"") : IREE_SV("'");
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, quote));
  if (!needs_escapes) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, value));
  } else {
    iree_host_size_t position = 0, span_start = 0;
    while (position < value.size) {
      const iree_host_size_t codepoint_start = position;
      const uint32_t codepoint = iree_unicode_utf8_decode(value, &position);
      const bool escaped_quote = codepoint == '"' || codepoint == '\\';
      const bool escaped_control = codepoint < 0x20 ||
                                   (codepoint >= 0x7F && codepoint <= 0x9F) ||
                                   codepoint == 0x2028 || codepoint == 0x2029;
      if (!escaped_quote && !escaped_control) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, iree_make_string_view(value.data + span_start,
                                         codepoint_start - span_start)));
      if (escaped_quote) {
        const char escape[] = {'\\', (char)codepoint};
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            builder, iree_make_string_view(escape, sizeof(escape))));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "\\u%04" PRIX32, codepoint));
      }
      span_start = position;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(value.data + span_start,
                                       value.size - span_start)));
  }
  return iree_string_builder_append_string(builder, quote);
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
  if (kernel->has_agpr_count) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "      .agpr_count: %" PRIu32 "\n", kernel->agpr_count));
  }
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
  if (kernel->has_workgroup_cluster_size) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder,
                                          "      .cluster_dims:\n"
                                          "        - %" PRIu32 "\n"
                                          "        - %" PRIu32 "\n"
                                          "        - %" PRIu32 "\n",
                                          kernel->workgroup_cluster_size.x,
                                          kernel->workgroup_cluster_size.y,
                                          kernel->workgroup_cluster_size.z));
  }
  for (uint16_t i = 0; i < kernel->target_extensions.count; ++i) {
    const loom_amdgpu_metadata_string_property_t* property =
        &kernel->target_extensions.entries[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "      %.*s: %.*s\n", (int)property->key.size,
        property->key.data, (int)property->value.size, property->value.data));
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDF));
  return loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)count);
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDD));
  return loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)count);
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
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_u8(builder, 0xDB));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_be_u32(builder, (uint32_t)value.size));
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
  // ROCr canonicalizes loaded AMDHSA metadata maps into lexicographic key
  // order. Match that order so strings retain their source ELF virtual
  // addresses when metadata is projected into the loaded code object.
  if (!iree_string_view_is_empty(argument->access)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".access"), argument->access));
  }
  if (!iree_string_view_is_empty(argument->actual_access)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".actual_access"), argument->actual_access));
  }
  if (!iree_string_view_is_empty(argument->address_space)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, IREE_SV(".address_space"), argument->address_space));
  }
  if (argument->alignment != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
        builder, IREE_SV(".align"), argument->alignment));
  }
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
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_metadata_append_target_extensions_before_msgpack(
    const loom_amdgpu_metadata_string_property_set_t* metadata,
    iree_string_view_t boundary, uint16_t* inout_ordinal,
    iree_string_builder_t* builder) {
  while (*inout_ordinal < metadata->count) {
    const loom_amdgpu_metadata_string_property_t* property =
        &metadata->entries[*inout_ordinal];
    if (!iree_string_view_is_empty(boundary) &&
        iree_string_view_compare(property->key, boundary) > 0) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
        builder, property->key, property->value));
    ++*inout_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_metadata_append_kernel_msgpack(
    const loom_amdgpu_metadata_kernel_t* kernel,
    iree_string_builder_t* builder) {
  const iree_host_size_t field_count =
      11 + (kernel->has_agpr_count ? 1 : 0) +
      (kernel->has_required_workgroup_size ? 1 : 0) +
      (kernel->has_workgroup_cluster_size ? 1 : 0) +
      kernel->target_extensions.count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_map(builder, field_count));
  uint16_t target_extension_ordinal = 0;
  if (kernel->has_agpr_count) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_metadata_append_target_extensions_before_msgpack(
            &kernel->target_extensions, IREE_SV(".agpr_count"),
            &target_extension_ordinal, builder));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
        builder, IREE_SV(".agpr_count"), kernel->agpr_count));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".args"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV(".args")));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_array(builder, kernel->argument_count));
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_argument_msgpack(
        &kernel->arguments[i], builder));
  }
  if (kernel->has_workgroup_cluster_size) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_metadata_append_target_extensions_before_msgpack(
            &kernel->target_extensions, IREE_SV(".cluster_dims"),
            &target_extension_ordinal, builder));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_msgpack_append_string(builder, IREE_SV(".cluster_dims")));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_array(builder, 3));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->workgroup_cluster_size.x));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->workgroup_cluster_size.y));
    IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
        builder, kernel->workgroup_cluster_size.z));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".group_segment_fixed_size"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".group_segment_fixed_size"),
      kernel->group_segment_fixed_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".kernarg_segment_align"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".kernarg_segment_align"),
      kernel->kernarg_segment_alignment));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".kernarg_segment_size"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".kernarg_segment_size"), kernel->kernarg_segment_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".max_flat_workgroup_size"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".max_flat_workgroup_size"),
      kernel->max_flat_workgroup_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".name"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV(".name"), kernel->name));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".private_segment_fixed_size"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".private_segment_fixed_size"),
      kernel->private_segment_fixed_size));
  if (kernel->has_required_workgroup_size) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_metadata_append_target_extensions_before_msgpack(
            &kernel->target_extensions, IREE_SV(".reqd_workgroup_size"),
            &target_extension_ordinal, builder));
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
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".sgpr_count"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".sgpr_count"), kernel->sgpr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".symbol"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV(".symbol"), kernel->descriptor_symbol));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".vgpr_count"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".vgpr_count"), kernel->vgpr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_metadata_append_target_extensions_before_msgpack(
          &kernel->target_extensions, IREE_SV(".wavefront_size"),
          &target_extension_ordinal, builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32_field(
      builder, IREE_SV(".wavefront_size"), kernel->wavefront_size));
  return loom_amdgpu_metadata_append_target_extensions_before_msgpack(
      &kernel->target_extensions, iree_string_view_empty(),
      &target_extension_ordinal, builder);
}

iree_status_t loom_amdgpu_metadata_append_msgpack(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_map(builder, 3));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV("amdhsa.kernels")));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_array(builder, metadata->kernel_count));
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_metadata_append_kernel_msgpack(
        &metadata->kernels[i], builder));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_string_field(
      builder, IREE_SV("amdhsa.target"), metadata->target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_msgpack_append_string(builder, IREE_SV("amdhsa.version")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_array(builder, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
      builder, LOOM_AMDGPU_METADATA_VERSION_MAJOR));
  IREE_RETURN_IF_ERROR(loom_amdgpu_msgpack_append_uint32(
      builder, LOOM_AMDGPU_METADATA_VERSION_MINOR));
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
  const iree_host_size_t padding_length = (4u - (field_length & 3u)) & 3u;
  if (padding_length == 0) {
    return iree_ok_status();
  }
  const uint8_t padding[3] = {0};
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)padding, padding_length));
}

iree_status_t loom_amdgpu_metadata_build_elf_note(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted, iree_const_byte_span_t* out_note) {
  *out_accepted = false;
  *out_note = iree_make_const_byte_span(NULL, 0);

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status =
      loom_amdgpu_metadata_append_msgpack(metadata, &measure_builder);
  const iree_host_size_t payload_size =
      iree_string_builder_size(&measure_builder);
  iree_string_builder_deinitialize(&measure_builder);
  IREE_RETURN_IF_ERROR(status);
  if (payload_size > UINT32_MAX) {
    return loom_target_artifact_emit_limit(
        emitter, NULL, IREE_SV("AMDGPU metadata"),
        IREE_SV("ELF note payload byte length"), payload_size, UINT32_MAX);
  }

  const iree_string_view_t note_name = IREE_SV("AMDGPU");
  const uint32_t note_name_size = (uint32_t)note_name.size + 1u;
  iree_host_size_t aligned_name_size = 0;
  iree_host_size_t aligned_payload_size = 0;
  iree_host_size_t note_size = 0;
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_align(note_name_size, 4, &aligned_name_size) ||
      !iree_host_size_checked_align(payload_size, 4, &aligned_payload_size) ||
      !iree_host_size_checked_add(12u, aligned_name_size, &note_size) ||
      !iree_host_size_checked_add(note_size, aligned_payload_size,
                                  &note_size) ||
      !iree_host_size_checked_add(note_size, 1u, &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU metadata ELF note size overflows");
  }

  char* note_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_size, (void**)&note_data));
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(note_data, storage_size,
                                              &builder);
  status = loom_amdgpu_metadata_append_le_u32(&builder, note_name_size);
  if (iree_status_is_ok(status)) {
    status =
        loom_amdgpu_metadata_append_le_u32(&builder, (uint32_t)payload_size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_metadata_append_le_u32(
        &builder, LOOM_AMDGPU_METADATA_ELF_NOTE_NT_AMDGPU_METADATA);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, note_name);
  }
  if (iree_status_is_ok(status)) {
    const uint8_t null_terminator = 0;
    status = iree_string_builder_append_string(
        &builder, iree_make_string_view((const char*)&null_terminator, 1));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_amdgpu_metadata_append_align4_padding(&builder, note_name_size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_metadata_append_msgpack(metadata, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_metadata_append_align4_padding(&builder, payload_size);
  }
  IREE_ASSERT(!iree_status_is_ok(status) ||
              iree_string_builder_size(&builder) == note_size);
  iree_string_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    *out_note = iree_make_const_byte_span(note_data, note_size);
    *out_accepted = true;
  }
  return status;
}
