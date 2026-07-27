// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/features.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/spirv/features_tables.inl"

loom_spirv_feature_bits_t loom_spirv_feature_atom_bit(
    loom_spirv_feature_atom_t atom) {
  if (atom <= LOOM_SPIRV_FEATURE_ATOM_UNKNOWN ||
      atom >= LOOM_SPIRV_FEATURE_ATOM_COUNT) {
    return 0;
  }
  return UINT64_C(1) << atom;
}

const loom_spirv_feature_atom_descriptor_t* loom_spirv_feature_atom_descriptor(
    loom_spirv_feature_atom_t atom) {
  if (atom <= LOOM_SPIRV_FEATURE_ATOM_UNKNOWN ||
      atom >= LOOM_SPIRV_FEATURE_ATOM_COUNT) {
    return NULL;
  }
  return &kSpirvFeatureAtoms[atom];
}

iree_string_view_t loom_spirv_feature_atom_name(
    loom_spirv_feature_atom_t atom) {
  const loom_spirv_feature_atom_descriptor_t* descriptor =
      loom_spirv_feature_atom_descriptor(atom);
  return descriptor ? descriptor->name : IREE_SV("unknown");
}

loom_spirv_feature_bits_t loom_spirv_known_feature_bits(void) {
  return LOOM_SPIRV_FEATURE_KNOWN_BITS;
}

bool loom_spirv_feature_set_has_atom(
    const loom_spirv_feature_set_t* feature_set,
    loom_spirv_feature_atom_t atom) {
  IREE_ASSERT_ARGUMENT(feature_set);
  return iree_any_bit_set(feature_set->atom_bits,
                          loom_spirv_feature_atom_bit(atom));
}

static bool loom_spirv_feature_set_has_extension(
    const loom_spirv_feature_set_t* feature_set, iree_string_view_t name) {
  for (uint8_t i = 0; i < feature_set->extension_count; ++i) {
    if (iree_string_view_equal(feature_set->extension_names[i], name)) {
      return true;
    }
  }
  return false;
}

static void loom_spirv_feature_set_append_extension(
    loom_spirv_feature_set_t* feature_set, iree_string_view_t name) {
  if (loom_spirv_feature_set_has_extension(feature_set, name)) {
    return;
  }
  feature_set->extension_names[feature_set->extension_count++] = name;
}

static bool loom_spirv_feature_set_has_uint32(const uint32_t* values,
                                              uint8_t count, uint32_t value) {
  for (uint8_t i = 0; i < count; ++i) {
    if (values[i] == value) {
      return true;
    }
  }
  return false;
}

static void loom_spirv_feature_set_append_uint32(uint32_t* values,
                                                 uint8_t* count,
                                                 uint32_t value) {
  if (loom_spirv_feature_set_has_uint32(values, *count, value)) {
    return;
  }
  values[(*count)++] = value;
}

static void loom_spirv_feature_set_apply_models(
    const loom_spirv_feature_atom_descriptor_t* descriptor,
    loom_spirv_feature_set_t* feature_set) {
  if (descriptor->addressing_model != LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED) {
    feature_set->addressing_model = descriptor->addressing_model;
  }
  if (descriptor->memory_model != LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED) {
    feature_set->memory_model = descriptor->memory_model;
  }
}

static const loom_spirv_feature_atom_descriptor_t*
loom_spirv_feature_set_find_missing_dependency(
    loom_spirv_feature_bits_t missing_atom_bits) {
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom = (loom_spirv_feature_atom_t)i;
    if (iree_any_bit_set(missing_atom_bits,
                         loom_spirv_feature_atom_bit(atom))) {
      return &kSpirvFeatureAtoms[atom];
    }
  }
  return NULL;
}

static iree_string_view_t loom_spirv_feature_atom_primary_extension(
    const loom_spirv_feature_atom_descriptor_t* descriptor) {
  return descriptor->extension_count != 0 ? descriptor->extension_names[0]
                                          : IREE_SV("<core>");
}

static uint32_t loom_spirv_feature_atom_primary_capability(
    const loom_spirv_feature_atom_descriptor_t* descriptor) {
  return descriptor->capability_count != 0 ? descriptor->capabilities[0]
                                           : UINT32_MAX;
}

static iree_status_t loom_spirv_feature_set_apply_atom(
    const loom_spirv_feature_atom_descriptor_t* descriptor,
    iree_string_view_t target_name,
    loom_spirv_feature_bits_t requested_atom_bits,
    loom_spirv_feature_set_t* feature_set) {
  const loom_spirv_feature_bits_t missing_bits =
      descriptor->required_atom_bits & ~requested_atom_bits;
  if (missing_bits != 0) {
    const loom_spirv_feature_atom_descriptor_t* missing_descriptor =
        loom_spirv_feature_set_find_missing_dependency(missing_bits);
    const iree_string_view_t missing_name = missing_descriptor != NULL
                                                ? missing_descriptor->name
                                                : IREE_SV("<unknown>");
    const iree_string_view_t extension_name =
        loom_spirv_feature_atom_primary_extension(descriptor);
    const uint32_t capability =
        loom_spirv_feature_atom_primary_capability(descriptor);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V target '%.*s' feature atom '%.*s' from extension '%.*s' and "
        "capability %" PRIu32
        " is missing dependency '%.*s' (bits 0x%016" PRIx64 ")",
        (int)target_name.size, target_name.data, (int)descriptor->name.size,
        descriptor->name.data, (int)extension_name.size, extension_name.data,
        capability, (int)missing_name.size, missing_name.data, missing_bits);
  }
  feature_set->minimum_spirv_version = iree_max(
      feature_set->minimum_spirv_version, descriptor->minimum_spirv_version);
  loom_spirv_feature_set_apply_models(descriptor, feature_set);
  for (uint8_t i = 0; i < descriptor->extension_count; ++i) {
    loom_spirv_feature_set_append_extension(feature_set,
                                            descriptor->extension_names[i]);
  }
  for (uint8_t i = 0; i < descriptor->capability_count; ++i) {
    loom_spirv_feature_set_append_uint32(feature_set->capabilities,
                                         &feature_set->capability_count,
                                         descriptor->capabilities[i]);
  }
  for (uint8_t i = 0; i < descriptor->opcode_count; ++i) {
    loom_spirv_feature_set_append_uint32(feature_set->opcodes,
                                         &feature_set->opcode_count,
                                         descriptor->opcodes[i]);
  }
  for (uint8_t i = 0; i < descriptor->storage_class_count; ++i) {
    loom_spirv_feature_set_append_uint32(feature_set->storage_classes,
                                         &feature_set->storage_class_count,
                                         descriptor->storage_classes[i]);
  }
  for (uint8_t i = 0; i < descriptor->decoration_count; ++i) {
    loom_spirv_feature_set_append_uint32(feature_set->decorations,
                                         &feature_set->decoration_count,
                                         descriptor->decorations[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_feature_set_prepare(
    iree_string_view_t target_name,
    loom_spirv_feature_bits_t requested_atom_bits,
    loom_spirv_feature_set_t* out_feature_set) {
  IREE_ASSERT_ARGUMENT(out_feature_set);

  const loom_spirv_feature_bits_t unknown_bits =
      requested_atom_bits & ~loom_spirv_known_feature_bits();
  if (unknown_bits != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V target '%.*s' requested unknown feature "
                            "bits 0x%016" PRIx64,
                            (int)target_name.size, target_name.data,
                            unknown_bits);
  }

  memset(out_feature_set, 0, sizeof(*out_feature_set));
  out_feature_set->atom_bits = requested_atom_bits;
  out_feature_set->addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED;
  out_feature_set->memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED;
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom = (loom_spirv_feature_atom_t)i;
    if (!iree_any_bit_set(requested_atom_bits,
                          loom_spirv_feature_atom_bit(atom))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_apply_atom(
        &kSpirvFeatureAtoms[atom], target_name, requested_atom_bits,
        out_feature_set));
  }
  if (out_feature_set->addressing_model ==
      LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED) {
    out_feature_set->addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_LOGICAL;
  }
  if (out_feature_set->memory_model == LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED) {
    out_feature_set->memory_model = LOOM_SPIRV_MEMORY_MODEL_GLSL450;
  }
  return iree_ok_status();
}
