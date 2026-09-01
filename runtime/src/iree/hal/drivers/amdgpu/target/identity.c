// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/target/identity.h"

typedef enum iree_hal_amdgpu_target_feature_support_bit_e {
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_NONE = 0u,
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC = 1u << 0,
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK = 1u << 1,
} iree_hal_amdgpu_target_feature_support_bit_t;
typedef uint32_t iree_hal_amdgpu_target_feature_support_flags_t;

typedef struct iree_hal_amdgpu_target_mapping_t {
  // Canonical exact, generic, or overlay target selector.
  iree_string_view_t target;
  // Backend processor selected by |target|.
  iree_string_view_t processor;
  // Code-object processor selected for an exact target.
  iree_string_view_t code_object_processor;
  // First generic code-object version supporting an exact target.
  uint32_t generic_introduction_version;
  // Bitset of iree_hal_amdgpu_target_feature_support_bit_t values.
  iree_hal_amdgpu_target_feature_support_flags_t supported_features;
  // Wavefront-size support derived from the processor table.
  iree_hal_amdgpu_wavefront_size_support_t wavefront;
} iree_hal_amdgpu_target_mapping_t;

typedef struct iree_hal_amdgpu_physical_target_mapping_t {
  // HSA ISA processor owning this physical target mapping.
  iree_string_view_t processor;
  // Physical ASIC revision reported by HSA.
  uint32_t asic_revision;
  // Canonical target selected by |processor| and |asic_revision|.
  iree_string_view_t target;
} iree_hal_amdgpu_physical_target_mapping_t;

static const iree_hal_amdgpu_target_mapping_t
    iree_hal_amdgpu_target_mappings[] = {
#define IREE_AMDGPU_TARGET_MAPPING(target, processor, code_object_processor,   \
                                   generic_introduction_version,               \
                                   supported_features, default_wavefront_size, \
                                   explicit_supported_wavefront_sizes)         \
  {IREE_SVL(target),                                                           \
   IREE_SVL(processor),                                                        \
   IREE_SVL(code_object_processor),                                            \
   generic_introduction_version,                                               \
   supported_features,                                                         \
   {default_wavefront_size, explicit_supported_wavefront_sizes}},
#include "iree/hal/drivers/amdgpu/target/identity_catalog.inl"
#undef IREE_AMDGPU_TARGET_MAPPING
};

static const iree_hal_amdgpu_physical_target_mapping_t
    iree_hal_amdgpu_physical_target_mappings[] = {
#define IREE_AMDGPU_PHYSICAL_TARGET(processor, asic_revision, target) \
  {IREE_SVL(processor), asic_revision, IREE_SVL(target)},
#include "iree/hal/drivers/amdgpu/target/identity_catalog.inl"
#undef IREE_AMDGPU_PHYSICAL_TARGET
};

static bool iree_hal_amdgpu_parse_decimal_digit(char c, uint32_t* out_value) {
  if (c < '0' || c > '9') return false;
  *out_value = (uint32_t)(c - '0');
  return true;
}

static bool iree_hal_amdgpu_parse_hex_digit(char c, uint32_t* out_value) {
  if (c >= '0' && c <= '9') {
    *out_value = (uint32_t)(c - '0');
    return true;
  } else if (c >= 'a' && c <= 'f') {
    *out_value = (uint32_t)(c - 'a' + 10);
    return true;
  } else if (c >= 'A' && c <= 'F') {
    *out_value = (uint32_t)(c - 'A' + 10);
    return true;
  }
  return false;
}

static bool iree_hal_amdgpu_parse_decimal_number(iree_string_view_t value,
                                                 uint32_t* out_number) {
  if (iree_string_view_is_empty(value)) return false;
  uint64_t number = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint32_t digit = 0;
    if (!iree_hal_amdgpu_parse_decimal_digit(value.data[i], &digit)) {
      return false;
    }
    number = number * 10 + digit;
    if (number > UINT32_MAX) return false;
  }
  *out_number = (uint32_t)number;
  return true;
}

static bool iree_hal_amdgpu_parse_exact_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_gfxip_version_t* out_version) {
  memset(out_version, 0, sizeof(*out_version));
  if (!iree_string_view_consume_prefix(&processor, IREE_SV("gfx"))) {
    return false;
  }

  uint32_t major0 = 0;
  uint32_t major1 = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
  if (processor.size == 4 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[0], &major0) &&
      major0 == 1 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[1], &major1) &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[2], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(processor.data[3], &stepping)) {
    out_version->major = 10 + major1;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return true;
  }
  if (processor.size == 3 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[0], &major0) &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[1], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(processor.data[2], &stepping)) {
    out_version->major = major0;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return true;
  }
  return false;
}

static const iree_hal_amdgpu_target_mapping_t*
iree_hal_amdgpu_target_lookup_mapping(iree_string_view_t target) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_target_mappings); ++i) {
    if (iree_string_view_equal(target,
                               iree_hal_amdgpu_target_mappings[i].target)) {
      return &iree_hal_amdgpu_target_mappings[i];
    }
  }
  return NULL;
}

static const iree_hal_amdgpu_physical_target_mapping_t*
iree_hal_amdgpu_target_find_physical_mapping(iree_string_view_t processor,
                                             uint32_t asic_revision) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_physical_target_mappings); ++i) {
    const iree_hal_amdgpu_physical_target_mapping_t* mapping =
        &iree_hal_amdgpu_physical_target_mappings[i];
    if (iree_string_view_equal(processor, mapping->processor) &&
        mapping->asic_revision == asic_revision) {
      return mapping;
    }
  }
  return NULL;
}

static bool iree_hal_amdgpu_target_has_physical_mappings(
    iree_string_view_t processor) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_physical_target_mappings); ++i) {
    if (iree_string_view_equal(
            processor, iree_hal_amdgpu_physical_target_mappings[i].processor)) {
      return true;
    }
  }
  return false;
}

static iree_hal_amdgpu_target_feature_state_t*
iree_hal_amdgpu_target_identity_feature_state(
    iree_hal_amdgpu_target_identity_t* identity,
    iree_hal_amdgpu_target_feature_support_bit_t feature) {
  switch (feature) {
    case IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC:
      return &identity->amdhsa_features.sramecc;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK:
      return &identity->amdhsa_features.xnack;
    default:
      return NULL;
  }
}

static void iree_hal_amdgpu_target_identity_apply_known_feature_support(
    iree_hal_amdgpu_target_identity_t* identity) {
  const iree_hal_amdgpu_target_mapping_t* mapping =
      iree_hal_amdgpu_target_lookup_mapping(identity->target);
  if (mapping == NULL) return;

  if (identity->amdhsa_features.sramecc ==
          IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY &&
      !iree_any_bit_set(mapping->supported_features,
                        IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC)) {
    identity->amdhsa_features.sramecc =
        IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
  }
  if (identity->amdhsa_features.xnack ==
          IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY &&
      !iree_any_bit_set(mapping->supported_features,
                        IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK)) {
    identity->amdhsa_features.xnack =
        IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
  }
}

static bool iree_hal_amdgpu_parse_generic_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_gfxip_version_t* out_version) {
  memset(out_version, 0, sizeof(*out_version));
  if (!iree_string_view_consume_prefix(&processor, IREE_SV("gfx")) ||
      !iree_string_view_consume_suffix(&processor, IREE_SV("-generic"))) {
    return false;
  }

  iree_string_view_t major = iree_string_view_empty();
  iree_string_view_t minor = iree_string_view_empty();
  if (iree_string_view_split(processor, '-', &major, &minor) == -1) {
    major = processor;
  } else if (iree_string_view_find_char(minor, '-', 0) !=
             IREE_STRING_VIEW_NPOS) {
    return false;
  }

  uint32_t major_value = 0;
  uint32_t minor_value = 0;
  if (!iree_hal_amdgpu_parse_decimal_number(major, &major_value)) {
    return false;
  }
  if (!iree_string_view_is_empty(minor) &&
      !iree_hal_amdgpu_parse_decimal_number(minor, &minor_value)) {
    return false;
  }
  out_version->major = major_value;
  out_version->minor = minor_value;
  out_version->stepping = 0;
  return true;
}

static iree_status_t iree_hal_amdgpu_target_identity_parse_processor_name(
    iree_string_view_t processor,
    iree_hal_amdgpu_target_identity_t* out_identity) {
  if (iree_string_view_is_empty(processor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target ID has an empty processor name");
  }

  out_identity->target = processor;
  out_identity->processor = processor;
  if (iree_hal_amdgpu_parse_generic_processor(processor,
                                              &out_identity->version)) {
    out_identity->kind = IREE_HAL_AMDGPU_TARGET_KIND_GENERIC;
    return iree_ok_status();
  }
  if (iree_hal_amdgpu_parse_exact_processor(processor,
                                            &out_identity->version)) {
    out_identity->kind = IREE_HAL_AMDGPU_TARGET_KIND_EXACT;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported AMDGPU target processor syntax: %.*s",
                          (int)processor.size, processor.data);
}

static iree_status_t iree_hal_amdgpu_target_identity_parse_target_name(
    iree_string_view_t target,
    iree_hal_amdgpu_target_identity_t* out_identity) {
  const iree_hal_amdgpu_target_mapping_t* mapping =
      iree_hal_amdgpu_target_lookup_mapping(target);
  if (mapping == NULL) {
    return iree_hal_amdgpu_target_identity_parse_processor_name(target,
                                                                out_identity);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_processor_name(
      mapping->processor, out_identity));
  out_identity->target = mapping->target;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_target_identity_parse_feature(
    iree_string_view_t feature, iree_hal_amdgpu_target_identity_t* identity) {
  if (feature.size < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target feature suffix is empty");
  }

  const char selector = feature.data[feature.size - 1];
  iree_hal_amdgpu_target_feature_state_t state =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  if (selector == '+') {
    state = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON;
  } else if (selector == '-') {
    state = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target feature suffix missing +/-: %.*s",
                            (int)feature.size, feature.data);
  }

  iree_string_view_t name = iree_string_view_remove_suffix(feature, /*n=*/1);
  iree_hal_amdgpu_target_feature_support_bit_t feature_support =
      IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_NONE;
  if (iree_string_view_equal(name, IREE_SV("sramecc"))) {
    feature_support = IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC;
  } else if (iree_string_view_equal(name, IREE_SV("xnack"))) {
    feature_support = IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU target feature suffix: %.*s",
                            (int)feature.size, feature.data);
  }
  const iree_hal_amdgpu_target_mapping_t* mapping =
      iree_hal_amdgpu_target_lookup_mapping(identity->target);
  if (mapping != NULL &&
      !iree_any_bit_set(mapping->supported_features, feature_support)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' does not support target feature '%.*s'",
        (int)identity->processor.size, identity->processor.data, (int)name.size,
        name.data);
  }
  iree_hal_amdgpu_target_feature_state_t* feature_state =
      iree_hal_amdgpu_target_identity_feature_state(identity, feature_support);
  if (*feature_state != IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate AMDGPU target feature suffix: %.*s",
                            (int)name.size, name.data);
  }
  *feature_state = state;
  return iree_ok_status();
}

typedef enum iree_hal_amdgpu_target_name_kind_e {
  IREE_HAL_AMDGPU_TARGET_NAME_KIND_TARGET = 0,
  IREE_HAL_AMDGPU_TARGET_NAME_KIND_PROCESSOR,
} iree_hal_amdgpu_target_name_kind_t;

static iree_status_t iree_hal_amdgpu_target_identity_parse_bare(
    iree_string_view_t value, iree_hal_amdgpu_target_name_kind_t name_kind,
    iree_hal_amdgpu_target_identity_t* out_identity) {
  memset(out_identity, 0, sizeof(*out_identity));
  out_identity->amdhsa_features.sramecc =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  out_identity->amdhsa_features.xnack =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  iree_string_view_t name = value;
  iree_string_view_t coordinates = iree_string_view_empty();
  if (iree_string_view_split(value, ':', &name, &coordinates) != -1) {
    if (iree_string_view_is_empty(coordinates) ||
        value.data[value.size - 1] == ':') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU target coordinate is empty");
    }
  }
  if (name_kind == IREE_HAL_AMDGPU_TARGET_NAME_KIND_PROCESSOR) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_processor_name(
        name, out_identity));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_target_identity_parse_target_name(name, out_identity));
  }

  while (!iree_string_view_is_empty(coordinates)) {
    iree_string_view_t coordinate = iree_string_view_empty();
    iree_string_view_t remaining_coordinates = iree_string_view_empty();
    if (iree_string_view_split(coordinates, ':', &coordinate,
                               &remaining_coordinates) == -1) {
      coordinate = coordinates;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_feature(
        coordinate, out_identity));
    coordinates = remaining_coordinates;
  }
  iree_hal_amdgpu_target_identity_apply_known_feature_support(out_identity);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_target_identity_parse_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(out_identity);
  memset(out_identity, 0, sizeof(*out_identity));
  out_identity->amdhsa_features.sramecc =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  out_identity->amdhsa_features.xnack =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_processor_name(
      processor, out_identity));
  iree_hal_amdgpu_target_identity_apply_known_feature_support(out_identity);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_identity_parse_artifact_key(
    iree_string_view_t value, iree_hal_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(out_identity);
  return iree_hal_amdgpu_target_identity_parse_bare(
      value, IREE_HAL_AMDGPU_TARGET_NAME_KIND_TARGET, out_identity);
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
    iree_string_view_t value, iree_hal_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(out_identity);
  if (!iree_string_view_consume_prefix(&value,
                                       IREE_SV("amdgcn-amd-amdhsa--"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSA ISA name requires the AMDHSA prefix");
  }
  return iree_hal_amdgpu_target_identity_parse_bare(
      value, IREE_HAL_AMDGPU_TARGET_NAME_KIND_PROCESSOR, out_identity);
}

IREE_API_EXPORT bool
iree_hal_amdgpu_target_identity_requires_physical_resolution(
    const iree_hal_amdgpu_target_identity_t* identity) {
  IREE_ASSERT_ARGUMENT(identity);
  return iree_hal_amdgpu_target_has_physical_mappings(identity->processor);
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_identity_resolve_physical_target(
    uint32_t asic_revision, iree_hal_amdgpu_target_identity_t* identity) {
  IREE_ASSERT_ARGUMENT(identity);
  if (!iree_hal_amdgpu_target_has_physical_mappings(identity->processor)) {
    return iree_ok_status();
  }

  const iree_hal_amdgpu_physical_target_mapping_t* mapping =
      iree_hal_amdgpu_target_find_physical_mapping(identity->processor,
                                                   asic_revision);
  if (mapping == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' has unknown physical ASIC revision %u",
        (int)identity->processor.size, identity->processor.data, asic_revision);
  }
  if (!iree_string_view_equal(identity->target, identity->processor) &&
      !iree_string_view_equal(identity->target, mapping->target)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target '%.*s' contradicts physical target '%.*s'",
        (int)identity->target.size, identity->target.data,
        (int)mapping->target.size, mapping->target.data);
  }
  identity->target = mapping->target;
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_hal_amdgpu_target_identity_equal(
    const iree_hal_amdgpu_target_identity_t* lhs,
    const iree_hal_amdgpu_target_identity_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->kind == rhs->kind && lhs->version.major == rhs->version.major &&
         lhs->version.minor == rhs->version.minor &&
         lhs->version.stepping == rhs->version.stepping &&
         lhs->generic_version == rhs->generic_version &&
         lhs->amdhsa_features.sramecc == rhs->amdhsa_features.sramecc &&
         lhs->amdhsa_features.xnack == rhs->amdhsa_features.xnack &&
         iree_string_view_equal(lhs->target, rhs->target) &&
         iree_string_view_equal(lhs->processor, rhs->processor);
}

typedef struct iree_hal_amdgpu_target_identity_formatter_t {
  // Caller-provided output buffer; NULL when only querying required length.
  char* buffer;
  // Caller-provided output buffer capacity in bytes.
  iree_host_size_t capacity;
  // Required output length excluding the NUL terminator.
  iree_host_size_t length;
} iree_hal_amdgpu_target_identity_formatter_t;

static void iree_hal_amdgpu_target_identity_formatter_append(
    iree_hal_amdgpu_target_identity_formatter_t* formatter,
    iree_string_view_t value) {
  if (formatter->buffer != NULL && formatter->capacity > 0 &&
      formatter->length < formatter->capacity - 1) {
    const iree_host_size_t available =
        formatter->capacity - 1 - formatter->length;
    const iree_host_size_t copy_length = iree_min(value.size, available);
    memcpy(formatter->buffer + formatter->length, value.data, copy_length);
    formatter->buffer[formatter->length + copy_length] = 0;
  }
  formatter->length += value.size;
}

static void iree_hal_amdgpu_target_identity_formatter_append_feature(
    iree_hal_amdgpu_target_identity_formatter_t* formatter,
    iree_string_view_t name, iree_hal_amdgpu_target_feature_state_t state) {
  if (state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF) {
    iree_hal_amdgpu_target_identity_formatter_append(formatter, IREE_SV(":"));
    iree_hal_amdgpu_target_identity_formatter_append(formatter, name);
    iree_hal_amdgpu_target_identity_formatter_append(formatter, IREE_SV("-"));
  } else if (state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON) {
    iree_hal_amdgpu_target_identity_formatter_append(formatter, IREE_SV(":"));
    iree_hal_amdgpu_target_identity_formatter_append(formatter, name);
    iree_hal_amdgpu_target_identity_formatter_append(formatter, IREE_SV("+"));
  }
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_identity_format_artifact_key(
    const iree_hal_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(identity);
  if (IREE_UNLIKELY(iree_string_view_is_empty(identity->target))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target identity has no target selector");
  }

  iree_hal_amdgpu_target_identity_formatter_t formatter = {
      .buffer = buffer,
      .capacity = buffer_capacity,
      .length = 0,
  };
  if (buffer != NULL && buffer_capacity > 0) buffer[0] = 0;
  iree_hal_amdgpu_target_identity_formatter_append(&formatter,
                                                   identity->target);
  iree_hal_amdgpu_target_identity_formatter_append_feature(
      &formatter, IREE_SV("sramecc"), identity->amdhsa_features.sramecc);
  iree_hal_amdgpu_target_identity_formatter_append_feature(
      &formatter, IREE_SV("xnack"), identity->amdhsa_features.xnack);
  if (out_buffer_length != NULL) {
    *out_buffer_length = formatter.length;
  }
  if (buffer != NULL && buffer_capacity <= formatter.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU target identity buffer capacity exceeded");
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_target_lookup_code_object_processor(
    iree_string_view_t exact_target,
    iree_string_view_t* out_code_object_processor) {
  const iree_hal_amdgpu_target_mapping_t* mapping =
      iree_hal_amdgpu_target_lookup_mapping(exact_target);
  if (mapping == NULL) return false;
  *out_code_object_processor = mapping->code_object_processor;
  return true;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_identity_project_code_object(
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_amdgpu_target_identity_t* out_code_object_identity) {
  IREE_ASSERT_ARGUMENT(exact_identity);
  IREE_ASSERT_ARGUMENT(out_code_object_identity);
  memset(out_code_object_identity, 0, sizeof(*out_code_object_identity));

  if (exact_identity->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code-object target lookup requires an "
                            "exact processor target ID");
  }
  iree_string_view_t code_object_processor = exact_identity->processor;
  iree_hal_amdgpu_target_lookup_code_object_processor(exact_identity->target,
                                                      &code_object_processor);
  if (iree_string_view_equal(code_object_processor,
                             exact_identity->processor)) {
    *out_code_object_identity = *exact_identity;
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_processor(
        code_object_processor, out_code_object_identity));
  }
  out_code_object_identity->amdhsa_features = exact_identity->amdhsa_features;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_amdgpu_wavefront_size_flags_t
iree_hal_amdgpu_wavefront_size_flag(uint32_t wavefront_size) {
  switch (wavefront_size) {
    case 32:
      return IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32;
    case 64:
      return IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64;
    default:
      return IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE;
  }
}

IREE_API_EXPORT bool
iree_hal_amdgpu_target_identity_lookup_wavefront_size_support(
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_amdgpu_wavefront_size_support_t* out_support) {
  IREE_ASSERT_ARGUMENT(exact_identity);
  IREE_ASSERT_ARGUMENT(out_support);
  memset(out_support, 0, sizeof(*out_support));

  if (exact_identity->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    return false;
  }
  const iree_hal_amdgpu_target_mapping_t* mapping =
      iree_hal_amdgpu_target_lookup_mapping(exact_identity->target);
  if (mapping == NULL) return false;
  *out_support = mapping->wavefront;
  return true;
}

static bool iree_hal_amdgpu_target_feature_compatible(
    iree_hal_amdgpu_target_feature_state_t code_object_feature,
    iree_hal_amdgpu_target_feature_state_t agent_feature) {
  if (code_object_feature == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON ||
      code_object_feature == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF) {
    return code_object_feature == agent_feature;
  }
  return true;
}

IREE_API_EXPORT iree_hal_amdgpu_target_compatibility_t
iree_hal_amdgpu_target_identity_check_compatible(
    const iree_hal_amdgpu_target_identity_t* artifact_identity,
    const iree_hal_amdgpu_target_identity_t* agent_identity) {
  IREE_ASSERT_ARGUMENT(artifact_identity);
  IREE_ASSERT_ARGUMENT(agent_identity);

  iree_hal_amdgpu_target_compatibility_t compatibility =
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE;
  if (artifact_identity->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    if (agent_identity->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT ||
        !iree_string_view_equal(artifact_identity->target,
                                agent_identity->target)) {
      compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_TARGET;
    }
  } else {
    uint32_t minimum_generic_version = 1;
    if (agent_identity->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
      const iree_hal_amdgpu_target_mapping_t* mapping =
          iree_hal_amdgpu_target_lookup_mapping(agent_identity->target);
      if (mapping == NULL ||
          !iree_string_view_equal(artifact_identity->processor,
                                  mapping->code_object_processor)) {
        compatibility |=
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY;
      } else {
        minimum_generic_version = mapping->generic_introduction_version;
      }
    } else if (!iree_string_view_equal(artifact_identity->processor,
                                       agent_identity->processor)) {
      compatibility |=
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY;
    }
    if (artifact_identity->generic_version != 0 &&
        artifact_identity->generic_version < minimum_generic_version) {
      compatibility |=
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION;
    }
  }
  if (!iree_hal_amdgpu_target_feature_compatible(
          artifact_identity->amdhsa_features.sramecc,
          agent_identity->amdhsa_features.sramecc)) {
    compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC;
  }
  if (!iree_hal_amdgpu_target_feature_compatible(
          artifact_identity->amdhsa_features.xnack,
          agent_identity->amdhsa_features.xnack)) {
    compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK;
  }
  return compatibility;
}

static void iree_hal_amdgpu_target_compatibility_formatter_append_reason(
    iree_hal_amdgpu_target_identity_formatter_t* formatter,
    iree_host_size_t* inout_reason_count, iree_string_view_t reason) {
  if (*inout_reason_count != 0) {
    iree_hal_amdgpu_target_identity_formatter_append(formatter, IREE_SV(", "));
  }
  iree_hal_amdgpu_target_identity_formatter_append(formatter, reason);
  ++*inout_reason_count;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_target_compatibility_format(
    iree_hal_amdgpu_target_compatibility_t compatibility,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length) {
  iree_hal_amdgpu_target_identity_formatter_t formatter = {
      .buffer = buffer,
      .capacity = buffer_capacity,
      .length = 0,
  };
  if (buffer != NULL && buffer_capacity > 0) buffer[0] = 0;

  iree_host_size_t reason_count = 0;
  if (compatibility == IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("compatible"));
  }
  if (iree_any_bit_set(compatibility,
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_TARGET)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("target"));
  }
  if (iree_any_bit_set(
          compatibility,
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("generic family"));
  }
  if (iree_any_bit_set(
          compatibility,
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("generic version"));
  }
  if (iree_any_bit_set(compatibility,
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("sramecc"));
  }
  if (iree_any_bit_set(compatibility,
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("xnack"));
  }
  if (out_buffer_length != NULL) {
    *out_buffer_length = formatter.length;
  }
  if (buffer != NULL && buffer_capacity <= formatter.length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU target compatibility buffer capacity exceeded");
  }
  return iree_ok_status();
}
