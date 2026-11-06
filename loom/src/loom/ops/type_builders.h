// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TYPE_BUILDERS_H_
#define LOOM_OPS_TYPE_BUILDERS_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/types.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;

// Exact-build family keys; serialized types use their public names.
enum {
  LOOM_PARAMETERIZED_TYPE_CHANNEL = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_PARAMETERIZED_TYPE_COMMON_NAMESPACE, 0),
  LOOM_PARAMETERIZED_TYPE_READ = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_PARAMETERIZED_TYPE_COMMON_NAMESPACE, 1),
  LOOM_PARAMETERIZED_TYPE_WRITE = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_PARAMETERIZED_TYPE_COMMON_NAMESPACE, 2),
};

extern const loom_bstring_t loom_encoding_type_role_enum_names[5];
static inline iree_string_view_t loom_encoding_type_role_name(loom_encoding_role_t value) {
  uint8_t index = (uint8_t)value;
  loom_bstring_t name = index < IREE_ARRAYSIZE(loom_encoding_type_role_enum_names) ? loom_encoding_type_role_enum_names[index] : NULL;
  return name ? loom_bstring_view(name) : iree_string_view_empty();
}
static inline bool loom_encoding_type_role_parse(iree_string_view_t name, loom_encoding_role_t* out_value) {
  iree_host_size_t value = 0;
  if (!loom_bstring_table_find(
          loom_encoding_type_role_enum_names, IREE_ARRAYSIZE(loom_encoding_type_role_enum_names), name, &value)) {
    return false;
  }
  *out_value = (loom_encoding_role_t)value;
  return true;
}

static inline bool loom_channel_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_kind(type) == LOOM_PARAMETERIZED_TYPE_CHANNEL;
}
enum { LOOM_CHANNEL_TYPE_PAYLOAD_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_channel_type_payload(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_CHANNEL_TYPE_PAYLOAD_PARAMETER_INDEX]);
}
iree_status_t loom_channel_type_make(
    loom_module_t* module,
    loom_type_id_t payload,
    loom_type_t* out_type);

// Permission on an owned consuming access; absence means immutable reading.
typedef enum loom_read_type_mode_e {
  LOOM_READ_TYPE_MODE_MUTABLE = 1,
  LOOM_READ_TYPE_MODE_COUNT_ = 2,
} loom_read_type_mode_t;

enum loom_read_type_build_flag_bits_e {
  LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE = 1u << 0,
};
typedef uint32_t loom_read_type_build_flags_t;

static inline bool loom_read_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_kind(type) == LOOM_PARAMETERIZED_TYPE_READ;
}
enum { LOOM_READ_TYPE_PAYLOAD_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_read_type_payload(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_READ_TYPE_PAYLOAD_PARAMETER_INDEX]);
}
enum { LOOM_READ_TYPE_MODE_PARAMETER_INDEX = 1 };
static inline bool loom_read_type_has_mode(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_READ_TYPE_MODE_PARAMETER_INDEX]);
}
static inline loom_read_type_mode_t loom_read_type_mode(loom_type_t type) {
  return (loom_read_type_mode_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_READ_TYPE_MODE_PARAMETER_INDEX]);
}
iree_status_t loom_read_type_make(
    loom_module_t* module,
    loom_read_type_build_flags_t build_flags,
    loom_type_id_t payload,
    loom_read_type_mode_t mode,
    loom_type_t* out_type);

static inline bool loom_write_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_kind(type) == LOOM_PARAMETERIZED_TYPE_WRITE;
}
enum { LOOM_WRITE_TYPE_PAYLOAD_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_write_type_payload(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_WRITE_TYPE_PAYLOAD_PARAMETER_INDEX]);
}
iree_status_t loom_write_type_make(
    loom_module_t* module,
    loom_type_id_t payload,
    loom_type_t* out_type);

extern const loom_bstring_t loom_low_storage_type_space_enum_names[4];
static inline iree_string_view_t loom_low_storage_type_space_name(loom_storage_space_t value) {
  uint8_t index = (uint8_t)value;
  loom_bstring_t name = index < IREE_ARRAYSIZE(loom_low_storage_type_space_enum_names) ? loom_low_storage_type_space_enum_names[index] : NULL;
  return name ? loom_bstring_view(name) : iree_string_view_empty();
}
static inline bool loom_low_storage_type_space_parse(iree_string_view_t name, loom_storage_space_t* out_value) {
  iree_host_size_t value = 0;
  if (!loom_bstring_table_find(
          loom_low_storage_type_space_enum_names, IREE_ARRAYSIZE(loom_low_storage_type_space_enum_names), name, &value)) {
    return false;
  }
  *out_value = (loom_storage_space_t)value;
  return true;
}

static inline bool loom_ref_type_isa(loom_type_t type) {
  return loom_type_is_reference(type) && loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_OWNED;
}
iree_status_t loom_ref_type_make(loom_module_t* module, loom_symbol_ref_t declaration, loom_type_t* out_type);

static inline bool loom_ptr_type_isa(loom_type_t type) {
  return loom_type_is_reference(type) && loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_BORROWED;
}
iree_status_t loom_ptr_type_make(loom_module_t* module, loom_symbol_ref_t declaration, loom_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TYPE_BUILDERS_H_
