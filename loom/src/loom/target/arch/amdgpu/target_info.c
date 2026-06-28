// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_info.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/target/arch/amdgpu/target_info_tables.h"

iree_host_size_t loom_amdgpu_target_info_processor_count(void) {
  return loom_amdgpu_target_info_processor_info_count;
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_processor_at(
    iree_host_size_t index) {
  if (index >= loom_amdgpu_target_info_processor_info_count) {
    return NULL;
  }
  return &loom_amdgpu_target_info_processor_infos[index];
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_find_processor(
    iree_string_view_t processor_name) {
  if (iree_string_view_is_empty(processor_name)) {
    return NULL;
  }
  iree_host_size_t low = 0;
  iree_host_size_t high = loom_amdgpu_target_info_processor_info_count;
  while (low < high) {
    const iree_host_size_t mid = low + (high - low) / 2;
    const loom_amdgpu_processor_info_t* processor =
        &loom_amdgpu_target_info_processor_infos[mid];
    const int comparison =
        iree_string_view_compare(processor->name, processor_name);
    if (comparison == 0) {
      return processor;
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return NULL;
}

iree_host_size_t loom_amdgpu_target_info_descriptor_set_count(void) {
  return loom_amdgpu_target_info_descriptor_set_info_count;
}

const loom_amdgpu_descriptor_set_info_t*
loom_amdgpu_target_info_descriptor_set_at(uint16_t descriptor_set_ordinal) {
  if (descriptor_set_ordinal >=
      loom_amdgpu_target_info_descriptor_set_info_count) {
    return NULL;
  }
  return &loom_amdgpu_target_info_descriptor_set_infos[descriptor_set_ordinal];
}

iree_status_t loom_amdgpu_target_info_lookup_processor(
    iree_string_view_t processor_name,
    const loom_amdgpu_processor_info_t** out_processor) {
  IREE_ASSERT_ARGUMENT(out_processor);
  *out_processor = NULL;
  if (iree_string_view_is_empty(processor_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU processor is required");
  }
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  if (processor != NULL) {
    *out_processor = processor;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU processor '%.*s' is not supported",
                          (int)processor_name.size, processor_name.data);
}

iree_status_t loom_amdgpu_target_info_lookup_descriptor_set(
    iree_string_view_t descriptor_set_key,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = NULL;
  if (iree_string_view_is_empty(descriptor_set_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU descriptor set key is required");
  }
  iree_host_size_t low = 0;
  iree_host_size_t high = loom_amdgpu_target_info_descriptor_set_info_count;
  while (low < high) {
    const iree_host_size_t mid = low + (high - low) / 2;
    const loom_amdgpu_descriptor_set_info_t* descriptor_set =
        &loom_amdgpu_target_info_descriptor_set_infos[mid];
    const int comparison =
        iree_string_view_compare(descriptor_set->key, descriptor_set_key);
    if (comparison == 0) {
      *out_descriptor_set = descriptor_set;
      return iree_ok_status();
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU descriptor set '%.*s' is not supported by native emission",
      (int)descriptor_set_key.size, descriptor_set_key.data);
}

iree_status_t loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
    uint16_t descriptor_set_ordinal,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = NULL;
  if (descriptor_set_ordinal == LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU descriptor set ordinal is required");
  }
  const loom_amdgpu_descriptor_set_info_t* descriptor_set =
      loom_amdgpu_target_info_descriptor_set_at(descriptor_set_ordinal);
  if (descriptor_set != NULL) {
    *out_descriptor_set = descriptor_set;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU descriptor set ordinal %" PRIu16
                          " is not supported by native emission",
                          descriptor_set_ordinal);
}

static iree_status_t loom_amdgpu_target_info_validate_target_id_chars(
    iree_string_view_t target_id) {
  for (iree_host_size_t i = 0; i < target_id.size; ++i) {
    const unsigned char c = (unsigned char)target_id.data[i];
    if (c <= ' ' || c == '"' || c == '\\') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU AMDHSA target id contains an unsupported character");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_target_info_parse_target_feature(
    iree_string_view_t feature,
    loom_amdgpu_target_feature_selection_t* inout_sramecc,
    loom_amdgpu_target_feature_selection_t* inout_xnack) {
  if (feature.size < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU AMDHSA target feature suffix is empty");
  }

  const char selector = feature.data[feature.size - 1];
  loom_amdgpu_target_feature_selection_t selection =
      LOOM_AMDGPU_TARGET_FEATURE_DEFAULT;
  if (selector == '+') {
    selection = LOOM_AMDGPU_TARGET_FEATURE_ON;
  } else if (selector == '-') {
    selection = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target feature suffix missing +/-: %.*s",
        (int)feature.size, feature.data);
  }

  const iree_string_view_t name = iree_string_view_remove_suffix(feature, 1);
  loom_amdgpu_target_feature_selection_t* feature_selection = NULL;
  if (iree_string_view_equal(name, IREE_SV("sramecc"))) {
    feature_selection = inout_sramecc;
  } else if (iree_string_view_equal(name, IREE_SV("xnack"))) {
    feature_selection = inout_xnack;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported AMDGPU AMDHSA target feature suffix: %.*s",
        (int)feature.size, feature.data);
  }
  if (*feature_selection != LOOM_AMDGPU_TARGET_FEATURE_DEFAULT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate AMDGPU AMDHSA target feature suffix: %.*s", (int)name.size,
        name.data);
  }
  *feature_selection = selection;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_info_parse_amdhsa_target_id(
    iree_string_view_t target_id,
    loom_amdgpu_amdhsa_target_id_t* out_target_id) {
  IREE_ASSERT_ARGUMENT(out_target_id);
  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){0};
  if (iree_string_view_is_empty(target_id)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU AMDHSA target id is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_validate_target_id_chars(target_id));
  iree_string_view_t processor_and_features = target_id;
  if (!iree_string_view_consume_prefix(
          &processor_and_features,
          loom_amdgpu_target_info_amdhsa_target_id_prefix)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target id '%.*s' does not start with '%.*s'",
        (int)target_id.size, target_id.data,
        (int)loom_amdgpu_target_info_amdhsa_target_id_prefix.size,
        loom_amdgpu_target_info_amdhsa_target_id_prefix.data);
  }
  iree_string_view_t processor_name = iree_string_view_empty();
  iree_string_view_t feature_suffix = iree_string_view_empty();
  const intptr_t split = iree_string_view_split(
      processor_and_features, ':', &processor_name, &feature_suffix);
  if (split == -1) {
    processor_name = processor_and_features;
  } else if (iree_string_view_is_empty(feature_suffix)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target id '%.*s' has an empty feature suffix",
        (int)target_id.size, target_id.data);
  } else if (target_id.data[target_id.size - 1] == ':') {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target id '%.*s' has an empty feature suffix",
        (int)target_id.size, target_id.data);
  }
  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor_name, &processor));
  loom_amdgpu_target_feature_selection_t sramecc =
      LOOM_AMDGPU_TARGET_FEATURE_DEFAULT;
  loom_amdgpu_target_feature_selection_t xnack =
      LOOM_AMDGPU_TARGET_FEATURE_DEFAULT;
  iree_string_view_t remaining_features = feature_suffix;
  while (!iree_string_view_is_empty(remaining_features)) {
    iree_string_view_t feature = iree_string_view_empty();
    iree_string_view_t next_features = iree_string_view_empty();
    if (iree_string_view_split(remaining_features, ':', &feature,
                               &next_features) == -1) {
      feature = remaining_features;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_parse_target_feature(
        feature, &sramecc, &xnack));
    remaining_features = next_features;
  }
  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){
      .processor = processor,
      .feature_suffix = feature_suffix,
      .sramecc = sramecc,
      .xnack = xnack,
  };
  return iree_ok_status();
}

static void loom_amdgpu_target_info_apply_feature_selection(
    loom_amdgpu_target_feature_selection_t selection, uint32_t feature_mask,
    uint32_t off_value, uint32_t on_value, uint32_t* inout_feature_flags) {
  if (selection == LOOM_AMDGPU_TARGET_FEATURE_DEFAULT) return;
  *inout_feature_flags &= ~feature_mask;
  if (selection == LOOM_AMDGPU_TARGET_FEATURE_ON) {
    *inout_feature_flags |= on_value;
  } else {
    *inout_feature_flags |= off_value;
  }
}

iree_status_t loom_amdgpu_target_info_amdhsa_target_id_elf_flags(
    const loom_amdgpu_amdhsa_target_id_t* target_id, uint32_t* out_elf_flags) {
  IREE_ASSERT_ARGUMENT(out_elf_flags);
  *out_elf_flags = 0;
  const loom_amdgpu_processor_info_t* processor = target_id->processor;
  if (processor->elf.machine_flags == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU processor '%.*s' has no ELF e_flags mapping",
        (int)processor->name.size, processor->name.data);
  }
  uint32_t feature_flags = processor->elf.feature_flags;
  loom_amdgpu_target_info_apply_feature_selection(
      target_id->sramecc, LOOM_AMDGPU_ELF_FEATURE_SRAMECC_MASK_V4,
      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_OFF_V4,
      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ON_V4, &feature_flags);
  loom_amdgpu_target_info_apply_feature_selection(
      target_id->xnack, LOOM_AMDGPU_ELF_FEATURE_XNACK_MASK_V4,
      LOOM_AMDGPU_ELF_FEATURE_XNACK_OFF_V4, LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4,
      &feature_flags);
  *out_elf_flags = processor->elf.machine_flags | feature_flags;
  return iree_ok_status();
}
