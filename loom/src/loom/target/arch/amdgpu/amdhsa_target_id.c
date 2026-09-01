// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/amdhsa_target_id.h"

#include "loom/target/arch/amdgpu/target_info.h"

const iree_string_view_t loom_amdgpu_amdhsa_target_id_prefix =
    IREE_SVL("amdgcn-amd-amdhsa--");

static iree_status_t loom_amdgpu_amdhsa_target_id_validate_chars(
    iree_string_view_t target_id) {
  for (iree_host_size_t i = 0; i < target_id.size; ++i) {
    const unsigned char character = (unsigned char)target_id.data[i];
    if (character <= ' ' || character == '"' || character == '\\') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU AMDHSA target ID contains an unsupported character");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_amdhsa_target_feature_parse(
    const loom_amdgpu_processor_info_t* processor, iree_string_view_t feature,
    loom_amdgpu_amdhsa_feature_states_t* inout_features) {
  if (feature.size < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU AMDHSA target feature suffix is empty");
  }

  const char selector = feature.data[feature.size - 1];
  loom_amdgpu_target_feature_state_t state = LOOM_AMDGPU_TARGET_FEATURE_ANY;
  if (selector == '+') {
    state = LOOM_AMDGPU_TARGET_FEATURE_ON;
  } else if (selector == '-') {
    state = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target feature suffix missing +/-: %.*s",
        (int)feature.size, feature.data);
  }

  const iree_string_view_t name = iree_string_view_remove_suffix(feature, 1);
  loom_amdgpu_target_feature_state_t* feature_state = NULL;
  loom_amdgpu_target_id_feature_support_bit_t feature_support =
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE;
  if (iree_string_view_equal(name, IREE_SV("sramecc"))) {
    feature_state = &inout_features->sramecc;
    feature_support = LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC;
  } else if (iree_string_view_equal(name, IREE_SV("xnack"))) {
    feature_state = &inout_features->xnack;
    feature_support = LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported AMDGPU AMDHSA target feature suffix: %.*s",
        (int)feature.size, feature.data);
  }
  if (!loom_amdgpu_processor_supports_target_id_features(processor,
                                                         feature_support)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' does not support target feature '%.*s'",
        (int)processor->name.size, processor->name.data, (int)name.size,
        name.data);
  }
  if (*feature_state != LOOM_AMDGPU_TARGET_FEATURE_ANY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate AMDGPU AMDHSA target feature suffix: %.*s", (int)name.size,
        name.data);
  }
  *feature_state = state;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_amdhsa_feature_suffix_parse(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix,
    loom_amdgpu_amdhsa_feature_states_t* out_features) {
  IREE_ASSERT_ARGUMENT(processor);
  IREE_ASSERT_ARGUMENT(out_features);
  loom_amdgpu_amdhsa_feature_states_initialize(processor, out_features);
  iree_string_view_t remaining_features = feature_suffix;
  while (!iree_string_view_is_empty(remaining_features)) {
    iree_string_view_t feature = iree_string_view_empty();
    iree_string_view_t next_features = iree_string_view_empty();
    if (iree_string_view_split(remaining_features, ':', &feature,
                               &next_features) == -1) {
      feature = remaining_features;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_feature_parse(
        processor, feature, out_features));
    remaining_features = next_features;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_amdhsa_feature_append(
    iree_string_view_t name, loom_amdgpu_target_feature_state_t state,
    iree_string_builder_t* builder) {
  if (state != LOOM_AMDGPU_TARGET_FEATURE_OFF &&
      state != LOOM_AMDGPU_TARGET_FEATURE_ON) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV(":")));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, name));
  return iree_string_builder_append_string(
      builder,
      state == LOOM_AMDGPU_TARGET_FEATURE_ON ? IREE_SV("+") : IREE_SV("-"));
}

iree_status_t loom_amdgpu_amdhsa_feature_suffix_append(
    const loom_amdgpu_amdhsa_feature_states_t* features,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_feature_append(
      IREE_SV("sramecc"), features->sramecc, builder));
  return loom_amdgpu_amdhsa_feature_append(IREE_SV("xnack"), features->xnack,
                                           builder);
}

iree_status_t loom_amdgpu_amdhsa_target_id_parse(
    iree_string_view_t value, loom_amdgpu_amdhsa_target_id_t* out_target_id) {
  IREE_ASSERT_ARGUMENT(out_target_id);
  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){0};
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU AMDHSA target ID is required");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_id_validate_chars(value));

  iree_string_view_t processor_and_features = value;
  if (!iree_string_view_consume_prefix(&processor_and_features,
                                       loom_amdgpu_amdhsa_target_id_prefix)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target ID '%.*s' does not start with '%.*s'",
        (int)value.size, value.data,
        (int)loom_amdgpu_amdhsa_target_id_prefix.size,
        loom_amdgpu_amdhsa_target_id_prefix.data);
  }

  iree_string_view_t processor_name = processor_and_features;
  iree_string_view_t feature_suffix = iree_string_view_empty();
  if (iree_string_view_split(processor_and_features, ':', &processor_name,
                             &feature_suffix) != -1 &&
      (iree_string_view_is_empty(feature_suffix) ||
       value.data[value.size - 1] == ':')) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU AMDHSA target ID '%.*s' has an empty feature suffix",
        (int)value.size, value.data);
  }

  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor_name, &processor));
  loom_amdgpu_amdhsa_feature_states_t features = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_feature_suffix_parse(
      processor, feature_suffix, &features));
  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){
      .processor = processor,
      .feature_suffix = feature_suffix,
      .features = features,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_amdhsa_target_id_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(identity->target);
  IREE_ASSERT(processor != NULL);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, loom_amdgpu_amdhsa_target_id_prefix));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, processor->name));
  return loom_amdgpu_amdhsa_feature_suffix_append(&identity->amdhsa_features,
                                                  builder);
}

iree_status_t loom_amdgpu_amdhsa_target_id_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id) {
  *out_target_id = iree_string_view_empty();

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status =
      loom_amdgpu_amdhsa_target_id_append(identity, &measure_builder);
  const iree_host_size_t target_id_length =
      iree_string_builder_size(&measure_builder);
  iree_string_builder_deinitialize(&measure_builder);
  IREE_RETURN_IF_ERROR(status);

  iree_host_size_t storage_length = 0;
  if (!iree_host_size_checked_add(target_id_length, 1, &storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU AMDHSA target-ID length overflows");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_length, (void**)&data));
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(data, storage_length, &builder);
  status = loom_amdgpu_amdhsa_target_id_append(identity, &builder);
  if (iree_status_is_ok(status)) {
    *out_target_id = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static void loom_amdgpu_amdhsa_target_id_apply_feature(
    loom_amdgpu_target_feature_state_t state, uint32_t feature_mask,
    uint32_t off_value, uint32_t on_value, uint32_t* inout_feature_flags) {
  if (state != LOOM_AMDGPU_TARGET_FEATURE_OFF &&
      state != LOOM_AMDGPU_TARGET_FEATURE_ON) {
    return;
  }
  *inout_feature_flags &= ~feature_mask;
  *inout_feature_flags |=
      state == LOOM_AMDGPU_TARGET_FEATURE_ON ? on_value : off_value;
}

iree_status_t loom_amdgpu_amdhsa_target_id_elf_flags(
    const loom_amdgpu_amdhsa_target_id_t* target_id, uint32_t* out_elf_flags) {
  IREE_ASSERT_ARGUMENT(out_elf_flags);
  *out_elf_flags = 0;
  const loom_amdgpu_processor_info_t* processor = target_id->processor;
  if (processor->properties.elf.machine_flags == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU processor '%.*s' has no ELF e_flags mapping",
        (int)processor->name.size, processor->name.data);
  }
  uint32_t feature_flags = processor->properties.elf.feature_flags;
  loom_amdgpu_amdhsa_target_id_apply_feature(
      target_id->features.sramecc, LOOM_AMDGPU_ELF_FEATURE_SRAMECC_MASK_V4,
      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_OFF_V4,
      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ON_V4, &feature_flags);
  loom_amdgpu_amdhsa_target_id_apply_feature(
      target_id->features.xnack, LOOM_AMDGPU_ELF_FEATURE_XNACK_MASK_V4,
      LOOM_AMDGPU_ELF_FEATURE_XNACK_OFF_V4, LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4,
      &feature_flags);
  *out_elf_flags = processor->properties.elf.machine_flags | feature_flags |
                   (processor->properties.elf.generic_version
                    << LOOM_AMDGPU_ELF_GENERIC_VERSION_OFFSET_V6);
  return iree_ok_status();
}
