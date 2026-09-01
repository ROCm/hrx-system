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

const loom_amdgpu_matrix_coexecution_profile_info_t*
loom_amdgpu_target_info_matrix_coexecution_profile(
    loom_amdgpu_matrix_coexecution_profile_t profile) {
  IREE_ASSERT_LT(profile, LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_COUNT);
  return &loom_amdgpu_target_info_matrix_coexecution_profile_infos[profile];
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

iree_host_size_t loom_amdgpu_target_info_target_count(void) {
  return loom_amdgpu_target_info_target_info_count;
}

const loom_amdgpu_target_info_t* loom_amdgpu_target_info_target_at(
    iree_host_size_t index) {
  if (index >= loom_amdgpu_target_info_target_info_count) {
    return NULL;
  }
  return &loom_amdgpu_target_info_target_infos[index];
}

const loom_amdgpu_target_info_t* loom_amdgpu_target_info_find_target(
    iree_string_view_t target_name) {
  if (iree_string_view_is_empty(target_name)) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < loom_amdgpu_target_info_target_info_count;
       ++i) {
    const loom_amdgpu_target_info_t* target =
        &loom_amdgpu_target_info_target_infos[i];
    if (iree_string_view_equal(target->name, target_name)) {
      return target;
    }
  }
  return NULL;
}

const loom_amdgpu_target_info_t* loom_amdgpu_target_info_find_target_by_kind(
    uint32_t target_kind) {
  if (target_kind == 0 ||
      target_kind > loom_amdgpu_target_info_target_info_count) {
    return NULL;
  }
  const loom_amdgpu_target_info_t* target =
      &loom_amdgpu_target_info_target_infos[target_kind - 1];
  IREE_ASSERT(target->target_kind == target_kind);
  return target;
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_target_processor(
    const loom_amdgpu_target_info_t* target) {
  return target != NULL
             ? loom_amdgpu_target_info_processor_at(target->processor_ordinal)
             : NULL;
}

bool loom_amdgpu_target_info_is_generic(
    const loom_amdgpu_target_info_t* target) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  return processor != NULL && processor->properties.elf.generic_version != 0;
}

iree_status_t loom_amdgpu_target_info_lookup_target(
    iree_string_view_t target_name,
    const loom_amdgpu_target_info_t** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (iree_string_view_is_empty(target_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target is required");
  }
  const loom_amdgpu_target_info_t* target =
      loom_amdgpu_target_info_find_target(target_name);
  if (target != NULL) {
    *out_target = target;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU target '%.*s' is not supported",
                          (int)target_name.size, target_name.data);
}

bool loom_amdgpu_target_info_requires_physical_resolution(
    const loom_amdgpu_processor_info_t* processor) {
  IREE_ASSERT_ARGUMENT(processor);
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_target_info_physical_target_info_count; ++i) {
    if (loom_amdgpu_target_info_physical_target_infos[i].processor_ordinal ==
        processor->ordinal) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_amdgpu_target_info_lookup_physical_target(
    const loom_amdgpu_processor_info_t* processor, uint32_t asic_revision,
    const loom_amdgpu_target_info_t** out_target) {
  IREE_ASSERT_ARGUMENT(processor);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  bool has_physical_targets = false;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_target_info_physical_target_info_count; ++i) {
    const loom_amdgpu_physical_target_info_t* physical_target =
        &loom_amdgpu_target_info_physical_target_infos[i];
    if (physical_target->processor_ordinal != processor->ordinal) {
      continue;
    }
    has_physical_targets = true;
    if (physical_target->asic_revision == asic_revision) {
      *out_target = loom_amdgpu_target_info_find_target_by_kind(
          physical_target->target_kind);
      IREE_ASSERT(*out_target != NULL);
      return iree_ok_status();
    }
  }
  if (has_physical_targets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' has unknown physical ASIC revision %" PRIu32,
        (int)processor->name.size, processor->name.data, asic_revision);
  }
  *out_target = loom_amdgpu_target_info_find_target(processor->name);
  if (*out_target != NULL) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "AMDGPU processor '%.*s' has no supported compiler target",
      (int)processor->name.size, processor->name.data);
}

bool loom_amdgpu_processor_satisfies_code_object_requirement(
    const loom_amdgpu_processor_info_t* effective_processor,
    const loom_amdgpu_processor_info_t* required_processor) {
  if (effective_processor->ordinal == required_processor->ordinal) {
    return true;
  }
  return effective_processor->generic_code_object.processor_ordinal ==
             required_processor->ordinal &&
         effective_processor->generic_code_object.introduction_version <=
             required_processor->properties.elf.generic_version;
}

bool loom_amdgpu_target_satisfies_code_object_requirement(
    const loom_amdgpu_target_info_t* effective_target,
    const loom_amdgpu_target_info_t* required_target) {
  if (effective_target->target_kind == required_target->target_kind) {
    return true;
  }
  const loom_amdgpu_processor_info_t* effective_processor =
      loom_amdgpu_target_info_target_processor(effective_target);
  const loom_amdgpu_processor_info_t* required_processor =
      loom_amdgpu_target_info_target_processor(required_target);
  return required_processor->properties.elf.generic_version != 0 &&
         loom_amdgpu_processor_satisfies_code_object_requirement(
             effective_processor, required_processor);
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

void loom_amdgpu_amdhsa_feature_states_initialize(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_amdhsa_feature_states_t* out_features) {
  IREE_ASSERT_ARGUMENT(processor);
  IREE_ASSERT_ARGUMENT(out_features);
  *out_features = (loom_amdgpu_amdhsa_feature_states_t){
      .sramecc = loom_amdgpu_processor_supports_target_id_features(
                     processor, LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC)
                     ? LOOM_AMDGPU_TARGET_FEATURE_ANY
                     : LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED,
      .xnack = loom_amdgpu_processor_supports_target_id_features(
                   processor, LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK)
                   ? LOOM_AMDGPU_TARGET_FEATURE_ANY
                   : LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED,
  };
}
