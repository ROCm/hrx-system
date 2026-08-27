// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/ops/target.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/records/target_records.h"

enum {
  LOOM_AMDGPU_TARGET_FEATURE_WORD_COUNT_ =
      (LOOM_AMDGPU_TARGET_FEATURES_COUNT_ + 63) / 64,
};
static_assert(LOOM_AMDGPU_TARGET_FEATURE_WORD_COUNT_ <=
                  LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT,
              "AMDGPU target features must fit the signed enum-set domain");
static_assert(LOOM_AMDGPU_TARGET_FEATURES_COUNT_ < 32,
              "AMDGPU target features must fit processor support flags");
static_assert((UINT32_C(1) << LOOM_AMDGPU_TARGET_FEATURES_COUNT_) - 1 ==
                  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
              "AMDGPU target feature ordinals must cover support flags");
static_assert((UINT32_C(1) << LOOM_AMDGPU_TARGET_FEATURES_SRAMECC) ==
                  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
              "sramecc target feature ordinal must map to its support bit");
static_assert((UINT32_C(1) << LOOM_AMDGPU_TARGET_FEATURES_XNACK) ==
                  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
              "xnack target feature ordinal must map to its support bit");

static loom_amdgpu_target_id_feature_support_bit_t
loom_amdgpu_target_feature_support_bit(uint8_t stable_value) {
  return (loom_amdgpu_target_id_feature_support_bit_t)(UINT32_C(1)
                                                       << stable_value);
}

static loom_signed_enum_set_t loom_amdgpu_target_materialize_features(
    const loom_amdgpu_target_identity_t* identity, uint64_t* words) {
  bool has_assertion = false;
  for (iree_host_size_t stable_value = 0;
       stable_value < LOOM_AMDGPU_TARGET_FEATURES_COUNT_; ++stable_value) {
    const loom_amdgpu_target_id_feature_support_bit_t support_bit =
        loom_amdgpu_target_feature_support_bit((uint8_t)stable_value);
    const loom_amdgpu_target_feature_state_t state =
        loom_amdgpu_amdhsa_feature_state_query(&identity->amdhsa_features,
                                               support_bit);
    const iree_host_size_t word_index = stable_value / 64u;
    const uint64_t bit = UINT64_C(1) << (stable_value % 64u);
    switch (state) {
      case LOOM_AMDGPU_TARGET_FEATURE_ANY:
      case LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
        break;
      case LOOM_AMDGPU_TARGET_FEATURE_OFF:
        words[LOOM_AMDGPU_TARGET_FEATURE_WORD_COUNT_ + word_index] |= bit;
        has_assertion = true;
        break;
      case LOOM_AMDGPU_TARGET_FEATURE_ON:
        words[word_index] |= bit;
        has_assertion = true;
        break;
      default:
        IREE_CHECK_UNREACHABLE("invalid normalized AMDGPU target feature");
        break;
    }
  }
  return has_assertion ? loom_make_signed_enum_set(
                             words, LOOM_AMDGPU_TARGET_FEATURE_WORD_COUNT_)
                       : loom_signed_enum_set_empty();
}

iree_status_t loom_amdgpu_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_amdgpu_target_facts_t* facts =
      loom_amdgpu_target_facts_cast(resolved_target->facts);
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "AMDGPU target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "AMDGPU target flags must follow target fact ordinals");
  static_assert(LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "AMDGPU target flags must follow target fact ordinals");

  loom_amdgpu_target_build_flags_t build_flags =
      (loom_amdgpu_target_build_flags_t)facts->base.explicit_fields;
  uint64_t feature_words[LOOM_AMDGPU_TARGET_FEATURE_WORD_COUNT_ * 2] = {0};
  const loom_signed_enum_set_t features =
      loom_amdgpu_target_materialize_features(&facts->identity, feature_words);
  if (features.word_count > 0) {
    build_flags |= LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_FEATURES;
  }

  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.export_plan.export_symbol,
        &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.config.contract_set_key,
        &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->base.storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &facts->base.storage.export_plan;
  const loom_target_config_t* config = &facts->base.storage.config;
  loom_op_t* target_op = NULL;
  return loom_amdgpu_target_build(
      builder, build_flags,
      (loom_amdgpu_target_kind_t)facts->identity.target->target_kind, symbol,
      snapshot->codegen_format, snapshot->artifact_format,
      snapshot->default_pointer_bitwidth, snapshot->index_bitwidth,
      snapshot->offset_bitwidth, snapshot->max_workgroup_size.x,
      snapshot->max_workgroup_size.y, snapshot->max_workgroup_size.z,
      snapshot->max_flat_workgroup_size, snapshot->max_workgroup_storage_bytes,
      snapshot->subgroup_size, snapshot->max_grid_size.x,
      snapshot->max_grid_size.y, snapshot->max_grid_size.z,
      snapshot->max_flat_grid_size, snapshot->max_workgroup_count.x,
      snapshot->max_workgroup_count.y, snapshot->max_workgroup_count.z,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, export_plan->abi_kind, export_symbol,
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      features, location, &target_op);
}

static iree_string_view_t loom_amdgpu_target_record_symbol_name(
    const loom_module_t* module, const loom_op_t* target_op) {
  loom_symbol_ref_t symbol_ref = loom_amdgpu_target_symbol(target_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_amdgpu_target_record_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_target_record_emit_wavefront_size_unsupported(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_amdgpu_processor_info_t* processor,
    uint32_t wavefront_size) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_u64(wavefront_size),
      loom_param_string(processor->name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_026,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_feature_processor_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t feature_name,
    iree_string_view_t required_state, iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_string(feature_name),
      loom_param_string(required_state),
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_048,
                                        params, IREE_ARRAYSIZE(params));
}

iree_string_view_t loom_amdgpu_target_record_target_name(
    const loom_op_t* target_op) {
  const loom_amdgpu_target_info_t* target =
      loom_amdgpu_target_record_target(target_op);
  return target != NULL ? target->name : iree_string_view_empty();
}

const loom_amdgpu_target_info_t* loom_amdgpu_target_record_target(
    const loom_op_t* target_op) {
  return loom_amdgpu_target_info_find_target_by_kind(
      (uint32_t)loom_amdgpu_target_kind(target_op));
}

iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_op_t* target_op) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(target_op);
  return processor != NULL ? processor->name : iree_string_view_empty();
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_op_t* target_op) {
  return loom_amdgpu_target_info_target_processor(
      loom_amdgpu_target_record_target(target_op));
}

void loom_amdgpu_target_record_resolve_identity(
    const loom_op_t* target_op, loom_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(target_op);
  IREE_ASSERT_ARGUMENT(out_identity);
  const loom_amdgpu_target_info_t* target =
      loom_amdgpu_target_record_target(target_op);
  IREE_ASSERT(target != NULL);
  const loom_signed_enum_set_t features =
      loom_amdgpu_target_features(target_op);
  loom_amdgpu_target_identity_initialize_with_features(
      target, features.words, features.word_count, out_identity);
}

void loom_amdgpu_target_record_resolve_properties(
    const loom_op_t* target_op, const loom_target_bundle_t* common,
    loom_amdgpu_target_properties_t* out_properties) {
  IREE_ASSERT_ARGUMENT(target_op);
  IREE_ASSERT_ARGUMENT(common);
  IREE_ASSERT_ARGUMENT(out_properties);
  loom_amdgpu_target_identity_t identity = {0};
  loom_amdgpu_target_record_resolve_identity(target_op, &identity);
  loom_amdgpu_target_properties_resolve(&identity, common, out_properties);
}

static iree_status_t loom_amdgpu_target_record_verify_features(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_amdgpu_processor_info_t* processor) {
  const loom_attribute_t attr =
      loom_op_const_attrs(op)[loom_amdgpu_target_features_ATTR_INDEX];
  if (loom_attr_is_absent(attr)) return iree_ok_status();
  const loom_signed_enum_set_t features = loom_attr_as_signed_enum_set(attr);
  if (features.word_count == 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
    };
    return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_050,
                                          params, IREE_ARRAYSIZE(params));
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  IREE_ASSERT(vtable != NULL && vtable->attr_descriptors != NULL);
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[loom_amdgpu_target_features_ATTR_INDEX];
  for (iree_host_size_t stable_value = 0;
       stable_value < LOOM_AMDGPU_TARGET_FEATURES_COUNT_; ++stable_value) {
    const bool positive =
        loom_signed_enum_set_contains_positive(features, (uint8_t)stable_value);
    const bool negative =
        loom_signed_enum_set_contains_negative(features, (uint8_t)stable_value);
    if (!positive && !negative) continue;
    IREE_ASSERT(!(positive && negative));
    const loom_amdgpu_target_id_feature_support_bit_t support_bit =
        loom_amdgpu_target_feature_support_bit((uint8_t)stable_value);
    if (loom_amdgpu_processor_supports_target_id_features(processor,
                                                          support_bit)) {
      continue;
    }
    const loom_bstring_t feature_name =
        loom_attr_descriptor_enum_case_name(descriptor, (uint8_t)stable_value);
    IREE_ASSERT(feature_name != NULL);
    return loom_amdgpu_target_record_emit_feature_processor_mismatch(
        module, emitter, op, loom_bstring_view(feature_name),
        positive ? IREE_SV("enabled") : IREE_SV("disabled"), processor->name);
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_target_record_default_wavefront_size(
    const loom_amdgpu_target_info_t* target,
    const loom_amdgpu_processor_info_t* processor) {
  const loom_target_bundle_t* bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          target->descriptor_set_ordinal);
  if (bundle != NULL && bundle->snapshot != NULL) {
    return bundle->snapshot->subgroup_size;
  }
  return processor->properties.wavefront.default_size;
}

static bool loom_amdgpu_target_record_effective_wavefront_size(
    const loom_op_t* target_op, const loom_amdgpu_processor_info_t* processor,
    uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_attribute_t subgroup_size =
      loom_op_attrs(target_op)[loom_amdgpu_target_subgroup_size_ATTR_INDEX];
  if (!loom_attr_is_absent(subgroup_size)) {
    const int64_t value = loom_attr_as_i64(subgroup_size);
    if (value < 0 || value > UINT32_MAX) {
      return false;
    }
    *out_wavefront_size = (uint32_t)value;
    return true;
  }
  *out_wavefront_size = loom_amdgpu_target_record_default_wavefront_size(
      loom_amdgpu_target_record_target(target_op), processor);
  return true;
}

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));

  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(op);

  IREE_RETURN_IF_ERROR(loom_amdgpu_target_record_verify_features(
      module, op, emitter, processor));

  uint32_t wavefront_size = 0;
  if (loom_amdgpu_target_record_effective_wavefront_size(op, processor,
                                                         &wavefront_size) &&
      !loom_amdgpu_processor_properties_support_wavefront_size(
          &processor->properties, wavefront_size)) {
    return loom_amdgpu_target_record_emit_wavefront_size_unsupported(
        module, emitter, op, processor, wavefront_size);
  }
  return iree_ok_status();
}
