// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/ops/target.h"

#include <stdint.h>

#include "loom/error/parameter.h"
#include "loom/error/x86_error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/records/target_records.h"

enum {
  LOOM_X86_TARGET_FEATURE_WORD_COUNT_ =
      (LOOM_X86_TARGET_FEATURES_COUNT_ + 63) / 64,
};
static_assert(LOOM_X86_TARGET_FEATURE_WORD_COUNT_ == 1,
              "x86 target features must fit one fact word");
static_assert(LOOM_X86_TARGET_FEATURE_WORD_COUNT_ <=
                  LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT,
              "x86 target features must fit the signed enum-set domain");
static_assert((UINT64_C(1) << LOOM_X86_TARGET_FEATURES_COUNT_) - 1 ==
                  LOOM_X86_FEATURE_KNOWN_BITS,
              "x86 target feature ordinals must cover known features");
#define LOOM_X86_TARGET_FEATURE(enum_suffix, feature_bit)                     \
  static_assert(                                                              \
      (UINT64_C(1) << LOOM_X86_TARGET_FEATURES_##enum_suffix) == feature_bit, \
      "x86 target feature ordinal must match private bit");
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_TARGET_FEATURE

static iree_status_t loom_x86_target_record_emit(
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

static iree_status_t loom_x86_target_record_emit_private_attr(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attribute_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(attribute_name),
  };
  return loom_x86_target_record_emit(emitter, op, LOOM_ERR_X86_006, params,
                                     IREE_ARRAYSIZE(params));
}

static iree_status_t loom_x86_target_record_verify_features(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (!loom_x86_target_has_features(op)) {
    return iree_ok_status();
  }
  const loom_signed_enum_set_t features = loom_x86_target_features(op);
  const loom_x86_feature_bits_t positive_features =
      features.word_count > 0 ? features.words[0] : 0;
  const loom_x86_target_policy_t* policy =
      loom_x86_target_policy_lookup((uint8_t)loom_x86_target_kind(op));
  const loom_x86_feature_bits_t unsupported_features =
      positive_features & ~policy->supported_feature_bits;
  if (unsupported_features == 0) {
    return iree_ok_status();
  }

  uint8_t stable_value = 0;
  while (!iree_any_bit_set(unsupported_features, UINT64_C(1) << stable_value)) {
    ++stable_value;
  }
  const loom_attr_descriptor_t* feature_descriptor =
      loom_x86_target_features_descriptor(module, op);
  const loom_attr_descriptor_t* selector_descriptor =
      loom_x86_target_kind_descriptor(module, op);
  const loom_bstring_t feature_name =
      loom_attr_descriptor_enum_case_name(feature_descriptor, stable_value);
  const loom_bstring_t selector_name = loom_attr_descriptor_enum_case_name(
      selector_descriptor, (uint8_t)loom_x86_target_kind(op));
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_bstring_view(feature_name)),
      loom_param_string(loom_bstring_view(selector_name)),
  };
  return loom_x86_target_record_emit(emitter, op, LOOM_ERR_X86_007, params,
                                     IREE_ARRAYSIZE(params));
}

iree_status_t loom_x86_target_record_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));
  if (loom_x86_target_has_contract_set_key(op)) {
    return loom_x86_target_record_emit_private_attr(
        emitter, op, IREE_SV("contract_set_key"));
  }
  if (loom_x86_target_has_contract_feature_bits(op)) {
    return loom_x86_target_record_emit_private_attr(
        emitter, op, IREE_SV("contract_feature_bits"));
  }
  return loom_x86_target_record_verify_features(module, op, emitter);
}

iree_status_t loom_x86_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_x86_target_facts_t* facts =
      loom_x86_target_facts_cast(resolved_target->facts);
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "x86 target flags reserve the first 30 bits for common facts");
  static_assert(LOOM_X86_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "x86 target flags must follow target fact ordinals");
  static_assert(LOOM_X86_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "x86 target flags must follow target fact ordinals");

  loom_x86_target_build_flags_t build_flags =
      (loom_x86_target_build_flags_t)facts->base.explicit_fields;
  build_flags &= ~(LOOM_X86_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY |
                   LOOM_X86_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS);
  const bool has_features = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
  if (has_features) {
    build_flags |= LOOM_X86_TARGET_BUILD_FLAG_HAS_FEATURES;
  }

  uint64_t feature_words[2] = {
      facts->positive_feature_assertions,
      facts->negative_feature_assertions,
  };
  const loom_signed_enum_set_t features =
      feature_words[0] != 0 || feature_words[1] != 0
          ? loom_make_signed_enum_set(feature_words, 1)
          : loom_signed_enum_set_empty();

  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_X86_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.export_plan.export_symbol,
        &export_symbol));
  }

  const loom_target_snapshot_t* snapshot = &facts->base.storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &facts->base.storage.export_plan;
  loom_op_t* target_op = NULL;
  return loom_x86_target_build(
      builder, build_flags, (loom_x86_target_kind_t)facts->base.selector,
      symbol, snapshot->codegen_format, snapshot->artifact_format,
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
      export_plan->linkage, LOOM_STRING_ID_INVALID, 0, features, location,
      &target_op);
}
