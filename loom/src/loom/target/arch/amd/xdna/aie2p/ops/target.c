// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/ops/target.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/ops.h"
#include "loom/target/arch/amd/xdna/device/profile.h"

static const loom_xdna_device_profile_t* loom_aie2p_target_record_profile(
    const loom_target_record_view_t* record) {
  const loom_attribute_t profile_attr = loom_target_record_view_attribute(
      record, loom_aie2p_target_device_profile_ATTR_INDEX);
  if (loom_attr_is_absent(profile_attr)) return NULL;
  const loom_string_id_t profile_id = loom_attr_as_string_id(profile_attr);
  IREE_ASSERT_LT(profile_id, record->strings.count);
  const loom_xdna_device_profile_t* profile =
      loom_xdna_device_profile_lookup(record->strings.values[profile_id]);
  return profile != NULL &&
                 loom_xdna_device_profile_array_family(profile)->architecture ==
                     LOOM_XDNA_ARCHITECTURE_AIE2P
             ? profile
             : NULL;
}

static void loom_aie2p_target_facts_project(
    const loom_target_record_view_t* record, loom_target_facts_t* base_facts) {
  loom_aie2p_target_facts_t* facts = (loom_aie2p_target_facts_t*)base_facts;
  facts->device_profile = loom_aie2p_target_record_profile(record);
  IREE_ASSERT(loom_target_record_view_attribute(
                  record, loom_aie2p_target_device_profile_ATTR_INDEX)
                      .kind == LOOM_ATTR_ABSENT ||
              facts->device_profile != NULL);
}

const loom_target_fact_projector_t loom_aie2p_target_fact_projector = {
    .project = loom_aie2p_target_facts_project,
};

iree_status_t loom_aie2p_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_aie2p_target_facts_t* facts =
      loom_aie2p_target_facts_cast(resolved_target->facts);
  IREE_ASSERT(facts != NULL);
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "AIE2P target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_AIE2P_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "AIE2P target flags must follow target fact ordinals");
  static_assert(LOOM_AIE2P_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "AIE2P target flags must follow target fact ordinals");

  loom_aie2p_target_build_flags_t build_flags =
      (loom_aie2p_target_build_flags_t)facts->base.explicit_fields;
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_AIE2P_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.export_plan.export_symbol,
        &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_AIE2P_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.config.contract_set_key,
        &contract_set_key));
  }
  loom_string_id_t device_profile = LOOM_STRING_ID_INVALID;
  if (facts->device_profile != NULL) {
    build_flags |= LOOM_AIE2P_TARGET_BUILD_FLAG_HAS_DEVICE_PROFILE;
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, iree_make_cstring_view(facts->device_profile->key),
        &device_profile));
  }

  const loom_target_snapshot_t* snapshot = &facts->base.storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &facts->base.storage.export_plan;
  const loom_target_config_t* config = &facts->base.storage.config;
  loom_op_t* target_op = NULL;
  return loom_aie2p_target_build(
      builder, build_flags, (loom_aie2p_target_kind_t)facts->base.selector,
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
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      device_profile, location, &target_op);
}

iree_status_t loom_aie2p_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));
  const loom_attribute_t profile_attr =
      loom_op_const_attrs(op)[loom_aie2p_target_device_profile_ATTR_INDEX];
  if (loom_attr_is_absent(profile_attr)) return iree_ok_status();

  const loom_string_id_t profile_id = loom_attr_as_string_id(profile_attr);
  IREE_ASSERT_LT(profile_id, module->strings.count);
  const iree_string_view_t profile = module->strings.entries[profile_id];
  const loom_target_record_view_t record = {
      .strings =
          {
              .values = module->strings.entries,
              .count = module->strings.count,
          },
      .attributes = loom_op_const_attrs(op),
      .attribute_count = op->attribute_count,
  };
  if (loom_aie2p_target_record_profile(&record) != NULL) {
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(profile),
          loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              loom_aie2p_target_device_profile_ATTR_INDEX)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_082,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
