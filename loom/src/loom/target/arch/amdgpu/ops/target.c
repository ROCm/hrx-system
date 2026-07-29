// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/ops/target.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/materialization.h"

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

static iree_status_t
loom_amdgpu_target_record_emit_asic_revision_processor_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, int64_t revision, iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_i64(revision),
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_046,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_feature_processor_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t feature_name,
    loom_amdgpu_target_feature_state_t feature_state,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_string(feature_name),
      loom_param_string(
          loom_amdgpu_target_feature_state_attr_name(feature_state)),
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_048,
                                        params, IREE_ARRAYSIZE(params));
}

iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_op_t* target_op) {
  const loom_amdgpu_target_record_info_t* target_info =
      loom_amdgpu_target_record_info_for_kind(
          (uint32_t)loom_amdgpu_target_kind(target_op));
  return target_info != NULL ? target_info->processor_name
                             : iree_string_view_empty();
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_op_t* target_op) {
  return loom_amdgpu_target_info_find_processor(
      loom_amdgpu_target_record_processor_name(target_op));
}

static bool loom_amdgpu_target_record_explicit_asic_revision(
    const loom_op_t* target_op, int64_t* out_revision) {
  const loom_attribute_t revision_attr = loom_op_const_attrs(
      target_op)[loom_amdgpu_target_asic_revision_ATTR_INDEX];
  if (loom_attr_is_absent(revision_attr)) {
    *out_revision = 0;
    return false;
  }
  *out_revision = loom_attr_as_i64(revision_attr);
  return true;
}

static void loom_amdgpu_target_record_apply_feature_attr(
    const loom_op_t* target_op, uint8_t attr_index,
    loom_amdgpu_target_feature_state_t* inout_feature_state) {
  const loom_attribute_t feature_attr =
      loom_op_const_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(feature_attr)) {
    return;
  }
  const loom_amdgpu_target_feature_state_t requested_state =
      (loom_amdgpu_target_feature_state_t)loom_attr_as_enum(feature_attr);
  if (requested_state != LOOM_AMDGPU_TARGET_FEATURE_ANY) {
    *inout_feature_state = requested_state;
  }
}

const loom_amdgpu_processor_asic_revision_info_t*
loom_amdgpu_target_record_asic_revision(const loom_op_t* target_op) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(target_op);
  if (processor == NULL || processor->asic_revisions.count == 0) {
    return NULL;
  }
  int64_t explicit_revision = 0;
  if (!loom_amdgpu_target_record_explicit_asic_revision(target_op,
                                                        &explicit_revision)) {
    IREE_ASSERT(processor->asic_revisions.default_ordinal <
                processor->asic_revisions.count);
    return &processor->asic_revisions
                .entries[processor->asic_revisions.default_ordinal];
  }
  IREE_ASSERT(explicit_revision >= 0 && explicit_revision <= UINT32_MAX);
  const loom_amdgpu_processor_asic_revision_info_t* revision =
      loom_amdgpu_target_info_find_asic_revision(processor,
                                                 (uint32_t)explicit_revision);
  IREE_ASSERT(revision != NULL);
  return revision;
}

void loom_amdgpu_target_record_resolve_identity(
    const loom_op_t* target_op, loom_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(target_op);
  IREE_ASSERT_ARGUMENT(out_identity);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(target_op);
  IREE_ASSERT(processor != NULL);
  loom_amdgpu_target_identity_initialize(processor, out_identity);
  loom_amdgpu_target_record_apply_feature_attr(
      target_op, loom_amdgpu_target_sramecc_ATTR_INDEX,
      &out_identity->amdhsa_features.sramecc);
  loom_amdgpu_target_record_apply_feature_attr(
      target_op, loom_amdgpu_target_xnack_ATTR_INDEX,
      &out_identity->amdhsa_features.xnack);
  out_identity->asic_revision =
      loom_amdgpu_target_record_asic_revision(target_op);
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

iree_status_t loom_amdgpu_target_record_build_for_profile(
    loom_builder_t* builder, const loom_amdgpu_target_profile_t* profile,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op) {
  if (builder == NULL || profile == NULL ||
      profile->identity.processor == NULL ||
      profile->base.target_bundle == NULL || out_target_op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target record builder requires a complete "
                            "profile, builder, and output pointer");
  }
  *out_target_op = NULL;
  const loom_amdgpu_processor_info_t* processor = profile->identity.processor;

  const loom_amdgpu_target_record_info_t* target_record =
      loom_amdgpu_target_record_info_for_processor(processor->name);
  if (target_record == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU processor '%.*s' has no target record",
                            (int)processor->name.size, processor->name.data);
  }

  loom_amdgpu_target_identity_t default_identity = {0};
  loom_amdgpu_target_identity_initialize(processor, &default_identity);

  loom_target_record_extension_attr_t extension_attrs[3] = {0};
  iree_host_size_t extension_attr_count = 0;
  if (profile->identity.asic_revision != default_identity.asic_revision) {
    IREE_ASSERT(profile->identity.asic_revision != NULL);
    extension_attrs[extension_attr_count++] =
        (loom_target_record_extension_attr_t){
            .attr_index = loom_amdgpu_target_asic_revision_ATTR_INDEX,
            .value =
                loom_attr_i64((int64_t)profile->identity.asic_revision->value),
        };
  }
  if (profile->identity.amdhsa_features.sramecc !=
      default_identity.amdhsa_features.sramecc) {
    extension_attrs[extension_attr_count++] =
        (loom_target_record_extension_attr_t){
            .attr_index = loom_amdgpu_target_sramecc_ATTR_INDEX,
            .value = loom_attr_enum(profile->identity.amdhsa_features.sramecc),
        };
  }
  if (profile->identity.amdhsa_features.xnack !=
      default_identity.amdhsa_features.xnack) {
    extension_attrs[extension_attr_count++] =
        (loom_target_record_extension_attr_t){
            .attr_index = loom_amdgpu_target_xnack_ATTR_INDEX,
            .value = loom_attr_enum(profile->identity.amdhsa_features.xnack),
        };
  }
  return loom_target_record_projection_build(
      builder, LOOM_OP_AMDGPU_TARGET, (uint8_t)target_record->target_kind,
      symbol, profile->base.target_bundle, extension_attrs,
      extension_attr_count, location, out_target_op);
}

static bool loom_amdgpu_target_record_feature_state_is_compatible(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_target_id_feature_support_bit_t feature,
    loom_amdgpu_target_feature_state_t state) {
  if (state == LOOM_AMDGPU_TARGET_FEATURE_ANY) {
    return true;
  }
  const bool supported =
      loom_amdgpu_processor_supports_target_id_features(processor, feature);
  return supported ? state == LOOM_AMDGPU_TARGET_FEATURE_OFF ||
                         state == LOOM_AMDGPU_TARGET_FEATURE_ON
                   : state == LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
}

static iree_status_t loom_amdgpu_target_record_verify_feature_attr(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_amdgpu_processor_info_t* processor, uint8_t attr_index,
    iree_string_view_t feature_name,
    loom_amdgpu_target_id_feature_support_bit_t feature) {
  const loom_attribute_t attr = loom_op_const_attrs(op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_ok_status();
  }
  const loom_amdgpu_target_feature_state_t state =
      (loom_amdgpu_target_feature_state_t)loom_attr_as_enum(attr);
  if (loom_amdgpu_target_record_feature_state_is_compatible(processor, feature,
                                                            state)) {
    return iree_ok_status();
  }
  return loom_amdgpu_target_record_emit_feature_processor_mismatch(
      module, emitter, op, feature_name, state, processor->name);
}

static uint32_t loom_amdgpu_target_record_default_wavefront_size(
    const loom_amdgpu_processor_info_t* processor) {
  const loom_target_bundle_t* bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->properties.descriptor_set.ordinal);
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
  *out_wavefront_size =
      loom_amdgpu_target_record_default_wavefront_size(processor);
  return true;
}

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));

  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(op);

  int64_t explicit_revision = 0;
  if (loom_amdgpu_target_record_explicit_asic_revision(op,
                                                       &explicit_revision) &&
      (explicit_revision < 0 || explicit_revision > UINT32_MAX ||
       loom_amdgpu_target_info_find_asic_revision(
           processor, (uint32_t)explicit_revision) == NULL)) {
    return loom_amdgpu_target_record_emit_asic_revision_processor_mismatch(
        module, emitter, op, explicit_revision, processor->name);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_record_verify_feature_attr(
      module, op, emitter, processor, loom_amdgpu_target_sramecc_ATTR_INDEX,
      IREE_SV("sramecc"), LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC));
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_record_verify_feature_attr(
      module, op, emitter, processor, loom_amdgpu_target_xnack_ATTR_INDEX,
      IREE_SV("xnack"), LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK));

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
