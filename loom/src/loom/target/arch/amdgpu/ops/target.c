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
#include "loom/target/arch/amdgpu/records/target_records.h"

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

static iree_string_view_t loom_amdgpu_gfx1250_revision_name(
    loom_amdgpu_gfx1250_revision_t revision) {
  switch (revision) {
    case LOOM_AMDGPU_GFX1250_REVISION_A0:
      return IREE_SV("a0");
    case LOOM_AMDGPU_GFX1250_REVISION_B0:
      return IREE_SV("b0");
    default:
      return IREE_SV("unspecified");
  }
}

static iree_status_t
loom_amdgpu_target_record_emit_gfx1250_revision_processor_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_amdgpu_gfx1250_revision_t revision,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_string(loom_amdgpu_gfx1250_revision_name(revision)),
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_046,
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

static loom_amdgpu_gfx1250_revision_t
loom_amdgpu_target_record_explicit_gfx1250_revision(
    const loom_op_t* target_op) {
  const loom_attribute_t revision_attr = loom_op_const_attrs(
      target_op)[loom_amdgpu_target_gfx1250_revision_ATTR_INDEX];
  return loom_attr_is_absent(revision_attr)
             ? LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED
             : (loom_amdgpu_gfx1250_revision_t)loom_attr_as_enum(revision_attr);
}

loom_amdgpu_gfx1250_revision_t
loom_amdgpu_target_record_effective_gfx1250_revision(
    const loom_op_t* target_op) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_record_processor(target_op);
  if (processor == NULL ||
      !iree_string_view_equal(processor->name, IREE_SV("gfx1250"))) {
    return LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED;
  }
  const loom_amdgpu_gfx1250_revision_t explicit_revision =
      loom_amdgpu_target_record_explicit_gfx1250_revision(target_op);
  return explicit_revision == LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED
             ? LOOM_AMDGPU_GFX1250_REVISION_B0
             : explicit_revision;
}

static iree_status_t loom_amdgpu_target_profile_validate(
    const loom_amdgpu_target_profile_t* profile) {
  if (profile == NULL || profile->processor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target profile requires a processor row");
  }
  const bool is_gfx1250 =
      iree_string_view_equal(profile->processor->name, IREE_SV("gfx1250"));
  if (is_gfx1250 &&
      profile->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_A0 &&
      profile->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_B0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "gfx1250 target profile requires an A0 or B0 silicon revision");
  }
  if (!is_gfx1250 &&
      profile->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "gfx1250 silicon revision cannot qualify processor '%.*s'",
        (int)profile->processor->name.size, profile->processor->name.data);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_record_build_for_profile(
    loom_builder_t* builder, const loom_amdgpu_target_profile_t* profile,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op) {
  if (builder == NULL || out_target_op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target record builder requires non-NULL "
                            "builder and output pointers");
  }
  *out_target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_validate(profile));
  const loom_amdgpu_processor_info_t* processor = profile->processor;

  const loom_amdgpu_target_record_info_t* target_record =
      loom_amdgpu_target_record_info_for_processor(processor->name);
  if (target_record == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU processor '%.*s' has no target record",
                            (int)processor->name.size, processor->name.data);
  }

  loom_amdgpu_target_build_flags_t build_flags = 0;
  if (profile->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED) {
    build_flags |= LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_GFX1250_REVISION;
  }
  return loom_amdgpu_target_build(
      builder, build_flags,
      (loom_amdgpu_target_kind_t)target_record->target_kind, symbol, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      LOOM_STRING_ID_INVALID, 0, 0, LOOM_STRING_ID_INVALID, 0,
      (uint8_t)profile->gfx1250_revision, location, out_target_op);
}

static uint32_t loom_amdgpu_target_record_default_wavefront_size(
    const loom_amdgpu_processor_info_t* processor) {
  const loom_target_bundle_t* bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set.ordinal);
  if (bundle != NULL && bundle->snapshot != NULL) {
    return bundle->snapshot->subgroup_size;
  }
  return processor->wavefront.default_size;
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

  const loom_amdgpu_gfx1250_revision_t explicit_revision =
      loom_amdgpu_target_record_explicit_gfx1250_revision(op);
  if (explicit_revision != LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED &&
      !iree_string_view_equal(processor->name, IREE_SV("gfx1250"))) {
    return loom_amdgpu_target_record_emit_gfx1250_revision_processor_mismatch(
        module, emitter, op, explicit_revision, processor->name);
  }

  uint32_t wavefront_size = 0;
  if (loom_amdgpu_target_record_effective_wavefront_size(op, processor,
                                                         &wavefront_size) &&
      !loom_amdgpu_processor_supports_wavefront_size(processor,
                                                     wavefront_size)) {
    return loom_amdgpu_target_record_emit_wavefront_size_unsupported(
        module, emitter, op, processor, wavefront_size);
  }
  return iree_ok_status();
}
