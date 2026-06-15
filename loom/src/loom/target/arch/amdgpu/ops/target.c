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

static iree_status_t loom_amdgpu_target_record_emit_unknown_processor(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_003,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_descriptor_set_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t record_descriptor_set) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->name),
      loom_param_string(processor->descriptor_set.key),
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_string(record_descriptor_set),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_005,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_no_descriptor_set(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_amdgpu_processor_info_t* processor) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->name),
  };
  return loom_amdgpu_target_record_emit(emitter, op, LOOM_ERR_AMDGPU_004,
                                        params, IREE_ARRAYSIZE(params));
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

static iree_string_view_t loom_amdgpu_target_record_string_attr(
    const loom_module_t* module, const loom_op_t* target_op,
    uint8_t attr_index) {
  loom_attribute_t attr = loom_op_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_empty();
  }
  loom_string_id_t string_id = loom_attr_as_string_id(attr);
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_module_t* module, const loom_op_t* target_op) {
  iree_string_view_t explicit_processor = loom_amdgpu_target_record_string_attr(
      module, target_op, loom_amdgpu_target_processor_ATTR_INDEX);
  if (!iree_string_view_is_empty(explicit_processor)) {
    return explicit_processor;
  }
  const loom_amdgpu_target_record_info_t* target_info =
      loom_amdgpu_target_record_info_for_kind(
          (uint32_t)loom_amdgpu_target_kind(target_op));
  return target_info != NULL ? target_info->default_processor_name
                             : iree_string_view_empty();
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_module_t* module, const loom_op_t* target_op) {
  return loom_amdgpu_target_info_find_processor(
      loom_amdgpu_target_record_processor_name(module, target_op));
}

iree_status_t loom_amdgpu_target_record_build_for_processor(
    loom_builder_t* builder, const loom_amdgpu_processor_info_t* processor,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op) {
  if (builder == NULL || out_target_op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target record builder requires non-NULL "
                            "builder and output pointers");
  }
  *out_target_op = NULL;
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target record builder requires a "
                            "processor row");
  }

  const loom_amdgpu_target_record_info_t* target_record =
      loom_amdgpu_target_record_info_for_processor(processor->name);
  if (target_record == NULL) {
    target_record = loom_amdgpu_target_record_default_info_for_descriptor_set(
        processor->descriptor_set.ordinal);
  }
  if (target_record == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU processor '%.*s' has unsupported descriptor set ordinal %u",
        (int)processor->name.size, processor->name.data,
        (unsigned)processor->descriptor_set.ordinal);
  }

  loom_amdgpu_target_build_flags_t build_flags = 0;
  loom_string_id_t processor_id = LOOM_STRING_ID_INVALID;
  if (!iree_string_view_equal(processor->name,
                              target_record->default_processor_name)) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        builder->module, processor->name, &processor_id));
    build_flags |= LOOM_AMDGPU_TARGET_BUILD_FLAG_HAS_PROCESSOR;
  }

  return loom_amdgpu_target_build(
      builder, build_flags,
      (loom_amdgpu_target_kind_t)target_record->target_kind, symbol, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      LOOM_STRING_ID_INVALID, 0, 0, LOOM_STRING_ID_INVALID, 0, processor_id,
      location, out_target_op);
}

static uint32_t loom_amdgpu_target_record_default_wavefront_size(
    const loom_amdgpu_processor_info_t* default_processor) {
  const loom_target_bundle_t* bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          default_processor->descriptor_set.ordinal);
  if (bundle != NULL && bundle->snapshot != NULL) {
    return bundle->snapshot->subgroup_size;
  }
  return default_processor->wavefront.default_size;
}

static bool loom_amdgpu_target_record_effective_wavefront_size(
    const loom_op_t* target_op,
    const loom_amdgpu_processor_info_t* default_processor,
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
      loom_amdgpu_target_record_default_wavefront_size(default_processor);
  return true;
}

iree_status_t loom_amdgpu_target_record_set_processor(
    loom_module_t* module, loom_op_t* target_op,
    const loom_amdgpu_processor_info_t* processor) {
  loom_string_id_t processor_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->name, &processor_id));
  loom_op_attrs(target_op)[loom_amdgpu_target_processor_ATTR_INDEX] =
      loom_attr_string(processor_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));

  const iree_string_view_t processor_name =
      loom_amdgpu_target_record_processor_name(module, op);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  if (processor == NULL) {
    return loom_amdgpu_target_record_emit_unknown_processor(emitter, op,
                                                            processor_name);
  }
  if (processor->descriptor_set.ordinal ==
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE ||
      iree_string_view_is_empty(processor->descriptor_set.key)) {
    return loom_amdgpu_target_record_emit_no_descriptor_set(emitter, op,
                                                            processor);
  }

  const loom_amdgpu_target_kind_t kind = loom_amdgpu_target_kind(op);
  const loom_amdgpu_target_record_info_t* target_info =
      loom_amdgpu_target_record_info_for_kind((uint32_t)kind);
  const iree_string_view_t default_processor_name =
      target_info != NULL ? target_info->default_processor_name
                          : iree_string_view_empty();
  const loom_amdgpu_processor_info_t* default_processor =
      loom_amdgpu_target_info_find_processor(default_processor_name);
  if (default_processor == NULL) {
    return loom_amdgpu_target_record_emit_unknown_processor(
        emitter, op, default_processor_name);
  }
  if (!iree_string_view_equal(processor->descriptor_set.key,
                              default_processor->descriptor_set.key)) {
    return loom_amdgpu_target_record_emit_descriptor_set_mismatch(
        module, emitter, op, processor, default_processor->descriptor_set.key);
  }

  uint32_t wavefront_size = 0;
  if (loom_amdgpu_target_record_effective_wavefront_size(op, default_processor,
                                                         &wavefront_size) &&
      !loom_amdgpu_processor_supports_wavefront_size(processor,
                                                     wavefront_size)) {
    return loom_amdgpu_target_record_emit_wavefront_size_unsupported(
        module, emitter, op, processor, wavefront_size);
  }
  return iree_ok_status();
}
