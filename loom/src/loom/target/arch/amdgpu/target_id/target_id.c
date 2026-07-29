// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_id/target_id.h"

#include "loom/codegen/low/target_binding.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"

const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_op(
    const loom_op_t* target_op) {
  if (!loom_amdgpu_target_isa(target_op)) {
    return NULL;
  }
  return loom_amdgpu_target_record_processor(target_op);
}

static const loom_op_t* loom_amdgpu_target_op_from_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  return module->symbols.entries[target_ref.symbol_id].defining_op;
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  return loom_amdgpu_target_processor_from_op(
      loom_amdgpu_target_op_from_ref(module, target_ref));
}

bool loom_amdgpu_target_identity_from_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref,
    loom_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(out_identity);
  *out_identity = (loom_amdgpu_target_identity_t){0};
  const loom_op_t* target_op =
      loom_amdgpu_target_op_from_ref(module, target_ref);
  if (!loom_amdgpu_target_isa(target_op)) {
    return false;
  }
  loom_amdgpu_target_record_resolve_identity(target_op, out_identity);
  return true;
}

const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_low_resolved_target_t* target) {
  return loom_amdgpu_target_processor_from_op(target->target_op);
}

const loom_amdgpu_processor_properties_t*
loom_amdgpu_target_processor_properties_from_resolved_target(
    const loom_low_resolved_target_t* target) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(target);
  return processor != NULL ? &processor->properties : NULL;
}

static iree_status_t loom_amdgpu_target_id_append_feature(
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

static iree_status_t loom_amdgpu_amdhsa_target_id_append(
    const loom_amdgpu_target_identity_t* identity, iree_string_view_t prefix,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, prefix));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, identity->processor->name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_id_append_feature(
      IREE_SV("sramecc"), identity->amdhsa_features.sramecc, builder));
  return loom_amdgpu_target_id_append_feature(
      IREE_SV("xnack"), identity->amdhsa_features.xnack, builder);
}

iree_status_t loom_amdgpu_artifact_target_key_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_id_append(
      identity, iree_string_view_empty(), builder));
  if (identity->asic_revision == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ":asic-revision="));
  return iree_string_builder_append_string(builder,
                                           identity->asic_revision->name);
}

iree_status_t loom_amdgpu_artifact_target_key_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_target_id) {
  *out_target_id = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(buffer, buffer_capacity,
                                              &builder);
  const iree_status_t status =
      loom_amdgpu_artifact_target_key_append(identity, &builder);
  if (iree_status_is_ok(status)) {
    *out_target_id = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

typedef iree_status_t (*loom_amdgpu_target_id_append_fn_t)(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder);

static iree_status_t loom_amdgpu_target_identity_format_with_appender(
    const loom_amdgpu_target_identity_t* identity,
    loom_amdgpu_target_id_append_fn_t append, iree_arena_allocator_t* arena,
    iree_string_view_t* out_target_id) {
  *out_target_id = iree_string_view_empty();

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status = append(identity, &measure_builder);
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
  status = append(identity, &builder);
  if (iree_status_is_ok(status)) {
    *out_target_id = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t loom_amdgpu_artifact_target_key_format_arena(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id) {
  return loom_amdgpu_target_identity_format_with_appender(
      identity, loom_amdgpu_artifact_target_key_append, arena, out_target_id);
}

static iree_status_t loom_amdgpu_amdhsa_code_object_target_id_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder) {
  return loom_amdgpu_amdhsa_target_id_append(
      identity, loom_amdgpu_target_info_amdhsa_target_id_prefix, builder);
}

iree_status_t loom_amdgpu_amdhsa_code_object_target_id_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id) {
  return loom_amdgpu_target_identity_format_with_appender(
      identity, loom_amdgpu_amdhsa_code_object_target_id_append, arena,
      out_target_id);
}
