// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <stdio.h>

#include "loom/target/launch.h"

void loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  *out_config = (loom_llvmir_target_config_t){0};
  *out_config = (loom_llvmir_target_config_t){
      .source_name = source_name,
      .target_triple = target_env->target_triple,
      .data_layout = target_env->data_layout,
      .default_pointer_bitwidth = target_env->default_pointer_bitwidth,
      .index_bitwidth = target_env->index_bitwidth,
      .offset_bitwidth = target_env->offset_bitwidth,
  };
}

void loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  loom_llvmir_target_env_module_config(profile->target_env, source_name,
                                       out_config);
}

static loom_llvmir_object_format_t loom_llvmir_object_format_from_artifact(
    loom_target_artifact_format_t artifact_format) {
  switch (artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      return LOOM_LLVMIR_OBJECT_FORMAT_ELF;
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      return LOOM_LLVMIR_OBJECT_FORMAT_COFF;
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return LOOM_LLVMIR_OBJECT_FORMAT_MACHO;
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT:
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE:
      return LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN;
  }
  return LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN;
}

static loom_llvmir_linkage_t loom_llvmir_linkage_from_target(
    loom_target_linkage_t linkage) {
  switch (linkage) {
    case LOOM_TARGET_LINKAGE_DEFAULT:
      return LOOM_LLVMIR_LINKAGE_DEFAULT;
    case LOOM_TARGET_LINKAGE_DSO_LOCAL:
      return LOOM_LLVMIR_LINKAGE_DSO_LOCAL;
  }
  return LOOM_LLVMIR_LINKAGE_DEFAULT;
}

static void loom_llvmir_resolve_hal_kernel_flat_workgroup_range(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel, uint32_t* out_min,
    uint32_t* out_max) {
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if (flat_min != 0) {
    *out_min = flat_min;
    *out_max = flat_max;
    return;
  }
  *out_min = 1;
  *out_max = snapshot->max_flat_workgroup_size;
}

void loom_llvmir_target_profile_storage_initialize_from_bundle(
    const loom_target_bundle_t* bundle,
    const loom_llvmir_target_profile_t* projected_profile,
    loom_llvmir_target_profile_storage_t* out_storage) {
  *out_storage = (loom_llvmir_target_profile_storage_t){0};
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  const bool use_projected_env =
      projected_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL;
  const loom_llvmir_target_env_t* projected_env = projected_profile->target_env;

  out_storage->target_env = (loom_llvmir_target_env_t){
      .name = use_projected_env ? projected_env->name : snapshot->name,
      .target_triple = projected_env->target_triple,
      .data_layout = projected_env->data_layout,
      .object_format = use_projected_env
                           ? projected_env->object_format
                           : loom_llvmir_object_format_from_artifact(
                                 snapshot->artifact_format),
      .default_pointer_bitwidth = use_projected_env
                                      ? projected_env->default_pointer_bitwidth
                                      : snapshot->default_pointer_bitwidth,
      .index_bitwidth = use_projected_env ? projected_env->index_bitwidth
                                          : snapshot->index_bitwidth,
      .offset_bitwidth = use_projected_env ? projected_env->offset_bitwidth
                                           : snapshot->offset_bitwidth,
      .address_spaces =
          {
              .generic = use_projected_env
                             ? projected_env->address_spaces.generic
                             : snapshot->memory_spaces.generic,
              .global = use_projected_env ? projected_env->address_spaces.global
                                          : snapshot->memory_spaces.global,
              .local = use_projected_env ? projected_env->address_spaces.local
                                         : snapshot->memory_spaces.workgroup,
              .constant = use_projected_env
                              ? projected_env->address_spaces.constant
                              : snapshot->memory_spaces.constant,
              .private_memory =
                  use_projected_env
                      ? projected_env->address_spaces.private_memory
                      : snapshot->memory_spaces.private_memory,
              .buffer_resource =
                  use_projected_env
                      ? projected_env->address_spaces.buffer_resource
                      : snapshot->memory_spaces.descriptor,
          },
  };

  out_storage->profile = (loom_llvmir_target_profile_t){
      .name = projected_profile->name,
      .target_env = &out_storage->target_env,
      .target_cpu = projected_profile->target_cpu,
      .target_features = projected_profile->target_features,
      .x86_packed_dot_feature_bits =
          projected_profile->x86_packed_dot_feature_bits,
      .exported_linkage =
          projected_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL
              ? projected_profile->exported_linkage
              : loom_llvmir_linkage_from_target(export_plan->linkage),
  };

  if (export_plan->abi_kind == LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT;
    return;
  }
  if (export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    uint32_t flat_workgroup_size_min = 0;
    uint32_t flat_workgroup_size_max = 0;
    loom_llvmir_resolve_hal_kernel_flat_workgroup_range(
        snapshot, &export_plan->hal_kernel, &flat_workgroup_size_min,
        &flat_workgroup_size_max);
    if (flat_workgroup_size_min == 0 || flat_workgroup_size_max == 0) {
      flat_workgroup_size_min =
          projected_profile->kernel.flat_workgroup_size_min;
      flat_workgroup_size_max =
          projected_profile->kernel.flat_workgroup_size_max;
    }
    const uint32_t buffer_resource_flags =
        export_plan->hal_kernel.buffer_resource_flags != 0
            ? export_plan->hal_kernel.buffer_resource_flags
            : projected_profile->kernel.binding_resource_flags;
    out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_KERNEL;
    out_storage->profile.kernel = projected_profile->kernel;
    out_storage->profile.kernel.required_workgroup_size =
        (loom_llvmir_workgroup_size_t){
            .x = export_plan->hal_kernel.required_workgroup_size.x,
            .y = export_plan->hal_kernel.required_workgroup_size.y,
            .z = export_plan->hal_kernel.required_workgroup_size.z,
        };
    out_storage->profile.kernel.flat_workgroup_size_min =
        flat_workgroup_size_min;
    out_storage->profile.kernel.flat_workgroup_size_max =
        flat_workgroup_size_max;
    out_storage->profile.kernel.subgroup_size =
        snapshot->subgroup_size != 0 ? snapshot->subgroup_size
                                     : projected_profile->kernel.subgroup_size;
    out_storage->profile.kernel.binding_resource_flags = buffer_resource_flags;
  }
}

static iree_status_t loom_llvmir_target_profile_append_llc_argument(
    iree_string_view_t prefix, iree_string_view_t value,
    loom_llvmir_target_profile_llc_arguments_t* arguments) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  if (arguments->count >= IREE_ARRAYSIZE(arguments->values)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM llc argument storage is too small");
  }
  char* storage = arguments->storage[arguments->count];
  iree_host_size_t storage_length =
      IREE_ARRAYSIZE(arguments->storage[arguments->count]);
  int length = snprintf(storage, storage_length, "%.*s%.*s", (int)prefix.size,
                        prefix.data, (int)value.size, value.data);
  if (length <= 0 || (iree_host_size_t)length >= storage_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM llc argument exceeds storage");
  }
  arguments->values[arguments->count] =
      iree_make_string_view(storage, (iree_host_size_t)length);
  ++arguments->count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_profile_llc_arguments(
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_target_profile_llc_arguments_t* out_arguments) {
  *out_arguments = (loom_llvmir_target_profile_llc_arguments_t){0};
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mtriple="), profile->target_env->target_triple, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mcpu="), profile->target_cpu, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mattr="), profile->target_features, out_arguments));
  return iree_ok_status();
}

void loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t* out_attr_count) {
  IREE_ASSERT_ARGUMENT(profile);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(out_attr_count);
  const iree_host_size_t attr_count =
      profile->kernel.binding_parameter_attr_count;
  IREE_ASSERT_LE(attr_count,
                 LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT);
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    attrs[i] = profile->kernel.binding_parameter_attrs[i];
  }
  *out_attr_count = attr_count;
}

static iree_string_view_t loom_llvmir_target_profile_flat_workgroup_size_attr(
    const loom_llvmir_target_profile_t* profile, char* storage,
    iree_host_size_t storage_capacity) {
  int length = snprintf(storage, storage_capacity, "%u,%u",
                        profile->kernel.flat_workgroup_size_min,
                        profile->kernel.flat_workgroup_size_max);
  return iree_make_string_view(storage, (iree_host_size_t)length);
}

iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id) {
  *out_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  char flat_workgroup_size_storage[32];
  iree_string_view_t flat_workgroup_size =
      loom_llvmir_target_profile_flat_workgroup_size_attr(
          profile, flat_workgroup_size_storage,
          IREE_ARRAYSIZE(flat_workgroup_size_storage));
  loom_llvmir_attr_t attrs[3];
  iree_host_size_t attr_count = 0;
  if (iree_any_bit_set(profile->kernel.flags,
                       LOOM_LLVMIR_KERNEL_PROFILE_FLAG_ALWAYSINLINE)) {
    attrs[attr_count++] = (loom_llvmir_attr_t){
        .kind = LOOM_LLVMIR_ATTR_ALWAYSINLINE,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    };
  }
  if (!iree_string_view_is_empty(
          profile->kernel.flat_workgroup_size_attr_name)) {
    attrs[attr_count++] = (loom_llvmir_attr_t){
        .kind = LOOM_LLVMIR_ATTR_STRING_KEY_VALUE,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
        .key = profile->kernel.flat_workgroup_size_attr_name,
        .string_value = flat_workgroup_size,
    };
  }
  if (!iree_string_view_is_empty(
          profile->kernel.uniform_workgroup_size_attr_name)) {
    attrs[attr_count++] = (loom_llvmir_attr_t){
        .kind = LOOM_LLVMIR_ATTR_STRING_KEY,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
        .key = profile->kernel.uniform_workgroup_size_attr_name,
    };
  }
  return loom_llvmir_module_add_attr_group(module, attrs, attr_count,
                                           out_group_id);
}

iree_status_t loom_llvmir_target_profile_attach_kernel_metadata(
    loom_llvmir_function_t* function,
    const loom_llvmir_target_profile_t* profile) {
  const loom_target_workgroup_size_t required_workgroup_size = {
      .x = profile->kernel.required_workgroup_size.x,
      .y = profile->kernel.required_workgroup_size.y,
      .z = profile->kernel.required_workgroup_size.z,
  };
  if (loom_target_workgroup_size_is_empty(&required_workgroup_size) ||
      iree_string_view_is_empty(
          profile->kernel.required_workgroup_size_metadata_name)) {
    return iree_ok_status();
  }
  int32_t workgroup_size_values[] = {
      (int32_t)profile->kernel.required_workgroup_size.x,
      (int32_t)profile->kernel.required_workgroup_size.y,
      (int32_t)profile->kernel.required_workgroup_size.z,
  };
  loom_llvmir_metadata_id_t workgroup_size = LOOM_LLVMIR_METADATA_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_metadata_i32_tuple(
      loom_llvmir_function_module(function),
      &(loom_llvmir_metadata_i32_tuple_t){
          .values = workgroup_size_values,
          .value_count = IREE_ARRAYSIZE(workgroup_size_values),
      },
      &workgroup_size));
  return loom_llvmir_function_add_metadata_attachment(
      function,
      &(loom_llvmir_metadata_attachment_t){
          .name = profile->kernel.required_workgroup_size_metadata_name,
          .metadata_id = workgroup_size,
      });
}
