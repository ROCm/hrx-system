// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/target_low_registry_manifest.h"

#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_status_t
loom_check_target_low_manifest_write_descriptor_set_summary(
    loom_output_stream_t* stream,
    const loom_low_descriptor_set_t* descriptor_set) {
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  iree_string_view_t target = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->target_key_string_offset);
  iree_string_view_t feature_namespace = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->feature_key_string_offset);

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("key"), key));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("target"), target));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("feature_namespace"), feature_namespace));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("abi_version"), descriptor_set->abi_version));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("generator_version"),
                                          descriptor_set->generator_version));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("table_counts")));
  loom_json_object_writer_t table_counts;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &table_counts));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("descriptors"), descriptor_set->descriptor_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("descriptor_refs"),
      descriptor_set->descriptor_ref_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("operands"), descriptor_set->operand_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("immediates"), descriptor_set->immediate_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("enum_domains"),
      descriptor_set->enum_domain_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("enum_values"), descriptor_set->enum_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("effects"), descriptor_set->effect_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("constraints"), descriptor_set->constraint_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("reg_classes"), descriptor_set->reg_class_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("reg_class_alts"),
      descriptor_set->reg_class_alt_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("schedule_classes"),
      descriptor_set->schedule_class_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("issue_uses"), descriptor_set->issue_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("resources"), descriptor_set->resource_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("hazards"), descriptor_set->hazard_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("pressure_deltas"),
      descriptor_set->pressure_delta_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &table_counts, IREE_SV("feature_mask_words"),
      descriptor_set->feature_mask_word_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&table_counts));
  return loom_json_object_end(&object);
}

iree_status_t loom_check_target_low_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(&registry->registry);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("descriptor_set_count"), descriptor_set_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("descriptor_sets")));
  loom_json_array_writer_t descriptor_sets;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &descriptor_sets));
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&descriptor_sets));
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(&registry->registry, i);
    IREE_RETURN_IF_ERROR(
        loom_check_target_low_manifest_write_descriptor_set_summary(
            &stream, descriptor_set));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&descriptor_sets));
  return loom_json_object_end(&object);
}
