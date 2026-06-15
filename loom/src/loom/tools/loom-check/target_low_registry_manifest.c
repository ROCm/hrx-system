// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/target_low_registry_manifest.h"

#include <inttypes.h>

#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_status_t loom_check_target_low_manifest_write_string_field(
    loom_output_stream_t* stream, const char* field_name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\":"));
  return loom_json_write_escaped_string(stream, value);
}

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

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(
      loom_check_target_low_manifest_write_string_field(stream, "key", key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_check_target_low_manifest_write_string_field(
      stream, "target", target));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_check_target_low_manifest_write_string_field(
      stream, "feature_namespace", feature_namespace));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"abi_version\":%" PRIu32 ",\"generator_version\":%" PRIu32
      ",\"table_counts\":{\"descriptors\":%" PRIu32
      ",\"descriptor_refs\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"enum_domains\":%" PRIu32
      ",\"enum_values\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"reg_class_alts\":%" PRIu32 ",\"schedule_classes\":%" PRIu32
      ",\"issue_uses\":%" PRIu32 ",\"resources\":%" PRIu32
      ",\"hazards\":%" PRIu32 ",\"pressure_deltas\":%" PRIu32
      ",\"feature_mask_words\":%" PRIu32 "}}",
      descriptor_set->abi_version, descriptor_set->generator_version,
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->operand_count, descriptor_set->immediate_count,
      descriptor_set->enum_domain_count, descriptor_set->enum_value_count,
      descriptor_set->effect_count, descriptor_set->constraint_count,
      descriptor_set->reg_class_count, descriptor_set->reg_class_alt_count,
      descriptor_set->schedule_class_count, descriptor_set->issue_use_count,
      descriptor_set->resource_count, descriptor_set->hazard_count,
      descriptor_set->pressure_delta_count,
      descriptor_set->feature_mask_word_count));
  return iree_ok_status();
}

iree_status_t loom_check_target_low_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(&registry->registry);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, "{\"descriptor_set_count\":%" PRIhsz ",\"descriptor_sets\":[",
      descriptor_set_count));
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(&registry->registry, i);
    IREE_RETURN_IF_ERROR(
        loom_check_target_low_manifest_write_descriptor_set_summary(
            &stream, descriptor_set));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
