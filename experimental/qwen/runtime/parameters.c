// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/parameters.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/printf.h"

// Parameter subranges are aligned for 128-bit vector memory accesses and every
// current HAL allocation base. All fixed encoded lengths are multiples of this
// alignment, so the resident slab introduces no inter-parameter padding.
#define QWEN_PARAMETER_BASE_ALIGNMENT 256ull

typedef enum qwen_layer_storage_selector_e {
  QWEN_LAYER_STORAGE_SELECTOR_NONE = 0,
  QWEN_LAYER_STORAGE_SELECTOR_VALUE = 1,
  QWEN_LAYER_STORAGE_SELECTOR_DOWN = 2,
} qwen_layer_storage_selector_t;

typedef struct qwen_layer_parameter_spec_t {
  // GGUF suffix following "blk.<layer>.".
  const char* suffix;
  // Required encoded length, or Q4_K length when an alternative is present.
  iree_device_size_t encoded_length;
  // Optional Q6_K encoded length.
  iree_device_size_t alternate_encoded_length;
  // Byte offset of the destination span in qwen_layer_parameters_t.
  iree_host_size_t span_offset;
  // Layer storage selection updated by this entry.
  qwen_layer_storage_selector_t storage_selector;
} qwen_layer_parameter_spec_t;

static const qwen_layer_parameter_spec_t qwen_layer_parameter_specs[] = {
    {
        .suffix = "attn_norm.weight",
        .encoded_length = 8192,
        .span_offset = offsetof(qwen_layer_parameters_t, attention_norm),
    },
    {
        .suffix = "attn_q.weight",
        .encoded_length = 4718592,
        .span_offset = offsetof(qwen_layer_parameters_t, query),
    },
    {
        .suffix = "attn_k.weight",
        .encoded_length = 589824,
        .span_offset = offsetof(qwen_layer_parameters_t, key),
    },
    {
        .suffix = "attn_v.weight",
        .encoded_length = 589824,
        .alternate_encoded_length = 860160,
        .span_offset = offsetof(qwen_layer_parameters_t, value),
        .storage_selector = QWEN_LAYER_STORAGE_SELECTOR_VALUE,
    },
    {
        .suffix = "attn_q_norm.weight",
        .encoded_length = 512,
        .span_offset = offsetof(qwen_layer_parameters_t, query_norm),
    },
    {
        .suffix = "attn_k_norm.weight",
        .encoded_length = 512,
        .span_offset = offsetof(qwen_layer_parameters_t, key_norm),
    },
    {
        .suffix = "attn_output.weight",
        .encoded_length = 4718592,
        .span_offset = offsetof(qwen_layer_parameters_t, attention_output),
    },
    {
        .suffix = "ffn_norm.weight",
        .encoded_length = 8192,
        .span_offset = offsetof(qwen_layer_parameters_t, feed_forward_norm),
    },
    {
        .suffix = "ffn_gate_inp.weight",
        .encoded_length = 1048576,
        .span_offset = offsetof(qwen_layer_parameters_t, router),
    },
    {
        .suffix = "ffn_gate_exps.weight",
        .encoded_length = 113246208,
        .span_offset = offsetof(qwen_layer_parameters_t, expert_gate),
    },
    {
        .suffix = "ffn_up_exps.weight",
        .encoded_length = 113246208,
        .span_offset = offsetof(qwen_layer_parameters_t, expert_up),
    },
    {
        .suffix = "ffn_down_exps.weight",
        .encoded_length = 113246208,
        .alternate_encoded_length = 165150720,
        .span_offset = offsetof(qwen_layer_parameters_t, expert_down),
        .storage_selector = QWEN_LAYER_STORAGE_SELECTOR_DOWN,
    },
};

_Static_assert(
    IREE_ARRAYSIZE(qwen_layer_parameter_specs) == 12,
    "fixed Qwen layer parameter count must preserve the 579-entry schema");
_Static_assert(
    1 + QWEN_MODEL_LAYER_COUNT * IREE_ARRAYSIZE(qwen_layer_parameter_specs) +
            2 ==
        QWEN_PARAMETER_COUNT,
    "top-level and layer parameter counts must cover the fixed index");

#define QWEN_LAYER_PARAMETER_COUNT IREE_ARRAYSIZE(qwen_layer_parameter_specs)
#define QWEN_FIRST_LAYER_PARAMETER_ORDINAL 1
#define QWEN_OUTPUT_NORM_PARAMETER_ORDINAL \
  (QWEN_FIRST_LAYER_PARAMETER_ORDINAL +    \
   QWEN_MODEL_LAYER_COUNT * QWEN_LAYER_PARAMETER_COUNT)
#define QWEN_OUTPUT_PARAMETER_ORDINAL (QWEN_OUTPUT_NORM_PARAMETER_ORDINAL + 1)

typedef struct qwen_parameter_descriptor_t {
  // Byte offset of the typed span within qwen_parameter_layout_t.
  iree_host_size_t span_offset;
  // Layer ordinal, or QWEN_MODEL_LAYER_COUNT for top-level parameters.
  iree_host_size_t layer;
  // Layer parameter specification, or NULL for top-level parameters.
  const qwen_layer_parameter_spec_t* layer_spec;
  // Required encoded length, or Q4_K length when an alternative is present.
  iree_device_size_t encoded_length;
  // Optional Q6_K encoded length.
  iree_device_size_t alternate_encoded_length;
} qwen_parameter_descriptor_t;

static qwen_parameter_descriptor_t
qwen_parameter_descriptor_for_logical_ordinal(
    iree_host_size_t logical_ordinal) {
  IREE_ASSERT(logical_ordinal < QWEN_PARAMETER_COUNT);
  if (logical_ordinal == 0) {
    return (qwen_parameter_descriptor_t){
        .span_offset = offsetof(qwen_parameter_layout_t, token_embedding),
        .layer = QWEN_MODEL_LAYER_COUNT,
        .encoded_length = 175030272,
    };
  }
  if (logical_ordinal < QWEN_OUTPUT_NORM_PARAMETER_ORDINAL) {
    const iree_host_size_t layer_ordinal =
        logical_ordinal - QWEN_FIRST_LAYER_PARAMETER_ORDINAL;
    const iree_host_size_t layer = layer_ordinal / QWEN_LAYER_PARAMETER_COUNT;
    const qwen_layer_parameter_spec_t* spec =
        &qwen_layer_parameter_specs[layer_ordinal % QWEN_LAYER_PARAMETER_COUNT];
    return (qwen_parameter_descriptor_t){
        .span_offset = offsetof(qwen_parameter_layout_t, layers) +
                       layer * sizeof(qwen_layer_parameters_t) +
                       spec->span_offset,
        .layer = layer,
        .layer_spec = spec,
        .encoded_length = spec->encoded_length,
        .alternate_encoded_length = spec->alternate_encoded_length,
    };
  }
  if (logical_ordinal == QWEN_OUTPUT_NORM_PARAMETER_ORDINAL) {
    return (qwen_parameter_descriptor_t){
        .span_offset = offsetof(qwen_parameter_layout_t, output_norm),
        .layer = QWEN_MODEL_LAYER_COUNT,
        .encoded_length = 8192,
    };
  }
  return (qwen_parameter_descriptor_t){
      .span_offset = offsetof(qwen_parameter_layout_t, output),
      .layer = QWEN_MODEL_LAYER_COUNT,
      .encoded_length = 255252480,
  };
}

static iree_status_t qwen_parameter_format_layer_key(
    iree_host_size_t layer, const char* suffix,
    char key_storage[QWEN_PARAMETER_KEY_CAPACITY],
    iree_string_view_t* out_key) {
  int key_length = iree_snprintf(key_storage, QWEN_PARAMETER_KEY_CAPACITY,
                                 "blk.%" PRIhsz ".%s", layer, suffix);
  if (IREE_UNLIKELY(key_length < 0 ||
                    key_length >= QWEN_PARAMETER_KEY_CAPACITY)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen parameter key exceeded %d bytes",
                            QWEN_PARAMETER_KEY_CAPACITY);
  }
  *out_key = iree_make_string_view(key_storage, (iree_host_size_t)key_length);
  return iree_ok_status();
}

static iree_status_t qwen_parameter_format_logical_key(
    iree_host_size_t logical_ordinal,
    char key_storage[QWEN_PARAMETER_KEY_CAPACITY],
    iree_string_view_t* out_key) {
  if (logical_ordinal == 0) {
    *out_key = IREE_SV("token_embd.weight");
    return iree_ok_status();
  }
  if (logical_ordinal < QWEN_OUTPUT_NORM_PARAMETER_ORDINAL) {
    const iree_host_size_t layer_ordinal =
        logical_ordinal - QWEN_FIRST_LAYER_PARAMETER_ORDINAL;
    const iree_host_size_t layer = layer_ordinal / QWEN_LAYER_PARAMETER_COUNT;
    const qwen_layer_parameter_spec_t* spec =
        &qwen_layer_parameter_specs[layer_ordinal % QWEN_LAYER_PARAMETER_COUNT];
    return qwen_parameter_format_layer_key(layer, spec->suffix, key_storage,
                                           out_key);
  }
  if (logical_ordinal == QWEN_OUTPUT_NORM_PARAMETER_ORDINAL) {
    *out_key = IREE_SV("output_norm.weight");
  } else {
    *out_key = IREE_SV("output.weight");
  }
  return iree_ok_status();
}

static iree_status_t qwen_parameter_resolve_logical_ordinal(
    iree_string_view_t key, iree_host_size_t* out_logical_ordinal) {
  if (iree_string_view_equal(key, IREE_SV("token_embd.weight"))) {
    *out_logical_ordinal = 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("output_norm.weight"))) {
    *out_logical_ordinal = QWEN_OUTPUT_NORM_PARAMETER_ORDINAL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("output.weight"))) {
    *out_logical_ordinal = QWEN_OUTPUT_PARAMETER_ORDINAL;
    return iree_ok_status();
  }

  iree_string_view_t layer_key = key;
  if (iree_string_view_consume_prefix(&layer_key, IREE_SV("blk."))) {
    iree_string_view_t layer_string = iree_string_view_empty();
    iree_string_view_t suffix = iree_string_view_empty();
    iree_string_view_split(layer_key, '.', &layer_string, &suffix);
    uint32_t layer = 0;
    if (iree_string_view_atoi_uint32(layer_string, &layer) &&
        layer < QWEN_MODEL_LAYER_COUNT) {
      for (iree_host_size_t i = 0; i < QWEN_LAYER_PARAMETER_COUNT; ++i) {
        if (iree_string_view_equal(
                suffix,
                iree_make_cstring_view(qwen_layer_parameter_specs[i].suffix))) {
          *out_logical_ordinal =
              QWEN_FIRST_LAYER_PARAMETER_ORDINAL +
              (iree_host_size_t)layer * QWEN_LAYER_PARAMETER_COUNT + i;
          return iree_ok_status();
        }
      }
    }
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected Qwen parameter '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t qwen_parameter_pack_index_entry(
    const iree_io_parameter_index_entry_t* entry,
    const qwen_parameter_descriptor_t* descriptor,
    iree_device_size_t* inout_cursor,
    qwen_parameter_statistics_t* inout_statistics,
    qwen_parameter_span_t* out_span, qwen_quantized_storage_t* out_storage) {
  qwen_quantized_storage_t storage = QWEN_QUANTIZED_STORAGE_Q4_K;
  if (entry->length == descriptor->alternate_encoded_length &&
      descriptor->alternate_encoded_length != 0) {
    storage = QWEN_QUANTIZED_STORAGE_Q6_K;
  } else if (entry->length != descriptor->encoded_length) {
    if (descriptor->alternate_encoded_length != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen parameter '%.*s' has encoded length %" PRIu64
          "; expected %" PRIu64 " (Q4_K) or %" PRIu64 " (Q6_K)",
          (int)entry->key.size, entry->key.data, entry->length,
          descriptor->encoded_length, descriptor->alternate_encoded_length);
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen parameter '%.*s' has encoded length %" PRIu64
                            "; expected %" PRIu64,
                            (int)entry->key.size, entry->key.data,
                            entry->length, descriptor->encoded_length);
  }

  const iree_device_size_t aligned_cursor =
      iree_align_uint64(*inout_cursor, QWEN_PARAMETER_BASE_ALIGNMENT);
  inout_statistics->parameter_padding_bytes += aligned_cursor - *inout_cursor;
  *out_span = (qwen_parameter_span_t){
      .offset = aligned_cursor,
      .length = (iree_device_size_t)entry->length,
  };
  *inout_cursor = aligned_cursor + entry->length;
  inout_statistics->encoded_parameter_bytes += entry->length;
  if (out_storage) *out_storage = storage;
  return iree_ok_status();
}

iree_status_t qwen_parameter_layout_build(
    iree_io_parameter_index_t* parameter_index,
    qwen_parameter_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(parameter_index);
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  const iree_host_size_t parameter_count =
      iree_io_parameter_index_count(parameter_index);
  if (parameter_count != QWEN_PARAMETER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen parameter index has %" PRIhsz
                            " entries; expected exactly %d",
                            parameter_count, QWEN_PARAMETER_COUNT);
  }

  iree_device_size_t cursor = 0;
  qwen_parameter_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));

  bool seen_logical_ordinals[QWEN_PARAMETER_COUNT] = {false};
  qwen_quantized_storage_t value_storage[QWEN_MODEL_LAYER_COUNT] = {0};
  qwen_quantized_storage_t down_storage[QWEN_MODEL_LAYER_COUNT] = {0};
  for (iree_host_size_t source_ordinal = 0;
       source_ordinal < QWEN_PARAMETER_COUNT; ++source_ordinal) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(
        iree_io_parameter_index_get(parameter_index, source_ordinal, &entry));

    iree_host_size_t logical_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        qwen_parameter_resolve_logical_ordinal(entry->key, &logical_ordinal));
    if (seen_logical_ordinals[logical_ordinal]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate Qwen parameter '%.*s'",
                              (int)entry->key.size, entry->key.data);
    }
    seen_logical_ordinals[logical_ordinal] = true;
    out_layout->source_parameter_ordinals[source_ordinal] =
        (uint16_t)logical_ordinal;

    const qwen_parameter_descriptor_t descriptor =
        qwen_parameter_descriptor_for_logical_ordinal(logical_ordinal);
    qwen_parameter_span_t* span =
        (qwen_parameter_span_t*)((uint8_t*)out_layout + descriptor.span_offset);
    qwen_quantized_storage_t* storage = NULL;
    if (descriptor.layer_spec && descriptor.layer_spec->storage_selector ==
                                     QWEN_LAYER_STORAGE_SELECTOR_VALUE) {
      storage = &value_storage[descriptor.layer];
    } else if (descriptor.layer_spec &&
               descriptor.layer_spec->storage_selector ==
                   QWEN_LAYER_STORAGE_SELECTOR_DOWN) {
      storage = &down_storage[descriptor.layer];
    }
    IREE_RETURN_IF_ERROR(qwen_parameter_pack_index_entry(
        entry, &descriptor, &cursor, &statistics, span, storage));
  }

  for (iree_host_size_t layer = 0; layer < QWEN_MODEL_LAYER_COUNT; ++layer) {
    if (value_storage[layer] != down_storage[layer]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen layer %" PRIhsz
          " uses different storage for attn_v and ffn_down_exps",
          layer);
    }
    out_layout->layers[layer].value_and_down_storage = value_storage[layer];
  }

  const iree_device_size_t auxiliary_offset =
      iree_align_uint64(cursor, QWEN_PARAMETER_BASE_ALIGNMENT);
  statistics.parameter_padding_bytes += auxiliary_offset - cursor;
  out_layout->rope_inverse_frequencies = (qwen_parameter_span_t){
      .offset = auxiliary_offset,
      .length = QWEN_MODEL_ROPE_FREQUENCY_COUNT * sizeof(float),
  };
  statistics.immutable_auxiliary_bytes =
      out_layout->rope_inverse_frequencies.length;
  statistics.allocation_bytes =
      auxiliary_offset + out_layout->rope_inverse_frequencies.length;
  out_layout->statistics = statistics;
  return iree_ok_status();
}

iree_status_t qwen_parameter_layout_enumerate(
    const qwen_parameter_layout_t* layout, iree_host_size_t index,
    char key_storage[QWEN_PARAMETER_KEY_CAPACITY], iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(key_storage);
  IREE_ASSERT_ARGUMENT(out_key);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_key = iree_string_view_empty();
  memset(out_span, 0, sizeof(*out_span));

  if (index >= QWEN_PARAMETER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen parameter ordinal %" PRIhsz
                            " out of range; expected [0, %d)",
                            index, QWEN_PARAMETER_COUNT);
  }

  const iree_host_size_t logical_ordinal =
      layout->source_parameter_ordinals[index];
  const qwen_parameter_descriptor_t descriptor =
      qwen_parameter_descriptor_for_logical_ordinal(logical_ordinal);
  IREE_RETURN_IF_ERROR(
      qwen_parameter_format_logical_key(logical_ordinal, key_storage, out_key));
  const qwen_parameter_span_t* parameter_span =
      (const qwen_parameter_span_t*)((const uint8_t*)layout +
                                     descriptor.span_offset);

  *out_span = (iree_io_parameter_span_t){
      .parameter_offset = 0,
      .buffer_offset = parameter_span->offset,
      .length = parameter_span->length,
  };
  return iree_ok_status();
}

void qwen_parameter_calculate_rope_inverse_frequencies(
    float out_values[QWEN_MODEL_ROPE_FREQUENCY_COUNT]) {
  IREE_ASSERT_ARGUMENT(out_values);
  for (iree_host_size_t i = 0; i < QWEN_MODEL_ROPE_FREQUENCY_COUNT; ++i) {
    const float exponent =
        -(2.0f * (float)i) / (2.0f * QWEN_MODEL_ROPE_FREQUENCY_COUNT);
    out_values[i] = powf(1000000.0f, exponent);
  }
}
