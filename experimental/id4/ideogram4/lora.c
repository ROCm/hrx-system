// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/lora.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/json.h"

typedef enum id4_ideogram4_lora_dimension_e {
  ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN = 0,
  ID4_IDEOGRAM4_LORA_DIMENSION_INTERMEDIATE,
  ID4_IDEOGRAM4_LORA_DIMENSION_ADALN,
} id4_ideogram4_lora_dimension_t;

typedef struct id4_ideogram4_lora_site_t {
  // External target path between the layer ordinal and LoRA suffix.
  iree_string_view_t path;
  // Model dimension used for the down-projection input.
  id4_ideogram4_lora_dimension_t input_dimension;
  // Multiplier applied to the selected input dimension.
  uint32_t input_multiplier;
  // Model dimension used for the up-projection output.
  id4_ideogram4_lora_dimension_t output_dimension;
  // Multiplier applied to the selected output dimension.
  uint32_t output_multiplier;
} id4_ideogram4_lora_site_t;

static const id4_ideogram4_lora_site_t id4_ideogram4_lora_sites[] = {
    {
        .path = IREE_SVL("adaln_modulation"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_ADALN,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .output_multiplier = 4,
    },
    {
        .path = IREE_SVL("attention.qkv"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .output_multiplier = 3,
    },
    {
        .path = IREE_SVL("attention.o"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .output_multiplier = 1,
    },
    {
        .path = IREE_SVL("feed_forward.w1"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_INTERMEDIATE,
        .output_multiplier = 1,
    },
    {
        .path = IREE_SVL("feed_forward.w3"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_INTERMEDIATE,
        .output_multiplier = 1,
    },
    {
        .path = IREE_SVL("feed_forward.w2"),
        .input_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_INTERMEDIATE,
        .input_multiplier = 1,
        .output_dimension = ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN,
        .output_multiplier = 1,
    },
};

typedef enum id4_ideogram4_lora_parameter_half_e {
  ID4_IDEOGRAM4_LORA_PARAMETER_HALF_DOWN = 0,
  ID4_IDEOGRAM4_LORA_PARAMETER_HALF_UP,
} id4_ideogram4_lora_parameter_half_t;

typedef struct id4_ideogram4_lora_tensor_t {
  // Indexed parameter entry borrowed during import.
  const iree_io_parameter_index_entry_t* entry;
  // First physical tensor dimension.
  uint32_t rows;
  // Second physical tensor dimension.
  uint32_t columns;
} id4_ideogram4_lora_tensor_t;

typedef struct id4_ideogram4_lora_pair_t {
  // BF16 down-projection tensor [rank, input].
  id4_ideogram4_lora_tensor_t down;
  // BF16 up-projection tensor [output, rank].
  id4_ideogram4_lora_tensor_t up;
  // Validated base-linear input feature count.
  uint32_t input_size;
  // Validated base-linear output feature count.
  uint32_t output_size;
} id4_ideogram4_lora_pair_t;

struct id4_ideogram4_lora_t {
  // Reference count for shared immutable catalog ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this packed catalog allocation.
  iree_allocator_t host_allocator;
  // Provider scope copied from the import options.
  iree_string_view_t source_scope;
  // Number of entries in |targets|.
  iree_host_size_t target_count;
  // Validated target table stored within the packed allocation.
  id4_ideogram4_lora_target_t* targets;
};

static uint32_t id4_ideogram4_lora_model_dimension(
    const id4_ideogram4_dit_model_config_t* model,
    id4_ideogram4_lora_dimension_t dimension) {
  switch (dimension) {
    case ID4_IDEOGRAM4_LORA_DIMENSION_HIDDEN:
      return model->hidden_size;
    case ID4_IDEOGRAM4_LORA_DIMENSION_INTERMEDIATE:
      return model->intermediate_size;
    case ID4_IDEOGRAM4_LORA_DIMENSION_ADALN:
      return model->adaln_size;
    default:
      return 0;
  }
}

static iree_status_t id4_ideogram4_lora_site_dimensions(
    const id4_ideogram4_dit_model_config_t* model,
    const id4_ideogram4_lora_site_t* site, uint32_t* out_input_size,
    uint32_t* out_output_size) {
  uint64_t input_size = (uint64_t)id4_ideogram4_lora_model_dimension(
                            model, site->input_dimension) *
                        site->input_multiplier;
  uint64_t output_size = (uint64_t)id4_ideogram4_lora_model_dimension(
                             model, site->output_dimension) *
                         site->output_multiplier;
  if (input_size == 0 || input_size > UINT32_MAX || output_size == 0 ||
      output_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA target %.*s has invalid model dimensions input=%" PRIu64
        " output=%" PRIu64,
        (int)site->path.size, site->path.data, input_size, output_size);
  }
  *out_input_size = (uint32_t)input_size;
  *out_output_size = (uint32_t)output_size;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_parse_parameter_key(
    iree_string_view_t key, const id4_ideogram4_dit_model_config_t* model,
    iree_host_size_t* out_pair_index,
    id4_ideogram4_lora_parameter_half_t* out_half) {
  iree_string_view_t remaining = key;
  if (!iree_string_view_consume_prefix(&remaining,
                                       IREE_SV("diffusion_model.layers."))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Ideogram 4 LoRA parameter %.*s",
                            (int)key.size, key.data);
  }

  iree_string_view_t layer_string = iree_string_view_empty();
  iree_string_view_t target_string = iree_string_view_empty();
  if (iree_string_view_split(remaining, '.', &layer_string, &target_string) <
      0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed Ideogram 4 LoRA parameter %.*s",
                            (int)key.size, key.data);
  }
  uint32_t layer_ordinal = 0;
  if ((layer_string.size > 1 &&
       iree_string_view_starts_with_char(layer_string, '0')) ||
      !iree_string_view_atoi_uint32(layer_string, &layer_ordinal) ||
      layer_ordinal >= model->layer_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter %.*s has invalid layer ordinal",
                            (int)key.size, key.data);
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(id4_ideogram4_lora_sites);
       ++i) {
    iree_string_view_t suffix = target_string;
    if (!iree_string_view_consume_prefix(&suffix,
                                         id4_ideogram4_lora_sites[i].path)) {
      continue;
    }
    if (iree_string_view_equal(suffix, IREE_SV(".lora_A.weight"))) {
      *out_half = ID4_IDEOGRAM4_LORA_PARAMETER_HALF_DOWN;
    } else if (iree_string_view_equal(suffix, IREE_SV(".lora_B.weight"))) {
      *out_half = ID4_IDEOGRAM4_LORA_PARAMETER_HALF_UP;
    } else {
      continue;
    }
    *out_pair_index = (iree_host_size_t)layer_ordinal *
                          IREE_ARRAYSIZE(id4_ideogram4_lora_sites) +
                      i;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unrecognized Ideogram 4 LoRA target %.*s",
                          (int)key.size, key.data);
}

static iree_status_t id4_ideogram4_lora_parse_tensor_metadata(
    const iree_io_parameter_index_entry_t* entry, uint32_t* out_rows,
    uint32_t* out_columns) {
  if (iree_const_byte_span_is_empty(entry->metadata)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter %.*s is missing tensor metadata",
                            (int)entry->key.size, entry->key.data);
  }
  iree_string_view_t metadata = iree_make_string_view(
      (const char*)entry->metadata.data, entry->metadata.data_length);
  char dtype_buffer[16];
  iree_host_size_t dtype_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_lookup_string(
      metadata, IREE_SV("dtype"),
      iree_make_mutable_string_view(dtype_buffer, IREE_ARRAYSIZE(dtype_buffer)),
      &dtype_length));
  iree_string_view_t dtype = iree_make_string_view(dtype_buffer, dtype_length);
  if (!iree_string_view_equal(dtype, IREE_SV("BF16"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter %.*s has dtype %.*s; expected BF16",
                            (int)entry->key.size, entry->key.data,
                            (int)dtype.size, dtype.data);
  }

  iree_string_view_t shape = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(metadata, IREE_SV("shape"), &shape));
  iree_host_size_t rank = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(shape, &rank));
  if (rank != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter %.*s has rank %" PRIhsz
                            "; expected rank 2",
                            (int)entry->key.size, entry->key.data, rank);
  }
  uint64_t dimensions[2] = {0, 0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(dimensions); ++i) {
    iree_string_view_t dimension = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_array_get(shape, i, &dimension));
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(dimension, &dimensions[i]));
    if (dimensions[i] == 0 || dimensions[i] > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LoRA parameter %.*s dimension %" PRIhsz
                              " is out of range",
                              (int)entry->key.size, entry->key.data, i);
    }
  }
  if (dimensions[0] > UINT64_MAX / dimensions[1] ||
      dimensions[0] * dimensions[1] > UINT64_MAX / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA parameter %.*s byte length overflows",
                            (int)entry->key.size, entry->key.data);
  }
  uint64_t expected_byte_length = dimensions[0] * dimensions[1] * 2;
  if (entry->length != expected_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter %.*s has byte length %" PRIu64
                            "; expected %" PRIu64,
                            (int)entry->key.size, entry->key.data,
                            entry->length, expected_byte_length);
  }
  *out_rows = (uint32_t)dimensions[0];
  *out_columns = (uint32_t)dimensions[1];
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_validate_pair(
    const id4_ideogram4_dit_model_config_t* model, iree_host_size_t pair_index,
    id4_ideogram4_lora_pair_t* pair) {
  uint32_t layer_ordinal =
      (uint32_t)(pair_index / IREE_ARRAYSIZE(id4_ideogram4_lora_sites));
  const id4_ideogram4_lora_site_t* site =
      &id4_ideogram4_lora_sites[pair_index %
                                IREE_ARRAYSIZE(id4_ideogram4_lora_sites)];
  if (!pair->down.entry || !pair->up.entry) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA target layers.%" PRIu32 ".%.*s is missing its %s parameter",
        layer_ordinal, (int)site->path.size, site->path.data,
        pair->down.entry ? "up-projection" : "down-projection");
  }
  uint32_t input_size = 0;
  uint32_t output_size = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_site_dimensions(
      model, site, &input_size, &output_size));
  if (pair->down.columns != input_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA down parameter %.*s has %" PRIu32
                            " input features; expected %" PRIu32,
                            (int)pair->down.entry->key.size,
                            pair->down.entry->key.data, pair->down.columns,
                            input_size);
  }
  if (pair->up.rows != output_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA up parameter %.*s has %" PRIu32
                            " output features; expected %" PRIu32,
                            (int)pair->up.entry->key.size,
                            pair->up.entry->key.data, pair->up.rows,
                            output_size);
  }
  if (pair->down.rows != pair->up.columns) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA target layers.%" PRIu32
                            ".%.*s rank mismatch: down=%" PRIu32 " up=%" PRIu32,
                            layer_ordinal, (int)site->path.size,
                            site->path.data, pair->down.rows, pair->up.columns);
  }
  pair->input_size = input_size;
  pair->output_size = output_size;
  return iree_ok_status();
}

static iree_host_size_t id4_ideogram4_lora_base_key_length(
    iree_string_view_t down_parameter_key) {
  iree_string_view_t body = iree_string_view_strip_prefix(
      down_parameter_key, IREE_SV("diffusion_model."));
  body = iree_string_view_strip_suffix(body, IREE_SV(".lora_A.weight"));
  return body.size + sizeof(".weight") - 1;
}

static iree_string_view_t id4_ideogram4_lora_copy_string(
    iree_string_view_t source, char** inout_cursor) {
  char* target = *inout_cursor;
  memcpy(target, source.data, source.size);
  *inout_cursor += source.size;
  return iree_make_string_view(target, source.size);
}

static iree_string_view_t id4_ideogram4_lora_copy_base_key(
    iree_string_view_t down_parameter_key, char** inout_cursor) {
  iree_string_view_t body = iree_string_view_strip_prefix(
      down_parameter_key, IREE_SV("diffusion_model."));
  body = iree_string_view_strip_suffix(body, IREE_SV(".lora_A.weight"));
  char* start = *inout_cursor;
  id4_ideogram4_lora_copy_string(body, inout_cursor);
  id4_ideogram4_lora_copy_string(IREE_SV(".weight"), inout_cursor);
  return iree_make_string_view(start,
                               (iree_host_size_t)(*inout_cursor - start));
}

static void id4_ideogram4_lora_destroy(id4_ideogram4_lora_t* lora) {
  iree_allocator_free(lora->host_allocator, lora);
}

iree_status_t id4_ideogram4_lora_import(
    const id4_ideogram4_lora_import_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_lora_t** out_lora) {
  IREE_ASSERT_ARGUMENT(out_lora);
  *out_lora = NULL;
  if (!options || options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA import options are required");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LoRA import extension structures are not supported");
  }
  if (!options->parameter_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter index is required");
  }
  if (iree_string_view_is_empty(options->source_scope)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA parameter source scope is required");
  }
  if (options->model.layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA model layer count must be nonzero");
  }

  iree_host_size_t pair_capacity = 0;
  if (!iree_host_size_checked_mul(options->model.layer_count,
                                  IREE_ARRAYSIZE(id4_ideogram4_lora_sites),
                                  &pair_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA target table size overflow");
  }
  id4_ideogram4_lora_pair_t* pairs = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, pair_capacity, sizeof(pairs[0]), (void**)&pairs));
  memset(pairs, 0, pair_capacity * sizeof(pairs[0]));

  iree_host_size_t target_count = 0;
  iree_status_t status = iree_ok_status();
  iree_host_size_t entry_count =
      iree_io_parameter_index_count(options->parameter_index);
  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    status = iree_io_parameter_index_get(options->parameter_index, i, &entry);
    iree_host_size_t pair_index = 0;
    id4_ideogram4_lora_parameter_half_t half =
        ID4_IDEOGRAM4_LORA_PARAMETER_HALF_DOWN;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_lora_parse_parameter_key(
          entry->key, &options->model, &pair_index, &half);
    }
    id4_ideogram4_lora_tensor_t* tensor = NULL;
    if (iree_status_is_ok(status)) {
      tensor = half == ID4_IDEOGRAM4_LORA_PARAMETER_HALF_DOWN
                   ? &pairs[pair_index].down
                   : &pairs[pair_index].up;
      if (tensor->entry) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "LoRA parameter %.*s duplicates target half %.*s",
            (int)entry->key.size, entry->key.data, (int)tensor->entry->key.size,
            tensor->entry->key.data);
      }
    }
    if (iree_status_is_ok(status)) {
      tensor->entry = entry;
      status = id4_ideogram4_lora_parse_tensor_metadata(entry, &tensor->rows,
                                                        &tensor->columns);
    }
  }

  for (iree_host_size_t i = 0; i < pair_capacity && iree_status_is_ok(status);
       ++i) {
    if (!pairs[i].down.entry && !pairs[i].up.entry) continue;
    status = id4_ideogram4_lora_validate_pair(&options->model, i, &pairs[i]);
    if (iree_status_is_ok(status)) ++target_count;
  }
  if (iree_status_is_ok(status) && target_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LoRA parameter index contains no targets");
  }

  iree_host_size_t string_byte_length = options->source_scope.size;
  for (iree_host_size_t i = 0; i < pair_capacity && iree_status_is_ok(status);
       ++i) {
    if (!pairs[i].down.entry) continue;
    iree_host_size_t target_string_byte_length =
        id4_ideogram4_lora_base_key_length(pairs[i].down.entry->key);
    if (!iree_host_size_checked_add(target_string_byte_length,
                                    pairs[i].down.entry->key.size,
                                    &target_string_byte_length) ||
        !iree_host_size_checked_add(target_string_byte_length,
                                    pairs[i].up.entry->key.size,
                                    &target_string_byte_length) ||
        !iree_host_size_checked_add(string_byte_length,
                                    target_string_byte_length,
                                    &string_byte_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LoRA catalog string storage overflow");
    }
  }

  id4_ideogram4_lora_t* lora = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t target_offset = 0;
    iree_host_size_t string_offset = 0;
    iree_host_size_t allocation_size = 0;
    status = IREE_STRUCT_LAYOUT(
        sizeof(*lora), &allocation_size,
        IREE_STRUCT_FIELD(target_count, id4_ideogram4_lora_target_t,
                          &target_offset),
        IREE_STRUCT_FIELD(string_byte_length, char, &string_offset));
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(host_allocator, allocation_size, (void**)&lora);
    }
    if (iree_status_is_ok(status)) {
      memset(lora, 0, sizeof(*lora));
      iree_atomic_ref_count_init(&lora->ref_count);
      lora->host_allocator = host_allocator;
      lora->target_count = target_count;
      lora->targets =
          (id4_ideogram4_lora_target_t*)((uint8_t*)lora + target_offset);
      char* string_cursor = (char*)lora + string_offset;
      lora->source_scope =
          id4_ideogram4_lora_copy_string(options->source_scope, &string_cursor);
      iree_host_size_t target_index = 0;
      for (iree_host_size_t i = 0; i < pair_capacity; ++i) {
        if (!pairs[i].down.entry) continue;
        id4_ideogram4_lora_target_t* target = &lora->targets[target_index++];
        target->base_parameter_key = id4_ideogram4_lora_copy_base_key(
            pairs[i].down.entry->key, &string_cursor);
        target->down_parameter_key = id4_ideogram4_lora_copy_string(
            pairs[i].down.entry->key, &string_cursor);
        target->up_parameter_key = id4_ideogram4_lora_copy_string(
            pairs[i].up.entry->key, &string_cursor);
        target->input_size = pairs[i].input_size;
        target->output_size = pairs[i].output_size;
        target->rank = pairs[i].down.rows;
      }
    }
  }

  iree_allocator_free(host_allocator, pairs);
  if (iree_status_is_ok(status)) {
    *out_lora = lora;
  } else {
    iree_allocator_free(host_allocator, lora);
  }
  return status;
}

void id4_ideogram4_lora_retain(id4_ideogram4_lora_t* lora) {
  if (!lora) return;
  iree_atomic_ref_count_inc(&lora->ref_count);
}

void id4_ideogram4_lora_release(id4_ideogram4_lora_t* lora) {
  if (lora && iree_atomic_ref_count_dec(&lora->ref_count) == 1) {
    id4_ideogram4_lora_destroy(lora);
  }
}

iree_string_view_t id4_ideogram4_lora_source_scope(
    const id4_ideogram4_lora_t* lora) {
  return lora ? lora->source_scope : iree_string_view_empty();
}

iree_host_size_t id4_ideogram4_lora_target_count(
    const id4_ideogram4_lora_t* lora) {
  return lora ? lora->target_count : 0;
}

const id4_ideogram4_lora_target_t* id4_ideogram4_lora_target_at(
    const id4_ideogram4_lora_t* lora, iree_host_size_t index) {
  if (!lora || index >= lora->target_count) return NULL;
  return &lora->targets[index];
}

const id4_ideogram4_lora_target_t* id4_ideogram4_lora_lookup_target(
    const id4_ideogram4_lora_t* lora, iree_string_view_t base_parameter_key) {
  if (!lora) return NULL;
  for (iree_host_size_t i = 0; i < lora->target_count; ++i) {
    if (iree_string_view_equal(lora->targets[i].base_parameter_key,
                               base_parameter_key)) {
      return &lora->targets[i];
    }
  }
  return NULL;
}
