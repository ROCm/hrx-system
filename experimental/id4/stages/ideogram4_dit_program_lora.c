// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program_lora.h"

#include <inttypes.h>
#include <stdint.h>

#include "experimental/id4/stages/ideogram4_dit_program_block.h"

enum {
  ID4_IDEOGRAM4_DIT_LORA_CONFIG_VALUE_BUFFER_CAPACITY = 16,
  ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT = 6,
  ID4_IDEOGRAM4_DIT_LORA_SEGMENT_METADATA_FIELD_COUNT = 4,
};

typedef struct id4_ideogram4_dit_program_lora_config_value_t {
  // Loom config key selected by the dispatch authoring site.
  iree_string_view_t key;
  // Unsigned integer config value formatted for loomc.
  uint32_t value;
} id4_ideogram4_dit_program_lora_config_value_t;

typedef enum id4_ideogram4_dit_program_lora_projection_e {
  ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN = 0,
  ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_UP = 1,
} id4_ideogram4_dit_program_lora_projection_t;

static iree_status_t id4_ideogram4_dit_program_lora_make_config_bindings(
    iree_host_size_t value_count,
    const id4_ideogram4_dit_program_lora_config_value_t* values,
    char value_buffers[ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT]
                      [ID4_IDEOGRAM4_DIT_LORA_CONFIG_VALUE_BUFFER_CAPACITY],
    id4_pipeline_kernel_config_binding_t* out_bindings) {
  if (value_count > ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT LoRA config binding count %" PRIhsz
                            " exceeds max count %u",
                            value_count,
                            ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT);
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    iree_string_view_t value_string = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
        value_buffers[i], ID4_IDEOGRAM4_DIT_LORA_CONFIG_VALUE_BUFFER_CAPACITY,
        &value_string, "%" PRIu32, values[i].value));
    out_bindings[i] =
        id4_pipeline_make_kernel_config_binding(values[i].key, value_string);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_lora_dispatch(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t function_name, iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  const id4_pipeline_program_dispatch_loom_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .kernel = id4_pipeline_make_kernel_ref(IREE_SV("ideogram4/lora_bf16"),
                                             function_name),
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = binding_count,
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &options);
}

static bool id4_ideogram4_dit_program_lora_checked_mul_u64(
    uint64_t lhs, uint64_t rhs, uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return false;
  *out_result = lhs * rhs;
  return true;
}

iree_status_t id4_ideogram4_dit_program_validate_lora_topology(
    id4_ideogram4_dit_lora_topology_t topology) {
  if (topology.adapter_count == 0 || topology.target_count == 0) {
    if (topology.adapter_count != 0 || topology.target_count != 0 ||
        topology.targets) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT LoRA topology must be entirely empty or contain "
          "adapters and targets");
    }
    return iree_ok_status();
  }
  if (!topology.targets) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LoRA targets are required");
  }
  if (topology.adapter_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT LoRA adapter count %" PRIhsz
                            " exceeds I32 metadata",
                            topology.adapter_count);
  }
  for (iree_host_size_t i = 0; i < topology.target_count; ++i) {
    const id4_ideogram4_dit_lora_target_t* target = &topology.targets[i];
    if (iree_string_view_is_empty(target->base_parameter_key) ||
        target->input_size == 0 || target->output_size == 0 ||
        target->total_rank == 0 || target->segment_count == 0 ||
        !target->segments) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT LoRA target %" PRIhsz " is incomplete", i);
    }
    if (target->total_rank > INT32_MAX || target->segment_count > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram4 DiT LoRA target `%.*s` exceeds I32 metadata limits",
          (int)target->base_parameter_key.size,
          target->base_parameter_key.data);
    }
    uint32_t expected_rank_offset = 0;
    uint64_t up_element_offset = 0;
    for (iree_host_size_t j = 0; j < target->segment_count; ++j) {
      const id4_ideogram4_dit_lora_segment_t* segment = &target->segments[j];
      if (segment->adapter_ordinal >= topology.adapter_count ||
          segment->rank == 0 || segment->rank > INT32_MAX ||
          segment->rank_offset > INT32_MAX ||
          segment->rank_offset != expected_rank_offset ||
          iree_string_view_is_empty(segment->source_scope) ||
          iree_string_view_is_empty(segment->down_parameter_key) ||
          iree_string_view_is_empty(segment->up_parameter_key)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT LoRA target `%.*s` segment %" PRIhsz " is invalid",
            (int)target->base_parameter_key.size,
            target->base_parameter_key.data, j);
      }
      if (segment->rank > UINT32_MAX - expected_rank_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Ideogram4 DiT LoRA target `%.*s` rank overflows U32",
            (int)target->base_parameter_key.size,
            target->base_parameter_key.data);
      }
      expected_rank_offset += segment->rank;
      uint64_t segment_up_element_count = 0;
      if (!id4_ideogram4_dit_program_lora_checked_mul_u64(
              target->output_size, segment->rank, &segment_up_element_count) ||
          segment_up_element_count > UINT64_MAX - up_element_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Ideogram4 DiT LoRA target `%.*s` up projection overflows",
            (int)target->base_parameter_key.size,
            target->base_parameter_key.data);
      }
      up_element_offset += segment_up_element_count;
    }
    if (expected_rank_offset != target->total_rank) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT LoRA target `%.*s` has inconsistent composed rank",
          (int)target->base_parameter_key.size,
          target->base_parameter_key.data);
    }
    if (up_element_offset > INT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram4 DiT LoRA target `%.*s` up projection exceeds I32 "
          "metadata",
          (int)target->base_parameter_key.size,
          target->base_parameter_key.data);
    }
    for (iree_host_size_t j = i + 1; j < topology.target_count; ++j) {
      if (iree_string_view_equal(target->base_parameter_key,
                                 topology.targets[j].base_parameter_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "Ideogram4 DiT LoRA target `%.*s` is duplicated",
            (int)target->base_parameter_key.size,
            target->base_parameter_key.data);
      }
    }
  }
  return iree_ok_status();
}

const id4_ideogram4_dit_lora_target_t*
id4_ideogram4_dit_program_lookup_lora_target(
    id4_ideogram4_dit_lora_topology_t topology,
    iree_string_view_t base_parameter_key) {
  for (iree_host_size_t i = 0; i < topology.target_count; ++i) {
    if (iree_string_view_equal(topology.targets[i].base_parameter_key,
                               base_parameter_key)) {
      return &topology.targets[i];
    }
  }
  return NULL;
}

static iree_status_t id4_ideogram4_dit_program_lora_assemble_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t key,
    const id4_ideogram4_dit_lora_target_t* target,
    id4_ideogram4_dit_program_lora_projection_t projection,
    id4_pipeline_program_tensor_t* out_tensor) {
  iree_allocator_t allocator = id4_pipeline_program_builder_allocator(builder);
  id4_pipeline_program_parameter_source_t* sources = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, target->segment_count, sizeof(*sources), (void**)&sources));
  id4_pipeline_program_parameter_source_span_t* spans = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, target->segment_count, sizeof(*spans), (void**)&spans));

  uint64_t target_byte_offset = 0;
  for (iree_host_size_t i = 0; i < target->segment_count; ++i) {
    const id4_ideogram4_dit_lora_segment_t* segment = &target->segments[i];
    uint64_t element_count = 0;
    const uint32_t row_count =
        projection == ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN
            ? segment->rank
            : target->output_size;
    const uint32_t column_count =
        projection == ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN
            ? target->input_size
            : segment->rank;
    if (!id4_ideogram4_dit_program_lora_checked_mul_u64(row_count, column_count,
                                                        &element_count) ||
        element_count > UINT64_MAX / sizeof(uint16_t)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram4 DiT LoRA parameter `%.*s` byte length overflows",
          (int)key.size, key.data);
    }
    const uint64_t byte_length = element_count * sizeof(uint16_t);
    if (byte_length > IREE_DEVICE_SIZE_MAX - target_byte_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram4 DiT LoRA parameter `%.*s` assembly overflows",
          (int)key.size, key.data);
    }
    sources[i] = (id4_pipeline_program_parameter_source_t){
        .source_scope = segment->source_scope,
        .key = projection == ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN
                   ? segment->down_parameter_key
                   : segment->up_parameter_key,
        .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .shape = id4_pipeline_program_make_shape_rank2(row_count, column_count),
    };
    spans[i] = (id4_pipeline_program_parameter_source_span_t){
        .source_offset = 0,
        .target_offset = (iree_device_size_t)target_byte_offset,
        .length = (iree_device_size_t)byte_length,
        .source_index = i,
    };
    target_byte_offset += byte_length;
  }

  const id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      .source_count = target->segment_count,
      .sources = sources,
      .key = key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .shape = projection == ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN
                   ? id4_pipeline_program_make_shape_rank2(target->total_rank,
                                                           target->input_size)
                   : id4_pipeline_program_make_shape_rank1(target_byte_offset /
                                                           sizeof(uint16_t)),
      .source_span_count = target->segment_count,
      .source_spans = spans,
      .domain = IREE_SV("lora_dynamic"),
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

static iree_status_t id4_ideogram4_dit_program_lora_segment_metadata(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_ideogram4_dit_lora_target_t* target,
    id4_pipeline_program_tensor_t* out_tensor) {
  uint64_t metadata_element_count = 0;
  if (!id4_ideogram4_dit_program_lora_checked_mul_u64(
          target->segment_count,
          ID4_IDEOGRAM4_DIT_LORA_SEGMENT_METADATA_FIELD_COUNT,
          &metadata_element_count) ||
      metadata_element_count > IREE_HOST_SIZE_MAX / sizeof(int32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT LoRA metadata size overflows");
  }
  iree_allocator_t allocator = id4_pipeline_program_builder_allocator(builder);
  int32_t* metadata = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, (iree_host_size_t)metadata_element_count, sizeof(*metadata),
      (void**)&metadata));

  uint64_t up_element_offset = 0;
  for (iree_host_size_t i = 0; i < target->segment_count; ++i) {
    const id4_ideogram4_dit_lora_segment_t* segment = &target->segments[i];
    int32_t* fields =
        &metadata[i * ID4_IDEOGRAM4_DIT_LORA_SEGMENT_METADATA_FIELD_COUNT];
    fields[0] = (int32_t)segment->adapter_ordinal;
    fields[1] = (int32_t)segment->rank_offset;
    fields[2] = (int32_t)segment->rank;
    fields[3] = (int32_t)up_element_offset;
    up_element_offset += (uint64_t)target->output_size * segment->rank;
  }

  const id4_pipeline_program_constant_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_I32,
      .shape = id4_pipeline_program_make_shape_rank2(
          target->segment_count,
          ID4_IDEOGRAM4_DIT_LORA_SEGMENT_METADATA_FIELD_COUNT),
      .data = iree_make_const_byte_span(
          metadata,
          (iree_host_size_t)metadata_element_count * sizeof(*metadata)),
  };
  return id4_pipeline_program_constant(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_begin_lora(
    id4_pipeline_program_builder_t* builder, iree_string_view_t operation_name,
    const id4_ideogram4_dit_lora_target_t* target,
    iree_host_size_t adapter_count, id4_pipeline_program_tensor_t strengths,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output, iree_host_size_t* target_use_count,
    id4_ideogram4_dit_program_lora_application_t* out_application) {
  if (!builder || !target_use_count || !out_application) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT LoRA authoring outputs are required");
  }
  *out_application = (id4_ideogram4_dit_program_lora_application_t){0};
  if (!target) return iree_ok_status();
  if (adapter_count == 0 || adapter_count > UINT32_MAX || token_count == 0 ||
      token_capacity < token_count || input_size != target->input_size ||
      output_size != target->output_size ||
      !id4_pipeline_program_tensor_is_valid(strengths) ||
      !id4_pipeline_program_tensor_is_valid(input) ||
      !id4_pipeline_program_tensor_is_valid(output)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT LoRA target `%.*s` does not match its projection",
        (int)target->base_parameter_key.size, target->base_parameter_key.data);
  }

  char down_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  char up_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  char metadata_name_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  char low_rank_name_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t down_key = iree_string_view_empty();
  iree_string_view_t up_key = iree_string_view_empty();
  iree_string_view_t metadata_name = iree_string_view_empty();
  iree_string_view_t low_rank_name = iree_string_view_empty();
  iree_string_view_t dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      down_key_buffer, IREE_ARRAYSIZE(down_key_buffer), &down_key,
      "%.*s.lora.down", (int)target->base_parameter_key.size,
      target->base_parameter_key.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      up_key_buffer, IREE_ARRAYSIZE(up_key_buffer), &up_key, "%.*s.lora.up",
      (int)target->base_parameter_key.size, target->base_parameter_key.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      metadata_name_buffer, IREE_ARRAYSIZE(metadata_name_buffer),
      &metadata_name, "%.*s.lora.segments",
      (int)target->base_parameter_key.size, target->base_parameter_key.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      low_rank_name_buffer, IREE_ARRAYSIZE(low_rank_name_buffer),
      &low_rank_name, "%.*s.lora.low_rank", (int)operation_name.size,
      operation_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      dispatch_name_buffer, IREE_ARRAYSIZE(dispatch_name_buffer),
      &dispatch_name, "%.*s.lora.down", (int)operation_name.size,
      operation_name.data));

  id4_pipeline_program_tensor_t down = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_assemble_parameter(
      builder, down_key, target, ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_DOWN,
      &down));
  id4_pipeline_program_tensor_t up = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_assemble_parameter(
      builder, up_key, target, ID4_IDEOGRAM4_DIT_PROGRAM_LORA_PROJECTION_UP,
      &up));
  id4_pipeline_program_tensor_t segment_metadata =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_segment_metadata(
      builder, metadata_name, target, &segment_metadata));
  id4_pipeline_program_tensor_t low_rank =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, low_rank_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, target->total_rank),
      &low_rank));

  const id4_ideogram4_dit_program_lora_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.lora.token_count"), token_count},
      {IREE_SV("id4.ideogram4.lora.token_capacity"), token_capacity},
      {IREE_SV("id4.ideogram4.lora.input_size"), input_size},
      {IREE_SV("id4.ideogram4.lora.total_rank"), target->total_rank},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_LORA_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  const id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(down),
      id4_pipeline_program_write(low_rank),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_dispatch(
      builder, dispatch_name, IREE_SV("id4_ideogram4_lora_down_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings));

  ++*target_use_count;
  *out_application = (id4_ideogram4_dit_program_lora_application_t){
      .target = target,
      .low_rank = low_rank,
      .up = up,
      .segment_metadata = segment_metadata,
      .strengths = strengths,
      .output = output,
      .token_count = token_count,
      .token_capacity = token_capacity,
      .output_size = output_size,
      .adapter_count = (uint32_t)adapter_count,
  };
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_finish_lora(
    id4_pipeline_program_builder_t* builder, iree_string_view_t operation_name,
    const id4_ideogram4_dit_program_lora_application_t* application) {
  if (!builder || !application) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LoRA finish arguments are required");
  }
  if (!application->target) return iree_ok_status();

  char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      dispatch_name_buffer, IREE_ARRAYSIZE(dispatch_name_buffer),
      &dispatch_name, "%.*s.lora.up_add", (int)operation_name.size,
      operation_name.data));
  const id4_ideogram4_dit_program_lora_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.lora.token_count"), application->token_count},
      {IREE_SV("id4.ideogram4.lora.token_capacity"),
       application->token_capacity},
      {IREE_SV("id4.ideogram4.lora.output_size"), application->output_size},
      {IREE_SV("id4.ideogram4.lora.total_rank"),
       application->target->total_rank},
      {IREE_SV("id4.ideogram4.lora.segment_count"),
       (uint32_t)application->target->segment_count},
      {IREE_SV("id4.ideogram4.lora.adapter_count"), application->adapter_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_LORA_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_LORA_MAX_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_lora_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  const id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(application->low_rank),
      id4_pipeline_program_read(application->up),
      id4_pipeline_program_read(application->segment_metadata),
      id4_pipeline_program_read(application->strengths),
      id4_pipeline_program_read_write(application->output),
  };
  return id4_ideogram4_dit_program_lora_dispatch(
      builder, dispatch_name, IREE_SV("id4_ideogram4_lora_up_add_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}
