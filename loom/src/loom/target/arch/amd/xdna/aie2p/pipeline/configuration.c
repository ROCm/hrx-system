// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/intern_table.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/channel/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/target.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_version.h"

typedef struct loom_aie2p_native_configuration_t {
  // Native occurrence supplying physical selection.
  loom_aie2p_native_context_t* context;
  // Typed configuration instruction descriptors.
  const loom_low_descriptor_set_t* descriptors;
  // Builder in the currently emitted configuration function.
  loom_builder_t builder;
  // Scalar constant results in the current straight-line function. Cleared at
  // each function boundary so every reused result dominates its consumers.
  loom_intern_table_t constants;
  // Configuration scalar type, independent of the core's machine word width.
  loom_type_t scalar_type;
  // Native invocation buffer identity.
  loom_type_t binding_type;
  // Relocated byte range in an invocation buffer.
  loom_type_t span_type;
  // Constant immediate name in configuration descriptors.
  loom_string_id_t value_name;
  // Explicit configuration representation contract.
  loom_string_id_t contract;
  // Exact deployment target declaration.
  loom_symbol_ref_t target;
} loom_aie2p_native_configuration_t;

typedef struct loom_aie2p_native_constant_key_t {
  // Module owning the canonical constant result IDs.
  const loom_module_t* module;
  // Exact scalar bit pattern being materialized.
  uint64_t value;
} loom_aie2p_native_constant_key_t;

static bool loom_aie2p_native_constant_equal(const void* user_data,
                                             uint32_t index) {
  const loom_aie2p_native_constant_key_t* key = user_data;
  const loom_op_t* op =
      loom_value_def_op(loom_module_value(key->module, index));
  return (uint64_t)loom_low_const_attrs(op).entries[0].value.i64 == key->value;
}

static iree_status_t loom_aie2p_native_config_constant(
    loom_aie2p_native_configuration_t* config, uint64_t value,
    loom_value_id_t* out_value) {
  const uint32_t hash = loom_structural_hash_finalize(
      loom_structural_hash_mix_u64(loom_structural_hash_initialize(), value));
  const loom_aie2p_native_constant_key_t key = {
      .module = config->context->code.module, .value = value};
  loom_intern_probe_t probe = loom_intern_table_probe(
      &config->constants, hash, loom_aie2p_native_constant_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_value = probe.index;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      config->context->pass->arena, &config->constants, hash, 1, &probe.slot));
  const loom_named_attr_t attr = {.name_id = config->value_name,
                                  .value = loom_attr_i64((int64_t)value)};
  loom_op_t* op;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &config->builder, config->descriptors,
      &config->descriptors->descriptors
           [AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_CONSTANT],
      loom_make_named_attr_slice(&attr, 1), config->scalar_type,
      LOOM_LOCATION_UNKNOWN, &op));
  *out_value = loom_low_const_result(op);
  loom_intern_table_insert(&config->constants, probe.slot, hash, *out_value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_config_op(
    loom_aie2p_native_configuration_t* config, uint32_t descriptor,
    const uint64_t* values, iree_host_size_t value_count,
    loom_named_attr_slice_t attributes) {
  loom_value_id_t* operands;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(config->context->pass->arena,
                                                 value_count, sizeof(*operands),
                                                 (void**)&operands));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_config_constant(config, values[i], &operands[i]));
  }
  loom_op_t* op;
  return loom_low_build_resolved_descriptor_op(
      &config->builder, config->descriptors,
      &config->descriptors->descriptors[descriptor], 0, operands, value_count,
      attributes, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op);
}

static iree_status_t loom_aie2p_native_config_function(
    loom_aie2p_native_configuration_t* config, loom_symbol_ref_t symbol,
    uint8_t visibility, uint8_t retain, loom_op_t** out_function) {
  loom_intern_table_clear(&config->constants);
  loom_builder_set_block(&config->builder,
                         loom_module_block(config->context->code.module));
  return loom_low_func_def_build(
      &config->builder,
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET |
          (visibility ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY : 0) |
          (retain ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN : 0) |
          (visibility || retain ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI : 0),
      visibility, retain, 0, 0, 0, 0, 0, config->contract, config->target,
      LOOM_TARGET_ABI_ARRAY_PROGRAM, loom_named_attr_slice_empty(),
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), symbol, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, out_function);
}

static iree_status_t loom_aie2p_native_config_register(
    loom_aie2p_native_configuration_t* config, loom_xdna_tile_coordinate_t tile,
    loom_xdna_register_field_id_t field, const uint16_t* indices,
    iree_host_size_t index_count, int64_t value, uint32_t instruction) {
  uint64_t address;
  uint32_t bits;
  loom_xdna_register_field_info_t info;
  IREE_RETURN_IF_ERROR(loom_xdna_register_field_info(field, &info));
  IREE_RETURN_IF_ERROR(loom_xdna_register_field_address(
      config->context->family, field, tile, index_count, indices, &address));
  IREE_RETURN_IF_ERROR(loom_xdna_register_field_encode(field, value, &bits));
  const uint64_t mask = ((UINT64_C(1) << info.bit_width) - 1)
                        << info.least_significant_bit;
  const uint64_t operands[] = {address, mask, bits};
  return loom_aie2p_native_config_op(config, instruction, operands, 3,
                                     loom_named_attr_slice_empty());
}

static iree_status_t loom_aie2p_native_config_write(
    loom_aie2p_native_configuration_t* config, uint64_t address,
    uint32_t value) {
  const uint64_t operands[] = {address, value};
  return loom_aie2p_native_config_op(
      config, AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE32,
      operands, 2, loom_named_attr_slice_empty());
}

static iree_status_t loom_aie2p_native_config_reset_core(
    loom_aie2p_native_configuration_t* config,
    loom_xdna_tile_coordinate_t tile) {
  // Disable and reset are bits in the same control register. Set the reset
  // state atomically while preserving the register's other fields.
  const uint32_t enable = loom_xdna_register_field_encode_admitted(
      LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_ENABLE, 1);
  const uint32_t reset = loom_xdna_register_field_encode_admitted(
      LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_RESET, 1);
  const uint64_t address = loom_xdna_register_field_address_admitted(
      config->context->family, LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_RESET,
      tile, NULL);
  const uint64_t operands[] = {address, enable | reset, reset};
  return loom_aie2p_native_config_op(
      config, AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_MASK32,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty());
}

static iree_status_t loom_aie2p_native_config_set_lock(
    loom_aie2p_native_configuration_t* config, loom_xdna_tile_coordinate_t tile,
    uint16_t lock, uint32_t value) {
  // The semaphore value occupies the complete writable register. A direct
  // write establishes its state without a redundant register read.
  return loom_aie2p_native_config_write(
      config,
      loom_xdna_register_field_address_admitted(
          config->context->family,
          LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE, tile,
          &lock),
      loom_xdna_register_field_encode_admitted(
          LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE, value));
}

static iree_status_t loom_aie2p_native_config_bank(
    loom_aie2p_native_configuration_t* config,
    const loom_aie2p_native_register_bank_t* bank, loom_value_id_t zero) {
  const iree_host_size_t operand_count = bank->word_count + 1;
  loom_value_id_t* operands;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(config->context->pass->arena, operand_count,
                                sizeof(*operands), (void**)&operands));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_config_constant(config, bank->address, &operands[0]));
  for (iree_host_size_t i = 0; i < bank->word_count; ++i) {
    operands[i + 1] = zero;
    if (bank->words[i]) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_constant(
          config, bank->words[i], &operands[i + 1]));
    }
  }
  loom_op_t* op;
  return loom_low_build_resolved_descriptor_op(
      &config->builder, config->descriptors,
      &config->descriptors->descriptors
           [AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_BLOCK32],
      0, operands, operand_count, loom_named_attr_slice_empty(), NULL, 0, NULL,
      0, LOOM_LOCATION_UNKNOWN, &op);
}

static iree_status_t loom_aie2p_native_config_routes(
    loom_aie2p_native_configuration_t* config) {
  const loom_aie2p_native_context_t* context = config->context;
  if (context->switches) {
    loom_value_id_t zero;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_constant(config, 0, &zero));
    for (const loom_aie2p_native_switch_t* state = context->switches; state;
         state = state->next) {
      for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_native_config_bank(config, &state->banks[i], zero));
      }
    }
  }
  static const loom_xdna_register_field_id_t mux_fields[8] = {
      0,
      0,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_MUX_CONFIG_SOUTH2,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_MUX_CONFIG_SOUTH3,
      0,
      0,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_MUX_CONFIG_SOUTH6,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_MUX_CONFIG_SOUTH7};
  static const loom_xdna_register_field_id_t demux_fields[6] = {
      0,
      0,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DEMUX_CONFIG_SOUTH2,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DEMUX_CONFIG_SOUTH3,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DEMUX_CONFIG_SOUTH4,
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DEMUX_CONFIG_SOUTH5};
  for (iree_host_size_t i = 0; i < context->routing.route_count; ++i) {
    const loom_aie2p_array_route_plan_t* edge = &context->routing.routes[i];
    if (edge->switch_kind != LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX) {
      continue;
    }
    const loom_xdna_register_field_id_t field =
        edge->source_port == LOOM_XDNA_STREAM_PORT_DMA
            ? mux_fields[edge->destination_channel]
            : demux_fields[edge->source_channel];
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
        config, edge->coordinate, field, NULL, 0, 1,
        AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_MASK32));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_config_transfers(
    loom_aie2p_native_configuration_t* config,
    const loom_pipeline_realization_t* realization) {
  const loom_aie2p_native_context_t* context = config->context;
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    for (const loom_aie2p_native_dma_path_t* path = worker->paths; path;
         path = path->next) {
      const loom_xdna_register_field_id_t local_control =
          path->ingress
              ? LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_CONTROL_RESET
              : LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_CONTROL_RESET;
      const uint16_t local_engine = path->local_engine;
      const uint64_t local_address = loom_xdna_register_field_address_admitted(
          context->family, local_control, path->local->coordinate,
          &local_engine);
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_write(
          config, local_address,
          loom_xdna_register_field_encode_admitted(local_control, 1)));
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_config_write(config, local_address, 0));
      const loom_xdna_register_field_id_t shim_control =
          path->ingress
              ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_CONTROL_CONTROLLER_ID
              : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_CONTROL_CONTROLLER_ID;
      const uint16_t shim_engine = path->shim_engine;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_write(
          config,
          loom_xdna_register_field_address_admitted(
              context->family, shim_control, path->shim->coordinate,
              &shim_engine),
          loom_xdna_register_field_encode_admitted(shim_control,
                                                   path->completion_packet)));
    }
    for (const loom_aie2p_native_transfer_t* transfer = worker->transfers;
         transfer; transfer = transfer->next) {
      uint64_t local[7] = {loom_xdna_register_field_address_admitted(
          context->family,
          LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD0_BASE_ADDRESS,
          transfer->path->local->coordinate, &transfer->local_descriptor)};
      uint64_t shim[9] = {loom_xdna_register_field_address_admitted(
          context->family,
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD0_BUFFER_LENGTH,
          transfer->path->shim->coordinate, &transfer->shim_descriptor)};
      for (unsigned j = 0; j < IREE_ARRAYSIZE(transfer->local_words); ++j) {
        local[j + 1] = transfer->local_words[j];
      }
      for (unsigned j = 0; j < IREE_ARRAYSIZE(transfer->shim_words); ++j) {
        shim[j + 1] = transfer->shim_words[j];
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
          config,
          AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_BLOCK32, local,
          IREE_ARRAYSIZE(local), loom_named_attr_slice_empty()));
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
          config,
          AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_BLOCK32, shim,
          IREE_ARRAYSIZE(shim), loom_named_attr_slice_empty()));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_config_bindings(
    loom_aie2p_native_configuration_t* config,
    const loom_pipeline_realization_t* realization) {
  const loom_aie2p_native_context_t* context = config->context;
  loom_value_id_t* bindings;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, context->binding_count,
                                sizeof(*bindings), (void**)&bindings));
  for (iree_host_size_t i = 0; i < context->binding_count; ++i) {
    const loom_aie2p_native_binding_t* binding = &context->bindings[i];
    if (!binding->access) {
      const uint64_t ordinal = i;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
          config,
          AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_BINDING_UNUSED,
          &ordinal, 1, loom_named_attr_slice_empty()));
      continue;
    }
    const uint64_t values[] = {i, binding->access, binding->byte_length,
                               binding->byte_alignment};
    loom_value_id_t operands[4];
    for (unsigned j = 0; j < IREE_ARRAYSIZE(values); ++j) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_config_constant(config, values[j], &operands[j]));
    }
    loom_op_t* op;
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
        &config->builder, config->descriptors,
        &config->descriptors->descriptors
             [AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_BINDING],
        0, operands, 4, loom_named_attr_slice_empty(), &config->binding_type, 1,
        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
    bindings[i] = loom_op_results(op)[0];
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    const uint64_t tile_address = (uint64_t)worker->tile->coordinate.column
                                      << context->family->column_shift |
                                  (uint64_t)worker->tile->coordinate.row
                                      << context->family->row_shift;
    for (const loom_aie2p_native_transfer_t* transfer = worker->transfers;
         transfer; transfer = transfer->next) {
      const loom_symbolic_expr_t* offset =
          &transfer->external_view->begin_byte_offset;
      const bool external_dynamic =
          !worker->repetition.count && !loom_symbolic_expr_is_constant(offset);
      const uint64_t begin = worker->repetition.count
                                 ? worker->repetition.external_byte_offset
                             : external_dynamic ? 0
                                                : offset->constant;
      loom_value_id_t range[] = {bindings[transfer->binding],
                                 LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_config_constant(config, begin, &range[1]));
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_constant(
          config, context->bindings[transfer->binding].byte_length - begin,
          &range[2]));
      loom_op_t* op;
      IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
          &config->builder, config->descriptors,
          &config->descriptors->descriptors
               [AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_RANGE],
          0, range, 3, loom_named_attr_slice_empty(), &config->span_type, 1,
          NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
      loom_value_id_t operands[] = {LOOM_VALUE_ID_INVALID,
                                    loom_op_results(op)[0]};
      // Constant offsets bind directly into this site's dedicated descriptor.
      // Dynamic offsets load a relocated base through the worker's aperture;
      // configuration writes use the memory owner's allocation address space.
      const uint64_t address =
          external_dynamic
              ? tile_address + worker->tile->facts->memory.local_base +
                    transfer->base_storage_offset
              : loom_xdna_register_field_address_admitted(
                    context->family,
                    LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD1_BASE_ADDRESS_LOW,
                    transfer->path->shim->coordinate,
                    &transfer->shim_descriptor);
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_config_constant(config, address, &operands[0]));
      IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
          &config->builder, config->descriptors,
          &config->descriptors->descriptors
               [AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_ADDRESS],
          0, operands, 2, loom_named_attr_slice_empty(), NULL, 0, NULL, 0,
          LOOM_LOCATION_UNKNOWN, &op));
      if (worker->repetition.count &&
          worker->repetition.count %
              transfer->local_channel->source->capacity) {
        // The descriptor advances its iteration when loaded. All publications
        // were consumed before the previous invocation returned, so its final
        // advancement is complete. Restore slot zero when the bounded repeat
        // does not wrap naturally; the consumer also starts at slot zero.
        IREE_RETURN_IF_ERROR(loom_aie2p_native_config_write(
            config,
            loom_xdna_register_field_address_admitted(
                context->family,
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD4_ITERATION_CURRENT,
                transfer->path->local->coordinate, &transfer->local_descriptor),
            transfer->local_words[4]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_config_submit(
    loom_aie2p_native_configuration_t* config,
    const loom_aie2p_native_transfer_t* transfer, uint32_t repetitions) {
  const loom_aie2p_native_dma_path_t* path = transfer->path;
  const loom_xdna_register_field_id_t local_queue =
      path->ingress
          ? LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_START_QUEUE_START_BD_ID
          : LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_START_QUEUE_START_BD_ID;
  const loom_xdna_register_field_id_t shim_queue =
      path->ingress
          ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
          : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID;
  uint32_t local_task = loom_xdna_register_field_encode_admitted(
      local_queue, transfer->local_descriptor);
  uint32_t shim_task = loom_xdna_register_field_encode_admitted(
      shim_queue, transfer->shim_descriptor);
  if (repetitions > 1) {
    local_task |= loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_START_QUEUE_REPEAT_COUNT,
        repetitions - 1);
    shim_task |= loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT,
        repetitions - 1);
  }
  if (!path->ingress) {
    shim_task |= loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
        1);
  }
  const uint16_t local_engine = path->local_engine;
  const uint16_t shim_engine = path->shim_engine;
  const uint64_t local_address = loom_xdna_register_field_address_admitted(
      config->context->family, local_queue, path->local->coordinate,
      &local_engine);
  const uint64_t shim_address = loom_xdna_register_field_address_admitted(
      config->context->family, shim_queue, path->shim->coordinate,
      &shim_engine);
  // Enqueue the receiver before the sender, as in core-issued movement.
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_write(
      config, path->ingress ? local_address : shim_address,
      path->ingress ? local_task : shim_task));
  return loom_aie2p_native_config_write(
      config, path->ingress ? shim_address : local_address,
      path->ingress ? shim_task : local_task);
}

static iree_status_t loom_aie2p_native_config_communicate(
    loom_aie2p_native_configuration_t* config,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index) {
  const loom_aie2p_native_context_t* context = config->context;
  const loom_pipeline_worker_t* source = &realization->workers[worker_index];
  const loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  const loom_aie2p_native_transfer_t* issue = worker->transfers;
  const loom_aie2p_native_transfer_t* complete = worker->transfers;
  for (iree_host_size_t i = 0; i < source->transport.count; ++i) {
    const loom_pipeline_transport_step_t* step = &source->transport.steps[i];
    switch (step->kind) {
      case LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER: {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_config_submit(config, issue, 1));
        issue = issue->next;
        break;
      }
      case LOOM_PIPELINE_TRANSPORT_STEP_WAIT:
        while (complete && complete->completion == step->op) {
          const loom_aie2p_native_dma_path_t* path = complete->path;
          if (path->ingress) {
            IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
                config, path->local->coordinate,
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE,
                &complete->completion_lock, 1, 1,
                AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WAIT_MASK32));
          } else {
            const uint64_t values[] = {path->shim->coordinate.column,
                                       path->shim->coordinate.row,
                                       LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
                                       path->shim_engine,
                                       1,
                                       1};
            IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
                config,
                AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_DMA_WAIT,
                values, IREE_ARRAYSIZE(values), loom_named_attr_slice_empty()));
          }
          complete = complete->next;
        }
        break;
      case LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL: {
        // Initial free credit cannot block and final reclamation has no later
        // admission observing it. Readiness is still an explicit source edge.
        if (!loom_channel_acquire_isa(step->op) &&
            !loom_channel_publish_isa(step->op)) {
          break;
        }
        if (issue && issue->admission == step->op) {
          break;
        }
        const loom_pipeline_resource_channel_t* binding =
            loom_pipeline_resources_lookup_channel(
                &realization->resources,
                step->source.channel->channel->value_id);
        const loom_aie2p_native_channel_t* channel =
            &context->channels[binding - realization->resources.channels];
        const loom_xdna_tile_coordinate_t tile =
            context->pool_tiles[channel->pool_index]->coordinate;
        const bool acquire = loom_channel_acquire_isa(step->op);
        if (acquire) {
          IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
              config, tile,
              LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE,
              &channel->ready_lock, 1, 1,
              AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WAIT_MASK32));
        }
        // Each endpoint acts once and has one cursor owner. No second producer
        // can race this ready-credit update with another publication.
        IREE_RETURN_IF_ERROR(loom_aie2p_native_config_set_lock(
            config, tile, channel->ready_lock, acquire ? 0 : 1));
        break;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_configuration(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization) {
  loom_aie2p_native_context_t* context = user_data;
  loom_module_t* module = rewriter->module;
  loom_aie2p_native_configuration_t config = {
      .context = context,
      .descriptors = loom_aie2p_configuration_descriptor_set()};
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(context->pass->arena, 0, &config.constants));
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &config.builder);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("amd.xdna.aie2p.configuration"), &config.contract));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("value"), &config.value_name));
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      module, config.descriptors,
      AIE2P_CONFIGURATION_REG_CLASS_ID_AIE2P_CONFIG_SCALAR, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I64), &config.scalar_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      config.descriptors, AIE2P_CONFIGURATION_REG_CLASS_ID_AIE2P_CONFIG_BINDING,
      1, &config.binding_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      config.descriptors, AIE2P_CONFIGURATION_REG_CLASS_ID_AIE2P_CONFIG_SPAN, 1,
      &config.span_type));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_symbol(&context->code, &config.target));
  loom_target_facts_t* facts;
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(&context->target->base,
                                                       &module->arena, &facts));
  loom_target_facts_builder_set_worker_contract(
      LOOM_AIE2P_TARGET_KIND_CONFIGURATION,
      &loom_aie2p_configuration_target_bundle, facts);
  loom_target_function_version_t* version =
      loom_target_function_version_cast(context->pass->function_version);
  const loom_resolved_target_t resolved = {
      .provider = version->resolved_target.provider, .facts = facts};
  IREE_RETURN_IF_ERROR(loom_aie2p_target_materialize_definition(
      &config.builder, &resolved, config.target, LOOM_LOCATION_UNKNOWN));
  loom_symbol_ref_t initialize_symbol, invoke_symbol;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_symbol(&context->code, &initialize_symbol));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_symbol(&context->code, &invoke_symbol));
  const loom_symbol_ref_t entry_symbol =
      loom_func_like_callee(realization->function);
  const uint8_t visibility = loom_func_like_visibility(realization->function);
  const uint8_t retain =
      iree_any_bit_set(module->symbols.entries[entry_symbol.symbol_id].flags,
                       LOOM_SYMBOL_FLAG_RETAIN)
          ? LOOM_LOW_RETAIN_RETAIN
          : 0;
  uint64_t columns = 1;
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const uint64_t limit = context->workers[i].tile->coordinate.column + 1;
    if (limit > columns) {
      columns = limit;
    }
  }
  loom_op_t* initialize;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_function(
      &config, initialize_symbol, 0, 0, &initialize));
  loom_builder_enter_region(&config.builder, initialize,
                            loom_low_func_def_body(initialize));
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_config_reset_core(&config, worker->tile->coordinate));
    const uint64_t position[] = {worker->tile->coordinate.column,
                                 worker->tile->coordinate.row};
    loom_string_id_t program_name;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, IREE_SV("program"), &program_name));
    const loom_named_attr_t program = {
        .name_id = program_name,
        .value = loom_attr_symbol(
            loom_func_like_callee(realization->workers[i].function))};
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
        &config, AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_PROGRAM_LOAD,
        position, 2, loom_make_named_attr_slice(&program, 1)));
  }
  for (iree_host_size_t i = 0; i < context->inventory.pool_count; ++i) {
    const uint64_t length = loom_source_storage_packing_requirement(
                                context->inventory.pools[i].packing)
                                .byte_length;
    if (!length) {
      continue;
    }
    const loom_xdna_tile_coordinate_t tile = context->pool_tiles[i]->coordinate;
    if ((uint64_t)tile.column + 1 > columns) {
      columns = tile.column + 1;
    }
    const uint64_t range[] = {tile.column, tile.row, 0, length};
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
        &config, AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_DATA_RESERVE,
        range, 4, loom_named_attr_slice_empty()));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_routes(&config));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_config_transfers(&config, realization));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&config.builder, NULL, 0));

  loom_op_t* invoke;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_config_function(&config, invoke_symbol, 0, 0, &invoke));
  loom_builder_enter_region(&config.builder, invoke,
                            loom_low_func_def_body(invoke));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_bindings(&config, realization));
  for (iree_host_size_t i = 0; i < realization->resources.channel_count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const loom_xdna_tile_coordinate_t tile =
        context->pool_tiles[channel->pool_index]->coordinate;
    // A configuration writer reserves its sole record from the initial
    // credit. Debit it here so a core reader's final release cannot overflow
    // the semaphore even when the channel starts at its maximum capacity.
    const bool reserve_initial =
        channel->cursor.writer.worker != UINT32_MAX &&
        context->workers[channel->cursor.writer.worker].execution ==
            LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_set_lock(
        &config, tile, channel->free_lock,
        channel->source->capacity - reserve_initial));
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_set_lock(
        &config, tile, channel->ready_lock, 0));
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_set_lock(
          &config, worker->tile->coordinate, worker->completion_lock, 0));
    }
    for (const loom_aie2p_native_transfer_t* transfer = worker->transfers;
         transfer; transfer = transfer->next) {
      if (!transfer->path->ingress ||
          worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_set_lock(
          &config, transfer->path->local->coordinate, transfer->completion_lock,
          0));
    }
  }
  // Every semaphore is initialized before any worker can produce a credit.
  // Ingress completions can belong to a neighboring worker's memory tile.
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    if (context->workers[i].execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    const loom_xdna_tile_coordinate_t tile =
        context->workers[i].tile->coordinate;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
        &config, tile, LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_RESET, NULL, 0, 0,
        AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_MASK32));
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
        &config, tile, LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_ENABLE, NULL, 0, 1,
        AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_MASK32));
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_config_submit(
          &config, worker->transfers, worker->repetition.count));
    }
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_config_communicate(&config, realization, i));
    }
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_register(
        &config, worker->tile->coordinate,
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE,
        &worker->completion_lock, 1, 1,
        AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WAIT_MASK32));
  }
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    if (context->workers[i].execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    const loom_xdna_tile_coordinate_t tile =
        context->workers[i].tile->coordinate;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_config_reset_core(&config, tile));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&config.builder, NULL, 0));

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, realization->function.op));
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    if (context->workers[i].execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase(rewriter, realization->workers[i].function.op));
    }
  }
  loom_op_t* entry;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_function(
      &config, entry_symbol, visibility, retain, &entry));
  loom_builder_enter_region(&config.builder, entry,
                            loom_low_func_def_body(entry));
  loom_string_id_t initialize_name, invoke_name;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("initialize"),
                                                 &initialize_name));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("invoke"), &invoke_name));
  const loom_named_attr_t entries[] = {
      {.name_id = initialize_name,
       .value = loom_attr_symbol(initialize_symbol)},
      {.name_id = invoke_name, .value = loom_attr_symbol(invoke_symbol)}};
  IREE_RETURN_IF_ERROR(loom_aie2p_native_config_op(
      &config, AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_ENTRY, &columns,
      1, loom_make_named_attr_slice(entries, 2)));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&config.builder, NULL, 0));
  loom_function_version_update(&version->base,
                               loom_func_like_cast(module, entry));
  version->resolved_target = resolved;
  version->function_target_facts = facts;
  version->target_requirement_facts = facts;
  version->authored_target_is_exact = true;
  version->target_context_ordinal = realization->entry_target_context;
  return iree_ok_status();
}
