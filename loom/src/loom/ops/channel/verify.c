// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/type_propagation.h"

typedef enum loom_channel_handle_kind_e {
  LOOM_CHANNEL_HANDLE_CHANNEL,
  LOOM_CHANNEL_HANDLE_READ,
  LOOM_CHANNEL_HANDLE_WRITE,
} loom_channel_handle_kind_t;

typedef enum loom_channel_field_kind_e {
  LOOM_CHANNEL_FIELD_OPERAND,
  LOOM_CHANNEL_FIELD_RESULT,
} loom_channel_field_kind_t;

static iree_status_t loom_channel_emit_type_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_channel_field_kind_t field_kind, iree_string_view_t field_name,
    loom_type_t actual_type, iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_type(actual_type),
      loom_param_string(constraint),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = field_kind == LOOM_CHANNEL_FIELD_OPERAND ? LOOM_ERR_TYPE_003
                                                        : LOOM_ERR_TYPE_004,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static bool loom_channel_payload(const loom_module_t* module, loom_type_t type,
                                 loom_channel_handle_kind_t kind,
                                 loom_type_t* out_payload) {
  loom_type_id_t payload_id = LOOM_TYPE_ID_INVALID;
  switch (kind) {
    case LOOM_CHANNEL_HANDLE_CHANNEL:
      if (!loom_channel_type_isa(type)) {
        return false;
      }
      payload_id = loom_channel_type_payload(type);
      break;
    case LOOM_CHANNEL_HANDLE_READ:
      if (!loom_read_type_isa(type)) {
        return false;
      }
      payload_id = loom_read_type_payload(type);
      break;
    case LOOM_CHANNEL_HANDLE_WRITE:
      if (!loom_write_type_isa(type)) {
        return false;
      }
      payload_id = loom_write_type_payload(type);
      break;
  }
  *out_payload = loom_type_table_get(&module->types, payload_id);
  return loom_type_is_tile(*out_payload);
}

static bool loom_channel_view_matches(loom_type_t payload, loom_type_t view,
                                      uint8_t prefix_rank) {
  if (!loom_type_is_view(view) ||
      loom_type_element_type(payload) != loom_type_element_type(view) ||
      loom_type_rank(view) != (uint16_t)loom_type_rank(payload) + prefix_rank) {
    return false;
  }
  for (uint8_t axis = 0; axis < loom_type_rank(payload); ++axis) {
    if (loom_type_dim(payload, axis) !=
        loom_type_dim(view, axis + prefix_rank)) {
      return false;
    }
  }
  return true;
}

// A payload's shape is one contract shared by its channel, owned accesses and
// borrowed views. A local type transaction must not specialize a view in
// isolation. Slot extents and view encodings remain independently refinable.
iree_status_t loom_channel_bind_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  const loom_type_t channel =
      loom_type_transfer_value_type(context, loom_channel_bind_result(op));
  const loom_type_t payload =
      loom_type_table_get(&module->types, loom_channel_type_payload(channel));
  const loom_type_t storage =
      loom_type_transfer_value_type(context, loom_channel_bind_storage(op));
  if (!loom_channel_view_matches(payload, storage, 1)) {
    loom_type_transfer_reject(context);
  }
  return iree_ok_status();
}

// Reserve and acquire both return a borrowed view as their second result.
iree_status_t loom_channel_selection_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  const loom_type_t channel =
      loom_type_transfer_value_type(context, loom_op_const_operands(op)[0]);
  const loom_type_t payload =
      loom_type_table_get(&module->types, loom_channel_type_payload(channel));
  const loom_type_t view =
      loom_type_transfer_value_type(context, loom_op_const_results(op)[1]);
  if (!loom_channel_view_matches(payload, view, 0)) {
    loom_type_transfer_reject(context);
  }
  return iree_ok_status();
}

iree_status_t loom_channel_wait_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  const loom_type_t read =
      loom_type_transfer_value_type(context, loom_channel_wait_read(op));
  const loom_type_t payload =
      loom_type_table_get(&module->types, loom_read_type_payload(read));
  const loom_type_t view =
      loom_type_transfer_value_type(context, loom_channel_wait_view(op));
  if (!loom_channel_view_matches(payload, view, 0)) {
    loom_type_transfer_reject(context);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_channel_handle_constraint(
    loom_channel_handle_kind_t kind) {
  static const iree_string_view_t constraints[] = {
      IREE_SVL("channel<tile<...>>"),
      IREE_SVL("read<tile<...>> with optional mutable permission"),
      IREE_SVL("write<tile<...>>"),
  };
  return constraints[kind];
}

iree_status_t loom_channel_bind_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t type =
      loom_module_value_type(module, loom_channel_bind_result(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, type, LOOM_CHANNEL_HANDLE_CHANNEL,
                            &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("result"), type,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_CHANNEL));
  }
  loom_type_t storage =
      loom_module_value_type(module, loom_channel_bind_storage(op));
  if (!loom_channel_view_matches(payload, storage, 1)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("storage"), storage,
        IREE_SV("one leading slot dimension followed by the channel payload "
                "shape"));
  }
  return iree_ok_status();
}

// Reserve, accept and acquire share a channel-to-owned-access type contract.
// Their descriptor-verified result lists distinguish access-only and access
// plus view forms; the view always borrows the selected record's obligation.
static iree_status_t loom_channel_verify_selection(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_channel_handle_kind_t access_kind,
    uint8_t mode) {
  loom_type_t channel =
      loom_module_value_type(module, loom_op_const_operands(op)[0]);
  loom_type_t payload;
  if (!loom_channel_payload(module, channel, LOOM_CHANNEL_HANDLE_CHANNEL,
                            &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("channel"), channel,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_CHANNEL));
  }
  loom_type_t access =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  loom_type_t access_payload;
  iree_string_view_t access_name = access_kind == LOOM_CHANNEL_HANDLE_READ
                                       ? IREE_SV("read")
                                       : IREE_SV("write");
  if (!loom_channel_payload(module, access, access_kind, &access_payload) ||
      !loom_type_equal(payload, access_payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_RESULT, access_name, access,
        IREE_SV("owned access matching the channel payload"));
  }
  if (access_kind == LOOM_CHANNEL_HANDLE_READ) {
    uint8_t access_mode = loom_read_type_has_mode(access)
                              ? (uint8_t)loom_read_type_mode(access)
                              : 0;
    if (access_mode != mode) {
      return loom_channel_emit_type_constraint(
          emitter, op, LOOM_CHANNEL_FIELD_RESULT, access_name, access,
          IREE_SV("read permission matching the operation's access mode"));
    }
  }
  if (op->result_count == 2) {
    loom_type_t view =
        loom_module_value_type(module, loom_op_const_results(op)[1]);
    if (!loom_channel_view_matches(payload, view, 0)) {
      return loom_channel_emit_type_constraint(
          emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("view"), view,
          IREE_SV("view matching the channel payload shape"));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_channel_reserve_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_channel_verify_selection(module, op, emitter,
                                       LOOM_CHANNEL_HANDLE_WRITE, 0);
}

iree_status_t loom_channel_accept_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  return loom_channel_verify_selection(module, op, emitter,
                                       LOOM_CHANNEL_HANDLE_READ,
                                       loom_channel_accept_mode(op));
}

iree_status_t loom_channel_acquire_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_channel_verify_selection(module, op, emitter,
                                       LOOM_CHANNEL_HANDLE_READ,
                                       loom_channel_acquire_mode(op));
}

iree_status_t loom_channel_wait_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t read = loom_module_value_type(module, loom_channel_wait_read(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, read, LOOM_CHANNEL_HANDLE_READ, &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("read"), read,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_READ));
  }
  loom_type_t view = loom_module_value_type(module, loom_channel_wait_view(op));
  if (!loom_channel_view_matches(payload, view, 0)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("view"), view,
        IREE_SV("view matching the read payload shape"));
  }
  return iree_ok_status();
}

iree_status_t loom_channel_publish_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t write =
      loom_module_value_type(module, loom_channel_publish_write(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, write, LOOM_CHANNEL_HANDLE_WRITE,
                            &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("write"), write,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_WRITE));
  }
  return iree_ok_status();
}

iree_status_t loom_channel_release_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t read =
      loom_module_value_type(module, loom_channel_release_read(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, read, LOOM_CHANNEL_HANDLE_READ, &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("read"), read,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_READ));
  }
  return iree_ok_status();
}

iree_status_t loom_channel_fanout_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t read =
      loom_module_value_type(module, loom_channel_fanout_read(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, read, LOOM_CHANNEL_HANDLE_READ, &payload) ||
      loom_read_type_has_mode(read)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("read"), read,
        IREE_SV("immutable read<tile<...>>"));
  }
  loom_value_slice_t reads = loom_channel_fanout_reads(op);
  if (reads.count < 2) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("channel.fanout")),
        loom_param_string(IREE_SV("reads")),
        loom_param_u32(reads.count),
        loom_param_u32(2),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_STRUCTURE_057,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  for (uint16_t i = 0; i < reads.count; ++i) {
    loom_type_t result = loom_module_value_type(module, reads.values[i]);
    if (!loom_type_equal(read, result)) {
      return loom_channel_emit_type_constraint(
          emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("reads"), result,
          IREE_SV("the same immutable read type as the source"));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_channel_copy_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t source =
      loom_module_value_type(module, loom_channel_copy_source(op));
  loom_type_t source_payload;
  if (!loom_channel_payload(module, source, LOOM_CHANNEL_HANDLE_READ,
                            &source_payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("source"), source,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_READ));
  }
  loom_type_t destination =
      loom_module_value_type(module, loom_channel_copy_destination(op));
  loom_type_t destination_payload;
  if (!loom_channel_payload(module, destination, LOOM_CHANNEL_HANDLE_WRITE,
                            &destination_payload) ||
      !loom_type_equal(source_payload, destination_payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("destination"),
        destination,
        IREE_SV("write with the same payload type as the source read"));
  }
  return iree_ok_status();
}

iree_status_t loom_channel_select_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  const loom_type_t channel =
      loom_module_value_type(module, loom_channel_select_channel(op));
  loom_type_t payload;
  if (!loom_channel_payload(module, channel, LOOM_CHANNEL_HANDLE_CHANNEL,
                            &payload)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_OPERAND, IREE_SV("channel"), channel,
        loom_channel_handle_constraint(LOOM_CHANNEL_HANDLE_CHANNEL));
  }

  const loom_type_t source_rank =
      loom_module_value_type(module, loom_channel_select_source_rank(op));
  if (!loom_type_equal(source_rank, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("source_rank"),
        source_rank, IREE_SV("index"));
  }

  const loom_type_t read =
      loom_module_value_type(module, loom_channel_select_read(op));
  loom_type_t read_payload;
  if (!loom_channel_payload(module, read, LOOM_CHANNEL_HANDLE_READ,
                            &read_payload) ||
      !loom_type_equal(payload, read_payload) ||
      loom_read_type_has_mode(read)) {
    return loom_channel_emit_type_constraint(
        emitter, op, LOOM_CHANNEL_FIELD_RESULT, IREE_SV("read"), read,
        IREE_SV("immutable read matching the channel payload"));
  }
  return iree_ok_status();
}
