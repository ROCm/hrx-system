// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/net/channel_transfer.h"

#include "iree/base/alignment.h"
#include "iree/hal/device.h"

enum {
  IREE_HAL_NET_CHANNEL_DATA_HEADER_SIZE =
      IREE_NET_QUEUE_MESSAGE_HEADER_SIZE + IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE,
  IREE_HAL_NET_CHANNEL_CREDIT_SIZE = IREE_HAL_NET_CHANNEL_DATA_HEADER_SIZE + 8,
};

// Returns the admitted operation before publishing completion so a woken
// submitter can reuse the slot immediately. Local copies own all references
// used after publication; the next generation cannot overwrite those captures.
static void iree_hal_net_channel_transfer_complete(
    iree_hal_net_channel_transfer_t* transfer, iree_status_t status) {
  iree_hal_net_channel_peer_t* peer = transfer->peer;
  iree_hal_net_channel_peer_fail(peer, iree_status_clone(status));
  iree_hal_semaphore_t* signals[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
  uint64_t values[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
  iree_slim_mutex_lock(&peer->mutex);
  iree_hal_queue_t* queue = transfer->queue;
  iree_hal_buffer_t* buffer = transfer->buffer.buffer;
  transfer->buffer = (iree_hal_buffer_ref_t){0};
  transfer->data = NULL;
  const iree_host_size_t signal_count = transfer->signals.count;
  memcpy(signals, transfer->signals.semaphores,
         signal_count * sizeof(signals[0]));
  memcpy(values, transfer->signals.values, signal_count * sizeof(values[0]));
  for (iree_host_size_t i = 0; i < transfer->waits.count; ++i) {
    iree_async_semaphore_release(transfer->waits.semaphores[i]);
  }
  transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_IDLE;
  --peer->pending_count;
  iree_slim_mutex_unlock(&peer->mutex);
  iree_hal_buffer_release(buffer);
  const iree_hal_semaphore_list_t signal_list = {
      .count = signal_count, .semaphores = signals, .payload_values = values};
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_list, NULL);
    iree_hal_net_channel_peer_fail(peer, iree_status_clone(status));
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_list, status);
  }
  iree_hal_semaphore_list_release(signal_list);
  iree_hal_queue_release(queue);
  iree_hal_net_channel_peer_service(peer);
  iree_hal_channel_transfer_release(&transfer->base);
}

// Abandoned dependency/receive work resolves with the owning channel failure.
// Native sends instead report their actual source-return result, even when
// local or peer shutdown races with that receipt.
static void iree_hal_net_channel_transfer_abort(
    iree_hal_net_channel_transfer_t* transfer, iree_status_t status) {
  iree_hal_net_channel_peer_t* peer = transfer->peer;
  iree_slim_mutex_lock(&peer->mutex);
  status = iree_status_join(iree_status_clone(peer->status), status);
  iree_slim_mutex_unlock(&peer->mutex);
  if (iree_status_is_ok(status)) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_hal_net_channel_transfer_complete(transfer, status);
}

// The pool prepared this slot before admission. Explicit dependencies publish
// its native value before this load, including for queued allocation epochs.
static void iree_hal_net_channel_transfer_resolve(
    iree_hal_net_channel_transfer_t* transfer) {
  const iree_hal_buffer_native_binding_slot_t slot = {
      .index = transfer->base.params.pool->memory_contract->binding_layout
                   .host_binding_index,
      .type = IREE_HAL_BUFFER_INTERFACE_HOST,
  };
  transfer->data =
      (uint8_t*)iree_hal_buffer_native_binding(transfer->buffer.buffer, slot)
          .host_pointer +
      transfer->buffer.offset;
}

static void iree_hal_net_channel_transfer_on_wait(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  iree_hal_net_channel_peer_t* peer = transfer->peer;
  iree_slim_mutex_lock(&peer->mutex);
  bool closing = peer->closing;
  transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_READY;
  iree_slim_mutex_unlock(&peer->mutex);
  if (closing) {
    iree_hal_net_channel_transfer_abort(transfer, status);
  } else if (!iree_status_is_ok(status)) {
    iree_hal_net_channel_transfer_complete(transfer, status);
  } else {
    iree_hal_net_channel_transfer_resolve(transfer);
    iree_hal_net_channel_peer_service(peer);
  }
}

static void iree_hal_net_channel_transfer_on_issue(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  iree_hal_net_channel_peer_t* peer = transfer->peer;
  iree_slim_mutex_lock(&peer->mutex);
  bool closing = peer->closing;
  transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_READY;
  iree_slim_mutex_unlock(&peer->mutex);
  if (closing) {
    iree_hal_net_channel_transfer_abort(transfer, status);
  } else if (!iree_status_is_ok(status)) {
    iree_hal_net_channel_transfer_complete(transfer, status);
  } else if (transfer->waits.count) {
    memset(&transfer->wait_operation, 0, sizeof(transfer->wait_operation));
    transfer->wait_operation.base.type =
        IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
    transfer->wait_operation.base.completion_fn =
        iree_hal_net_channel_transfer_on_wait;
    transfer->wait_operation.base.user_data = transfer;
    transfer->wait_operation.semaphores = transfer->waits.semaphores;
    transfer->wait_operation.values = transfer->waits.values;
    transfer->wait_operation.count = transfer->waits.count;
    transfer->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
    status = iree_async_proactor_submit_one(peer->proactor,
                                            &transfer->wait_operation.base);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&peer->mutex);
      transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_WAITING;
      iree_slim_mutex_unlock(&peer->mutex);
    } else {
      iree_hal_net_channel_transfer_complete(transfer, status);
    }
  } else {
    iree_hal_net_channel_transfer_resolve(transfer);
    iree_hal_net_channel_peer_service(peer);
  }
}

static iree_status_t iree_hal_net_channel_transfer_submit(
    iree_hal_channel_transfer_t* base_transfer, iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    uint64_t sequence, iree_hal_buffer_ref_t buffer) {
  iree_hal_net_channel_transfer_t* transfer =
      (iree_hal_net_channel_transfer_t*)base_transfer;
  iree_hal_net_channel_peer_t* peer = transfer->peer;
  if (waits.count > IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY ||
      signals.count > IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "channel exchange supports at most eight waits "
                            "and eight signals");
  }
  // Host progress requires both timeline access and payload publication at
  // host scope. Device-only edges can be used by a device execution strategy,
  // but cannot safely be consumed by this mapped-memory route.
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    if (!iree_any_bit_set(iree_hal_device_query_semaphore_compatibility(
                              peer->device, waits.semaphores[i]),
                          IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "channel wait requires host-visible publication");
    }
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    if (!iree_any_bit_set(iree_hal_device_query_semaphore_compatibility(
                              peer->device, signals.semaphores[i]),
                          IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "channel signal requires host-visible publication");
    }
  }
  iree_slim_mutex_lock(&peer->mutex);
  iree_status_t status = iree_ok_status();
  if (peer->closing) {
    status = iree_status_is_ok(peer->status)
                 ? iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                    "channel admission is closed")
                 : iree_status_clone(peer->status);
  } else if (transfer->phase != IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_IDLE) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "prepared transfer already has an exchange");
  } else if (sequence <= transfer->sequence) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "channel sequence must increase strictly");
  }
  if (iree_status_is_ok(status)) {
    transfer->queue = queue;
    transfer->buffer = buffer;
    iree_hal_buffer_retain(buffer.buffer);
    transfer->waits.count = waits.count;
    transfer->signals.count = signals.count;
    for (iree_host_size_t i = 0; i < waits.count; ++i) {
      transfer->waits.semaphores[i] =
          (iree_async_semaphore_t*)waits.semaphores[i];
      transfer->waits.values[i] = waits.payload_values[i];
    }
    for (iree_host_size_t i = 0; i < signals.count; ++i) {
      transfer->signals.semaphores[i] = signals.semaphores[i];
      transfer->signals.values[i] = signals.payload_values[i];
    }
    iree_hal_semaphore_list_retain(waits);
    iree_hal_semaphore_list_retain(signals);
    iree_hal_queue_retain(queue);
    iree_hal_channel_transfer_retain(base_transfer);
    memset(&transfer->issue_operation, 0, sizeof(transfer->issue_operation));
    transfer->issue_operation.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    transfer->issue_operation.base.completion_fn =
        iree_hal_net_channel_transfer_on_issue;
    transfer->issue_operation.base.user_data = transfer;
    status = iree_async_proactor_submit_one(peer->proactor,
                                            &transfer->issue_operation.base);
    if (iree_status_is_ok(status)) {
      transfer->sequence = sequence;
      transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_QUEUED;
      ++peer->pending_count;
    } else {
      iree_hal_semaphore_list_release(waits);
      iree_hal_semaphore_list_release(signals);
      iree_hal_queue_release(queue);
      iree_hal_buffer_release(buffer.buffer);
      transfer->buffer = (iree_hal_buffer_ref_t){0};
      iree_hal_channel_transfer_release(base_transfer);
    }
  }
  iree_slim_mutex_unlock(&peer->mutex);
  return status;
}

static void iree_hal_net_channel_transfer_destroy(
    iree_hal_channel_transfer_t* transfer) {
  // Slot storage belongs to the channel, not this public handle.
  iree_hal_channel_transfer_deinitialize(transfer);
}

static const iree_hal_channel_transfer_vtable_t
    iree_hal_net_channel_transfer_vtable = {
        .destroy = iree_hal_net_channel_transfer_destroy,
        .submit = iree_hal_net_channel_transfer_submit,
};

iree_status_t iree_hal_net_channel_prepare_transfer(
    iree_hal_net_channel_peer_t* peer,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t** out_transfer) {
  if (params->tag >= peer->options.tag_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transfer tag exceeds prepared channel capacity");
  }
  if (params->capacity >
      IREE_HOST_SIZE_MAX - IREE_HAL_NET_CHANNEL_DATA_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prepared message exceeds host address space");
  }
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(params->pool, &capabilities);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      capabilities.memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  const iree_hal_memory_contract_t* contract = params->pool->memory_contract;
  const iree_hal_memory_access_t access =
      params->mode == IREE_HAL_CHANNEL_TRANSFER_MODE_SEND
          ? IREE_HAL_MEMORY_ACCESS_READ
          : IREE_HAL_MEMORY_ACCESS_WRITE;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_access(contract->host.access, access));
  if (!iree_all_bits_set(contract->host.modes,
                         IREE_HAL_MAPPING_MODE_PERSISTENT) ||
      contract->binding_layout.host_binding_index ==
          IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "channel pool requires prepared persistent host "
                            "access");
  }
  iree_hal_net_channel_transfer_t* transfer =
      &peer->transfers[(iree_host_size_t)params->mode *
                           peer->options.tag_capacity +
                       params->tag];
  iree_slim_mutex_lock(&peer->mutex);
  iree_status_t status = iree_ok_status();
  if (peer->closing || transfer->prepared) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "channel closed or transfer tag already prepared");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_channel_transfer_initialize(
        &iree_hal_net_channel_transfer_vtable, peer->owner, peer->device,
        peer->queue_family_affinity, params, &transfer->base);
    transfer->prepared = true;
    *out_transfer = &transfer->base;
  }
  iree_slim_mutex_unlock(&peer->mutex);
  return status;
}

static iree_status_t iree_hal_net_channel_transfer_build(
    void* user_data, const iree_net_queue_message_builder_t* builder) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  iree_net_queue_frontier_builder_set(
      &builder->signal_frontier, 0,
      (iree_async_frontier_entry_t){.axis = transfer->base.params.tag,
                                    .epoch = transfer->sequence});
  if (builder->generated_payload.data_length) {
    iree_unaligned_store_le_u64(builder->generated_payload.data,
                                transfer->buffer.length);
  }
  return iree_ok_status();
}

static void iree_hal_net_channel_transfer_on_source_returned(
    void* user_data, iree_status_t status, iree_host_size_t bytes) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  if (iree_status_is_ok(status) &&
      bytes !=
          IREE_HAL_NET_CHANNEL_DATA_HEADER_SIZE + transfer->buffer.length) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS, "short channel send");
  }
  iree_hal_net_channel_transfer_complete(transfer, status);
}

static void iree_hal_net_channel_transfer_on_credit_returned(
    void* user_data, iree_status_t status, iree_host_size_t bytes) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  transfer->credit_pending = false;
  if (iree_status_is_ok(status) && bytes != IREE_HAL_NET_CHANNEL_CREDIT_SIZE) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS, "short channel credit");
  }
  iree_hal_net_channel_peer_fail(transfer->peer, status);
  iree_hal_net_channel_peer_service(transfer->peer);
}

static void iree_hal_net_channel_transfer_on_target_returned(
    void* user_data, iree_status_t status, iree_host_size_t bytes) {
  iree_hal_net_channel_transfer_t* transfer = user_data;
  transfer->receive.pending = false;
  // Matching validated the exact extent before native writes. Success returns
  // the complete target; a failed return only authorizes local reclamation.
  if (iree_status_is_ok(status)) {
    iree_hal_net_channel_transfer_complete(transfer, status);
  } else {
    iree_hal_net_channel_transfer_abort(transfer, status);
  }
}

void iree_hal_net_channel_transfers_service(iree_hal_net_channel_peer_t* peer) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < peer->transfer_count && iree_status_is_ok(status); ++i) {
    iree_hal_net_channel_transfer_t* transfer = &peer->transfers[i];
    iree_slim_mutex_lock(&peer->mutex);
    const bool ready =
        !peer->closing &&
        transfer->phase == IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_READY;
    iree_slim_mutex_unlock(&peer->mutex);
    if (!ready) {
      continue;
    }
    const bool sending =
        transfer->base.params.mode == IREE_HAL_CHANNEL_TRANSFER_MODE_SEND;
    if (sending && transfer->credit.sequence < transfer->sequence) {
      continue;
    }
    if (!sending && transfer->credit_pending) {
      continue;
    }
    if (sending && (transfer->credit.sequence != transfer->sequence ||
                    transfer->credit.length != transfer->buffer.length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "send and receive sequence or extent differ");
      continue;
    }
    const iree_net_carrier_send_budget_t budget =
        iree_net_queue_channel_query_send_budget(peer->protocol);
    if (!budget.slots || !budget.bytes) {
      break;
    }
    iree_net_queue_channel_send_params_t params = {
        .signal_frontier_count = 1,
        .build = iree_hal_net_channel_transfer_build,
        .build_user_data = transfer,
        .completion_callback =
            {.fn = sending ? iree_hal_net_channel_transfer_on_source_returned
                           : iree_hal_net_channel_transfer_on_credit_returned,
             .user_data = transfer},
    };
    if (sending) {
      iree_async_span_t source =
          iree_async_span_from_ptr(transfer->data, transfer->buffer.length);
      params.payload = iree_async_span_list_make(&source, 1);
      status = iree_net_queue_channel_send_command(
          peer->protocol, IREE_NET_QUEUE_ID_NONE, &params);
    } else {
      const iree_async_single_frontier_t coordinate = {
          .entry_count = 1,
          .entries = {{.axis = transfer->base.params.tag,
                       .epoch = transfer->sequence}}};
      status = iree_net_queue_channel_receive_command(
          peer->protocol, IREE_NET_QUEUE_ID_NONE, NULL,
          iree_async_single_frontier_as_const_frontier(&coordinate),
          &transfer->receive.operation,
          iree_make_byte_span(transfer->data, transfer->buffer.length),
          (iree_net_receive_completion_callback_t){
              .fn = iree_hal_net_channel_transfer_on_target_returned,
              .user_data = transfer});
      if (iree_status_is_ok(status)) {
        transfer->receive.pending = true;
        params.generated_payload_length = 8;
        status = iree_net_queue_channel_send_advance(peer->protocol, &params);
      }
    }
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&peer->mutex);
      transfer->phase = IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_ISSUED;
      iree_slim_mutex_unlock(&peer->mutex);
      if (sending) {
        transfer->credit.used_sequence = transfer->sequence;
      } else {
        transfer->credit_pending = true;
      }
    }
  }
  iree_hal_net_channel_peer_fail(peer, status);
}

void iree_hal_net_channel_transfers_abort(iree_hal_net_channel_peer_t* peer) {
  for (iree_host_size_t i = 0; i < peer->transfer_count; ++i) {
    iree_hal_net_channel_transfer_t* transfer = &peer->transfers[i];
    iree_slim_mutex_lock(&peer->mutex);
    const iree_hal_net_channel_transfer_phase_t phase = transfer->phase;
    iree_slim_mutex_unlock(&peer->mutex);
    if (phase == IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_WAITING) {
      // Software semaphore cancellation has no native address-key borrower.
      // Its terminal callback joins every timepoint callback before reuse.
      iree_hal_net_channel_peer_fail(
          peer, iree_async_proactor_cancel(peer->proactor,
                                           &transfer->wait_operation.base));
    } else if (!transfer->receive.pending &&
               (phase == IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_READY ||
                (phase == IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_ISSUED &&
                 transfer->base.params.mode ==
                     IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE))) {
      iree_hal_net_channel_transfer_abort(transfer, iree_ok_status());
    }
  }
}

iree_status_t iree_hal_net_channel_on_command(
    void* user_data, uint32_t queue_id,
    const iree_net_queue_frontier_view_t* waits,
    const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  // Every valid grant already posted its exact prefix and current target.
  // Matched commands complete that receive without whole-message delivery.
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "payload does not name a posted receive");
}

iree_status_t iree_hal_net_channel_on_advance(
    void* user_data, const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_hal_net_channel_peer_t* peer = user_data;
  if (signals->count != 1 || payload.data_length != 8) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "channel credit requires one coordinate and extent");
  }
  const iree_async_frontier_entry_t coordinate =
      iree_net_queue_frontier_view_get(signals, 0);
  if (coordinate.axis >= peer->options.tag_capacity) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "channel tag out of range");
  }
  iree_hal_net_channel_transfer_t* transfer = &peer->transfers[coordinate.axis];
  const uint64_t length = iree_unaligned_load_le_u64(payload.data);
  if (!length || coordinate.epoch <= transfer->credit.sequence ||
      transfer->credit.sequence != transfer->credit.used_sequence) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "credit replaced an unused or newer receive grant");
  }
  transfer->credit.sequence = coordinate.epoch;
  transfer->credit.length = length;
  iree_hal_net_channel_peer_service(peer);
  return iree_ok_status();
}
