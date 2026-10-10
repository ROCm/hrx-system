// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/connection_control.h"

#include <limits.h>

#include "iree/async/operations/scheduling.h"
#include "iree/net/carrier/rdma/connection_events.h"
#include "iree/net/carrier/rdma/device_failure.h"
#include "iree/net/rdma/region.h"

#define IREE_NET_RDMA_CONTROL_RECEIVE_ID_BIT (UINT64_C(1) << 62)
#define IREE_NET_RDMA_CONTROL_PRIVATE_DATA_SIZE 56u

typedef enum iree_net_rdma_connection_control_state_e {
  IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CREATED = 0,
  IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTING,
  IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED,
  IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DRAINING,
  IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DEACTIVATED,
} iree_net_rdma_connection_control_state_t;

enum iree_net_rdma_connection_control_flag_bits_e {
  IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_ESTABLISHED = 1u << 0,
  IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_DISCONNECT_REQUESTED = 1u << 1,
};
typedef uint32_t iree_net_rdma_connection_control_flags_t;

struct iree_net_rdma_connection_control_t {
  // Allocator for control ownership and its fixed send-slot free list.
  iree_allocator_t host_allocator;
  // Retained native device/PD owner, shared with application registrations.
  iree_net_rdma_context_t* context;
  // Retained callback executor; no private thread or progress loop is created.
  iree_async_proactor_t* proactor;
  // Immutable enforced native bounds and setup policy.
  iree_net_rdma_connection_control_options_t options;
  // Stable containing-connection callbacks, joined by deactivation.
  iree_net_rdma_connection_control_callbacks_t callbacks;
  // Poll-owner-only lifecycle, independent of an optional terminal failure.
  iree_net_rdma_connection_control_state_t state;
  // Native handshake obligations preserved while lifecycle is draining.
  iree_net_rdma_connection_control_flags_t flags;
  // Owned terminal failure; the first error is also delivered to the owner.
  iree_status_t failure;
  // Registered private send/receive records, never exposed as user leases.
  iree_async_region_t* region;
  // CQ shared with independently owned connection data QPs.
  iree_net_rdma_completion_queue_t* completions;
  // CM channel with callbacks bound to this stable owner from its creation.
  iree_net_rdma_connection_events_t* events;
  // Native CQ/QP/device error subscription and owner-thread handoff.
  iree_net_rdma_device_failure_t* device_failure;
  // Owned CM identity, never implicitly owning or modifying a native QP.
  struct rdma_cm_id* id;
  // Independently owned private control QP, configured before CM publication.
  struct ibv_qp* queue;
  // Value snapshot available after establishment for independent data QPs.
  iree_net_rdma_connection_route_t route;
  // Copied native handshake bytes; neither view borrows an acknowledged event.
  struct {
    // Number of meaningful local request/reply bytes.
    uint8_t local_length;
    // Stable local bytes borrowed by native connect/accept submission.
    uint8_t local[IREE_NET_RDMA_CONTROL_PRIVATE_DATA_SIZE];
    // Native peer byte count, including provider padding.
    uint8_t peer_length;
    // Captured native private data from CONNECT_REQUEST or CONNECT_RESPONSE.
    uint8_t peer[UINT8_MAX];
  } handshake;
  // Number of currently reusable registered send records.
  uint32_t free_send_count;
  // CQ and CM notifications can arrive in either order. Completed control
  // receives stay in their original registered storage until on_ready returns.
  struct {
    // Borrowed trailing ring of completed native receive indices.
    uint32_t* slots;
    // Next completed receive to deliver in native control-QP order.
    uint32_t head;
    // Number of completed receives not yet delivered/reposted.
    uint32_t count;
  } receives;
  // Preallocated handoff that starts monitor retirement outside callbacks.
  iree_async_nop_operation_t deactivate_operation;
  // Monitor callbacks plus the deactivation NOP body still borrowing the owner.
  uint32_t pending_joins;
  // Terminal ownership callback, allowed to destroy this object.
  iree_async_event_source_unregistered_callback_t deactivated_callback;
  // Send free-list followed by the receive-index ring; no payload staging.
  uint32_t free_send_slots[];
};

static iree_status_t iree_net_rdma_connection_control_native_error(
    const char* operation, int error) {
  return iree_make_status(iree_status_code_from_errno(error), "%s: %s",
                          operation, strerror(error));
}

static void iree_net_rdma_connection_control_fail(
    iree_net_rdma_connection_control_t* control, iree_status_t status) {
  bool first_failure = iree_status_is_ok(control->failure);
  control->failure = iree_status_join(control->failure, status);
  if (first_failure) {
    control->callbacks.on_error(control->callbacks.user_data,
                                iree_status_clone(control->failure));
  }
}

static struct ibv_sge iree_net_rdma_connection_control_record_span(
    iree_net_rdma_connection_control_t* control, uint32_t index) {
  return (struct ibv_sge){
      .addr = control->region->handles.rdma.address +
              (uint64_t)index * IREE_NET_RDMA_CONTROL_RECORD_SIZE,
      .length = IREE_NET_RDMA_CONTROL_RECORD_SIZE,
      .lkey = control->region->handles.rdma.lkey,
  };
}

static iree_status_t iree_net_rdma_connection_control_post_receive(
    iree_net_rdma_connection_control_t* control, uint32_t index) {
  struct ibv_sge span = iree_net_rdma_connection_control_record_span(
      control, control->options.send_count + index);
  struct ibv_recv_wr request = {
      .wr_id = IREE_NET_RDMA_CONTROL_WORK_ID_BIT |
               IREE_NET_RDMA_CONTROL_RECEIVE_ID_BIT | index,
      .sg_list = &span,
      .num_sge = 1,
  };
  struct ibv_recv_wr* rejected = NULL;
  int error = ibv_post_recv(control->queue, &request, &rejected);
  if (error) {
    return iree_net_rdma_connection_control_native_error("ibv_post_recv",
                                                         error);
  }
  return iree_ok_status();
}

static void iree_net_rdma_connection_control_dispatch_records(
    iree_net_rdma_connection_control_t* control) {
  iree_status_t status = iree_ok_status();
  while (control->receives.count &&
         control->state == IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED &&
         iree_status_is_ok(control->failure) && iree_status_is_ok(status)) {
    uint32_t index = control->receives.slots[control->receives.head];
    control->receives.head =
        (control->receives.head + 1) % control->options.receive_count;
    --control->receives.count;
    const uint8_t* bytes = control->region->base_ptr;
    bytes += (iree_host_size_t)(control->options.send_count + index) *
             IREE_NET_RDMA_CONTROL_RECORD_SIZE;
    status = control->callbacks.on_record(
        control->callbacks.user_data,
        iree_make_const_byte_span(bytes, IREE_NET_RDMA_CONTROL_RECORD_SIZE));
    if (iree_status_is_ok(status) &&
        control->state == IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED &&
        iree_status_is_ok(control->failure)) {
      status = iree_net_rdma_connection_control_post_receive(control, index);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
  }
}

static void iree_net_rdma_connection_control_complete_record(
    iree_net_rdma_connection_control_t* control,
    const struct ibv_wc* completion, uint32_t* out_send_count) {
  if (control->state >= IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DRAINING ||
      !iree_status_is_ok(control->failure)) {
    return;
  }
  if (completion->status != IBV_WC_SUCCESS) {
    iree_net_rdma_connection_control_fail(
        control, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "RDMA control completion failed (%u)",
                                  (unsigned)completion->status));
    return;
  }
  uint32_t index = (uint32_t)completion->wr_id;
  if (!(completion->wr_id & IREE_NET_RDMA_CONTROL_RECEIVE_ID_BIT)) {
    control->free_send_slots[control->free_send_count++] = index;
    ++*out_send_count;
    return;
  }
  if (completion->byte_len != IREE_NET_RDMA_CONTROL_RECORD_SIZE) {
    iree_net_rdma_connection_control_fail(
        control, iree_make_status(
                     IREE_STATUS_DATA_LOSS,
                     "RDMA control record has %u bytes, expected %u",
                     completion->byte_len, IREE_NET_RDMA_CONTROL_RECORD_SIZE));
    return;
  }
  uint32_t tail = (control->receives.head + control->receives.count) %
                  control->options.receive_count;
  control->receives.slots[tail] = index;
  ++control->receives.count;
  iree_net_rdma_connection_control_dispatch_records(control);
}

static void iree_net_rdma_connection_control_completions(
    void* user_data, iree_host_size_t count, const struct ibv_wc* completions) {
  iree_net_rdma_connection_control_t* control = user_data;
  uint32_t send_count = 0;
  iree_host_size_t data_begin = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!(completions[i].wr_id & IREE_NET_RDMA_CONTROL_WORK_ID_BIT)) {
      continue;
    }
    if (i != data_begin) {
      control->callbacks.on_completions(control->callbacks.user_data,
                                        i - data_begin,
                                        completions + data_begin);
    }
    iree_net_rdma_connection_control_complete_record(control, &completions[i],
                                                     &send_count);
    data_begin = i + 1;
  }
  if (data_begin != count) {
    control->callbacks.on_completions(control->callbacks.user_data,
                                      count - data_begin,
                                      completions + data_begin);
  }
  if (send_count &&
      control->state == IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED &&
      iree_status_is_ok(control->failure)) {
    control->callbacks.on_capacity(control->callbacks.user_data);
  }
}

static void iree_net_rdma_connection_control_service_failed(
    void* user_data, iree_status_t status) {
  iree_net_rdma_connection_control_fail(user_data, status);
}

static iree_status_t iree_net_rdma_connection_control_create_queue(
    iree_net_rdma_connection_control_t* control) {
  if (control->id->verbs->device !=
      iree_net_rdma_context_device(control->context)->device) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA connection resolved to a different device");
  }
  struct ibv_qp_init_attr attributes = {
      .send_cq = iree_net_rdma_completion_queue_handle(control->completions),
      .recv_cq = iree_net_rdma_completion_queue_handle(control->completions),
      .cap = {.max_send_wr = control->options.send_count,
              .max_recv_wr = control->options.receive_count,
              .max_send_sge = 1,
              .max_recv_sge = 1},
      .qp_type = IBV_QPT_RC,
  };
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  control->queue = library->ibv_create_qp(
      iree_net_rdma_context_protection_domain(control->context), &attributes);
  if (!control->queue) {
    return iree_net_rdma_connection_control_native_error("ibv_create_qp",
                                                         errno);
  }
  struct ibv_qp_attr initial = {.qp_state = IBV_QPS_INIT};
  int mask = 0;
  if (library->rdma_init_qp_attr(control->id, &initial, &mask)) {
    return iree_net_rdma_connection_control_native_error("rdma_init_qp_attr",
                                                         errno);
  }
  initial.qp_access_flags = 0;
  if (initial.port_num != iree_net_rdma_context_port_number(control->context)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA connection resolved to a different port");
  }
  int error = library->ibv_modify_qp(control->queue, &initial, mask);
  if (error) {
    return iree_net_rdma_connection_control_native_error("ibv_modify_qp",
                                                         error);
  }
  // A partial or ambiguous native posting failure keeps the entire private
  // region owned until QP destruction. No guessed CQE count gates cleanup.
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < control->options.receive_count && iree_status_is_ok(status); ++i) {
    status = iree_net_rdma_connection_control_post_receive(control, i);
  }
  return status;
}

static struct rdma_conn_param iree_net_rdma_connection_control_parameters(
    iree_net_rdma_connection_control_t* control) {
  return (struct rdma_conn_param){
      .private_data = control->handshake.local,
      .private_data_len = control->handshake.local_length,
      .responder_resources = 0,
      .initiator_depth = 0,
      .retry_count = 7,
      .rnr_retry_count = 7,
      .qp_num = control->queue->qp_num,
  };
}

static iree_status_t iree_net_rdma_connection_control_connect_queue(
    iree_net_rdma_connection_control_t* control) {
  IREE_RETURN_IF_ERROR(iree_net_rdma_connection_route_initialize(
      control->context, control->id, &control->route));
  // CM supplies the control queue's negotiated identities. Independent data
  // queues will instead apply this route with their own exchanged identities.
  struct ibv_qp_attr receive = control->route.receive.attributes;
  struct ibv_qp_attr send = control->route.send.attributes;
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  receive.min_rnr_timer = control->options.minimum_rnr_timer;
  int error = library->ibv_modify_qp(control->queue, &receive,
                                     control->route.receive.mask);
  if (!error) {
    error =
        library->ibv_modify_qp(control->queue, &send, control->route.send.mask);
  }
  if (error) {
    return iree_net_rdma_connection_control_native_error("ibv_modify_qp",
                                                         error);
  }
  return iree_ok_status();
}

static void iree_net_rdma_connection_control_ready(
    iree_net_rdma_connection_control_t* control) {
  control->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_ESTABLISHED;
  control->state = IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED;
  control->callbacks.on_ready(
      control->callbacks.user_data,
      iree_make_const_byte_span(control->handshake.peer,
                                control->handshake.peer_length));
  iree_net_rdma_connection_control_dispatch_records(control);
}

static iree_status_t iree_net_rdma_connection_control_disconnect(
    iree_net_rdma_connection_control_t* control) {
  control->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_DISCONNECT_REQUESTED;
  if (iree_net_rdma_context_library(control->context)
          ->rdma_disconnect(control->id)) {
    return iree_net_rdma_connection_control_native_error("rdma_disconnect",
                                                         errno);
  }
  return iree_ok_status();
}

static void iree_net_rdma_connection_control_event(
    void* user_data, const struct rdma_cm_event* event) {
  iree_net_rdma_connection_control_t* control = user_data;
  if (control->state >= IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DRAINING ||
      !iree_status_is_ok(control->failure)) {
    return;
  }
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  iree_status_t status = iree_ok_status();
  switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
      if (library->rdma_resolve_route(
              control->id, (int)control->options.resolution_timeout_ms)) {
        status = iree_net_rdma_connection_control_native_error(
            "rdma_resolve_route", errno);
      }
      break;
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
      status = iree_net_rdma_connection_control_create_queue(control);
      if (iree_status_is_ok(status)) {
        struct rdma_conn_param parameters =
            iree_net_rdma_connection_control_parameters(control);
        if (library->rdma_connect(control->id, &parameters)) {
          status = iree_net_rdma_connection_control_native_error("rdma_connect",
                                                                 errno);
        }
      }
      break;
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
      control->handshake.peer_length = event->param.conn.private_data_len;
      if (control->handshake.peer_length) {
        memcpy(control->handshake.peer, event->param.conn.private_data,
               control->handshake.peer_length);
      }
      status = iree_net_rdma_connection_control_connect_queue(control);
      if (iree_status_is_ok(status) && library->rdma_establish(control->id)) {
        status = iree_net_rdma_connection_control_native_error("rdma_establish",
                                                               errno);
      }
      if (iree_status_is_ok(status)) {
        iree_net_rdma_connection_control_ready(control);
      }
      break;
    case RDMA_CM_EVENT_ESTABLISHED:
      iree_net_rdma_connection_control_ready(control);
      break;
    case RDMA_CM_EVENT_DISCONNECTED:
      // The passive side must answer DREQ; recording departure alone leaves
      // the native initiator waiting for DREP.
      if (!iree_any_bit_set(
              control->flags,
              IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_DISCONNECT_REQUESTED)) {
        status = iree_net_rdma_connection_control_disconnect(control);
      }
      status = iree_status_join(
          status,
          iree_make_status(IREE_STATUS_UNAVAILABLE, "RDMA peer disconnected"));
      break;
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
      break;
    default:
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "RDMA connection event %u failed (%d)",
                                (unsigned)event->event, event->status);
      break;
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
  }
}

static iree_status_t iree_net_rdma_connection_control_start(
    iree_net_rdma_connection_control_t* control) {
  control->state = IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTING;
  iree_net_rdma_completion_queue_options_t options = {
      .capacity = control->options.send_count + control->options.receive_count +
                  control->options.data_work_capacity,
      .service_batch_size = control->options.service_batch_size,
      .mode = control->options.completion_mode,
  };
  IREE_RETURN_IF_ERROR(iree_net_rdma_completion_queue_create(
      control->context, control->proactor, options,
      (iree_net_rdma_completion_queue_callbacks_t){
          .on_completions = iree_net_rdma_connection_control_completions,
          .on_error = iree_net_rdma_connection_control_service_failed,
          .user_data = control,
      },
      control->host_allocator, &control->completions));
  IREE_RETURN_IF_ERROR(iree_net_rdma_device_failure_create(
      control->context, control->proactor,
      iree_net_rdma_completion_queue_handle(control->completions),
      control->options.service_batch_size,
      iree_net_rdma_connection_control_service_failed, control,
      control->host_allocator, &control->device_failure));
  return iree_net_rdma_connection_events_create(
      control->context, control->proactor, control->options.service_batch_size,
      (iree_net_rdma_connection_events_callbacks_t){
          .on_event = iree_net_rdma_connection_control_event,
          .on_error = iree_net_rdma_connection_control_service_failed,
          .user_data = control,
      },
      control->host_allocator, &control->events);
}

void iree_net_rdma_connection_control_connect(
    iree_net_rdma_connection_control_t* control,
    const iree_async_address_t* address) {
  iree_status_t status = iree_net_rdma_connection_control_start(control);
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  if (iree_status_is_ok(status) &&
      library->rdma_create_id(
          iree_net_rdma_connection_events_handle(control->events), &control->id,
          control, RDMA_PS_TCP)) {
    status =
        iree_net_rdma_connection_control_native_error("rdma_create_id", errno);
  }
  if (iree_status_is_ok(status) &&
      library->rdma_resolve_addr(control->id, NULL,
                                 (struct sockaddr*)address->storage,
                                 (int)control->options.resolution_timeout_ms)) {
    status = iree_net_rdma_connection_control_native_error("rdma_resolve_addr",
                                                           errno);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_connection_events_activate(control->events);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
  }
}

void iree_net_rdma_connection_control_accept(
    iree_net_rdma_connection_control_t* control, struct rdma_cm_id* id,
    iree_const_byte_span_t private_data) {
  control->id = id;
  control->handshake.peer_length = (uint8_t)private_data.data_length;
  if (private_data.data_length) {
    memcpy(control->handshake.peer, private_data.data,
           private_data.data_length);
  }
  iree_status_t status = iree_net_rdma_connection_control_start(control);
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  if (iree_status_is_ok(status) &&
      library->rdma_migrate_id(
          id, iree_net_rdma_connection_events_handle(control->events))) {
    status =
        iree_net_rdma_connection_control_native_error("rdma_migrate_id", errno);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_connection_control_create_queue(control);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_connection_control_connect_queue(control);
  }
  if (iree_status_is_ok(status)) {
    struct rdma_conn_param parameters =
        iree_net_rdma_connection_control_parameters(control);
    if (library->rdma_accept(id, &parameters)) {
      status =
          iree_net_rdma_connection_control_native_error("rdma_accept", errno);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_connection_events_activate(control->events);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
  }
}

bool iree_net_rdma_connection_control_try_send(
    iree_net_rdma_connection_control_t* control,
    const uint8_t record[IREE_NET_RDMA_CONTROL_RECORD_SIZE]) {
  if (control->state != IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CONNECTED ||
      !iree_status_is_ok(control->failure) || !control->free_send_count) {
    return false;
  }
  uint32_t index = control->free_send_slots[--control->free_send_count];
  memcpy((uint8_t*)control->region->base_ptr +
             (iree_host_size_t)index * IREE_NET_RDMA_CONTROL_RECORD_SIZE,
         record, IREE_NET_RDMA_CONTROL_RECORD_SIZE);
  struct ibv_sge span =
      iree_net_rdma_connection_control_record_span(control, index);
  struct ibv_send_wr request = {
      .wr_id = IREE_NET_RDMA_CONTROL_WORK_ID_BIT | index,
      .sg_list = &span,
      .num_sge = 1,
      .opcode = IBV_WR_SEND,
      .send_flags = IBV_SEND_SIGNALED,
  };
  struct ibv_send_wr* rejected = NULL;
  int error = ibv_post_send(control->queue, &request, &rejected);
  if (error) {
    iree_net_rdma_connection_control_fail(
        control,
        iree_net_rdma_connection_control_native_error("ibv_post_send", error));
  }
  return true;
}

struct ibv_cq* iree_net_rdma_connection_control_completion_queue(
    iree_net_rdma_connection_control_t* control) {
  return iree_net_rdma_completion_queue_handle(control->completions);
}

const iree_net_rdma_connection_route_t* iree_net_rdma_connection_control_route(
    iree_net_rdma_connection_control_t* control) {
  return &control->route;
}

static void iree_net_rdma_connection_control_joined(void* user_data,
                                                    iree_status_t status) {
  iree_net_rdma_connection_control_t* control = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
    return;
  }
  if (--control->pending_joins != 0) {
    return;
  }
  control->state = IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DEACTIVATED;
  control->deactivated_callback.fn(control->deactivated_callback.user_data,
                                   iree_ok_status());
}

static void iree_net_rdma_connection_control_deactivate_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_connection_control_t* control = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_control_fail(control, status);
  }
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(control->context);
  if (control->id) {
    if (iree_any_bit_set(control->flags,
                         IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_ESTABLISHED) &&
        !iree_any_bit_set(
            control->flags,
            IREE_NET_RDMA_CONNECTION_CONTROL_FLAG_DISCONNECT_REQUESTED)) {
      status = iree_net_rdma_connection_control_disconnect(control);
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connection_control_fail(control, status);
      }
    }
    if (control->queue) {
      // Native destruction establishes quiescence even when setup/posting
      // failed partway through. Control records need no guessed flush count.
      int error = library->ibv_destroy_qp(control->queue);
      if (error) {
        iree_status_abort(iree_net_rdma_connection_control_native_error(
            "destroying control QP", error));
      }
      control->queue = NULL;
    }
    if (library->rdma_destroy_id(control->id)) {
      iree_status_abort(iree_net_rdma_connection_control_native_error(
          "destroying control CM ID", errno));
    }
    control->id = NULL;
  }
  // Count every join before starting any: POSIX monitor retirement can invoke
  // its callback inline. The NOP body itself holds the final count.
  control->pending_joins = 1 + (control->events != NULL) +
                           (control->completions != NULL) +
                           (control->device_failure != NULL);
  iree_async_event_source_unregistered_callback_t callback = {
      .fn = iree_net_rdma_connection_control_joined,
      .user_data = control,
  };
  if (control->events) {
    iree_net_rdma_connection_events_deactivate(control->events, callback);
  }
  if (control->completions) {
    iree_net_rdma_completion_queue_deactivate(control->completions, callback);
  }
  if (control->device_failure) {
    iree_net_rdma_device_failure_deactivate(control->device_failure, callback);
  }
  iree_net_rdma_connection_control_joined(control, iree_ok_status());
}

void iree_net_rdma_connection_control_deactivate(
    iree_net_rdma_connection_control_t* control,
    iree_async_event_source_unregistered_callback_t callback) {
  control->state = IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DRAINING;
  control->deactivated_callback = callback;
  iree_async_operation_initialize(
      &control->deactivate_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_rdma_connection_control_deactivate_complete, control);
  IREE_CHECK_OK(iree_async_proactor_submit_one(
      control->proactor, &control->deactivate_operation.base));
}

void iree_net_rdma_connection_control_destroy(
    iree_net_rdma_connection_control_t* control) {
  if (!control) {
    return;
  }
  IREE_ASSERT(
      control->state == IREE_NET_RDMA_CONNECTION_CONTROL_STATE_CREATED ||
          control->state == IREE_NET_RDMA_CONNECTION_CONTROL_STATE_DEACTIVATED,
      "started RDMA control must join deactivation before destruction");
  iree_net_rdma_connection_events_destroy(control->events);
  iree_net_rdma_device_failure_destroy(control->device_failure);
  iree_net_rdma_completion_queue_destroy(control->completions);
  iree_async_region_release(control->region);
  iree_status_free(control->failure);
  iree_net_rdma_context_release(control->context);
  iree_async_proactor_release(control->proactor);
  iree_allocator_free(control->host_allocator, control);
}

iree_status_t iree_net_rdma_connection_control_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    iree_net_rdma_connection_control_options_t options,
    iree_const_byte_span_t private_data,
    iree_net_rdma_connection_control_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_net_rdma_connection_control_t** out_control) {
  *out_control = NULL;
  uint64_t capacity = (uint64_t)options.send_count + options.receive_count +
                      options.data_work_capacity;
  if (!context || !proactor || !options.send_count || !options.receive_count ||
      !options.service_batch_size || options.service_batch_size > capacity ||
      capacity > INT_MAX || !options.resolution_timeout_ms ||
      options.resolution_timeout_ms > INT_MAX ||
      options.minimum_rnr_timer > 31 || !callbacks.on_ready ||
      !callbacks.on_record || !callbacks.on_capacity ||
      !callbacks.on_completions || !callbacks.on_error ||
      private_data.data_length > IREE_NET_RDMA_CONTROL_PRIVATE_DATA_SIZE ||
      (private_data.data_length && !private_data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid RDMA control configuration");
  }
  if (iree_net_rdma_context_device(context)->device->transport_type !=
      IBV_TRANSPORT_IB) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA host control requires IB/RoCE routing");
  }
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_rdma_connection_control_t), &allocation_size,
      IREE_STRUCT_FIELD_FAM(options.send_count + options.receive_count,
                            uint32_t)));
  iree_net_rdma_connection_control_t* control = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&control));
  control->host_allocator = host_allocator;
  control->context = context;
  iree_net_rdma_context_retain(context);
  control->proactor = proactor;
  iree_async_proactor_retain(proactor);
  control->options = options;
  control->callbacks = callbacks;
  control->free_send_count = options.send_count;
  control->receives.slots = control->free_send_slots + options.send_count;
  for (uint32_t i = 0; i < options.send_count; ++i) {
    control->free_send_slots[i] = i;
  }
  control->handshake.local_length = (uint8_t)private_data.data_length;
  if (private_data.data_length) {
    memcpy(control->handshake.local, private_data.data,
           private_data.data_length);
  }
  iree_async_slab_t* slab = NULL;
  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = IREE_NET_RDMA_CONTROL_RECORD_SIZE;
  slab_options.buffer_count = options.send_count + options.receive_count;
  iree_status_t status =
      iree_async_slab_create(slab_options, host_allocator, &slab);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_region_register_slab(
        context, slab, (uint64_t)(uintptr_t)slab->base_ptr,
        IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
            IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        host_allocator, &control->region);
  }
  iree_async_slab_release(slab);
  if (iree_status_is_ok(status)) {
    *out_control = control;
  } else {
    iree_net_rdma_connection_control_destroy(control);
  }
  return status;
}
