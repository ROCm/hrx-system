// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/alignment.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/net/demo/worker.h"

IREE_FLAG(int32_t, rank, 0, "Local worker rank in [0, count).");
IREE_FLAG(int32_t, count, 2, "Number of workers in [2, 8].");
IREE_FLAG(string, address, "127.0.0.1:0", "Local numeric bind host:port.");
IREE_FLAG_LIST(string, peer_address,
               "Lower-rank worker contacts, repeated in rank order.");
IREE_FLAG(int32_t, iterations, 8,
          "Alternating projection requests while another consumer is held.");

enum {
  DEMO_ALLOCATED,
  DEMO_PRODUCED,
  DEMO_CONSUMED,
  DEMO_RELEASED,
  DEMO_COMPLETION_COUNT,
  DEMO_MAX_RANKS = 8,
  DEMO_MAX_PAYLOAD_BYTES = 13 * 24 * sizeof(float),
};

typedef struct demo_projection_t {
  // Rows of the input and output matrices.
  uint32_t rows;
  // Columns of the weight and output matrices.
  uint32_t columns;
  // Full reduction dimension partitioned across all workers.
  uint32_t depth;
} demo_projection_t;

typedef struct demo_peer_t {
  // Reusable send plan; reserves no tensor storage.
  iree_hal_channel_transfer_t* send;
  // Reusable receive plan over the same shared workspace contract.
  iree_hal_channel_transfer_t* receive;
  // Source-return observer for the admitted send.
  demo_completion_t sent;
  // Data-readiness observer for the admitted receive.
  demo_completion_t received;
} demo_peer_t;

typedef struct demo_lane_t {
  // Cold communication plans and observers indexed by remote rank.
  demo_peer_t* peers;
  // Number of ranks sharing this projection, including the local worker.
  uint32_t count;
  // Independent receipts for every admitted operation.
  demo_completion_t completions[DEMO_COMPLETION_COUNT];
  // Explicit final-consumer permission, independent of network arrival.
  iree_hal_semaphore_t* consumer_gate;
  // Current queue allocation epoch, owned until all operations retire.
  iree_hal_buffer_t* buffer;
  // Monotonic exchange identity within this tag.
  uint64_t sequence;
  // Current model shape, stable until the queued callbacks retire.
  demo_projection_t shape;
  // Local shard of the projection.
  uint32_t rank;
  // Request seed shared by all workers.
  uint32_t request;
  // Current payload layout within this request's shared-slab range.
  struct {
    // Source starts at this changing byte offset.
    iree_device_size_t offset;
    // Aligned spacing between each rank partial and the final output.
    iree_device_size_t stride;
    // Bytes exchanged in each direction.
    iree_device_size_t length;
  } payload;
  // Final-output checksum, published by the consumer's completion.
  double checksum;
} demo_lane_t;

// Small exactly representable values make full-output checking independent of
// floating-point reassociation between the sharded and unsharded sums.
static float demo_input(uint32_t row, uint32_t depth, uint32_t request) {
  return ((int)((row * 3 + depth * 5 + request) % 13) - 6) * 0.25f;
}

static float demo_weight(uint32_t depth, uint32_t column) {
  return ((int)((depth * 7 + column * 3) % 11) - 5) * 0.125f;
}

static float demo_residual(uint32_t row, uint32_t column, uint32_t request) {
  return ((int)((row + column + request) % 7) - 3) * 0.5f;
}

static float* demo_payload(demo_lane_t* lane, uint32_t index) {
  const iree_hal_buffer_native_binding_slot_t slot = {
      .index = lane->buffer->memory.contract->binding_layout.host_binding_index,
      .type = IREE_HAL_BUFFER_INTERFACE_HOST};
  uint8_t* base =
      iree_hal_buffer_native_binding(lane->buffer, slot).host_pointer;
  return (float*)(base + lane->payload.offset + index * lane->payload.stride);
}

static iree_status_t demo_project(void* user_data, const uint64_t args[4],
                                  iree_hal_host_call_context_t* context) {
  demo_lane_t* lane = user_data;
  float* partial = demo_payload(lane, lane->rank);
  const uint32_t begin = lane->shape.depth * lane->rank / lane->count;
  const uint32_t end = lane->shape.depth * (lane->rank + 1) / lane->count;
  for (uint32_t row = 0; row < lane->shape.rows; ++row) {
    for (uint32_t column = 0; column < lane->shape.columns; ++column) {
      float value = 0;
      for (uint32_t k = begin; k < end; ++k) {
        value += demo_input(row, k, lane->request) * demo_weight(k, column);
      }
      partial[row * lane->shape.columns + column] = value;
    }
  }
  return iree_ok_status();
}

// This application-authored epilogue performs the reduction, residual and
// final check in one queued consumer. Communication imposes no reduction ABI.
static iree_status_t demo_consume(void* user_data, const uint64_t args[4],
                                  iree_hal_host_call_context_t* context) {
  demo_lane_t* lane = user_data;
  float* output = demo_payload(lane, lane->count);
  lane->checksum = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t row = 0; iree_status_is_ok(status) && row < lane->shape.rows;
       ++row) {
    for (uint32_t column = 0;
         iree_status_is_ok(status) && column < lane->shape.columns; ++column) {
      const uint32_t index = row * lane->shape.columns + column;
      const float residual = demo_residual(row, column, lane->request);
      float reduced = 0;
      for (uint32_t rank = 0; rank < lane->count; ++rank) {
        reduced += demo_payload(lane, rank)[index];
      }
      output[index] = reduced + residual;
      float expected = 0;
      for (uint32_t k = 0; k < lane->shape.depth; ++k) {
        expected += demo_input(row, k, lane->request) * demo_weight(k, column);
      }
      expected += residual;
      if (output[index] != expected) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS, "request %u output [%u,%u]: %g != %g",
            lane->request, row, column, output[index], expected);
      }
      lane->checksum += output[index];
    }
  }
  return status;
}

static iree_status_t demo_lane_initialize(demo_worker_t* worker, uint32_t rank,
                                          uint32_t count, uint32_t tag,
                                          demo_lane_t* lane) {
  lane->rank = rank;
  lane->count = count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             count * sizeof(*lane->peers),
                                             (void**)&lane->peers));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < DEMO_COMPLETION_COUNT;
       ++i) {
    status = demo_completion_initialize(worker, &lane->completions[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      worker->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, tag,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &lane->consumer_gate));
  for (uint32_t peer_rank = 0; iree_status_is_ok(status) && peer_rank < count;
       ++peer_rank) {
    if (peer_rank == rank) {
      continue;
    }
    demo_peer_t* peer = &lane->peers[peer_rank];
    status = demo_completion_initialize(worker, &peer->sent);
    if (iree_status_is_ok(status)) {
      status = demo_completion_initialize(worker, &peer->received);
    }
    iree_hal_channel_transfer_params_t params = {
        .mode = IREE_HAL_CHANNEL_TRANSFER_MODE_SEND,
        .peer_rank = (int32_t)peer_rank,
        .tag = tag,
        .pool = worker->workspace,
        .capacity = DEMO_MAX_PAYLOAD_BYTES};
    if (iree_status_is_ok(status)) {
      status = iree_hal_channel_prepare_transfer(worker->channel, &params,
                                                 &peer->send);
    }
    if (iree_status_is_ok(status)) {
      params.mode = IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE;
      status = iree_hal_channel_prepare_transfer(worker->channel, &params,
                                                 &peer->receive);
    }
  }
  return status;
}

static iree_status_t demo_allocate(demo_worker_t* worker, demo_lane_t* lane) {
  demo_completion_t* allocated = &lane->completions[DEMO_ALLOCATED];
  IREE_RETURN_IF_ERROR(demo_completion_arm(worker, allocated));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size =
          lane->payload.offset + (lane->count + 1) * lane->payload.stride,
      .min_alignment = 64};
  return demo_completion_admit(
      allocated,
      iree_hal_queue_alloca(worker->queue, iree_hal_semaphore_list_empty(),
                            demo_completion_point(allocated), worker->workspace,
                            1, &request, &lane->buffer));
}

static iree_status_t demo_exchange(demo_worker_t* worker, demo_lane_t* lane,
                                   iree_hal_semaphore_list_t send_waits) {
  iree_status_t status = iree_ok_status();
  for (uint32_t rank = 0; iree_status_is_ok(status) && rank < lane->count;
       ++rank) {
    if (rank == lane->rank) {
      continue;
    }
    demo_peer_t* peer = &lane->peers[rank];
    status = demo_completion_arm(worker, &peer->received);
    if (iree_status_is_ok(status)) {
      status = demo_completion_admit(
          &peer->received,
          iree_hal_queue_channel_recv(
              worker->queue,
              demo_completion_point(&lane->completions[DEMO_ALLOCATED]),
              demo_completion_point(&peer->received), peer->receive,
              lane->sequence, lane->buffer,
              lane->payload.offset + rank * lane->payload.stride,
              lane->payload.length));
    }
    if (iree_status_is_ok(status)) {
      status = demo_completion_arm(worker, &peer->sent);
    }
    if (iree_status_is_ok(status)) {
      status = demo_completion_admit(
          &peer->sent,
          iree_hal_queue_channel_send(
              worker->queue, send_waits, demo_completion_point(&peer->sent),
              peer->send, lane->sequence, lane->buffer,
              lane->payload.offset + lane->rank * lane->payload.stride,
              lane->payload.length));
    }
  }
  return status;
}

static iree_status_t demo_release(demo_worker_t* worker, demo_lane_t* lane,
                                  iree_hal_semaphore_list_t waits) {
  demo_completion_t* released = &lane->completions[DEMO_RELEASED];
  IREE_RETURN_IF_ERROR(demo_completion_arm(worker, released));
  return demo_completion_admit(
      released, iree_hal_queue_dealloca(worker->queue, waits,
                                        demo_completion_point(released), 1,
                                        &lane->buffer));
}

static iree_status_t demo_submit(demo_worker_t* worker, demo_lane_t* lane,
                                 uint32_t request) {
  const demo_projection_t shapes[] = {{8, 16, 32}, {13, 24, 48}};
  lane->shape = shapes[request % IREE_ARRAYSIZE(shapes)];
  lane->request = request;
  ++lane->sequence;
  lane->payload.length = lane->shape.rows * lane->shape.columns * sizeof(float);
  lane->payload.stride = iree_device_align(lane->payload.length, 64);
  lane->payload.offset = 64 * (1 + request % 4);
  IREE_RETURN_IF_ERROR(demo_allocate(worker, lane));
  demo_completion_t* produced = &lane->completions[DEMO_PRODUCED];
  IREE_RETURN_IF_ERROR(demo_completion_arm(worker, produced));
  const uint64_t args[4] = {0};
  IREE_RETURN_IF_ERROR(demo_completion_admit(
      produced, iree_hal_queue_host_call(
                    worker->queue,
                    demo_completion_point(&lane->completions[DEMO_ALLOCATED]),
                    demo_completion_point(produced),
                    iree_hal_make_host_call(demo_project, lane), args,
                    IREE_HAL_HOST_CALL_FLAG_NONE)));
  IREE_RETURN_IF_ERROR(
      demo_exchange(worker, lane, demo_completion_point(produced)));
  demo_completion_t* consumed = &lane->completions[DEMO_CONSUMED];
  IREE_RETURN_IF_ERROR(demo_completion_arm(worker, consumed));
  iree_hal_semaphore_t* dependencies[DEMO_MAX_RANKS + 1];
  uint64_t values[DEMO_MAX_RANKS + 1];
  dependencies[0] = produced->semaphore;
  values[0] = produced->value;
  iree_host_size_t count = 1;
  for (uint32_t rank = 0; rank < lane->count; ++rank) {
    if (rank == lane->rank) {
      continue;
    }
    dependencies[count] = lane->peers[rank].received.semaphore;
    values[count++] = lane->peers[rank].received.value;
  }
  dependencies[count] = lane->consumer_gate;
  values[count++] = 1;
  IREE_RETURN_IF_ERROR(demo_completion_admit(
      consumed, iree_hal_queue_host_call(
                    worker->queue,
                    (iree_hal_semaphore_list_t){count, dependencies, values},
                    demo_completion_point(consumed),
                    iree_hal_make_host_call(demo_consume, lane), args,
                    IREE_HAL_HOST_CALL_FLAG_NONE)));
  dependencies[0] = consumed->semaphore;
  values[0] = consumed->value;
  count = 1;
  for (uint32_t rank = 0; rank < lane->count; ++rank) {
    if (rank == lane->rank) {
      continue;
    }
    dependencies[count] = lane->peers[rank].sent.semaphore;
    values[count++] = lane->peers[rank].sent.value;
  }
  return demo_release(worker, lane,
                      (iree_hal_semaphore_list_t){count, dependencies, values});
}

// Success joins every operation. Failure returns promptly so the caller can
// close the channel and revoke held consumer gates before joining everything.
static iree_status_t demo_await(demo_worker_t* worker, demo_lane_t* lane) {
  bool pending = true;
  iree_status_t status = iree_ok_status();
  while (pending && iree_status_is_ok(status)) {
    pending = false;
    for (uint32_t i = 0; i < DEMO_COMPLETION_COUNT; ++i) {
      pending |= lane->completions[i].pending;
      status = iree_status_join(status,
                                iree_status_clone(lane->completions[i].status));
    }
    for (uint32_t rank = 0; rank < lane->count; ++rank) {
      const demo_peer_t* peer = &lane->peers[rank];
      pending |= peer->sent.pending || peer->received.pending;
      status = iree_status_join(status, iree_status_clone(peer->sent.status));
      status =
          iree_status_join(status, iree_status_clone(peer->received.status));
    }
    if (pending && iree_status_is_ok(status)) {
      demo_worker_poll(worker);
    }
  }
  return status;
}

static iree_status_t demo_finish(demo_worker_t* worker, demo_lane_t* lane) {
  IREE_RETURN_IF_ERROR(demo_await(worker, lane));
  iree_hal_buffer_release(lane->buffer);
  lane->buffer = NULL;
  printf("rank=%u request=%u shape=%ux%ux%u checked=%u checksum=%.8g\n",
         lane->rank, lane->request, lane->shape.rows, lane->shape.depth,
         lane->shape.columns, lane->shape.rows * lane->shape.columns,
         lane->checksum);
  fflush(stdout);
  return iree_ok_status();
}

static iree_status_t demo_run(demo_worker_t* worker, demo_lane_t lanes[2],
                              uint32_t iterations) {
  IREE_RETURN_IF_ERROR(demo_submit(worker, &lanes[0], 0));
  iree_status_t status = iree_ok_status();
  for (uint32_t request = 1; iree_status_is_ok(status) && request <= iterations;
       ++request) {
    status = demo_submit(worker, &lanes[1], request);
    if (iree_status_is_ok(status)) {
      status = demo_finish(worker, &lanes[1]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  uint64_t consumed = 0;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
      lanes[0].completions[DEMO_CONSUMED].semaphore, &consumed));
  if (consumed != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "held consumer ran before its dependency");
  }
  printf("rank=%u independent_requests=%u held_consumer=pending\n",
         lanes[0].rank, iterations);
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_signal(lanes[0].consumer_gate, 1, NULL));
  IREE_RETURN_IF_ERROR(demo_finish(worker, &lanes[0]));
  iree_hal_pool_trim(worker->workspace, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(worker->workspace, &stats);
  if (stats.reservation_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "idle workspace still has %u tensor reservations",
                            stats.reservation_count);
  }
  printf("rank=%u live_tensor_reservations=0 reusable_plans=%u\n",
         lanes[0].rank, 4 * (lanes[0].count - 1));
  printf("rank=%u ready_to_close=1\n", lanes[0].rank);
  fflush(stdout);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1 || FLAG_rank < 0 || FLAG_count < 2 ||
      FLAG_count > DEMO_MAX_RANKS || FLAG_rank >= FLAG_count ||
      FLAG_iterations < 1 || FLAG_iterations > 1000000) {
    fprintf(stderr,
            "Expected count in [2, 8], rank in [0, count), and iterations in "
            "[1, 1000000].\n");
    return 1;
  }
  demo_worker_t worker = {0};
  demo_lane_t lanes[2] = {0};
  const iree_hal_net_channel_options_t options = {
      .rank = FLAG_rank,
      .count = FLAG_count,
      .address = iree_make_cstring_view(FLAG_address),
      .peer_addresses = FLAG_peer_address_list(),
      .tag_capacity = 2};
  iree_status_t status = demo_worker_initialize(&options, &worker);
  for (uint32_t tag = 0; iree_status_is_ok(status) && tag < 2; ++tag) {
    status = demo_lane_initialize(&worker, (uint32_t)FLAG_rank,
                                  (uint32_t)FLAG_count, tag, &lanes[tag]);
  }
  if (iree_status_is_ok(status)) {
    status = demo_run(&worker, lanes, (uint32_t)FLAG_iterations);
  }
  if (iree_status_is_ok(status)) {
    // The launch owner joins every worker's final output before closing group
    // membership. This is a single teardown decision, outside all requests.
    char command[16];
    if (!fgets(command, sizeof(command), stdin) || strcmp(command, "close\n")) {
      status = iree_make_status(IREE_STATUS_ABORTED,
                                "coordinator did not confirm shutdown");
    }
  }
  if (!iree_status_is_ok(status)) {
    for (uint32_t tag = 0; tag < 2; ++tag) {
      if (lanes[tag].consumer_gate) {
        iree_hal_semaphore_fail(lanes[tag].consumer_gate,
                                iree_status_clone(status));
      }
    }
  }
  status = demo_worker_close(&worker, status);
  for (uint32_t tag = 0; tag < 2; ++tag) {
    demo_lane_t* lane = &lanes[tag];
    for (uint32_t i = 0; i < DEMO_COMPLETION_COUNT; ++i) {
      status = iree_status_join(
          status, demo_completion_join(&worker, &lane->completions[i]));
      iree_hal_semaphore_release(lane->completions[i].semaphore);
    }
    // Failed deallocation preserves the epoch. These references are dropped
    // only after every admitted borrower and channel callback has retired.
    iree_hal_buffer_release(lane->buffer);
    iree_hal_semaphore_release(lane->consumer_gate);
    for (uint32_t rank = 0; lane->peers && rank < lane->count; ++rank) {
      demo_peer_t* peer = &lane->peers[rank];
      status =
          iree_status_join(status, demo_completion_join(&worker, &peer->sent));
      status = iree_status_join(status,
                                demo_completion_join(&worker, &peer->received));
      iree_hal_semaphore_release(peer->sent.semaphore);
      iree_hal_semaphore_release(peer->received.semaphore);
      iree_hal_channel_transfer_release(peer->send);
      iree_hal_channel_transfer_release(peer->receive);
    }
    iree_allocator_free(iree_allocator_system(), lane->peers);
  }
  demo_worker_deinitialize(&worker);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  fprintf(stderr, "Worker verified X*W + residual. All work retired.\n");
  return 0;
}
