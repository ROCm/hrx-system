// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Socket carrier implementing reliable ordered byte-stream transport over TCP
// (IPv4 or IPv6) and Unix domain stream sockets. No framing is added.
//
// One logical send owns the ordered socket write lane at a time. Once all its
// bytes are accepted, a zero-copy send may await source retirement in its own
// bounded slot while the next send starts. Partial writes retain the lane until
// their operation can be reused. Additional sends wait in a bounded FIFO.
// Registered regions are retained from admission through logical completion;
// payload bytes remain caller-owned under the generic asynchronous carrier
// contract. Send callbacks return source ownership and may run in a different
// order than the bytes were written.
//
// Receive progress uses one pool-backed operation. A consumer may move a
// receive lease out of its callback. The last available native buffer is
// delivered without a movable lease so retained data cannot block later
// control messages. Framing gives borrowed messages independent storage.

#ifndef IREE_NET_CARRIER_STREAM_CARRIER_H_
#define IREE_NET_CARRIER_STREAM_CARRIER_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default number of accepted sends per carrier.
#define IREE_NET_STREAM_DEFAULT_MAX_SEND_OPERATIONS 64u

// Default generated-prefix bytes retained per send without allocation.
#define IREE_NET_STREAM_DEFAULT_GENERATED_PREFIX_CAPACITY (16u * 1024u)

// Default minimum native send extent for requesting kernel copy avoidance.
#define IREE_NET_STREAM_DEFAULT_ZERO_COPY_MIN_SEND_SIZE (64u * 1024u)

// Options controlling bounded stream carrier resources and send policy.
typedef struct iree_net_stream_carrier_options_t {
  // Maximum accepted sends.
  uint32_t max_send_operations;

  // Generated-prefix bytes retained per admitted send without allocation.
  // Zero disables preallocated storage without limiting accepted messages;
  // larger prefixes allocate exact completion-scoped storage.
  uint32_t generated_prefix_capacity;

  // Minimum bytes in a native socket send for requesting zero-copy. Smaller
  // sends use kernel copying to avoid page-retirement overhead. The extent
  // includes any generated prefix; each scatter-vector or partial-write
  // continuation is selected independently. Zero permits all eligible sends;
  // IREE_HOST_SIZE_MAX disables zero-copy. The socket and proactor must still
  // support it. Both paths retain source ownership until the final callback.
  iree_host_size_t zero_copy_min_send_size;
} iree_net_stream_carrier_options_t;

// Returns default stream carrier options.
static inline iree_net_stream_carrier_options_t
iree_net_stream_carrier_options_default(void) {
  iree_net_stream_carrier_options_t options;
  options.max_send_operations = IREE_NET_STREAM_DEFAULT_MAX_SEND_OPERATIONS;
  options.generated_prefix_capacity =
      IREE_NET_STREAM_DEFAULT_GENERATED_PREFIX_CAPACITY;
  options.zero_copy_min_send_size =
      IREE_NET_STREAM_DEFAULT_ZERO_COPY_MIN_SEND_SIZE;
  return options;
}

// Creates a carrier over a connected TCP, TCP6, or UNIX_STREAM socket.
// Datagram sockets are not supported. Socket creation, connection
// establishment, and listener lifetime remain owned by the caller.
//
// |proactor| must own |socket| and the registered region backing
// |receive_pool|. The receive region must permit writes and contain at least
// one buffer. The pool is dedicated to this carrier while it is active; sharing
// its buffers with another I/O producer would violate lease exclusivity. The
// carrier retains all three resources without consuming the caller's
// references.
//
// The returned carrier begins in CREATED state. Install handlers and activate
// it before sending. The carrier accepts at most
// |options.max_send_operations| sends and at most
// IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS borrowed spans per send. A generated
// prefix does not reduce the accepted borrowed-span count. When the prefix plus
// borrowed spans exceed one physical socket vector the carrier continues the
// same logical send in another socket operation without copying the payload.
IREE_API_EXPORT iree_status_t iree_net_stream_carrier_create(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* receive_pool,
    const iree_net_stream_carrier_options_t* options,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_carrier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_STREAM_CARRIER_H_
