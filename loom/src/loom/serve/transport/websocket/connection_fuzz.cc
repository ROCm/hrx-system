// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "iree/base/threading/notification.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/http/server.h"
#include "loom/serve/transport/websocket/connection.h"
#include "loom/serve/transport/websocket/handshake.h"

namespace {

constexpr size_t kMaximumInputLength = 64 * 1024;

class Client {
 public:
  explicit Client(const iree_async_address_t* address) {
    const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
    descriptor_ = socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (descriptor_ < 0) {
      return;
    }
    int result;
    do {
      result = connect(descriptor_, native, address->length);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      close(descriptor_);
      descriptor_ = -1;
    }
  }

  ~Client() {
    if (descriptor_ >= 0) {
      close(descriptor_);
    }
  }

  bool connected() const { return descriptor_ >= 0; }

  bool Write(iree_string_view_t bytes) {
    size_t position = 0;
    while (position < bytes.size) {
      const ssize_t count = send(descriptor_, bytes.data + position,
                                 bytes.size - position, MSG_NOSIGNAL);
      if (count > 0) {
        position += static_cast<size_t>(count);
      } else if (count < 0 && errno == EINTR) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  }

  void Finish() { shutdown(descriptor_, SHUT_WR); }

 private:
  // Connected loopback peer, or -1 when native setup failed.
  int descriptor_ = -1;
};

class Environment {
 public:
  Environment() {
    IREE_CHECK_OK(iree_async_proactor_pool_create(
        2, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto server_options = loom_serve_http_server_options_default();
    server_options.connection_capacity = 4;
    IREE_CHECK_OK(loom_serve_http_server_allocate(
        pool_, &server_options, iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_CHECK_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, iree_allocator_system(), &listener_));
  }

  ~Environment() {
    IREE_CHECK_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
  }

  const iree_async_address_t* address() const {
    return loom_serve_http_listener_address(listener_);
  }

  loom_serve_http_server_t* server() const { return server_; }

 private:
  // Shared asynchronous I/O owner retained for the process campaign.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // HTTP connection owner reused after each claimed connection retires.
  loom_serve_http_server_t* server_ = nullptr;
  // Real ephemeral loopback listener feeding the shared server.
  loom_serve_http_listener_t* listener_ = nullptr;
};

Environment& GetEnvironment() {
  static Environment environment;
  return environment;
}

loom_serve_http_connection_t* AwaitRequest(
    loom_serve_http_server_t* server,
    const loom_serve_http_request_t** out_request) {
  loom_serve_http_connection_t* connection = nullptr;
  auto* notification = loom_serve_http_server_notification(server);
  while (!connection) {
    const auto token = iree_notification_prepare_wait(notification);
    connection = loom_serve_http_server_take_request(server, out_request);
    if (connection) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  return connection;
}

std::string ClientFrame(uint8_t type, iree_string_view_t payload,
                        bool final = true) {
  std::string frame(1, static_cast<char>((final ? 0x80 : 0) | type));
  if (payload.size < 126) {
    frame.push_back(static_cast<char>(0x80 | payload.size));
  } else {
    const size_t width = payload.size <= 65535 ? 2 : 8;
    frame.push_back(static_cast<char>(0x80 | (width == 2 ? 126 : 127)));
    for (size_t i = 0; i < width; ++i) {
      frame.push_back(static_cast<char>(static_cast<uint64_t>(payload.size) >>
                                        (8 * (width - i - 1))));
    }
  }
  constexpr uint8_t mask[] = {0x37, 0xFA, 0x21, 0x3D};
  frame.append(reinterpret_cast<const char*>(mask), sizeof(mask));
  for (size_t i = 0; i < payload.size; ++i) {
    frame.push_back(payload.data[i] ^ mask[i % sizeof(mask)]);
  }
  return frame;
}

std::string CloseFrame() {
  constexpr char payload[] = {'\x03', static_cast<char>(0xE8)};
  return ClientFrame(LOOM_SERVE_WEBSOCKET_CLOSE,
                     iree_make_string_view(payload, sizeof(payload)));
}

std::string GenerateWire(const uint8_t* data, size_t size) {
  const char mode = static_cast<char>(data[0]);
  const iree_string_view_t payload =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), size - 1);
  switch (mode) {
    case 't':
      return ClientFrame(LOOM_SERVE_WEBSOCKET_TEXT, payload) + CloseFrame();
    case 'b':
      return ClientFrame(LOOM_SERVE_WEBSOCKET_BINARY, payload) + CloseFrame();
    case 'f': {
      const iree_string_view_t first =
          iree_string_view_substr(payload, 0, payload.size / 2);
      const iree_string_view_t second = iree_string_view_substr(
          payload, payload.size / 2, IREE_HOST_SIZE_MAX);
      return ClientFrame(LOOM_SERVE_WEBSOCKET_TEXT, first, false) +
             ClientFrame(LOOM_SERVE_WEBSOCKET_PING, IREE_SV("ping")) +
             ClientFrame(0, second) + CloseFrame();
    }
    case 'c':
      return ClientFrame(LOOM_SERVE_WEBSOCKET_CLOSE, payload);
    case 'w':
      return std::string(payload.data, payload.size);
    default:
      return std::string(reinterpret_cast<const char*>(data), size);
  }
}

void AwaitPeerEnd(loom_serve_http_server_t* server,
                  loom_serve_http_connection_t* connection,
                  loom_serve_websocket_t* socket) {
  auto* notification = loom_serve_http_server_notification(server);
  while (!loom_serve_websocket_peer_closed(socket) &&
         !loom_serve_http_connection_failed(connection)) {
    const auto token = iree_notification_prepare_wait(notification);
    if (loom_serve_websocket_peer_closed(socket) ||
        loom_serve_http_connection_failed(connection)) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
}

void DrainMessages(loom_serve_websocket_t* socket) {
  while (true) {
    loom_serve_websocket_message_type_t type;
    iree_const_byte_span_t payload;
    loom_serve_websocket_message_t* message =
        loom_serve_websocket_take(socket, &type, &payload);
    if (!message) {
      return;
    }
    loom_serve_websocket_message_release(socket, message);
  }
}

void Exercise(iree_string_view_t wire) {
  Environment& environment = GetEnvironment();
  Client client(environment.address());
  if (!client.connected() ||
      !client.Write(IREE_SV(
          "GET /stream HTTP/1.1\r\nHost: local\r\nConnection: Upgrade\r\n"
          "Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n"
          "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"))) {
    return;
  }

  const loom_serve_http_request_t* request = nullptr;
  loom_serve_http_connection_t* connection =
      AwaitRequest(environment.server(), &request);
  char response_storage[512];
  iree_string_builder_t response;
  iree_string_builder_initialize_with_storage(
      response_storage, sizeof(response_storage), &response);
  IREE_CHECK_OK(loom_serve_websocket_handshake(
      request, loom_serve_http_connection_origin_policy(connection),
      iree_string_view_empty(), &response));

  const loom_serve_websocket_limits_t limits = {
      .message_byte_capacity = kMaximumInputLength,
      .retained_byte_capacity = 2 * kMaximumInputLength,
      .message_capacity = 64,
  };
  loom_serve_websocket_t* socket = nullptr;
  IREE_CHECK_OK(loom_serve_websocket_upgrade(connection, limits,
                                             iree_allocator_system(), &socket));
  IREE_CHECK_OK(loom_serve_http_connection_send(
      connection, iree_string_builder_view(&response)));
  client.Write(wire);
  client.Finish();
  AwaitPeerEnd(environment.server(), connection, socket);
  DrainMessages(socket);
  loom_serve_http_connection_abort(connection);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!size || size > kMaximumInputLength) {
    return 0;
  }
  const std::string wire = GenerateWire(data, size);
  Exercise(iree_make_string_view(wire.data(), wire.size()));
  return 0;
}
