// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/websocket/handshake.h"

#include <string.h>

#include "iree/base/internal/base64.h"

static uint32_t loom_serve_websocket_rotate(uint32_t value, uint32_t count) {
  return (value << count) | (value >> (32 - count));
}

// RFC 6455's fixed 60-byte challenge occupies exactly two SHA-1 blocks after
// padding. This is a protocol challenge, not an authentication primitive.
static void loom_serve_websocket_challenge(iree_string_view_t key,
                                           uint8_t digest[20]) {
  uint8_t blocks[128] = {0};
  memcpy(blocks, key.data, 24);
  memcpy(blocks + 24, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", 36);
  blocks[60] = 0x80;
  blocks[126] = 0x01;
  blocks[127] = 0xE0;  // 60 * 8 bits.
  uint32_t state[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u,
                       0xC3D2E1F0u};
  const uint32_t constants[4] = {0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu,
                                 0xCA62C1D6u};
  for (iree_host_size_t block = 0; block < 2; ++block) {
    uint32_t words[80];
    for (iree_host_size_t i = 0; i < 16; ++i) {
      const uint8_t* bytes = blocks + block * 64 + i * 4;
      words[i] = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
                 ((uint32_t)bytes[2] << 8) | bytes[3];
    }
    for (iree_host_size_t i = 16; i < 80; ++i) {
      words[i] = loom_serve_websocket_rotate(
          words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16], 1);
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3],
             e = state[4];
    for (iree_host_size_t i = 0; i < 80; ++i) {
      uint32_t combine = 0;
      switch (i / 20) {
        case 0:
          combine = (b & c) | (~b & d);
          break;
        case 2:
          combine = (b & c) | (b & d) | (c & d);
          break;
        default:
          combine = b ^ c ^ d;
          break;
      }
      const uint32_t next = loom_serve_websocket_rotate(a, 5) + combine + e +
                            constants[i / 20] + words[i];
      e = d;
      d = c;
      c = loom_serve_websocket_rotate(b, 30);
      b = a;
      a = next;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
  }
  for (iree_host_size_t i = 0; i < 20; ++i) {
    digest[i] = (uint8_t)(state[i / 4] >> (24 - 8 * (i % 4)));
  }
}

static bool loom_serve_websocket_has_token(iree_string_view_t list,
                                           iree_string_view_t token,
                                           bool case_sensitive) {
  while (!iree_string_view_is_empty(list)) {
    iree_string_view_t value;
    iree_string_view_split(list, ',', &value, &list);
    value = iree_string_view_trim(value);
    if (case_sensitive ? iree_string_view_equal(value, token)
                       : iree_string_view_equal_case(value, token)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_serve_websocket_handshake(
    const loom_serve_http_request_t* request,
    iree_string_view_list_t allowed_origins, iree_string_view_t subprotocol,
    iree_string_builder_t* response) {
  iree_string_view_t value = iree_string_view_empty();
  if (!iree_string_view_equal(request->method, IREE_SV("GET")) ||
      !iree_string_view_is_empty(request->body) ||
      !loom_serve_http_request_lookup_header(request, IREE_SV("Host"),
                                             &value) ||
      iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket requires a body-free GET with Host");
  }
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Upgrade"),
                                             &value) ||
      !loom_serve_websocket_has_token(value, IREE_SV("websocket"), false) ||
      !loom_serve_http_request_lookup_header(request, IREE_SV("Connection"),
                                             &value) ||
      !loom_serve_websocket_has_token(value, IREE_SV("upgrade"), false)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket requires Upgrade and Connection tokens");
  }
  if (!loom_serve_http_request_lookup_header(
          request, IREE_SV("Sec-WebSocket-Version"), &value) ||
      !iree_string_view_equal(value, IREE_SV("13"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket requires version 13");
  }
  if (loom_serve_http_request_lookup_header(request, IREE_SV("Origin"),
                                            &value)) {
    bool allowed = false;
    for (iree_host_size_t i = 0; i < allowed_origins.count; ++i) {
      allowed |= iree_string_view_equal(value, allowed_origins.values[i]);
    }
    if (!allowed) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "WebSocket Origin is not allowed");
    }
  }
  iree_string_view_t key;
  if (!loom_serve_http_request_lookup_header(
          request, IREE_SV("Sec-WebSocket-Key"), &key) ||
      key.size != 24 || key.data[22] != '=' || key.data[23] != '=') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket key must encode a 16-byte nonce");
  }
  uint8_t nonce[16];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_base64_decode(
      key, iree_make_byte_span(nonce, sizeof(nonce)), &length));
  if (length != sizeof(nonce)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket key must encode a 16-byte nonce");
  }
  uint8_t digest[20];
  char accept[28];
  loom_serve_websocket_challenge(key, digest);
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      iree_make_const_byte_span(digest, sizeof(digest)),
      iree_make_mutable_string_view(accept, sizeof(accept)), &length));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      response,
      "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
      "Connection: Upgrade\r\nSec-WebSocket-Accept: %.*s\r\n",
      (int)length, accept));
  if (!iree_string_view_is_empty(subprotocol) &&
      loom_serve_http_request_lookup_header(
          request, IREE_SV("Sec-WebSocket-Protocol"), &value) &&
      loom_serve_websocket_has_token(value, subprotocol, true)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        response, "Sec-WebSocket-Protocol: %.*s\r\n", (int)subprotocol.size,
        subprotocol.data));
  }
  return iree_string_builder_append_cstring(response, "\r\n");
}
