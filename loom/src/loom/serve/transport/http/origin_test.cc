// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/http/origin.h"

#include <string>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_status_t Check(const std::string& host, const std::string& headers,
                    loom_serve_http_origin_policy_t policy = {}) {
  const std::string wire =
      "GET /v1/observe HTTP/1.1\r\nHost: " + host + "\r\n" + headers + "\r\n";
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_allocate(
      {4096, 512, 32, 4096, 32}, iree_allocator_system(), &parser));
  const loom_serve_http_request_t* request = nullptr;
  iree_status_t status = loom_serve_http_request_parser_feed(
      parser, iree_make_const_byte_span(wire.data(), wire.size()), &request,
      nullptr);
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_origin_check(request, policy);
  }
  loom_serve_http_request_parser_free(parser);
  return status;
}

TEST(HttpOriginTest, NativeClientsNeedNoBrowserConfiguration) {
  IREE_EXPECT_OK(Check("local", ""));
  IREE_EXPECT_OK(Check("backend", "X-Forwarded-Proto: https\r\n"));
}

TEST(HttpOriginTest, DirectLocalAndNumericAuthoritiesWorkWithoutConfiguration) {
  const std::pair<const char*, const char*> cases[] = {
      {"localhost", "http://localhost"},
      {"LOCALHOST:80", "http://localhost"},
      {"localhost", "http://localhost:80"},
      {"localhost:19080", "http://localhost:19080"},
      {"127.0.0.1:13000", "http://127.0.0.1:13000"},
      {"127.0.0.2:8080", "http://127.0.0.2:8080"},
      {"192.0.2.1:8080", "http://192.0.2.1:8080"},
      {"[::1]:8080", "http://[::1]:8080"},
      {"[2001:DB8::1]", "http://[2001:db8::1]:80"},
      {"[::ffff:127.0.0.1]:8080", "http://[::ffff:127.0.0.1]:8080"},
  };
  for (const auto& [host, origin] : cases) {
    SCOPED_TRACE(origin);
    IREE_EXPECT_OK(Check(host, std::string("Origin: ") + origin + "\r\n"));
    IREE_EXPECT_OK(Check(host, std::string("Origin: ") + origin +
                                   "\r\nSec-Fetch-Site: same-origin\r\n"));
  }
}

TEST(HttpOriginTest, MatchingDnsNamesDoNotAuthorizeReboundBrowserRequests) {
  for (const char* host :
       {"attacker.example:8080", "localhost.attacker:8080",
        "127.0.0.1.attacker:8080", "serve.example", "localhost."}) {
    SCOPED_TRACE(host);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        Check(host, std::string("Origin: http://") + host +
                        "\r\nSec-Fetch-Site: same-origin\r\n"));
  }
}

TEST(HttpOriginTest, SchemeHostAndEffectivePortMustMatch) {
  for (const char* origin : {"http://localhost:8081", "https://localhost:8080",
                             "http://127.0.0.1:8080", "http://localhost",
                             "http://attacker.example:8080"}) {
    SCOPED_TRACE(origin);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        Check("localhost:8080", std::string("Origin: ") + origin + "\r\n"));
  }
  for (const char* site : {"same-site", "cross-site", "none", "unknown", ""}) {
    SCOPED_TRACE(site);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        Check("localhost", std::string("Origin: http://localhost\r\n"
                                       "Sec-Fetch-Site: ") +
                               site + "\r\n"));
  }
}

TEST(HttpOriginTest, RejectsMalformedAndOpaqueOriginsAndAuthorities) {
  for (const char* origin : {
           "",
           "null",
           "*",
           "file:///dashboard.html",
           "ws://localhost",
           "http://",
           "http://localhost/",
           "http://localhost?query",
           "http://localhost#fragment",
           "http://user@localhost",
           "http://localhost\\attacker",
           "http://localhost%2eattacker",
           "http://localhost http://attacker",
           "http://localhost,attacker",
           "http://localhost:",
           "http://localhost:-1",
           "http://localhost:+80",
           "http://localhost:65536",
           "http://localhost:999999999999999999999",
           "http://127.1",
           "http://127.0.0.999",
           "http://127.000.0.1",
           "http://[::1",
           "http://[]",
           "http://[::1]trailing",
           "http://[::1]:",
           "http://[not-an-address]",
           "http://[fe80::1%1]",
       }) {
    SCOPED_TRACE(origin);
    const auto value = iree_make_cstring_view(origin);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_http_origin_validate_allowed({1, &value}));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Check("localhost", std::string("Origin: ") + origin + "\r\n"));
  }
  for (const char* host : {"localhost:", "user@localhost", "[::1", "[]",
                           "localhost/path", "localhost:65536"}) {
    SCOPED_TRACE(host);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Check(host, "Origin: http://localhost\r\n"));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Check("localhost",
                              "Origin: http://localhost\r\n"
                              "origin: http://attacker.example\r\n"));
}

TEST(HttpOriginTest, ExtraOriginsAreExactAndExplicit) {
  const iree_string_view_t values[] = {IREE_SV("https://console.example"),
                                       IREE_SV("http://serve.example:8080")};
  const iree_string_view_list_t allowed = {IREE_ARRAYSIZE(values), values};
  IREE_ASSERT_OK(loom_serve_http_origin_validate_allowed(allowed));
  const loom_serve_http_origin_policy_t policy = {.allowed_origins = allowed};
  IREE_EXPECT_OK(Check("localhost",
                       "Origin: https://console.example\r\n"
                       "Sec-Fetch-Site: cross-site\r\n",
                       policy));
  IREE_EXPECT_OK(Check("serve.example:8080",
                       "Origin: http://serve.example:8080\r\n", policy));
  IREE_EXPECT_OK(Check("localhost", "Origin: http://localhost\r\n", policy));
  for (const char* origin :
       {"https://console.example.attacker", "http://console.example",
        "http://serve.example:8081"}) {
    SCOPED_TRACE(origin);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        Check("localhost", std::string("Origin: ") + origin + "\r\n", policy));
  }
}

TEST(HttpOriginTest, TrustedRelaySuppliesPublicAuthorityAndScheme) {
  const loom_serve_http_origin_policy_t relay = {
      .source = LOOM_SERVE_HTTP_ORIGIN_RELAY,
  };
  IREE_EXPECT_OK(
      Check("serve.example", "Origin: http://serve.example\r\n", relay));
  IREE_EXPECT_OK(Check("serve.example:443",
                       "Origin: https://serve.example\r\n"
                       "X-Forwarded-Proto: https\r\n"
                       "Sec-Fetch-Site: same-origin\r\n",
                       relay));
  IREE_EXPECT_OK(Check("serve.example:8443",
                       "Origin: https://serve.example:8443\r\n"
                       "X-Forwarded-Proto: https\r\n",
                       relay));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Check("serve.example",
                              "Origin: https://attacker.example\r\n"
                              "X-Forwarded-Proto: https\r\n",
                              relay));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      Check("serve.example", "Origin: https://serve.example\r\n", relay));
  for (const char* scheme : {"", "ftp", "https,http", "https, http"}) {
    SCOPED_TRACE(scheme);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Check("serve.example",
                                std::string("Origin: https://serve.example\r\n"
                                            "X-Forwarded-Proto: ") +
                                    scheme + "\r\n",
                                relay));
  }
}

TEST(HttpOriginTest, HeadersCannotCreateRelayTrustOrReplaceThePublicHost) {
  const std::string headers =
      "Origin: https://serve.example\r\n"
      "X-Forwarded-Proto: https\r\n"
      "X-Forwarded-Host: serve.example\r\n"
      "Forwarded: host=serve.example;proto=https\r\n"
      "Sec-Fetch-Site: same-origin\r\n";
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Check("serve.example", headers));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Check("localhost", headers));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      Check("localhost", headers, {.source = LOOM_SERVE_HTTP_ORIGIN_RELAY}));
}

}  // namespace
