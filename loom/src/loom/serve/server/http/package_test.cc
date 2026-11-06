// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/package.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string Part(const std::string& name, const std::string& filename,
                 const std::string& bytes) {
  return "--b\r\nContent-Disposition: form-data; name=\"" + name +
         "\"; filename=\"" + filename + "\"\r\n\r\n" + bytes + "\r\n";
}

class HttpPackageTest : public ::testing::Test {
 protected:
  void TearDown() override { ReleaseRequest(); }

  void ReleaseRequest() {
    loom_serve_http_request_parser_free(parser_);
    parser_ = nullptr;
  }

  iree_status_t Parse(const std::string& body, iree_allocator_t host_allocator,
                      iree_string_view_t* json,
                      loom_runtime_package_t** package) {
    ReleaseRequest();
    *json = iree_string_view_empty();
    *package = nullptr;
    const loom_serve_http_request_limits_t limits = {
        .header_byte_capacity = 16384,
        .body_byte_capacity = 8 * 1024 * 1024,
        .header_capacity = 64,
        .multipart_header_byte_capacity = 64 * 1024,
        .multipart_part_capacity = 1024,
    };
    IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_allocate(
        limits, host_allocator, &parser_));
    const std::string wire =
        "POST /v1/models/register HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: multipart/form-data; boundary=b\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
    const loom_serve_http_request_t* request = nullptr;
    IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_feed(
        parser_, iree_make_const_byte_span(wire.data(), wire.size()),
        &request));
    IREE_RETURN_IF_ERROR(
        loom_serve_http_request_parser_finalize(parser_, &request));
    return loom_serve_http_package_parse(request, host_allocator, json,
                                         package);
  }

  // Parsed request owns retained metadata and JSON through package creation.
  loom_serve_http_request_parser_t* parser_ = nullptr;
};

TEST_F(HttpPackageTest, ArbitraryPartOrderAndOwnedBinaryPayload) {
  const std::string bytes("\0\xff\r\n", 4);
  std::string body =
      Part("file", "model/provider.loombc", bytes) +
      Part("request", "model/model.json", "{\"model\":\"demo\"}") + "--b--\r\n";
  iree_string_view_t json;
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(Parse(body, iree_allocator_system(), &json, &package));
  EXPECT_EQ(std::string(json.data, json.size), "{\"model\":\"demo\"}");
  ReleaseRequest();
  body.assign(body.size(), '?');
  loomc_source_t* source = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_package_open(package, IREE_SV("provider.loombc"), &source));
  loom_runtime_package_release(package);
  const auto contents = loomc_source_contents(source);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(contents.data),
                        contents.data_length),
            bytes);
  loomc_source_release(source);
}

TEST_F(HttpPackageTest, ManyProvidersHaveBoundedAllocationAndExactContents) {
  const iree_allocator_t bounded_allocator = {
      nullptr,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** inout_pointer) -> iree_status_t {
        if (command != IREE_ALLOCATOR_COMMAND_FREE &&
            static_cast<const iree_allocator_alloc_params_t*>(params)
                    ->byte_length > 1024 * 1024) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "source package allocation exceeded 1 MiB");
        }
        const auto system = iree_allocator_system();
        return system.ctl(system.self, command, params, inout_pointer);
      }};
  std::string body = Part("request", "model/model.json", "{}");
  for (int i = 0; i < 128; ++i) {
    body += Part("file", "model/provider" + std::to_string(i) + ".loombc",
                 std::to_string(i) + std::string("\0\xff\r\n", 4));
  }
  body += "--b--\r\n";
  iree_string_view_t json;
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(Parse(body, bounded_allocator, &json, &package));
  for (int i = 0; i < 128; ++i) {
    const std::string name = "provider" + std::to_string(i) + ".loombc";
    loomc_source_t* source = nullptr;
    IREE_ASSERT_OK(loom_runtime_package_open(
        package, iree_make_cstring_view(name.c_str()), &source));
    const auto contents = loomc_source_contents(source);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(contents.data),
                          contents.data_length),
              std::to_string(i) + std::string("\0\xff\r\n", 4));
    loomc_source_release(source);
  }
  loom_runtime_package_release(package);
}

TEST_F(HttpPackageTest, InvalidPackagePublishesNothing) {
  const auto request = Part("request", "model.json", "{}");
  for (const auto& body : {
           Part("file", "a.loom", "source") + "--b--",
           request + request + "--b--",
           request + Part("unknown", "a.loom", "source") + "--b--",
           request + Part("file", "../a.loom", "source") + "--b--",
           request + Part("file", "a.loom", "one") +
               Part("file", "a.loom", "two") + "--b--",
           request + Part("file", "a.loom", "source"),
       }) {
    SCOPED_TRACE(body);
    iree_string_view_t json;
    loom_runtime_package_t* package = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(body, iree_allocator_system(), &json, &package));
    EXPECT_EQ(package, nullptr);
    EXPECT_EQ(json.size, 0u);
  }
}

TEST_F(HttpPackageTest, OrdinaryJsonHasNoPackage) {
  const loom_serve_http_request_t request = {.body = IREE_SV("{}")};
  iree_string_view_t json;
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_serve_http_package_parse(
      &request, iree_allocator_system(), &json, &package));
  EXPECT_EQ(package, nullptr);
  EXPECT_EQ(json.data, request.body.data);
}

}  // namespace
