// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-serve-device/transport.h"

#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

void ExpectBind(iree_string_view_t uri, iree_string_view_t transport_name,
                iree_string_view_t bind_address) {
  iree_serve_device_bind_t bind;
  IREE_ASSERT_OK(iree_serve_device_parse_bind_uri(uri, &bind));
  EXPECT_TRUE(iree_string_view_equal(bind.transport_name, transport_name));
  EXPECT_TRUE(iree_string_view_equal(bind.bind_address, bind_address));
}

TEST(ServeDeviceTransportTest, ParsesTcpBindUri) {
  ExpectBind(IREE_SV("tcp://0.0.0.0:5000"), IREE_SV("tcp"),
             IREE_SV("0.0.0.0:5000"));
  ExpectBind(IREE_SV("tcp://[::]:5000"), IREE_SV("tcp"), IREE_SV("[::]:5000"));
}

TEST(ServeDeviceTransportTest, ParsesShmBindUri) {
  ExpectBind(IREE_SV("shm:///dev/shm/iree-gpu"), IREE_SV("shm"),
             IREE_SV("/dev/shm/iree-gpu"));
}

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
TEST(ServeDeviceTransportTest, ParsesRdmaBindUri) {
  ExpectBind(IREE_SV("rdma://192.0.2.10:7471"), IREE_SV("rdma"),
             IREE_SV("192.0.2.10:7471"));
}
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

TEST(ServeDeviceTransportTest, RejectsBindUriWithoutKnownTransport) {
  iree_serve_device_bind_t bind;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_serve_device_parse_bind_uri(IREE_SV("0.0.0.0:5000"), &bind));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_serve_device_parse_bind_uri(IREE_SV("udp://0.0.0.0:5000"), &bind));
}

TEST(ServeDeviceTransportTest, ClassifiesLocalOnlyListeners) {
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(
                IREE_SV("tcp"), IREE_SV("127.0.0.1:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(
                IREE_SV("tcp"), IREE_SV("127.42.0.1:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(IREE_SV("tcp"),
                                                       IREE_SV("[::1]:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(
                IREE_SV("shm"), IREE_SV("/dev/shm/iree-gpu")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY);
}

TEST(ServeDeviceTransportTest, ClassifiesNetworkListeners) {
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(
                IREE_SV("tcp"), IREE_SV("192.0.2.10:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(
                IREE_SV("rdma"), IREE_SV("192.0.2.10:7471")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(IREE_SV("unknown"),
                                                       IREE_SV("endpoint")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK);
}

TEST(ServeDeviceTransportTest, ClassifiesWildcardListeners) {
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(IREE_SV("tcp"),
                                                       IREE_SV("0.0.0.0:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(IREE_SV("tcp"),
                                                       IREE_SV("[::]:5000")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD);
  EXPECT_EQ(iree_serve_device_classify_bind_visibility(IREE_SV("rdma"),
                                                       IREE_SV("0.0.0.0:7471")),
            IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD);
}

TEST(ServeDeviceTransportTest, CreatesTcpTransportFactory) {
  iree_net_transport_factory_t* factory = nullptr;
  IREE_ASSERT_OK(iree_serve_device_create_transport_factory(
      IREE_SV("tcp"), iree_allocator_system(), &factory));
  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
  iree_net_transport_factory_release(factory);
}

TEST(ServeDeviceTransportTest, CreatesShmTransportFactory) {
  iree_net_transport_factory_t* factory = nullptr;
  IREE_ASSERT_OK(iree_serve_device_create_transport_factory(
      IREE_SV("shm"), iree_allocator_system(), &factory));
  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
  iree_net_transport_factory_release(factory);
}

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
TEST(ServeDeviceTransportTest, CreatesRdmaTransportFactory) {
  iree_net_transport_factory_t* factory = nullptr;
  IREE_ASSERT_OK(iree_serve_device_create_transport_factory(
      IREE_SV("rdma"), iree_allocator_system(), &factory));
  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RDMA);
  iree_net_transport_factory_release(factory);
}
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

TEST(ServeDeviceTransportTest, RejectsUnsupportedTransportFactory) {
  iree_net_transport_factory_t* factory = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_serve_device_create_transport_factory(
                            IREE_SV("udp"), iree_allocator_system(), &factory));
  EXPECT_EQ(factory, nullptr);
}

}  // namespace
}  // namespace iree
