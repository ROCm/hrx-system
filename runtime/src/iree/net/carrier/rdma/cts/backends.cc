// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA carrier CTS backend registration.
//
// Registers the RDMA transport through the generic carrier/factory CTS. The
// backend derives a routable numeric address for the selected RDMA device so
// wildcard listener bindings do not accidentally turn into unroutable
// 0.0.0.0 connects.

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "iree/async/address.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/rdma/context.h"
#include "iree/net/carrier/rdma/factory.h"
#include "iree/net/carrier/rdma/region.h"
#include "iree/net/connection.h"
#include "iree/net/transport_factory.h"

namespace iree::net::carrier::cts {
namespace {

static constexpr const char* kRdmaCtsBindAddressEnv =
    "IREE_NET_RDMA_CTS_BIND_ADDRESS";
static constexpr const char* kRdmaCtsDeviceEnv = "IREE_NET_RDMA_CTS_DEVICE";
static constexpr const char* kRdmaCtsGidIndexEnv =
    "IREE_NET_RDMA_CTS_GID_INDEX";
static constexpr const char* kRdmaCtsHostEnv = "IREE_NET_RDMA_CTS_HOST";
static constexpr const char* kRdmaCtsPortEnv = "IREE_NET_RDMA_CTS_PORT";
static constexpr const char* kRdmaCtsUnreachableAddressEnv =
    "IREE_NET_RDMA_CTS_UNREACHABLE_ADDRESS";

static constexpr uint32_t kRdmaCtsEndpointCount = 4;
static constexpr uint32_t kRdmaCtsQueueDepth = 64;
static constexpr uint32_t kRdmaCtsBufferSize = 128 * 1024;

const char* GetEnvironmentValue(const char* name) {
  const char* value = std::getenv(name);
  return value && value[0] ? value : nullptr;
}

iree_status_t ParseEnvironmentUint8(const char* name, uint8_t* out_value) {
  const char* value = GetEnvironmentValue(name);
  if (!value) return iree_ok_status();

  errno = 0;
  char* end = nullptr;
  unsigned long parsed = strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s must be an integer in [0, 255]", name);
  }
  *out_value = (uint8_t)parsed;
  return iree_ok_status();
}

iree_status_t ConfigureRdmaFactoryOptions(
    iree_net_rdma_factory_options_t* out_options) {
  *out_options = iree_net_rdma_factory_options_default();
  out_options->max_endpoint_count = kRdmaCtsEndpointCount;
  out_options->carrier_options.send_queue_depth = kRdmaCtsQueueDepth;
  out_options->carrier_options.recv_queue_depth = kRdmaCtsQueueDepth;
  out_options->carrier_options.recv_buffer_size = kRdmaCtsBufferSize;
  out_options->carrier_options.send_staging_buffer_size = kRdmaCtsBufferSize;

  const char* device_name = GetEnvironmentValue(kRdmaCtsDeviceEnv);
  if (device_name) {
    out_options->context_options.device_name =
        iree_make_cstring_view(device_name);
  }
  IREE_RETURN_IF_ERROR(ParseEnvironmentUint8(
      kRdmaCtsPortEnv, &out_options->context_options.port_number));
  IREE_RETURN_IF_ERROR(ParseEnvironmentUint8(
      kRdmaCtsGidIndexEnv, &out_options->context_options.gid_index));
  return iree_ok_status();
}

void TrimAsciiWhitespace(std::string* value) {
  size_t begin = 0;
  while (begin < value->size() &&
         ((*value)[begin] == ' ' || (*value)[begin] == '\t' ||
          (*value)[begin] == '\n' || (*value)[begin] == '\r')) {
    ++begin;
  }
  size_t end = value->size();
  while (end > begin &&
         ((*value)[end - 1] == ' ' || (*value)[end - 1] == '\t' ||
          (*value)[end - 1] == '\n' || (*value)[end - 1] == '\r')) {
    --end;
  }
  *value = value->substr(begin, end - begin);
}

iree_status_t ReadTextFile(const std::string& path, std::string* out_value) {
  out_value->clear();
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "file not found: %s",
                            path.c_str());
  }

  std::getline(stream, *out_value);
  TrimAsciiWhitespace(out_value);
  if (out_value->empty()) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "file is empty: %s",
                            path.c_str());
  }
  return iree_ok_status();
}

iree_status_t ReadFirstDirectoryEntry(const std::string& path,
                                      std::string* out_value) {
  out_value->clear();
  DIR* directory = opendir(path.c_str());
  if (!directory) {
    return iree_status_from_errno(__FILE__, __LINE__, errno, "opendir");
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    errno = 0;
    dirent* entry = readdir(directory);
    if (!entry) {
      if (errno != 0) {
        status = iree_status_from_errno(__FILE__, __LINE__, errno, "readdir");
      }
      break;
    }
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    *out_value = entry->d_name;
    break;
  }

  int close_result = closedir(directory);
  if (close_result != 0) {
    status = iree_status_join(
        status, iree_status_from_errno(__FILE__, __LINE__, errno, "closedir"));
  }
  if (iree_status_is_ok(status) && out_value->empty()) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "directory has no entries: %s", path.c_str());
  }
  return status;
}

iree_status_t GetRdmaNetworkDeviceName(iree_net_rdma_context_t* context,
                                       std::string* out_network_device_name) {
  iree_string_view_t device_view = iree_net_rdma_context_device_name(context);
  std::string rdma_device_name(device_view.data, device_view.size);
  uint8_t port_number = iree_net_rdma_context_port_number(context);
  uint8_t gid_index = iree_net_rdma_context_gid_index(context);

  std::string gid_network_device_path =
      "/sys/class/infiniband/" + rdma_device_name + "/ports/" +
      std::to_string(port_number) + "/gid_attrs/ndevs/" +
      std::to_string(gid_index);
  iree_status_t status =
      ReadTextFile(gid_network_device_path, out_network_device_name);
  if (iree_status_is_ok(status)) return status;
  IREE_ATTRIBUTE_UNUSED iree::Status gid_status(std::move(status));

  std::string fallback_path =
      "/sys/class/infiniband/" + rdma_device_name + "/device/net";
  status = ReadFirstDirectoryEntry(fallback_path, out_network_device_name);
  if (!iree_status_is_ok(status)) return status;
  return iree_ok_status();
}

socklen_t SockaddrLength(const sockaddr* address) {
  switch (address->sa_family) {
    case AF_INET:
      return sizeof(sockaddr_in);
    case AF_INET6:
      return sizeof(sockaddr_in6);
    default:
      return 0;
  }
}

iree_status_t FindInterfaceAddressFamily(const std::string& interface_name,
                                         int address_family,
                                         std::string* out_host) {
  out_host->clear();
  ifaddrs* interface_addresses = nullptr;
  if (getifaddrs(&interface_addresses) != 0) {
    return iree_status_from_errno(__FILE__, __LINE__, errno, "getifaddrs");
  }

  iree_status_t status = iree_ok_status();
  for (ifaddrs* interface_address = interface_addresses; interface_address;
       interface_address = interface_address->ifa_next) {
    if (!interface_address->ifa_addr ||
        interface_address->ifa_addr->sa_family != address_family ||
        strcmp(interface_address->ifa_name, interface_name.c_str()) != 0 ||
        !iree_all_bits_set(interface_address->ifa_flags, IFF_UP)) {
      continue;
    }

    char host_buffer[NI_MAXHOST];
    socklen_t address_length = SockaddrLength(interface_address->ifa_addr);
    if (address_length == 0) continue;
    int result =
        getnameinfo(interface_address->ifa_addr, address_length, host_buffer,
                    sizeof(host_buffer), nullptr, 0, NI_NUMERICHOST);
    if (result != 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "getnameinfo failed for interface %s: %s",
                                interface_name.c_str(), gai_strerror(result));
      break;
    }
    *out_host = host_buffer;
    break;
  }

  freeifaddrs(interface_addresses);
  if (iree_status_is_ok(status) && out_host->empty()) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND, "interface %s has no usable %s address",
        interface_name.c_str(), address_family == AF_INET ? "IPv4" : "IPv6");
  }
  return status;
}

iree_status_t FindInterfaceAddress(const std::string& interface_name,
                                   std::string* out_host) {
  iree_status_t status =
      FindInterfaceAddressFamily(interface_name, AF_INET, out_host);
  if (iree_status_is_ok(status)) return status;
  IREE_ATTRIBUTE_UNUSED iree::Status ipv4_status(std::move(status));

  status = FindInterfaceAddressFamily(interface_name, AF_INET6, out_host);
  if (iree_status_is_ok(status)) return status;
  IREE_ATTRIBUTE_UNUSED iree::Status ipv6_status(std::move(status));
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "interface %s has no routable numeric address",
                          interface_name.c_str());
}

iree_status_t DiscoverRdmaConnectHost(
    iree_net_rdma_context_options_t context_options, std::string* out_host) {
  out_host->clear();
  iree_net_rdma_context_t* context = nullptr;
  iree_status_t status = iree_net_rdma_context_create(
      context_options, iree_allocator_system(), &context);

  std::string network_device_name;
  if (iree_status_is_ok(status)) {
    status = GetRdmaNetworkDeviceName(context, &network_device_name);
  }
  if (iree_status_is_ok(status)) {
    status = FindInterfaceAddress(network_device_name, out_host);
  }
  iree_net_rdma_context_release(context);
  return status;
}

iree_status_t GetRdmaConnectHost(std::string* out_host) {
  const char* host = GetEnvironmentValue(kRdmaCtsHostEnv);
  if (host) {
    *out_host = host;
    return iree_ok_status();
  }

  iree_net_rdma_factory_options_t options;
  IREE_RETURN_IF_ERROR(ConfigureRdmaFactoryOptions(&options));
  return DiscoverRdmaConnectHost(options.context_options, out_host);
}

std::string FormatHostPort(const std::string& host, uint16_t port) {
  std::string port_string = std::to_string(port);
  if (!host.empty() && host.front() == '[') {
    return host + ":" + port_string;
  }
  if (host.find(':') != std::string::npos) {
    return "[" + host + "]:" + port_string;
  }
  return host + ":" + port_string;
}

iree_status_t QueryListenerPort(iree_net_listener_t* listener,
                                uint16_t* out_port) {
  *out_port = 0;
  char address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address_string;
  IREE_RETURN_IF_ERROR(iree_net_listener_query_bound_address(
      listener, sizeof(address_buffer), address_buffer, &address_string));

  iree_async_address_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_from_string(address_string, &address));
  const sockaddr* socket_address =
      reinterpret_cast<const sockaddr*>(address.storage);
  switch (socket_address->sa_family) {
    case AF_INET: {
      const sockaddr_in* ipv4_address =
          reinterpret_cast<const sockaddr_in*>(socket_address);
      *out_port = ntohs(ipv4_address->sin_port);
      return iree_ok_status();
    }
    case AF_INET6: {
      const sockaddr_in6* ipv6_address =
          reinterpret_cast<const sockaddr_in6*>(socket_address);
      *out_port = ntohs(ipv6_address->sin6_port);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported RDMA listener address family %d",
                              (int)socket_address->sa_family);
  }
}

iree_status_t PollUntil(iree_async_proactor_t* proactor,
                        const std::function<bool()>& condition) {
  while (!condition()) {
    iree_host_size_t completed_count = 0;
    iree_status_t poll_status = PollProactorOnce(proactor, &completed_count);
    if (!iree_status_is_ok(poll_status)) return poll_status;
  }
  return iree_ok_status();
}

iree_status_t StopAndFreeListener(iree_async_proactor_t* proactor,
                                  iree_net_listener_t* listener) {
  if (!listener) return iree_ok_status();

  struct StopState {
    // True once the listener stop callback has fired.
    bool fired = false;
  } stop_state;
  iree_status_t status = iree_net_listener_stop(
      listener, {[](void* user_data) {
                   auto* state = static_cast<StopState*>(user_data);
                   state->fired = true;
                 },
                 &stop_state});
  if (iree_status_is_ok(status)) {
    status = PollUntil(proactor, [&]() { return stop_state.fired; });
  }
  if (iree_status_is_ok(status)) {
    iree_net_listener_free(listener);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Factory-level CTS support
//===----------------------------------------------------------------------===//

iree_status_t CreateRdmaCtsAvailabilityStatus();

static iree_status_t CreateUncheckedRdmaFactory(
    iree_allocator_t allocator, iree_net_transport_factory_t** out_factory) {
  iree_net_rdma_factory_options_t options;
  IREE_RETURN_IF_ERROR(ConfigureRdmaFactoryOptions(&options));
  if (!GetEnvironmentValue(kRdmaCtsHostEnv)) {
    std::string host;
    IREE_RETURN_IF_ERROR(
        DiscoverRdmaConnectHost(options.context_options, &host));
  }
  return iree_net_rdma_factory_create(options, allocator, out_factory);
}

static iree_status_t CreateRdmaFactory(
    iree_allocator_t allocator, iree_net_transport_factory_t** out_factory) {
  IREE_RETURN_IF_ERROR(CreateRdmaCtsAvailabilityStatus());
  return CreateUncheckedRdmaFactory(allocator, out_factory);
}

static iree_status_t CreateRdmaRegisteredRegion(
    iree_net_carrier_t* carrier, iree_host_size_t byte_length,
    iree_async_buffer_access_flags_t access_flags,
    iree_allocator_t host_allocator, RegisteredRegion* out_region) {
  memset(out_region, 0, sizeof(*out_region));
  iree_net_rdma_carrier_t* rdma_carrier = iree_net_rdma_carrier_cast(carrier);
  iree_net_rdma_context_t* context =
      iree_net_rdma_carrier_context(rdma_carrier);
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier is not an RDMA carrier");
  }

  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = byte_length;
  slab_options.buffer_count = 1;

  iree_status_t status =
      iree_async_slab_create(slab_options, host_allocator, &out_region->slab);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_region_register_slab(context, out_region->slab,
                                                access_flags, host_allocator,
                                                &out_region->region);
  }
  if (!iree_status_is_ok(status)) {
    iree_async_region_release(out_region->region);
    iree_async_slab_release(out_region->slab);
    memset(out_region, 0, sizeof(*out_region));
  }
  return status;
}

static iree::StatusOr<std::string> MakeRdmaBindAddress() {
  const char* bind_address = GetEnvironmentValue(kRdmaCtsBindAddressEnv);
  if (bind_address) return bind_address;

  std::string host;
  IREE_RETURN_IF_ERROR(GetRdmaConnectHost(&host));
  return host.find(':') != std::string::npos ? "[::]:0" : "0.0.0.0:0";
}

static iree::StatusOr<std::string> ResolveRdmaConnectAddress(
    const std::string& /*bind_address*/, iree_net_listener_t* listener) {
  uint16_t port = 0;
  IREE_RETURN_IF_ERROR(QueryListenerPort(listener, &port));

  std::string host;
  IREE_RETURN_IF_ERROR(GetRdmaConnectHost(&host));
  return FormatHostPort(host, port);
}

static iree::StatusOr<std::string> MakeRdmaUnreachableAddress(
    iree_async_proactor_t* /*proactor*/) {
  const char* address = GetEnvironmentValue(kRdmaCtsUnreachableAddressEnv);
  if (address) return address;

  std::string host;
  IREE_RETURN_IF_ERROR(GetRdmaConnectHost(&host));
  return FormatHostPort(host, 1);
}

//===----------------------------------------------------------------------===//
// Carrier-pair CTS support
//===----------------------------------------------------------------------===//

struct RdmaPairContext {
  // Factory shared by the client and server connections.
  iree_net_transport_factory_t* factory = nullptr;

  // Client-side connection retaining the returned carrier view.
  iree_net_connection_t* client_connection = nullptr;

  // Server-side connection retaining the returned carrier view.
  iree_net_connection_t* server_connection = nullptr;

  ~RdmaPairContext() {
    iree_net_connection_release(client_connection);
    iree_net_connection_release(server_connection);
    iree_net_transport_factory_release(factory);
  }
};

void CleanupRdmaPair(void* context) {
  delete static_cast<RdmaPairContext*>(context);
}

struct ConnectionState {
  // Number of accept/connect callbacks observed by the polling thread.
  std::atomic<int> completion_count{0};

  // Status reported by the listener accept callback.
  iree::Status accept_status;

  // Status reported by the client connect callback.
  iree::Status connect_status;

  // Accepted server-side connection, owned by this state until transferred.
  iree_net_connection_t* server_connection = nullptr;

  // Connected client-side connection, owned by this state until transferred.
  iree_net_connection_t* client_connection = nullptr;
};

void AcceptConnection(void* user_data, iree_status_t status,
                      iree_net_connection_t* connection) {
  auto* state = static_cast<ConnectionState*>(user_data);
  state->accept_status = iree::Status(std::move(status));
  state->server_connection = connection;
  state->completion_count.fetch_add(1, std::memory_order_release);
}

void ConnectConnection(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
  auto* state = static_cast<ConnectionState*>(user_data);
  state->connect_status = iree::Status(std::move(status));
  state->client_connection = connection;
  state->completion_count.fetch_add(1, std::memory_order_release);
}

iree_status_t ProbeRdmaCtsConnection() {
  iree_async_proactor_t* proactor = nullptr;
  iree_net_transport_factory_t* factory = nullptr;
  iree_net_listener_t* listener = nullptr;
  ConnectionState connection_state;

  iree_status_t status =
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor);
  if (iree_status_is_ok(status)) {
    status = CreateUncheckedRdmaFactory(iree_allocator_system(), &factory);
  }
  if (iree_status_is_ok(status)) {
    iree::StatusOr<std::string> bind_address = MakeRdmaBindAddress();
    if (bind_address.ok()) {
      status = iree_net_transport_factory_create_listener(
          factory, iree_make_cstring_view(bind_address.value().c_str()),
          proactor, /*recv_pool=*/nullptr, AcceptConnection, &connection_state,
          iree_allocator_system(), &listener);
    } else {
      status = std::move(bind_address).status();
    }
  }
  if (iree_status_is_ok(status)) {
    iree::StatusOr<std::string> connect_address =
        ResolveRdmaConnectAddress("", listener);
    if (connect_address.ok()) {
      status = iree_net_transport_factory_connect(
          factory, iree_make_cstring_view(connect_address.value().c_str()),
          proactor, /*recv_pool=*/nullptr, ConnectConnection,
          &connection_state);
    } else {
      status = std::move(connect_address).status();
    }
  }
  if (iree_status_is_ok(status)) {
    status = PollUntil(proactor, [&]() {
      return connection_state.completion_count.load(
                 std::memory_order_acquire) >= 2;
    });
  }
  if (iree_status_is_ok(status) && !connection_state.accept_status.ok()) {
    status = std::move(connection_state.accept_status);
  }
  if (iree_status_is_ok(status) && !connection_state.connect_status.ok()) {
    status = std::move(connection_state.connect_status);
  }

  iree_status_t stop_status = StopAndFreeListener(proactor, listener);
  listener = nullptr;
  if (!iree_status_is_ok(stop_status)) {
    status = iree_status_join(status, stop_status);
  }
  iree_net_connection_release(connection_state.client_connection);
  iree_net_connection_release(connection_state.server_connection);
  iree_net_transport_factory_release(factory);
  iree_async_proactor_release(proactor);
  return status;
}

struct RdmaCtsAvailability {
  // Status code returned by the cached availability probe.
  iree_status_code_t status_code = IREE_STATUS_OK;

  // Diagnostic copied from the first availability probe failure.
  std::string message;
};

RdmaCtsAvailability& GetRdmaCtsAvailability() {
  static RdmaCtsAvailability availability;
  return availability;
}

std::once_flag& GetRdmaCtsAvailabilityOnce() {
  static std::once_flag availability_once;
  return availability_once;
}

void ProbeRdmaCtsAvailabilityOnce() {
  iree_status_t status = ProbeRdmaCtsConnection();
  if (iree_status_is_ok(status)) return;

  RdmaCtsAvailability& availability = GetRdmaCtsAvailability();
  availability.status_code = iree_status_code(status);
  iree_string_view_t message = iree_status_message(status);
  if (iree_string_view_is_empty(message)) {
    availability.message = iree_status_code_string(availability.status_code);
  } else {
    availability.message.assign(message.data, message.size);
  }
  status = iree_status_ignore(status);
  if (availability.status_code == IREE_STATUS_DEADLINE_EXCEEDED) {
    availability.status_code = IREE_STATUS_UNAVAILABLE;
    availability.message =
        "RDMA CTS backend unavailable: " + availability.message;
  }
}

iree_status_t CreateRdmaCtsAvailabilityStatus() {
  std::call_once(GetRdmaCtsAvailabilityOnce(), ProbeRdmaCtsAvailabilityOnce);

  const RdmaCtsAvailability& availability = GetRdmaCtsAvailability();
  if (availability.status_code == IREE_STATUS_OK) return iree_ok_status();
  return iree_make_status(availability.status_code, "%s",
                          availability.message.c_str());
}

static iree::StatusOr<CarrierPair> CreateRdmaCarrierPair(
    iree_async_proactor_t* proactor) {
  iree_async_proactor_t* owned_proactor = nullptr;
  if (!proactor) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &owned_proactor));
    proactor = owned_proactor;
  }

  auto context = std::make_unique<RdmaPairContext>();
  iree_status_t status =
      CreateRdmaFactory(iree_allocator_system(), &context->factory);
  iree_net_listener_t* listener = nullptr;
  ConnectionState connection_state;
  if (iree_status_is_ok(status)) {
    iree::StatusOr<std::string> bind_address = MakeRdmaBindAddress();
    if (bind_address.ok()) {
      status = iree_net_transport_factory_create_listener(
          context->factory,
          iree_make_cstring_view(bind_address.value().c_str()), proactor,
          /*recv_pool=*/nullptr, AcceptConnection, &connection_state,
          iree_allocator_system(), &listener);
    } else {
      status = std::move(bind_address).status();
    }
  }

  if (iree_status_is_ok(status)) {
    iree::StatusOr<std::string> connect_address =
        ResolveRdmaConnectAddress("", listener);
    if (connect_address.ok()) {
      status = iree_net_transport_factory_connect(
          context->factory,
          iree_make_cstring_view(connect_address.value().c_str()), proactor,
          /*recv_pool=*/nullptr, ConnectConnection, &connection_state);
    } else {
      status = std::move(connect_address).status();
    }
  }
  if (iree_status_is_ok(status)) {
    status = PollUntil(proactor, [&]() {
      int completion_count =
          connection_state.completion_count.load(std::memory_order_acquire);
      return completion_count >= 2 ||
             (completion_count >= 1 && (!connection_state.accept_status.ok() ||
                                        !connection_state.connect_status.ok()));
    });
  }
  if (iree_status_is_ok(status) && !connection_state.accept_status.ok()) {
    status = std::move(connection_state.accept_status);
  }
  if (iree_status_is_ok(status) && !connection_state.connect_status.ok()) {
    status = std::move(connection_state.connect_status);
  }
  if (iree_status_is_ok(status)) {
    context->client_connection = connection_state.client_connection;
    context->server_connection = connection_state.server_connection;
    connection_state.client_connection = nullptr;
    connection_state.server_connection = nullptr;
  }

  iree_status_t stop_status = StopAndFreeListener(proactor, listener);
  listener = nullptr;
  if (!iree_status_is_ok(stop_status)) {
    status = iree_status_join(status, stop_status);
  }

  iree_net_carrier_t* client_carrier = nullptr;
  iree_net_carrier_t* server_carrier = nullptr;
  if (iree_status_is_ok(status)) {
    client_carrier = iree_net_connection_carrier(context->client_connection);
    server_carrier = iree_net_connection_carrier(context->server_connection);
    if (!client_carrier || !server_carrier) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "RDMA connection did not expose endpoint carriers");
    }
  }

  if (iree_status_is_ok(status)) {
    iree_net_carrier_retain(client_carrier);
    iree_net_carrier_retain(server_carrier);

    CarrierPair pair;
    pair.client = client_carrier;
    pair.server = server_carrier;
    pair.proactor = proactor;
    pair.context = context.release();
    pair.cleanup = CleanupRdmaPair;
    owned_proactor = nullptr;
    return pair;
  }

  iree_net_connection_release(connection_state.client_connection);
  iree_net_connection_release(connection_state.server_connection);
  iree_async_proactor_release(owned_proactor);
  return iree::Status(std::move(status));
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

static bool rdma_registered =
    (CtsRegistry::RegisterBackend(
         {"rdma",
          {"rdma", CreateRdmaCarrierPair, CreateRdmaFactory,
           MakeRdmaBindAddress, ResolveRdmaConnectAddress,
           MakeRdmaUnreachableAddress, CreateRdmaRegisteredRegion},
          {"reliable", "ordered", "zerocopy_tx", "zerocopy_rx",
           "registered_regions", "direct_write", "direct_read", "factory"}}),
     true);

}  // namespace
}  // namespace iree::net::carrier::cts
