// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/net/channel_provider.h"

#include "iree/hal/net/channel.h"

typedef struct iree_hal_net_channel_provider_t {
  // Base HAL resource, at offset zero.
  iree_hal_resource_t resource;
  // Allocator owning this provider and its copied configuration.
  iree_allocator_t host_allocator;
  // Retained transport shared by constructed channels.
  iree_net_transport_factory_t* factory;
  // Retained service driven by the application.
  iree_async_proactor_t* proactor;
  // Immutable construction policy with owned address storage.
  iree_hal_net_channel_options_t* options;
} iree_hal_net_channel_provider_t;

IREE_API_EXPORT iree_hal_net_channel_options_t
iree_hal_net_channel_options_default(void) {
  return (iree_hal_net_channel_options_t){.rank = 0,
                                          .count = 2,
                                          .address = IREE_SV("127.0.0.1:0"),
                                          .tag_capacity = 64};
}

iree_status_t iree_hal_net_channel_options_clone(
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_net_channel_options_t** out_options) {
  *out_options = NULL;
  iree_host_size_t string_length = options->address.size;
  for (iree_host_size_t i = 0; i < options->peer_addresses.count; ++i) {
    if (!iree_host_size_checked_add(string_length,
                                    options->peer_addresses.values[i].size,
                                    &string_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "channel contacts exceed host address space");
    }
  }
  iree_host_size_t allocation_size = 0;
  iree_host_size_t addresses_offset = 0;
  iree_host_size_t strings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*options), &allocation_size,
      IREE_STRUCT_FIELD(options->peer_addresses.count, iree_string_view_t,
                        &addresses_offset),
      IREE_STRUCT_FIELD(string_length, char, &strings_offset)));
  iree_hal_net_channel_options_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&copy));
  *copy = *options;
  iree_string_view_t* addresses =
      (iree_string_view_t*)((uint8_t*)copy + addresses_offset);
  char* strings = (char*)copy + strings_offset;
  memcpy(strings, options->address.data, options->address.size);
  copy->address = iree_make_string_view(strings, options->address.size);
  strings += options->address.size;
  for (iree_host_size_t i = 0; i < options->peer_addresses.count; ++i) {
    const iree_string_view_t address = options->peer_addresses.values[i];
    if (address.size) {
      memcpy(strings, address.data, address.size);
    }
    addresses[i] = iree_make_string_view(strings, address.size);
    strings += address.size;
  }
  copy->peer_addresses.values = addresses;
  *out_options = copy;
  return iree_ok_status();
}

static void iree_hal_net_channel_provider_destroy(
    iree_hal_channel_provider_t* base_provider) {
  iree_hal_net_channel_provider_t* provider =
      (iree_hal_net_channel_provider_t*)base_provider;
  iree_net_transport_factory_release(provider->factory);
  iree_async_proactor_release(provider->proactor);
  iree_allocator_free(provider->host_allocator, provider->options);
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t iree_hal_net_channel_provider_query_default_rank_and_count(
    iree_hal_channel_provider_t* base_provider, int32_t* out_rank,
    int32_t* out_count) {
  const iree_hal_net_channel_provider_t* provider =
      (const iree_hal_net_channel_provider_t*)base_provider;
  if (out_rank) {
    *out_rank = provider->options->rank;
  }
  if (out_count) {
    *out_count = provider->options->count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_net_channel_provider_exchange_default_id(
    iree_hal_channel_provider_t* provider, iree_byte_span_t id) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "net channels use explicit group identities");
}

static iree_status_t iree_hal_net_channel_provider_create_channel(
    iree_hal_channel_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_net_channel_provider_t* provider =
      (iree_hal_net_channel_provider_t*)base_provider;
  if (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT) {
    params.rank = provider->options->rank;
  }
  if (params.count == IREE_HAL_CHANNEL_COUNT_DEFAULT) {
    params.count = provider->options->count;
  }
  if (params.rank != provider->options->rank ||
      params.count != provider->options->count ||
      params.flags != IREE_HAL_CHANNEL_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "net provider requires its configured rank and count "
        "without flags");
  }
  return iree_hal_net_channel_create(device, queue_family_affinity, params,
                                     provider->factory, provider->proactor,
                                     provider->options,
                                     provider->host_allocator, out_channel);
}

static const iree_hal_channel_provider_vtable_t
    iree_hal_net_channel_provider_vtable = {
        .destroy = iree_hal_net_channel_provider_destroy,
        .query_default_rank_and_count =
            iree_hal_net_channel_provider_query_default_rank_and_count,
        .exchange_default_id =
            iree_hal_net_channel_provider_exchange_default_id,
        .create_channel = iree_hal_net_channel_provider_create_channel,
};

IREE_API_EXPORT iree_status_t iree_hal_net_channel_provider_create(
    iree_net_transport_factory_t* factory, iree_async_proactor_t* proactor,
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_channel_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(factory);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (options->rank < 0 || options->count <= 0 ||
      options->rank >= options->count || !options->tag_capacity ||
      !options->address.data || iree_string_view_is_empty(options->address)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid net channel rank, address or tag capacity");
  }
  if (options->peer_addresses.count != (iree_host_size_t)options->rank ||
      (options->rank && !options->peer_addresses.values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "net channel requires one contact per lower rank");
  }
  for (int32_t rank = 0; rank < options->rank; ++rank) {
    const iree_string_view_t address = options->peer_addresses.values[rank];
    if (!address.data || iree_string_view_is_empty(address)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "net channel peer %d has no contact", rank);
    }
  }
  if (!iree_all_bits_set(iree_net_transport_factory_query_capabilities(factory),
                         IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
                             IREE_NET_TRANSPORT_CAPABILITY_ORDERED)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "channel requires reliable ordered messages");
  }
  iree_hal_net_channel_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_resource_initialize(&iree_hal_net_channel_provider_vtable,
                               &provider->resource);
  provider->host_allocator = host_allocator;
  provider->factory = factory;
  iree_net_transport_factory_retain(factory);
  provider->proactor = proactor;
  iree_async_proactor_retain(proactor);
  iree_status_t status = iree_hal_net_channel_options_clone(
      options, host_allocator, &provider->options);
  if (iree_status_is_ok(status)) {
    *out_provider = (iree_hal_channel_provider_t*)provider;
  } else {
    iree_hal_channel_provider_release((iree_hal_channel_provider_t*)provider);
  }
  return status;
}
