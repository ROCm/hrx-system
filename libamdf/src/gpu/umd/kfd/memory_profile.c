// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/memory_profile.h"

#include "amdf/gpu.h"
#include "libamdf/src/device_state.h"
#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/platform/linux/dma_buf.h"

static amdf_memory_site_description_t amdf_gpu_kfd_describe_program_site(
    amdf_memory_access_t access, const amdf_device_t* device,
    amdf_memory_atomic_capabilities_t operations) {
  return amdf_gpu_umd_memory_describe_program_site(access, &device->endpoint_id,
                                                   operations);
}

static amdf_memory_atomic_capabilities_t amdf_gpu_kfd_query_program_atomics(
    const amdf_memory_native_profile_t* profile, amdf_memory_flags_t flags) {
  const amdf_gpu_kfd_topology_t* topology = profile->construction.data;
  const amdf_gpu_kfd_ip_version_t* gc = &topology->gc_ip;
  if (!gc->exact || gc->major < 9 || gc->major > 12) {
    return (amdf_memory_atomic_capabilities_t){0};
  }
  // gmc_v9_0_get_coherence_flags and gmc_v11_0_get_vm_pte select UC for
  // explicit uncached and coherent USERPTR mappings. GC12 retains NC for
  // coherent mappings and uses the instruction scope to select atomic reach.
  const bool uncached = (flags & AMDF_MEMORY_FLAG_DEVICE_UNCACHED) != 0;
  const bool registered =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  const bool modern_cdna =
      gc->major == 9 &&
      (gc->minor > 4 || (gc->minor == 4 && gc->revision >= 3));
  const bool forwarded =
      uncached || (registered && gc->major < 12) || modern_cdna;
  // MI300+ implements unsupported outgoing PCIe atomics with device-atomic
  // load/op/store serialization. It does not thereby grant system atomicity.
  const bool device_integer = !forwarded || modern_cdna || gc->major == 12;
  const amdf_atomic_operations_t integer =
      amdf_gpu_umd_program_atomic_integer_operations();
  const amdf_atomic_operations_t remote_32 =
      topology->integrated && topology->host_atomics.supports_32
          ? integer
          : amdf_gpu_umd_program_atomic_pcie_operations(
                topology->host_atomics.supports_32);
  const amdf_atomic_operations_t remote_64 =
      topology->integrated && topology->host_atomics.supports_64
          ? integer
          : amdf_gpu_umd_program_atomic_pcie_operations(
                topology->host_atomics.supports_64);
  amdf_memory_atomic_capabilities_t result = {
      .device_scope_32 = device_integer ? integer : remote_32,
      .device_scope_64 = device_integer ? integer : remote_64,
  };
  // Cached GC9-11 ordinary backing deliberately has only device scope.
  // Coherent SYSTEM accesses use the exact CPU-facing route; LOCAL can be
  // evicted into GTT and cannot inherit a stronger permanent residency promise.
  if (profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
      (flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0 &&
      (forwarded || gc->major == 12)) {
    result.system_scope_32 = remote_32;
    result.system_scope_64 = remote_64;
  }
  return result;
}

// Selected backing policies need no SDMA payload cache operation. Explicit
// transfer completion and dependency ordering remain caller obligations.
static amdf_status_t amdf_gpu_kfd_sdma_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  const amdf_queue_family_info_t* family = query->queue_family_info;
  if (family->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA ||
      family->format_version != AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 ||
      (family->roles & AMDF_QUEUE_ROLE_TRANSFER) == 0) {
    return amdf_gpu_umd_memory_describe_site(query, out_description);
  }
  amdf_memory_site_description_t description = {0};
  if ((query->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  description.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
                              AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
  *out_description = description;
  return AMDF_STATUS_OK;
}

// GTT and registered pages follow the engine's no-GCR transfer path.
// This does not imply that compute accesses bypass their own data caches.
static amdf_status_t amdf_gpu_kfd_sdma_system_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  if (query->queue_family_info->command_type !=
      AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
    return amdf_gpu_umd_memory_describe_system_store_site(query,
                                                          out_description);
  }
  // A remote LOCAL projection retains consumer policy but cannot inherit
  // the owned SYSTEM backing's SDMA path.
  if ((query->flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) != 0) {
    return amdf_gpu_umd_memory_describe_site(query, out_description);
  }
  return amdf_gpu_kfd_sdma_describe_site(query, out_description);
}

// Same-GPU VRAM follows the engine's transfer policy. Host aperture writes
// require the separately queried host release, including HDP maintenance;
// compute queues still require their own payload fences.
static amdf_status_t amdf_gpu_kfd_sdma_local_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  if ((query->flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) == 0) {
    return amdf_gpu_umd_memory_describe_site(query, out_description);
  }
  return amdf_gpu_kfd_sdma_describe_site(query, out_description);
}

// SDMA 4.4 uses the V4 no-GCR visibility path across CDNA. This is independent
// of compiler ISA and compute PTE MTYPE.
static bool amdf_gpu_kfd_supports_legacy_sdma_visibility(
    const amdf_gpu_kfd_topology_t* topology) {
  const amdf_gpu_kfd_ip_version_t* ip = &topology->sdma.ip;
  return ip->exact && ip->major == 4 && ip->minor == 4 &&
         (ip->revision == 0 || ip->revision == 2 || ip->revision == 4 ||
          ip->revision == 5);
}

static bool amdf_gpu_kfd_supports_gfx1151_system_visibility(
    const amdf_gpu_kfd_topology_t* topology) {
  return topology->properties.gfx_ip.major == 11 &&
         topology->properties.gfx_ip.minor == 5 &&
         topology->properties.gfx_ip.stepping == 1 && topology->sdma.ip.exact &&
         topology->sdma.ip.major == 6 && topology->sdma.ip.minor == 1 &&
         topology->sdma.ip.revision == 1;
}

// KFD maps a fixed consumer set into the backing owner's native allocation.
// Local placement is directional: the consumer must reach the selected source,
// regardless of whether it offers a local allocation scope of its own.
static bool amdf_gpu_kfd_query_group_access(
    const amdf_memory_native_profile_t* backing,
    const amdf_memory_native_profile_t* candidate,
    amdf_memory_native_profile_t* out_profile) {
  const amdf_memory_profile_roles_t role =
      (backing->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0
          ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
          : AMDF_MEMORY_PROFILE_ROLE_CREATE;
  if (candidate->memory_class != AMDF_MEMORY_CLASS_SYSTEM ||
      (candidate->roles & role) == 0) {
    return false;
  }
  if (backing->memory_class == AMDF_MEMORY_CLASS_SYSTEM) {
    // The shared KFD allocation fixes the GPU cache policy for every consumer.
    // Cached and control backing both snoop the host but cannot share GPU
    // cache policies. Registered profiles all use the same native policy.
    if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE &&
        backing->ordinal != candidate->ordinal) {
      return false;
    }
    *out_profile = *candidate;
    return true;
  }
  const amdf_gpu_kfd_topology_t* source = backing->construction.data;
  const amdf_gpu_kfd_topology_t* consumer = candidate->construction.data;
  bool reachable =
      source->gpu_id == consumer->gpu_id ||
      (source->memory_peers.hive_id != 0 &&
       source->memory_peers.hive_id == consumer->memory_peers.hive_id &&
       source->memory_peers.hive_sharing_enabled &&
       consumer->memory_peers.hive_sharing_enabled);
  for (uint32_t i = 0; !reachable && i < consumer->memory_peers.count; ++i) {
    reachable = consumer->memory_peers.gpu_ids[i] == source->gpu_id;
  }
  if (!reachable) {
    return false;
  }
  amdf_memory_native_profile_t profile = *backing;
  profile.ordinal = candidate->ordinal;
  profile.device_address = candidate->device_address;
  profile.allocation = candidate->allocation;
  profile.construction = candidate->construction;
  profile.visibility = candidate->visibility;
  // A remote LOCAL attachment has a different native route from either the
  // owner's local access or this consumer's SYSTEM allocation profile.
  profile.query_program_atomics = NULL;
  if (source->gpu_id == consumer->gpu_id) {
    // A second handle for this GPU keeps the backing's LOCAL site policy;
    // its construction, address and host-policy facts remain consumer-owned.
    profile.visibility.describe_site = backing->visibility.describe_site;
    profile.query_program_atomics = backing->query_program_atomics;
  }
  // The backing determines VRAM cache semantics. In particular, a consumer's
  // GTT profile must not add HOST_COHERENT to an access of local memory.
  *out_profile = profile;
  return true;
}

static uint64_t amdf_gpu_kfd_maximum_byte_length(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size) {
  const uint64_t virtual_address_span =
      topology->virtual_address.end - topology->virtual_address.begin;
  uint64_t maximum_byte_length = virtual_address_span / 2;
  if (maximum_byte_length > SIZE_MAX / 2) {
    maximum_byte_length = SIZE_MAX / 2;
  }
  return maximum_byte_length & ~(uint64_t)(page_size - 1);
}

static uint64_t amdf_gpu_kfd_maximum_alignment(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size) {
  uint64_t limit = amdf_gpu_kfd_maximum_byte_length(topology, page_size);
  uint64_t alignment = 1;
  while (alignment <= limit / 2) {
    alignment <<= 1;
  }
  return alignment;
}

static uint32_t amdf_gpu_kfd_address_bit_count(uint64_t maximum_address) {
  uint32_t bit_count = 0;
  do {
    ++bit_count;
    maximum_address >>= 1;
  } while (maximum_address != 0);
  return bit_count;
}

amdf_status_t amdf_gpu_kfd_query_memory_profile(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    amdf_native_lifetime_t native_lifetime, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  amdf_memory_native_profile_t profile = {
      .ordinal = memory_profile_ordinal,
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
      .query_program_atomics = amdf_gpu_kfd_query_program_atomics,
      .visibility = {.describe_site = amdf_gpu_umd_memory_describe_site,
                     .describe_program = amdf_gpu_kfd_describe_program_site},
      .construction =
          {
              .query_access = amdf_gpu_kfd_query_group_access,
              .data = topology,
          },
  };
  const uint64_t maximum_byte_length =
      amdf_gpu_kfd_maximum_byte_length(topology, page_size);
  const uint64_t maximum_alignment =
      amdf_gpu_kfd_maximum_alignment(topology, page_size);
  const amdf_memory_construction_capabilities_t allocation = {
      .maximum_byte_length = maximum_byte_length,
      .byte_length_granularity = 1,
      .minimum_alignment = page_size,
      .maximum_alignment = maximum_alignment,
      .native_byte_length_granularity = page_size,
  };
  const amdf_memory_construction_capabilities_t registration = {
      .maximum_byte_length = maximum_byte_length,
      .byte_length_granularity = 1,
      .registered_host_pointer_alignment = 1,
      .registered_host_properties = (AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                                     AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK),
      .minimum_alignment = 1,
      .maximum_alignment = maximum_alignment,
      .native_byte_length_granularity = page_size,
  };
  const amdf_memory_address_capabilities_t page_address = {
      .address_domain_ordinal = 0,
      .address_bit_count =
          amdf_gpu_kfd_address_bit_count(topology->virtual_address.end - 1),
      .minimum_address = topology->virtual_address.begin,
      .maximum_address = topology->virtual_address.end - 1,
      .minimum_alignment = page_size,
  };
  const amdf_host_mapping_capabilities_t host_mapping = {
      .properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                    AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK,
      .maximum_byte_length = maximum_byte_length,
      .byte_offset_granularity = 1,
      .byte_length_granularity = 1,
      .supported_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  // Owned GTT uses CPU-cached TTM pages with snooped GPU PTEs. Host coherence
  // is independent of GPU MTYPE: only the uncached profile requests GPU UC.
  const bool uncached_system =
      memory_profile_ordinal == AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_UNCACHED;
  uint32_t ordinal = AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_COUNT;
  if (memory_profile_ordinal < ordinal) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                    AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                    AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE |
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
        AMDF_MEMORY_FLAG_HOST_ADDRESS_IDENTITY;
    if (uncached_system) {
      profile.guaranteed_flags |= AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    }
    profile.supported_flags = profile.guaranteed_flags |
                              AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                              AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
    // Unified backing is local to both CPU and GPU regardless of GPU cache
    // policy. Keep its SYSTEM acquisition and write-back host mapping.
    if (topology->integrated) {
      profile.supported_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    }
    if (uncached_system) {
      profile.supported_flags |= AMDF_MEMORY_FLAG_QUEUE_STORAGE;
    }
    profile.guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
    profile.supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                      AMDF_MEMORY_ACCESS_WRITE |
                                      AMDF_MEMORY_ACCESS_EXECUTE;
    profile.device_address = page_address;
    profile.allocation = allocation;
    profile.host_mapping = host_mapping;
    // Buffered I/O copies through the CPU mapping. GUP cannot pin these
    // VM_IO/PFNMAP views, so direct I/O and registered buffers remain excluded.
    profile.host_mapping.features |= AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS |
                                     AMDF_HOST_MAPPING_FEATURE_BUFFERED_FILE_IO;
    profile.external_memory_support_count = 2;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS,
        .source_offset_alignment = 1,
        .byte_length_alignment = 1,
    };
    profile.external_memory_support[1] = profile.external_memory_support[0];
    profile.external_memory_support[1].type =
        AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    profile.external_memory_support[1].provenance =
        (amdf_external_memory_provenance_t)
            AMDF_LINUX_DMA_BUF_DIRECT_HOST_PROVENANCE;
    profile.external_memory_support[1].flags &=
        ~AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API;
    // Control allocations request UC on GMC9/11/12. Aligned control-word
    // loads/stores do not use PCIe AtomicOps transactions. Only read-modify-
    // write operations require the corresponding native CPU route width.
    // GC12.0 additionally routes CP atomics through MALL. PCIe supports
    // integer ADD, unlike raw SUB/AND/OR/XOR. Wait comparisons remain an
    // independent queue capability.
    const amdf_gpu_kfd_ip_version_t* gc = &topology->gc_ip;
    if (uncached_system && gc->exact &&
        (gc->major == 9 || gc->major == 11 ||
         (gc->major == 12 && gc->minor == 0 && gc->revision <= 1))) {
      profile.atomic_operations_32 =
          AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE;
      profile.atomic_operations_64 = profile.atomic_operations_32;
      if (topology->host_atomics.supports_32) {
        profile.atomic_operations_32 |= AMDF_ATOMIC_OPERATION_ADD;
      }
      if (topology->host_atomics.supports_64) {
        profile.atomic_operations_64 |= AMDF_ATOMIC_OPERATION_ADD;
      }
    }
    profile.visibility.describe_site =
        amdf_gpu_umd_memory_describe_system_store_site;
  } else if ((topology->memory_features &
              AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) != 0 &&
             memory_profile_ordinal == ordinal++) {
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    profile.supported_flags =
        profile.guaranteed_flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE |
        AMDF_MEMORY_FLAG_MAPPING_SOURCE | AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    if ((topology->memory_features &
         AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY) != 0) {
      profile.roles |= AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                 AMDF_MEMORY_FLAG_HOST_ADDRESS_IDENTITY;
    }
    profile.guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
    profile.supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                      AMDF_MEMORY_ACCESS_WRITE |
                                      AMDF_MEMORY_ACCESS_EXECUTE;
    profile.device_address = page_address;
    profile.allocation = allocation;
    if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0) {
      profile.host_mapping = host_mapping;
      // TTM maps ordinary VRAM write-combined, but KFD can restore an evicted
      // allocation in cached GTT. Only CPU-connected XGMI stays cached in both
      // domains. Cache operations cover every admitted state independently of
      // these lifetime guarantees.
      if (!topology->vram.host_cached) {
        profile.host_mapping.properties = AMDF_HOST_MEMORY_PROPERTY_NONE;
      }
    }
    if (amdf_gpu_kfd_supports_legacy_sdma_visibility(topology)) {
      profile.visibility.describe_site = amdf_gpu_kfd_sdma_local_describe_site;
    }
  } else if (native_lifetime == AMDF_NATIVE_LIFETIME_PROCESS &&
             memory_profile_ordinal == ordinal++) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                               AMDF_MEMORY_FLAG_HOST_COHERENT |
                               AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    profile.supported_flags = profile.guaranteed_flags |
                              AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                              AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    profile.guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
    profile.supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                      AMDF_MEMORY_ACCESS_WRITE |
                                      AMDF_MEMORY_ACCESS_EXECUTE;
    profile.device_address = page_address;
    profile.device_address.minimum_alignment = 1;
    profile.registration = registration;
    profile.host_mapping = host_mapping;
    profile.host_mapping.features |= AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
  } else {
    // Foreign host DMA-BUF imports create separate KFD residency and render
    // alias objects. Both charge the full payload against the GTT limit, so
    // this acquisition cannot satisfy the single-charge backing contract.
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
      (amdf_gpu_kfd_supports_legacy_sdma_visibility(topology) ||
       amdf_gpu_kfd_supports_gfx1151_system_visibility(topology))) {
    // Physical page ownership and compute PTE cache policy do not change the
    // selected SDMA engine's cache protocol. CPU transitions remain separate.
    profile.visibility.describe_site = amdf_gpu_kfd_sdma_system_describe_site;
  }
  *out_profile = profile;
  return AMDF_STATUS_OK;
}
