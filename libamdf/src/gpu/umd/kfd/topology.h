// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_TOPOLOGY_H_
#define AMDF_SRC_GPU_UMD_KFD_TOPOLOGY_H_

#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/platform/endpoint.h"

// Native facts belonging to one identity-checked DRM endpoint.
typedef struct amdf_gpu_kfd_topology_t {
  // GPU target and compute geometry reported by KFD topology.
  amdf_gpu_endpoint_properties_t properties;
  // KFD node identity used by memory and queue ioctls.
  uint32_t gpu_id;
  // Physical local-memory reachability reported by the installed driver.
  struct {
    // Cached native DRM hive identity; zero when this GPU has no hive.
    uint64_t hive_id;
    // Native xGMI peer mapping policy for this hive.
    bool hive_sharing_enabled;
    // Number of GPU backing owners reachable through directed PCIe links.
    uint32_t count;
    // Owned GPU identities in consumer-to-backing direction, excluding CPUs.
    uint32_t* gpu_ids;
  } memory_peers;
  // Physical placement features supported by the native device.
  amdf_gpu_device_features_t memory_features;
  // Cached physical heap totals, independent of current allocation usage.
  struct {
    // Physical VRAM capacity in bytes, not KFD's possibly substituted GTT size.
    uint64_t total_byte_length;
    // CPU-visible VRAM capacity in bytes.
    uint64_t visible_byte_length;
  } vram;
  // Number of native compute queues exposed by this KFD node.
  uint32_t compute_queue_count;
  // Native SDMA engines and exact packet-ABI identity.
  struct {
    // Number of ordinary SDMA engines exposed by KFD.
    uint32_t engine_count;
    // Number of XGMI-specialized SDMA engines exposed by KFD.
    uint32_t xgmi_engine_count;
    // Number of constructible queues per SDMA engine.
    uint32_t queue_count_per_engine;
    // Cached IP discovery version used to select exact packet encodings.
    struct {
      // SDMA IP major version.
      uint32_t major;
      // SDMA IP minor version.
      uint32_t minor;
      // SDMA IP revision.
      uint32_t revision;
      // Whether sysfs supplied the full discovery version.
      bool exact;
    } ip;
  } sdma;
  // Native per-XCC context-save/restore bytes, or zero when not reported.
  // Compute queue plans require this group; memory and SDMA do not.
  uint32_t context_save_restore_byte_length;
  // Native per-XCC control-stack bytes, or zero when not reported.
  uint32_t control_stack_byte_length;
  // Ordinary GPU virtual-address interval reported by DRM.
  struct {
    // First usable GPU virtual byte address.
    uint64_t begin;
    // Exclusive end of the usable GPU virtual interval.
    uint64_t end;
    // Required GPU mapping alignment in bytes.
    uint32_t alignment;
  } virtual_address;
} amdf_gpu_kfd_topology_t;

#ifdef __cplusplus
extern "C" {
#endif

// Reads a coherent cached topology snapshot, including physical heap totals
// and owned peer metadata. Success transfers discovery metadata to the caller;
// failure leaves the output unchanged. No execution resources are acquired.
// Memory features and the virtual-address interval are supplied separately by
// target expectations or native refinement on the consuming connection.
// A missing KFD node returns UNSUPPORTED; malformed or changing state is an
// error.
amdf_status_t amdf_gpu_kfd_topology_initialize(
    const amdf_platform_endpoint_t* endpoint, amdf_allocator_t host_allocator,
    amdf_gpu_kfd_topology_t* out_topology);

// Releases only peer metadata owned by a successful initialization.
void amdf_gpu_kfd_topology_deinitialize(amdf_gpu_kfd_topology_t* topology,
                                        amdf_allocator_t host_allocator);

// Refines cached topology with the native memory placement and usable address
// interval of this exact render connection. Failure leaves topology unchanged.
amdf_status_t amdf_gpu_kfd_topology_refine_memory(
    int render_descriptor, uint32_t pci_device_id,
    amdf_gpu_kfd_topology_t* topology);

// Physical placement supported for this package and its cached heap totals.
amdf_gpu_device_features_t amdf_gpu_kfd_topology_memory_features(
    const amdf_gpu_kfd_topology_t* topology, bool integrated);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_KFD_TOPOLOGY_H_
