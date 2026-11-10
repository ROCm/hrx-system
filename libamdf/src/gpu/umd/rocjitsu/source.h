// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_SOURCE_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_SOURCE_H_

#include <pthread.h>

#include "amdf/rocjitsu.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/umd/aql_descriptor.h"
#include "libamdf/src/gpu/umd/rocjitsu/library.h"
#include "libamdf/src/instance.h"

typedef struct amdf_rocjitsu_source_t amdf_rocjitsu_source_t;

typedef struct amdf_rocjitsu_gpu_t {
  // Owning source retained by the provider instance.
  amdf_rocjitsu_source_t* source;
  // Immutable passive simulator metadata.
  rj_vm_gpu_info_t native;
  // Core facts with an explicit configured identity and no host device node.
  amdf_endpoint_info_t info;
  // Qualified implementation services and physical compiler target.
  amdf_gpu_endpoint_profile_t profile;
  // Descriptor geometry; process apertures are filled before VM execution.
  amdf_gpu_aql_descriptor_profile_t aql;
} amdf_rocjitsu_gpu_t;

struct amdf_rocjitsu_source_t {
  // Allocator copied from the owning instance.
  amdf_allocator_t host_allocator;
  // Borrowed instance owner of the cold native-state lock.
  amdf_platform_instance_t* platform;
  // Qualified C entry points and owned loader reference.
  amdf_rocjitsu_library_t library;
  // Owned passive configuration snapshot.
  rj_vm_config_t* configuration;
  // Number of immutable GPU records following this source.
  uint32_t gpu_count;
  // Fresh generation discriminator shared by this source's endpoint IDs.
  uint64_t generation;
  // Lazily activated VM, retained through instance teardown.
  rj_vm_t* machine;
  // Explicit process domain, or zero before opening it.
  uint32_t process;
  // Host-owned thread executing the VM.
  pthread_t thread;
  // Whether thread requires joining before VM destruction.
  bool thread_started;
  // Nonzero terminal failure published when the engine returns.
  amdf_atomic_uint64_t terminal_status;
  // Shared native notification backing, serialized by the instance lock.
  struct {
    // Owned event table, including incomplete backing preparation.
    struct amdf_rocjitsu_buffer_t* storage;
    // Whether storage has a complete host view that EventState may borrow.
    bool ready;
  } events;
  // Immutable GPU facts, allocated with the source metadata.
  amdf_rocjitsu_gpu_t gpus[];
};

// Immutable bootstrap API negotiation; performs no initialization.
amdf_status_t amdf_rocjitsu_extension_query(uint32_t minimum_version,
                                            uint32_t maximum_version,
                                            const void** out_api);

// Releases owned sources after all instance children are gone. Completed
// releases are removed even if a later source retains cleanup progress.
amdf_status_t amdf_rocjitsu_sources_destroy(amdf_instance_t* instance);

// Enumerates the configured sources as the instance's complete inventory.
uint32_t amdf_rocjitsu_endpoint_enumerate(amdf_instance_t* instance,
                                          uint32_t capacity,
                                          amdf_endpoint_summary_t* summaries);

// Resolves an explicit source ID and creates an ordinary core endpoint.
amdf_status_t amdf_rocjitsu_endpoint_open(amdf_instance_t* instance,
                                          const amdf_endpoint_id_t* id,
                                          amdf_endpoint_t** out_endpoint);

// Prepares shared execution state under the instance's cold native-state lock.
amdf_status_t amdf_rocjitsu_source_prepare(amdf_rocjitsu_source_t* source);

// Dispatches a wire command; C transport failures and driver errno stay
// distinct.
amdf_status_t amdf_rocjitsu_source_command(amdf_rocjitsu_source_t* source,
                                           uint32_t opcode, void* arguments,
                                           size_t byte_length);

// Maps an explicit process view and preserves the driver-reported mmap errno.
amdf_status_t amdf_rocjitsu_source_map(amdf_rocjitsu_source_t* source,
                                       rj_vm_map_t* mapping);

// Cached engine liveness, with no native query or wait.
static inline amdf_status_t amdf_rocjitsu_source_status(
    const amdf_rocjitsu_source_t* source) {
  return amdf_atomic_uint64_load_acquire(&source->terminal_status);
}

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_SOURCE_H_
