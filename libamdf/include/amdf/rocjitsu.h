// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_ROCJITSU_H_
#define AMDF_ROCJITSU_H_

#include "amdf/base.h"

/// Explicit machine configuration with instance-owned simulator lifetime.
#define AMDF_ROCJITSU_EXTENSION_VERSION_1 1u
#define AMDF_ROCJITSU_EXTENSION_VERSION_LATEST AMDF_ROCJITSU_EXTENSION_VERSION_1

/// Describes one independently configured simulator machine.
#define AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO \
  ((amdf_structure_type_t)0x00040001u)

/// Inputs borrowed only for instance_add_machine.
typedef struct amdf_rocjitsu_machine_create_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_rocjitsu_machine_create_info_t).
  uint32_t structure_size;
  /// Optional input chain; no extensions are defined in version 1.
  const void* next;
  /// Optional NUL-terminated path or loader name of a qualified, trusted
  /// library. A nonempty value overrides LIBAMDF_ROCJITSU_PATH. If NULL or
  /// empty, a nonempty LIBAMDF_ROCJITSU_PATH selects the library; otherwise the
  /// system loader searches for librocjitsu.so, including its LD_LIBRARY_PATH
  /// policy. Failure to load or negotiate the selected library is terminal; no
  /// other library is substituted. Dependencies use the system loader's search
  /// rules. The library must exclude libc interposition and host fault
  /// recovery.
  const char* library_path;
  /// NUL-terminated rocjitsu JSON configuration copied during this call.
  const char* configuration_json;
} amdf_rocjitsu_machine_create_info_t;

/// Explicit host bootstrap for simulated devices in an ordinary instance.
///
/// Negotiation performs no initialization. Each configured machine owns an
/// independent VM and may contain multiple GPUs. The configured machines define
/// the instance's complete endpoint inventory. Native endpoints are neither
/// enumerated nor openable in that instance. An instance without configured
/// machines retains native discovery. Device consumers use the ordinary core
/// and GPU API tables.
/// The build's pinned rocjitsu revision defines its accepted C ABI; another
/// revision returns VERSION_MISMATCH. Valid memory backing remains live through
/// queue retirement; invalid simulator accesses are process-fatal.
/// Activation requires the host to provide a soft RLIMIT_NOFILE of at least
/// 8192; an insufficient budget returns RESOURCE_EXHAUSTED. The backend leaves
/// process resource limits and signal handlers under host ownership.
typedef struct amdf_rocjitsu_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// Implemented rocjitsu extension version.
  uint32_t extension_version;
  /// Adds one machine transactionally, preserving existing configuration on
  /// failure. Requires exclusive instance access and no open endpoints.
  /// Native GPU activation seals the instance's implementation choice; adding
  /// a simulator after that point returns FAILED_PRECONDITION even after all
  /// native endpoints have closed.
  /// The first successful addition replaces native discovery with the
  /// configured topology; subsequent additions extend that topology. Previously
  /// enumerated native identities can no longer be opened in this instance.
  /// Resolves the library and parses passive metadata without starting
  /// simulator execution. First device creation activates the machine. The
  /// instance retains configuration and execution state through destruction;
  /// all queues, memory, devices, and endpoints must be released first.
  /// This cold operation may allocate, load a library, and perform host IO.
  amdf_status_t(AMDF_CALL* instance_add_machine)(
      amdf_instance_t* instance,
      const amdf_rocjitsu_machine_create_info_t* create_info);
} amdf_rocjitsu_api_t;

#endif  // AMDF_ROCJITSU_H_
