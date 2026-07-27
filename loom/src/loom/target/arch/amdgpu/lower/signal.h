// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low helpers for AMDGPU runtime signal producers.
//
// These helpers mirror the device-visible signal operations used by the AMDGPU
// runtime without depending on HSA headers. They build descriptor-backed
// target-low packets for the common producer path: update a user signal with
// system-release semantics and notify a host mailbox when the signal carries
// one. The optional mailbox null check is intentionally left to callers so the
// same primitives can be composed into cold reporting blocks, host-call paths,
// or future kernel-print producers without baking in one CFG shape.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SIGNAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SIGNAL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

typedef struct loom_amdgpu_signal_values_t {
  // Device-visible iree_amd_signal_t address.
  loom_value_id_t address;
  // Host interrupt mailbox pointer loaded from iree_amd_signal_t.
  loom_value_id_t event_mailbox_ptr;
  // Host interrupt event id loaded from iree_amd_signal_t.
  loom_value_id_t event_id;
} loom_amdgpu_signal_values_t;

// Emits target-low IR that loads the signal fields needed for host notification
// from host-visible system memory.
//
// |signal_address| must be an SGPRx2 pointer to a device-visible
// iree_amd_signal_t. The loaded values are uniform across the dispatch and are
// emitted as SGPR values.
iree_status_t loom_amdgpu_build_signal_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location,
    loom_amdgpu_signal_values_t* out_values);

// Emits target-low IR that atomically adds one to signal->value with
// system-release semantics.
//
// This helper only updates the signal value. It does not load or poke the host
// mailbox; callers that need host wakeups should compose this with
// loom_amdgpu_build_signal_poke_mailbox after checking whether the mailbox
// pointer is non-zero.
iree_status_t loom_amdgpu_build_signal_add_one_release(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location);

// Emits target-low IR that applies the AMDGPU mailbox message-id mask to an
// event id.
//
// |event_id| must be an SGPR. The result is an SGPR suitable for moving into
// M0 before S_SENDMSG.
iree_status_t loom_amdgpu_build_signal_mailbox_message_id(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_id, loom_location_id_t location,
    loom_value_id_t* out_message_id);

// Emits target-low IR that release-stores |event_id| to |event_mailbox_ptr| and
// sends the host-interrupt message.
//
// |event_mailbox_ptr| must be an SGPRx2 pointer and |event_id| must be an SGPR.
// This helper does not check for a null mailbox pointer; callers must guard the
// call when the runtime ABI permits a null mailbox.
iree_status_t loom_amdgpu_build_signal_poke_mailbox(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_mailbox_ptr, loom_value_id_t event_id,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SIGNAL_H_
