// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low builders for AMDGPU scalar control packets.
//
// These helpers construct descriptor-backed packets such as S_SENDMSG, S_TRAP,
// and S_SETHALT directly in target-low IR. They are intentionally separate from
// branch/control-flow lowering: callers use these packets for runtime
// facilities such as signal notification, feedback, host calls, and traps.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PACKET_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Emits an AMDGPU message-send packet with the given target-defined message
// immediate and M0 payload.
//
// |m0_payload| must be the implicit M0 resource value consumed by the selected
// S_SENDMSG descriptor. The hardware assembly does not print M0 as an explicit
// operand, but target-low IR carries it as a packet operand so scheduling,
// allocation, and dead-code elimination preserve the dependency on the producer
// that wrote M0.
iree_status_t loom_amdgpu_build_control_packet_send_message_with_m0(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_value_id_t m0_payload, loom_location_id_t location);

// Emits an AMDGPU message-send packet with the given target-defined message
// immediate and a zero M0 payload.
iree_status_t loom_amdgpu_build_control_packet_send_message(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location);

// Emits an AMDGPU message-send packet that returns a 32-bit scalar result.
iree_status_t loom_amdgpu_build_control_packet_send_message_rtn_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location, loom_value_id_t* out_value);

// Emits an AMDGPU halt packet with the given target-defined reason immediate.
iree_status_t loom_amdgpu_build_control_packet_halt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t reason, loom_location_id_t location);

// Emits an AMDGPU trap packet with the given target-defined trap id.
iree_status_t loom_amdgpu_build_control_packet_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t trap_id, loom_location_id_t location);

// Emits an LLVM AMDHSA debugtrap packet.
//
// Debug traps are explicitly resumable by debuggers. Sanitizer assertion
// failures should use loom_amdgpu_build_control_packet_fatal_trap instead.
iree_status_t loom_amdgpu_build_control_packet_debug_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

// Emits an LLVM AMDHSA fatal trap notification packet sequence.
//
// This follows LLVM's trap ABI for AMDHSA kernels. Targets where S_TRAP 2 is
// sufficient emit that packet directly. Targets where S_TRAP 2 can be a nop
// under privileged execution emit LLVM's simulated trap sequence that requests
// the doorbell id and sends MSG_INTERRUPT. This helper emits only the target
// packets; callers own the no-return control-flow edge, usually a branch to an
// S_SETHALT loop.
iree_status_t loom_amdgpu_build_control_packet_fatal_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PACKET_H_
