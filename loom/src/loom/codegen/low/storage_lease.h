// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-defined physical storage leases over scheduled low packets.
//
// Storage leases model target-visible physical register/storage ownership that
// can outlive semantic SSA liveness. A packet may keep reading an issued source
// register, or may keep owning a pending result register, after the semantic
// value interval would otherwise be dead. Allocation consumes these facts to
// decide whether physical reuse is legal, whether a release action is required,
// or whether another location must be chosen.

#ifndef LOOM_CODEGEN_LOW_STORAGE_LEASE_H_
#define LOOM_CODEGEN_LOW_STORAGE_LEASE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent target storage release classes.
#define LOOM_LOW_STORAGE_LEASE_RELEASE_CLASS_NONE UINT16_MAX
// Sentinel for absent target storage release reason identifiers.
#define LOOM_LOW_STORAGE_RELEASE_REASON_NONE UINT16_MAX
// Sentinel for absent target storage release action identifiers.
#define LOOM_LOW_STORAGE_RELEASE_ACTION_NONE 0
// Sentinel for absent allocation storage release action indices.
#define LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE UINT32_MAX
// Sentinel for absent storage-lease packet indices.
#define LOOM_LOW_STORAGE_LEASE_PACKET_NONE IREE_HOST_SIZE_MAX
// Sentinel for absent storage-lease node indices.
#define LOOM_LOW_STORAGE_LEASE_NODE_NONE UINT32_MAX
// Sentinel for absent storage-lease scheduled ordinals.
#define LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE UINT32_MAX

// Storage-lease fact emitted by a target provider for the current scheduled
// node.
typedef struct loom_low_storage_lease_event_t {
  // Target-visible lease kind.
  loom_low_storage_lease_kind_t kind;
  // Scheduled-node attachment kind.
  loom_low_storage_lease_attachment_t attachment;
  // Operand or result index within the scheduled node.
  uint16_t attachment_index;
  // First allocation unit leased within the attached value.
  uint32_t unit_offset;
  // Number of allocation units leased.
  uint32_t unit_count;
  // Target progress model used to release the lease.
  loom_low_storage_lease_release_scope_t release_scope;
  // Target-owned release class identifier.
  uint16_t release_class_id;
  // Borrowed stable release-class name for diagnostics.
  iree_string_view_t release_class_name;
  // Target-owned residual action identifier used when allocation requests a
  // release.
  uint16_t release_action_id;
  // Borrowed stable target residual action name.
  iree_string_view_t release_action_name;
  // Target-owned hazard reason identifier used for release diagnostics.
  uint16_t release_reason_id;
  // Borrowed stable target release reason name.
  iree_string_view_t release_reason_name;
  // Lease flags.
  loom_low_storage_lease_flags_t flags;
} loom_low_storage_lease_event_t;

// Emits one storage-lease event for the scheduled node currently being queried.
typedef iree_status_t (*loom_low_storage_lease_emit_fn_t)(
    void* user_data, const loom_low_storage_lease_event_t* event);

// Queries target storage leases for one scheduled node.
//
// The builder may call this function more than once for the same node while
// sizing and populating the output table. Implementations must be pure for a
// given schedule and node.
typedef iree_status_t (*loom_low_storage_lease_query_fn_t)(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data);

// Target storage-lease provider used to populate a lease table.
typedef struct loom_low_storage_lease_provider_t {
  // Target-owned context passed to |query|.
  void* user_data;
  // Storage-lease query callback.
  loom_low_storage_lease_query_fn_t query;
} loom_low_storage_lease_provider_t;

// Queries descriptor-attached storage-lease rows for |node|. This is a
// provider callback; targets may use it directly or wrap it to apply a target
// identity check before consuming descriptor rows.
iree_status_t loom_low_storage_lease_query_descriptor_rows(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data);

// One target storage lease attached to a scheduled packet.
typedef struct loom_low_storage_lease_record_t {
  // Packet ordinal in final scheduled order.
  iree_host_size_t packet_index;
  // Schedule node represented by |packet_index|.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Target-visible lease kind.
  loom_low_storage_lease_kind_t kind;
  // Scheduled-node attachment kind.
  loom_low_storage_lease_attachment_t attachment;
  // Operand or result index within the scheduled node.
  uint16_t attachment_index;
  // First allocation unit leased within the attached value.
  uint32_t unit_offset;
  // Number of allocation units leased.
  uint32_t unit_count;
  // Target progress model used to release the lease.
  loom_low_storage_lease_release_scope_t release_scope;
  // Target-owned release class identifier.
  uint16_t release_class_id;
  // Borrowed stable release-class name for diagnostics.
  iree_string_view_t release_class_name;
  // Target-owned residual action identifier used when allocation requests a
  // release.
  uint16_t release_action_id;
  // Borrowed stable target residual action name.
  iree_string_view_t release_action_name;
  // Target-owned hazard reason identifier used for release diagnostics.
  uint16_t release_reason_id;
  // Borrowed stable target release reason name.
  iree_string_view_t release_reason_name;
  // Lease flags.
  loom_low_storage_lease_flags_t flags;
} loom_low_storage_lease_record_t;

// Allocator-requested release action over target storage leases.
typedef struct loom_low_storage_release_action_t {
  // Packet ordinal that must observe the release before it executes.
  iree_host_size_t insertion_packet_index;
  // Schedule node at the release insertion point.
  uint32_t insertion_node_index;
  // Region block containing the insertion point.
  uint32_t block_index;
  // Scheduled ordinal of the insertion point within |block_index|.
  uint32_t scheduled_ordinal;
  // Target-owned release class identifier.
  uint16_t release_class_id;
  // Borrowed stable release-class name for diagnostics.
  iree_string_view_t release_class_name;
  // Target-owned residual action identifier.
  uint16_t release_action_id;
  // Borrowed stable target residual action name.
  iree_string_view_t release_action_name;
  // Target-owned hazard reason identifier.
  uint16_t release_reason_id;
  // Borrowed stable target release reason name.
  iree_string_view_t release_reason_name;
  // Target-defined release progress needed before the insertion point.
  uint32_t required_progress;
  // Storage-lease record covered by this release action.
  uint32_t lease_record_index;
} loom_low_storage_release_action_t;

// Ordered storage-lease sidecar for one scheduled low function.
typedef struct loom_low_storage_lease_table_t {
  // Schedule table walked to build this storage-lease table.
  const loom_low_schedule_table_t* schedule;
  // Storage-lease records in scheduled packet order.
  const loom_low_storage_lease_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
} loom_low_storage_lease_table_t;

// Builds target storage-lease records for |schedule| using |provider|.
iree_status_t loom_low_storage_lease_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_storage_lease_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_storage_lease_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_LEASE_H_
