// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Channel membership and admission provenance, independent of storage.

#ifndef LOOM_ANALYSIS_CHANNEL_PLAN_H_
#define LOOM_ANALYSIS_CHANNEL_PLAN_H_

#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// One protocol instance in a retained compiler plan. The construction owner
// allocates a distinct object for every binding occurrence and shares its
// pointer through captures and calls. Equal source values in different
// invocations still have distinct objects; equal backing addresses never
// establish protocol identity. The object outlives all plans borrowing it.
typedef struct loom_channel_identity_t {
  // Source binding or formal argument, for source correspondence only.
  loom_value_id_t value_id;
} loom_channel_identity_t;

// A source argument or capture bound by the enclosing communication plan.
// This correspondence comes from callable/capture substitution, never from
// equal addresses, storage roots, argument names, or payload types.
typedef struct loom_channel_plan_binding_t {
  // Channel or owned access in the analyzed region's value domain.
  loom_value_id_t value_id;
  // Borrowed protocol instance in the enclosing plan. Several arguments may
  // select it, including arguments forwarded through different invocations.
  const loom_channel_identity_t* channel;
} loom_channel_plan_binding_t;

// A visible callable's channel membership in formal argument coordinates.
// The enclosing plan retains its body for physical specialization. This is not
// an ownership or effect summary: returning a read from an argument's channel
// does not imply returning that argument's particular record or obligation.
typedef struct loom_channel_plan_callable_t {
  // Same-module definition whose source body remains available.
  loom_op_t* function;
  // Formal argument supplying each returned channel/read/write's membership;
  // UINT16_MAX for an ordinary result. Indexed by function result ordinal.
  const uint16_t* result_arguments;
} loom_channel_plan_callable_t;

// Semantic action retained by channel planning. Concrete operation identity is
// classified once while collecting the plan; downstream completion, transport,
// and materialization stages consume this stable vocabulary directly.
typedef uint8_t loom_channel_plan_action_kind_t;
enum loom_channel_plan_action_kind_e {
  LOOM_CHANNEL_PLAN_ACTION_KIND_INVALID = 0,
  LOOM_CHANNEL_PLAN_ACTION_KIND_CALL = 1,
  LOOM_CHANNEL_PLAN_ACTION_KIND_BIND = 2,
  LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE = 3,
  LOOM_CHANNEL_PLAN_ACTION_KIND_ACCEPT = 4,
  LOOM_CHANNEL_PLAN_ACTION_KIND_WAIT = 5,
  LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE = 6,
  LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH = 7,
  LOOM_CHANNEL_PLAN_ACTION_KIND_RELEASE = 8,
  LOOM_CHANNEL_PLAN_ACTION_KIND_FANOUT = 9,
  LOOM_CHANNEL_PLAN_ACTION_KIND_COPY = 10,
  LOOM_CHANNEL_PLAN_ACTION_KIND_SELECT = 11,
  LOOM_CHANNEL_PLAN_ACTION_KIND_COUNT_ = 12,
};

// One operation's channel identities, established before carrier rewriting.
typedef struct loom_channel_plan_action_t {
  // Source operation, borrowed until the consuming rewrite erases it.
  loom_op_t* op;
  // Channel of the first operand, or the result for channel.bind; NULL for
  // a call, whose actual arguments have independent entries in the value table.
  const loom_channel_identity_t* channel;
  union {
    // Destination channel for channel.copy; NULL for other channel actions.
    const loom_channel_identity_t* destination_channel;
    // Retained formal summary for a channel-aware call; NULL for channel ops.
    const loom_channel_plan_callable_t* callable;
  };
  // Producer-classified semantic action kind.
  loom_channel_plan_action_kind_t kind;
} loom_channel_plan_action_t;

static_assert(sizeof(loom_channel_plan_action_t) == 4 * sizeof(void*),
              "channel plan actions must remain four pointers wide");

typedef enum loom_channel_plan_rejection_kind_e {
  LOOM_CHANNEL_PLAN_REJECTION_NONE = 0,
  // A call or other operation still needs a channel-aware representation.
  LOOM_CHANNEL_PLAN_REJECTION_OPAQUE_USE = 1,
  // Control chooses between different channels and needs a dynamic carrier.
  LOOM_CHANNEL_PLAN_REJECTION_DYNAMIC_CHANNEL = 2,
  // An incoming access lacks its enclosing plan's channel correspondence.
  LOOM_CHANNEL_PLAN_REJECTION_UNBOUND_ACCESS = 3,
} loom_channel_plan_rejection_kind_t;

typedef struct loom_channel_plan_rejection_t {
  // Requirement that prevented this fixed-channel realization.
  loom_channel_plan_rejection_kind_t kind;
  // Authored operation requiring the unsupported behavior, if present.
  const loom_op_t* op;
  // Channel or access whose binding could not be retained.
  loom_value_id_t value_id;
} loom_channel_plan_rejection_t;

typedef struct loom_channel_plan_t {
  // Borrowed domain retained by the owner through materialization.
  const loom_local_value_domain_t* value_domain;
  // Direct protocol instance by value ordinal; NULL for unrelated values.
  // Equal entries mean membership only: records and consuming obligations
  // remain distinct SSA values with their original lifetime requirements.
  const loom_channel_identity_t* const* channels;
  // Common admission result ordinal forwarded to each owned access ordinal.
  // INVALID means that the origin is external, opaque, or path dependent.
  // A repeated admission creates different records: equality identifies its
  // source site, not a dynamic record across different loop iterations.
  // Ordinals preserve the relation when cloning projects the value domain.
  const loom_value_ordinal_t* admissions;
  // Original domain extent, excluding values created during materialization.
  loom_value_ordinal_t value_count;
  // Arena-owned channel actions in block/source order.
  const loom_channel_plan_action_t* actions;
  // Number of retained actions.
  iree_host_size_t action_count;
  // Function exits collected with the source operations.
  loom_op_t* const* returns;
  // Number of retained exits.
  iree_host_size_t return_count;
} loom_channel_plan_t;

// Builds fixed-channel membership for one verified flat execution region.
// SCF policies and control lowering precede this boundary. The enclosing plan
// supplies the identity of every incoming channel/read/write argument; a local
// channel.bind creates an arena-owned identity for this analyzed occurrence,
// independent of its backing allocation and other analyses of the same source.
//
// A single collection owns the channel actions and forwarding edges. The
// resulting direct table includes loop-carried accesses and fanout results.
// |callables| is an optional symbol-indexed table of previously summarized
// definitions, owned by the enclosing call-graph plan. Calls carrying channels
// or accesses require an entry there; ordinary borrowed-view calls are opaque
// to this analysis. Dynamic choices between channels are reported explicitly,
// rather than merged because their storage or payload shapes happen to match.
//
// Status reports allocation failures. A non-NONE rejection describes authored
// behavior needing another realization; no partial plan is published. The
// plan is invalidated by source mutation, except its consuming rewrite may use
// the retained identities/actions while replacing their physical carriers.
iree_status_t loom_channel_plan_build(
    const loom_local_value_domain_t* value_domain,
    const loom_channel_plan_binding_t* bindings, iree_host_size_t binding_count,
    const loom_channel_plan_callable_t* const* callables,
    iree_arena_allocator_t* arena, loom_channel_plan_t* out_plan,
    loom_channel_plan_rejection_t* out_rejection);

// Summarizes an already analyzed callable without walking its body again.
// Build the formal plan with each incoming channel/read/write bound to its own
// fresh formal identity carrying that argument's source ID. All returned
// communication values must belong to a formal argument's channel; a
// dynamically chosen or locally created escaping channel needs a runtime
// channel carrier and produces DYNAMIC_CHANNEL. Ordinary returned values need
// no membership. The summary outlives the local domain.
iree_status_t loom_channel_plan_summarize(
    const loom_channel_plan_t* plan, loom_op_t* function,
    iree_arena_allocator_t* arena, loom_channel_plan_callable_t* out_callable,
    loom_channel_plan_rejection_t* out_rejection);

// Direct membership query for an original channel/read/write value.
static inline const loom_channel_identity_t* loom_channel_plan_channel(
    const loom_channel_plan_t* plan, loom_value_id_t value_id) {
  return plan
      ->channels[loom_local_value_domain_ordinal(plan->value_domain, value_id)];
}

// Source admission shared by every forwarding path to this original access.
static inline loom_value_id_t loom_channel_plan_admission(
    const loom_channel_plan_t* plan, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      plan->admissions[loom_local_value_domain_ordinal(plan->value_domain,
                                                       value_id)];
  return ordinal == LOOM_VALUE_ORDINAL_INVALID
             ? LOOM_VALUE_ID_INVALID
             : plan->value_domain->value_ids[ordinal];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CHANNEL_PLAN_H_
