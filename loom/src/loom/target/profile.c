// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/target/facts_builder.h"

static bool loom_target_profile_bundle_is_complete(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL;
}

void loom_target_profile_selection_release(
    loom_target_profile_selection_t* selection, iree_allocator_t allocator) {
  if (selection->profile != NULL && selection->destroy != NULL) {
    selection->destroy(selection->profile, allocator);
  }
  *selection = (loom_target_profile_selection_t){0};
}

iree_status_t loom_target_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t** out_facts) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;
  if (profile == NULL || profile->type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target profile has no family type");
  }
  const loom_target_profile_type_t* profile_type = profile->type;
  if (!loom_target_profile_bundle_is_complete(profile->target_bundle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile family '%.*s' has no complete target bundle",
        (int)profile_type->name.size, profile_type->name.data);
  }
  if (profile_type->fact_type == NULL ||
      profile_type->fact_type->storage_size < sizeof(loom_target_facts_t) ||
      profile_type->project_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target profile family '%.*s' has no typed fact projector",
        (int)profile_type->name.size, profile_type->name.data);
  }

  loom_target_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, profile_type->fact_type->storage_size, (void**)&facts));
  memset(facts, 0, profile_type->fact_type->storage_size);
  loom_target_facts_builder_initialize(profile_type->fact_type,
                                       profile->target_bundle, facts);
  facts->explicit_fields = profile->explicit_fields;
  IREE_RETURN_IF_ERROR(profile_type->project_facts(profile, arena, facts));
  if (facts->fact_type->rebind != NULL) {
    facts->fact_type->rebind(facts);
  }
  *out_facts = facts;
  return iree_ok_status();
}
