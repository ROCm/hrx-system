// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile.h"

#include <string.h>

#include "loom/target/facts_builder.h"

static bool loom_target_profile_bundle_is_complete(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL;
}

static iree_status_t loom_target_profile_validate(
    const loom_target_profile_t* profile,
    const loom_target_profile_type_t** out_profile_type) {
  *out_profile_type = NULL;
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
  *out_profile_type = profile_type;
  return iree_ok_status();
}

iree_status_t loom_target_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t** out_facts) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;
  const loom_target_profile_type_t* profile_type = NULL;
  IREE_RETURN_IF_ERROR(loom_target_profile_validate(profile, &profile_type));

  loom_target_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, profile_type->fact_type->storage_size, (void**)&facts));
  memset(facts, 0, profile_type->fact_type->storage_size);
  loom_target_facts_builder_initialize(profile_type->fact_type,
                                       profile->target_bundle, facts);
  IREE_RETURN_IF_ERROR(
      loom_target_profile_project_facts_into(profile, arena, facts));
  *out_facts = facts;
  return iree_ok_status();
}

iree_status_t loom_target_profile_project_facts_into(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);
  const loom_target_profile_type_t* profile_type = NULL;
  IREE_RETURN_IF_ERROR(loom_target_profile_validate(profile, &profile_type));
  if (out_facts->fact_type != profile_type->fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile family '%.*s' cannot project into fact family '%.*s'",
        (int)profile_type->name.size, profile_type->name.data,
        out_facts->fact_type ? (int)out_facts->fact_type->name.size : 0,
        out_facts->fact_type ? out_facts->fact_type->name.data : "");
  }
  return profile_type->project_facts(profile, arena, out_facts);
}
