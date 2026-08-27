// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

static_assert(LOOM_LOCATION_TAG_SANITIZER_SITE == 1, "LOOM_LOCATION_TAG_SANITIZER_SITE value does not match Python");
static_assert(LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION == 2, "LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION value does not match Python");
static_assert(LOOM_LOCATION_TAG_TILE_LOWERING == 3, "LOOM_LOCATION_TAG_TILE_LOWERING value does not match Python");
static_assert(LOOM_LOCATION_TAG_UKERNEL_SELECTION == 4, "LOOM_LOCATION_TAG_UKERNEL_SELECTION value does not match Python");
static_assert(LOOM_LOCATION_TAG_USER_BASE == 32768, "location tag user base does not match Python");

static iree_string_view_t loom_location_tag_builtin_name(loom_location_tag_t tag) {
  switch (tag) {
    case LOOM_LOCATION_TAG_SANITIZER_SITE:
      return IREE_SV("sanitizer_site");
    case LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION:
      return IREE_SV("template_instantiation");
    case LOOM_LOCATION_TAG_TILE_LOWERING:
      return IREE_SV("tile_lowering");
    case LOOM_LOCATION_TAG_UKERNEL_SELECTION:
      return IREE_SV("ukernel_selection");
    default:
      return iree_string_view_empty();
  }
}

static loom_location_tag_t loom_location_tag_classify_name(iree_string_view_t name) {
  switch (name.size) {
    case 13:
      if (iree_string_view_equal(name, IREE_SV("tile_lowering"))) {
        return LOOM_LOCATION_TAG_TILE_LOWERING;
      }
      break;
    case 14:
      if (iree_string_view_equal(name, IREE_SV("sanitizer_site"))) {
        return LOOM_LOCATION_TAG_SANITIZER_SITE;
      }
      break;
    case 17:
      if (iree_string_view_equal(name, IREE_SV("ukernel_selection"))) {
        return LOOM_LOCATION_TAG_UKERNEL_SELECTION;
      }
      break;
    case 22:
      if (iree_string_view_equal(name, IREE_SV("template_instantiation"))) {
        return LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION;
      }
      break;
  }
  return LOOM_LOCATION_TAG_INVALID;
}
