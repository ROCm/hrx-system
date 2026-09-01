// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

static_assert(LOOM_SCALAR_TYPE_INDEX == 1, "LOOM_SCALAR_TYPE_INDEX ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_OFFSET == 2, "LOOM_SCALAR_TYPE_OFFSET ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_I1 == 3, "LOOM_SCALAR_TYPE_I1 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_I8 == 4, "LOOM_SCALAR_TYPE_I8 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_I16 == 5, "LOOM_SCALAR_TYPE_I16 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_I32 == 6, "LOOM_SCALAR_TYPE_I32 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_I64 == 7, "LOOM_SCALAR_TYPE_I64 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_F8E4M3 == 8, "LOOM_SCALAR_TYPE_F8E4M3 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_F8E5M2 == 9, "LOOM_SCALAR_TYPE_F8E5M2 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_F16 == 10, "LOOM_SCALAR_TYPE_F16 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_BF16 == 11, "LOOM_SCALAR_TYPE_BF16 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_F32 == 12, "LOOM_SCALAR_TYPE_F32 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_F64 == 13, "LOOM_SCALAR_TYPE_F64 ordinal does not match Python");
static_assert(LOOM_SCALAR_TYPE_COUNT_ == 14, "scalar type count does not match Python");
static_assert(LOOM_SCALAR_TYPE_NONE == 0, "scalar type none sentinel does not match Python");

static const char* const loom_scalar_type_names[LOOM_SCALAR_TYPE_COUNT_] = {
    [LOOM_SCALAR_TYPE_INDEX] = "index",
    [LOOM_SCALAR_TYPE_OFFSET] = "offset",
    [LOOM_SCALAR_TYPE_I1] = "i1",
    [LOOM_SCALAR_TYPE_I8] = "i8",
    [LOOM_SCALAR_TYPE_I16] = "i16",
    [LOOM_SCALAR_TYPE_I32] = "i32",
    [LOOM_SCALAR_TYPE_I64] = "i64",
    [LOOM_SCALAR_TYPE_F8E4M3] = "f8E4M3",
    [LOOM_SCALAR_TYPE_F8E5M2] = "f8E5M2",
    [LOOM_SCALAR_TYPE_F16] = "f16",
    [LOOM_SCALAR_TYPE_BF16] = "bf16",
    [LOOM_SCALAR_TYPE_F32] = "f32",
    [LOOM_SCALAR_TYPE_F64] = "f64",
};

static loom_scalar_type_t loom_scalar_type_classify_name(iree_string_view_t name) {
  switch (name.size) {
    case 2:
      if (iree_string_view_equal(name, IREE_SV("i1"))) {
        return LOOM_SCALAR_TYPE_I1;
      }
      if (iree_string_view_equal(name, IREE_SV("i8"))) {
        return LOOM_SCALAR_TYPE_I8;
      }
      break;
    case 3:
      if (iree_string_view_equal(name, IREE_SV("i16"))) {
        return LOOM_SCALAR_TYPE_I16;
      }
      if (iree_string_view_equal(name, IREE_SV("i32"))) {
        return LOOM_SCALAR_TYPE_I32;
      }
      if (iree_string_view_equal(name, IREE_SV("i64"))) {
        return LOOM_SCALAR_TYPE_I64;
      }
      if (iree_string_view_equal(name, IREE_SV("f16"))) {
        return LOOM_SCALAR_TYPE_F16;
      }
      if (iree_string_view_equal(name, IREE_SV("f32"))) {
        return LOOM_SCALAR_TYPE_F32;
      }
      if (iree_string_view_equal(name, IREE_SV("f64"))) {
        return LOOM_SCALAR_TYPE_F64;
      }
      break;
    case 4:
      if (iree_string_view_equal(name, IREE_SV("bf16"))) {
        return LOOM_SCALAR_TYPE_BF16;
      }
      break;
    case 5:
      if (iree_string_view_equal(name, IREE_SV("index"))) {
        return LOOM_SCALAR_TYPE_INDEX;
      }
      break;
    case 6:
      if (iree_string_view_equal(name, IREE_SV("offset"))) {
        return LOOM_SCALAR_TYPE_OFFSET;
      }
      if (iree_string_view_equal(name, IREE_SV("f8E4M3"))) {
        return LOOM_SCALAR_TYPE_F8E4M3;
      }
      if (iree_string_view_equal(name, IREE_SV("f8E5M2"))) {
        return LOOM_SCALAR_TYPE_F8E5M2;
      }
      break;
  }
  return LOOM_SCALAR_TYPE_NONE;
}
