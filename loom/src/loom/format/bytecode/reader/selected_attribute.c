// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_attribute.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/selected_tables.h"
#include "loom/format/bytecode/reader/type.h"
#include "loom/ir/parameterized_type.h"

typedef loom_bytecode_selected_table_materializer_t
    loom_bytecode_attribute_policy_materializer_t;

#define LOOM_BYTECODE_ATTRIBUTE_SELECTED 1
#define LOOM_BYTECODE_ATTRIBUTE_TABLES(materializer) ((materializer)->metadata)
#define LOOM_BYTECODE_ATTRIBUTE_DECODE_STATIC_TYPE \
  loom_bytecode_selected_attribute_decode_static_type
#define LOOM_BYTECODE_ATTRIBUTE_DECODE_COMPLETE_TYPE \
  loom_bytecode_selected_attribute_decode_complete_type
#define LOOM_BYTECODE_ATTRIBUTE_DECODE_NAMED \
  loom_bytecode_selected_attribute_decode_named
#define LOOM_BYTECODE_ATTRIBUTE_MATERIALIZE_NAMED \
  loom_bytecode_selected_attribute_materialize_named
#define LOOM_BYTECODE_ATTRIBUTE_MATERIALIZE_SSA \
  loom_bytecode_selected_attribute_materialize_ssa
#define LOOM_BYTECODE_TYPE_PROJECT_COMPLETED \
  loom_bytecode_selected_type_project_completed
#define LOOM_BYTECODE_TYPE_MATERIALIZE_BINDINGS \
  loom_bytecode_selected_type_materialize_bindings
#include "loom/format/bytecode/reader/attribute_materializer_impl.inl"
#include "loom/format/bytecode/reader/scoped_type_impl.inl"
#undef LOOM_BYTECODE_TYPE_MATERIALIZE_BINDINGS
#undef LOOM_BYTECODE_TYPE_PROJECT_COMPLETED
#undef LOOM_BYTECODE_ATTRIBUTE_MATERIALIZE_SSA
#undef LOOM_BYTECODE_ATTRIBUTE_MATERIALIZE_NAMED
#undef LOOM_BYTECODE_ATTRIBUTE_DECODE_NAMED
#undef LOOM_BYTECODE_ATTRIBUTE_DECODE_COMPLETE_TYPE
#undef LOOM_BYTECODE_ATTRIBUTE_DECODE_STATIC_TYPE
#undef LOOM_BYTECODE_ATTRIBUTE_TABLES
#undef LOOM_BYTECODE_ATTRIBUTE_SELECTED
