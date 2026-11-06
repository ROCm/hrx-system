// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/ops/source_types.h"

#include "iree/base/string_view.h"

loom_aie2p_source_type_kind_t loom_aie2p_source_type_kind(
    const loom_module_t* module, loom_type_t type) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return LOOM_AIE2P_SOURCE_TYPE_NONE;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return LOOM_AIE2P_SOURCE_TYPE_NONE;
  }
  const iree_string_view_t name =
      loom_string_table_get(&module->strings, name_id);
  switch (name.size) {
    case sizeof("aie2p.load_fifo") - 1:
      return iree_string_view_equal(name, IREE_SV("aie2p.load_fifo"))
                 ? LOOM_AIE2P_SOURCE_TYPE_LOAD_FIFO
                 : LOOM_AIE2P_SOURCE_TYPE_NONE;
    case sizeof("aie2p.store_fifo") - 1:
      return iree_string_view_equal(name, IREE_SV("aie2p.store_fifo"))
                 ? LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO
                 : LOOM_AIE2P_SOURCE_TYPE_NONE;
    case sizeof("aie2p.bfp16ebs8.operand") - 1:
      return iree_string_view_equal(name, IREE_SV("aie2p.bfp16ebs8.operand"))
                 ? LOOM_AIE2P_SOURCE_TYPE_BFP16EBS8_OPERAND
                 : LOOM_AIE2P_SOURCE_TYPE_NONE;
    case sizeof("aie2p.store_fifo.pointer") - 1:
      return iree_string_view_equal(name, IREE_SV("aie2p.store_fifo.pointer"))
                 ? LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO_POINTER
                 : LOOM_AIE2P_SOURCE_TYPE_NONE;
    case sizeof("aie2p.store_fifo.position") - 1:
      return iree_string_view_equal(name, IREE_SV("aie2p.store_fifo.position"))
                 ? LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO_POSITION
                 : LOOM_AIE2P_SOURCE_TYPE_NONE;
    default:
      return LOOM_AIE2P_SOURCE_TYPE_NONE;
  }
}
