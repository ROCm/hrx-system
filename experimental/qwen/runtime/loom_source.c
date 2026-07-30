// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_source.h"

#include <string.h>

#include "experimental/qwen/kernels/runtime_modules.h"

typedef struct qwen_loom_source_mapping_t {
  // Stable runtime module path.
  iree_string_view_t module_path;
  // Generated source filename in the embedded table of contents.
  iree_string_view_t embedded_name;
} qwen_loom_source_mapping_t;

static const qwen_loom_source_mapping_t qwen_loom_source_mappings[] = {
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ATTENTION_METADATA),
        .embedded_name = IREE_SVL("qwen_attention_metadata.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED),
        .embedded_name = IREE_SVL("qwen3_moe_attention_prepare_quantized.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED),
        .embedded_name = IREE_SVL("qwen3_moe_attention_qkv_quantized.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16),
        .embedded_name =
            IREE_SVL("qwen3_moe_attention_postprocess_f32_f16.loom"),
    },
    {
        .module_path =
            IREE_SVL(QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16),
        .embedded_name =
            IREE_SVL("qwen3_moe_flash_attention_prefill_f32_f16.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        .embedded_name = IREE_SVL("qwen3_moe_dense_linear_quantized_f16.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32),
        .embedded_name = IREE_SVL("qwen3_moe_router_projection_f32.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ROUTER_TOP8_F32),
        .embedded_name = IREE_SVL("qwen3_moe_router_top8_f32.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        .embedded_name = IREE_SVL("qwen3_moe_routed_gate_up_f16.loom"),
    },
    {
        .module_path = IREE_SVL(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        .embedded_name = IREE_SVL("qwen3_moe_routed_down_f16.loom"),
    },
};

iree_status_t qwen_loom_source_lookup(
    iree_string_view_t module_path,
    qwen_loom_source_module_t* out_source_module) {
  IREE_ASSERT_ARGUMENT(out_source_module);
  memset(out_source_module, 0, sizeof(*out_source_module));
  if (iree_string_view_is_empty(module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Loom module path is required");
  }

  const qwen_loom_source_mapping_t* mapping = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(qwen_loom_source_mappings);
       ++i) {
    if (iree_string_view_equal(module_path,
                               qwen_loom_source_mappings[i].module_path)) {
      mapping = &qwen_loom_source_mappings[i];
      break;
    }
  }
  if (!mapping) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "embedded Loom module `%.*s` was not found",
                            (int)module_path.size, module_path.data);
  }

  const iree_file_toc_t* embedded_files = qwen_loom_runtime_modules_create();
  for (iree_host_size_t i = 0; i < qwen_loom_runtime_modules_size(); ++i) {
    const iree_file_toc_t* embedded_file = &embedded_files[i];
    const iree_string_view_t embedded_name =
        iree_make_cstring_view(embedded_file->name);
    if (!iree_string_view_equal(mapping->embedded_name, embedded_name)) {
      continue;
    }
    *out_source_module = (qwen_loom_source_module_t){
        .module_path = mapping->module_path,
        .source_identifier = embedded_name,
        .source_contents =
            iree_make_const_byte_span(embedded_file->data, embedded_file->size),
    };
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "embedded Loom table is missing mapped source `%.*s`",
                          (int)mapping->embedded_name.size,
                          mapping->embedded_name.data);
}
