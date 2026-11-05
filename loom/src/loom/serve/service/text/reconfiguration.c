// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/reconfiguration.h"

static iree_status_t loom_serve_text_reconfiguration_compile(
    void* self, const loom_runtime_sources_t* sources,
    loom_runtime_variant_completion_fn_t completion, void* user_data) {
  return loom_serve_text_model_compile_variant_async(self, sources, NULL,
                                                     completion, user_data);
}

static iree_status_t loom_serve_text_reconfiguration_select(void* self,
                                                            void* value) {
  return loom_serve_text_model_select_variant(self, value);
}

static void loom_serve_text_reconfiguration_dispose(void* value) {
  loom_serve_text_variant_free(value);
}

loom_service_execution_update_t loom_serve_text_model_execution_update(
    loom_serve_text_model_t* model) {
  return (loom_service_execution_update_t){
      .self = model,
      .compile = loom_serve_text_reconfiguration_compile,
      .select = loom_serve_text_reconfiguration_select,
      .dispose = loom_serve_text_reconfiguration_dispose};
}
