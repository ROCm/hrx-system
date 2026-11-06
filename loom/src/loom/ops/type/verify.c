// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/type/ops.h"

iree_status_t loom_type_decl_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  const loom_string_id_t key[] = {
      loom_type_decl_namespace(op),
      loom_type_decl_type_name(op),
  };
  const iree_string_view_t field_names[] = {
      IREE_SVL("namespace"),
      IREE_SVL("type_name"),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(key) && iree_status_is_ok(status); ++i) {
    const iree_string_view_t value =
        loom_string_table_get(&module->strings, key[i]);
    if (value.size && !memchr(value.data, 0, value.size) &&
        iree_unicode_utf8_validate(value)) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_names[i]),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_TYPE_021,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    status = iree_diagnostic_emit(emitter, &emission);
  }
  return status;
}
