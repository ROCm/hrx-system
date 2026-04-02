// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Memory pool stubs. Full implementation lives in the binding layer
// (hip_mem_pool.c) during the migration period. These stubs export the
// symbols so the DSO link succeeds; the binding overrides them once
// the streaming layer is removed.

#include "pyre_internal.h"

pyre_status_t pyre_mem_pool_create(pyre_device_t device,
                                   const pyre_mem_pool_props_t* props,
                                   pyre_mem_pool_t* out_pool) {
  (void)device;
  (void)props;
  (void)out_pool;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_mem_pool_create: use streaming layer");
}

void pyre_mem_pool_retain(pyre_mem_pool_t pool) { (void)pool; }

void pyre_mem_pool_release(pyre_mem_pool_t pool) { (void)pool; }

pyre_status_t pyre_mem_pool_get_attribute(pyre_mem_pool_t pool,
                                          pyre_mem_pool_attr_t attr,
                                          uint64_t* out_value) {
  (void)pool;
  (void)attr;
  (void)out_value;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_mem_pool_get_attribute: use streaming layer");
}

pyre_status_t pyre_mem_pool_set_attribute(pyre_mem_pool_t pool,
                                          pyre_mem_pool_attr_t attr,
                                          uint64_t value) {
  (void)pool;
  (void)attr;
  (void)value;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_mem_pool_set_attribute: use streaming layer");
}

pyre_status_t pyre_mem_pool_trim(pyre_mem_pool_t pool,
                                 size_t min_bytes_to_keep) {
  (void)pool;
  (void)min_bytes_to_keep;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_mem_pool_trim: use streaming layer");
}
