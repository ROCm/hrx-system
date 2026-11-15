// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/entry_interface.h"

static iree_status_t loom_spirv_entry_interface_append(
    loom_spirv_entry_interface_t* interface, uint32_t function_id,
    loom_spirv_interface_variable_t* variable, iree_arena_allocator_t* arena) {
  if (variable->last_function_id == function_id) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(arena, interface->count, interface->count + 1,
                            sizeof(*interface->variables), &interface->capacity,
                            (void**)&interface->variables));
  interface->variables[interface->count++] = variable;
  variable->last_function_id = function_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_entry_interface_declare(
    loom_spirv_entry_interface_t* interface, uint32_t function_id,
    uint32_t variable_id, iree_arena_allocator_t* arena) {
  loom_spirv_interface_variable_t* variable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*variable), (void**)&variable));
  *variable = (loom_spirv_interface_variable_t){.id = variable_id};
  return loom_spirv_entry_interface_append(interface, function_id, variable,
                                           arena);
}

iree_status_t loom_spirv_entry_interface_include(
    loom_spirv_entry_interface_t* interface, uint32_t function_id,
    const loom_spirv_entry_interface_t* callee, iree_arena_allocator_t* arena) {
  for (iree_host_size_t i = 0; i < callee->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_entry_interface_append(
        interface, function_id, callee->variables[i], arena));
  }
  return iree_ok_status();
}
