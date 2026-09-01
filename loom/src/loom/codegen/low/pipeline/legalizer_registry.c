// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/legalizer_registry.h"

#include "loom/transforms/buffer/target_legalization.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/view/target_legalization.h"

iree_status_t loom_low_legalizer_registry_storage_initialize(
    loom_target_legalizer_provider_list_t target_provider_list,
    iree_allocator_t allocator,
    loom_target_legalizer_registry_storage_t* out_storage) {
  const loom_target_legalizer_provider_t* generic_providers[] = {
      loom_buffer_target_legalizer_provider(),
      loom_scalar_target_legalizer_provider(),
      loom_vector_target_legalizer_provider(),
      loom_view_target_legalizer_provider(),
  };
  const loom_target_legalizer_provider_list_t provider_lists[] = {
      target_provider_list,
      loom_target_legalizer_provider_list_make(
          generic_providers, IREE_ARRAYSIZE(generic_providers)),
  };
  return loom_target_legalizer_registry_storage_initialize(
      provider_lists, IREE_ARRAYSIZE(provider_lists), allocator, out_storage);
}
