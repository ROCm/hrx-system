// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/bank.h"

#include "iree/base/internal/math.h"

struct loom_serve_parameter_bank_t {
  // Allocator owning this bank and its trailing buffer/plan arrays.
  iree_allocator_t host_allocator;
  // Number of domain slots, including NULL slots during partial construction.
  iree_host_size_t count;
  // Owned physical group; every invocation residency borrows this registration.
  loom_serve_residency_group_t* group;
  // Owned fixed-root views followed by count owned weight-plan pointers and
  // count alignment exponents (log2 of each root's promised byte alignment).
  iree_hal_buffer_t* buffers[];
};

static loom_serve_weights_t** loom_serve_parameter_bank_plans(
    const loom_serve_parameter_bank_t* bank) {
  return (loom_serve_weights_t**)(bank->buffers + bank->count);
}

static uint8_t* loom_serve_parameter_bank_alignments(
    const loom_serve_parameter_bank_t* bank) {
  return (uint8_t*)(loom_serve_parameter_bank_plans(bank) + bank->count);
}

static iree_status_t loom_serve_parameter_bank_allocate_range(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    uint32_t first, uint32_t count, iree_allocator_t host_allocator,
    loom_serve_parameter_bank_t** out_bank) {
  *out_bank = NULL;
  if (!count) {
    return iree_ok_status();
  }
  loom_serve_parameter_bank_t* bank = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*bank), count,
      sizeof(iree_hal_buffer_t*) + sizeof(loom_serve_weights_t*) +
          sizeof(uint8_t),
      (void**)&bank));
  bank->host_allocator = host_allocator;
  bank->count = count;
  *out_bank = bank;
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_root_t reflected =
        loom_cmd_program_parameter_root_at(program, first + i);
    loom_serve_parameter_bank_alignments(bank)[i] =
        (uint8_t)iree_math_count_trailing_zeros_u64(
            iree_max(reflected.minimum_alignment, 1));
    const loom_serve_weight_root_t root = {program, reflected,
                                           &bank->buffers[i]};
    status =
        loom_serve_weights_allocate(loom_serve_device_handle(device),
                                    loom_serve_device_transfer_queue(device),
                                    loom_serve_device_dispatch_queue(device),
                                    loom_serve_device_storage_pool(device),
                                    loom_serve_device_memory_pool(device),
                                    loom_serve_device_weight_geometry(device),
                                    NULL, loom_serve_device_workspace(device),
                                    0, 1, &root, host_allocator, &plans[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_group_allocate(
        loom_serve_device_residency_cache(device), count, plans, host_allocator,
        &bank->group);
  }
  return status;
}

iree_status_t loom_serve_parameter_bank_allocate(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    iree_allocator_t host_allocator, loom_serve_parameter_bank_t** out_bank) {
  return loom_serve_parameter_bank_allocate_range(
      device, program, 0,
      program ? program->requirements.fixed_buffer_count : 0, host_allocator,
      out_bank);
}

iree_status_t loom_serve_parameter_bank_allocate_domain(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    uint32_t domain, iree_allocator_t host_allocator,
    loom_serve_parameter_bank_t** out_bank) {
  return loom_serve_parameter_bank_allocate_range(device, program, domain, 1,
                                                  host_allocator, out_bank);
}

iree_host_size_t loom_serve_parameter_bank_count(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->count : 0;
}

loom_serve_weights_t* loom_serve_parameter_bank_weights(
    const loom_serve_parameter_bank_t* bank, iree_host_size_t index) {
  return loom_serve_parameter_bank_plans(bank)[index];
}

iree_hal_buffer_t* const* loom_serve_parameter_bank_buffers(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->buffers : NULL;
}

loom_serve_residency_group_t* loom_serve_parameter_bank_group(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->group : NULL;
}

static iree_status_t loom_serve_parameter_bindings_check(
    iree_host_size_t count, const loom_serve_parameter_binding_t* bindings,
    const loom_cmd_program_t* program, uint32_t* cursors) {
  if (count != (program ? program->requirements.fixed_buffer_count : 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage checkpoint domain count differs");
  }
  if (!count) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_parameter_binding_t binding = bindings[i];
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, i);
    const uint8_t alignment =
        loom_serve_parameter_bank_alignments(binding.bank)[binding.domain];
    if (iree_hal_buffer_byte_length(binding.bank->buffers[binding.domain]) !=
            root.required_byte_length ||
        (UINT64_C(1) << alignment) != iree_max(root.minimum_alignment, 1)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter root %zu differs", i);
    }
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < program->parameters.count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_t actual =
        loom_cmd_program_parameter_at(program, i);
    const uint32_t ordinal = actual.fixed_buffer_index;
    const loom_serve_parameter_binding_t binding = bindings[ordinal];
    const loom_serve_weights_t* plan =
        loom_serve_parameter_bank_weights(binding.bank, binding.domain);
    const uint32_t cursor = cursors[ordinal]++;
    if (cursor == loom_serve_weights_parameter_count(plan)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "stage parameter domain %u has extra rows", ordinal);
    } else {
      const loom_cmd_program_parameter_t expected =
          loom_serve_weights_parameter_at(plan, cursor);
      if (!iree_string_view_equal(expected.key, actual.key) ||
          expected.byte_offset != actual.byte_offset ||
          expected.byte_length != actual.byte_length ||
          expected.minimum_alignment != actual.minimum_alignment) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stage parameter placement %u differs", i);
      }
    }
  }
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_serve_parameter_binding_t binding = bindings[i];
    if (cursors[i] !=
        loom_serve_weights_parameter_count(
            loom_serve_parameter_bank_weights(binding.bank, binding.domain))) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "stage parameter domain %zu is missing rows", i);
    }
  }
  return status;
}

iree_status_t loom_serve_parameter_bindings_check_layout(
    iree_host_size_t count, const loom_serve_parameter_binding_t* bindings,
    const loom_cmd_program_t* program, iree_allocator_t host_allocator) {
  uint32_t inline_cursors[8] = {0};
  uint32_t* cursors = inline_cursors;
  if (count > IREE_ARRAYSIZE(inline_cursors)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, count, sizeof(*cursors), (void**)&cursors));
  }
  iree_status_t status =
      loom_serve_parameter_bindings_check(count, bindings, program, cursors);
  if (cursors != inline_cursors) {
    iree_allocator_free(host_allocator, cursors);
  }
  return status;
}

iree_status_t loom_serve_parameter_bank_check_layout(
    const loom_serve_parameter_bank_t* bank,
    const loom_cmd_program_t* program) {
  const iree_host_size_t count = loom_serve_parameter_bank_count(bank);
  loom_serve_parameter_binding_t inline_bindings[8];
  uint32_t inline_cursors[8] = {0};
  loom_serve_parameter_binding_t* bindings = inline_bindings;
  uint32_t* cursors = inline_cursors;
  if (count > IREE_ARRAYSIZE(inline_bindings)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        bank->host_allocator, count, sizeof(*bindings) + sizeof(*cursors),
        (void**)&bindings));
    cursors = (uint32_t*)(bindings + count);
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    bindings[i] = (loom_serve_parameter_binding_t){bank, (uint32_t)i};
  }
  iree_status_t status =
      loom_serve_parameter_bindings_check(count, bindings, program, cursors);
  if (bindings != inline_bindings) {
    iree_allocator_free(bank->host_allocator, bindings);
  }
  return status;
}

bool loom_serve_parameter_bank_matches_domain(
    const loom_serve_parameter_bank_t* bank, const loom_cmd_program_t* program,
    uint32_t domain) {
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(program, domain);
  if (iree_hal_buffer_byte_length(bank->buffers[0]) !=
          root.required_byte_length ||
      (UINT64_C(1) << loom_serve_parameter_bank_alignments(bank)[0]) !=
          iree_max(root.minimum_alignment, 1)) {
    return false;
  }
  const loom_serve_weights_t* weights =
      loom_serve_parameter_bank_plans(bank)[0];
  const iree_host_size_t count = loom_serve_weights_parameter_count(weights);
  iree_host_size_t cursor = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t actual =
        loom_cmd_program_parameter_at(program, i);
    if (actual.fixed_buffer_index != domain) {
      continue;
    }
    if (cursor == count) {
      return false;
    }
    const loom_cmd_program_parameter_t expected =
        loom_serve_weights_parameter_at(weights, cursor++);
    if (!iree_string_view_equal(expected.key, actual.key) ||
        expected.byte_offset != actual.byte_offset ||
        expected.byte_length != actual.byte_length ||
        expected.minimum_alignment != actual.minimum_alignment) {
      return false;
    }
  }
  return cursor == count;
}

void loom_serve_parameter_bank_retire(loom_serve_parameter_bank_t* bank,
                                      void (*wake)(void* user_data),
                                      void* user_data) {
  if (!bank) {
    return;
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    loom_serve_weights_set_retirement_observer(plans[i], wake, user_data);
    iree_hal_buffer_release(bank->buffers[i]);
    bank->buffers[i] = NULL;
  }
}

bool loom_serve_parameter_bank_is_retired(loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return true;
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  bool retired = true;
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    retired &= loom_serve_weights_is_retired(plans[i]);
  }
  return retired;
}

iree_status_t loom_serve_parameter_bank_await_retirement(
    loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return iree_ok_status();
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    status =
        iree_status_join(status, loom_serve_weights_await_retirement(plans[i]));
  }
  return status;
}

iree_status_t loom_serve_parameter_bank_free(
    loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return iree_ok_status();
  }
  loom_serve_residency_group_free(bank->group);
  bank->group = NULL;
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bank->count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_weights_free(plans[i]);
    if (iree_status_is_ok(status)) {
      plans[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(bank->host_allocator, bank);
  }
  return status;
}
