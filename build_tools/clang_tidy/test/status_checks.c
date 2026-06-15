// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

typedef struct iree_status_handle_t* iree_status_t;
typedef struct iree_hal_amdgpu_reclaim_entry_t iree_hal_amdgpu_reclaim_entry_t;
typedef intptr_t iree_host_size_t;
typedef int iree_status_code_t;
typedef void (*iree_clang_tidy_status_const_callback_t)(
    void* user_data, const iree_status_t status);

iree_status_t iree_ok_status(void);
int iree_status_is_ok(iree_status_t status);
void iree_status_free(iree_status_t status);
iree_status_t iree_status_ignore(iree_status_t status);
iree_status_t iree_status_join(iree_status_t base_status,
                               iree_status_t new_status);
iree_status_t iree_status_annotate(iree_status_t base_status,
                                   const char* message);
iree_status_code_t iree_status_consume_code(iree_status_t status);
int iree_status_format(iree_status_t status, iree_host_size_t buffer_capacity,
                       char* buffer, iree_host_size_t* out_buffer_length);
void iree_status_fprint(void* file, iree_status_t status);
int iree_atomic_compare_exchange_strong(int* ptr, int* expected,
                                        intptr_t desired, int success_order,
                                        int failure_order);
iree_status_t iree_async_socket_query_failure(void* socket);
iree_status_t iree_status_clone(iree_status_t status);
void iree_async_proactor_io_uring_push_software_completion(
    void* proactor, void* operation, iree_status_t status);

iree_status_t iree_clang_tidy_status_assigned_source(void);
iree_status_t iree_clang_tidy_status_cleanup_source(void);
iree_status_t iree_clang_tidy_status_dropped_source(void);
iree_status_t iree_clang_tidy_status_fix_source(void);
iree_status_t iree_clang_tidy_status_ignored_source(void);
iree_status_t iree_clang_tidy_status_named_call_source(void);
iree_status_t iree_clang_tidy_status_returned_source(void);
iree_status_t iree_clang_tidy_status_void_cast_source(void);
iree_status_t iree_clang_tidy_status_transfer_order_sink(iree_status_t status);
void iree_clang_tidy_status_lifetime_const_borrowed_observer(
    const iree_status_t status);

#define IREE_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_STATUS_OK 0
#define IREE_STATUS_UNAVAILABLE 14
#define IREE_STATUS_CODE_MASK 0x1Fu
#define iree_status_code(value) \
  ((iree_status_code_t)(((uintptr_t)(value)) & IREE_STATUS_CODE_MASK))
#define iree_status_is_ok(value) \
  IREE_LIKELY((uintptr_t)(value) == IREE_STATUS_OK)
#define iree_status_is_unavailable(value) \
  (iree_status_code(value) == IREE_STATUS_UNAVAILABLE)
#define IREE_RETURN_IF_ERROR(expr)            \
  do {                                        \
    iree_status_t __status_macro = (expr);    \
    if (!iree_status_is_ok(__status_macro)) { \
      return __status_macro;                  \
    }                                         \
  } while (0)

iree_status_t iree_clang_tidy_status_returned(void) {
  return iree_clang_tidy_status_returned_source();
}

void iree_clang_tidy_status_assigned(void) {
  iree_status_t status = iree_clang_tidy_status_assigned_source();
  (void)status;
}

void iree_clang_tidy_status_ignored(void) {
  iree_status_ignore(iree_clang_tidy_status_ignored_source());
}

void iree_clang_tidy_status_dropped(void) {
  iree_clang_tidy_status_dropped_source();
}

void iree_clang_tidy_status_void_cast(void) {
  (void)iree_clang_tidy_status_void_cast_source();
}

void iree_clang_tidy_status_named_call_dropped(void) {
  iree_clang_tidy_status_named_call_source();
}

void iree_clang_tidy_status_lifetime_leaked(void) {
  iree_status_t leaked_status = iree_clang_tidy_status_assigned_source();
  (void)leaked_status;
}

void iree_clang_tidy_status_lifetime_overwritten(void) {
  iree_status_t overwritten_status = iree_clang_tidy_status_assigned_source();
  overwritten_status = iree_clang_tidy_status_cleanup_source();
  iree_status_ignore(overwritten_status);
}

void iree_clang_tidy_status_lifetime_double_consumed(void) {
  iree_status_t double_consumed_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_ignore(double_consumed_status);
  iree_status_free(double_consumed_status);
}

iree_status_t iree_clang_tidy_status_lifetime_used_after_consume(void) {
  iree_status_t used_after_consume_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_ignore(used_after_consume_status);
  return used_after_consume_status;
}

iree_status_t iree_clang_tidy_status_lifetime_used_after_transfer(void) {
  iree_status_t used_after_transfer_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t transferred_status = used_after_transfer_status;
  iree_status_ignore(transferred_status);
  return used_after_transfer_status;
}

iree_status_t iree_clang_tidy_status_lifetime_lost_on_return(void) {
  iree_status_t lost_on_return_status =
      iree_clang_tidy_status_assigned_source();
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_status_lifetime_lost_on_code_predicate(void) {
  iree_status_t code_predicate_status =
      iree_clang_tidy_status_assigned_source();
  if (iree_status_is_unavailable(code_predicate_status)) {
    return iree_clang_tidy_status_cleanup_source();
  }
  return code_predicate_status;
}

iree_status_t iree_clang_tidy_status_lifetime_returned(void) {
  iree_status_t returned_status = iree_clang_tidy_status_assigned_source();
  return returned_status;
}

iree_status_t iree_clang_tidy_status_lifetime_return_if_error(void) {
  iree_status_t return_if_error_status =
      iree_clang_tidy_status_assigned_source();
  if (!iree_status_is_ok(return_if_error_status)) {
    return return_if_error_status;
  }
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_status_lifetime_return_if_error_macro(void) {
  iree_status_t macro_return_status = iree_clang_tidy_status_assigned_source();
  IREE_RETURN_IF_ERROR(macro_return_status);
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_status_lifetime_return_if_error_fallthrough(
    void) {
  iree_status_t macro_fallthrough_status =
      iree_clang_tidy_status_assigned_source();
  IREE_RETURN_IF_ERROR(macro_fallthrough_status);
  if (!iree_status_is_ok(macro_fallthrough_status)) {
    return macro_fallthrough_status;
  }
  return iree_clang_tidy_status_cleanup_source();
}

iree_status_t iree_clang_tidy_status_lifetime_transferred_local(void) {
  iree_status_t transfer_source_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t transfer_target_status = transfer_source_status;
  return transfer_target_status;
}

iree_status_t iree_clang_tidy_status_lifetime_transferred_conditional(void) {
  iree_status_t conditional_source_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t conditional_target_status =
      iree_status_is_ok(conditional_source_status)
          ? iree_clang_tidy_status_cleanup_source()
          : conditional_source_status;
  return conditional_target_status;
}

iree_status_t iree_clang_tidy_status_lifetime_joined_secondary_error(void) {
  iree_status_t joined_primary_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t joined_secondary_status =
      iree_clang_tidy_status_cleanup_source();
  if (!iree_status_is_ok(joined_secondary_status)) {
    return iree_status_join(joined_primary_status, joined_secondary_status);
  }
  return joined_primary_status;
}

iree_status_t iree_clang_tidy_status_lifetime_loop_break_overwrite(void) {
  iree_status_t loop_break_status = iree_ok_status();
  for (int i = 0; i < 4; ++i) {
    loop_break_status = iree_clang_tidy_status_assigned_source();
    if (!iree_status_is_ok(loop_break_status)) break;
    loop_break_status = iree_clang_tidy_status_cleanup_source();
    if (!iree_status_is_ok(loop_break_status)) break;
  }
  return loop_break_status;
}

iree_status_t iree_clang_tidy_status_lifetime_ok_and_overwrite(int replace) {
  iree_status_t ok_and_status = iree_clang_tidy_status_assigned_source();
  if (iree_status_is_ok(ok_and_status) && replace) {
    ok_and_status = iree_clang_tidy_status_cleanup_source();
  }
  return ok_and_status;
}

iree_status_t iree_clang_tidy_status_lifetime_immediate_ok_overwrite(void) {
  iree_status_t immediate_ok_overwrite_status = iree_ok_status();
  immediate_ok_overwrite_status = iree_clang_tidy_status_assigned_source();
  return immediate_ok_overwrite_status;
}

iree_status_t iree_clang_tidy_status_lifetime_fix_immediate_ok_overwrite(void) {
  iree_status_t fix_status = iree_ok_status();
  fix_status = iree_clang_tidy_status_fix_source();
  return fix_status;
}

iree_status_t iree_clang_tidy_status_lifetime_conditional_ok_assignment(
    int replace) {
  iree_status_t conditional_ok_status = iree_ok_status();
  if (replace) {
    conditional_ok_status = iree_clang_tidy_status_assigned_source();
  }
  return conditional_ok_status;
}

iree_status_t iree_clang_tidy_status_lifetime_joined_ok_status(void) {
  iree_status_t joined_ok_status = iree_ok_status();
  joined_ok_status = iree_status_join(joined_ok_status,
                                      iree_clang_tidy_status_cleanup_source());
  return joined_ok_status;
}

iree_status_t iree_clang_tidy_status_lifetime_code_predicate_consumed(void) {
  iree_status_t consumed_predicate_status =
      iree_clang_tidy_status_assigned_source();
  if (iree_status_is_unavailable(consumed_predicate_status)) {
    iree_status_ignore(consumed_predicate_status);
    return iree_clang_tidy_status_cleanup_source();
  }
  return consumed_predicate_status;
}

void iree_clang_tidy_status_lifetime_atomic_escape(int* ptr, int* expected) {
  iree_status_t atomic_escape_status = iree_clang_tidy_status_assigned_source();
  if (!iree_atomic_compare_exchange_strong(
          ptr, expected, (intptr_t)atomic_escape_status, 0, 0)) {
    iree_status_ignore(atomic_escape_status);
  }
}

void iree_clang_tidy_status_lifetime_atomic_builtin_escape(
    _Atomic(intptr_t)* ptr, intptr_t* expected) {
  iree_status_t atomic_builtin_status =
      iree_clang_tidy_status_assigned_source();
  if (!__c11_atomic_compare_exchange_strong(
          ptr, expected, (intptr_t)atomic_builtin_status, __ATOMIC_ACQ_REL,
          __ATOMIC_RELAXED)) {
    iree_status_ignore(atomic_builtin_status);
  }
}

void iree_clang_tidy_status_lifetime_sink_argument(void* proactor,
                                                   void* operation) {
  iree_status_t sink_argument_status = iree_clang_tidy_status_assigned_source();
  iree_async_proactor_io_uring_push_software_completion(proactor, operation,
                                                        sink_argument_status);
}

iree_status_t iree_clang_tidy_status_lifetime_borrowed_status(void* socket) {
  iree_status_t borrowed_status = iree_async_socket_query_failure(socket);
  if (!iree_status_is_ok(borrowed_status)) {
    return iree_status_clone(borrowed_status);
  }
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_status_lifetime_joined(void) {
  iree_status_t joined_status = iree_clang_tidy_status_assigned_source();
  joined_status =
      iree_status_join(joined_status, iree_clang_tidy_status_cleanup_source());
  return joined_status;
}

iree_status_t iree_clang_tidy_status_lifetime_annotated(void) {
  iree_status_t annotated_status = iree_clang_tidy_status_assigned_source();
  if (annotated_status) {
    return iree_status_annotate(annotated_status, "message");
  }
  return iree_ok_status();
}

void iree_clang_tidy_status_lifetime_stored(iree_status_t* out_status) {
  iree_status_t stored_status = iree_clang_tidy_status_assigned_source();
  *out_status = stored_status;
}

iree_status_code_t iree_clang_tidy_status_lifetime_consumed_code(void) {
  iree_status_t consumed_code_status = iree_clang_tidy_status_assigned_source();
  return iree_status_consume_code(consumed_code_status);
}

void iree_clang_tidy_status_lifetime_reported_ignore(void* file) {
  iree_status_t reported_ignore_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_fprint(file, reported_ignore_status);
  iree_status_ignore(reported_ignore_status);
}

void iree_clang_tidy_status_lifetime_formatted_ignore(char* buffer) {
  iree_host_size_t buffer_length = 0;
  iree_status_t formatted_ignore_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_format(formatted_ignore_status, 64, buffer, &buffer_length);
  iree_status_ignore(formatted_ignore_status);
}

void iree_clang_tidy_status_lifetime_reported_free(void* file) {
  iree_status_t reported_free_status = iree_clang_tidy_status_assigned_source();
  iree_status_fprint(file, reported_free_status);
  iree_status_free(reported_free_status);
}

iree_status_t iree_clang_tidy_status_lifetime_const_borrowed_observed(void) {
  iree_status_t const_borrowed_observed_status =
      iree_clang_tidy_status_assigned_source();
  iree_clang_tidy_status_lifetime_const_borrowed_observer(
      const_borrowed_observed_status);
  return const_borrowed_observed_status;
}

iree_status_t iree_clang_tidy_status_lifetime_const_callback_observed(
    iree_clang_tidy_status_const_callback_t callback, void* user_data) {
  iree_status_t const_callback_observed_status =
      iree_clang_tidy_status_assigned_source();
  callback(user_data, const_callback_observed_status);
  return const_callback_observed_status;
}

int iree_clang_tidy_status_borrowed_parameter(
    iree_status_t borrowed_parameter_status) {
  return iree_status_is_ok(borrowed_parameter_status);
}

int iree_clang_tidy_status_borrowed_parameter_code(
    iree_status_code_t status_code) {
  return status_code == IREE_STATUS_OK;
}

int iree_clang_tidy_status_borrowed_parameter_const(
    const iree_status_t const_borrowed_parameter_status) {
  iree_status_t cloned_status =
      iree_status_clone(const_borrowed_parameter_status);
  iree_status_free(cloned_status);
  return iree_status_is_ok(const_borrowed_parameter_status);
}

void iree_clang_tidy_status_borrowed_parameter_const_consumed(
    const iree_status_t const_status_consumed) {
  iree_status_free(const_status_consumed);
}

void iree_clang_tidy_status_borrowed_parameter_const_clone_stored(
    const iree_status_t const_clone_stored_parameter_status,
    iree_status_t* out_status) {
  *out_status = iree_status_clone(const_clone_stored_parameter_status);
}

void iree_clang_tidy_status_borrowed_parameter_const_callback(
    iree_clang_tidy_status_const_callback_t callback, void* user_data,
    const iree_status_t const_callback_parameter_status) {
  callback(user_data, const_callback_parameter_status);
}

iree_status_t iree_clang_tidy_status_borrowed_parameter_returned(
    iree_status_t returned_parameter_status) {
  return returned_parameter_status;
}

void iree_clang_tidy_status_borrowed_parameter_consumed(
    iree_status_t consumed_parameter_status) {
  iree_status_free(consumed_parameter_status);
}

void iree_clang_tidy_status_borrowed_parameter_stored(
    iree_status_t stored_parameter_status, iree_status_t* out_status) {
  *out_status = stored_parameter_status;
}

iree_status_t iree_clang_tidy_status_borrowed_parameter_joined(
    iree_status_t joined_parameter_status) {
  return iree_status_join(joined_parameter_status,
                          iree_clang_tidy_status_cleanup_source());
}

iree_status_t iree_clang_tidy_status_borrowed_parameter_annotated(
    iree_status_t annotated_parameter_status) {
  return iree_status_annotate(annotated_parameter_status, "message");
}

iree_status_t iree_clang_tidy_status_borrowed_parameter_cloned(
    iree_status_t cloned_parameter_status) {
  return iree_status_clone(cloned_parameter_status);
}

void iree_clang_tidy_status_borrowed_parameter_sink(
    void* proactor, void* operation, iree_status_t sink_parameter_status) {
  iree_async_proactor_io_uring_push_software_completion(proactor, operation,
                                                        sink_parameter_status);
}

void iree_clang_tidy_status_borrowed_parameter_reclaim_callback(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t reclaim_callback_parameter_status) {
  (void)entry;
  *(iree_status_code_t*)user_data =
      iree_status_code(reclaim_callback_parameter_status);
}

iree_status_t iree_clang_tidy_status_transfer_order_join_same(void) {
  iree_status_t transfer_order_join_same_status =
      iree_clang_tidy_status_assigned_source();
  return iree_status_join(transfer_order_join_same_status,
                          transfer_order_join_same_status);
}

iree_status_t iree_clang_tidy_status_transfer_order_join_callback(void) {
  iree_status_t transfer_order_nested_status =
      iree_clang_tidy_status_assigned_source();
  return iree_status_join(
      transfer_order_nested_status,
      iree_clang_tidy_status_transfer_order_sink(transfer_order_nested_status));
}

iree_status_t iree_clang_tidy_status_transfer_order_explicit_sequence(void) {
  iree_status_t explicit_sequence_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t cleanup_status = iree_clang_tidy_status_cleanup_source();
  return iree_status_join(explicit_sequence_status, cleanup_status);
}

iree_status_t iree_clang_tidy_status_transfer_order_clone_before_fanout(void) {
  iree_status_t clone_before_fanout_status =
      iree_clang_tidy_status_assigned_source();
  iree_status_t cloned_status = iree_status_clone(clone_before_fanout_status);
  iree_status_t callback_status =
      iree_clang_tidy_status_transfer_order_sink(cloned_status);
  return iree_status_join(clone_before_fanout_status, callback_status);
}

int iree_clang_tidy_status_transfer_order_observer_only(void) {
  iree_status_t observer_only_status = iree_clang_tidy_status_assigned_source();
  int result = iree_status_is_ok(observer_only_status) ||
               iree_status_is_unavailable(observer_only_status);
  iree_status_ignore(observer_only_status);
  return result;
}
