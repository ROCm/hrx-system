// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef int iree_status_t;
typedef int iree_zone_id_t;

#define IREE_STATUS_OK 0

iree_status_t iree_clang_tidy_trace_status_source(void);

#define IREE_TRACE_ZONE_BEGIN(zone_id) int zone_id = 0
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) int zone_id = 0
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length) \
  int zone_id = 0
#define IREE_TRACE_ZONE_END(zone_id) (void)(zone_id)
#define IREE_TRACE_ZONE_ADOPT(zone_id) (void)(zone_id)
#define IREE_TRACE_ZONE_TRANSFER(zone_id) (zone_id) = 0

#define IREE_RETURN_IF_ERROR(expr)                           \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) return status_macro; \
  } while (0)

#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, expr) \
  IREE_RETURN_IF_ERROR(expr)

#define IREE_RETURN_AND_END_ZONE(zone_id, expr) return (expr)

#define HRX_TRACE_ZONE_BEGIN(zone_id, name_literal) int zone_id = 0
#define HRX_TRACE_ZONE_END(zone_id) (void)(zone_id)
#define HRX_RETURN_IF_IREE_ERROR(expr)                       \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) return status_macro; \
  } while (0)
#define HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(zone_id, expr) \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) {                    \
      HRX_TRACE_ZONE_END(zone_id);                           \
      return status_macro;                                   \
    }                                                        \
  } while (0)
#define HRX_RETURN_AND_END_ZONE(zone_id, expr) \
  do {                                         \
    HRX_TRACE_ZONE_END(zone_id);               \
    return (expr);                             \
  } while (0)

#define HIP_RETURN_ERROR(error) \
  do {                          \
    int error_macro = (error);  \
    return error_macro;         \
  } while (0)

iree_status_t iree_clang_tidy_trace_zone_balanced(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_nested_balanced(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "inner");
  IREE_TRACE_ZONE_END(z1);
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_single_exit_cleanup(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_STATUS_OK;
  if (fail) {
    status = iree_clang_tidy_trace_status_source();
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_clang_tidy_trace_zone_branch_return_balanced(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fail) {
    IREE_TRACE_ZONE_END(z0);
    return iree_clang_tidy_trace_status_source();
  }
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

int iree_clang_tidy_trace_zone_branch_macro_return_balanced(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fail) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(1);
  }
  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_clang_tidy_trace_zone_plain_return(void) {
  IREE_TRACE_ZONE_BEGIN(raw_return_zone);
  return iree_clang_tidy_trace_status_source();
}

int iree_clang_tidy_trace_zone_macro_return(void) {
  IREE_TRACE_ZONE_BEGIN(hip_return_zone);
  HIP_RETURN_ERROR(1);
}

iree_status_t iree_clang_tidy_trace_zone_branch_local_zone(int flag) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (flag) {
    IREE_TRACE_ZONE_BEGIN(z1);
    IREE_TRACE_ZONE_END(z1);
  }
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_branch_replaces_handle(int flag) {
  IREE_TRACE_ZONE_BEGIN_NAMED(zi, "tick");
  if (flag) {
    IREE_TRACE_ZONE_END(zi);
    IREE_TRACE_ZONE_BEGIN_NAMED(zi_next, "tick");
    zi = zi_next;
  }
  IREE_TRACE_ZONE_END(zi);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_adopted_branch(
    int flag, iree_zone_id_t zone_id) {
  if (flag) {
    IREE_TRACE_ZONE_BEGIN(z0);
    zone_id = z0;
  } else {
    IREE_TRACE_ZONE_ADOPT(zone_id);
  }
  IREE_TRACE_ZONE_END(zone_id);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_transfer(int flag) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_zone_id_t zone_id = z0;
  if (flag) {
    IREE_TRACE_ZONE_TRANSFER(zone_id);
    return IREE_STATUS_OK;
  }
  IREE_TRACE_ZONE_END(zone_id);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_branch_missing_end(int flag) {
  IREE_TRACE_ZONE_BEGIN(branch_missing_end_outer_zone);
  if (flag) {
    IREE_TRACE_ZONE_BEGIN(branch_missing_end_inner_zone);
  }
  IREE_TRACE_ZONE_END(branch_missing_end_outer_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_branch_conditional_end(int flag) {
  IREE_TRACE_ZONE_BEGIN(branch_conditional_end_zone);
  if (flag) {
    IREE_TRACE_ZONE_END(branch_conditional_end_zone);
  }
  IREE_TRACE_ZONE_END(branch_conditional_end_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_return_and_end_if_error(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_unmatched_end(void) {
  IREE_TRACE_ZONE_BEGIN(unmatched_end_zone);
  IREE_TRACE_ZONE_END(unmatched_end_zone);
  IREE_TRACE_ZONE_END(unmatched_end_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_mismatched_end(void) {
  IREE_TRACE_ZONE_BEGIN(mismatch_outer_zone);
  IREE_TRACE_ZONE_BEGIN(mismatch_inner_zone);
  IREE_TRACE_ZONE_END(mismatch_outer_zone);
  IREE_TRACE_ZONE_END(mismatch_inner_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_mismatched_return_and_end(void) {
  IREE_TRACE_ZONE_BEGIN(return_helper_wrong_zone);
  IREE_TRACE_ZONE_BEGIN(return_helper_active_zone);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(return_helper_wrong_zone,
                                    iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(return_helper_active_zone);
  IREE_TRACE_ZONE_END(return_helper_wrong_zone);
  return IREE_STATUS_OK;
}

void iree_clang_tidy_trace_zone_missing_end(void) {
  IREE_TRACE_ZONE_BEGIN(missing_end_zone);
}

iree_status_t iree_clang_tidy_trace_zone_dynamic_end(iree_zone_id_t zone_id) {
  IREE_TRACE_ZONE_END(zone_id);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_dynamic_return_helper(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_zone_id_t zone_id = z0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id,
                                    iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_switch_return_balanced(int selector) {
  IREE_TRACE_ZONE_BEGIN(z0);
  switch (selector) {
    case 0:
      IREE_TRACE_ZONE_END(z0);
      return iree_clang_tidy_trace_status_source();
    case 1:
      IREE_TRACE_ZONE_END(z0);
      return iree_clang_tidy_trace_status_source();
    default:
      break;
  }
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

int iree_clang_tidy_trace_zone_infinite_loop_return_balanced(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  while (1) {
    IREE_TRACE_ZONE_END(z0);
    return 1;
  }
}

iree_status_t iree_clang_tidy_trace_zone_return_and_end(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE(z0, iree_clang_tidy_trace_status_source());
}

iree_status_t iree_clang_tidy_trace_zone_hrx_return_and_end_if_error(void) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx");
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(z0,
                                        iree_clang_tidy_trace_status_source());
  HRX_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_plain_return_if_error(void) {
  IREE_TRACE_ZONE_BEGIN(plain_return_if_error_zone);
  IREE_RETURN_IF_ERROR(iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(plain_return_if_error_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_hrx_plain_return_if_error(void) {
  HRX_TRACE_ZONE_BEGIN(hrx_plain_return_zone, "hrx");
  HRX_RETURN_IF_IREE_ERROR(iree_clang_tidy_trace_status_source());
  HRX_TRACE_ZONE_END(hrx_plain_return_zone);
  return IREE_STATUS_OK;
}
