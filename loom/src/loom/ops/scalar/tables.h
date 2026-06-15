// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.

#ifndef LOOM_OPS_SCALAR_TABLES_H_
#define LOOM_OPS_SCALAR_TABLES_H_

#include "loom/ops/scalar/ops.h"

#define _BSTRING(length, value) LOOM_BSTRING_REF(length, value)
#define _OP_NAME(length, namespace_length, value) \
  LOOM_OP_NAME_REF(length, namespace_length, value)

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_op_vtable_t loom_scalar_addi_vtable;
extern const loom_op_vtable_t loom_scalar_subi_vtable;
extern const loom_op_vtable_t loom_scalar_muli_vtable;
extern const loom_op_vtable_t loom_scalar_divsi_vtable;
extern const loom_op_vtable_t loom_scalar_divui_vtable;
extern const loom_op_vtable_t loom_scalar_remsi_vtable;
extern const loom_op_vtable_t loom_scalar_remui_vtable;
extern const loom_op_vtable_t loom_scalar_ceildivsi_vtable;
extern const loom_op_vtable_t loom_scalar_ceildivui_vtable;
extern const loom_op_vtable_t loom_scalar_floordivsi_vtable;
extern const loom_op_vtable_t loom_scalar_negi_vtable;
extern const loom_op_vtable_t loom_scalar_absi_vtable;
extern const loom_op_vtable_t loom_scalar_minsi_vtable;
extern const loom_op_vtable_t loom_scalar_maxsi_vtable;
extern const loom_op_vtable_t loom_scalar_minui_vtable;
extern const loom_op_vtable_t loom_scalar_maxui_vtable;
extern const loom_op_vtable_t loom_scalar_fmai_vtable;
extern const loom_op_vtable_t loom_scalar_addf_vtable;
extern const loom_op_vtable_t loom_scalar_subf_vtable;
extern const loom_op_vtable_t loom_scalar_mulf_vtable;
extern const loom_op_vtable_t loom_scalar_divf_vtable;
extern const loom_op_vtable_t loom_scalar_remf_vtable;
extern const loom_op_vtable_t loom_scalar_negf_vtable;
extern const loom_op_vtable_t loom_scalar_absf_vtable;
extern const loom_op_vtable_t loom_scalar_minimumf_vtable;
extern const loom_op_vtable_t loom_scalar_maximumf_vtable;
extern const loom_op_vtable_t loom_scalar_minnumf_vtable;
extern const loom_op_vtable_t loom_scalar_maxnumf_vtable;
extern const loom_op_vtable_t loom_scalar_clampf_vtable;
extern const loom_op_vtable_t loom_scalar_copysignf_vtable;
extern const loom_op_vtable_t loom_scalar_expf_vtable;
extern const loom_op_vtable_t loom_scalar_exp2f_vtable;
extern const loom_op_vtable_t loom_scalar_expm1f_vtable;
extern const loom_op_vtable_t loom_scalar_logf_vtable;
extern const loom_op_vtable_t loom_scalar_log2f_vtable;
extern const loom_op_vtable_t loom_scalar_log10f_vtable;
extern const loom_op_vtable_t loom_scalar_log1pf_vtable;
extern const loom_op_vtable_t loom_scalar_powf_vtable;
extern const loom_op_vtable_t loom_scalar_sqrtf_vtable;
extern const loom_op_vtable_t loom_scalar_rsqrtf_vtable;
extern const loom_op_vtable_t loom_scalar_cbrtf_vtable;
extern const loom_op_vtable_t loom_scalar_sinf_vtable;
extern const loom_op_vtable_t loom_scalar_cosf_vtable;
extern const loom_op_vtable_t loom_scalar_sinturnsf_vtable;
extern const loom_op_vtable_t loom_scalar_costurnsf_vtable;
extern const loom_op_vtable_t loom_scalar_tanf_vtable;
extern const loom_op_vtable_t loom_scalar_asinf_vtable;
extern const loom_op_vtable_t loom_scalar_acosf_vtable;
extern const loom_op_vtable_t loom_scalar_atanf_vtable;
extern const loom_op_vtable_t loom_scalar_atan2f_vtable;
extern const loom_op_vtable_t loom_scalar_sinhf_vtable;
extern const loom_op_vtable_t loom_scalar_coshf_vtable;
extern const loom_op_vtable_t loom_scalar_tanhf_vtable;
extern const loom_op_vtable_t loom_scalar_asinhf_vtable;
extern const loom_op_vtable_t loom_scalar_acoshf_vtable;
extern const loom_op_vtable_t loom_scalar_atanhf_vtable;
extern const loom_op_vtable_t loom_scalar_erff_vtable;
extern const loom_op_vtable_t loom_scalar_erfcf_vtable;
extern const loom_op_vtable_t loom_scalar_logisticf_vtable;
extern const loom_op_vtable_t loom_scalar_siluf_vtable;
extern const loom_op_vtable_t loom_scalar_softplusf_vtable;
extern const loom_op_vtable_t loom_scalar_geluf_vtable;
extern const loom_op_vtable_t loom_scalar_fmaf_vtable;
extern const loom_op_vtable_t loom_scalar_ceilf_vtable;
extern const loom_op_vtable_t loom_scalar_floorf_vtable;
extern const loom_op_vtable_t loom_scalar_roundf_vtable;
extern const loom_op_vtable_t loom_scalar_roundevenf_vtable;
extern const loom_op_vtable_t loom_scalar_truncf_vtable;
extern const loom_op_vtable_t loom_scalar_cmpi_vtable;
extern const loom_op_vtable_t loom_scalar_cmpf_vtable;
extern const loom_op_vtable_t loom_scalar_isnanf_vtable;
extern const loom_op_vtable_t loom_scalar_isinff_vtable;
extern const loom_op_vtable_t loom_scalar_isfinitef_vtable;
extern const loom_op_vtable_t loom_scalar_signf_vtable;
extern const loom_op_vtable_t loom_scalar_signi_vtable;
extern const loom_op_vtable_t loom_scalar_sitofp_vtable;
extern const loom_op_vtable_t loom_scalar_uitofp_vtable;
extern const loom_op_vtable_t loom_scalar_fptosi_vtable;
extern const loom_op_vtable_t loom_scalar_fptoui_vtable;
extern const loom_op_vtable_t loom_scalar_extf_vtable;
extern const loom_op_vtable_t loom_scalar_fptrunc_vtable;
extern const loom_op_vtable_t loom_scalar_extsi_vtable;
extern const loom_op_vtable_t loom_scalar_extui_vtable;
extern const loom_op_vtable_t loom_scalar_trunci_vtable;
extern const loom_op_vtable_t loom_scalar_bitcast_vtable;
extern const loom_op_vtable_t loom_scalar_constant_vtable;
extern const loom_op_vtable_t loom_scalar_poison_vtable;
extern const loom_op_vtable_t loom_scalar_andi_vtable;
extern const loom_op_vtable_t loom_scalar_ori_vtable;
extern const loom_op_vtable_t loom_scalar_xori_vtable;
extern const loom_op_vtable_t loom_scalar_shli_vtable;
extern const loom_op_vtable_t loom_scalar_shrsi_vtable;
extern const loom_op_vtable_t loom_scalar_shrui_vtable;
extern const loom_op_vtable_t loom_scalar_rotli_vtable;
extern const loom_op_vtable_t loom_scalar_rotri_vtable;
extern const loom_op_vtable_t loom_scalar_ctlzi_vtable;
extern const loom_op_vtable_t loom_scalar_cttzi_vtable;
extern const loom_op_vtable_t loom_scalar_ctpopi_vtable;
extern const loom_op_vtable_t loom_scalar_bitfield_extractu_vtable;
extern const loom_op_vtable_t loom_scalar_bitfield_extracts_vtable;
extern const loom_op_vtable_t loom_scalar_assume_vtable;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCALAR_TABLES_H_
