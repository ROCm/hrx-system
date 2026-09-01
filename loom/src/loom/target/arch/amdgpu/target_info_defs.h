// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU processor and descriptor-set facts.
//
// The declarations here are the stable C contract for target-owned AMDGPU
// facts. Generated target-info tables provide row data; checked-in C owns the
// lookup and parsing algorithms. This header owns the public types so C API
// design does not live inside the table generator. Target-info generators may
// include this header, but must not emit public struct or enum definitions for
// these facts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable target-family identity for AMDGPU low descriptor sets.
#define LOOM_AMDGPU_TARGET_STABLE_ID UINT64_C(0x6c46df5542915cc5)

// Sentinel for a processor relation without a referenced processor.
#define LOOM_AMDGPU_PROCESSOR_ORDINAL_NONE UINT16_MAX

// Sentinel for a processor without target-low descriptor support.
#define LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE UINT16_MAX

// Sentinel for a target without an exact LDS bank-service model set.
#define LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE UINT16_MAX

// Dense generated reference to an immutable LDS bank-service model set.
typedef uint16_t loom_amdgpu_lds_bank_service_model_set_ordinal_t;

typedef enum loom_amdgpu_elf_feature_flag_bits_e {
  // Mask selecting the AMDHSA code-object v4+ XNACK feature state.
  LOOM_AMDGPU_ELF_FEATURE_XNACK_MASK_V4 = UINT32_C(0x300),
  // AMDHSA code-object v4+ XNACK feature state is unsupported.
  LOOM_AMDGPU_ELF_FEATURE_XNACK_UNSUPPORTED_V4 = UINT32_C(0x000),
  // AMDHSA code-object v4+ XNACK feature state accepts any agent setting.
  LOOM_AMDGPU_ELF_FEATURE_XNACK_ANY_V4 = UINT32_C(0x100),
  // AMDHSA code-object v4+ XNACK feature state requires XNACK disabled.
  LOOM_AMDGPU_ELF_FEATURE_XNACK_OFF_V4 = UINT32_C(0x200),
  // AMDHSA code-object v4+ XNACK feature state requires XNACK enabled.
  LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4 = UINT32_C(0x300),
  // Mask selecting the AMDHSA code-object v4+ SRAM ECC feature state.
  LOOM_AMDGPU_ELF_FEATURE_SRAMECC_MASK_V4 = UINT32_C(0xc00),
  // AMDHSA code-object v4+ SRAM ECC feature state is unsupported.
  LOOM_AMDGPU_ELF_FEATURE_SRAMECC_UNSUPPORTED_V4 = UINT32_C(0x000),
  // AMDHSA code-object v4+ SRAM ECC feature state accepts any agent setting.
  LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ANY_V4 = UINT32_C(0x400),
  // AMDHSA code-object v4+ SRAM ECC feature state requires SRAM ECC disabled.
  LOOM_AMDGPU_ELF_FEATURE_SRAMECC_OFF_V4 = UINT32_C(0x800),
  // AMDHSA code-object v4+ SRAM ECC feature state requires SRAM ECC enabled.
  LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ON_V4 = UINT32_C(0xc00),
} loom_amdgpu_elf_feature_flag_bits_t;

// Bitset of AMDGPU ELF EF_AMDGPU_FEATURE_* values.
typedef uint32_t loom_amdgpu_elf_feature_flags_t;

enum {
  // Bit offset of the generic code-object version in AMDHSA v6 e_flags.
  LOOM_AMDGPU_ELF_GENERIC_VERSION_OFFSET_V6 = 24u,
  // Mask selecting the generic code-object version in AMDHSA v6 e_flags.
  LOOM_AMDGPU_ELF_GENERIC_VERSION_MASK_V6 = UINT32_C(0xff000000),
};

// Normalized state of one AMDHSA target-ID feature.
typedef uint8_t loom_amdgpu_target_feature_state_t;

enum loom_amdgpu_target_feature_state_e {
  // Feature is unconstrained by the target identity.
  LOOM_AMDGPU_TARGET_FEATURE_ANY = 0,
  // Feature is known not to be supported by the selected processor.
  LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED = 1,
  // Feature is explicitly disabled, such as `xnack-`.
  LOOM_AMDGPU_TARGET_FEATURE_OFF = 2,
  // Feature is explicitly enabled, such as `sramecc+`.
  LOOM_AMDGPU_TARGET_FEATURE_ON = 3,
};

// Returns the canonical attribute spelling for |state|.
static inline iree_string_view_t loom_amdgpu_target_feature_state_attr_name(
    loom_amdgpu_target_feature_state_t state) {
  switch (state) {
    case LOOM_AMDGPU_TARGET_FEATURE_ANY:
      return IREE_SV("any");
    case LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
      return IREE_SV("unsupported");
    case LOOM_AMDGPU_TARGET_FEATURE_OFF:
      return IREE_SV("off");
    case LOOM_AMDGPU_TARGET_FEATURE_ON:
      return IREE_SV("on");
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU target feature state");
      return iree_string_view_empty();
  }
}

typedef struct loom_amdgpu_amdhsa_feature_states_t {
  // SRAM ECC target-ID feature state.
  loom_amdgpu_target_feature_state_t sramecc;
  // XNACK target-ID feature state.
  loom_amdgpu_target_feature_state_t xnack;
} loom_amdgpu_amdhsa_feature_states_t;

typedef enum loom_amdgpu_target_id_feature_support_bit_e {
  // No AMDHSA target-ID features are supported.
  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE = 0u,
  // Processor supports the AMDHSA `sramecc` target-ID feature.
  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC = 1u << 0,
  // Processor supports the AMDHSA `xnack` target-ID feature.
  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK = 1u << 1,
  // AMDHSA target-ID feature bits known by the AMDGPU target package.
  LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS =
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC |
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
} loom_amdgpu_target_id_feature_support_bit_t;

// Bitset of loom_amdgpu_target_id_feature_support_bit_t values.
typedef uint32_t loom_amdgpu_target_id_feature_support_flags_t;

// Returns mutable storage for one finite AMDHSA feature state, or NULL when
// |feature| is not a singular known feature.
static inline loom_amdgpu_target_feature_state_t*
loom_amdgpu_amdhsa_feature_state_select(
    loom_amdgpu_amdhsa_feature_states_t* features,
    loom_amdgpu_target_id_feature_support_bit_t feature) {
  switch (feature) {
    case LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC:
      return &features->sramecc;
    case LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK:
      return &features->xnack;
    default:
      return NULL;
  }
}

// Returns one finite AMDHSA feature state, or UNSUPPORTED when |feature| is
// not a singular known feature.
static inline loom_amdgpu_target_feature_state_t
loom_amdgpu_amdhsa_feature_state_query(
    const loom_amdgpu_amdhsa_feature_states_t* features,
    loom_amdgpu_target_id_feature_support_bit_t feature) {
  switch (feature) {
    case LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC:
      return features->sramecc;
    case LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK:
      return features->xnack;
    default:
      return LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
  }
}

typedef enum loom_amdgpu_kernel_descriptor_profile_e {
  // No kernel descriptor writer is implemented for this processor yet.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = 0,
  // GFX9/CDNA AMDHSA code-object v5 kernel descriptor packing.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9 = 1,
  // GFX11 AMDHSA code-object v5 kernel descriptor packing.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = 2,
  // GFX12 AMDHSA code-object v5 kernel descriptor packing.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12 = 3,
  // GFX125x AMDHSA code-object v5 kernel descriptor packing.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125 = 4,
} loom_amdgpu_kernel_descriptor_profile_t;

// Hardware kernel-entry behavior selected independently of processor identity.
typedef enum loom_amdgpu_kernel_entry_profile_e {
  // Scheduled instructions begin at the hardware kernel entry point.
  LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE = 0,
  // Prepend an unclaused VMEM access and establish wave replay mode before the
  // scheduled instruction body.
  LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY = 1,
} loom_amdgpu_kernel_entry_profile_t;

// Exact matrix instruction-shape inventories. Profiles are not cumulative ISA
// generations: a later processor may replace an earlier operand or fragment
// layout while retaining the same semantic operation.
typedef enum loom_amdgpu_matrix_feature_profile_e {
  // No matrix instruction feature profile is defined.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE = 0,
  // GFX908 MFMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908 = 1,
  // GFX90A MFMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A = 2,
  // GFX940 MFMA/SMFMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940 = 3,
  // GFX950 MFMA/SMFMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 = 4,
  // GFX11 WMMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11 = 5,
  // GFX12 WMMA/SWMMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12 = 6,
  // GFX1250 WMMA/SWMMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250 = 7,
  // Portable GFX12.5 WMMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC = 8,
  // Portable intersection of GFX940- and GFX950-family MFMA features.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC = 9,
  // Number of matrix feature profiles.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_COUNT = 10,
} loom_amdgpu_matrix_feature_profile_t;

// Exact processor scheduling models for matrix/VALU coexecution. These are
// independent of matrix instruction inventories: processors may expose the
// same descriptors while assigning them different release distances.
typedef enum loom_amdgpu_matrix_coexecution_profile_e {
  // No qualified matrix coexecution model is available.
  LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE = 0,
  // 4/8/16-cycle XDL WMMA/SWMMAC vector-issue release model.
  LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_4_8_16 = 1,
  // Conservative 16/32-cycle XDL WMMA/SWMMAC vector-issue release model.
  LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_16_32 = 2,
  // Number of matrix coexecution profiles.
  LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_COUNT = 3,
} loom_amdgpu_matrix_coexecution_profile_t;

typedef enum loom_amdgpu_matrix_coexecution_source_kind_e {
  // Dense wave matrix multiply-accumulate source.
  LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA = 0,
  // Sparse wave matrix multiply-accumulate source.
  LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC = 1,
  // Number of matrix coexecution source kinds.
  LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_COUNT = 2,
} loom_amdgpu_matrix_coexecution_source_kind_t;

enum {
  // Number of entries required to index every uint8_t schedule latency.
  LOOM_AMDGPU_MATRIX_COEXECUTION_LATENCY_COUNT = UINT8_MAX + 1,
};

typedef struct loom_amdgpu_matrix_coexecution_release_t {
  // Required vector issues before a dependent matrix packet.
  uint8_t matrix_issue_distance;
  // Required vector issues before a dependent ordinary vector packet.
  uint8_t vector_issue_distance;
} loom_amdgpu_matrix_coexecution_release_t;

typedef struct loom_amdgpu_matrix_coexecution_profile_info_t {
  // Generated releases indexed directly by source kind and schedule latency.
  const loom_amdgpu_matrix_coexecution_release_t (
      *releases)[LOOM_AMDGPU_MATRIX_COEXECUTION_LATENCY_COUNT];
  // Largest release distance, measured in vector issue slots, in |releases|.
  uint8_t maximum_issue_distance;
} loom_amdgpu_matrix_coexecution_profile_info_t;

typedef enum loom_amdgpu_processor_info_flag_bits_e {
  // Processor has enough target-owned facts for native HSACO emission.
  LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION = 1u << 0,
  // Clustered dispatches provide workgroup and cluster identity in the GFX1250
  // launch-state TTMP and IB_STS2 ABI.
  LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE = 1u << 1,
  // Workgroup coordinates arrive in architected TTMP launch state.
  LOOM_AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS = 1u << 2,
  // Processor info flags known by the AMDGPU target package.
  LOOM_AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS =
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION |
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE |
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS,
} loom_amdgpu_processor_info_flag_bits_t;

// Bitset of loom_amdgpu_processor_info_flag_bits_t values.
typedef uint32_t loom_amdgpu_processor_info_flags_t;

typedef enum loom_amdgpu_processor_scheduling_bit_e {
  // Nearby VALU uses of TRANS results require va_vdst depctr drains.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR = 1u << 0,
  // Nearby VALU uses of TRANS results require fixed wait states.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES = 1u << 1,
  // Nearby VALU reads of SGPRs written by VALU require fixed wait states.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES = 1u << 2,
  // Destination-selected sub-DWORD writes require fixed wait states.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES = 1u << 3,
  // Nearby VALU reads of SGPRs written by VALU require depctr drains.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR = 1u << 4,
  // GFX11+ processors support s_delay_alu for short ALU dependency delays.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU = 1u << 5,
  // Same-class VMEM instructions write vector-register results in issue order.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER = 1u << 6,
  // Processor scheduling bits known by the AMDGPU target package.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS =
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
} loom_amdgpu_processor_scheduling_bit_t;

// Bitset of loom_amdgpu_processor_scheduling_bit_t values.
typedef uint32_t loom_amdgpu_processor_scheduling_bits_t;

// Instruction restrictions that require target-owned legalization or hazard
// handling before native emission.
typedef enum loom_amdgpu_instruction_constraint_bit_e {
  // Paired DS addresses require target-specific alignment legalization.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT = 1u << 0,
  // DS ADDTID packets require an explicitly materialized address.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION = 1u
                                                                         << 1,
  // Cluster multicast operations require mask preservation around the packet.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION = 1u
                                                                           << 2,
  // Tensor multicast operations require mask preservation around the packet.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION = 1u
                                                                          << 3,
  // K64 FP8/BF8 WMMA requires a neutral regular-scale prefix.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX = 1u << 4,
  // K128 FP8/BF8 WMMA must be split into regular-scale K64 packets.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT = 1u << 5,
  // 32x16 F4 WMMA must be split into 16x16 packets.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT = 1u << 6,
  // Scaled WMMA packets require target-specific encoding legalization.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING = 1u << 7,
  // Low-precision SWMMAC packets require a target-specific lowering.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING = 1u << 8,
  // Integer matrix packets require coexecution spacing.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING = 1u
                                                                          << 9,
  // Instruction constraints known by the AMDGPU target package.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS =
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING |
      LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING,
} loom_amdgpu_instruction_constraint_bit_t;

// Bitset of loom_amdgpu_instruction_constraint_bit_t values.
typedef uint32_t loom_amdgpu_instruction_constraint_bits_t;

// Maximum nearby vector ALU packet interval covered by the
// LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR hazard window.
#define LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL 5u

// Maximum nearby TRANS packet interval covered by the
// LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR hazard window.
#define LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL 1u

typedef enum loom_amdgpu_wavefront_size_flag_bits_e {
  // Processor supports wavefront-size-32 kernels.
  LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32 = 1u << 0,
  // Processor supports wavefront-size-64 kernels.
  LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64 = 1u << 1,
  // Wavefront-size flags known by the AMDGPU target package.
  LOOM_AMDGPU_WAVEFRONT_SIZE_KNOWN_FLAGS =
      LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32 | LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64,
} loom_amdgpu_wavefront_size_flag_bits_t;

// Bitset of loom_amdgpu_wavefront_size_flag_bits_t values.
typedef uint32_t loom_amdgpu_wavefront_size_flags_t;

typedef enum loom_amdgpu_descriptor_set_info_flag_bits_e {
  // Descriptor packets have implemented native binary encoding.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING = UINT64_C(1)
                                                                    << 0,
  // Descriptor set supports native VOPD packetization for wave32 kernels.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION = UINT64_C(1) << 1,
  // VOPD floating-point min/max use the numeric-minimum/maximum mnemonics.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS =
      UINT64_C(1) << 2,
  // Descriptor set supports native packed BF16 add, multiply, and FMA.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC =
      UINT64_C(1) << 3,
  // VOP3 packets may read two distinct scalar-source registers.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES = UINT64_C(1)
                                                                 << 4,
  // Native OCP FP8 encodes require exact NaN canonicalization after conversion.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN =
      UINT64_C(1) << 5,
  // Descriptor set supports native F16/F32 scalar floating-point arithmetic.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC =
      UINT64_C(1) << 6,
  // Descriptor set supports native F16/F32 scalar conversion instructions.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION =
      UINT64_C(1) << 7,
  // Descriptor set supports native F16/F32 scalar comparison instructions.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE = UINT64_C(1)
                                                                     << 8,
  // A dual v_mov_b32 pair routes the Y source through the SRC2 cache.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE = UINT64_C(1)
                                                                  << 9,
  // Descriptor-set info flags known by the AMDGPU target package.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS =
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE |
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE,
} loom_amdgpu_descriptor_set_info_flag_bits_t;

// Bitset of loom_amdgpu_descriptor_set_info_flag_bits_t values.
typedef uint64_t loom_amdgpu_descriptor_set_info_flags_t;

typedef enum loom_amdgpu_kernel_descriptor_abi_flag_bits_e {
  // Flat scratch is architected and legacy user SGPRs are invalid.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH = UINT64_C(1)
                                                                    << 0,
  // SGPR resource counts use the GFX10+ encoding rule.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING = UINT64_C(1) << 1,
  // COMPUTE_PGM_RSRC3.ACCUM_OFFSET must be encoded.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET = UINT64_C(1) << 2,
  // COMPUTE_PGM_RSRC1 DX10 clamp and IEEE mode bits are supported.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE = UINT64_C(1)
                                                                    << 3,
  // Workitem IDs are packed into v0 instead of separate VGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID = UINT64_C(1) << 4,
  // Kernel descriptor ABI flags known by the AMDGPU target package.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS =
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
} loom_amdgpu_kernel_descriptor_abi_flag_bits_t;

// Bitset of loom_amdgpu_kernel_descriptor_abi_flag_bits_t values.
typedef uint64_t loom_amdgpu_kernel_descriptor_abi_flags_t;

typedef enum loom_amdgpu_buffer_resource_record_encoding_e {
  // No buffer-resource descriptor record encoding is selected.
  LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE = 0,
  // 48-bit base, 32-bit num_records, and legacy format fields.
  LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE48_NUM_RECORDS32_LEGACY_FORMAT =
      1,
  // 48-bit base, 32-bit num_records, and unified format fields.
  LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE48_NUM_RECORDS32_UNIFIED_FORMAT =
      2,
  // 57-bit base, 45-bit num_records, 16-bit stride, and 4-bit control.
  LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE57_NUM_RECORDS45 = 3,
  // Number of buffer-resource descriptor record encodings.
  LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_COUNT = 4,
} loom_amdgpu_buffer_resource_record_encoding_t;

typedef enum loom_amdgpu_buffer_resource_cache_swizzle_e {
  // Buffer resource descriptors do not support cache swizzle.
  LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE = 0,
  // Descriptor word 1 carries a 14-bit byte stride and one enable bit.
  LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT = 1,
} loom_amdgpu_buffer_resource_cache_swizzle_t;

typedef enum loom_amdgpu_vector_memory_cache_policy_encoding_e {
  // Vector memory descriptors do not expose cache-policy immediates.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE = 0,
  // Vector memory descriptors expose GLC/SLC/DLC cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC = 1,
  // Vector memory descriptors expose NV/SCOPE/TH cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH = 2,
  // Vector memory descriptors expose NT/SC0/SC1 cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1 = 3,
} loom_amdgpu_vector_memory_cache_policy_encoding_t;

typedef struct loom_amdgpu_descriptor_set_sopp_opcodes_t {
  // Opcode for S_NOP.
  uint16_t nop;
  // Opcode for S_DELAY_ALU, or 0 when the descriptor set has no packet.
  uint16_t delay_alu;
  // Opcode for S_ENDPGM.
  uint16_t endpgm;
  // Opcode for S_BRANCH.
  uint16_t branch;
  // Opcode for S_CBRANCH_SCC0.
  uint16_t conditional_branch_scc0;
  // Opcode for S_CBRANCH_SCC1.
  uint16_t conditional_branch_scc1;
} loom_amdgpu_descriptor_set_sopp_opcodes_t;

typedef struct loom_amdgpu_descriptor_set_buffer_resource_info_t {
  // Physical buffer-resource descriptor record encoding.
  loom_amdgpu_buffer_resource_record_encoding_t record_encoding;
  // Buffer resource descriptor cache-swizzle encoding shape.
  loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle;
} loom_amdgpu_descriptor_set_buffer_resource_info_t;

typedef struct loom_amdgpu_descriptor_set_vector_memory_info_t {
  // Vector memory packet cache-policy immediate encoding shape.
  loom_amdgpu_vector_memory_cache_policy_encoding_t cache_policy_encoding;
} loom_amdgpu_descriptor_set_vector_memory_info_t;

typedef struct loom_amdgpu_descriptor_set_info_t {
  // Target-low descriptor set key such as `amdgpu.rdna3.core`.
  iree_string_view_t key;
  // Dense generated descriptor-set ordinal within the AMDGPU target package.
  uint16_t ordinal;
  // SOPP opcodes required by structural control-flow materialization.
  loom_amdgpu_descriptor_set_sopp_opcodes_t sopp;
  // Descriptor-set capability and encoding flags.
  loom_amdgpu_descriptor_set_info_flags_t flags;
  // Buffer resource descriptor encoding facts.
  loom_amdgpu_descriptor_set_buffer_resource_info_t buffer_resource;
  // Vector memory descriptor encoding facts.
  loom_amdgpu_descriptor_set_vector_memory_info_t vector_memory;
} loom_amdgpu_descriptor_set_info_t;

typedef struct loom_amdgpu_processor_descriptor_set_info_t {
  // Target-low descriptor set key selected for this processor.
  iree_string_view_t key;
  // Dense generated descriptor-set ordinal selected for this processor.
  uint16_t ordinal;
} loom_amdgpu_processor_descriptor_set_info_t;

typedef struct loom_amdgpu_processor_elf_info_t {
  // ELF EF_AMDGPU_MACH bits for this processor, or 0 when unknown.
  uint32_t machine_flags;
  // ELF EF_AMDGPU_FEATURE_* bits implied by the selected target-id policy.
  uint32_t feature_flags;
  // Generic code-object version, or 0 for an exact processor.
  uint32_t generic_version;
} loom_amdgpu_processor_elf_info_t;

typedef struct loom_amdgpu_processor_target_id_info_t {
  // AMDHSA target-ID features supported by this processor.
  loom_amdgpu_target_id_feature_support_flags_t supported_features;
} loom_amdgpu_processor_target_id_info_t;

// One string-valued AMDHSA kernel metadata extension.
typedef struct loom_amdgpu_metadata_string_property_t {
  // External metadata map key including its leading period.
  iree_string_view_t key;
  // External metadata string value.
  iree_string_view_t value;
} loom_amdgpu_metadata_string_property_t;

// String-valued AMDHSA kernel metadata extensions.
typedef struct loom_amdgpu_metadata_string_property_set_t {
  // Metadata properties in canonical serialization order.
  const loom_amdgpu_metadata_string_property_t* entries;
  // Number of metadata properties in |entries|.
  uint16_t count;
} loom_amdgpu_metadata_string_property_set_t;

typedef struct loom_amdgpu_processor_generic_code_object_info_t {
  // Dense ordinal of the compatible generic processor, or NONE.
  uint16_t processor_ordinal;
  // First generic code-object version compatible with this exact processor.
  uint16_t introduction_version;
} loom_amdgpu_processor_generic_code_object_info_t;

typedef struct loom_amdgpu_processor_wavefront_info_t {
  // Default metadata wavefront size in lanes.
  uint32_t default_size;
  // Wavefront-size modes supported by the processor kernel descriptor ABI.
  loom_amdgpu_wavefront_size_flags_t supported_sizes;
} loom_amdgpu_processor_wavefront_info_t;

typedef struct loom_amdgpu_kernel_descriptor_vgpr_granules_t {
  // VGPR encoding granule when wavefront-size-32 mode is enabled.
  uint32_t wave32;
  // VGPR encoding granule when wavefront-size-64 mode is enabled.
  uint32_t wave64;
} loom_amdgpu_kernel_descriptor_vgpr_granules_t;

typedef struct loom_amdgpu_processor_kernel_descriptor_info_t {
  // Kernel descriptor packing profile implemented for this processor.
  loom_amdgpu_kernel_descriptor_profile_t profile;
  // Kernel descriptor ABI flags implemented for this processor.
  loom_amdgpu_kernel_descriptor_abi_flags_t flags;
  // VGPR encoding granules for wave32 and wave64 modes.
  loom_amdgpu_kernel_descriptor_vgpr_granules_t vgpr_granules;
} loom_amdgpu_processor_kernel_descriptor_info_t;

typedef struct loom_amdgpu_processor_kernel_entry_info_t {
  // Hardware kernel-entry behavior required before the scheduled body.
  loom_amdgpu_kernel_entry_profile_t profile;
} loom_amdgpu_processor_kernel_entry_info_t;

typedef struct loom_amdgpu_processor_instruction_info_t {
  // Constraints active for the processor's same-named base target.
  loom_amdgpu_instruction_constraint_bits_t base_constraints;
} loom_amdgpu_processor_instruction_info_t;

typedef struct loom_amdgpu_processor_feature_info_t {
  // Matrix instruction feature profile implemented for this processor.
  loom_amdgpu_matrix_feature_profile_t matrix;
  // Matrix/VALU coexecution scheduling model implemented for this processor.
  loom_amdgpu_matrix_coexecution_profile_t matrix_coexecution;
  // Target-local scheduling and hazard facts for this processor.
  loom_amdgpu_processor_scheduling_bits_t scheduling;
  // LDS bank-service models selected by the processor's base target.
  loom_amdgpu_lds_bank_service_model_set_ordinal_t
      lds_bank_service_model_set_ordinal;
} loom_amdgpu_processor_feature_info_t;

// Compiler-semantic properties selected by one exact or generic processor.
//
// These properties deliberately exclude the external processor name,
// exact/generic compatibility, and target-ID feature applicability. Compiler
// policy consumes this view instead of branching on processor identity.
typedef struct loom_amdgpu_processor_properties_t {
  // Dense generated occupancy-model selector.
  uint16_t occupancy_model_ordinal;
  // Processor-wide target-info capability flags.
  loom_amdgpu_processor_info_flags_t flags;
  // Target-low descriptor-set identity selected for this processor.
  loom_amdgpu_processor_descriptor_set_info_t descriptor_set;
  // AMDHSA ELF code-object identity for this processor.
  loom_amdgpu_processor_elf_info_t elf;
  // Wavefront facts selected for this processor.
  loom_amdgpu_processor_wavefront_info_t wavefront;
  // Kernel descriptor ABI facts selected for this processor.
  loom_amdgpu_processor_kernel_descriptor_info_t kernel_descriptor;
  // Hardware kernel-entry behavior selected for this processor.
  loom_amdgpu_processor_kernel_entry_info_t kernel_entry;
  // Instruction constraints active for the processor's base target.
  loom_amdgpu_processor_instruction_info_t instructions;
  // Instruction and scheduling feature profiles for this processor.
  loom_amdgpu_processor_feature_info_t features;
} loom_amdgpu_processor_properties_t;

typedef struct loom_amdgpu_processor_info_t {
  // Exact or generic processor name used in AMDHSA target IDs, such as
  // `gfx1151` or `gfx11-generic`.
  iree_string_view_t name;
  // Dense generated processor identity ordinal.
  uint16_t ordinal;
  // AMDHSA target-ID qualification facts for this processor identity.
  loom_amdgpu_processor_target_id_info_t target_id;
  // Versioned generic code-object relation for this exact processor identity.
  loom_amdgpu_processor_generic_code_object_info_t generic_code_object;
  // Compiler-semantic properties selected by this processor identity.
  loom_amdgpu_processor_properties_t properties;
} loom_amdgpu_processor_info_t;

// One exact, generic, or overlay compiler target.
typedef struct loom_amdgpu_target_info_t {
  // Canonical target selector such as `gfx1151` or `gfx1250-a0`.
  iree_string_view_t name;
  // Numeric selector value used by amdgpu.target.
  uint32_t target_kind;
  // Dense ordinal of the backend processor selected by this target.
  uint16_t processor_ordinal;
  // Target-low descriptor-set key selected by this target.
  iree_string_view_t descriptor_set_key;
  // Dense generated descriptor-set ordinal selected by this target.
  uint16_t descriptor_set_ordinal;
  // Complete instruction constraints active for this target.
  loom_amdgpu_instruction_constraint_bits_t instruction_constraints;
  // Complete LDS bank-service model set selected for this target.
  loom_amdgpu_lds_bank_service_model_set_ordinal_t
      lds_bank_service_model_set_ordinal;
  // AMDHSA kernel metadata projected by this target.
  loom_amdgpu_metadata_string_property_set_t kernel_metadata_extensions;
} loom_amdgpu_target_info_t;

// One physical device observation mapped to a canonical compiler target.
typedef struct loom_amdgpu_physical_target_info_t {
  // Dense ordinal of the HSA-reported processor.
  uint16_t processor_ordinal;
  // Physical ASIC revision reported by HSA.
  uint32_t asic_revision;
  // Canonical target selector resolved for the physical device.
  uint32_t target_kind;
} loom_amdgpu_physical_target_info_t;

// Returns true when |processor| supports every requested non-empty target-ID
// feature.
static inline bool loom_amdgpu_processor_supports_target_id_features(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_target_id_feature_support_flags_t features) {
  return processor != NULL &&
         features != LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE &&
         iree_all_bits_set(processor->target_id.supported_features, features);
}

// Returns the support flag for |wavefront_size|, or zero when unsupported.
static inline loom_amdgpu_wavefront_size_flags_t
loom_amdgpu_wavefront_size_flag(uint32_t wavefront_size) {
  switch (wavefront_size) {
    case 32:
      return LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32;
    case 64:
      return LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64;
    default:
      return 0;
  }
}

// Returns true for wavefront sizes represented by AMDGPU target lowering.
static inline bool loom_amdgpu_wavefront_size_is_valid(
    uint32_t wavefront_size) {
  return loom_amdgpu_wavefront_size_flag(wavefront_size) != 0;
}

// Returns true when |properties| can execute kernels with |wavefront_size|.
static inline bool loom_amdgpu_processor_properties_support_wavefront_size(
    const loom_amdgpu_processor_properties_t* properties,
    uint32_t wavefront_size) {
  const loom_amdgpu_wavefront_size_flags_t requested_size =
      loom_amdgpu_wavefront_size_flag(wavefront_size);
  return properties != NULL && requested_size != 0 &&
         iree_all_bits_set(properties->wavefront.supported_sizes,
                           requested_size);
}

// Returns true when |properties| advertises every requested kernel descriptor
// ABI flag.
static inline bool loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
    const loom_amdgpu_processor_properties_t* properties,
    loom_amdgpu_kernel_descriptor_abi_flags_t flags) {
  return properties != NULL &&
         iree_all_bits_set(properties->kernel_descriptor.flags, flags);
}

// Returns true when |properties| advertises every requested processor-info
// flag.
static inline bool loom_amdgpu_processor_properties_have_flags(
    const loom_amdgpu_processor_properties_t* properties,
    loom_amdgpu_processor_info_flags_t flags) {
  return properties != NULL && iree_all_bits_set(properties->flags, flags);
}

// Returns true when |properties| advertises every requested scheduling bit.
static inline bool loom_amdgpu_processor_properties_have_scheduling(
    const loom_amdgpu_processor_properties_t* properties,
    loom_amdgpu_processor_scheduling_bits_t bits) {
  return properties != NULL &&
         iree_all_bits_set(properties->features.scheduling, bits);
}

// Returns true when |descriptor_set| advertises every requested descriptor-set
// info flag.
static inline bool loom_amdgpu_descriptor_set_info_has_flags(
    const loom_amdgpu_descriptor_set_info_t* descriptor_set,
    loom_amdgpu_descriptor_set_info_flags_t flags) {
  return descriptor_set != NULL &&
         iree_all_bits_set(descriptor_set->flags, flags);
}

// Returns true when |descriptor_set| supports native VOPD packetization.
static inline bool loom_amdgpu_descriptor_set_info_supports_vopd(
    const loom_amdgpu_descriptor_set_info_t* descriptor_set) {
  return loom_amdgpu_descriptor_set_info_has_flags(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION);
}

// Returns true when |properties| supports native AMDHSA HSACO emission.
static inline bool loom_amdgpu_processor_properties_support_hsaco(
    const loom_amdgpu_processor_properties_t* properties) {
  return loom_amdgpu_processor_properties_have_flags(
      properties, LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION);
}

// Returns the number of known AMDGPU processor fact rows.
iree_host_size_t loom_amdgpu_target_info_processor_count(void);

// Returns the |index|-th known AMDGPU processor fact row, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_processor_at(
    iree_host_size_t index);

// Finds known AMDGPU processor facts by processor name, or NULL.
//
// Some known processors do not yet have target-low or HSACO support.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_find_processor(
    iree_string_view_t processor);

// Returns the number of supported AMDGPU target-low descriptor-set rows.
iree_host_size_t loom_amdgpu_target_info_descriptor_set_count(void);

// Returns the generated descriptor-set facts for |descriptor_set_ordinal|, or
// NULL when the ordinal is NONE or outside the generated table.
const loom_amdgpu_descriptor_set_info_t*
loom_amdgpu_target_info_descriptor_set_at(uint16_t descriptor_set_ordinal);

// Returns generated matrix coexecution facts for |profile|.
const loom_amdgpu_matrix_coexecution_profile_info_t*
loom_amdgpu_target_info_matrix_coexecution_profile(
    loom_amdgpu_matrix_coexecution_profile_t profile);

// Looks up known AMDGPU processor facts by processor name.
//
// Some known processors do not yet have target-low or HSACO support.
iree_status_t loom_amdgpu_target_info_lookup_processor(
    iree_string_view_t processor,
    const loom_amdgpu_processor_info_t** out_processor);

// Returns the number of known AMDGPU compiler target rows.
iree_host_size_t loom_amdgpu_target_info_target_count(void);

// Returns the |index|-th known AMDGPU compiler target row, or NULL.
const loom_amdgpu_target_info_t* loom_amdgpu_target_info_target_at(
    iree_host_size_t index);

// Finds a compiler target by canonical selector, or NULL.
const loom_amdgpu_target_info_t* loom_amdgpu_target_info_find_target(
    iree_string_view_t target);

// Finds a compiler target by amdgpu.target selector value, or NULL.
const loom_amdgpu_target_info_t* loom_amdgpu_target_info_find_target_by_kind(
    uint32_t target_kind);

// Returns the backend processor selected by |target|, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_target_processor(
    const loom_amdgpu_target_info_t* target);

// Returns true when |target| selects a generic code-object processor.
bool loom_amdgpu_target_info_is_generic(
    const loom_amdgpu_target_info_t* target);

// Looks up a compiler target by canonical selector.
iree_status_t loom_amdgpu_target_info_lookup_target(
    iree_string_view_t target, const loom_amdgpu_target_info_t** out_target);

// Returns whether |processor| requires ASIC revision to resolve its canonical
// target.
bool loom_amdgpu_target_info_requires_physical_resolution(
    const loom_amdgpu_processor_info_t* processor);

// Resolves an HSA processor and physical ASIC revision to a compiler target.
//
// Processors without physical target rows ignore |asic_revision| and resolve
// to their same-named target. Processors with physical rows reject unknown
// revisions instead of guessing target semantics.
iree_status_t loom_amdgpu_target_info_lookup_physical_target(
    const loom_amdgpu_processor_info_t* processor, uint32_t asic_revision,
    const loom_amdgpu_target_info_t** out_target);

// Returns true when |effective_processor| refines |required_processor| under
// AMDGPU's versioned code-object processor relation.
//
// Identical exact or generic processors satisfy themselves. An exact processor
// may additionally satisfy its canonical generic processor when that generic
// record's current version includes the exact processor. Generic processors do
// not satisfy exact processors or other generic families.
bool loom_amdgpu_processor_satisfies_code_object_requirement(
    const loom_amdgpu_processor_info_t* effective_processor,
    const loom_amdgpu_processor_info_t* required_processor);

// Returns true when |effective_target| satisfies |required_target|.
//
// Overlay and exact targets satisfy only identical exact requirements. Their
// backend processors may additionally satisfy a generated generic code-object
// target. A shared backend processor does not imply exact-target
// substitutability.
bool loom_amdgpu_target_satisfies_code_object_requirement(
    const loom_amdgpu_target_info_t* effective_target,
    const loom_amdgpu_target_info_t* required_target);

// Looks up a supported AMDGPU target-low descriptor set by key.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set(
    iree_string_view_t descriptor_set_key,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Looks up a supported AMDGPU target-low descriptor set by generated ordinal.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
    uint16_t descriptor_set_ordinal,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Initializes AMDHSA feature states from generated processor support facts.
//
// Supported features are unconstrained and unsupported features are explicit.
// Physical observations resolve to a canonical target before identity
// construction.
void loom_amdgpu_amdhsa_feature_states_initialize(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_amdhsa_feature_states_t* out_features);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_
