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

// Sentinel for processors or descriptor sets without target-low support.
#define LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE UINT16_MAX

// Default raw buffer-resource descriptor control word for global HAL bindings.
//
// This is the final descriptor word consumed by MUBUF/MTBUF packets. It matches
// the word emitted by LLVM/IREE for amdgcn-amd-amdhsa raw buffers with 32-bit
// element format, resource-level OOB behavior, and the standard memory
// properties used for HAL binding resources.
#define LOOM_AMDGPU_HAL_BUFFER_RESOURCE_FLAGS UINT32_C(0x31027000)

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

// Target feature selector parsed from an AMDHSA target-id suffix.
typedef uint8_t loom_amdgpu_target_feature_selection_t;

enum loom_amdgpu_target_feature_selection_e {
  // Feature inherits the processor's default code-object policy.
  LOOM_AMDGPU_TARGET_FEATURE_DEFAULT = 0,
  // Feature is explicitly disabled, such as `xnack-`.
  LOOM_AMDGPU_TARGET_FEATURE_OFF = 1,
  // Feature is explicitly enabled, such as `sramecc+`.
  LOOM_AMDGPU_TARGET_FEATURE_ON = 2,
};

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
} loom_amdgpu_matrix_feature_profile_t;

typedef enum loom_amdgpu_processor_info_flag_bits_e {
  // Processor has enough target-owned facts for native HSACO emission.
  LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION = 1u << 0,
  // Processor info flags known by the AMDGPU target package.
  LOOM_AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS =
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
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
  // Sub-DWORD SDWA dst_sel writes require fixed wait states.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES = 1u << 3,
  // Nearby VALU reads of SGPRs written by VALU require depctr drains.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR = 1u << 4,
  // GFX11+ processors support s_delay_alu for short ALU dependency delays.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU = 1u << 5,
  // Processor scheduling bits known by the AMDGPU target package.
  LOOM_AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS =
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR |
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
} loom_amdgpu_processor_scheduling_bit_t;

// Bitset of loom_amdgpu_processor_scheduling_bit_t values.
typedef uint32_t loom_amdgpu_processor_scheduling_bits_t;

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
  // Descriptor-set info flags known by the AMDGPU target package.
  LOOM_AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS =
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
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
} loom_amdgpu_processor_elf_info_t;

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

typedef struct loom_amdgpu_processor_feature_info_t {
  // Matrix instruction feature profile implemented for this processor.
  loom_amdgpu_matrix_feature_profile_t matrix;
  // Target-local scheduling and hazard facts for this processor.
  loom_amdgpu_processor_scheduling_bits_t scheduling;
} loom_amdgpu_processor_feature_info_t;

typedef struct loom_amdgpu_processor_info_t {
  // Processor name used in AMDHSA target IDs, such as `gfx1100`.
  iree_string_view_t name;
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
  // Instruction and scheduling feature profiles for this processor.
  loom_amdgpu_processor_feature_info_t features;
} loom_amdgpu_processor_info_t;

typedef struct loom_amdgpu_amdhsa_target_id_t {
  // Processor row selected by the target-id processor component.
  const loom_amdgpu_processor_info_t* processor;
  // Target-id feature suffix after ':', or empty when no suffix is present.
  iree_string_view_t feature_suffix;
  // SRAM ECC feature selection parsed from the target-id suffix.
  loom_amdgpu_target_feature_selection_t sramecc;
  // XNACK feature selection parsed from the target-id suffix.
  loom_amdgpu_target_feature_selection_t xnack;
} loom_amdgpu_amdhsa_target_id_t;

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

// Returns true when |processor| can execute kernels with |wavefront_size|.
static inline bool loom_amdgpu_processor_supports_wavefront_size(
    const loom_amdgpu_processor_info_t* processor, uint32_t wavefront_size) {
  const loom_amdgpu_wavefront_size_flags_t requested_size =
      loom_amdgpu_wavefront_size_flag(wavefront_size);
  return processor != NULL && requested_size != 0 &&
         iree_all_bits_set(processor->wavefront.supported_sizes,
                           requested_size);
}

// Returns true when |processor| advertises every requested kernel descriptor
// ABI flag.
static inline bool loom_amdgpu_processor_kernel_descriptor_has_flags(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_kernel_descriptor_abi_flags_t flags) {
  return processor != NULL &&
         iree_all_bits_set(processor->kernel_descriptor.flags, flags);
}

// Returns true when |processor| advertises every requested processor-info flag.
static inline bool loom_amdgpu_processor_info_has_flags(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_processor_info_flags_t flags) {
  return processor != NULL && iree_all_bits_set(processor->flags, flags);
}

// Returns true when |descriptor_set| advertises every requested descriptor-set
// info flag.
static inline bool loom_amdgpu_descriptor_set_info_has_flags(
    const loom_amdgpu_descriptor_set_info_t* descriptor_set,
    loom_amdgpu_descriptor_set_info_flags_t flags) {
  return descriptor_set != NULL &&
         iree_all_bits_set(descriptor_set->flags, flags);
}

// Returns true when |processor| supports native AMDHSA HSACO emission.
static inline bool loom_amdgpu_processor_supports_hsaco(
    const loom_amdgpu_processor_info_t* processor) {
  return loom_amdgpu_processor_info_has_flags(
      processor, LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION);
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

// Looks up known AMDGPU processor facts by processor name.
//
// Some known processors do not yet have target-low or HSACO support.
iree_status_t loom_amdgpu_target_info_lookup_processor(
    iree_string_view_t processor,
    const loom_amdgpu_processor_info_t** out_processor);

// Looks up a supported AMDGPU target-low descriptor set by key.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set(
    iree_string_view_t descriptor_set_key,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Looks up a supported AMDGPU target-low descriptor set by generated ordinal.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
    uint16_t descriptor_set_ordinal,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Parses an AMDHSA target ID such as `amdgcn-amd-amdhsa--gfx1100`.
iree_status_t loom_amdgpu_target_info_parse_amdhsa_target_id(
    iree_string_view_t target_id,
    loom_amdgpu_amdhsa_target_id_t* out_target_id);

// Resolves the AMDGPU ELF e_flags implied by |target_id|.
iree_status_t loom_amdgpu_target_info_amdhsa_target_id_elf_flags(
    const loom_amdgpu_amdhsa_target_id_t* target_id, uint32_t* out_elf_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_
