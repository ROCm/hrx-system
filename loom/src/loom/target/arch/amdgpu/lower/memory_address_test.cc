// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_address.h"

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"

namespace {

// Constructs a canonical or realization term with retained byte-range facts.
loom_low_source_memory_dynamic_term_t Term(loom_value_id_t index, int64_t low,
                                           int64_t high) {
  loom_low_source_memory_dynamic_term_t term = {};
  term.index = index;
  term.axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE;
  term.byte_stride = 1;
  term.byte_facts = loom_value_facts_make(low, high, 1);
  term.byte_shift = 0;
  return term;
}

// Models a selected global access whose canonical vector sum already fits u32.
// A realization is the exact sum of its covered byte terms; its retained facts
// may be stronger than independently adding their ranges.
loom_amdgpu_memory_access_t MixedAccess() {
  loom_amdgpu_memory_access_t access = {
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR};
  access.source.dynamic_term_count = 2;
  access.source.dynamic_terms[0] = Term(1, 0, 1024);
  access.source.dynamic_terms[1] = Term(2, 0, 252);
  access.dynamic_term_kinds[0] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  access.dynamic_term_kinds[1] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  access.source.dynamic_realization_count = 1;
  access.source.dynamic_realizations[0].term = Term(3, 0, 1276);
  access.source.dynamic_realizations[0].first_term = 0;
  access.source.dynamic_realizations[0].term_count = 2;
  return access;
}

TEST(AmdgpuMemoryAddressTest, MixedRealizationFitsVectorOperand) {
  auto access = MixedAccess();
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 1);
  EXPECT_EQ(access.dynamic_term_kinds[0],
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET);
  EXPECT_EQ(access.source.dynamic_term_count, 2);
}

TEST(AmdgpuMemoryAddressTest, RemainingWideScalarTermPreservesPartition) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 3;
  access.source.dynamic_terms[2] = Term(4, 0, INT64_MAX);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  access.scalar_base_byte_offset = UINT64_C(1) << 40;
  access.scalar_offset_placement =
      LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, RemainingNarrowScalarTermPreservesPartition) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 3;
  access.source.dynamic_terms[2] = Term(4, 0, 1024);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, VectorStaticOffsetParticipatesInBound) {
  auto access = MixedAccess();
  access.vaddr_static_byte_offset = UINT32_MAX - 1276;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 1);
  ++access.vaddr_static_byte_offset;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, OtherVectorTermsParticipateInBound) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 3;
  access.source.dynamic_terms[2] = Term(4, 0, UINT32_MAX - 1275);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, MultipleRealizationsRequireCombinedBound) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 4;
  access.source.dynamic_terms[2] = Term(4, 0, UINT32_MAX - 252);
  access.source.dynamic_terms[3] = Term(5, 0, 252);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  access.dynamic_term_kinds[3] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  access.source.dynamic_realization_count = 2;
  access.source.dynamic_realizations[1].term = Term(6, 0, UINT32_MAX);
  access.source.dynamic_realizations[1].first_term = 2;
  access.source.dynamic_realizations[1].term_count = 2;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);

  access.source.dynamic_terms[2] = Term(4, 0, 1024);
  access.source.dynamic_realizations[1].term = Term(6, 0, 1276);
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 3);
}

TEST(AmdgpuMemoryAddressTest, UnknownScalarRealizationPreservesPartition) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 4;
  access.source.dynamic_terms[2] = Term(4, 0, INT64_MAX - 252);
  access.source.dynamic_terms[3] = Term(5, 0, 252);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  access.dynamic_term_kinds[3] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  access.source.dynamic_realization_count = 2;
  access.source.dynamic_realizations[1].term = Term(6, 0, INT64_MAX);
  access.source.dynamic_realizations[1].first_term = 2;
  access.source.dynamic_realizations[1].term_count = 2;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, WholeRealizationFactsRetainCorrelations) {
  auto access = MixedAccess();
  access.source.dynamic_terms[0] = Term(1, 0, UINT32_MAX);
  access.source.dynamic_realizations[0].term = Term(3, 0, UINT32_MAX);
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 1);
}

TEST(AmdgpuMemoryAddressTest, CompleteBoundPreservesSignedTermCorrelation) {
  auto access = MixedAccess();
  access.source.dynamic_terms[0] = Term(1, -64, 64);
  access.source.dynamic_terms[1] = Term(2, 64, 252);
  access.source.dynamic_realizations[0].term = Term(3, 0, 316);
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 1);
}

TEST(AmdgpuMemoryAddressTest, UnknownWholeWidthDoesNotAuthorizeTruncation) {
  auto access = MixedAccess();
  access.source.dynamic_terms[0] = Term(1, 0, INT64_MAX - 252);
  access.source.dynamic_realizations[0].term.byte_facts =
      loom_value_facts_unknown();
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, SameBankRealizationsNeedNoPromotion) {
  auto access = MixedAccess();
  access.dynamic_term_kinds[0] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
  access.dynamic_term_kinds[0] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  access.dynamic_term_kinds[1] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
}

TEST(AmdgpuMemoryAddressTest, OtherAddressFormsKeepTheirOperandContracts) {
  for (auto form : {LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
                    LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT,
                    LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM}) {
    auto access = MixedAccess();
    access.address_form = form;
    if (form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      access.dynamic_term_kinds[0] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    } else if (form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
      access.dynamic_term_kinds[1] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
    }
    loom_amdgpu_memory_access_select_vaddr_realizations(&access);
    EXPECT_EQ(access.vaddr_realization_mask, 0);
  }
}

TEST(AmdgpuMemoryAddressTest, RetainedNoncontiguousComponentLeavesWideScalar) {
  auto access = MixedAccess();
  access.source.dynamic_term_count = 3;
  access.source.dynamic_terms[2] = access.source.dynamic_terms[1];
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  access.source.dynamic_terms[1] = Term(4, 0, INT64_MAX);
  access.dynamic_term_kinds[1] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  const auto component = Term(5, 0, 1276);
  access.source.retained_component = {&component, 0b101};
  access.scalar_offset_placement =
      LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR);
  EXPECT_EQ(access.vaddr_realization_mask, 0);
  EXPECT_EQ(access.dynamic_term_kinds[1],
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET);
  EXPECT_EQ(access.source.dynamic_terms[1].byte_facts.range_hi, INT64_MAX);
}

TEST(AmdgpuMemoryAddressTest, RetainedComponentMustFitCompleteVectorOffset) {
  auto access = MixedAccess();
  auto component = Term(5, 0, 1276);
  access.source.retained_component = {&component, 0b11};
  access.vaddr_static_byte_offset = UINT32_MAX - 1276;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR);
  ++access.vaddr_static_byte_offset;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE);

  access.vaddr_static_byte_offset = 0;
  access.source.dynamic_term_count = 3;
  access.source.dynamic_terms[2] = Term(6, 0, UINT32_MAX - 1275);
  access.dynamic_term_kinds[2] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE);
}

TEST(AmdgpuMemoryAddressTest, RetainedComponentKeepsCorrelatedBound) {
  auto access = MixedAccess();
  access.source.dynamic_terms[0] = Term(1, -64, 64);
  access.source.dynamic_terms[1] = Term(2, 64, 252);
  auto component = Term(5, 0, 316);
  access.source.retained_component = {&component, 0b11};
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR);
  component.byte_facts = loom_value_facts_unknown();
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE);
}

TEST(AmdgpuMemoryAddressTest, RetainedScalarComponentDoesNotNarrow) {
  auto access = MixedAccess();
  access.dynamic_term_kinds[1] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  const auto component = Term(5, 0, INT64_MAX);
  access.source.retained_component = {&component, 0b11};
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET);
}

TEST(AmdgpuMemoryAddressTest,
     RetainedComponentPreservesImplicitLaneAddressing) {
  auto access = MixedAccess();
  access.address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID;
  access.dynamic_term_kinds[0] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  const auto component = Term(5, 0, 1276);
  access.source.retained_component = {&component, 0b11};
  loom_amdgpu_memory_access_select_vaddr_realizations(&access);
  EXPECT_EQ(access.retained_component_kind,
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE);
}

}  // namespace
