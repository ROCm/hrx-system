// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packets.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct ExpectedImmediate {
  // Expected selected immediate name.
  iree_string_view_t name;
  // Expected selected immediate value.
  uint16_t value;
};

struct ExpectedWaitSelection {
  // Descriptor set key to test.
  iree_string_view_t descriptor_set_key;
  // Logical wait counter mask to materialize.
  uint32_t counter_mask;
  // Expected descriptor ref selected for the logical counter.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Expected immediate row populated on the selected descriptor.
  ExpectedImmediate immediate;
};

class AmdgpuWaitPacketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  const loom_low_descriptor_t* DescriptorForRef(
      const loom_low_descriptor_set_t* descriptor_set,
      loom_amdgpu_descriptor_ref_t descriptor_ref) {
    const uint32_t descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
    return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                                 descriptor_ordinal);
  }

  const loom_amdgpu_wait_packet_immediate_t* FindImmediate(
      const loom_amdgpu_wait_packet_selection_t& selection,
      iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
      if (iree_string_view_equal(selection.immediates[i].name, name)) {
        return &selection.immediates[i];
      }
    }
    return nullptr;
  }

  void ExpectSelection(const ExpectedWaitSelection& expected) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                            expected.descriptor_set_key);
    if (descriptor_set == nullptr) {
      GTEST_SKIP() << "AMDGPU descriptor set is not linked: "
                   << ToString(expected.descriptor_set_key);
    }
    loom_amdgpu_wait_packet_selection_t selection = {};
    ASSERT_TRUE(loom_amdgpu_wait_packet_try_select_counter_mask(
        descriptor_set, expected.counter_mask, /*target_count=*/0, &selection));

    EXPECT_EQ(selection.counter_mask, expected.counter_mask);
    const loom_low_descriptor_t* expected_descriptor =
        DescriptorForRef(descriptor_set, expected.descriptor_ref);
    ASSERT_NE(expected_descriptor, nullptr);
    EXPECT_EQ(selection.descriptor, expected_descriptor);
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        FindImmediate(selection, expected.immediate.name);
    ASSERT_NE(immediate, nullptr)
        << "missing immediate " << ToString(expected.immediate.name);
    EXPECT_EQ(immediate->value, expected.immediate.value);
  }

  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(AmdgpuWaitPacketTest, SelectsLegacyCombinedAndVscntWaits) {
  const ExpectedWaitSelection cases[] = {
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          /*.immediate=*/{IREE_SV("vmcnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
          /*.immediate=*/{IREE_SV("vscnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          /*.immediate=*/{IREE_SV("lgkmcnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.cdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          /*.immediate=*/{IREE_SV("vmcnt"), 0},
      },
  };
  for (const ExpectedWaitSelection& expected : cases) {
    SCOPED_TRACE(ToString(expected.descriptor_set_key));
    ExpectSelection(expected);
  }
}

TEST_F(AmdgpuWaitPacketTest, SelectsArchitectureSpecificCombinedNoWaitValues) {
  const ExpectedWaitSelection cases[] = {
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          /*.immediate=*/{IREE_SV("lgkmcnt"), 63},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.cdna3.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          /*.immediate=*/{IREE_SV("lgkmcnt"), 15},
      },
  };
  for (const ExpectedWaitSelection& expected : cases) {
    SCOPED_TRACE(ToString(expected.descriptor_set_key));
    ExpectSelection(expected);
  }
}

TEST_F(AmdgpuWaitPacketTest, ClampsTargetCountToNoWaitEncoding) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                          IREE_SV("amdgpu.rdna3.core"));
  ASSERT_NE(descriptor_set, nullptr);

  loom_amdgpu_wait_packet_selection_t selection = {};
  ASSERT_TRUE(loom_amdgpu_wait_packet_try_select_counter_mask(
      descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
      /*target_count=*/UINT16_MAX, &selection));

  const loom_amdgpu_wait_packet_immediate_t* vmcnt =
      FindImmediate(selection, IREE_SV("vmcnt"));
  ASSERT_NE(vmcnt, nullptr);
  EXPECT_EQ(vmcnt->value, 63);
}

TEST_F(AmdgpuWaitPacketTest, SelectsRdna4SplitWaits) {
  const ExpectedWaitSelection cases[] = {
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna4.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT,
          /*.immediate=*/{IREE_SV("loadcnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna4.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT,
          /*.immediate=*/{IREE_SV("storecnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna4.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_DSCNT,
          /*.immediate=*/{IREE_SV("dscnt"), 0},
      },
      {
          /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna4.core"),
          /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
          /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_KMCNT,
          /*.immediate=*/{IREE_SV("kmcnt"), 0},
      },
  };
  for (const ExpectedWaitSelection& expected : cases) {
    SCOPED_TRACE(ToString(expected.descriptor_set_key));
    ExpectSelection(expected);
  }
}

TEST_F(AmdgpuWaitPacketTest, SelectsGfx125xTranslationWait) {
  ExpectSelection({
      /*.descriptor_set_key=*/IREE_SV("amdgpu.rdna4.gfx125x.core"),
      /*.counter_mask=*/LOOM_AMDGPU_WAIT_COUNTER_MASK_X,
      /*.descriptor_ref=*/LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_XCNT,
      /*.immediate=*/{IREE_SV("xcnt"), 0},
  });

  const loom_low_descriptor_set_t* rdna4_descriptor_set =
      loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                          IREE_SV("amdgpu.rdna4.core"));
  ASSERT_NE(rdna4_descriptor_set, nullptr);
  loom_amdgpu_wait_packet_selection_t selection = {};
  EXPECT_FALSE(loom_amdgpu_wait_packet_try_select_counter_mask(
      rdna4_descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_X,
      /*target_count=*/0, &selection));
}

}  // namespace
}  // namespace loom
