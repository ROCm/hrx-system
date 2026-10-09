// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/cts/pipeline_completion.h"

#include "experimental/xdna/cts/execution_fixture.h"

namespace iree::experimental::xdna::testing {
namespace {

class XdnaPipelineCompletionTest : public XdnaExecutionFixture {
 protected:
  void LoadPipeline(iree_string_view_t name) {
    const auto* images = iree_xdna_pipeline_completion_create();
    const iree_file_toc_t* image = nullptr;
    if (target_.identity.device_profile_id == UINT64_C(0x5354524958000001)) {
      image = &images[0];
    } else if (target_.identity.device_profile_id ==
               UINT64_C(0x535848414C4F0001)) {
      image = &images[1];
    } else {
      FAIL() << "no pipeline fixture for profile "
             << target_.identity.device_profile_id;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(image->data);
    auto sequence = iree::hal::amd::xdna::testing::MakeOwnedByteSequence(
        std::vector<uint8_t>(bytes, bytes + image->size));
    iree_hal_amd_xdna_image_destroy(executable_);
    executable_ = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        sequence.get(), &target_, iree_allocator_system(), &executable_));
    IREE_ASSERT_OK(
        iree_hal_amd_xdna_image_find_entry(executable_, name, &entry_ordinal_));
  }

  void BindPipeline(
      iree::span<const iree_hal_amd_xdna_executable_binding_t> bindings) {
    // Rebind only after the previous invocation has retired. The same context
    // and command storage must use the newly supplied buffer DMA addresses.
    iree_hal_amd_xdna_executable_storage_t storage = {};
    storage.memory = first_.instructions.memory;
    storage.mapping =
        iree_make_byte_span(first_.instructions.pointer, first_.byte_length);
    ASSERT_EQ(api_->memory_query_address(storage.memory, 0,
                                         AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                         &storage.device_address),
              AMDF_STATUS_OK);
    IREE_ASSERT_OK(iree_xdna_executable_storage_bind(
        executable_, entry_ordinal_, 1, &storage, bindings.size(),
        bindings.data()));
    first_.original_instructions.assign(
        first_.instructions.pointer,
        first_.instructions.pointer + first_.byte_length);
    ASSERT_EQ(api_->host_mapping_cache_control(first_.instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, first_.byte_length),
              AMDF_STATUS_OK);
  }

  void CheckPipeline(iree_string_view_t name,
                     iree::span<const uint32_t> results) {
    ASSERT_NO_FATAL_FAILURE(LoadPipeline(name));
    ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
    ASSERT_NO_FATAL_FAILURE(
        PrepareExecution({resolved_bindings_.data(), 1}, &first_));
    for (uint32_t generation = 0; generation < 3; ++generation) {
      SCOPED_TRACE(generation);
      const size_t output_binding = generation % resolved_bindings_.size();
      std::array<BindingValues, 3> expected;
      for (size_t binding = 0; binding < expected.size(); ++binding) {
        for (size_t i = 0; i < kElementCount; ++i) {
          expected[binding][i] =
              kValues[(i + binding + generation) % kElementCount];
        }
      }
      for (size_t i = 0; i < results.size(); ++i) {
        expected[output_binding][i] = ~results[i];
      }
      for (size_t binding = 0; binding < expected.size(); ++binding) {
        ASSERT_NO_FATAL_FAILURE(WriteBinding(binding, expected[binding]));
      }
      ASSERT_NO_FATAL_FAILURE(
          BindPipeline({&resolved_bindings_[output_binding], 1}));
      std::copy(results.begin(), results.end(),
                expected[output_binding].begin());
      ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
      ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
    }
  }
};

TEST_F(XdnaPipelineCompletionTest, StreamsStridedRecordsThroughPartialRing) {
  ASSERT_NO_FATAL_FAILURE(LoadPipeline(IREE_SV("autonomous_strided_ingress")));
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  ASSERT_NO_FATAL_FAILURE(
      PrepareExecution({resolved_bindings_.data(), 2}, &first_));
  for (size_t generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    const size_t input_binding = generation;
    const size_t output_binding = (generation + 1) % resolved_bindings_.size();
    std::array<BindingValues, 3> expected;
    for (size_t binding = 0; binding < expected.size(); ++binding) {
      for (size_t i = 0; i < kElementCount; ++i) {
        expected[binding][i] =
            kValues[(i + binding + generation) % kElementCount];
      }
      ASSERT_NO_FATAL_FAILURE(WriteBinding(binding, expected[binding]));
    }
    const std::array<iree_hal_amd_xdna_executable_binding_t, 2> bindings = {
        resolved_bindings_[input_binding], resolved_bindings_[output_binding]};
    ASSERT_NO_FATAL_FAILURE(BindPipeline(bindings));
    for (size_t record = 0; record < 5; ++record) {
      for (size_t field = 0; field < 2; ++field) {
        const size_t index = record * 3 + field * 2;
        expected[output_binding][index] = expected[input_binding][index] * 2;
      }
    }
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

TEST_F(XdnaPipelineCompletionTest, PublishesReservationOrder) {
  constexpr std::array<uint32_t, 2> results = {11, 22};
  CheckPipeline(IREE_SV("reordered_publication"), results);
}

TEST_F(XdnaPipelineCompletionTest, UsesFirstSlotOfMultiSlotChannel) {
  constexpr std::array<uint32_t, 1> results = {13};
  CheckPipeline(IREE_SV("single_admission"), results);
}

TEST_F(XdnaPipelineCompletionTest, RetainsEarlierReadThroughDma) {
  constexpr std::array<uint32_t, 1> results = {0};
  CheckPipeline(IREE_SV("reordered_retirement"), results);
}

TEST_F(XdnaPipelineCompletionTest, RotatesOwnedReadsAcrossLoopIterations) {
  constexpr std::array<uint32_t, 3> results = {0, 2, 4};
  CheckPipeline(IREE_SV("rotating_retirement"), results);
}

TEST_F(XdnaPipelineCompletionTest, ProjectsFixedRecordAddresses) {
  constexpr std::array<uint32_t, 16> results = {1,  0,  1,  11, 10, 11, 21, 20,
                                                21, 31, 30, 31, 41, 40, 41, 41};
  CheckPipeline(IREE_SV("projected_single_slot"), results);
}

TEST_F(XdnaPipelineCompletionTest, ProjectsRotatingRecordAddresses) {
  constexpr std::array<uint32_t, 16> results = {1,  0,  1,  11, 10, 11, 21, 20,
                                                21, 31, 30, 31, 41, 40, 41, 41};
  CheckPipeline(IREE_SV("projected_ring"), results);
}

}  // namespace
}  // namespace iree::experimental::xdna::testing
