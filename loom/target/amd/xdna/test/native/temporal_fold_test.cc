// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/driver.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, executable_file_npu4, "",
          "Compiler-built Strix fold artifact.");
IREE_FLAG(string, executable_file_npu5, "",
          "Compiler-built Halo fold artifact.");

namespace iree::hal::cts {
namespace {

iree_status_t CreateDevice(const iree_hal_device_create_params_t* params,
                           iree_hal_driver_t** out_driver,
                           iree_hal_device_t** out_device) {
  iree_hal_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_create(
      AMDF_NATIVE_LIFETIME_PROCESS, iree_allocator_system(), &driver));
  iree_status_t status = iree_hal_driver_create_device_by_uri(
      driver, IREE_SV("xdna://0?columns=1"), params, iree_allocator_system(),
      out_device);
  if (iree_status_is_ok(status)) {
    *out_driver = driver;
  } else {
    iree_hal_driver_release(driver);
  }
  return status;
}

// Each shape names an independently compiled worker.fold export.
struct FoldShape {
  // Array-program export in the compiler-built executable.
  const char* entry_point;
  // Number of words in each folded vector output.
  size_t output_width;
  // Number of payload words in each input contribution.
  size_t input_width;
  // Ordered contributions consumed before publishing one output record.
  size_t contribution_count;
  // Whether a separate scalar output is folded alongside the vector.
  bool scalar_output;
};

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint32_t FirstContributionBits(size_t fold, size_t element) {
  return element % 8 == 0
             ? 0x80000000u
             : static_cast<uint32_t>(fold * 1024 + element + 17) * 0x9E3779B9u;
}

std::array<float, 3> Contributions(size_t fold, size_t element) {
  switch (element % 8) {
    case 0:
      return {0.0f, 0.0f, 0.0f};
    case 1:
      return {16777216.0f, 1.0f, -16777216.0f};
    case 2:
      return {16777216.0f, -16777216.0f, 1.0f};
    case 3:
      return {-16777216.0f, -1.0f, 16777216.0f};
    case 4:
      return {static_cast<float>(fold * 1024 + element), 0.25f, -0.125f};
    case 5:
      return {0.5f * static_cast<float>(element + 1),
              -static_cast<float>(fold + 1), 0.0625f};
    case 6:
      return {0x1p-125f, 0x1p-125f, 0x1p-125f};
    default:
      return {-0.0f, 0.0f, -0.0f};
  }
}

uint32_t OrderedSumBits(size_t fold, size_t element) {
  const auto values = Contributions(fold, element);
  float sum = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    // Materialization fixes the f32 rounding boundary after each ordered add.
    volatile float rounded = sum + values[i];
    sum = rounded;
  }
  return FloatBits(sum);
}

class TemporalFoldTest : public CtsTestBase<> {
 protected:
  void RunFold(const FoldShape& shape) {
    SCOPED_TRACE(shape.entry_point);
    auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    ASSERT_NE(queue, nullptr);
    const auto target =
        SelectExecutableTarget(IREE_SV("xdna"), iree_string_view_empty());
    ASSERT_EQ(target.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    const auto key = target.target->target_key;
    SCOPED_TRACE(std::string(key.data, key.size));
    const char* image_path = nullptr;
    if (iree_string_view_equal(key, IREE_SV("amd.xdna.strix_halo.17f0_11"))) {
      image_path = FLAG_executable_file_npu5;
    } else if (iree_string_view_equal(key, IREE_SV("amd.xdna.strix.17f0_10")) ||
               iree_string_view_equal(key,
                                      IREE_SV("amd.xdna.krackan.17f0_20"))) {
      image_path = FLAG_executable_file_npu4;
    }
    ASSERT_NE(image_path, nullptr)
        << "No fold artifact for this device profile";
    Ref<iree_hal_executable_t> executable;
    {
      std::ifstream file(image_path, std::ios::binary);
      ASSERT_TRUE(file.good()) << image_path;
      const std::vector<char> image((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
      ASSERT_FALSE(image.empty());
      IREE_ASSERT_OK(
          LoadExecutable(iree_hal_queue_family(queue), target.target,
                         IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
                         iree_make_const_byte_span(image.data(), image.size()),
                         executable.out()));
    }
    iree_hal_executable_function_t function;
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, iree_make_cstring_view(shape.entry_point), &function));

    constexpr size_t kGuardWords = 16;
    constexpr size_t kHeaderWords = 16;
    constexpr size_t kOutputs = 8;
    const size_t record_words = kHeaderWords + shape.input_width;
    std::vector<uint32_t> input(
        2 * kGuardWords + kOutputs * shape.contribution_count * record_words);
    std::vector<uint32_t> expected(2 * kGuardWords +
                                   kOutputs * shape.output_width);
    std::vector<uint32_t> initial_output(expected.size());
    std::vector<uint32_t> scalar_expected(
        shape.scalar_output ? 2 * kGuardWords + kOutputs : 0);
    std::vector<uint32_t> scalar_initial(scalar_expected.size());
    Ref<iree_hal_buffer_t> input_buffer, output_buffer, scalar_buffer;
    IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(
        input.size() * sizeof(uint32_t), input_buffer.out()));
    IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(
        expected.size() * sizeof(uint32_t), output_buffer.out()));
    std::vector<iree_hal_buffer_ref_t> binding_refs = {
        iree_hal_make_buffer_ref(input_buffer, 0,
                                 input.size() * sizeof(uint32_t)),
    };
    if (shape.scalar_output) {
      IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(
          scalar_expected.size() * sizeof(uint32_t), scalar_buffer.out()));
      binding_refs.push_back(iree_hal_make_buffer_ref(
          scalar_buffer, 0, scalar_expected.size() * sizeof(uint32_t)));
    }
    binding_refs.push_back(iree_hal_make_buffer_ref(
        output_buffer, 0, expected.size() * sizeof(uint32_t)));
    const iree_hal_buffer_ref_list_t bindings = {binding_refs.size(),
                                                 binding_refs.data()};
    for (uint32_t trial = 0; trial < 3; ++trial) {
      SCOPED_TRACE(trial);
      const uint32_t guard = 0xA5A5A5A5u ^ trial;
      std::fill(input.begin(), input.end(), guard);
      std::fill(expected.begin(), expected.end(), guard);
      std::fill(initial_output.begin(), initial_output.end(), guard);
      std::fill(scalar_expected.begin(), scalar_expected.end(), guard);
      std::fill(scalar_initial.begin(), scalar_initial.end(), guard);
      for (size_t fold = 0; fold < kOutputs; ++fold) {
        const size_t epoch = trial * kOutputs + fold;
        for (size_t record = 0; record < shape.contribution_count; ++record) {
          const size_t base =
              kGuardWords +
              (fold * shape.contribution_count + record) * record_words;
          std::fill_n(input.begin() + base, kHeaderWords, 0);
          input[base] = (epoch + record) & 1;
          input[base + 1] = shape.output_width * sizeof(uint32_t);
          for (size_t element = 0; element < shape.input_width; ++element) {
            input[base + kHeaderWords + element] =
                shape.contribution_count == 1
                    ? FirstContributionBits(epoch, element)
                    : FloatBits(Contributions(epoch, element)[record]);
          }
        }
        for (size_t element = 0; element < shape.output_width; ++element) {
          // Large outputs expand compact input into four differently permuted
          // blocks, so repeating or misaddressing a private span cannot pass.
          const size_t source =
              (element % shape.input_width) ^ (element / shape.input_width);
          expected[kGuardWords + fold * shape.output_width + element] =
              shape.contribution_count == 1
                  ? FirstContributionBits(epoch, source)
                  : OrderedSumBits(epoch, source);
        }
        if (shape.scalar_output) {
          scalar_expected[kGuardWords + fold] = OrderedSumBits(epoch, 4);
        }
      }
      IREE_ASSERT_OK(UploadBufferData(input.data(), input_buffer, 0,
                                      input.size() * sizeof(uint32_t)));
      IREE_ASSERT_OK(
          UploadBufferData(initial_output.data(), output_buffer, 0,
                           initial_output.size() * sizeof(uint32_t)));
      if (shape.scalar_output) {
        IREE_ASSERT_OK(
            UploadBufferData(scalar_initial.data(), scalar_buffer, 0,
                             scalar_initial.size() * sizeof(uint32_t)));
      }
      SemaphoreList completed(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, iree_hal_semaphore_list_empty(), completed, executable,
          function, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, nullptr,
          IREE_HAL_DISPATCH_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          completed, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_THAT(ReadBufferData<uint32_t>(input_buffer),
                  ::testing::ContainerEq(input));
      EXPECT_THAT(ReadBufferData<uint32_t>(output_buffer),
                  ::testing::ContainerEq(expected));
      if (shape.scalar_output) {
        EXPECT_THAT(ReadBufferData<uint32_t>(scalar_buffer),
                    ::testing::ContainerEq(scalar_expected));
      }
    }
  }
};

TEST_P(TemporalFoldTest, FirstRecordPreservesBitsAndRingWrap) {
  RunFold({"temporal_fold_first_copy", 1024, 1024, 1, false});
}

TEST_P(TemporalFoldTest, OrderedWideContributionsAcrossCfgExits) {
  RunFold({"temporal_fold_packets", 1024, 1024, 3, false});
}

TEST_P(TemporalFoldTest, OrderedPartialVectorTail) {
  RunFold({"temporal_fold_tail", 80, 80, 3, false});
}

TEST_P(TemporalFoldTest, LargeFirstRecordPreservesEveryPrivateSpan) {
  RunFold({"temporal_fold_large_first_copy", 4096, 1024, 1, false});
}

TEST_P(TemporalFoldTest, OrderedLargeContributionsAcrossSpanLoops) {
  RunFold({"temporal_fold_large_packets", 4096, 1024, 3, false});
}

TEST_P(TemporalFoldTest, MixedScalarAndVectorWithSourcePrivateStorage) {
  RunFold({"temporal_fold_mixed", 144, 144, 3, true});
}

CTS_REGISTER_TEST_SUITE(TemporalFoldTest);
static bool registered =
    (CtsRegistry::RegisterBackend({"xdna", {"xdna", CreateDevice}, {}}), true);

}  // namespace
}  // namespace iree::hal::cts
