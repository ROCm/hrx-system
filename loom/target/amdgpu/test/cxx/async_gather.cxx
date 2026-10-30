// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

// Each workitem transfers one global element into its LDS lane and consumes
// the mirrored lane. The serial kernel proves slot reuse after a complete
// wait; the two-group kernel proves a partial wait with one younger transfer.
constexpr unsigned kSubgroupSize = 64u;
constexpr unsigned kRecordCount = 257u;
constexpr unsigned kInputElementCount = kRecordCount * kSubgroupSize;

void async_gather_reference(unsigned record_count,
                            loom::type::buffer<int> input,
                            loom::type::buffer<int> output) {
  loom::assume(record_count <= kRecordCount);
  auto source = loom::buffer::view<kRecordCount, kSubgroupSize>(
      input, {}, loom::encoding::layout::dense<2>());
  auto destination = loom::buffer::view<kSubgroupSize>(
      output, {}, loom::encoding::layout::dense<1>());
  for (unsigned lane = 0; lane < kSubgroupSize; ++lane) {
    int total = 0;
    unsigned neighbor = kSubgroupSize - 1u - lane;
    for (unsigned record = 0; record < record_count; ++record) {
      total += loom::view::load(source, record, neighbor);
    }
    loom::view::store(total, destination, lane);
  }
}

[[loom::kernel, loom::workgroup_size(kSubgroupSize, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void async_gather_serial(
    unsigned record_count,
    [[loom::noalias, loom::assume_aligned(64)]] const int* input,
    [[loom::noalias, loom::assume_aligned(64)]] int* output) {
  loom::assume(record_count <= kRecordCount);
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned neighbor = kSubgroupSize - 1u - lane;
  auto source = loom::buffer::view<kRecordCount, kSubgroupSize>(
      input, {}, loom::encoding::layout::dense<2>());
  auto* scratch = loom::buffer::alloca<int, loom::memory_space::workgroup, 16>(
      kSubgroupSize);
  auto stage = loom::buffer::view<kSubgroupSize, 1, 1>(
      scratch, {}, loom::encoding::layout::dense<3>());

  int total = 0;
  for (unsigned record = 0; record < record_count; ++record) {
    auto record_lane = loom::view::subview<1, 1>(source, {record, lane}, {});
    auto transfer = loom::kernel::async::gather<loom::cache::scope::device,
                                                loom::cache::temporal::regular>(
        record_lane, stage);
    auto transfers = loom::kernel::async::group(transfer);
    loom::kernel::async::wait<0>(transfers);
    loom::kernel::barrier<loom::memory_space::workgroup,
                          loom::atomic::scope::workgroup,
                          loom::atomic::ordering::acq_rel>();
    total += loom::view::load(stage, neighbor, 0, 0);
    loom::kernel::barrier<loom::memory_space::workgroup,
                          loom::atomic::scope::workgroup,
                          loom::atomic::ordering::acq_rel>();
  }
  output[lane] = total;
}

[[loom::kernel, loom::workgroup_size(kSubgroupSize, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void async_gather_two_groups(
    [[loom::noalias, loom::assume_aligned(64)]] const int* input,
    [[loom::noalias, loom::assume_aligned(64)]] int* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned neighbor = kSubgroupSize - 1u - lane;
  auto source = loom::buffer::view<kRecordCount, kSubgroupSize>(
      input, {}, loom::encoding::layout::dense<2>());
  auto* scratch = loom::buffer::alloca<int, loom::memory_space::workgroup, 16>(
      2u * kSubgroupSize);
  auto first_stage = loom::buffer::view<kSubgroupSize, 1, 1>(
      scratch, {}, loom::encoding::layout::dense<3>());
  auto second_stage = loom::buffer::view<kSubgroupSize, 1, 1>(
      scratch + kSubgroupSize, {}, loom::encoding::layout::dense<3>());

  loom::view::store(-1, first_stage, lane, 0, 0);
  loom::view::store(-2, second_stage, lane, 0, 0);
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();

  auto first_source = loom::view::subview<1, 1>(source, {0, lane}, {});
  auto first_transfer =
      loom::kernel::async::gather<loom::cache::scope::device,
                                  loom::cache::temporal::regular>(first_source,
                                                                  first_stage);
  auto first_group = loom::kernel::async::group(first_transfer);

  auto second_source = loom::view::subview<1, 1>(source, {1, lane}, {});
  auto second_transfer =
      loom::kernel::async::gather<loom::cache::scope::device,
                                  loom::cache::temporal::regular>(second_source,
                                                                  second_stage);
  auto second_group = loom::kernel::async::group(second_transfer);

  loom::kernel::async::wait<1>(first_group);
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  int first = loom::view::load(first_stage, neighbor, 0, 0);

  loom::kernel::async::wait<0>(second_group);
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  int second = loom::view::load(second_stage, neighbor, 0, 0);
  output[lane] = first + second;
}

unsigned async_gather_record_count(loom::check::ordinal trial) {
  if (trial == 0) {
    return 0u;
  }
  if (trial == 1) {
    return 1u;
  }
  if (trial == 2) {
    return 2u;
  }
  if (trial == 3) {
    return 3u;
  }
  if (trial == 4) {
    return 8u;
  }
  if (trial == 5) {
    return 127u;
  }
  if (trial == 6) {
    return 128u;
  }
  return kRecordCount;
}

LOOM_CHECK_SCENARIO(async_gather_finite_counts) {
  loom::check::trial<8>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto count = loom::check::generate<async_gather_record_count>(trial);
    const auto input = loom::check::iota<kInputElementCount, int>(1, 1);
    const auto output = loom::check::fill<kSubgroupSize, int>(-1);
    loom::check::compare<async_gather_serial, async_gather_reference>(
        count, input, output,
        [&] { loom::check::expect_bitwise(output, output); });
  });
}

LOOM_CHECK_CASE(async_gather_repeated_reuse) {
  loom::check::require("hal.amdgpu.descriptor_set", "descriptor_set",
                       "amdgpu.cdna3.core");
  const auto input = loom::check::iota<kInputElementCount, int>(1, 1);
  const auto output = loom::check::fill<kSubgroupSize, int>(-1);
  loom::check::launch<async_gather_serial>(kRecordCount, input, output);
  // Input[record, lane] is 1 + 64*record + lane. Mirrored-lane consumption
  // across N records therefore produces N*(64-lane) + 32*N*(N-1).
  loom::check::expect_bitwise(
      output, loom::check::iota<kSubgroupSize, int>(2121792, -257));
}

LOOM_CHECK_CASE(async_gather_distinct_groups) {
  loom::check::require("hal.amdgpu.descriptor_set", "descriptor_set",
                       "amdgpu.cdna3.core");
  const auto input = loom::check::iota<kInputElementCount, int>(1, 1);
  const auto output = loom::check::fill<kSubgroupSize, int>(-1);
  loom::check::launch<async_gather_two_groups>(input, output);
  // The first two records contribute (64-lane) + (128-lane).
  loom::check::expect_bitwise(output,
                              loom::check::iota<kSubgroupSize, int>(192, -2));
}
