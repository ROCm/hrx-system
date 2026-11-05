// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Routed Q4_K gate/up SwiGLU compiler workload.
//
// One wave computes one routed output channel. Each loop iteration assigns one
// 256-element Q4_K block to every eight-lane group for paired Q4_K decode, Q8_1
// dot products, and gate/up accumulation. Varying input_size changes the loop
// bound while keeping the compiled body bounded.

#include <loomcxx/buffer.h>
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/scalar.h>
#include <loomcxx/target.h>
#include <loomcxx/vector.h>
#include <loomcxx/view.h>

#include <stdfloat>

using F16x4 = std::float16_t __attribute__((ext_vector_type(4)));
using F16x8 = std::float16_t __attribute__((ext_vector_type(8)));
using I8x16 = signed char __attribute__((ext_vector_type(16)));
using I32x4 = int __attribute__((ext_vector_type(4)));
using U8x16 = unsigned char __attribute__((ext_vector_type(16)));
using U32x4 = unsigned __attribute__((ext_vector_type(4)));

struct alignas(16) Q4KBlock {
  std::float16_t scale;
  std::float16_t minimum;
  unsigned char scales[12];
  unsigned char quants[128];
};
static_assert(sizeof(Q4KBlock) == 144);
static_assert(__builtin_offsetof(Q4KBlock, scales) == 4);
static_assert(__builtin_offsetof(Q4KBlock, quants) == 16);

struct alignas(16) Q8_1Group {
  F16x8 scales_and_sums;
  signed char quants[128];
};
static_assert(sizeof(Q8_1Group) == 144);
static_assert(__builtin_offsetof(Q8_1Group, quants) == 16);

template <loom::type::size_type Rank>
using Layout = loom::type::encoding<loom::encoding::role::layout, Rank>;
using Q8F16View = loom::type::view<
    loom::type::shape<loom::type::dynamic, loom::type::dynamic,
                      sizeof(Q8_1Group) / sizeof(std::float16_t)>,
    const std::float16_t, Layout<3>>;
using Q8I8View =
    loom::type::view<loom::type::shape<loom::type::dynamic, loom::type::dynamic,
                                       sizeof(Q8_1Group)>,
                     const signed char, Layout<3>>;
using WeightU32View = loom::type::view<loom::type::shape<loom::type::dynamic>,
                                       const unsigned, Layout<1>>;

struct Q4ScaleMinimum {
  unsigned scale;
  unsigned minimum;
};

struct Q4ChunkPair {
  U8x16 low;
  float low_scale;
  float low_minimum;
  U8x16 high;
  float high_scale;
  float high_minimum;
};

[[loom::config("ffn_routed_gate_up.input_size"),
  loom::where(loom::predicate::range(512u, 32768u) &&
              loom::predicate::multiple_of(512u))]]
extern const unsigned ffn_input_size;
[[loom::config("ffn_routed_gate_up.expert_count")]]
const unsigned ffn_expert_count = 128u;
[[loom::config("ffn_routed_gate_up.route_count")]]
const unsigned ffn_route_count = 8u;
[[loom::config("ffn_routed_gate_up.output_size")]]
const unsigned ffn_output_size = 768u;
[[loom::config("ffn_routed_gate_up.token_capacity")]]
const unsigned ffn_token_capacity = 16u;

static LOOM_INLINE Q4ScaleMinimum q4k_scale_from_header(unsigned scale0,
                                                        unsigned scale1,
                                                        unsigned scale2,
                                                        unsigned group) {
  loom::assume(group < 8u);
  bool low_group = group < 4u;
  unsigned scale_shift = (group % 4u) * 8u;
  unsigned high_shift = scale_shift + 2u;
  unsigned minimum_shift = scale_shift + 4u;
  unsigned scale_source = low_group ? scale0 : scale2;
  unsigned minimum_source = low_group ? scale1 : scale2;
  unsigned scale_high_shift = low_group ? scale_shift : high_shift;
  unsigned minimum_low_shift = low_group ? scale_shift : minimum_shift;
  unsigned scale = ((scale_source >> scale_shift) & 15u) |
                   ((scale0 >> scale_high_shift) & 48u);
  unsigned minimum = ((minimum_source >> minimum_low_shift) & 15u) |
                     ((scale1 >> scale_high_shift) & 48u);
  return {scale, minimum};
}

static LOOM_INLINE Q4ChunkPair q4k_chunk_pair(
    WeightU32View weights, unsigned weight_word_count, unsigned block_word_base,
    unsigned group_pair, unsigned half, U32x4 header_words) {
  loom::assume(group_pair < 4u && half < 2u);
  F16x8 header_halves = __builtin_bit_cast(F16x8, header_words);
  float scale = float(header_halves[0]);
  float minimum = float(header_halves[1]);
  unsigned scale0 = header_words[1];
  unsigned scale1 = header_words[2];
  unsigned scale2 = header_words[3];

  unsigned packed_word = block_word_base + 4u + (group_pair * 2u + half) * 4u;
  unsigned packed_word_end = packed_word + 4u;
  loom::assume(packed_word_end <= weight_word_count);
  U32x4 packed = loom::vector::load<U32x4>(weights, {packed_word});
  U32x4 low_words = packed & 0x0f0f0f0fu;
  U32x4 high_words = (packed >> 4u) & 0x0f0f0f0fu;
  U8x16 low = __builtin_bit_cast(U8x16, low_words);
  U8x16 high = __builtin_bit_cast(U8x16, high_words);

  Q4ScaleMinimum low_values =
      q4k_scale_from_header(scale0, scale1, scale2, group_pair * 2u);
  Q4ScaleMinimum high_values =
      q4k_scale_from_header(scale0, scale1, scale2, group_pair * 2u + 1u);
  return {low,
          scale * float(low_values.scale),
          minimum * float(low_values.minimum),
          high,
          scale * float(high_values.scale),
          minimum * float(high_values.minimum)};
}

static LOOM_INLINE float q4k_q8_1_dot(U8x16 q4_values, float scale,
                                      float minimum, I8x16 q8_values,
                                      float q8_scale, float q8_sum) {
  I32x4 partial = {};
  partial = loom::vector::dot4i(q4_values, q8_values, partial);
  int sum = loom::vector::reduce::addi(partial, 0);
  return q8_scale * scale * float(sum) - minimum * (q8_sum * 0.5f);
}

LOOM_DEVICE
LOOM_TEMPLATE_DECL("ffn_routed_gate_up.q4k_q8.body")
void ffn_routed_gate_up_q4k_q8_body(
    bool publish_output, unsigned token_count, unsigned token,
    unsigned route_count, unsigned route, unsigned route_stride,
    unsigned expert_count, unsigned output_size, unsigned channel,
    unsigned lane, Q8F16View q8_f16, Q8I8View q8_i8, const int* route_ids,
    WeightU32View gate_weight, WeightU32View up_weight,
    unsigned weight_word_count, float* output);

LOOM_TEMPLATE_DEF(ffn_routed_gate_up_q4k_q8_body)
void ffn_routed_gate_up_swiglu_q4k_q8_body(
    bool publish_output, unsigned token_count, unsigned token,
    unsigned route_count, unsigned route, unsigned route_stride,
    unsigned expert_count, unsigned output_size, unsigned channel,
    unsigned lane, Q8F16View q8_f16, Q8I8View q8_i8, const int* route_ids,
    WeightU32View gate_weight, WeightU32View up_weight,
    unsigned weight_word_count, float* output) {
  loom::assume(token_count >= 1u && token_count <= 512u);
  loom::assume(route_count >= 1u && route_count <= 8u);
  loom::assume(route_stride >= 1u && route_stride <= 128u);
  loom::assume(route_count <= route_stride);
  loom::assume(expert_count >= 1u && expert_count <= 128u);
  loom::assume(output_size >= 1u && output_size <= 4096u);
  loom::assume(token < token_count && route < route_count &&
               channel < output_size);

  unsigned q4_block_count = ffn_input_size / 256u;
  unsigned q8_group_count = ffn_input_size / 128u;
  unsigned expert = unsigned(route_ids[token * route_stride + route]);
  loom::assume(expert < expert_count);
  loom::assume(expert < 128u);
  unsigned row_block_base = (expert * output_size + channel) * q4_block_count;
  unsigned q4_group_pair = (lane / 2u) % 4u;
  unsigned q4_half = lane % 2u;
  unsigned lane_q4_block = lane / 8u;
  unsigned q8_group_in_block = q4_group_pair / 2u;
  unsigned pair_in_q8_group = q4_group_pair % 2u;
  unsigned q8_low_inner_block = pair_in_q8_group * 2u;
  unsigned q8_high_inner_block = q8_low_inner_block + 1u;

  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  unsigned blocks_per_iteration = loom::kernel::subgroup::size() / 8u;
  unsigned iteration_count =
      (q4_block_count + blocks_per_iteration - 1u) / blocks_per_iteration;
  for (unsigned iteration = 0; iteration < iteration_count; ++iteration) {
    unsigned q4_block = iteration * blocks_per_iteration + lane_q4_block;
    if (q4_block < q4_block_count) {
      unsigned q8_group = q4_block * 2u + q8_group_in_block;
      F16x4 q8_scales = loom::vector::load<F16x4>(
          q8_f16, {token, q8_group, pair_in_q8_group * 4u});
      float q8_low_scale = float(q8_scales[0]);
      float q8_low_sum = float(q8_scales[1]);
      float q8_high_scale = float(q8_scales[2]);
      float q8_high_sum = float(q8_scales[3]);
      unsigned q8_low_offset = __builtin_offsetof(Q8_1Group, quants) +
                               (q8_low_inner_block * 2u + q4_half) * 16u;
      unsigned q8_high_offset = __builtin_offsetof(Q8_1Group, quants) +
                                (q8_high_inner_block * 2u + q4_half) * 16u;
      I8x16 q8_low =
          loom::vector::load<I8x16>(q8_i8, {token, q8_group, q8_low_offset});
      I8x16 q8_high =
          loom::vector::load<I8x16>(q8_i8, {token, q8_group, q8_high_offset});

      unsigned block_word_base =
          (row_block_base + q4_block) * (sizeof(Q4KBlock) / sizeof(unsigned));
      unsigned block_header_end = block_word_base + 4u;
      loom::assume(block_header_end <= weight_word_count);
      U32x4 gate_header =
          loom::vector::load<U32x4>(gate_weight, {block_word_base});
      U32x4 up_header = loom::vector::load<U32x4>(up_weight, {block_word_base});
      Q4ChunkPair gate =
          q4k_chunk_pair(gate_weight, weight_word_count, block_word_base,
                         q4_group_pair, q4_half, gate_header);
      Q4ChunkPair up =
          q4k_chunk_pair(up_weight, weight_word_count, block_word_base,
                         q4_group_pair, q4_half, up_header);

      gate_accumulator +=
          q4k_q8_1_dot(gate.low, gate.low_scale, gate.low_minimum, q8_low,
                       q8_low_scale, q8_low_sum) +
          q4k_q8_1_dot(gate.high, gate.high_scale, gate.high_minimum, q8_high,
                       q8_high_scale, q8_high_sum);
      up_accumulator += q4k_q8_1_dot(up.low, up.low_scale, up.low_minimum,
                                     q8_low, q8_low_scale, q8_low_sum) +
                        q4k_q8_1_dot(up.high, up.high_scale, up.high_minimum,
                                     q8_high, q8_high_scale, q8_high_sum);
    }
  }

  float gate = loom::kernel::subgroup::reduce::addf(gate_accumulator);
  float up = loom::kernel::subgroup::reduce::addf(up_accumulator);
  if (publish_output && lane == 0u) {
    output[(token * route_count + route) * output_size + channel] =
        loom::scalar::siluf(gate) * up;
  }
}

static loom::kernel::configuration ffn_routed_gate_up_configuration(
    unsigned, unsigned, unsigned, unsigned, unsigned) {
  return {{(ffn_output_size + 3u) / 4u, ffn_route_count, ffn_token_capacity},
          {4u * loom::target::subgroup::size(), 1u, 1u}};
}

[[loom::kernel(ffn_routed_gate_up_configuration)]] void
ffn_routed_gate_up_swiglu_q4k_q8(
    unsigned token_count, unsigned route_count, unsigned route_stride,
    unsigned expert_count, unsigned output_size,
    [[loom::noalias,
      loom::assume_aligned(64)]] const unsigned char* q8_input_storage,
    [[loom::noalias, loom::assume_aligned(64)]] const int* route_ids,
    [[loom::noalias,
      loom::assume_aligned(64)]] const unsigned char* gate_weight_storage,
    [[loom::noalias,
      loom::assume_aligned(64)]] const unsigned char* up_weight_storage,
    [[loom::noalias, loom::assume_aligned(64)]] float* output) {
  loom::assume(token_count >= 1u && token_count <= ffn_token_capacity);
  loom::assume(route_count == ffn_route_count);
  loom::assume(route_stride >= 1u && route_stride <= 128u);
  loom::assume(expert_count == ffn_expert_count);
  loom::assume(output_size == ffn_output_size);

  unsigned token_candidate = loom::kernel::workgroup::id.z;
  unsigned channel_candidate =
      loom::kernel::workgroup::id.x * 4u + loom::kernel::subgroup::id();
  unsigned route = loom::kernel::workgroup::id.y;
  unsigned lane = loom::kernel::subgroup::lane_id();
  bool valid_token = token_candidate < token_count;
  unsigned token = valid_token ? token_candidate : 0u;
  unsigned channel = channel_candidate;
  loom::assume(token < token_count && route < route_count &&
               channel < output_size);

  unsigned q4_block_count = ffn_input_size / 256u;
  unsigned q8_group_count = ffn_input_size / 128u;
  auto q8_f16 = loom::buffer::view<loom::type::dynamic, loom::type::dynamic,
                                   sizeof(Q8_1Group) / sizeof(std::float16_t)>(
      reinterpret_cast<const std::float16_t*>(q8_input_storage),
      {token_count, q8_group_count}, loom::encoding::layout::dense<3>());
  auto q8_i8 = loom::buffer::view<loom::type::dynamic, loom::type::dynamic,
                                  sizeof(Q8_1Group)>(
      reinterpret_cast<const signed char*>(q8_input_storage),
      {token_count, q8_group_count}, loom::encoding::layout::dense<3>());
  unsigned weight_word_count = expert_count * output_size * q4_block_count *
                               (sizeof(Q4KBlock) / sizeof(unsigned));
  auto gate_weight = loom::buffer::view<loom::type::dynamic>(
      reinterpret_cast<const unsigned*>(gate_weight_storage),
      {weight_word_count}, loom::encoding::layout::dense<1>());
  auto up_weight = loom::buffer::view<loom::type::dynamic>(
      reinterpret_cast<const unsigned*>(up_weight_storage), {weight_word_count},
      loom::encoding::layout::dense<1>());

  ffn_routed_gate_up_q4k_q8_body(
      valid_token, token_count, token, route_count, route, route_stride,
      expert_count, output_size, channel, lane, q8_f16, q8_i8, route_ids,
      gate_weight, up_weight, weight_word_count, output);
}

constexpr unsigned kQ4KQ8TokenCount = 2u;
constexpr unsigned kQ4KQ8RouteCount = 8u;
constexpr unsigned kQ4KQ8RouteStride = 11u;
constexpr unsigned kQ4KQ8ExpertCount = 128u;
constexpr unsigned kQ4KQ8StoredExpertCount = 2u;
constexpr unsigned kQ4KQ8OutputSize = 768u;
constexpr unsigned kQ4KQ8BlockCount = 2u;
constexpr unsigned kQ4KQ8GroupCount = kQ4KQ8BlockCount * 2u;
constexpr unsigned kQ4KQ8InputByteCount =
    kQ4KQ8TokenCount * kQ4KQ8GroupCount * sizeof(Q8_1Group);
constexpr unsigned kQ4KQ8WeightByteCount = kQ4KQ8StoredExpertCount *
                                           kQ4KQ8OutputSize * kQ4KQ8BlockCount *
                                           sizeof(Q4KBlock);
constexpr unsigned kQ4KQ8OutputElementCount =
    kQ4KQ8TokenCount * kQ4KQ8RouteCount * kQ4KQ8OutputSize;

static std::float16_t q4k_q8_power_of_two(unsigned exponent) {
  return static_cast<std::float16_t>(1.0f / float(1u << exponent));
}

static int q4k_q8_input_value(unsigned token, unsigned record, unsigned group,
                              unsigned lane) {
  return int((token * 11u + record * 7u + group * 5u + lane * 3u + lane / 5u) %
             15u) -
         7;
}

template <unsigned Kind>
static unsigned q4k_scale_code(unsigned expert, unsigned channel,
                               unsigned block, unsigned group) {
  return (expert * 17u + channel * 3u + block * 11u + group * 7u + Kind * 13u) %
         64u;
}

template <unsigned Kind>
static unsigned q4k_minimum_code(unsigned expert, unsigned channel,
                                 unsigned block, unsigned group) {
  return (expert * 13u + channel * 5u + block * 3u + group * 9u + Kind * 7u) %
         64u;
}

template <unsigned Kind>
static unsigned q4k_quant_code(unsigned expert, unsigned channel,
                               unsigned block, unsigned group, unsigned lane) {
  return (expert * 13u + channel * 7u + block * 5u + group * 3u + lane * 11u +
          lane / 7u + Kind * 9u) %
         16u;
}

void generate_q4k_q8_input(loom::type::buffer<unsigned char> input) {
  auto bytes = loom::buffer::view<kQ4KQ8InputByteCount>(
      input, {}, loom::encoding::layout::dense<1>());
  for (unsigned token = 0; token < kQ4KQ8TokenCount; ++token) {
    for (unsigned record = 0; record < kQ4KQ8GroupCount; ++record) {
      unsigned base = (token * kQ4KQ8GroupCount + record) * sizeof(Q8_1Group);
      for (unsigned group = 0; group < 4u; ++group) {
        std::float16_t scale =
            q4k_q8_power_of_two(4u + (token + record + group) % 3u);
        int sum = 0;
        for (unsigned lane = 0; lane < 32u; ++lane) {
          int value = q4k_q8_input_value(token, record, group, lane);
          sum += value;
          loom::view::store(static_cast<unsigned char>(value), bytes,
                            base + 16u + group * 32u + lane);
        }
        std::float16_t scaled_sum =
            static_cast<std::float16_t>(float(scale) * float(sum));
        unsigned short scale_bits = __builtin_bit_cast(unsigned short, scale);
        unsigned short sum_bits =
            __builtin_bit_cast(unsigned short, scaled_sum);
        unsigned header = base + group * 4u;
        loom::view::store(static_cast<unsigned char>(scale_bits), bytes,
                          header);
        loom::view::store(static_cast<unsigned char>(scale_bits >> 8u), bytes,
                          header + 1u);
        loom::view::store(static_cast<unsigned char>(sum_bits), bytes,
                          header + 2u);
        loom::view::store(static_cast<unsigned char>(sum_bits >> 8u), bytes,
                          header + 3u);
      }
    }
  }
}

void generate_q4k_q8_route_ids(loom::type::buffer<int> route_ids) {
  auto values = loom::buffer::view<kQ4KQ8TokenCount * kQ4KQ8RouteStride>(
      route_ids, {}, loom::encoding::layout::dense<1>());
  for (unsigned token = 0; token < kQ4KQ8TokenCount; ++token) {
    for (unsigned route = 0; route < kQ4KQ8RouteStride; ++route) {
      int expert = route < kQ4KQ8RouteCount
                       ? int((token + route) % kQ4KQ8StoredExpertCount)
                       : int(kQ4KQ8ExpertCount - 1u);
      loom::view::store(expert, values, token * kQ4KQ8RouteStride + route);
    }
  }
}

template <unsigned Kind>
void generate_q4k_weight(loom::type::buffer<unsigned char> weight) {
  auto bytes = loom::buffer::view<kQ4KQ8WeightByteCount>(
      weight, {}, loom::encoding::layout::dense<1>());
  for (unsigned expert = 0; expert < kQ4KQ8StoredExpertCount; ++expert) {
    for (unsigned channel = 0; channel < kQ4KQ8OutputSize; ++channel) {
      for (unsigned block = 0; block < kQ4KQ8BlockCount; ++block) {
        unsigned record =
            (expert * kQ4KQ8OutputSize + channel) * kQ4KQ8BlockCount + block;
        unsigned base = record * sizeof(Q4KBlock);
        std::float16_t scale =
            q4k_q8_power_of_two(9u + (expert + channel + block + Kind) % 3u);
        std::float16_t minimum = q4k_q8_power_of_two(
            11u + (expert + channel + 2u * block + Kind) % 2u);
        unsigned short scale_bits = __builtin_bit_cast(unsigned short, scale);
        unsigned short minimum_bits =
            __builtin_bit_cast(unsigned short, minimum);
        loom::view::store(static_cast<unsigned char>(scale_bits), bytes, base);
        loom::view::store(static_cast<unsigned char>(scale_bits >> 8u), bytes,
                          base + 1u);
        loom::view::store(static_cast<unsigned char>(minimum_bits), bytes,
                          base + 2u);
        loom::view::store(static_cast<unsigned char>(minimum_bits >> 8u), bytes,
                          base + 3u);
        for (unsigned group = 0; group < 4u; ++group) {
          unsigned low_scale =
              q4k_scale_code<Kind>(expert, channel, block, group);
          unsigned high_scale =
              q4k_scale_code<Kind>(expert, channel, block, group + 4u);
          unsigned low_minimum =
              q4k_minimum_code<Kind>(expert, channel, block, group);
          unsigned high_minimum =
              q4k_minimum_code<Kind>(expert, channel, block, group + 4u);
          loom::view::store(static_cast<unsigned char>(
                                (low_scale & 15u) | ((low_scale >> 4u) << 4u) |
                                ((high_scale >> 4u) << 6u)),
                            bytes, base + 4u + group);
          loom::view::store(
              static_cast<unsigned char>((low_minimum & 15u) |
                                         ((low_minimum >> 4u) << 4u) |
                                         ((high_minimum >> 4u) << 6u)),
              bytes, base + 8u + group);
          loom::view::store(
              static_cast<unsigned char>((high_scale & 15u) |
                                         ((high_minimum & 15u) << 4u)),
              bytes, base + 12u + group);
        }
        for (unsigned pair = 0; pair < 4u; ++pair) {
          for (unsigned half = 0; half < 2u; ++half) {
            for (unsigned lane = 0; lane < 16u; ++lane) {
              unsigned value_lane = half * 16u + lane;
              unsigned low_code = q4k_quant_code<Kind>(expert, channel, block,
                                                       pair * 2u, value_lane);
              unsigned high_code = q4k_quant_code<Kind>(
                  expert, channel, block, pair * 2u + 1u, value_lane);
              unsigned offset = base + 16u + pair * 32u + half * 16u + lane;
              loom::view::store(
                  static_cast<unsigned char>(low_code | (high_code << 4u)),
                  bytes, offset);
            }
          }
        }
      }
    }
  }
}

template <unsigned Kind>
double q4k_q8_projection_oracle(unsigned token, unsigned expert,
                                unsigned channel) {
  double accumulator = 0.0;
  for (unsigned block = 0; block < kQ4KQ8BlockCount; ++block) {
    float scale =
        float(q4k_q8_power_of_two(9u + (expert + channel + block + Kind) % 3u));
    float minimum = float(
        q4k_q8_power_of_two(11u + (expert + channel + 2u * block + Kind) % 2u));
    for (unsigned group = 0; group < 8u; ++group) {
      unsigned input_record = block * 2u + group / 4u;
      unsigned input_group = group % 4u;
      float input_scale = float(
          q4k_q8_power_of_two(4u + (token + input_record + input_group) % 3u));
      int input_sum = 0;
      int dot = 0;
      for (unsigned lane = 0; lane < 32u; ++lane) {
        int input_value =
            q4k_q8_input_value(token, input_record, input_group, lane);
        input_sum += input_value;
        dot += int(q4k_quant_code<Kind>(expert, channel, block, group, lane)) *
               input_value;
      }
      unsigned group_scale =
          q4k_scale_code<Kind>(expert, channel, block, group);
      unsigned group_minimum =
          q4k_minimum_code<Kind>(expert, channel, block, group);
      accumulator += double(input_scale) * double(scale) * double(group_scale) *
                     double(dot);
      accumulator -= double(minimum) * double(group_minimum) *
                     double(input_scale) * double(input_sum);
    }
  }
  return accumulator;
}

void q4k_q8_swiglu_oracle(unsigned token_count, unsigned route_count,
                          unsigned route_stride, unsigned expert_count,
                          unsigned output_size,
                          loom::type::buffer<unsigned char> q8_input,
                          loom::type::buffer<int> route_ids,
                          loom::type::buffer<unsigned char> gate_weight,
                          loom::type::buffer<unsigned char> up_weight,
                          loom::type::buffer<float> output) {
  loom::assume(
      token_count == kQ4KQ8TokenCount && route_count == kQ4KQ8RouteCount &&
      route_stride == kQ4KQ8RouteStride && expert_count == kQ4KQ8ExpertCount &&
      output_size == kQ4KQ8OutputSize);
  auto output_values = loom::buffer::view<kQ4KQ8OutputElementCount>(
      output, {}, loom::encoding::layout::dense<1>());
  for (unsigned token = 0; token < token_count; ++token) {
    for (unsigned route = 0; route < route_count; ++route) {
      unsigned expert = (token + route) % kQ4KQ8StoredExpertCount;
      for (unsigned channel = 0; channel < output_size; ++channel) {
        float gate = static_cast<float>(
            q4k_q8_projection_oracle<0>(token, expert, channel));
        float up = static_cast<float>(
            q4k_q8_projection_oracle<1>(token, expert, channel));
        float value =
            gate / (1.0f + loom::scalar::approximate::expf(-gate)) * up;
        loom::view::store(
            value, output_values,
            (token * kQ4KQ8RouteCount + route) * kQ4KQ8OutputSize + channel);
      }
    }
  }
}

LOOM_CHECK_SCENARIO(q4k_q8_swiglu_values) {
  loom::check::trial<1>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto q8_input =
        loom::check::fill<kQ4KQ8InputByteCount, unsigned char>(0u);
    loom::check::generate<generate_q4k_q8_input>(q8_input);
    const auto route_ids =
        loom::check::fill<kQ4KQ8TokenCount * kQ4KQ8RouteStride, int>(0);
    loom::check::generate<generate_q4k_q8_route_ids>(route_ids);
    const auto gate_weight =
        loom::check::fill<kQ4KQ8WeightByteCount, unsigned char>(0u);
    loom::check::generate<generate_q4k_weight<0>>(gate_weight);
    const auto up_weight =
        loom::check::fill<kQ4KQ8WeightByteCount, unsigned char>(0u);
    loom::check::generate<generate_q4k_weight<1>>(up_weight);
    const auto output_storage =
        loom::check::fill<kQ4KQ8OutputElementCount + 32u, float>(-123.0f);
    const auto output_prefix = loom::check::slice<16>(output_storage, 0);
    const auto output =
        loom::check::slice<kQ4KQ8OutputElementCount>(output_storage, 16);
    const auto output_suffix =
        loom::check::slice<16>(output_storage, kQ4KQ8OutputElementCount + 16u);
    loom::check::compare<ffn_routed_gate_up_swiglu_q4k_q8,
                         q4k_q8_swiglu_oracle>(
        loom::kernel::workload(kQ4KQ8TokenCount, kQ4KQ8RouteCount,
                               kQ4KQ8RouteStride, kQ4KQ8ExpertCount,
                               kQ4KQ8OutputSize),
        kQ4KQ8TokenCount, kQ4KQ8RouteCount, kQ4KQ8RouteStride,
        kQ4KQ8ExpertCount, kQ4KQ8OutputSize, q8_input, route_ids, gate_weight,
        up_weight, output, [&] {
          loom::check::expect_bitwise(q8_input, q8_input);
          loom::check::expect_bitwise(route_ids, route_ids);
          loom::check::expect_bitwise(gate_weight, gate_weight);
          loom::check::expect_bitwise(up_weight, up_weight);
          loom::check::expect_approximate(output, output, 0.002, 0.002);
          loom::check::expect_bitwise(output_prefix, output_prefix);
          loom::check::expect_bitwise(output_suffix, output_suffix);
        });
  });
}

LOOM_CHECK_BENCHMARK(q4k_q8_swiglu_latency, q4k_q8_swiglu_values);
