// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/buffer.h>
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

#include <stdfloat>

// IQ4_XS stores 256 nonlinear four-bit codes and eight signed six-bit scales.
struct IQ4XSBlock {
  // Base multiplier shared by all eight groups.
  std::float16_t scale;
  // Two high bits of each group scale code.
  unsigned short scales_high;
  // Two low four-bit scale codes per byte.
  unsigned char scales_low[4];
  // Sixteen packed bytes per group: low half followed by high half on decode.
  unsigned char quants[128];
};
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2);
static_assert(__builtin_offsetof(IQ4XSBlock, scales_high) == 2);
static_assert(__builtin_offsetof(IQ4XSBlock, scales_low) == 4);
static_assert(__builtin_offsetof(IQ4XSBlock, quants) == 8);

static float group_scale(const IQ4XSBlock* block, unsigned group) {
  auto* low = block->scales_low;
  unsigned low_bits = (low[group / 2] >> (4 * (group % 2))) & 15;
  unsigned high_bits = (block->scales_high >> (2 * group)) & 3;
  int signed_scale = int(low_bits | (high_bits << 4)) - 32;
  return float(block->scale) * float(signed_scale);
}

// Each workgroup decodes one 32-value group directly from the stored block.
[[loom::kernel, loom::workgroup_size(32, 1, 1),
  loom::workgroup_count(64, 1, 1)]]
void decode_iq4xs(
    [[loom::assume_aligned(2)]] const unsigned char* block_storage,
    const signed char* codebook, float* output) {
  const auto* blocks = reinterpret_cast<const IQ4XSBlock*>(block_storage);
  unsigned group_index = loom::kernel::workgroup::id.x;
  unsigned lane = loom::kernel::workitem::id.x;
  auto* block = &blocks[group_index / 8];
  unsigned group = group_index % 8;
  unsigned packed = block->quants[group * 16 + lane % 16];
  unsigned code = (packed >> (4 * (lane / 16))) & 15;
  output[group_index * 32 + lane] = group_scale(block, group) * codebook[code];
}

using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));
using Codes4 = signed char __attribute__((ext_vector_type(4)));
using Codebook16 = signed char __attribute__((ext_vector_type(16)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Half8 = std::float16_t __attribute__((ext_vector_type(8)));

static Codes4 lookup(Codebook16 table, Bytes4 indices) {
  return {table[indices[0]], table[indices[1]], table[indices[2]],
          table[indices[3]]};
}

// One workgroup decodes a whole block. Each workitem expands four packed bytes
// into four weights in each half of its group using register table lookups.
[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(8, 1, 1)]]
void decode_iq4xs_packed(
    [[loom::noalias,
      loom::assume_aligned(2)]] const unsigned char* block_storage,
    [[loom::noalias,
      loom::assume_aligned(64)]] const signed char* codebook_storage,
    [[loom::noalias]] float* output_storage) {
  const auto* blocks = reinterpret_cast<const IQ4XSBlock*>(block_storage);
  const auto* codebook = reinterpret_cast<const Codebook16*>(codebook_storage);
  auto* output = reinterpret_cast<Float4*>(output_storage);
  unsigned block_index = loom::kernel::workgroup::id.x;
  unsigned lane = loom::kernel::workitem::id.x;
  auto* block = &blocks[block_index];
  unsigned group = lane / 4;
  unsigned chunk = lane % 4;
  auto* quants = reinterpret_cast<const Bytes4*>(block->quants);
  Bytes4 packed = quants[group * 4 + chunk];
  auto low_codes = lookup(*codebook, packed & 15);
  auto high_codes = lookup(*codebook, packed >> 4);
  Float4 scales = group_scale(block, group);
  unsigned position = block_index * 64 + group * 8 + chunk;
  output[position] = __builtin_convertvector(low_codes, Float4) * scales;
  output[position + 4] = __builtin_convertvector(high_codes, Float4) * scales;
}

// Updating both inline byte arrays must preserve the base scale, high scale
// bits and neighboring records. Each byte has exactly one writing workitem.
[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(8, 1, 1)]]
void update_iq4xs([[loom::assume_aligned(2)]] unsigned char* block_storage) {
  auto* blocks = reinterpret_cast<IQ4XSBlock*>(block_storage);
  auto* block = &blocks[loom::kernel::workgroup::id.x];
  unsigned lane = loom::kernel::workitem::id.x;
  for (unsigned index = 0; index < 4; ++index) {
    block->quants[lane * 4 + index] ^= 0x11;
  }
  if (lane < 4) {
    block->scales_low[lane] ^= 0x11;
  }
}

static std::float16_t iq4xs_base_scale(unsigned block) {
  loom::assume(block < 8u);
  Half8 scales = {0.5f, -0.25f, 2.0f, -4.0f, 0.0625f, -0.125f, 1.0f, -2.0f};
  return scales[block];
}

static signed char iq4nl_code(unsigned code) {
  loom::assume(code < 16u);
  Codebook16 values = {-127, -104, -83, -65, -49, -35, -22, -10,
                       1,    13,   25,  38,  53,  69,  89,  113};
  return values[code];
}

void generate_iq4xs_blocks(loom::type::buffer<unsigned char> blocks) {
  auto bytes =
      loom::buffer::view<1088>(blocks, {}, loom::encoding::layout::dense<1>());
  for (unsigned block = 0; block < 8u; ++block) {
    unsigned base = block * 136u;
    unsigned short scale_bits =
        __builtin_bit_cast(unsigned short, iq4xs_base_scale(block));
    loom::view::store(static_cast<unsigned char>(scale_bits), bytes, base);
    loom::view::store(static_cast<unsigned char>(scale_bits >> 8u), bytes,
                      base + 1u);
    unsigned scales_high = 0u;
    for (unsigned group = 0; group < 8u; ++group) {
      unsigned scale_code = block * 8u + group;
      scales_high |= (scale_code >> 4u) << (group * 2u);
    }
    loom::view::store(static_cast<unsigned char>(scales_high), bytes,
                      base + 2u);
    loom::view::store(static_cast<unsigned char>(scales_high >> 8u), bytes,
                      base + 3u);
    for (unsigned pair = 0; pair < 4u; ++pair) {
      unsigned low_group = pair * 2u;
      unsigned low_code = block * 8u + low_group;
      unsigned high_code = low_code + 1u;
      unsigned packed = (low_code & 15u) | ((high_code & 15u) << 4u);
      loom::view::store(static_cast<unsigned char>(packed), bytes,
                        base + 4u + pair);
    }
    for (unsigned group = 0; group < 8u; ++group) {
      for (unsigned lane = 0; lane < 16u; ++lane) {
        unsigned low_code = (3u * lane + 5u * group + 7u * block) % 16u;
        unsigned high_lane = lane + 16u;
        unsigned high_code =
            (3u * high_lane + 5u * group + 7u * block + group + 8u * block) %
            16u;
        loom::view::store(
            static_cast<unsigned char>(low_code | (high_code << 4u)), bytes,
            base + 8u + group * 16u + lane);
      }
    }
  }
}

void generate_iq4nl_codebook(loom::type::buffer<signed char> codebook) {
  auto values =
      loom::buffer::view<16>(codebook, {}, loom::encoding::layout::dense<1>());
  for (unsigned code = 0; code < 16u; ++code) {
    loom::view::store(iq4nl_code(code), values, code);
  }
}

void decode_iq4xs_oracle(loom::type::buffer<unsigned char> blocks,
                         loom::type::buffer<signed char> codebook,
                         loom::type::buffer<float> output) {
  auto bytes =
      loom::buffer::view<1088>(blocks, {}, loom::encoding::layout::dense<1>());
  auto codes =
      loom::buffer::view<16>(codebook, {}, loom::encoding::layout::dense<1>());
  auto output_values =
      loom::buffer::view<2048>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned block = 0; block < 8u; ++block) {
    unsigned base = block * 136u;
    unsigned short scale_bits = loom::view::load(bytes, base);
    scale_bits |=
        static_cast<unsigned short>(loom::view::load(bytes, base + 1u)) << 8u;
    float scale =
        static_cast<float>(__builtin_bit_cast(std::float16_t, scale_bits));
    unsigned scales_high = loom::view::load(bytes, base + 2u);
    scales_high |= unsigned(loom::view::load(bytes, base + 3u)) << 8u;
    for (unsigned group = 0; group < 8u; ++group) {
      unsigned low_byte = loom::view::load(bytes, base + 4u + group / 2u);
      unsigned low_bits = (low_byte >> (4u * (group % 2u))) & 15u;
      unsigned high_bits = (scales_high >> (2u * group)) & 3u;
      int signed_scale = int(low_bits | (high_bits << 4u)) - 32;
      float group_multiplier = scale * float(signed_scale);
      for (unsigned lane = 0; lane < 32u; ++lane) {
        unsigned packed =
            loom::view::load(bytes, base + 8u + group * 16u + lane % 16u);
        unsigned code = (packed >> (4u * (lane / 16u))) & 15u;
        signed char code_value = loom::view::load(codes, code);
        loom::view::store(group_multiplier * float(code_value), output_values,
                          (block * 8u + group) * 32u + lane);
      }
    }
  }
}

void update_iq4xs_oracle(loom::type::buffer<unsigned char> blocks) {
  auto bytes =
      loom::buffer::view<1088>(blocks, {}, loom::encoding::layout::dense<1>());
  for (unsigned block = 0; block < 8u; ++block) {
    unsigned base = block * 136u;
    for (unsigned index = 0; index < 4u; ++index) {
      unsigned position = base + 4u + index;
      unsigned char value = loom::view::load(bytes, position);
      loom::view::store(static_cast<unsigned char>(value ^ 0x11u), bytes,
                        position);
    }
    for (unsigned index = 0; index < 128u; ++index) {
      unsigned position = base + 8u + index;
      unsigned char value = loom::view::load(bytes, position);
      loom::view::store(static_cast<unsigned char>(value ^ 0x11u), bytes,
                        position);
    }
  }
}

LOOM_CHECK_SCENARIO(iq4xs_block_values) {
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto storage = loom::check::fill<1152, unsigned char>(0xa5u);
        const auto blocks = loom::check::slice<1088>(storage, 32);
        loom::check::generate<generate_iq4xs_blocks>(blocks);
        const auto codebook = loom::check::fill<16, signed char>(0);
        loom::check::generate<generate_iq4nl_codebook>(codebook);
        const auto output_storage = loom::check::fill<2080, float>(-123.0f);
        const auto output_prefix = loom::check::slice<16>(output_storage, 0);
        const auto output = loom::check::slice<2048>(output_storage, 16);
        const auto output_suffix = loom::check::slice<16>(output_storage, 2064);
        loom::check::compare<decode_iq4xs, decode_iq4xs_oracle>(
            blocks, codebook, output, [&] {
              loom::check::expect_bitwise(storage, storage);
              loom::check::expect_bitwise(codebook, codebook);
              loom::check::expect_bitwise(output, output);
              loom::check::expect_bitwise(output_prefix, output_prefix);
              loom::check::expect_bitwise(output_suffix, output_suffix);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto storage = loom::check::fill<1152, unsigned char>(0xa5u);
        const auto blocks = loom::check::slice<1088>(storage, 32);
        loom::check::generate<generate_iq4xs_blocks>(blocks);
        const auto codebook = loom::check::fill<16, signed char>(0);
        loom::check::generate<generate_iq4nl_codebook>(codebook);
        const auto output_storage = loom::check::fill<2080, float>(-123.0f);
        const auto output_prefix = loom::check::slice<16>(output_storage, 0);
        const auto output = loom::check::slice<2048>(output_storage, 16);
        const auto output_suffix = loom::check::slice<16>(output_storage, 2064);
        loom::check::compare<decode_iq4xs_packed, decode_iq4xs_oracle>(
            blocks, codebook, output, [&] {
              loom::check::expect_bitwise(storage, storage);
              loom::check::expect_bitwise(codebook, codebook);
              loom::check::expect_bitwise(output, output);
              loom::check::expect_bitwise(output_prefix, output_prefix);
              loom::check::expect_bitwise(output_suffix, output_suffix);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto storage = loom::check::fill<1152, unsigned char>(0xa5u);
        const auto blocks = loom::check::slice<1088>(storage, 32);
        loom::check::generate<generate_iq4xs_blocks>(blocks);
        loom::check::compare<update_iq4xs, update_iq4xs_oracle>(
            blocks, [&] { loom::check::expect_bitwise(storage, storage); });
      });
}
