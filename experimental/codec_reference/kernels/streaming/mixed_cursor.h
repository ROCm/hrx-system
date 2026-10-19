#pragma once
#include <loomcxx/buffer.h>
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/target.h>

#include "mixed_image.h"
[[loom::config("jpeg.chunk_bytes"),
  loom::where(loom::predicate::range(1u, 65536u))]]
extern const unsigned chunk_bytes;
static constexpr unsigned component_count = 3;
static constexpr unsigned table_count = 8;
static constexpr bool cache_tables = false;

static LOOM_FORCE_INLINE unsigned block_descriptor(BlockLayout layout,
                                                   unsigned block) {
  unsigned word =
      block < 4 ? layout.first : (block < 8 ? layout.second : layout.third);
  return (word >> ((block & 3u) * 8u)) & 255u;
}

using Predictors = int __attribute__((ext_vector_type(4)));

struct Cursor {
  // Next bit in the unstuffed restart segment.
  unsigned position;
  // Every consumed bit, including invalid-prefix revisits.
  unsigned reads;
  // Maximum permitted reads for this trajectory.
  unsigned budget;
  // Completed blocks since the most recent prefix reset.
  unsigned blocks;
  // Block index within the scan MCU, independent of component identity.
  unsigned block_index;
  // Next coefficient in zigzag order; zero starts a DC codeword.
  unsigned coefficient;
  // Component-relative DC sums since the most recent prefix reset.
  Predictors predictors;
  // Word index of the cached 64-bit lookahead.
  unsigned cached_word;
  // Two adjacent big-endian words, retained across syntax transitions.
  unsigned long long cached_bits;
  // Zero is valid; one is invalid syntax/extent; two exhausts the read cap.
  unsigned error;
};

struct Read {
  // Updated entropy trajectory and cached input words.
  Cursor cursor;
  // Codeword or amplitude bits, interpreted by the transition's syntax.
  unsigned value;
};

static LOOM_FORCE_INLINE unsigned minimum(unsigned left, unsigned right) {
  return left < right ? left : right;
}

static LOOM_FORCE_INLINE Read peek(const unsigned* words, Cursor cursor) {
  unsigned word = cursor.position >> 5;
  if (word != cursor.cached_word) {
    cursor.cached_bits =
        (static_cast<unsigned long long>(words[word]) << 32) | words[word + 1];
    cursor.cached_word = word;
  }
  return {cursor, static_cast<unsigned>(cursor.cached_bits >>
                                        (32 - (cursor.position & 31)))};
}

// Header-derived canonical tables retain a final root entry or a bounded
// offset/mask/shift into their secondary suffix table. Undefined prefixes stay
// zero; extent/read-cap and syntax errors remain the cursor transition's job.
static LOOM_FORCE_INLINE unsigned lookup_huffman(const unsigned* table,
                                                 unsigned bits) {
  unsigned packed = table[bits >> 23];
  if ((packed & 0x80000000u) != 0) {
    unsigned offset = packed & 2047u;
    unsigned shift = (packed >> 24) & 31u;
    unsigned mask = (packed >> 16) & 127u;
    unsigned index = offset + ((bits >> shift) & mask);
    loom::assume(index < 2048u);
    packed = table[index];
  }
  return packed;
}

struct Codeword {
  // Cursor after Huffman bits; amplitude extent checking remains separate.
  Cursor cursor;
  // Decoded Huffman symbol.
  unsigned value;
  // Original 32-bit lookahead, including the symbol's amplitude bits.
  unsigned lookahead;
  // Huffman code length, no more than sixteen bits for a validated table.
  unsigned length;
};

static LOOM_FORCE_INLINE Codeword codeword(const unsigned* words,
                                           const unsigned* table,
                                           unsigned limit, Cursor cursor) {
  Read input = peek(words, cursor);
  cursor = input.cursor;
  unsigned bits = input.value;
  unsigned packed = lookup_huffman(table, bits);
  unsigned length = packed >> 8;
  unsigned symbol = packed & 255;

  unsigned consumed = length ? length : 16;
  unsigned available = limit - cursor.position;
  unsigned remaining = cursor.budget - cursor.reads;
  if (consumed > available || consumed > remaining) {
    unsigned taken = minimum(consumed, minimum(available, remaining));
    cursor.position += taken;
    cursor.reads += taken;
    cursor.error = remaining <= available ? 2 : 1;
  } else {
    cursor.position += consumed;
    cursor.reads += consumed;
    if (length == 0) {
      cursor.error = 1;
    }
  }
  return {cursor, symbol, bits, length};
}

static LOOM_FORCE_INLINE Read amplitude(unsigned lookahead, unsigned length,
                                        unsigned count, unsigned limit,
                                        Cursor cursor) {
  int result = 0;
  if (count != 0 && cursor.error == 0) {
    if (count > cursor.budget - cursor.reads) {
      cursor.error = 2;
    } else if (count > limit - cursor.position) {
      cursor.error = 1;
    } else {
      // A codeword (<=16 bits) and its amplitude (<=11 bits) fit in the
      // already loaded window. Preserve the existing extent/read-cap errors.
      loom::assume(length <= 16u);
      unsigned value = (lookahead << length) >> (32 - count);
      result = static_cast<int>(value);
      if (value < (1u << (count - 1))) {
        result -= (1u << count) - 1;
      }
      cursor.position += count;
      cursor.reads += count;
    }
  }
  return {cursor, static_cast<unsigned>(result)};
}

struct Symbol {
  // Cursor after one complete syntax transition or an explicit error.
  Cursor cursor;
  // Zigzag coefficient index, or all ones for EOB/ZRL/error.
  unsigned index;
  // Absolute DC or signed AC coefficient when index is present.
  int value;
};

static LOOM_FORCE_INLINE Symbol advance_symbol_slow(const unsigned* words,
                                                    const unsigned* tables,
                                                    BlockLayout layout,
                                                    unsigned limit,
                                                    Cursor cursor) {
  unsigned output_index = 0xffffffffu;
  int output_value = 0;
  // The cursor advances cyclically through the scan's verified block order.
  unsigned block_index = cursor.block_index;
  unsigned block_count = layout.count;
  loom::assume(block_index < block_count);
  unsigned descriptor = block_descriptor(layout, block_index);
  unsigned table_index =
      cursor.coefficient == 0 ? (descriptor >> 2) & 7 : descriptor >> 5;
  // Header preparation resolves every block to a table in this allocation.
  unsigned available_tables = table_count;
  loom::assume(table_index < available_tables);
  Codeword decoded =
      codeword(words, tables + table_index * 2048, limit, cursor);
  cursor = decoded.cursor;
  unsigned symbol = decoded.value;
  if (cursor.error == 0) {
    if (cursor.coefficient == 0) {
      Read value =
          amplitude(decoded.lookahead, decoded.length, symbol, limit, cursor);
      cursor = value.cursor;
      if (cursor.error == 0) {
        unsigned component = descriptor & 3;
        cursor.predictors[component] += static_cast<int>(value.value);
        cursor.coefficient = 1;
        output_index = 0;
        output_value = cursor.predictors[component];
      }
    } else {
      unsigned category = symbol & 15;
      unsigned run = symbol >> 4;
      if (category == 0) {
        if (run == 0) {
          cursor.coefficient = 0;
        } else {
          unsigned next = cursor.coefficient + 16;
          if (next > 64) {
            cursor.error = 1;
          }
          cursor.coefficient = next & 63;
        }
      } else {
        unsigned index = cursor.coefficient + run;
        if (index >= 64) {
          cursor.error = 1;
        } else {
          Read value = amplitude(decoded.lookahead, decoded.length, category,
                                 limit, cursor);
          cursor = value.cursor;
          cursor.coefficient = (index + 1) & 63;
          if (cursor.error == 0) {
            output_index = index;
            output_value = static_cast<int>(value.value);
          }
        }
      }
      if (cursor.error == 0 && cursor.coefficient == 0) {
        ++cursor.blocks;
        ++cursor.block_index;
        if (cursor.block_index == layout.count) {
          cursor.block_index = 0;
        }
      }
    }
  }
  return {cursor, output_index, output_value};
}

// The fast admission proves the entire symbol is present. The original
// transition owns long codes and every partial-consumption/error outcome.
static LOOM_FORCE_INLINE Symbol advance_symbol(const unsigned* words,
                                               const unsigned* tables,
                                               BlockLayout layout,
                                               unsigned limit, Cursor cursor) {
  unsigned block_index = cursor.block_index;
  unsigned block_count = layout.count;
  loom::assume(block_index < block_count);
  unsigned descriptor = block_descriptor(layout, block_index);
  unsigned dc_mask = 0u - static_cast<unsigned>(cursor.coefficient == 0);
  unsigned dc_table = (descriptor >> 2) & 7;
  unsigned ac_table = descriptor >> 5;
  unsigned table_index = ac_table ^ ((ac_table ^ dc_table) & dc_mask);
  unsigned available_tables = table_count;
  loom::assume(table_index < available_tables);
  Read input = peek(words, cursor);
  cursor = input.cursor;
  unsigned packed = lookup_huffman(tables + table_index * 2048, input.value);
  unsigned length = packed >> 8;
  unsigned symbol = packed & 255;
  unsigned category = symbol & (15u | dc_mask);
  unsigned consumed = length + category;
  if (length == 0 || consumed > limit - cursor.position ||
      consumed > cursor.budget - cursor.reads) {
    return advance_symbol_slow(words, tables, layout, limit, cursor);
  }
  loom::assume(length <= 16u);
  loom::assume(category <= 11u);
  unsigned amplitude_mask = (1u << category) - 1;
  unsigned value = (input.value >> (32 - consumed)) & amplitude_mask;
  unsigned negative_mask =
      0u - static_cast<unsigned>(value < ((1u << category) >> 1));
  int amplitude =
      static_cast<int>(value + ((0u - amplitude_mask) & negative_mask));
  unsigned output_index = 0xffffffffu;
  int output_value = 0;
  if (cursor.coefficient == 0) {
    unsigned component = descriptor & 3;
    cursor.predictors[component] += amplitude;
    cursor.coefficient = 1;
    output_index = 0;
    output_value = cursor.predictors[component];
  } else {
    unsigned run = symbol >> 4;
    if (category == 0) {
      unsigned nonzero_run_mask = 0u - static_cast<unsigned>(run != 0);
      unsigned next = (cursor.coefficient + 16) & nonzero_run_mask;
      if (next > 64) {
        cursor.error = 1;
      }
      cursor.coefficient = next & 63;
    } else {
      unsigned index = cursor.coefficient + run;
      if (index >= 64) {
        // An invalid run consumes its codeword, but never its amplitude.
        consumed = length;
        cursor.error = 1;
      } else {
        cursor.coefficient = (index + 1) & 63;
        output_index = index;
        output_value = amplitude;
      }
    }
    if (cursor.coefficient == 0 && cursor.error == 0) {
      ++cursor.blocks;
      ++cursor.block_index;
      unsigned wrap_mask =
          0u - static_cast<unsigned>(cursor.block_index != layout.count);
      cursor.block_index &= wrap_mask;
    }
  }
  cursor.position += consumed;
  cursor.reads += consumed;
  return {cursor, output_index, output_value};
}

static LOOM_FORCE_INLINE Cursor advance(const unsigned* words,
                                        const unsigned* tables,
                                        BlockLayout layout, unsigned limit,
                                        Cursor cursor) {
  return advance_symbol(words, tables, layout, limit, cursor).cursor;
}

// The restart-rooted certificate proves every complete symbol before its
// exclusive frontier. Absolute DC bounds remain the output consumer's job.
static LOOM_FORCE_INLINE Symbol advance_symbol_certified(const unsigned* words,
                                                         const unsigned* tables,
                                                         BlockLayout layout,
                                                         Cursor cursor) {
  unsigned block_index = cursor.block_index;
  unsigned block_count = layout.count;
  loom::assume(block_index < block_count);
  unsigned descriptor = block_descriptor(layout, block_index);
  unsigned dc_mask = 0u - static_cast<unsigned>(cursor.coefficient == 0);
  unsigned dc_table = (descriptor >> 2) & 7;
  unsigned ac_table = descriptor >> 5;
  unsigned table_index = ac_table ^ ((ac_table ^ dc_table) & dc_mask);
  unsigned available_tables = table_count;
  loom::assume(table_index < available_tables);
  const unsigned* table = tables + table_index * 2048;
  Read input = peek(words, cursor);
  cursor = input.cursor;
  unsigned packed = lookup_huffman(table, input.value);
  unsigned length = packed >> 8;
  unsigned symbol = packed & 255;

  loom::assume(length != 0 && length <= 16u);
  unsigned category = symbol & (15u | dc_mask);
  loom::assume(category <= 11u);
  unsigned consumed = length + category;
  unsigned amplitude_mask = (1u << category) - 1;
  unsigned value = (input.value >> (32 - consumed)) & amplitude_mask;
  unsigned negative_mask =
      0u - static_cast<unsigned>(value < ((1u << category) >> 1));
  int amplitude =
      static_cast<int>(value + ((0u - amplitude_mask) & negative_mask));
  unsigned output_index = 0xffffffffu;
  int output_value = 0;
  if (cursor.coefficient == 0) {
    unsigned component = descriptor & 3;
    cursor.predictors[component] += amplitude;
    cursor.coefficient = 1;
    output_index = 0;
    output_value = cursor.predictors[component];
  } else {
    unsigned run = symbol >> 4;
    if (category == 0) {
      unsigned nonzero_run_mask = 0u - static_cast<unsigned>(run != 0);
      cursor.coefficient = ((cursor.coefficient + 16) & nonzero_run_mask) & 63;
    } else {
      unsigned index = cursor.coefficient + run;
      loom::assume(index < 64u);
      cursor.coefficient = (index + 1) & 63;
      output_index = index;
      output_value = amplitude;
    }
    if (cursor.coefficient == 0) {
      ++cursor.blocks;
      ++cursor.block_index;
      unsigned wrap_mask =
          0u - static_cast<unsigned>(cursor.block_index != layout.count);
      cursor.block_index &= wrap_mask;
    }
  }
  cursor.position += consumed;
  cursor.reads += consumed;
  return {cursor, output_index, output_value};
}
