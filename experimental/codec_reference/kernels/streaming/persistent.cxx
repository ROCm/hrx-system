
#include <loomcxx/scalar.h>

#include "mixed_cursor.h"
[[loom::config("input.entropy_padding_words"),
  loom::where(loom::predicate::range(2u, 1024u))]]
extern const unsigned entropy_padding_words;
[[loom::config("jpeg.output_partitions"),
  loom::where(loom::predicate::range(2u, 8u))]]
extern const unsigned output_partitions;

static loom::kernel::configuration configure_chunks(const unsigned* input,
                                                    const unsigned* output) {
  const unsigned* schedule = output + input[24] / 4;
  unsigned x = schedule[1];
  unsigned y = schedule[2];
  unsigned z = schedule[3];
  // Device job planning uses at most the configured chunk capacity and256
  // images; its grid fits the target's signed index domain.
  loom::assume(x <= 1048576u);
  loom::assume(y <= 256u);
  loom::assume(z == 1u);
  return {{x, y, z}, {loom::target::subgroup::size() * 4, 1, 1}};
}

static LOOM_FORCE_INLINE void jpeg_speculate_body(
    const unsigned* entropy, const unsigned* jobs, const unsigned* tables,
    const unsigned* blocks, unsigned* summaries, unsigned* points,
    const unsigned* schedule, const Image* headers) {
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned threads = loom::kernel::workgroup::size.x;
  if (cache_tables) {
    auto* table_storage =
        loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 16>(
            table_count * 2048);
    for (unsigned index = lane; index < table_count * 2048; index += threads) {
      table_storage[index] = tables[index];
    }
    loom::kernel::workgroup::barrier();
    tables = table_storage;
  }
  unsigned job =
      loom::kernel::workgroup::id.x * loom::kernel::workgroup::size.x +
      loom::kernel::workitem::id.x;
  if (job >= schedule[0]) {
    return;
  }
  unsigned image = jobs[job * 4 + 3];
  BlockLayout layout = reinterpret_cast<const BlockLayout*>(blocks)[image];
  tables += headers[image].table_word_offset;

  unsigned offset = jobs[job * 4];
  unsigned limit = jobs[job * 4 + 1];
  unsigned interval = jobs[job * 4 + 2];
  unsigned chunk_bits = chunk_bytes * 8;
  unsigned first_boundary = interval * chunk_bits;
  unsigned last_boundary = first_boundary + chunk_bits;
  const unsigned* words = entropy + offset;
  Cursor cursor = {};
  cursor.position = interval ? first_boundary - chunk_bits : 0;
  cursor.budget = chunk_bits * 4 + 26;
  cursor.cached_word = 0xffffffffu;
  while (cursor.position < first_boundary && cursor.error == 0) {
    unsigned before = cursor.position;
    cursor = advance(words, tables, layout, limit, cursor);
    if (cursor.error == 1) {
      cursor.position = before + 1;
      cursor.blocks = 0;
      cursor.block_index = 0;
      cursor.coefficient = 0;
      cursor.predictors = {};
      cursor.error = 0;
    }
  }
  unsigned first_bit = cursor.position;
  unsigned first_phase = cursor.block_index * 64 + cursor.coefficient;
  unsigned first_blocks = cursor.blocks;
  Predictors first_predictors = cursor.predictors;
  unsigned point_stride = (3 + component_count + 3) & ~3u;
  unsigned* job_points = points + job * (output_partitions - 1) * point_stride;
  for (unsigned partition = 1; partition < output_partitions; ++partition) {
    unsigned boundary =
        first_boundary + chunk_bits * partition / output_partitions;
    while (cursor.position < boundary && cursor.error == 0) {
      cursor = advance(words, tables, layout, limit, cursor);
    }
    unsigned* point = job_points + (partition - 1) * point_stride;
    [[loom::unroll]]
    for (unsigned field = 0; field < point_stride; ++field) {
      point[field] = 0;
    }
    point[0] = cursor.position;
    point[1] = cursor.blocks - first_blocks;
    point[2] = cursor.block_index * 64 + cursor.coefficient;
    [[loom::unroll]]
    for (unsigned component = 0; component < component_count; ++component) {
      point[3 + component] =
          cursor.predictors[component] - first_predictors[component];
    }
  }
  while (cursor.position < last_boundary && cursor.error == 0) {
    cursor = advance(words, tables, layout, limit, cursor);
  }
  unsigned stride = (5 + component_count + 3) & ~3u;
  unsigned* output = summaries + job * stride;
  [[loom::unroll]]
  for (unsigned index = 0; index < stride; ++index) {
    output[index] = 0;
  }
  output[0] = 0xffffffffu;
  if (cursor.error == 0) {
    output[0] = first_bit;
    output[1] = first_phase;
    output[2] = cursor.position;
    output[3] = cursor.block_index * 64 + cursor.coefficient;
    output[4] = cursor.blocks - first_blocks;
    [[loom::unroll]]
    for (unsigned component = 0; component < component_count; ++component) {
      output[5 + component] =
          cursor.predictors[component] - first_predictors[component];
    }
  } else {
    // Invalid speculative summaries never publish partial subdivision facts.
    for (unsigned partition = 0; partition + 1 < output_partitions;
         ++partition) {
      [[loom::unroll]]
      for (unsigned field = 0; field < point_stride; ++field) {
        job_points[partition * point_stride + field] = 0;
      }
    }
  }
}

static loom::kernel::configuration configure_bridges(const unsigned* input,
                                                     const unsigned* output) {
  const unsigned* schedule = output + input[24] / 4;
  unsigned x = schedule[1];
  unsigned y = schedule[2];
  unsigned z = schedule[3];
  // Device job planning uses at most the configured chunk capacity and256
  // images; its grid fits the target's signed index domain.
  loom::assume(x <= 1048576u);
  loom::assume(y <= 256u);
  loom::assume(z == 1u);
  return {{x, y, z}, {loom::target::subgroup::size() * 4, 1, 1}};
}

// A record is a later summary index, completed-block delta, and DC deltas.
// An all-ones target means no local connection; only the restart-rooted path
// may certify a record. No absolute predictor or entropy checkpoint is supplied
// by the host. Every candidate trajectory has its own fixed read cap.
static LOOM_FORCE_INLINE void jpeg_bridge_body(
    const unsigned* entropy, const unsigned* jobs, const unsigned* tables,
    const unsigned* blocks, const unsigned* summaries, unsigned* bridges,
    const unsigned* schedule, const Image* headers) {
  unsigned job =
      loom::kernel::workgroup::id.x * loom::kernel::workgroup::size.x +
      loom::kernel::workitem::id.x;
  if (job >= schedule[0]) {
    return;
  }
  unsigned image = jobs[job * 4 + 3];
  BlockLayout layout = reinterpret_cast<const BlockLayout*>(blocks)[image];
  tables += headers[image].table_word_offset;

  unsigned stride = (2 + component_count + 3) & ~3u;
  unsigned* output = bridges + job * stride;
  [[loom::unroll]]
  for (unsigned index = 0; index < stride; ++index) {
    output[index] = 0;
  }
  output[0] = 0xffffffffu;
  unsigned interval = jobs[job * 4 + 2];
  unsigned first_job = job - interval;
  unsigned summary_stride = (5 + component_count + 3) & ~3u;
  unsigned start_bit = 0;
  unsigned start_phase = 0;
  bool active = interval == 0;
  if (interval != 0) {
    const unsigned* previous = summaries + (job - 1) * summary_stride;
    start_bit = previous[2];
    start_phase = previous[3];
    const unsigned* current = summaries + job * summary_stride;
    active = previous[0] != 0xffffffffu &&
             (current[0] != start_bit || current[1] != start_phase);
  }
  if (!active) {
    return;
  }
  unsigned bit_limit = jobs[job * 4 + 1];
  unsigned chunk_bits = chunk_bytes * 8;
  unsigned intervals = bit_limit / chunk_bits + (bit_limit % chunk_bits != 0);
  const unsigned* words = entropy + jobs[job * 4];
  Cursor cursor = {};
  cursor.position = start_bit;
  cursor.block_index = start_phase >> 6;
  cursor.blocks = cursor.block_index;
  unsigned first_blocks = cursor.blocks;
  cursor.coefficient = start_phase & 63;
  cursor.budget = chunk_bits * 4 + 26;
  cursor.cached_word = 0xffffffffu;
  unsigned target = 0xffffffffu;
  while (cursor.error == 0 && target == 0xffffffffu) {
    // A saved entry can overshoot its fixed boundary by at most 26 bits.
    // Tiny chunks may have several entries at one symbol; choose the first.
    unsigned lower =
        cursor.position > 26 ? (cursor.position - 26) / chunk_bits : 0;
    if (cursor.position - lower * chunk_bits > 26) {
      ++lower;
    }
    if (lower < interval) {
      lower = interval;
    }
    unsigned upper = cursor.position / chunk_bits;
    for (unsigned index = lower;
         index <= upper && index < intervals && target == 0xffffffffu;
         ++index) {
      const unsigned* candidate =
          summaries + (first_job + index) * summary_stride;
      if (candidate[0] == cursor.position &&
          candidate[1] == cursor.block_index * 64 + cursor.coefficient) {
        target = index;
      }
    }
    if (target != 0xffffffffu ||
        cursor.position - start_bit >= chunk_bits * 4) {
      break;
    }
    cursor = advance(words, tables, layout, bit_limit, cursor);
  }
  if (target != 0xffffffffu) {
    output[0] = target;
    output[1] = cursor.blocks - first_blocks;
    [[loom::unroll]]
    for (unsigned component = 0; component < component_count; ++component) {
      output[2 + component] = cursor.predictors[component];
    }
  }
}

[[loom::config("jpeg.segments"), loom::where(loom::predicate::range(1u, 256u))]]
extern const unsigned certification_segments;

using Sums = unsigned __attribute__((ext_vector_type(4)));
using loom::kernel::scan_direction;
using loom::kernel::scan_mode;

static loom::kernel::configuration configure_segment_groups() {
  return {{certification_segments, 1, 1},
          {loom::target::subgroup::size() * 4, 1, 1}};
}

static loom::kernel::configuration configure_segment_roots() {
  unsigned threads = loom::target::subgroup::size();
  return {{(certification_segments + threads - 1) / threads, 1, 1},
          {threads, 1, 1}};
}

// Suffix records retain sequence limits and additive block/DC totals. The
// difference between two suffixes describes a contiguous summary range; it
// does not certify that range as part of the actual JPEG trajectory.
static LOOM_FORCE_INLINE void jpeg_index_body(const unsigned* segments,
                                              const unsigned* summaries,
                                              unsigned* suffixes,
                                              unsigned* ranges,
                                              unsigned* intervals) {
  unsigned segment = loom::kernel::workgroup::id.x;
  unsigned item = loom::kernel::workitem::id.x;
  unsigned threads = loom::kernel::workgroup::size.x;
  const unsigned* descriptor = segments + segment * 8;
  unsigned first_job = descriptor[2];
  unsigned count = descriptor[3];
  unsigned first_record = descriptor[5];
  unsigned summary_stride = (5 + component_count + 3) & ~3u;
  unsigned suffix_stride = (2 + component_count + 3) & ~3u;
  unsigned interval_stride = (4 + component_count + 3) & ~3u;
  unsigned* suffix = suffixes + first_record * suffix_stride;
  unsigned* range = ranges + first_record * suffix_stride;
  unsigned* output =
      intervals +
      ((output_partitions + 1) * first_job + segment) * interval_stride;
  if (item == 0) {
    [[loom::unroll]]
    for (unsigned field = 0; field < suffix_stride; ++field) {
      suffix[count * suffix_stride + field] = 0;
      range[count * suffix_stride + field] = 0;
    }
    suffix[count * suffix_stride] = count;
    range[count * suffix_stride] = 0xffffffffu;
    [[loom::unroll]]
    for (unsigned field = 0; field < interval_stride; ++field) {
      output[(output_partitions + 1) * count * interval_stride + field] = 0;
    }
  }
  unsigned end_carry = count;
  unsigned block_carry = 0;
  Sums dc_carry = {};
  for (unsigned tiles = (count + threads - 1) / threads; tiles != 0; --tiles) {
    unsigned index = (tiles - 1) * threads + item;
    unsigned end = count;
    unsigned blocks = 0;
    Sums dc = {};
    if (index < count) {
      const unsigned* summary =
          summaries + (first_job + index) * summary_stride;
      if (summary[0] != 0xffffffffu) {
        blocks = summary[4];
        [[loom::unroll]]
        for (unsigned component = 0; component < component_count; ++component) {
          dc[component] = summary[5 + component];
        }
      }
      bool connected = false;
      if (index + 1 < count) {
        const unsigned* next = summary + summary_stride;
        connected = summary[0] != 0xffffffffu && next[0] == summary[2] &&
                    next[1] == summary[3];
      }
      if (!connected) {
        end = index + 1;
      }
    }
    unsigned local_end =
        loom::kernel::workgroup::scan::minui<scan_mode::inclusive,
                                             scan_direction::reverse>(end);
    unsigned local_blocks =
        loom::kernel::workgroup::scan::addi<scan_mode::inclusive,
                                            scan_direction::reverse>(blocks);
    Sums local_dc =
        loom::kernel::workgroup::scan::addi<scan_mode::inclusive,
                                            scan_direction::reverse>(dc);
    if (index < count) {
      [[loom::unroll]]
      for (unsigned field = 0; field < suffix_stride; ++field) {
        suffix[index * suffix_stride + field] = 0;
        range[index * suffix_stride + field] = 0;
      }
      suffix[index * suffix_stride] = minimum(end_carry, local_end);
      suffix[index * suffix_stride + 1] = block_carry + local_blocks;
      range[index * suffix_stride] = 0xffffffffu;
      [[loom::unroll]]
      for (unsigned component = 0; component < component_count; ++component) {
        suffix[index * suffix_stride + 2 + component] =
            dc_carry[component] + local_dc[component];
      }
      [[loom::unroll]]
      for (unsigned field = 0; field < interval_stride; ++field) {
        for (unsigned partition = 0; partition <= output_partitions;
             ++partition) {
          output[(partition * count + index) * interval_stride + field] = 0;
        }
      }
    }
    end_carry = minimum(end_carry, loom::kernel::workgroup::reduce::minui(end));
    block_carry += loom::kernel::workgroup::reduce::addi(blocks);
    dc_carry += loom::kernel::workgroup::reduce::addi(dc);
  }
}

static LOOM_FORCE_INLINE void write_interval(unsigned* output, unsigned first,
                                             unsigned last, unsigned blocks,
                                             unsigned phase, Sums dc) {
  output[0] = first;
  output[1] = last;
  output[2] = blocks;
  output[3] = phase;
  [[loom::unroll]]
  for (unsigned component = 0; component < component_count; ++component) {
    output[4 + component] = dc[component];
  }
}

// Only this restart-rooted traversal certifies local transitions. It consumes
// indexed sequence ends and additive totals, never compressed entropy bytes.
static LOOM_FORCE_INLINE void jpeg_certify_body(const unsigned* segments,
                                                const unsigned* summaries,
                                                const unsigned* bridges,
                                                const unsigned* suffixes,
                                                unsigned* ranges,
                                                unsigned* intervals) {
  unsigned segment =
      loom::kernel::workgroup::id.x * loom::kernel::workgroup::size.x +
      loom::kernel::workitem::id.x;
  if (segment >= certification_segments) {
    return;
  }
  const unsigned* descriptor = segments + segment * 8;
  unsigned first_job = descriptor[2];
  unsigned count = descriptor[3];
  unsigned total_blocks = descriptor[4];
  unsigned first_record = descriptor[5];
  unsigned summary_stride = (5 + component_count + 3) & ~3u;
  unsigned stride = (2 + component_count + 3) & ~3u;
  unsigned interval_stride = (4 + component_count + 3) & ~3u;
  const unsigned* suffix = suffixes + first_record * stride;
  unsigned* range = ranges + first_record * stride;
  unsigned* output =
      intervals +
      ((output_partitions + 1) * first_job + segment) * interval_stride;
  unsigned position = 0;
  unsigned blocks = 0;
  unsigned phase = 0;
  Sums dc = {};
  unsigned source = 0;
  while (source < count) {
    const unsigned* bridge = bridges + (first_job + source) * stride;
    unsigned target = bridge[0];
    bool joined = target != 0xffffffffu;
    bool fits = bridge[1] < total_blocks - blocks;
    if (joined & fits) {
      const unsigned* entry = summaries + (first_job + target) * summary_stride;
      write_interval(
          output + (output_partitions * count + source) * interval_stride,
          position, entry[0], blocks, phase, dc);
      position = entry[0];
      blocks += bridge[1];
      phase = entry[1];
      [[loom::unroll]]
      for (unsigned component = 0; component < component_count; ++component) {
        dc[component] += bridge[2 + component];
      }
      unsigned sequence_end = suffix[target * stride];
      unsigned limit = sequence_end;
      unsigned base_blocks = suffix[target * stride + 1];
      if (base_blocks - suffix[limit * stride + 1] >= total_blocks - blocks) {
        unsigned lower = target;
        unsigned upper = sequence_end;
        while (lower < upper) {
          unsigned middle = lower + (upper - lower + 1) / 2;
          if (base_blocks - suffix[middle * stride + 1] <
              total_blocks - blocks) {
            lower = middle;
          } else {
            upper = middle - 1;
          }
        }
        limit = lower;
      }
      if (limit != target) {
        range[target * stride] = limit;
        range[target * stride + 1] = blocks;
        [[loom::unroll]]
        for (unsigned component = 0; component < component_count; ++component) {
          range[target * stride + 2 + component] = dc[component];
          dc[component] += suffix[target * stride + 2 + component] -
                           suffix[limit * stride + 2 + component];
        }
        const unsigned* last =
            summaries + (first_job + limit - 1) * summary_stride;
        position = last[2];
        phase = last[3];
        blocks += base_blocks - suffix[limit * stride + 1];
      }
      source = limit == sequence_end ? sequence_end : count;
    } else {
      source = count;
    }
  }
  // The output consumer owns the exact final MCU and padding validation.
  write_interval(output + (output_partitions + 1) * count * interval_stride,
                 position, descriptor[1], blocks, phase, dc);
}

static LOOM_FORCE_INLINE void jpeg_intervals_body(const unsigned* segments,
                                                  const unsigned* summaries,
                                                  const unsigned* suffixes,
                                                  const unsigned* ranges,
                                                  const unsigned* points,
                                                  unsigned* intervals) {
  unsigned segment = loom::kernel::workgroup::id.x;
  unsigned item = loom::kernel::workitem::id.x;
  unsigned threads = loom::kernel::workgroup::size.x;
  const unsigned* descriptor = segments + segment * 8;
  unsigned first_job = descriptor[2];
  unsigned count = descriptor[3];
  unsigned first_record = descriptor[5];
  unsigned summary_stride = (5 + component_count + 3) & ~3u;
  unsigned stride = (2 + component_count + 3) & ~3u;
  unsigned interval_stride = (4 + component_count + 3) & ~3u;
  const unsigned* suffix = suffixes + first_record * stride;
  const unsigned* range = ranges + first_record * stride;
  unsigned* output =
      intervals +
      ((output_partitions + 1) * first_job + segment) * interval_stride;
  unsigned carry = 0;
  for (unsigned begin = 0; begin < count; begin += threads) {
    unsigned index = begin + item;
    unsigned start = 0;
    if (index < count && range[index * stride] != 0xffffffffu) {
      start = index + 1;
    }
    unsigned prefix = loom::kernel::workgroup::scan::maxui(start);
    if (prefix < carry) {
      prefix = carry;
    }
    if (index < count && prefix != 0) {
      unsigned target = prefix - 1;
      if (index < range[target * stride]) {
        const unsigned* summary =
            summaries + (first_job + index) * summary_stride;
        unsigned blocks = range[target * stride + 1] +
                          suffix[target * stride + 1] -
                          suffix[index * stride + 1];
        Sums dc = {};
        [[loom::unroll]]
        for (unsigned component = 0; component < component_count; ++component) {
          dc[component] = range[target * stride + 2 + component] +
                          suffix[target * stride + 2 + component] -
                          suffix[index * stride + 2 + component];
        }
        unsigned point_stride = (3 + component_count + 3) & ~3u;
        const unsigned* job_points = points + (first_job + index) *
                                                  (output_partitions - 1) *
                                                  point_stride;
        unsigned position = summary[0];
        unsigned phase = summary[1];
        unsigned absolute_blocks = blocks;
        Sums absolute_dc = dc;
        for (unsigned partition = 0; partition + 1 < output_partitions;
             ++partition) {
          const unsigned* point = job_points + partition * point_stride;
          write_interval(output + (index * output_partitions + partition) *
                                      interval_stride,
                         position, point[0], absolute_blocks, phase,
                         absolute_dc);
          position = point[0];
          absolute_blocks = blocks + point[1];
          phase = point[2];
          [[loom::unroll]]
          for (unsigned component = 0; component < component_count;
               ++component) {
            absolute_dc[component] = dc[component] + point[3 + component];
          }
        }
        write_interval(
            output + (index * output_partitions + output_partitions - 1) *
                         interval_stride,
            position, summary[2], absolute_blocks, phase, absolute_dc);
      }
    }
    unsigned total = loom::kernel::workgroup::reduce::maxui(start);
    if (total > carry) {
      carry = total;
    }
  }
}

using WideValues = float __attribute__((ext_vector_type(8)));

static loom::kernel::configuration configure_output(const unsigned* schedule) {
  unsigned x = schedule[4];
  unsigned y = schedule[5];
  unsigned z = schedule[6];
  // Device job planning uses at most the configured chunk capacity and256
  // images; its grid fits the target's signed index domain.
  loom::assume(x <= 1048576u);
  loom::assume(y <= 256u);
  loom::assume(z == 1u);
  return {{x, y, z}, {loom::target::subgroup::size(), 1, 1}};
}

static LOOM_FORCE_INLINE WideValues transform_block_axis(WideValues input) {
  float sum = input[0] + input[4];
  float difference = input[0] - input[4];
  float rotation = (input[2] + input[6]) * 0x1.1517a80000000p-1f;
  float two = loom::scalar::fmaf(input[6], -0x1.d906bc0000000p+0f, rotation);
  float six = loom::scalar::fmaf(input[2], 0x1.87de2a0000000p-1f, rotation);
  float even0 = sum + six;
  float even1 = difference + two;
  float even2 = difference - two;
  float even3 = sum - six;
  float odd_rotation =
      (input[1] + input[3] + input[5] + input[7]) * 0x1.2d062e0000000p+0f;
  float seven_three = loom::scalar::fmaf(input[7] + input[3],
                                         -0x1.f6297c0000000p+0f, odd_rotation);
  float five_one = loom::scalar::fmaf(input[5] + input[1],
                                      -0x1.8f8b840000000p-2f, odd_rotation);
  float seven_one = (input[7] + input[1]) * -0x1.ccc9ae0000000p-1f;
  float five_three = (input[5] + input[3]) * -0x1.480d9e0000000p+1f;
  float odd0 =
      loom::scalar::fmaf(input[1], 0x1.8056940000000p+0f, seven_one) + five_one;
  float odd1 = loom::scalar::fmaf(input[3], 0x1.894e980000000p+1f, five_three) +
               seven_three;
  float odd2 = loom::scalar::fmaf(input[5], 0x1.06cca20000000p+1f, five_three) +
               five_one;
  float odd3 = loom::scalar::fmaf(input[7], 0x1.31cc6a0000000p-2f, seven_one) +
               seven_three;
  return {even0 + odd0, even1 + odd1, even2 + odd2, even3 + odd3,
          even3 - odd3, even2 - odd2, even1 - odd1, even0 - odd0};
}

static LOOM_FORCE_INLINE unsigned sample_byte(int value) {
  unsigned nonnegative = static_cast<unsigned>(value & ~(value >> 31));
  unsigned saturated =
      nonnegative | (0u - static_cast<unsigned>(nonnegative > 255u));
  return saturated & 255u;
}

static LOOM_FORCE_INLINE Symbol
output_symbol(const unsigned* words, const unsigned* tables, BlockLayout layout,
              unsigned limit, unsigned certified_end, Cursor cursor) {
  Symbol symbol = {};
  if (cursor.position < certified_end) {
    symbol = advance_symbol_certified(words, tables, layout, cursor);
  } else {
    symbol = advance_symbol(words, tables, layout, limit, cursor);
  }
  if (symbol.index == 0 && (symbol.value < -1024 || symbol.value > 1023)) {
    symbol.cursor.error = 5;
  }
  return symbol;
}

// The block owner has already consumed DC. Its retained table and the
// restart-rooted certificate establish the entire AC syntax transition.
static LOOM_FORCE_INLINE Symbol advance_ac_certified(const unsigned* words,
                                                     const unsigned* table,
                                                     Cursor cursor,
                                                     BlockLayout layout) {
  Read input = peek(words, cursor);
  cursor = input.cursor;
  unsigned packed = lookup_huffman(table, input.value);
  unsigned length = packed >> 8;
  unsigned symbol = packed & 255;

  loom::assume(length != 0 && length <= 16u);
  unsigned category = symbol & 15u;
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
  cursor.position += consumed;
  cursor.reads += consumed;
  return {cursor, output_index, output_value};
}

using Values = int __attribute__((ext_vector_type(8)));

// Exchange the low three bits of the register index with the lane index.
// Every shuffle is confined to a fully participating eight-lane block.
static LOOM_FORCE_INLINE Values transpose_block(Values values, unsigned axis) {
  [[loom::unroll]]
  for (unsigned stage = 0; stage < 3; ++stage) {
    unsigned mask = 1u << stage;
    Values exchanged =
        loom::kernel::subgroup::shuffle_xor(values, mask, 8).value;
    Values next = {};
    [[loom::unroll]]
    for (unsigned index = 0; index < 8; ++index) {
      int exchange_mask = -static_cast<int>(((axis ^ index) & mask) != 0);
      next[index] = values[index] ^
                    ((values[index] ^ exchanged[index ^ mask]) & exchange_mask);
    }
    values = next;
  }
  return values;
}

// Row-local owner permutation preserves the payload while distributing
// each eight-lane consumer's column across independent LDS banks.
static LOOM_FORCE_INLINE unsigned coefficient_index(unsigned natural,
                                                    unsigned owner,
                                                    unsigned stride) {
  return natural * stride + (owner ^ ((natural & 7u) * 2u));
}

// All lanes enter; each group of eight consumes one producer's block.
static LOOM_FORCE_INLINE void reconstruct_cooperative(
    const short* coefficients, unsigned coefficient_stride, unsigned threads,
    unsigned lane, unsigned owner, bool valid, bool any_ac, unsigned offset,
    const unsigned* quantization, unsigned* pixels) {
  bool owner_valid = loom::kernel::subgroup::shuffle_index(
                         static_cast<unsigned>(valid), owner, threads)
                         .value != 0;
  bool owner_ac = loom::kernel::subgroup::shuffle_index(
                      static_cast<unsigned>(any_ac), owner, threads)
                      .value != 0;
  unsigned owner_offset =
      loom::kernel::subgroup::shuffle_index(offset, owner, threads).value;
  unsigned axis = lane & 7;
  if (owner_valid) {
    unsigned long long packed = 0;
    if (!owner_ac) {
      int dc = static_cast<int>(coefficients[owner]) *
               static_cast<int>(quantization[0]);
      unsigned value = sample_byte(((dc + 4) >> 3) + 128);
      packed = static_cast<unsigned long long>(value) * 0x0101010101010101ull;
    } else {
      WideValues input = {};
      int ac = 0;
      [[loom::unroll]]
      for (unsigned row = 0; row < 8; ++row) {
        unsigned natural = row * 8 + axis;
        unsigned step = quantization[natural];
        loom::assume(step <= 65535u);
        int value = static_cast<int>(coefficients[coefficient_index(
                        natural, owner, coefficient_stride)]) *
                    static_cast<int>(step);
        input[row] = value;
        if (row != 0) {
          ac |= value;
        }
      }
      WideValues columns = {};
      if (ac == 0) {
        [[loom::unroll]]
        for (unsigned row = 0; row < 8; ++row) {
          columns[row] = input[0];
        }
      } else {
        columns = transform_block_axis(input);
      }
      WideValues row = __builtin_bit_cast(
          WideValues,
          transpose_block(__builtin_bit_cast(Values, columns), axis));
      WideValues result = transform_block_axis(row);
      [[loom::unroll]]
      for (unsigned column = 0; column < 8; ++column) {
        int value = static_cast<int>(loom::scalar::floorf(
            loom::scalar::fmaf(result[column], 0.125f, 128.5f)));
        packed |= static_cast<unsigned long long>(sample_byte(value))
                  << (column * 8);
      }
    }
    reinterpret_cast<unsigned long long*>(pixels + owner_offset)[axis] = packed;
  }
}

// A single subgroup owns the workgroup so its active vote controls every
// barrier uniformly. Producer lanes retain bounded interval cursors; consumers
// exchange matrix ownership only while a complete coefficient block is live.
[[loom::kernel(configure_output)]]
void jpeg_reconstruct(const unsigned* entropy, const unsigned* segments,
                      const unsigned* tables, const unsigned* blocks,
                      const unsigned* intervals, const unsigned* geometry,
                      const unsigned* quantization, const unsigned* zigzag,
                      unsigned* pixels, unsigned* status,
                      const Image* headers) {
  unsigned segment = loom::kernel::workgroup::id.y;
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned threads = loom::kernel::workgroup::size.x;
  unsigned stripe = threads * certification_segments;
  unsigned group_count = loom::kernel::workgroup::count.x;
  const unsigned* descriptor = segments + segment * 8;
  Image image = headers[segment];
  BlockLayout layout = reinterpret_cast<const BlockLayout*>(blocks)[segment];
  tables += image.table_word_offset;
  geometry += image.geometry_word_offset;
  quantization += image.quantization_word_offset;

  unsigned count = descriptor[3];
  unsigned total_blocks = descriptor[4];
  unsigned first_interval = descriptor[2] * (output_partitions + 1) + segment;
  unsigned stride = (4 + component_count + 3) & ~3u;
  unsigned certified_end =
      intervals[(first_interval + (output_partitions + 1) * count) * stride];
  unsigned certified_blocks =
      intervals[(first_interval + (output_partitions + 1) * count) * stride +
                2];
  const unsigned* words = entropy + descriptor[0];
  unsigned bit_limit = descriptor[1];
  // The row-local permutation avoids padding the coefficient payload.
  unsigned coefficient_stride = threads;
  auto* coefficients =
      loom::buffer::alloca<short, loom::memory_space::workgroup, 16>(
          coefficient_stride * 64);
  for (unsigned first = loom::kernel::workgroup::id.x * threads;
       first < count * (output_partitions + 1) + 1;
       first += group_count * threads) {
    unsigned interval = first + lane;
    bool present = interval < count * (output_partitions + 1) + 1;
    Cursor cursor = {};
    cursor.budget = bit_limit;
    cursor.cached_word = 0xffffffffu;
    unsigned end = 0;
    if (present) {
      const unsigned* entry = intervals + (first_interval + interval) * stride;
      cursor.position = entry[0];
      end = entry[1];
      cursor.blocks = entry[2];
      cursor.block_index = entry[3] >> 6;
      cursor.coefficient = entry[3] & 63;
      [[loom::unroll]]
      for (unsigned component = 0; component < component_count; ++component) {
        cursor.predictors[component] = static_cast<int>(entry[4 + component]);
      }
      if (cursor.position < end) {
        while ((cursor.block_index != 0 || cursor.coefficient != 0) &&
               cursor.error == 0) {
          cursor = output_symbol(words, tables, layout, bit_limit,
                                 certified_end, cursor)
                       .cursor;
        }
      }
    }
    bool active = present && cursor.blocks < total_blocks &&
                  cursor.position < end && cursor.error == 0;
    while (loom::kernel::subgroup::vote::any(active)) {
      unsigned mcu = descriptor[6] + cursor.blocks / layout.count;
      unsigned mcu_row = mcu / image.mcu_columns;
      unsigned mcu_column = mcu % image.mcu_columns;
      for (unsigned block = 0; block < layout.count; ++block) {
        bool any_ac = false;
        bool output_valid = false;
        unsigned offset = 0;
        unsigned table_index = block_descriptor(layout, block) >> 5;
        unsigned available_tables = table_count;
        loom::assume(table_index < available_tables);
        const unsigned* ac_table = tables + table_index * 2048;
        if (active && cursor.error == 0) {
          [[loom::unroll]]
          for (unsigned index = 0; index < 64; ++index) {
            coefficients[coefficient_index(index, lane, coefficient_stride)] =
                0;
          }
          if (cursor.blocks < certified_blocks) {
            // The root certificate covers this whole block. Syntax is proven;
            // the absolute DC predictor remains a consumer-owned constraint.
            unsigned first_coefficient = cursor.coefficient;
            loom::assume(first_coefficient == 0);
            cursor.coefficient = first_coefficient;
            Symbol dc = advance_symbol_certified(words, tables, layout, cursor);
            cursor = dc.cursor;
            if (dc.value < -1024 || dc.value > 1023) {
              cursor.error = 5;
            }
            if (cursor.error == 0) {
              coefficients[lane] = static_cast<short>(dc.value);
              do {
                Symbol symbol =
                    advance_ac_certified(words, ac_table, cursor, layout);
                cursor = symbol.cursor;
                if (symbol.index != 0xffffffffu) {
                  unsigned natural = zigzag[symbol.index];
                  loom::assume(natural < 64u);
                  coefficients[coefficient_index(natural, lane,
                                                 coefficient_stride)] =
                      static_cast<short>(symbol.value);
                  any_ac = true;
                }
              } while (cursor.coefficient != 0);
            }
          } else {
            do {
              Symbol symbol =
                  advance_symbol(words, tables, layout, bit_limit, cursor);
              cursor = symbol.cursor;
              if (symbol.index == 0 &&
                  (symbol.value < -1024 || symbol.value > 1023)) {
                cursor.error = 5;
              }
              if (symbol.index != 0xffffffffu && cursor.error == 0) {
                unsigned natural = zigzag[symbol.index];
                loom::assume(natural < 64u);
                coefficients[coefficient_index(natural, lane,
                                               coefficient_stride)] =
                    static_cast<short>(symbol.value);
                if (symbol.index != 0) {
                  any_ac = true;
                }
              }
            } while (cursor.coefficient != 0 && cursor.error == 0);
          }
          const unsigned* shape = geometry + block * 8;
          unsigned column = mcu_column * shape[3] + shape[5];
          unsigned row = mcu_row * shape[4] + shape[6];
          output_valid =
              cursor.error == 0 && column < shape[1] && row < shape[2];
          offset = descriptor[7] + shape[0] + (row * shape[1] + column) * 16;
        }
        loom::kernel::barrier<loom::memory_space::workgroup,
                              loom::atomic::scope::subgroup,
                              loom::atomic::ordering::acq_rel>();
        unsigned component = block_descriptor(layout, block) & 3;
        unsigned components = component_count;
        loom::assume(component < components);
        for (unsigned round = 0; round < 8; ++round) {
          unsigned owner = round * (threads / 8) + lane / 8;
          reconstruct_cooperative(coefficients, coefficient_stride, threads,
                                  lane, owner, output_valid, any_ac, offset,
                                  quantization + component * 64, pixels);
        }
        loom::kernel::barrier<loom::memory_space::workgroup,
                              loom::atomic::scope::subgroup,
                              loom::atomic::ordering::acq_rel>();
      }
      active = present && cursor.blocks < total_blocks &&
               cursor.position < end && cursor.error == 0;
    }
    if (present) {
      if (interval == count * (output_partitions + 1) && cursor.error == 0) {
        if (cursor.blocks != total_blocks || cursor.coefficient != 0) {
          cursor.error = 3;
        } else {
          unsigned padding = bit_limit - cursor.position;
          if (padding >= 8) {
            cursor.error = 4;
          } else if (padding != 0) {
            Read input = peek(words, cursor);
            if ((input.value >> (32 - padding)) != (1u << padding) - 1) {
              cursor.error = 4;
            }
          }
        }
      }
      status[first_interval + interval] = cursor.error;
    }
  }
}

#include <loomcxx/buffer.h>
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/vector.h>

struct ColorGeometry {
  // Per-image encoded width, independent of the sampling specialization.
  unsigned width;
  // Per-image encoded height, independent of the sampling specialization.
  unsigned height;
};
using ColorSamples = int __attribute__((ext_vector_type(4)));

LOOM_FORCE_INLINE unsigned color_divide_up(ColorGeometry dimensions,
                                           unsigned value, unsigned divisor) {
  return (value + divisor - 1) / divisor;
}

LOOM_FORCE_INLINE unsigned color_plane_bytes(ColorGeometry dimensions,
                                             unsigned horizontal,
                                             unsigned vertical) {
  return color_divide_up(
             dimensions,
             color_divide_up(dimensions, dimensions.width, horizontal), 8) *
         color_divide_up(
             dimensions,
             color_divide_up(dimensions, dimensions.height, vertical), 8) *
         64;
}

LOOM_FORCE_INLINE unsigned color_address(ColorGeometry dimensions,
                                         unsigned column, unsigned row,
                                         unsigned width) {
  return ((row >> 3) * color_divide_up(dimensions, width, 8) + (column >> 3)) *
             64 +
         (row & 7u) * 8 + (column & 7u);
}

LOOM_FORCE_INLINE unsigned color_edge(ColorGeometry dimensions, int value,
                                      unsigned extent) {
  if (value < 0) {
    return 0;
  }
  unsigned positive = static_cast<unsigned>(value);
  return positive < extent ? positive : extent - 1;
}

LOOM_FORCE_INLINE bool color_horizontal_halo(ColorGeometry dimensions,
                                             unsigned horizontal,
                                             unsigned vertical) {
  unsigned width = color_divide_up(dimensions, dimensions.width, horizontal);
  return horizontal == 2 && (vertical == 1 || vertical == 2) && width > 2;
}

LOOM_FORCE_INLINE bool color_vertical_halo(ColorGeometry dimensions,
                                           unsigned horizontal,
                                           unsigned vertical) {
  return vertical == 2 &&
         (horizontal == 1 ||
          color_horizontal_halo(dimensions, horizontal, vertical));
}

LOOM_FORCE_INLINE unsigned color_cache_columns(ColorGeometry dimensions,
                                               unsigned horizontal,
                                               unsigned vertical) {
  return color_divide_up(dimensions, 64, horizontal) +
         2 * color_horizontal_halo(dimensions, horizontal, vertical);
}

LOOM_FORCE_INLINE unsigned color_cache_rows(ColorGeometry dimensions,
                                            unsigned horizontal,
                                            unsigned vertical) {
  return color_divide_up(dimensions, 16, vertical) +
         2 * color_vertical_halo(dimensions, horizontal, vertical);
}

LOOM_FORCE_INLINE unsigned color_cache_size(ColorGeometry dimensions,
                                            unsigned horizontal,
                                            unsigned vertical) {
  return horizontal == 1 && vertical == 1
             ? 1
             : color_cache_columns(dimensions, horizontal, vertical) *
                   color_cache_rows(dimensions, horizontal, vertical);
}

// One copy of each reduced-plane sample, including only the interpolation halo,
// feeds every four-pixel consumer in this output tile.
LOOM_FORCE_INLINE void color_stage(ColorGeometry dimensions,
                                   const unsigned char* plane, unsigned* cache,
                                   unsigned horizontal, unsigned vertical,
                                   unsigned first_x, unsigned first_y) {
  if (horizontal == 1 && vertical == 1) {
    return;
  }
  unsigned width = color_divide_up(dimensions, dimensions.width, horizontal);
  unsigned height = color_divide_up(dimensions, dimensions.height, vertical);
  unsigned columns = color_cache_columns(dimensions, horizontal, vertical);
  unsigned count = columns * color_cache_rows(dimensions, horizontal, vertical);
  int first_column = static_cast<int>(first_x / horizontal) -
                     color_horizontal_halo(dimensions, horizontal, vertical);
  int first_row = static_cast<int>(first_y / vertical) -
                  color_vertical_halo(dimensions, horizontal, vertical);
  unsigned threads = loom::kernel::workgroup::size.x;
  for (unsigned index = loom::kernel::workitem::id.x; index < count;
       index += threads) {
    unsigned column = color_edge(
        dimensions, first_column + static_cast<int>(index % columns), width);
    unsigned row = color_edge(
        dimensions, first_row + static_cast<int>(index / columns), height);
    cache[index] = plane[color_address(dimensions, column, row, width)];
  }
}

LOOM_FORCE_INLINE ColorSamples
color_samples(ColorGeometry dimensions, const unsigned char* plane,
              const unsigned* cache, unsigned horizontal, unsigned vertical,
              unsigned first_x, unsigned first_y, unsigned x, unsigned y) {
  ColorSamples result = {};
  if (horizontal == 1 && vertical == 1) {
    unsigned address = color_address(dimensions, x, y, dimensions.width);
    unsigned packed = *reinterpret_cast<const unsigned*>(plane + address);
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      result[index] = (packed >> (index * 8)) & 255u;
    }
    return result;
  }
  unsigned columns = color_cache_columns(dimensions, horizontal, vertical);
  unsigned cache_count =
      columns * color_cache_rows(dimensions, horizontal, vertical);
  unsigned column = x / horizontal - first_x / horizontal +
                    color_horizontal_halo(dimensions, horizontal, vertical);
  unsigned row = y / vertical - first_y / vertical +
                 color_vertical_halo(dimensions, horizontal, vertical);
  if (color_horizontal_halo(dimensions, horizontal, vertical)) {
    ColorSamples weighted = {};
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      unsigned position = row * columns + column + index - 1;
      // The four-pixel owner lies inside the tile; staged halos include both
      // neighbors. Retain that geometric bound across global/local division.
      loom::assume(position < cache_count);
      unsigned value = cache[position];
      if (vertical == 2) {
        unsigned neighbor = (y & 1u) ? position + columns : position - columns;
        loom::assume(neighbor < cache_count);
        value = 3 * value + cache[neighbor];
      }
      weighted[index] = value;
    }
    unsigned shift = vertical == 2 ? 4 : 2;
    unsigned even_bias = vertical == 2 ? 8 : 1;
    unsigned odd_bias = vertical == 2 ? 7 : 2;
    result[0] = (3 * weighted[1] + weighted[0] + even_bias) >> shift;
    result[1] = (3 * weighted[1] + weighted[2] + odd_bias) >> shift;
    result[2] = (3 * weighted[2] + weighted[1] + even_bias) >> shift;
    result[3] = (3 * weighted[2] + weighted[3] + odd_bias) >> shift;
  } else {
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      unsigned current_column = (x + index) / horizontal - first_x / horizontal;
      unsigned position = row * columns + current_column;
      loom::assume(position < cache_count);
      unsigned value = cache[position];
      if (color_vertical_halo(dimensions, horizontal, vertical)) {
        unsigned neighbor = (y & 1u) ? position + columns : position - columns;
        loom::assume(neighbor < cache_count);
        value = (3 * value + cache[neighbor] + ((y & 1u) ? 2 : 1)) >> 2;
      }
      result[index] = value;
    }
  }
  return result;
}

LOOM_FORCE_INLINE unsigned color_clamp(ColorGeometry dimensions, int value) {
  unsigned positive = static_cast<unsigned>(value & ~(value >> 31));
  return (positive | (0u - static_cast<unsigned>(positive > 255u))) & 255u;
}

template <unsigned image_color_space, unsigned horizontal0, unsigned vertical0,
          unsigned horizontal1, unsigned vertical1, unsigned horizontal2,
          unsigned vertical2>
static LOOM_FORCE_INLINE void color_run(const unsigned char* components,
                                        unsigned char* rgb, Image image,
                                        const unsigned* tile, unsigned* cache) {
  ColorGeometry dimensions{image.width, image.height};
  unsigned bytes0 = image.first_plane_bytes;
  unsigned bytes1 = image.second_plane_bytes;
  const unsigned char* plane0 = components + image.component_word_offset * 4;
  const unsigned char* plane1 = plane0 + bytes0;
  const unsigned char* plane2 = plane1 + bytes1;
  unsigned* cache0 = cache;
  unsigned* cache1 = cache;
  unsigned* cache2 = cache + 640;
  unsigned first_x = tile[1];
  unsigned first_y = tile[2];
  color_stage(dimensions, plane0, cache0, horizontal0, vertical0, first_x,
              first_y);
  if (image_color_space != 1) {
    color_stage(dimensions, plane1, cache1, horizontal1, vertical1, first_x,
                first_y);
    color_stage(dimensions, plane2, cache2, horizontal2, vertical2, first_x,
                first_y);
  }
  loom::kernel::workgroup::barrier();
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned x = first_x + (lane & 15u) * 4;
  unsigned y = first_y + (lane >> 4);
  if (x >= dimensions.width || y >= dimensions.height) {
    return;
  }
  ColorSamples values0 = color_samples(dimensions, plane0, cache0, horizontal0,
                                       vertical0, first_x, first_y, x, y);
  ColorSamples values1 = {};
  ColorSamples values2 = {};
  if (image_color_space != 1) {
    values1 = color_samples(dimensions, plane1, cache1, horizontal1, vertical1,
                            first_x, first_y, x, y);
    values2 = color_samples(dimensions, plane2, cache2, horizontal2, vertical2,
                            first_x, first_y, x, y);
  }
  ColorSamples red = values0;
  ColorSamples green = image_color_space == 1 ? values0 : values1;
  ColorSamples blue = image_color_space == 1 ? values0 : values2;
  if (image_color_space == 3) {
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      int cb = values1[index] - 128;
      int cr = values2[index] - 128;
      red[index] = values0[index] + ((91881 * cr + 32768) >> 16);
      green[index] =
          values0[index] + ((-22554 * cb - 46802 * cr + 32768) >> 16);
      blue[index] = values0[index] + ((116130 * cb + 32768) >> 16);
    }
  }
  ColorSamples packed = {};
  [[loom::unroll]]
  for (unsigned index = 0; index < 4; ++index) {
    packed[index] = color_clamp(dimensions, red[index]) |
                    (color_clamp(dimensions, green[index]) << 8) |
                    (color_clamp(dimensions, blue[index]) << 16);
  }
  unsigned offset = image.rgb_byte_offset + (y * dimensions.width + x) * 3;
  if ((dimensions.width & 3u) == 0) {
    unsigned* output = reinterpret_cast<unsigned*>(rgb + offset);
    using Words = unsigned __attribute__((ext_vector_type(4)));
    using Triple = loom::type::vector<loom::type::shape<3>, unsigned>;
    Words words = {static_cast<unsigned>(packed[0]) |
                       (static_cast<unsigned>(packed[1]) << 24),
                   (static_cast<unsigned>(packed[1]) >> 8) |
                       (static_cast<unsigned>(packed[2]) << 16),
                   (static_cast<unsigned>(packed[2]) >> 16) |
                       (static_cast<unsigned>(packed[3]) << 8),
                   0};
    auto destination =
        loom::buffer::view<3>(output, {}, loom::encoding::layout::dense<1>());
    loom::vector::store(loom::vector::slice<Triple>(words, 0), destination,
                        {0});
  } else {
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      if (x + index < dimensions.width) {
        rgb[offset + index * 3] = static_cast<unsigned char>(packed[index]);
        rgb[offset + index * 3 + 1] = static_cast<unsigned char>(
            static_cast<unsigned>(packed[index]) >> 8);
        rgb[offset + index * 3 + 2] = static_cast<unsigned char>(
            static_cast<unsigned>(packed[index]) >> 16);
      }
    }
  }
}

static loom::kernel::configuration configure_color(
    const unsigned* color_schedule) {
  unsigned count = color_schedule[0];
  loom::assume(count <= 1048576u);
  return {{count, 1, 1}, {256, 1, 1}};
}

[[loom::kernel(configure_color)]]
void jpeg_color(const unsigned char* components, unsigned char* rgb,
                const Image* headers, const unsigned* color_tiles,
                const unsigned* validity) {
  const unsigned* tile = color_tiles + loom::kernel::workgroup::id.x * 3;
  unsigned ordinal = tile[0];
  if (validity[ordinal] != 0) {
    return;
  }
  Image image = headers[ordinal];
  // At most two reduced planes are live. The largest supported halo is
  // 64 columns by ten rows; mutually exclusive sampling cases reuse it.
  auto* cache =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 16>(1280);
  if (image.color_kind == 1) {
    color_run<1, 1, 1, 1, 1, 1, 1>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 2) {
    color_run<2, 1, 1, 1, 1, 1, 1>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 3) {
    color_run<3, 1, 1, 1, 1, 1, 1>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 4) {
    color_run<3, 1, 1, 2, 1, 2, 1>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 5) {
    color_run<3, 1, 1, 2, 2, 2, 2>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 6) {
    color_run<3, 1, 1, 1, 2, 1, 2>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 7) {
    color_run<3, 1, 1, 4, 1, 4, 1>(components, rgb, image, tile, cache);
  }
  if (image.color_kind == 8) {
    color_run<3, 1, 1, 4, 2, 4, 2>(components, rgb, image, tile, cache);
  }
}

[[loom::config("input.tiles"), loom::where(loom::predicate::range(1u, 1024u))]]
extern const unsigned tile_count;

using PacketWords = unsigned __attribute__((ext_vector_type(4)));

struct Packet {
  // Original little-endian bytes in one aligned sixteen-byte input packet.
  PacketWords words;
  // Bit i retains byte i after removing stuffing and marker prefixes.
  unsigned keep;
  // First marker/fill prefix, or all ones when absent.
  unsigned marker_prefix;
  // First FF followed by a marker code, or all ones when absent.
  unsigned marker_code;
  // First fill-before-stuffed-zero error, or all ones when absent.
  unsigned invalid;
};

static LOOM_FORCE_INLINE unsigned byte_minimum(unsigned a, unsigned b) {
  return a < b ? a : b;
}

static LOOM_FORCE_INLINE Packet classify(const unsigned* source, unsigned first,
                                         unsigned length, unsigned start) {
  PacketWords words = *reinterpret_cast<const PacketWords*>(source + first / 4);
  unsigned previous = first ? source[first / 4 - 1] : 0;
  unsigned following = source[first / 4 + 4];
  unsigned keep = 0;
  unsigned prefix = 0xffffffffu;
  unsigned code = 0xffffffffu;
  unsigned invalid = 0xffffffffu;
  [[loom::unroll]]
  for (unsigned index = 0; index < 16; ++index) {
    unsigned position = first + index;
    unsigned value = (words[index / 4] >> ((index & 3u) * 8)) & 255u;
    unsigned before =
        index ? (words[(index - 1) / 4] >> (((index - 1) & 3u) * 8)) & 255u
              : previous >> 24;
    unsigned before_previous =
        index >= 2 ? (words[(index - 2) / 4] >> (((index - 2) & 3u) * 8)) & 255u
                   : (previous >> ((index + 2) * 8)) & 255u;
    unsigned after =
        index < 15 ? (words[(index + 1) / 4] >> (((index + 1) & 3u) * 8)) & 255u
                   : following & 255u;
    if (position >= start && position < length) {
      if (position + 1 == length) {
        after = 255;
      }
      if (value == 255) {
        if (after == 0) {
          keep |= 1u << index;
        } else {
          prefix = byte_minimum(prefix, position);
          if (after != 255) {
            code = byte_minimum(code, position);
          }
        }
      } else if (value == 0 && position > start && before == 255) {
        if (position > start + 1 && before_previous == 255) {
          invalid = byte_minimum(invalid, position);
        }
      } else {
        keep |= 1u << index;
      }
    }
  }
  return {words, keep, prefix, code, invalid};
}

static LOOM_FORCE_INLINE unsigned before_marker(unsigned mask, unsigned first,
                                                unsigned marker) {
  if (marker <= first) {
    return 0;
  }
  unsigned available = marker - first;
  if (available < 16) {
    mask &= (1u << available) - 1;
  }
  return mask;
}

static loom::kernel::configuration configure_tiles() {
  return {{tile_count, certification_segments, 1}, {256, 1, 1}};
}

static loom::kernel::configuration configure_images() {
  return {{certification_segments, 1, 1}, {256, 1, 1}};
}

// Headers contain original byte length, SOS payload end, MCU block count, and
// destination component-word base. The input allocation has one guard packet.
static LOOM_FORCE_INLINE void jpeg_classify_bytes_body(
    const unsigned* compressed, const Image* headers, unsigned* records) {
  unsigned image = loom::kernel::workgroup::id.y;
  unsigned tile = loom::kernel::workgroup::id.x;
  if (tile >= headers[image].tile_count) {
    return;
  }
  unsigned lane = loom::kernel::workitem::id.x;
  const unsigned* source = compressed + headers[image].compressed_word_offset;
  unsigned first = tile * 4096 + lane * 16;
  Packet packet = classify(source, first, headers[image].compressed_bytes,
                           headers[image].entropy_start);
  unsigned prefix =
      loom::kernel::workgroup::reduce::minui(packet.marker_prefix);
  unsigned code = loom::kernel::workgroup::reduce::minui(packet.marker_code);
  unsigned invalid = loom::kernel::workgroup::reduce::minui(packet.invalid);
  unsigned mask = before_marker(packet.keep, first, prefix);
  unsigned count =
      loom::kernel::workgroup::reduce::addi(loom::scalar::ctpopi(mask));
  if (lane == 0) {
    unsigned* record = records + (headers[image].tile_offset + tile) * 4;
    record[0] = count;
    record[1] = prefix;
    record[2] = code;
    record[3] = invalid;
  }
}

static LOOM_FORCE_INLINE void jpeg_prefix_bytes_body(
    const unsigned* compressed, const Image* headers, const unsigned* records,
    unsigned* prefixes, unsigned* segments, unsigned* errors) {
  unsigned image = loom::kernel::workgroup::id.x;
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned marker = 0xffffffffu;
  unsigned code = 0xffffffffu;
  unsigned invalid = 0xffffffffu;
  PacketWords counts = {};
  [[loom::unroll]]
  for (unsigned item = 0; item < 4; ++item) {
    unsigned tile = lane * 4 + item;
    if (tile < headers[image].tile_count) {
      const unsigned* record =
          records + (headers[image].tile_offset + tile) * 4;
      counts[item] = record[0];
      marker = byte_minimum(marker, record[1]);
      code = byte_minimum(code, record[2]);
      invalid = byte_minimum(invalid, record[3]);
    }
  }
  marker = loom::kernel::workgroup::reduce::minui(marker);
  code = loom::kernel::workgroup::reduce::minui(code);
  invalid = loom::kernel::workgroup::reduce::minui(invalid);
  unsigned error = 0;
  if (code == 0xffffffffu) {
    error = 1;
  } else {
    const unsigned char* source = reinterpret_cast<const unsigned char*>(
        compressed + headers[image].compressed_word_offset);
    if (source[code + 1] != 0xd9) {
      error = 2;
    }
    if (code + 2 != headers[image].compressed_bytes || invalid < code) {
      error = 1;
    }
  }
  unsigned count = 0;
  [[loom::unroll]]
  for (unsigned item = 0; item < 4; ++item) {
    unsigned tile = lane * 4 + item;
    if (tile * 4096 > marker || error) {
      counts[item] = 0;
    }
    count += counts[item];
  }
  unsigned prefix =
      loom::kernel::workgroup::scan::addi<loom::kernel::scan_mode::exclusive>(
          count);
  unsigned total = loom::kernel::workgroup::reduce::addi(count);
  [[loom::unroll]]
  for (unsigned item = 0; item < 4; ++item) {
    unsigned tile = lane * 4 + item;
    if (tile < headers[image].tile_count) {
      prefixes[headers[image].tile_offset + tile] = prefix;
    }
    prefix += counts[item];
  }
  unsigned job_count = (total + chunk_bytes - 1) / chunk_bytes;
  unsigned word_base = headers[image].entropy_word_offset;
  if (lane == 0) {
    unsigned* segment = segments + image * 8;
    segment[0] = word_base;
    segment[1] = total * 8;
    segment[2] = 0;
    segment[3] = job_count;
    segment[4] = headers[image].block_count;
    segment[5] = 0;
    segment[6] = 0;
    segment[7] = headers[image].component_word_offset;
    errors[image * 4] = error;
    errors[image * 4 + 1] = marker;
    errors[image * 4 + 2] = code;
    errors[image * 4 + 3] = total;
  }
}

static loom::kernel::configuration configure_job_prefix() {
  return {{1, 1, 1}, {256, 1, 1}};
}

static LOOM_FORCE_INLINE void jpeg_plan_jobs_body(unsigned* segments,
                                                  unsigned* schedule) {
  unsigned image = loom::kernel::workitem::id.x;
  unsigned count = image < certification_segments ? segments[image * 8 + 3] : 0;
  unsigned prefix =
      loom::kernel::workgroup::scan::addi<loom::kernel::scan_mode::exclusive>(
          count);
  unsigned total = loom::kernel::workgroup::reduce::addi(count);
  if (image < certification_segments) {
    segments[image * 8 + 2] = prefix;
    segments[image * 8 + 5] = prefix + image;
  }
  if (image == 0) {
    unsigned subgroup = loom::kernel::subgroup::size();
    unsigned intervals =
        total * (output_partitions + 1) + certification_segments;
    unsigned stripe = subgroup * certification_segments;
    schedule[0] = total;
    schedule[1] = (total + subgroup * 4 - 1) / (subgroup * 4);
    schedule[2] = 1;
    schedule[3] = 1;
    schedule[4] = (intervals + stripe - 1) / stripe;
    schedule[5] = certification_segments;
    schedule[6] = 1;
    schedule[7] = intervals;
  }
}

static LOOM_FORCE_INLINE void jpeg_compact_bytes_body(
    const unsigned* compressed, const Image* headers, const unsigned* records,
    const unsigned* prefixes, const unsigned* segments, const unsigned* errors,
    unsigned* entropy, unsigned* jobs) {
  unsigned image = loom::kernel::workgroup::id.y;
  unsigned tile = loom::kernel::workgroup::id.x;
  if (tile >= headers[image].tile_count) {
    return;
  }
  unsigned lane = loom::kernel::workitem::id.x;
  const unsigned* segment = segments + image * 8;
  for (unsigned job = tile * 256 + lane; job < segment[3];
       job += tile_count * 256) {
    unsigned* output = jobs + (segment[2] + job) * 4;
    output[0] = segment[0];
    output[1] = segment[1];
    output[2] = job;
    output[3] = image;
  }
  unsigned total = errors[image * 4 + 3];
  unsigned* destination = entropy + segments[image * 8];
  unsigned rounded = (total + 3) / 4;
  // One group owns the tail padding, including the unused bytes of its last
  // logical word. Partial tile stores below overwrite only valid byte lanes.
  if (tile == 0 && lane == 0) {
    unsigned char* bytes = reinterpret_cast<unsigned char*>(destination);
    for (unsigned byte = total; byte < rounded * 4; ++byte) {
      bytes[(byte & ~3u) + 3 - (byte & 3u)] = 0;
    }
    destination[rounded] = 0;
    destination[rounded + 1] = 0;
  }
  if (errors[image * 4] != 0 || tile * 4096 > errors[image * 4 + 1]) {
    return;
  }
  unsigned count = records[(headers[image].tile_offset + tile) * 4];
  unsigned output_start = prefixes[headers[image].tile_offset + tile];
  unsigned bias = output_start & 3u;
  auto* storage =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 16>(1025);
  unsigned char* bytes = reinterpret_cast<unsigned char*>(storage);
  const unsigned* source = compressed + headers[image].compressed_word_offset;
  unsigned first = tile * 4096 + lane * 16;
  Packet packet = classify(source, first, headers[image].compressed_bytes,
                           headers[image].entropy_start);
  unsigned mask = before_marker(
      packet.keep, first, records[(headers[image].tile_offset + tile) * 4 + 1]);
  unsigned offset =
      loom::kernel::workgroup::scan::addi<loom::kernel::scan_mode::exclusive>(
          loom::scalar::ctpopi(mask));
  [[loom::unroll]]
  for (unsigned index = 0; index < 16; ++index) {
    if ((mask & (1u << index)) != 0) {
      unsigned local_byte = (bias + offset) ^ 3u;
      // The exclusive packet count plus this packet's retained prefix owns
      // one of at most4096 payload bytes, with at most three alignment bytes.
      loom::assume(local_byte < 4100u);
      bytes[local_byte] = static_cast<unsigned char>(packet.words[index / 4] >>
                                                     ((index & 3u) * 8));
      ++offset;
    }
  }
  loom::kernel::workgroup::barrier();
  unsigned first_word = (output_start + 3) / 4;
  unsigned final_word = (output_start + count) / 4;
  for (unsigned word = first_word + lane; word < final_word; word += 256) {
    unsigned local_word = word - output_start / 4;
    loom::assume(local_word < 1025u);
    unsigned value = storage[local_word];
    destination[word] = value;
  }
  if (lane == 0) {
    unsigned char* output_bytes = reinterpret_cast<unsigned char*>(destination);
    for (unsigned index = 0; index < count; ++index) {
      unsigned position = output_start + index;
      if (position >= first_word * 4 && position < final_word * 4) {
        index = final_word * 4 - output_start - 1;
      } else {
        unsigned local_byte = (bias + index) ^ 3u;
        loom::assume(local_byte < 4100u);
        output_bytes[(position & ~3u) + 3 - (position & 3u)] =
            bytes[local_byte];
      }
    }
  }
}

// One status belongs to one stable input ordinal; input failures do not become
// device failures and no later image consumer reads a failed image's pixels.
[[loom::kernel(configure_images)]]
void jpeg_validate(const unsigned* segments, const unsigned* errors,
                   const unsigned* status, unsigned* validity) {
  unsigned image = loom::kernel::workgroup::id.x;
  const unsigned* segment = segments + image * 8;
  unsigned first = segment[2] * (output_partitions + 1) + image;
  unsigned count = segment[3] * (output_partitions + 1) + 1;
  unsigned error = 0;
  for (unsigned index = loom::kernel::workitem::id.x; index < count;
       index += 256) {
    unsigned value = status[first + index];
    if (value > error) {
      error = value;
    }
  }
  error = loom::kernel::workgroup::reduce::maxui(error);
  if (loom::kernel::workitem::id.x == 0) {
    validity[image] = errors[image * 4] | (error << 8);
  }
}

#include <loomcxx/atomic.h>

[[loom::config("stream.rows"), loom::where(loom::predicate::range(3u, 64u))]]
extern const unsigned stream_rows;
[[loom::config("stream.workers"),
  loom::where(loom::predicate::range(1u, 1024u))]]
extern const unsigned stream_workers;

struct Stream {
  // First logical row in the disjoint queue/completion arrays.
  unsigned row_offset;
  // Number of encoded MCU rows, including partial bottom-edge MCUs.
  unsigned row_count;
  // First word of the bounded component payload allocation.
  unsigned pixel_offset;
  // Current color row; completing its readers retires the preceding row.
  unsigned color_row;
  // Number of interval owners, including the mandatory terminal validator.
  unsigned interval_count;
  // Fully retired interval owners, including zero-output intervals.
  unsigned interval_done;
  // First observed decode failure; provisional output remains private.
  unsigned error;
};

static constexpr unsigned absent = 0xffffffffu;

// JPEG dimensions and the admitted 256-image cohort bound logical row indices.
static LOOM_FORCE_INLINE unsigned* row_address(unsigned* rows, unsigned index) {
  loom::assume(index < 2097152u);
  return rows + index;
}

static LOOM_FORCE_INLINE unsigned observe(const unsigned* address) {
  return loom::view::atomic::load<loom::atomic::ordering::acquire,
                                  loom::atomic::scope::device>(address);
}

static LOOM_FORCE_INLINE void publish(unsigned* address, unsigned value) {
  loom::view::atomic::store<loom::atomic::ordering::release,
                            loom::atomic::scope::device>(value, address);
}

static LOOM_FORCE_INLINE unsigned add(unsigned* address, unsigned value) {
  return loom::view::atomic::rmw<loom::atomic::kind::addi,
                                 loom::atomic::ordering::acq_rel,
                                 loom::atomic::scope::device>(value, address);
}

static LOOM_FORCE_INLINE unsigned compare(unsigned* address, unsigned expected,
                                          unsigned replacement) {
  return loom::view::atomic::cmpxchg<loom::atomic::ordering::acq_rel,
                                     loom::atomic::ordering::acquire,
                                     loom::atomic::scope::device>(
      expected, replacement, address);
}

// A task enters any logical row at most once. Reusing its link in a later row
// cannot make the same head value reappear in the old row's compare/exchange.
static LOOM_FORCE_INLINE void enqueue(unsigned* head, unsigned* links,
                                      unsigned task) {
  // The admitted 4 MiB/image, 1 KiB/chunk, and five records/chunk fit this
  // pool.
  loom::assume(task < 8388608u);
  unsigned previous = observe(head);
  bool done = false;
  while (!done) {
    publish(links + task, previous);
    unsigned actual = compare(head, previous, task);
    done = actual == previous;
    previous = actual;
  }
}

static loom::kernel::configuration configure_stream_init() {
  return {{certification_segments, 1, 1}, {256, 1, 1}};
}

static LOOM_FORCE_INLINE void jpeg_stream_init_body(
    const unsigned* segments, const unsigned* intervals, const Image* images,
    const unsigned* blocks, const Stream* plans, Stream* streams,
    unsigned* heads, unsigned* pending, unsigned* decoded,
    unsigned* color_claimed, unsigned* color_done, unsigned* links,
    unsigned* cursors) {
  unsigned image_index = loom::kernel::workgroup::id.x;
  unsigned lane = loom::kernel::workitem::id.x;
  Stream* stream = streams + image_index;
  unsigned row_offset = plans[image_index].row_offset;
  unsigned row_count = plans[image_index].row_count;
  if (lane == 0) {
    stream->row_offset = row_offset;
    stream->row_count = row_count;
    stream->pixel_offset = plans[image_index].pixel_offset;
    stream->color_row = 0;
    stream->interval_done = 0;
    stream->error = 0;
  }
  for (unsigned row = lane; row < row_count; row += 256) {
    heads[row_offset + row] = absent;
    pending[row_offset + row] = 0;
    decoded[row_offset + row] = 0;
    color_claimed[row_offset + row] = 0;
    color_done[row_offset + row] = 0;
  }
  loom::kernel::barrier<loom::memory_space::global,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  const unsigned* segment = segments + image_index * 8;
  unsigned first = segment[2] * (output_partitions + 1) + image_index;
  unsigned count = segment[3] * (output_partitions + 1) + 1;
  unsigned blocks_per_mcu =
      reinterpret_cast<const BlockLayout*>(blocks)[image_index].count;
  unsigned columns = images[image_index].mcu_columns;
  unsigned retained = 0;
  for (unsigned index = lane; index < count; index += 256) {
    unsigned task = first + index;
    const unsigned* entry = intervals + task * 8;
    if (entry[0] < entry[1] || index + 1 == count) {
      [[loom::unroll]]
      for (unsigned word = 0; word < 8; ++word) {
        cursors[task * 8 + word] = entry[word];
      }
      unsigned mcu = entry[2] / blocks_per_mcu + (entry[3] != 0);
      unsigned row = mcu / columns;
      row = row < row_count ? row : row_count - 1;
      add(row_address(pending, row_offset + row), 1);
      enqueue(row_address(heads, row_offset + row), links, task);
      ++retained;
    }
  }
  unsigned total = loom::kernel::workgroup::reduce::addi(retained);
  if (lane == 0) {
    stream->interval_count = total;
  }
}

// Try one finite batch claim. Contention returns to the ready-work selector;
// it never holds a decoded row hostage to an unscheduled workgroup.
static LOOM_FORCE_INLINE bool claim_decode(unsigned* head,
                                           const unsigned* links,
                                           unsigned* selected) {
  unsigned first = observe(head);
  unsigned next = first;
  for (unsigned index = 0; index < 32; ++index) {
    selected[index] = next;
    if (next != absent) {
      loom::assume(next < 8388608u);
      next = observe(links + next);
    }
  }
  return first != absent && compare(head, first, next) == first;
}

static LOOM_FORCE_INLINE unsigned stream_decode(
    const unsigned* entropy, const unsigned* segments,
    const unsigned* all_tables, const unsigned* blocks,
    const unsigned* intervals, const unsigned* all_geometry,
    const unsigned* all_quantization, const unsigned* zigzag, unsigned* pixels,
    const Image* images, unsigned segment, unsigned selected_row, unsigned task,
    short* coefficients, unsigned* cursors, Stream* streams, unsigned* heads,
    unsigned* pending, unsigned* links) {
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned threads = 32;
  unsigned coefficient_stride = 32;
  Stream* stream = streams + segment;
  Image image = images[segment];
  BlockLayout layout = reinterpret_cast<const BlockLayout*>(blocks)[segment];
  const unsigned* tables = all_tables + image.table_word_offset;
  const unsigned* geometry = all_geometry + image.geometry_word_offset;
  const unsigned* quantization =
      all_quantization + image.quantization_word_offset;
  const unsigned* descriptor = segments + segment * 8;
  unsigned total_blocks = descriptor[4];
  unsigned first_interval = descriptor[2] * (output_partitions + 1) + segment;
  unsigned terminal = first_interval + descriptor[3] * (output_partitions + 1);
  unsigned certified_end = intervals[terminal * 8];
  unsigned certified_blocks = intervals[terminal * 8 + 2];
  const unsigned* words = entropy + descriptor[0];
  unsigned bit_limit = descriptor[1];
  bool present = task != absent;
  unsigned produced = 0;
  Cursor cursor = {};
  cursor.budget = bit_limit;
  cursor.cached_word = 0xffffffffu;
  unsigned end = 0;
  if (present) {
    const unsigned* entry = cursors + task * 8;
    cursor.position = entry[0];
    end = entry[1];
    cursor.blocks = entry[2];
    cursor.block_index = entry[3] >> 6;
    cursor.coefficient = entry[3] & 63;
    [[loom::unroll]]
    for (unsigned component = 0; component < component_count; ++component) {
      cursor.predictors[component] = static_cast<int>(entry[4 + component]);
    }
    if (cursor.position < end) {
      while ((cursor.block_index != 0 || cursor.coefficient != 0) &&
             cursor.error == 0) {
        cursor = output_symbol(words, tables, layout, bit_limit, certified_end,
                               cursor)
                     .cursor;
      }
    }
  }
  bool active = present && cursor.blocks < total_blocks &&
                cursor.position < end && cursor.error == 0;
  active = active &&
           cursor.blocks / layout.count / image.mcu_columns == selected_row;
  while (loom::kernel::subgroup::vote::any(active)) {
    unsigned mcu = descriptor[6] + cursor.blocks / layout.count;
    unsigned mcu_row = mcu / image.mcu_columns;
    unsigned mcu_column = mcu % image.mcu_columns;
    for (unsigned block = 0; block < layout.count; ++block) {
      bool any_ac = false;
      bool output_valid = false;
      unsigned offset = 0;
      unsigned table_index = block_descriptor(layout, block) >> 5;
      unsigned available_tables = table_count;
      loom::assume(table_index < available_tables);
      const unsigned* ac_table = tables + table_index * 2048;
      if (active && cursor.error == 0) {
        [[loom::unroll]]
        for (unsigned index = 0; index < 64; ++index) {
          coefficients[coefficient_index(index, lane, coefficient_stride)] = 0;
        }
        if (cursor.blocks < certified_blocks) {
          // The root certificate covers this whole block. Syntax is proven;
          // the absolute DC predictor remains a consumer-owned constraint.
          unsigned first_coefficient = cursor.coefficient;
          loom::assume(first_coefficient == 0);
          cursor.coefficient = first_coefficient;
          Symbol dc = advance_symbol_certified(words, tables, layout, cursor);
          cursor = dc.cursor;
          if (dc.value < -1024 || dc.value > 1023) {
            cursor.error = 5;
          }
          if (cursor.error == 0) {
            coefficients[lane] = static_cast<short>(dc.value);
            do {
              Symbol symbol =
                  advance_ac_certified(words, ac_table, cursor, layout);
              cursor = symbol.cursor;
              if (symbol.index != 0xffffffffu) {
                unsigned natural = zigzag[symbol.index];
                loom::assume(natural < 64u);
                coefficients[coefficient_index(natural, lane,
                                               coefficient_stride)] =
                    static_cast<short>(symbol.value);
                any_ac = true;
              }
            } while (cursor.coefficient != 0);
          }
        } else {
          do {
            Symbol symbol =
                advance_symbol(words, tables, layout, bit_limit, cursor);
            cursor = symbol.cursor;
            if (symbol.index == 0 &&
                (symbol.value < -1024 || symbol.value > 1023)) {
              cursor.error = 5;
            }
            if (symbol.index != 0xffffffffu && cursor.error == 0) {
              unsigned natural = zigzag[symbol.index];
              loom::assume(natural < 64u);
              coefficients[coefficient_index(natural, lane,
                                             coefficient_stride)] =
                  static_cast<short>(symbol.value);
              if (symbol.index != 0) {
                any_ac = true;
              }
            }
          } while (cursor.coefficient != 0 && cursor.error == 0);
        }
        const unsigned* shape = geometry + block * 8;
        unsigned column = mcu_column * shape[3] + shape[5];
        unsigned row = mcu_row * shape[4] + shape[6];
        output_valid = cursor.error == 0 && column < shape[1] && row < shape[2];
        offset = stream->pixel_offset + shape[0] +
                 (((mcu_row % stream_rows) * shape[4] + shape[6]) * shape[1] +
                  column) *
                     16;
      }
      loom::kernel::barrier<loom::memory_space::workgroup,
                            loom::atomic::scope::subgroup,
                            loom::atomic::ordering::acq_rel>();
      unsigned component = block_descriptor(layout, block) & 3;
      unsigned components = component_count;
      loom::assume(component < components);
      for (unsigned round = 0; round < 8; ++round) {
        unsigned owner = round * (threads / 8) + lane / 8;
        reconstruct_cooperative(coefficients, coefficient_stride, threads, lane,
                                owner, output_valid, any_ac, offset,
                                quantization + component * 64, pixels);
      }
      loom::kernel::barrier<loom::memory_space::workgroup,
                            loom::atomic::scope::subgroup,
                            loom::atomic::ordering::acq_rel>();
    }
    if (active && cursor.error == 0) {
      ++produced;
    }
    active = present && cursor.blocks < total_blocks && cursor.position < end &&
             cursor.error == 0 &&
             cursor.blocks / layout.count / image.mcu_columns == selected_row;
  }

  if (present) {
    bool continued = cursor.blocks < total_blocks && cursor.position < end &&
                     cursor.error == 0;
    if (continued) {
      unsigned* entry = cursors + task * 8;
      entry[0] = cursor.position;
      entry[2] = cursor.blocks;
      entry[3] = cursor.block_index * 64 + cursor.coefficient;
      for (unsigned component = 0; component < 3; ++component) {
        entry[4 + component] = cursor.predictors[component];
      }
      unsigned row = cursor.blocks / layout.count / image.mcu_columns;
      add(row_address(pending, stream->row_offset + row), 1);
      enqueue(row_address(heads, stream->row_offset + row), links, task);
    } else {
      if (task == terminal && cursor.error == 0) {
        if (cursor.blocks != total_blocks || cursor.coefficient != 0) {
          cursor.error = 3;
        } else {
          unsigned padding = bit_limit - cursor.position;
          if (padding > 7) {
            cursor.error = 3;
          } else if (padding != 0) {
            Read input = peek(words, cursor);
            if ((input.value >> (32 - padding)) != ((1u << padding) - 1)) {
              cursor.error = 4;
            }
          }
        }
      }
      if (cursor.error != 0) {
        compare(&stream->error, 0, cursor.error);
      }
      add(&stream->interval_done, 1);
    }
  }
  return produced;
}

struct StreamColorGeometry {
  // Per-image encoded width, independent of the sampling specialization.
  unsigned width;
  // Per-image encoded height, independent of the sampling specialization.
  unsigned height;
  // Encoded luma rows per interleaved MCU.
  unsigned mcu_height;
};
using ColorSamples = int __attribute__((ext_vector_type(4)));

LOOM_FORCE_INLINE unsigned stream_color_divide_up(
    StreamColorGeometry dimensions, unsigned value, unsigned divisor) {
  return (value + divisor - 1) / divisor;
}

LOOM_FORCE_INLINE unsigned stream_color_plane_bytes(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  return stream_color_divide_up(
             dimensions,
             stream_color_divide_up(dimensions, dimensions.width, horizontal),
             8) *
         stream_color_divide_up(
             dimensions,
             stream_color_divide_up(dimensions, dimensions.height, vertical),
             8) *
         64;
}

LOOM_FORCE_INLINE unsigned stream_color_address(StreamColorGeometry dimensions,
                                                unsigned column, unsigned row,
                                                unsigned width,
                                                unsigned vertical) {
  row %= stream_rows * (dimensions.mcu_height / vertical);
  return ((row >> 3) * stream_color_divide_up(dimensions, width, 8) +
          (column >> 3)) *
             64 +
         (row & 7u) * 8 + (column & 7u);
}

LOOM_FORCE_INLINE unsigned stream_color_edge(StreamColorGeometry dimensions,
                                             int value, unsigned extent) {
  if (value < 0) {
    return 0;
  }
  unsigned positive = static_cast<unsigned>(value);
  return positive < extent ? positive : extent - 1;
}

LOOM_FORCE_INLINE bool stream_color_horizontal_halo(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  unsigned width =
      stream_color_divide_up(dimensions, dimensions.width, horizontal);
  return horizontal == 2 && (vertical == 1 || vertical == 2) && width > 2;
}

LOOM_FORCE_INLINE bool stream_color_vertical_halo(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  return vertical == 2 &&
         (horizontal == 1 ||
          stream_color_horizontal_halo(dimensions, horizontal, vertical));
}

LOOM_FORCE_INLINE unsigned stream_color_cache_columns(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  return stream_color_divide_up(dimensions, 64, horizontal) +
         2 * stream_color_horizontal_halo(dimensions, horizontal, vertical);
}

LOOM_FORCE_INLINE unsigned stream_color_cache_rows(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  return stream_color_divide_up(dimensions, 8, vertical) +
         2 * stream_color_vertical_halo(dimensions, horizontal, vertical);
}

LOOM_FORCE_INLINE unsigned stream_color_cache_size(
    StreamColorGeometry dimensions, unsigned horizontal, unsigned vertical) {
  return horizontal == 1 && vertical == 1
             ? 1
             : stream_color_cache_columns(dimensions, horizontal, vertical) *
                   stream_color_cache_rows(dimensions, horizontal, vertical);
}

// One copy of each reduced-plane sample, including only the interpolation halo,
// feeds every four-pixel consumer in this output tile.
LOOM_FORCE_INLINE void stream_color_stage(StreamColorGeometry dimensions,
                                          const unsigned char* plane,
                                          unsigned* cache, unsigned horizontal,
                                          unsigned vertical, unsigned first_x,
                                          unsigned first_y) {
  if (horizontal == 1 && vertical == 1) {
    return;
  }
  unsigned width =
      stream_color_divide_up(dimensions, dimensions.width, horizontal);
  unsigned height =
      stream_color_divide_up(dimensions, dimensions.height, vertical);
  unsigned columns =
      stream_color_cache_columns(dimensions, horizontal, vertical);
  unsigned count =
      columns * stream_color_cache_rows(dimensions, horizontal, vertical);
  int first_column =
      static_cast<int>(first_x / horizontal) -
      stream_color_horizontal_halo(dimensions, horizontal, vertical);
  int first_row = static_cast<int>(first_y / vertical) -
                  stream_color_vertical_halo(dimensions, horizontal, vertical);
  unsigned threads = loom::kernel::workgroup::size.x;
  for (unsigned index = loom::kernel::workitem::id.x; index < count;
       index += threads) {
    unsigned column = stream_color_edge(
        dimensions, first_column + static_cast<int>(index % columns), width);
    unsigned row = stream_color_edge(
        dimensions, first_row + static_cast<int>(index / columns), height);
    cache[index] =
        plane[stream_color_address(dimensions, column, row, width, vertical)];
  }
}

LOOM_FORCE_INLINE ColorSamples stream_color_samples(
    StreamColorGeometry dimensions, const unsigned char* plane,
    const unsigned* cache, unsigned horizontal, unsigned vertical,
    unsigned first_x, unsigned first_y, unsigned x, unsigned y) {
  ColorSamples result = {};
  if (horizontal == 1 && vertical == 1) {
    unsigned address =
        stream_color_address(dimensions, x, y, dimensions.width, vertical);
    unsigned packed = *reinterpret_cast<const unsigned*>(plane + address);
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      result[index] = (packed >> (index * 8)) & 255u;
    }
    return result;
  }
  unsigned columns =
      stream_color_cache_columns(dimensions, horizontal, vertical);
  unsigned cache_count =
      columns * stream_color_cache_rows(dimensions, horizontal, vertical);
  unsigned column =
      x / horizontal - first_x / horizontal +
      stream_color_horizontal_halo(dimensions, horizontal, vertical);
  unsigned row = y / vertical - first_y / vertical +
                 stream_color_vertical_halo(dimensions, horizontal, vertical);
  if (stream_color_horizontal_halo(dimensions, horizontal, vertical)) {
    ColorSamples weighted = {};
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      unsigned position = row * columns + column + index - 1;
      // The four-pixel owner lies inside the tile; staged halos include both
      // neighbors. Retain that geometric bound across global/local division.
      loom::assume(position < cache_count);
      loom::assume(position < 512u);
      unsigned value = cache[position];
      if (vertical == 2) {
        unsigned neighbor = (y & 1u) ? position + columns : position - columns;
        loom::assume(neighbor < cache_count);
        loom::assume(neighbor < 512u);
        value = 3 * value + cache[neighbor];
      }
      weighted[index] = value;
    }
    unsigned shift = vertical == 2 ? 4 : 2;
    unsigned even_bias = vertical == 2 ? 8 : 1;
    unsigned odd_bias = vertical == 2 ? 7 : 2;
    result[0] = (3 * weighted[1] + weighted[0] + even_bias) >> shift;
    result[1] = (3 * weighted[1] + weighted[2] + odd_bias) >> shift;
    result[2] = (3 * weighted[2] + weighted[1] + even_bias) >> shift;
    result[3] = (3 * weighted[2] + weighted[3] + odd_bias) >> shift;
  } else {
    [[loom::unroll]]
    for (unsigned index = 0; index < 4; ++index) {
      unsigned current_column = (x + index) / horizontal - first_x / horizontal;
      unsigned position = row * columns + current_column;
      loom::assume(position < cache_count);
      loom::assume(position < 512u);
      unsigned value = cache[position];
      if (stream_color_vertical_halo(dimensions, horizontal, vertical)) {
        unsigned neighbor = (y & 1u) ? position + columns : position - columns;
        loom::assume(neighbor < cache_count);
        loom::assume(neighbor < 512u);
        value = (3 * value + cache[neighbor] + ((y & 1u) ? 2 : 1)) >> 2;
      }
      result[index] = value;
    }
  }
  return result;
}

LOOM_FORCE_INLINE unsigned stream_color_clamp(StreamColorGeometry dimensions,
                                              int value) {
  unsigned positive = static_cast<unsigned>(value & ~(value >> 31));
  return (positive | (0u - static_cast<unsigned>(positive > 255u))) & 255u;
}

template <unsigned image_stream_color_space, unsigned horizontal0,
          unsigned vertical0, unsigned horizontal1, unsigned vertical1,
          unsigned horizontal2, unsigned vertical2>
static LOOM_FORCE_INLINE void stream_color_run(const unsigned char* components,
                                               unsigned char* rgb, Image image,
                                               unsigned first_x,
                                               unsigned first_y,
                                               unsigned* cache) {
  StreamColorGeometry dimensions{
      image.width, image.height,
      image.color_kind == 5 || image.color_kind == 6 || image.color_kind == 8
          ? 16u
          : 8u};
  unsigned bytes0 = image.first_plane_bytes;
  unsigned bytes1 = image.second_plane_bytes;
  const unsigned char* plane0 = components + image.component_word_offset * 4;
  const unsigned char* plane1 = plane0 + bytes0;
  const unsigned char* plane2 = plane1 + bytes1;
  unsigned* cache0 = cache;
  unsigned* cache1 = cache;
  unsigned* cache2 = cache + 512;
  stream_color_stage(dimensions, plane0, cache0, horizontal0, vertical0,
                     first_x, first_y);
  if (image_stream_color_space != 1) {
    stream_color_stage(dimensions, plane1, cache1, horizontal1, vertical1,
                       first_x, first_y);
    stream_color_stage(dimensions, plane2, cache2, horizontal2, vertical2,
                       first_x, first_y);
  }
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::subgroup,
                        loom::atomic::ordering::acq_rel>();
  unsigned lane = loom::kernel::workitem::id.x;
  for (unsigned item = lane; item < 128; item += 32) {
    unsigned x = first_x + (item & 15u) * 4;
    unsigned y = first_y + (item >> 4);
    if (x < dimensions.width && y < dimensions.height) {
      ColorSamples values0 =
          stream_color_samples(dimensions, plane0, cache0, horizontal0,
                               vertical0, first_x, first_y, x, y);
      ColorSamples values1 = {};
      ColorSamples values2 = {};
      if (image_stream_color_space != 1) {
        values1 = stream_color_samples(dimensions, plane1, cache1, horizontal1,
                                       vertical1, first_x, first_y, x, y);
        values2 = stream_color_samples(dimensions, plane2, cache2, horizontal2,
                                       vertical2, first_x, first_y, x, y);
      }
      ColorSamples red = values0;
      ColorSamples green = image_stream_color_space == 1 ? values0 : values1;
      ColorSamples blue = image_stream_color_space == 1 ? values0 : values2;
      if (image_stream_color_space == 3) {
        [[loom::unroll]]
        for (unsigned index = 0; index < 4; ++index) {
          int cb = values1[index] - 128;
          int cr = values2[index] - 128;
          red[index] = values0[index] + ((91881 * cr + 32768) >> 16);
          green[index] =
              values0[index] + ((-22554 * cb - 46802 * cr + 32768) >> 16);
          blue[index] = values0[index] + ((116130 * cb + 32768) >> 16);
        }
      }
      ColorSamples packed = {};
      [[loom::unroll]]
      for (unsigned index = 0; index < 4; ++index) {
        packed[index] = stream_color_clamp(dimensions, red[index]) |
                        (stream_color_clamp(dimensions, green[index]) << 8) |
                        (stream_color_clamp(dimensions, blue[index]) << 16);
      }
      unsigned offset = image.rgb_byte_offset + (y * dimensions.width + x) * 3;
      if ((dimensions.width & 3u) == 0) {
        unsigned* output = reinterpret_cast<unsigned*>(rgb + offset);
        using Words = unsigned __attribute__((ext_vector_type(4)));
        using Triple = loom::type::vector<loom::type::shape<3>, unsigned>;
        Words words = {static_cast<unsigned>(packed[0]) |
                           (static_cast<unsigned>(packed[1]) << 24),
                       (static_cast<unsigned>(packed[1]) >> 8) |
                           (static_cast<unsigned>(packed[2]) << 16),
                       (static_cast<unsigned>(packed[2]) >> 16) |
                           (static_cast<unsigned>(packed[3]) << 8),
                       0};
        auto destination = loom::buffer::view<3>(
            output, {}, loom::encoding::layout::dense<1>());
        loom::vector::store(loom::vector::slice<Triple>(words, 0), destination,
                            {0});
      } else {
        [[loom::unroll]]
        for (unsigned index = 0; index < 4; ++index) {
          if (x + index < dimensions.width) {
            rgb[offset + index * 3] = static_cast<unsigned char>(packed[index]);
            rgb[offset + index * 3 + 1] = static_cast<unsigned char>(
                static_cast<unsigned>(packed[index]) >> 8);
            rgb[offset + index * 3 + 2] = static_cast<unsigned char>(
                static_cast<unsigned>(packed[index]) >> 16);
          }
        }
      }
    }
  }
}

static LOOM_FORCE_INLINE void stream_color(const unsigned char* components,
                                           unsigned char* rgb, Image image,
                                           unsigned first_x, unsigned first_y,
                                           unsigned* cache) {
  if (image.color_kind == 1) {
    stream_color_run<1, 1, 1, 1, 1, 1, 1>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 2) {
    stream_color_run<2, 1, 1, 1, 1, 1, 1>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 3) {
    stream_color_run<3, 1, 1, 1, 1, 1, 1>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 4) {
    stream_color_run<3, 1, 1, 2, 1, 2, 1>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 5) {
    stream_color_run<3, 1, 1, 2, 2, 2, 2>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 6) {
    stream_color_run<3, 1, 1, 1, 2, 1, 2>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 7) {
    stream_color_run<3, 1, 1, 4, 1, 4, 1>(components, rgb, image, first_x,
                                          first_y, cache);
  }
  if (image.color_kind == 8) {
    stream_color_run<3, 1, 1, 4, 2, 4, 2>(components, rgb, image, first_x,
                                          first_y, cache);
  }
}

static loom::kernel::configuration configure_stream_workers() {
  return {{stream_workers, 1, 1}, {32, 1, 1}};
}

static LOOM_FORCE_INLINE void jpeg_stream_body(
    const unsigned* entropy, const unsigned* segments, const unsigned* tables,
    const unsigned* blocks, const unsigned* intervals, const unsigned* geometry,
    const unsigned* quantization, const unsigned* zigzag, unsigned* pixels,
    unsigned char* rgb, const Image* images, Stream* streams, unsigned* heads,
    unsigned* pending, unsigned* decoded, unsigned* color_claimed,
    unsigned* color_done, unsigned* links, unsigned* cursors) {
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned search = loom::kernel::workgroup::id.x % certification_segments;
  // The color halo and task claims reuse the exact coefficient allocation.
  auto* storage =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 16>(1024);
  while (true) {
    unsigned role = 0;
    unsigned selected_image = 0;
    unsigned selected_row = 0;
    unsigned selected_tile = 0;
    unsigned done_count = 0;
    if (lane == 0) {
      for (unsigned attempt = 0; attempt < certification_segments && role == 0;
           ++attempt) {
        unsigned image_index = (search + attempt) % certification_segments;
        Stream* stream = streams + image_index;
        Image image = images[image_index];
        unsigned row = observe(&stream->color_row);
        unsigned rows = stream->row_count;
        if (observe(&stream->error) != 0 ||
            (row == rows &&
             observe(&stream->interval_done) == stream->interval_count)) {
          ++done_count;
        } else {
          unsigned height = image.color_kind == 5 || image.color_kind == 6 ||
                                    image.color_kind == 8
                                ? 16
                                : 8;
          unsigned columns = (image.width + 63) / 64;
          unsigned tiles = columns * (height / 8);
          unsigned row_offset = stream->row_offset;
          unsigned previous = row ? row - 1 : 0;
          unsigned next = row + 1 < rows ? row + 1 : rows - 1;
          if (row < rows &&
              observe(row_address(pending, row_offset + previous)) == 0 &&
              observe(row_address(pending, row_offset + row)) == 0 &&
              observe(row_address(pending, row_offset + next)) == 0 &&
              observe(row_address(decoded, row_offset + previous)) ==
                  image.mcu_columns &&
              observe(row_address(decoded, row_offset + row)) ==
                  image.mcu_columns &&
              observe(row_address(decoded, row_offset + next)) ==
                  image.mcu_columns) {
            unsigned tile =
                observe(row_address(color_claimed, row_offset + row));
            if (tile < tiles &&
                compare(row_address(color_claimed, row_offset + row), tile,
                        tile + 1) == tile) {
              role = 2;
              selected_image = image_index;
              selected_row = row;
              selected_tile = tile;
            }
          }
          unsigned first_row = row ? row - 1 : 0;
          for (unsigned relative = 0; relative < stream_rows && role == 0;
               ++relative) {
            unsigned candidate = first_row + relative;
            if (candidate < rows &&
                claim_decode(row_address(heads, row_offset + candidate), links,
                             storage)) {
              role = 1;
              selected_image = image_index;
              selected_row = candidate;
            }
          }
        }
      }
    }
    role = loom::kernel::subgroup::broadcast(role, 0);
    selected_image = loom::kernel::subgroup::broadcast(selected_image, 0);
    // Every selected image is an in-range scheduler candidate, including idle
    // 0.
    unsigned available_images = certification_segments;
    loom::assume(selected_image < available_images);
    selected_row = loom::kernel::subgroup::broadcast(selected_row, 0);
    selected_tile = loom::kernel::subgroup::broadcast(selected_tile, 0);
    done_count = loom::kernel::subgroup::broadcast(done_count, 0);
    if (done_count == certification_segments) {
      break;
    }
    // Transfer the leader's acquired global payload and selected LDS task IDs
    // to all consumers. This is subgroup synchronization, never a grid barrier.
    loom::kernel::barrier<loom::memory_space::global,
                          loom::atomic::scope::subgroup,
                          loom::atomic::ordering::acq_rel>();
    loom::kernel::barrier<loom::memory_space::workgroup,
                          loom::atomic::scope::subgroup,
                          loom::atomic::ordering::acq_rel>();
    Stream* stream = streams + selected_image;
    Image image = images[selected_image];
    if (role == 1) {
      unsigned task = storage[lane];
      loom::kernel::barrier<loom::memory_space::workgroup,
                            loom::atomic::scope::subgroup,
                            loom::atomic::ordering::acq_rel>();
      unsigned produced =
          stream_decode(entropy, segments, tables, blocks, intervals, geometry,
                        quantization, zigzag, pixels, images, selected_image,
                        selected_row, task, reinterpret_cast<short*>(storage),
                        cursors, streams, heads, pending, links);
      unsigned total = loom::kernel::subgroup::reduce::addi(produced);
      unsigned retired = loom::kernel::subgroup::reduce::addi(
          static_cast<unsigned>(task != absent));
      loom::kernel::barrier<loom::memory_space::global,
                            loom::atomic::scope::subgroup,
                            loom::atomic::ordering::acq_rel>();
      if (lane == 0) {
        if (total != 0) {
          add(row_address(decoded, stream->row_offset + selected_row), total);
        }
        add(row_address(pending, stream->row_offset + selected_row),
            0u - retired);
      }
    }
    if (role == 2) {
      unsigned height = image.color_kind == 5 || image.color_kind == 6 ||
                                image.color_kind == 8
                            ? 16
                            : 8;
      unsigned columns = (image.width + 63) / 64;
      unsigned first_x = selected_tile % columns * 64;
      unsigned first_y = selected_row * height + selected_tile / columns * 8;
      stream_color(reinterpret_cast<const unsigned char*>(pixels), rgb, image,
                   first_x, first_y, storage);
      loom::kernel::barrier<loom::memory_space::global,
                            loom::atomic::scope::subgroup,
                            loom::atomic::ordering::acq_rel>();
      if (lane == 0) {
        unsigned done =
            add(row_address(color_done, stream->row_offset + selected_row), 1);
        if (done + 1 == columns * (height / 8)) {
          publish(&stream->color_row, selected_row + 1);
        }
      }
    }
    search = (search + 1) % certification_segments;
  }
}

static LOOM_FORCE_INLINE unsigned stream_offset(const unsigned* input,
                                                unsigned index) {
  unsigned offset = loom::kernel::subgroup::broadcast_first(input[index]);
  // The bounded probe's complete roots are checked below 1 GiB when packed.
  loom::assume(offset < 1073741824u);
  loom::assume(loom::predicate::multiple_of(offset, 16u));
  return offset / 4;
}

[[loom::kernel(configure_chunks)]]
void jpeg_speculate(const unsigned* input, unsigned* output) {
  jpeg_speculate_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 8)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 25)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 0)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 1)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 26)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 27)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 24)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)));
}

[[loom::kernel(configure_bridges)]]
void jpeg_bridge(const unsigned* input, unsigned* output) {
  jpeg_bridge_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 8)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 25)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 0)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 1)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 26)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 28)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 24)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)));
}

[[loom::kernel(configure_segment_groups)]]
void jpeg_index(const unsigned* input, unsigned* output) {
  jpeg_index_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 26)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 29)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 30)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 10)));
}

[[loom::kernel(configure_segment_roots)]]
void jpeg_certify(const unsigned* input, unsigned* output) {
  jpeg_certify_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 26)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 28)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 29)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 30)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 10)));
}

[[loom::kernel(configure_segment_groups)]]
void jpeg_intervals(const unsigned* input, unsigned* output) {
  jpeg_intervals_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 26)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 29)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 30)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 27)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 10)));
}

[[loom::kernel(configure_tiles)]]
void jpeg_classify_bytes(const unsigned* input, unsigned* output) {
  jpeg_classify_bytes_body(
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 7)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 21)));
}

[[loom::kernel(configure_images)]]
void jpeg_prefix_bytes(const unsigned* input, unsigned* output) {
  jpeg_prefix_bytes_body(
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 7)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 21)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 22)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 23)));
}

[[loom::kernel(configure_job_prefix)]]
void jpeg_plan_jobs(const unsigned* input, unsigned* output) {
  jpeg_plan_jobs_body(
      reinterpret_cast<unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 24)));
}

[[loom::kernel(configure_tiles)]]
void jpeg_compact_bytes(const unsigned* input, unsigned* output) {
  jpeg_compact_bytes_body(
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 7)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 21)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 22)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 23)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 8)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 25)));
}

[[loom::kernel(configure_stream_init)]]
void jpeg_stream_init(const unsigned* input, unsigned* output) {
  jpeg_stream_init_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 10)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 1)),
      reinterpret_cast<const Stream*>(input + stream_offset(input, 6)),
      reinterpret_cast<Stream*>(output + stream_offset(input, 13)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 14)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 15)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 16)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 17)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 18)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 19)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 20)));
}

[[loom::kernel(configure_stream_workers)]]
void jpeg_stream(const unsigned* input, unsigned* output) {
  jpeg_stream_body(
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 8)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 9)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 0)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 1)),
      reinterpret_cast<const unsigned*>(output + stream_offset(input, 10)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 2)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 3)),
      reinterpret_cast<const unsigned*>(input + stream_offset(input, 4)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 11)),
      reinterpret_cast<unsigned char*>(output + stream_offset(input, 12)),
      reinterpret_cast<const Image*>(input + stream_offset(input, 5)),
      reinterpret_cast<Stream*>(output + stream_offset(input, 13)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 14)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 15)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 16)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 17)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 18)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 19)),
      reinterpret_cast<unsigned*>(output + stream_offset(input, 20)));
}
