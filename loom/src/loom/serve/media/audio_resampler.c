// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_resampler.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"

// I0 on [0, 10]. Thirty terms resolve the cold filter construction in F64;
// neither the series nor transcendental operations occur during resampling.
static double loom_serve_audio_resampler_bessel(double value) {
  double sum = 1.0;
  double term = 1.0;
  for (int order = 1; order <= 30; ++order) {
    term *= value * value / (4.0 * order * order);
    sum += term;
  }
  return sum;
}

iree_status_t loom_serve_audio_resampler_filter_initialize(
    uint32_t input_rate, uint32_t output_rate, iree_host_size_t storage_limit,
    iree_allocator_t host_allocator,
    loom_serve_audio_resampler_filter_t* out_filter) {
  *out_filter =
      (loom_serve_audio_resampler_filter_t){.host_allocator = host_allocator};
  if (!input_rate || !output_rate) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resampling requires positive input/output rates");
  }
  const uint32_t divisor = (uint32_t)iree_math_gcd_u64(input_rate, output_rate);
  out_filter->numerator = input_rate / divisor;
  out_filter->denominator = output_rate / divisor;
  if (input_rate == output_rate) {
    return iree_ok_status();
  }
  const uint32_t phase_count = iree_min(out_filter->denominator, 512u);
  const uint64_t center =
      (64ull * iree_max(input_rate, output_rate) + output_rate - 1) /
      output_rate;
  const uint64_t taps = (2 * center + 5) / 4 * 4;
  // Rates are U32 and the phase grid is bounded, so these U64 products fit.
  const uint64_t storage_size = taps * (phase_count + 1) * sizeof(float);
  const uint64_t temporary_size = taps * sizeof(double);
  if (storage_size + temporary_size > storage_limit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "resampling filter requires %" PRIu64
                            " bytes, exceeding the %" PRIhsz " byte limit",
                            storage_size + temporary_size, storage_limit);
  }
  out_filter->phase_count = phase_count;
  out_filter->center_offset = (iree_host_size_t)center;
  out_filter->tap_count = (iree_host_size_t)taps;
  out_filter->storage_size = (iree_host_size_t)storage_size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, out_filter->storage_size,
      (void**)&out_filter->coefficients));
  double* values = NULL;
  iree_status_t status = iree_allocator_malloc_uninitialized(
      host_allocator, (iree_host_size_t)temporary_size, (void**)&values);
  if (!iree_status_is_ok(status)) {
    loom_serve_audio_resampler_filter_deinitialize(out_filter);
    return status;
  }
  const double pi = 3.14159265358979323846264338327950288;
  const double cutoff = 0.95 * iree_min(1.0, (double)output_rate / input_rate);
  const double inverse_bessel = 1.0 / loom_serve_audio_resampler_bessel(10.0);
  for (uint32_t phase = 0; phase <= phase_count; ++phase) {
    double sum = 0;
    for (iree_host_size_t i = 0; i < out_filter->tap_count; ++i) {
      const double distance =
          (double)i - (double)center - (double)phase / phase_count;
      values[i] = 0;
      if (fabs(distance) <= center) {
        const double window =
            loom_serve_audio_resampler_bessel(
                10.0 * sqrt(iree_max(
                           0.0, 1.0 - distance * distance /
                                          ((double)center * (double)center)))) *
            inverse_bessel;
        const double sinc = distance == 0.0
                                ? cutoff
                                : sin(pi * cutoff * distance) / (pi * distance);
        values[i] = sinc * window;
        sum += values[i];
      }
    }
    for (iree_host_size_t i = 0; i < out_filter->tap_count; ++i) {
      out_filter->coefficients[phase * out_filter->tap_count + i] =
          (float)(values[i] / sum);
    }
  }
  iree_allocator_free(host_allocator, values);
  return iree_ok_status();
}

void loom_serve_audio_resampler_filter_deinitialize(
    loom_serve_audio_resampler_filter_t* filter) {
  iree_allocator_free(filter->host_allocator, filter->coefficients);
  memset(filter, 0, sizeof(*filter));
}

iree_status_t loom_serve_audio_resampler_initialize(
    const loom_serve_audio_resampler_filter_t* filter, uint32_t channels,
    iree_host_size_t input_capacity, iree_host_size_t storage_limit,
    iree_allocator_t host_allocator,
    loom_serve_audio_resampler_t* out_resampler) {
  *out_resampler =
      (loom_serve_audio_resampler_t){.host_allocator = host_allocator};
  if (!channels || !input_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resampling requires channels and input capacity");
  }
  uint64_t capacity = 0;
  uint64_t storage_size = 0;
  if (!iree_checked_add_u64(filter->tap_count, input_capacity, &capacity) ||
      !iree_checked_mul_u64(capacity, (uint64_t)channels * sizeof(float),
                            &storage_size) ||
      storage_size > storage_limit) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "resampling history exceeds the %" PRIhsz " byte limit", storage_limit);
  }
  out_resampler->filter = filter;
  out_resampler->channels = channels;
  out_resampler->capacity = (iree_host_size_t)capacity;
  out_resampler->count = filter->center_offset;
  out_resampler->input_end = filter->center_offset;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, (iree_host_size_t)storage_size,
      (void**)&out_resampler->samples));
  memset(out_resampler->samples, 0,
         filter->center_offset * channels * sizeof(float));
  return iree_ok_status();
}

void loom_serve_audio_resampler_deinitialize(
    loom_serve_audio_resampler_t* resampler) {
  iree_allocator_free(resampler->host_allocator, resampler->samples);
  memset(resampler, 0, sizeof(*resampler));
}

static void loom_serve_audio_resampler_compact(
    loom_serve_audio_resampler_t* resampler) {
  if (!resampler->cursor) {
    return;
  }
  memmove(resampler->samples,
          resampler->samples + resampler->cursor * resampler->channels,
          (resampler->count - resampler->cursor) * resampler->channels *
              sizeof(float));
  resampler->count -= resampler->cursor;
  resampler->input_end -= resampler->cursor;
  resampler->cursor = 0;
}

float* loom_serve_audio_resampler_reserve(
    loom_serve_audio_resampler_t* resampler,
    iree_host_size_t* out_frame_capacity) {
  *out_frame_capacity = 0;
  if (resampler->finished) {
    return NULL;
  }
  if (resampler->count == resampler->capacity) {
    loom_serve_audio_resampler_compact(resampler);
  }
  *out_frame_capacity = resampler->capacity - resampler->count;
  return resampler->samples + resampler->count * resampler->channels;
}

void loom_serve_audio_resampler_commit(loom_serve_audio_resampler_t* resampler,
                                       iree_host_size_t frame_count) {
  resampler->count += frame_count;
  resampler->input_end = resampler->count;
}

// Independent accumulators retain a stable reduction order across packet and
// output-credit boundaries. Coefficients and neighboring input windows are
// reused from the stream's bounded storage; channels remain independent.
static float loom_serve_audio_resampler_convolve(const float* samples,
                                                 iree_host_size_t stride,
                                                 const float* coefficients,
                                                 iree_host_size_t tap_count) {
  float sums[4] = {0};
  for (iree_host_size_t i = 0; i < tap_count; i += 4) {
    sums[0] += samples[(i + 0) * stride] * coefficients[i + 0];
    sums[1] += samples[(i + 1) * stride] * coefficients[i + 1];
    sums[2] += samples[(i + 2) * stride] * coefficients[i + 2];
    sums[3] += samples[(i + 3) * stride] * coefficients[i + 3];
  }
  return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

iree_host_size_t loom_serve_audio_resampler_read(
    loom_serve_audio_resampler_t* resampler, iree_host_size_t frame_capacity,
    float* frames) {
  const loom_serve_audio_resampler_filter_t* filter = resampler->filter;
  const iree_host_size_t channels = resampler->channels;
  if (!filter->tap_count) {
    const iree_host_size_t count =
        iree_min(resampler->count - resampler->cursor, frame_capacity);
    if (count) {
      memcpy(frames, resampler->samples + resampler->cursor * channels,
             count * channels * sizeof(float));
    }
    resampler->cursor += count;
    return count;
  }
  iree_host_size_t generated = 0;
  while (generated < frame_capacity &&
         !loom_serve_audio_resampler_is_complete(resampler)) {
    if (resampler->count - resampler->cursor < filter->tap_count) {
      if (!resampler->finished) {
        break;
      }
      if (resampler->cursor + filter->tap_count > resampler->capacity) {
        loom_serve_audio_resampler_compact(resampler);
      }
      const iree_host_size_t end = resampler->cursor + filter->tap_count;
      memset(resampler->samples + resampler->count * channels, 0,
             (end - resampler->count) * channels * sizeof(float));
      resampler->count = end;
    }
    const uint64_t scaled = (uint64_t)resampler->fraction * filter->phase_count;
    const iree_host_size_t phase = scaled / filter->denominator;
    const uint32_t remainder = scaled % filter->denominator;
    const float* coefficients =
        filter->coefficients + phase * filter->tap_count;
    for (iree_host_size_t channel = 0; channel < channels; ++channel) {
      const float* samples =
          resampler->samples + resampler->cursor * channels + channel;
      float value = loom_serve_audio_resampler_convolve(
          samples, channels, coefficients, filter->tap_count);
      if (remainder) {
        const float next = loom_serve_audio_resampler_convolve(
            samples, channels, coefficients + filter->tap_count,
            filter->tap_count);
        value += (next - value) * ((float)remainder / filter->denominator);
      }
      frames[generated * channels + channel] = value;
    }
    const uint64_t next = (uint64_t)resampler->fraction + filter->numerator;
    resampler->cursor += next / filter->denominator;
    resampler->fraction = next % filter->denominator;
    ++generated;
  }
  return generated;
}

void loom_serve_audio_resampler_finish(
    loom_serve_audio_resampler_t* resampler) {
  resampler->finished = true;
}

bool loom_serve_audio_resampler_is_complete(
    const loom_serve_audio_resampler_t* resampler) {
  return resampler->finished &&
         resampler->cursor + resampler->filter->center_offset >=
             resampler->input_end;
}
