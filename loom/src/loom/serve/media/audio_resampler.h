// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_AUDIO_RESAMPLER_H_
#define LOOM_SERVE_MEDIA_AUDIO_RESAMPLER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable fixed-rate filter, independent of channel count and stream history.
// A normalized Kaiser-windowed sinc preserves the lower rate's passband through
// 90% of Nyquist and suppresses aliasing above its Nyquist. Its radius is 64
// frames at the lower rate. Equal rates are exact identity with no lookahead.
// Coefficients may be shared by independently owned streams; their owner keeps
// this object stable until every borrowing stream has been deinitialized.
typedef struct loom_serve_audio_resampler_filter_t {
  // Allocator owning the coefficient storage.
  iree_allocator_t host_allocator;
  // Reduced input-rate numerator of each output frame's input-time advance.
  uint32_t numerator;
  // Reduced output-rate denominator of the retained fractional input position.
  uint32_t denominator;
  // Exact phases when the denominator fits; otherwise an interpolation grid.
  uint32_t phase_count;
  // Input frames preceding the center of each convolution window.
  iree_host_size_t center_offset;
  // Coefficients in one padded convolution row; zero denotes identity.
  iree_host_size_t tap_count;
  // Owned coefficient bytes, excluding temporary initialization storage.
  iree_host_size_t storage_size;
  // Immutable rows, including the endpoint needed for phase interpolation.
  float* coefficients;
} loom_serve_audio_resampler_filter_t;

// Constructs on a CPU worker. Positive integer rates are limited by checked
// storage size, not a rate whitelist. storage_limit bounds coefficients plus
// temporary construction storage. Failure leaves a deinitializable value.
iree_status_t loom_serve_audio_resampler_filter_initialize(
    uint32_t input_rate, uint32_t output_rate, iree_host_size_t storage_limit,
    iree_allocator_t host_allocator,
    loom_serve_audio_resampler_filter_t* out_filter);
void loom_serve_audio_resampler_filter_deinitialize(
    loom_serve_audio_resampler_filter_t* filter);

// Independent, bounded native-F32 history. Calls are serialized by its owner;
// workers operating on different streams may share the immutable filter. No
// operation waits, schedules work, or accesses a model. Accepted workers retain
// this value and its filter until they retire, including after cancellation.
typedef struct loom_serve_audio_resampler_t {
  // Allocator owning the input/history storage.
  iree_allocator_t host_allocator;
  // Borrowed immutable filter, stable through deinitialization.
  const loom_serve_audio_resampler_filter_t* filter;
  // Independent interleaved samples per frame; conversion never mixes channels.
  uint32_t channels;
  // Total retained frame capacity, including convolution history.
  iree_host_size_t capacity;
  // Owned native-F32 frames filled by the producer or explicit EOF padding.
  float* samples;
  // Initialized frames in samples, including initial and final zero extension.
  iree_host_size_t count;
  // First tap of the next output window, relative to samples.
  iree_host_size_t cursor;
  // Real input endpoint in this window, excluding final zero extension.
  iree_host_size_t input_end;
  // Fractional input position in units of the filter's denominator.
  uint32_t fraction;
  // Explicit input EOF; packet exhaustion never sets this field.
  bool finished;
} loom_serve_audio_resampler_t;

// Adds input_capacity frames of producer storage to the filter history.
// storage_limit bounds this allocation independently of shared coefficients.
// Construction is fallible; subsequent trusted numerical operations are not.
iree_status_t loom_serve_audio_resampler_initialize(
    const loom_serve_audio_resampler_filter_t* filter, uint32_t channels,
    iree_host_size_t input_capacity, iree_host_size_t storage_limit,
    iree_allocator_t host_allocator,
    loom_serve_audio_resampler_t* out_resampler);
void loom_serve_audio_resampler_deinitialize(
    loom_serve_audio_resampler_t* resampler);

// Borrows writable native-F32 storage until the next resampler operation.
// Compacts retained history only when storage fills, independently of packet
// boundaries. Zero capacity means output must be read first, or input has
// ended.
float* loom_serve_audio_resampler_reserve(
    loom_serve_audio_resampler_t* resampler,
    iree_host_size_t* out_frame_capacity);

// Publishes a prefix of the last reservation. The producer has filled those
// whole frames with finite normalized samples in [-1, 1]. No input follows
// finish. These are producer invariants, established by PCM decoding or the
// caller's external input validation before entering the numerical path.
void loom_serve_audio_resampler_commit(loom_serve_audio_resampler_t* resampler,
                                       iree_host_size_t frame_count);

// Produces at most frame_capacity native-F32 frames into disjoint caller
// storage. Zero capacity does nothing; zero output otherwise needs input or
// means completion. Filtering may overshoot [-1, 1]; it never clips samples.
// Output timestamps are j/output_rate. Lookahead delays availability without
// shifting that clock. Explicit EOF yields ceil(N * output_rate / input_rate)
// frames for N real input frames, with no extra filter-tail duration.
iree_host_size_t loom_serve_audio_resampler_read(
    loom_serve_audio_resampler_t* resampler, iree_host_size_t frame_capacity,
    float* frames);

// Marks EOF without padding or processing. Repeated finish is harmless.
void loom_serve_audio_resampler_finish(loom_serve_audio_resampler_t* resampler);
// All output timestamps within the real input duration have been emitted.
bool loom_serve_audio_resampler_is_complete(
    const loom_serve_audio_resampler_t* resampler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_AUDIO_RESAMPLER_H_
