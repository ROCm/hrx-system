// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_completion_carrier_trace.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(IREE_PLATFORM_LINUX)
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX

#include "iree/base/threading/thread.h"
#include "iree/base/time.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

enum {
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY = 131072,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY = 32768,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_EVENT_CAPACITY = 512 * 1024,
};

typedef struct iree_hal_amdgpu_completion_carrier_clock_sample_t {
  // Monotonic time sampled immediately before the boot-time clock.
  uint64_t monotonic_before_ns;
  // Boot time sampled between the monotonic bounds.
  uint64_t boottime_ns;
  // Monotonic time sampled immediately after the boot-time clock.
  uint64_t monotonic_after_ns;
  // Errno from sampling the boot-time clock, or zero on success.
  int error;
} iree_hal_amdgpu_completion_carrier_clock_sample_t;

typedef struct iree_hal_amdgpu_completion_carrier_slot_t {
  // Absolute packet ID currently owning this physical slot.
  uint64_t packet_id;
  // Reservation containing |packet_id|.
  uint64_t reservation_id;
  // Absolute packet ID of the slot's preceding owner.
  uint64_t previous_packet_id;
  // Completion signal observed for the preceding owner.
  uint64_t previous_completion_signal;
  // Completion signal observed for the current owner.
  uint64_t completion_signal;
  // Number of packets in the current reservation.
  uint32_t reservation_count;
  // Offset of this packet within the current reservation.
  uint32_t reservation_offset;
} iree_hal_amdgpu_completion_carrier_slot_t;

typedef struct iree_hal_amdgpu_completion_carrier_packet_t {
  // Absolute AQL packet ID.
  uint64_t packet_id;
  // Reservation identity assigned when the packet span was claimed.
  uint64_t reservation_id;
  // Absolute packet ID of the preceding physical-slot owner.
  uint64_t previous_packet_id;
  // Completion signal carried by the preceding physical-slot owner.
  uint64_t previous_completion_signal;
  // Completion signal present immediately before ringing the doorbell.
  uint64_t completion_signal;
  // First notification epoch possibly assigned by this doorbell batch.
  uint64_t first_epoch;
  // Last notification epoch assigned by this doorbell batch.
  uint64_t last_epoch;
  // Host publication epoch immediately before the doorbell.
  uint64_t published_epoch;
  // Monotonic doorbell sequence containing this packet.
  uint64_t doorbell_sequence;
  // Number of packets in the owning reservation.
  uint32_t reservation_count;
  // Offset of this packet within the owning reservation.
  uint32_t reservation_offset;
  // Physical AQL ring slot used by this packet.
  uint32_t slot;
  // Number of epochs assigned by this doorbell batch.
  uint32_t batch_epoch_count;
  // Number of queue-epoch signal bodies found in this doorbell batch.
  uint32_t batch_carrier_count;
  // PM4 sidecar record index, or UINT32_MAX when not applicable.
  uint32_t pm4_record_index;
  // Header observed immediately before ringing the doorbell.
  uint16_t header;
  // Setup or vendor format observed with |header|.
  uint16_t setup;
  // True when reservation tracking owned the physical slot.
  bool has_reservation;
  // Complete packet bytes sampled immediately after ringing the doorbell.
  uint8_t raw[64];
} iree_hal_amdgpu_completion_carrier_packet_t;

typedef struct iree_hal_amdgpu_completion_carrier_pm4_t {
  // Complete queue-private PM4 IB slot bytes.
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY];
} iree_hal_amdgpu_completion_carrier_pm4_t;

typedef enum iree_hal_amdgpu_completion_carrier_position_phase_e {
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_INITIAL = 0,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_RESERVE = 1,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_BEFORE_DOORBELL = 2,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_AFTER_DOORBELL = 3,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_EXPORT = 4,
} iree_hal_amdgpu_completion_carrier_position_phase_t;

typedef struct iree_hal_amdgpu_completion_carrier_position_sample_t {
  // Hardware-reported next AQL packet position.
  uint64_t read_dispatch_id;
  // Read position from the immediately preceding recorder sample.
  uint64_t previous_read_dispatch_id;
  // Producer-reserved next AQL packet position.
  uint64_t write_dispatch_id;
  // Write position from the immediately preceding recorder sample.
  uint64_t previous_write_dispatch_id;
  // Doorbell sequence active when the positions were sampled.
  uint64_t doorbell_sequence;
  // Recorder boundary at which the positions were sampled.
  iree_hal_amdgpu_completion_carrier_position_phase_t phase;
} iree_hal_amdgpu_completion_carrier_position_sample_t;

typedef struct iree_hal_amdgpu_completion_carrier_position_history_t {
  // Queue positions observed before recorder callbacks are installed.
  iree_hal_amdgpu_completion_carrier_position_sample_t initial;
  // Most recent queue positions observed by the recorder.
  iree_hal_amdgpu_completion_carrier_position_sample_t last;
  // First observation whose read position regressed from the prior sample.
  iree_hal_amdgpu_completion_carrier_position_sample_t first_read_regression;
  // First observation whose write position regressed from the prior sample.
  iree_hal_amdgpu_completion_carrier_position_sample_t first_write_regression;
  // First observation whose read position exceeded the producer write position.
  iree_hal_amdgpu_completion_carrier_position_sample_t first_read_ahead;
  // Greatest read position observed across all samples.
  uint64_t max_read_dispatch_id;
  // Greatest write position observed across all samples.
  uint64_t max_write_dispatch_id;
  // Number of queue-position samples taken.
  uint64_t sample_count;
  // Number of samples whose read position regressed from the prior sample.
  uint64_t read_regression_count;
  // Number of samples whose write position regressed from the prior sample.
  uint64_t write_regression_count;
  // Number of samples whose read position exceeded the producer write position.
  uint64_t read_ahead_count;
} iree_hal_amdgpu_completion_carrier_position_history_t;

typedef struct iree_hal_amdgpu_completion_carrier_trace_t {
  // Queue whose reservation and doorbell boundaries are observed.
  iree_hal_amdgpu_host_queue_t* queue;
  // Allocator owning all trace storage.
  iree_allocator_t allocator;
  // Pre-opened trace output, never touched from an observer callback.
  FILE* output_file;
  // KFD SMI descriptor reporting this process's queue lifecycle events.
  int kfd_event_fd;
  // Event descriptor used to stop the KFD observer without polling.
  int kfd_stop_fd;
  // Dedicated owner continuously draining the bounded kernel event FIFO.
  iree_thread_t* kfd_event_thread;
  // Tail buffer retaining the KFD events nearest trace export.
  uint8_t* kfd_events;
  // First retained byte in the circular KFD event buffer.
  uint32_t kfd_event_begin;
  // Number of retained bytes in the circular KFD event buffer.
  uint32_t kfd_event_length;
  // KFD GPU identity used to subscribe to SMI events.
  uint32_t kfd_driver_uid;
  // Queue lifecycle event mask written to the SMI descriptor.
  uint64_t kfd_event_mask;
  // Clock-domain correlation sampled immediately after SMI subscription.
  iree_hal_amdgpu_completion_carrier_clock_sample_t kfd_begin_clocks;
  // Clock-domain correlation sampled after the SMI observer stops.
  iree_hal_amdgpu_completion_carrier_clock_sample_t kfd_end_clocks;
  // First errno observed by the KFD event reader, or zero.
  int kfd_event_error;
  // Errno observed while stopping the KFD event reader, or zero.
  int kfd_stop_error;
  // True when early KFD event bytes were displaced by newer events.
  bool kfd_event_overflow;
  // Hardware and host AQL queue-position history.
  iree_hal_amdgpu_completion_carrier_position_history_t positions;
  // Per-physical-slot reservation and reuse identity.
  iree_hal_amdgpu_completion_carrier_slot_t* slots;
  // Bounded chronological packet ledger.
  iree_hal_amdgpu_completion_carrier_packet_t* packets;
  // Number of populated packet records.
  uint32_t packet_count;
  // Bounded PM4 snapshots referenced by packet records.
  iree_hal_amdgpu_completion_carrier_pm4_t* pm4_records;
  // Number of populated PM4 records.
  uint32_t pm4_record_count;
  // Monotonic reservation identity.
  uint64_t reservation_count;
  // Total number of packets claimed by observed reservations.
  uint64_t reserved_packet_count;
  // Total number of packet IDs covered by observed doorbells.
  uint64_t doorbelled_packet_count;
  // Total notification epochs assigned before observed doorbells.
  uint64_t assigned_epoch_count;
  // Total queue-epoch signal bodies observed before doorbells.
  uint64_t carrier_count;
  // Number of doorbell batches whose carrier and epoch counts differ.
  uint64_t carrier_mismatch_count;
  // First packet ID expected in the next reservation.
  uint64_t next_reservation_packet_id;
  // First packet ID expected in the next doorbell batch.
  uint64_t next_doorbell_packet_id;
  // Notification epoch observed at the preceding doorbell.
  uint64_t last_doorbell_epoch;
  // Number of doorbell operations observed.
  uint64_t doorbell_count;
  // First packet record populated by the active doorbell callback pair.
  uint32_t active_record_begin;
  // Number of packet records populated by the active callback pair.
  uint32_t active_record_count;
  // Number of packets in the active doorbell batch.
  uint64_t active_packet_count;
  // True after the first reservation establishes the expected packet ID.
  bool has_reservation_position;
  // True after the first reservation establishes the doorbell position.
  bool has_doorbell_position;
  // True when reservation IDs were not contiguous.
  bool has_reservation_gap;
  // True when a doorbelled slot lacked the expected reservation identity.
  bool has_unowned_packet;
  // True when the chronological packet ledger filled.
  bool packet_overflow;
  // True when the PM4 sidecar ledger filled.
  bool pm4_overflow;
  // True after the trace has been exported.
  bool has_exported;
} iree_hal_amdgpu_completion_carrier_trace_t;

static bool iree_hal_amdgpu_completion_carrier_trace_is_enabled(void) {
  const char* override = getenv("IREE_AMDGPU_COMPLETION_CARRIER_TRACE");
  if (override && override[0] != '\0' && strcmp(override, "0") != 0) {
    return true;
  }
  const char* test_target = getenv("TEST_TARGET");
  return test_target &&
         strstr(test_target,
                "indirection_access_test_execute_amdgpu_access_test") != NULL;
}

#if defined(IREE_PLATFORM_LINUX)

enum {
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_QUEUE_EVICTION_EVENT = 9,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_QUEUE_RESTORE_EVENT = 10,
};

typedef struct iree_hal_amdgpu_completion_carrier_kfd_smi_events_args_t {
  // KFD GPU identity to observe.
  uint32_t gpu_id;
  // Anonymous event descriptor returned by KFD.
  uint32_t event_fd;
} iree_hal_amdgpu_completion_carrier_kfd_smi_events_args_t;

#define IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_SMI_EVENTS_IOCTL \
  _IOWR('K', 0x1F, iree_hal_amdgpu_completion_carrier_kfd_smi_events_args_t)

static void iree_hal_amdgpu_completion_carrier_trace_sample_clocks(
    iree_hal_amdgpu_completion_carrier_clock_sample_t* out_sample) {
  memset(out_sample, 0, sizeof(*out_sample));
  out_sample->monotonic_before_ns = (uint64_t)iree_time_now();
  struct timespec boottime = {0};
  if (clock_gettime(CLOCK_BOOTTIME, &boottime) != 0) {
    out_sample->error = errno;
  } else {
    out_sample->boottime_ns =
        (uint64_t)boottime.tv_sec * 1000000000ull + (uint64_t)boottime.tv_nsec;
  }
  out_sample->monotonic_after_ns = (uint64_t)iree_time_now();
}

static void iree_hal_amdgpu_completion_carrier_trace_append_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace, const uint8_t* data,
    iree_host_size_t data_length) {
  const uint32_t capacity =
      IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_EVENT_CAPACITY;
  if (data_length >= capacity) {
    memcpy(trace->kfd_events, data + data_length - capacity, capacity);
    trace->kfd_event_begin = 0;
    trace->kfd_event_length = capacity;
    trace->kfd_event_overflow = true;
    return;
  }

  const uint32_t append_length = (uint32_t)data_length;
  const uint32_t available = capacity - trace->kfd_event_length;
  if (append_length > available) {
    const uint32_t displaced_length = append_length - available;
    trace->kfd_event_begin =
        (trace->kfd_event_begin + displaced_length) % capacity;
    trace->kfd_event_length -= displaced_length;
    trace->kfd_event_overflow = true;
  }

  const uint32_t end =
      (trace->kfd_event_begin + trace->kfd_event_length) % capacity;
  const uint32_t first_length = iree_min(append_length, capacity - end);
  memcpy(trace->kfd_events + end, data, first_length);
  memcpy(trace->kfd_events, data + first_length, append_length - first_length);
  trace->kfd_event_length += append_length;
}

static void iree_hal_amdgpu_completion_carrier_trace_drain_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  uint8_t buffer[8192];
  while (trace->kfd_event_error == 0) {
    const ssize_t read_length =
        read(trace->kfd_event_fd, buffer, sizeof(buffer));
    if (read_length > 0) {
      iree_hal_amdgpu_completion_carrier_trace_append_kfd_events(
          trace, buffer, (iree_host_size_t)read_length);
    } else if (read_length == 0) {
      trace->kfd_event_error = EPIPE;
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      break;
    } else {
      trace->kfd_event_error = errno;
    }
  }
}

static int iree_hal_amdgpu_completion_carrier_trace_kfd_thread_main(
    void* user_data) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  struct pollfd poll_fds[2] = {
      {
          .fd = trace->kfd_event_fd,
          .events = POLLIN,
      },
      {
          .fd = trace->kfd_stop_fd,
          .events = POLLIN,
      },
  };
  while (trace->kfd_event_error == 0) {
    int poll_result = 0;
    do {
      poll_result = poll(poll_fds, IREE_ARRAYSIZE(poll_fds), -1);
    } while (poll_result < 0 && errno == EINTR);
    if (poll_result < 0) {
      trace->kfd_event_error = errno;
      break;
    }

    if (poll_fds[0].revents & POLLIN) {
      iree_hal_amdgpu_completion_carrier_trace_drain_kfd_events(trace);
    }
    if (poll_fds[1].revents & POLLIN) {
      iree_hal_amdgpu_completion_carrier_trace_drain_kfd_events(trace);
      break;
    }
    if (poll_fds[0].revents & (POLLERR | POLLHUP | POLLNVAL) ||
        poll_fds[1].revents & (POLLERR | POLLHUP | POLLNVAL)) {
      trace->kfd_event_error = EIO;
      break;
    }
  }
  return 0;
}

static void iree_hal_amdgpu_completion_carrier_trace_stop_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  if (trace->kfd_event_thread) {
    const uint64_t stop_value = 1;
    ssize_t write_length = 0;
    do {
      write_length = write(trace->kfd_stop_fd, &stop_value, sizeof(stop_value));
    } while (write_length < 0 && errno == EINTR);
    if (write_length != (ssize_t)sizeof(stop_value)) {
      trace->kfd_stop_error = write_length < 0 ? errno : EIO;
    }
    iree_thread_release(trace->kfd_event_thread);
    trace->kfd_event_thread = NULL;
  }
  iree_hal_amdgpu_completion_carrier_trace_sample_clocks(
      &trace->kfd_end_clocks);
  if (trace->kfd_event_fd >= 0) {
    if (close(trace->kfd_event_fd) != 0 && trace->kfd_stop_error == 0) {
      trace->kfd_stop_error = errno;
    }
    trace->kfd_event_fd = -1;
  }
  if (trace->kfd_stop_fd >= 0) {
    if (close(trace->kfd_stop_fd) != 0 && trace->kfd_stop_error == 0) {
      trace->kfd_stop_error = errno;
    }
    trace->kfd_stop_fd = -1;
  }
}

static iree_status_t
iree_hal_amdgpu_completion_carrier_trace_initialize_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  hsa_agent_t gpu_agent = {0};
  IREE_RETURN_IF_ERROR(iree_hsa_amd_queue_get_info(
      IREE_LIBHSA(trace->queue->libhsa), trace->queue->hardware_queue,
      HSA_AMD_QUEUE_INFO_AGENT, &gpu_agent));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(trace->queue->libhsa), gpu_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID, &trace->kfd_driver_uid));

  int kfd_fd = -1;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kfd_open(&kfd_fd));
  iree_hal_amdgpu_completion_carrier_kfd_smi_events_args_t args = {
      .gpu_id = trace->kfd_driver_uid,
  };
  iree_status_t status = iree_ok_status();
  if (iree_hal_amdgpu_ioctl(
          kfd_fd, IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_SMI_EVENTS_IOCTL,
          &args) != 0) {
    const int errsv = errno;
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "AMDKFD_IOC_SMI_EVENTS(gpu_id=%" PRIu32
                              ") failed: errno=%d (%s)",
                              trace->kfd_driver_uid, errsv, strerror(errsv));
  } else if (args.event_fd > (uint32_t)INT_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "KFD returned an invalid SMI descriptor %" PRIu32,
                              args.event_fd);
  } else {
    trace->kfd_event_fd = (int)args.event_fd;
  }
  iree_hal_amdgpu_kfd_close(kfd_fd);

  if (iree_status_is_ok(status)) {
    if (fcntl(trace->kfd_event_fd, F_SETFL, O_NONBLOCK) != 0 ||
        fcntl(trace->kfd_event_fd, F_SETFD, FD_CLOEXEC) != 0) {
      const int errsv = errno;
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "failed to configure KFD SMI descriptor: errno=%d (%s)", errsv,
          strerror(errsv));
    }
  }

  if (iree_status_is_ok(status)) {
    trace->kfd_event_mask =
        (1ull << (IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_QUEUE_EVICTION_EVENT -
                  1)) |
        (1ull << (IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_QUEUE_RESTORE_EVENT -
                  1));
    ssize_t write_length = 0;
    do {
      write_length = write(trace->kfd_event_fd, &trace->kfd_event_mask,
                           sizeof(trace->kfd_event_mask));
    } while (write_length < 0 && errno == EINTR);
    if (write_length != (ssize_t)sizeof(trace->kfd_event_mask)) {
      const int errsv = write_length < 0 ? errno : EIO;
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "failed to subscribe KFD queue events: errno=%d (%s)", errsv,
          strerror(errsv));
    }
  }

  if (iree_status_is_ok(status)) {
    trace->kfd_stop_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (trace->kfd_stop_fd < 0) {
      const int errsv = errno;
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "failed to create KFD observer stop event: errno=%d (%s)", errsv,
          strerror(errsv));
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_completion_carrier_trace_sample_clocks(
        &trace->kfd_begin_clocks);
    if (trace->kfd_begin_clocks.error != 0) {
      status =
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "failed to sample CLOCK_BOOTTIME: errno=%d (%s)",
                           trace->kfd_begin_clocks.error,
                           strerror(trace->kfd_begin_clocks.error));
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_thread_create_params_t thread_params = {
        .name = IREE_SV("amdgpu-kfd-smi"),
    };
    status = iree_thread_create(
        iree_hal_amdgpu_completion_carrier_trace_kfd_thread_main, trace,
        thread_params, trace->allocator, &trace->kfd_event_thread);
  }
  return status;
}

#else

static void iree_hal_amdgpu_completion_carrier_trace_stop_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  (void)trace;
}

static iree_status_t
iree_hal_amdgpu_completion_carrier_trace_initialize_kfd_events(
    iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  (void)trace;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU completion-carrier KFD correlation requires Linux");
}

#endif  // IREE_PLATFORM_LINUX

static uint64_t iree_hal_amdgpu_completion_carrier_load_signal(
    const iree_hal_amdgpu_aql_packet_t* packet) {
  uint64_t signal = 0;
  memcpy(&signal, &packet->raw[sizeof(packet->raw) - sizeof(signal)],
         sizeof(signal));
  return signal;
}

static iree_hal_amdgpu_completion_carrier_position_sample_t
iree_hal_amdgpu_completion_carrier_trace_sample_positions(
    iree_hal_amdgpu_completion_carrier_trace_t* trace,
    iree_hal_amdgpu_completion_carrier_position_phase_t phase) {
  iree_hal_amdgpu_completion_carrier_position_history_t* history =
      &trace->positions;
  iree_hal_amdgpu_completion_carrier_position_sample_t sample = {
      .doorbell_sequence = trace->doorbell_count,
      .phase = phase,
  };
  // Sample read before write. Producers can advance both positions while this
  // diagnostic is active, but write is monotonic. A read position that exceeds
  // the subsequently sampled write position therefore cannot be an artifact of
  // sampling a newly consumed packet before its concurrent reservation.
  sample.read_dispatch_id = (uint64_t)iree_atomic_load(
      (iree_atomic_int64_t*)trace->queue->aql_ring.read_dispatch_id,
      iree_memory_order_acquire);
  sample.write_dispatch_id = (uint64_t)iree_atomic_load(
      trace->queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);
  sample.previous_read_dispatch_id = history->sample_count == 0
                                         ? sample.read_dispatch_id
                                         : history->last.read_dispatch_id;
  sample.previous_write_dispatch_id = history->sample_count == 0
                                          ? sample.write_dispatch_id
                                          : history->last.write_dispatch_id;
  if (history->sample_count == 0) {
    history->initial = sample;
  } else if (sample.read_dispatch_id < history->last.read_dispatch_id) {
    if (history->read_regression_count == 0) {
      history->first_read_regression = sample;
    }
    ++history->read_regression_count;
  }
  if (sample.read_dispatch_id > sample.write_dispatch_id) {
    if (history->read_ahead_count == 0) {
      history->first_read_ahead = sample;
    }
    ++history->read_ahead_count;
  }
  if (history->sample_count != 0 &&
      sample.write_dispatch_id < history->last.write_dispatch_id) {
    if (history->write_regression_count == 0) {
      history->first_write_regression = sample;
    }
    ++history->write_regression_count;
  }
  history->max_read_dispatch_id =
      iree_max(history->max_read_dispatch_id, sample.read_dispatch_id);
  history->max_write_dispatch_id =
      iree_max(history->max_write_dispatch_id, sample.write_dispatch_id);
  history->last = sample;
  ++history->sample_count;
  return sample;
}

static void iree_hal_amdgpu_completion_carrier_trace_reserve(
    void* user_data, uint64_t first_packet_id, uint32_t count) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  iree_hal_amdgpu_completion_carrier_trace_sample_positions(
      trace, IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_RESERVE);
  const uint64_t reservation_id = ++trace->reservation_count;
  if (!trace->has_reservation_position) {
    trace->next_reservation_packet_id = first_packet_id;
    trace->next_doorbell_packet_id = first_packet_id;
    trace->has_reservation_position = true;
    trace->has_doorbell_position = true;
  }
  if (first_packet_id != trace->next_reservation_packet_id) {
    trace->has_reservation_gap = true;
  }
  trace->next_reservation_packet_id = first_packet_id + count;
  trace->reserved_packet_count += count;

  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t packet_id = first_packet_id + i;
    iree_hal_amdgpu_completion_carrier_slot_t* slot =
        &trace->slots[packet_id & trace->queue->aql_ring.mask];
    slot->previous_packet_id = slot->packet_id;
    slot->previous_completion_signal = slot->completion_signal;
    slot->packet_id = packet_id;
    slot->reservation_id = reservation_id;
    slot->completion_signal = 0;
    slot->reservation_count = count;
    slot->reservation_offset = i;
  }
}

static void iree_hal_amdgpu_completion_carrier_trace_before_doorbell(
    void* user_data, uint64_t final_packet_id) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  ++trace->doorbell_count;
  iree_hal_amdgpu_completion_carrier_trace_sample_positions(
      trace, IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_BEFORE_DOORBELL);
  trace->active_record_begin = trace->packet_count;
  trace->active_record_count = 0;
  trace->active_packet_count = 0;

  const uint64_t current_epoch =
      trace->queue->notification_ring.epoch.next_submission;
  const uint64_t published_epoch = (uint64_t)iree_atomic_load(
      &trace->queue->notification_ring.epoch.last_published,
      iree_memory_order_acquire);
  const uint64_t batch_epoch_count = current_epoch - trace->last_doorbell_epoch;
  trace->assigned_epoch_count += batch_epoch_count;

  if (!trace->has_doorbell_position ||
      final_packet_id < trace->next_doorbell_packet_id) {
    if (batch_epoch_count != 0) {
      ++trace->carrier_mismatch_count;
    }
    trace->last_doorbell_epoch = current_epoch;
    return;
  }

  const uint64_t first_packet_id = trace->next_doorbell_packet_id;
  const uint64_t packet_count = final_packet_id - first_packet_id + 1;
  trace->active_packet_count = packet_count;
  trace->doorbelled_packet_count += packet_count;

  uint32_t batch_carrier_count = 0;
  const uint64_t epoch_signal =
      trace->queue->notification_ring.epoch.signal.handle;
  for (uint64_t i = 0; i < packet_count; ++i) {
    const uint64_t packet_id = first_packet_id + i;
    const uint32_t slot_index =
        (uint32_t)(packet_id & trace->queue->aql_ring.mask);
    iree_hal_amdgpu_completion_carrier_slot_t* slot = &trace->slots[slot_index];
    iree_hal_amdgpu_aql_packet_t* packet =
        &trace->queue->aql_ring.base[slot_index];
    uint32_t header_setup = 0;
    memcpy(&header_setup, packet->raw, sizeof(header_setup));
    const uint64_t completion_signal =
        iree_hal_amdgpu_completion_carrier_load_signal(packet);
    if (completion_signal == epoch_signal) {
      ++batch_carrier_count;
    }

    const bool has_reservation = slot->packet_id == packet_id;
    if (!has_reservation) {
      trace->has_unowned_packet = true;
    }
    slot->completion_signal = completion_signal;

    if (trace->packet_count >=
        IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY) {
      trace->packet_overflow = true;
      continue;
    }
    iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[trace->packet_count++];
    memset(record, 0, sizeof(*record));
    record->packet_id = packet_id;
    record->reservation_id = has_reservation ? slot->reservation_id : 0;
    record->previous_packet_id =
        has_reservation ? slot->previous_packet_id : UINT64_MAX;
    record->previous_completion_signal =
        has_reservation ? slot->previous_completion_signal : 0;
    record->completion_signal = completion_signal;
    record->first_epoch =
        batch_epoch_count != 0 ? trace->last_doorbell_epoch + 1 : 0;
    record->last_epoch = batch_epoch_count != 0 ? current_epoch : 0;
    record->published_epoch = published_epoch;
    record->doorbell_sequence = trace->doorbell_count;
    record->reservation_count = has_reservation ? slot->reservation_count : 0;
    record->reservation_offset = has_reservation ? slot->reservation_offset : 0;
    record->slot = slot_index;
    record->batch_epoch_count = batch_epoch_count <= UINT32_MAX
                                    ? (uint32_t)batch_epoch_count
                                    : UINT32_MAX;
    record->pm4_record_index = UINT32_MAX;
    record->header = (uint16_t)header_setup;
    record->setup = (uint16_t)(header_setup >> 16);
    record->has_reservation = has_reservation;
    ++trace->active_record_count;
  }

  trace->carrier_count += batch_carrier_count;
  if (batch_carrier_count != batch_epoch_count) {
    ++trace->carrier_mismatch_count;
  }
  for (uint32_t i = 0; i < trace->active_record_count; ++i) {
    trace->packets[trace->active_record_begin + i].batch_carrier_count =
        batch_carrier_count;
  }
  trace->last_doorbell_epoch = current_epoch;
}

static void iree_hal_amdgpu_completion_carrier_trace_after_doorbell(
    void* user_data, uint64_t final_packet_id) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  for (uint32_t i = 0; i < trace->active_record_count; ++i) {
    const uint32_t record_index = trace->active_record_begin + i;
    iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[record_index];
    iree_hal_amdgpu_aql_packet_t* packet =
        &trace->queue->aql_ring.base[record->slot];
    memcpy(record->raw, packet->raw, sizeof(record->raw));

    const uint8_t packet_type =
        (uint8_t)(record->header >> IREE_HSA_PACKET_HEADER_TYPE);
    const uint8_t vendor_format = (uint8_t)record->setup;
    if (packet_type != IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC ||
        vendor_format != IREE_HSA_AMD_AQL_FORMAT_PM4_IB ||
        !trace->queue->pm4_ib_slots) {
      continue;
    }
    if (trace->pm4_record_count >=
        IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY) {
      trace->pm4_overflow = true;
      continue;
    }
    const uint32_t pm4_record_index = trace->pm4_record_count++;
    iree_hal_amdgpu_completion_carrier_pm4_t* pm4_record =
        &trace->pm4_records[pm4_record_index];
    memcpy(pm4_record->dwords, trace->queue->pm4_ib_slots[record->slot].dwords,
           sizeof(pm4_record->dwords));
    record->pm4_record_index = pm4_record_index;
  }
  if (trace->active_packet_count != 0) {
    trace->next_doorbell_packet_id = final_packet_id + 1;
  }
  iree_hal_amdgpu_completion_carrier_trace_sample_positions(
      trace, IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_AFTER_DOORBELL);
  trace->active_record_count = 0;
  trace->active_packet_count = 0;
}

static void iree_hal_amdgpu_completion_carrier_trace_write_hex(
    FILE* file, const void* data, iree_host_size_t data_length) {
  static const char kHex[] = "0123456789abcdef";
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < data_length; ++i) {
    fputc(kHex[bytes[i] >> 4], file);
    fputc(kHex[bytes[i] & 0x0F], file);
  }
}

static void iree_hal_amdgpu_completion_carrier_trace_write_kfd_events(
    FILE* file, const iree_hal_amdgpu_completion_carrier_trace_t* trace) {
  const uint32_t capacity =
      IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_EVENT_CAPACITY;
  uint32_t remaining_length = trace->kfd_event_length;
  uint32_t buffer_offset = trace->kfd_event_begin;
  uint32_t stream_offset = 0;
  uint32_t sequence = 0;
  while (remaining_length != 0) {
    const uint32_t chunk_length =
        iree_min(remaining_length, iree_min(capacity - buffer_offset, 4096u));
    fprintf(file,
            "{\"kind\":\"kfd_event_chunk\",\"sequence\":%u,"
            "\"stream_offset\":%u,\"payload\":\"",
            sequence++, stream_offset);
    iree_hal_amdgpu_completion_carrier_trace_write_hex(
        file, trace->kfd_events + buffer_offset, chunk_length);
    fputs("\"}\n", file);
    remaining_length -= chunk_length;
    stream_offset += chunk_length;
    buffer_offset = (buffer_offset + chunk_length) % capacity;
  }
}

static const char* iree_hal_amdgpu_completion_carrier_position_phase_name(
    iree_hal_amdgpu_completion_carrier_position_phase_t phase) {
  switch (phase) {
    case IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_INITIAL:
      return "initial";
    case IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_RESERVE:
      return "reserve";
    case IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_BEFORE_DOORBELL:
      return "before_doorbell";
    case IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_AFTER_DOORBELL:
      return "after_doorbell";
    case IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_EXPORT:
      return "export";
  }
  return "unknown";
}

static void iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
    FILE* file,
    const iree_hal_amdgpu_completion_carrier_position_sample_t* sample) {
  fprintf(
      file,
      "{\"read\":%" PRIu64 ",\"previous_read\":%" PRIu64 ",\"write\":%" PRIu64
      ",\"previous_write\":%" PRIu64 ",\"doorbell\":%" PRIu64
      ",\"phase\":\"%s\"}",
      sample->read_dispatch_id, sample->previous_read_dispatch_id,
      sample->write_dispatch_id, sample->previous_write_dispatch_id,
      sample->doorbell_sequence,
      iree_hal_amdgpu_completion_carrier_position_phase_name(sample->phase));
}

static void iree_hal_amdgpu_completion_carrier_trace_write_position_history(
    FILE* file,
    const iree_hal_amdgpu_completion_carrier_position_history_t* history) {
  fprintf(file,
          "{\"kind\":\"queue_positions\",\"samples\":%" PRIu64
          ",\"max_read\":%" PRIu64 ",\"max_write\":%" PRIu64
          ",\"read_regressions\":%" PRIu64 ",\"write_regressions\":%" PRIu64
          ",\"read_ahead\":%" PRIu64 ",\"initial\":",
          history->sample_count, history->max_read_dispatch_id,
          history->max_write_dispatch_id, history->read_regression_count,
          history->write_regression_count, history->read_ahead_count);
  iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
      file, &history->initial);
  fputs(",\"final\":", file);
  iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
      file, &history->last);
  fputs(",\"first_read_regression\":", file);
  if (history->read_regression_count != 0) {
    iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
        file, &history->first_read_regression);
  } else {
    fputs("null", file);
  }
  fputs(",\"first_write_regression\":", file);
  if (history->write_regression_count != 0) {
    iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
        file, &history->first_write_regression);
  } else {
    fputs("null", file);
  }
  fputs(",\"first_read_ahead\":", file);
  if (history->read_ahead_count != 0) {
    iree_hal_amdgpu_completion_carrier_trace_write_position_sample(
        file, &history->first_read_ahead);
  } else {
    fputs("null", file);
  }
  fputs("}\n", file);
}

static void iree_hal_amdgpu_completion_carrier_trace_export(
    iree_hal_amdgpu_completion_carrier_trace_t* trace, const char* reason,
    uint64_t completed_epoch, uint64_t published_epoch,
    uint64_t submitted_epoch, uint64_t drained_epoch) {
  if (trace->has_exported) {
    return;
  }
  iree_hal_amdgpu_completion_carrier_trace_sample_positions(
      trace, IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_EXPORT);
  iree_hal_amdgpu_completion_carrier_trace_stop_kfd_events(trace);
  trace->has_exported = true;

  FILE* file = trace->output_file;
  fprintf(file,
          "{\"kind\":\"summary\",\"reason\":\"%s\","
          "\"device\":%" PRIhsz
          ",\"queue\":%u,"
          "\"ring_capacity\":%u,\"epoch_signal\":\"0x%016" PRIx64
          "\",\"completed\":%" PRIu64 ",\"submitted\":%" PRIu64
          ",\"published\":%" PRIu64 ",\"drained\":%" PRIu64
          ",\"reservations\":%" PRIu64 ",\"reserved_packets\":%" PRIu64
          ",\"doorbells\":%" PRIu64 ",\"doorbelled_packets\":%" PRIu64
          ",\"assigned_epochs\":%" PRIu64 ",\"carriers\":%" PRIu64
          ",\"carrier_mismatches\":%" PRIu64
          ",\"reservation_gap\":%s,\"unowned_packet\":%s,"
          "\"packet_overflow\":%s,\"pm4_overflow\":%s}\n",
          reason, trace->queue->device_ordinal,
          (uint32_t)trace->queue->physical_queue_ordinal,
          trace->queue->aql_ring.mask + 1,
          trace->queue->notification_ring.epoch.signal.handle, completed_epoch,
          submitted_epoch, published_epoch, drained_epoch,
          trace->reservation_count, trace->reserved_packet_count,
          trace->doorbell_count, trace->doorbelled_packet_count,
          trace->assigned_epoch_count, trace->carrier_count,
          trace->carrier_mismatch_count,
          trace->has_reservation_gap ? "true" : "false",
          trace->has_unowned_packet ? "true" : "false",
          trace->packet_overflow ? "true" : "false",
          trace->pm4_overflow ? "true" : "false");

  iree_hal_amdgpu_completion_carrier_trace_write_position_history(
      file, &trace->positions);

  fprintf(
      file,
      "{\"kind\":\"kfd_smi\",\"gpu_id\":%" PRIu32
      ",\"event_mask\":\"0x%016" PRIx64
      "\",\"retained_bytes\":%u,\"overflow\":%s,"
      "\"read_errno\":%d,\"stop_errno\":%d,"
      "\"begin_monotonic_before_ns\":%" PRIu64 ",\"begin_boottime_ns\":%" PRIu64
      ",\"begin_monotonic_after_ns\":%" PRIu64
      ",\"begin_clock_errno\":%d,"
      "\"end_monotonic_before_ns\":%" PRIu64 ",\"end_boottime_ns\":%" PRIu64
      ",\"end_monotonic_after_ns\":%" PRIu64 ",\"end_clock_errno\":%d}\n",
      trace->kfd_driver_uid, trace->kfd_event_mask, trace->kfd_event_length,
      trace->kfd_event_overflow ? "true" : "false", trace->kfd_event_error,
      trace->kfd_stop_error, trace->kfd_begin_clocks.monotonic_before_ns,
      trace->kfd_begin_clocks.boottime_ns,
      trace->kfd_begin_clocks.monotonic_after_ns, trace->kfd_begin_clocks.error,
      trace->kfd_end_clocks.monotonic_before_ns,
      trace->kfd_end_clocks.boottime_ns,
      trace->kfd_end_clocks.monotonic_after_ns, trace->kfd_end_clocks.error);
  iree_hal_amdgpu_completion_carrier_trace_write_kfd_events(file, trace);

  for (uint32_t i = 0; i < trace->packet_count; ++i) {
    const iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[i];
    fprintf(file,
            "{\"kind\":\"packet\",\"packet_id\":%" PRIu64
            ",\"slot\":%u,\"reservation_id\":%" PRIu64
            ",\"reservation_offset\":%u,\"reservation_count\":%u,"
            "\"previous_packet_id\":",
            record->packet_id, record->slot, record->reservation_id,
            record->reservation_offset, record->reservation_count);
    if (record->previous_packet_id == UINT64_MAX) {
      fputs("null", file);
    } else {
      fprintf(file, "%" PRIu64, record->previous_packet_id);
    }
    fprintf(file,
            ",\"previous_completion\":\"0x%016" PRIx64
            "\",\"completion\":\"0x%016" PRIx64
            "\",\"header\":\"0x%04x\",\"setup\":\"0x%04x\","
            "\"first_epoch\":%" PRIu64 ",\"last_epoch\":%" PRIu64
            ",\"published_epoch\":%" PRIu64 ",\"doorbell\":%" PRIu64
            ",\"batch_epochs\":%u,"
            "\"batch_carriers\":%u,\"has_reservation\":%s,\"raw\":\"",
            record->previous_completion_signal, record->completion_signal,
            record->header, record->setup, record->first_epoch,
            record->last_epoch, record->published_epoch,
            record->doorbell_sequence, record->batch_epoch_count,
            record->batch_carrier_count,
            record->has_reservation ? "true" : "false");
    iree_hal_amdgpu_completion_carrier_trace_write_hex(file, record->raw,
                                                       sizeof(record->raw));
    fputs("\"", file);
    if (record->pm4_record_index != UINT32_MAX) {
      fputs(",\"pm4\":\"", file);
      const iree_hal_amdgpu_completion_carrier_pm4_t* pm4_record =
          &trace->pm4_records[record->pm4_record_index];
      iree_hal_amdgpu_completion_carrier_trace_write_hex(
          file, pm4_record->dwords, sizeof(pm4_record->dwords));
      fputs("\"", file);
    }
    fputs("}\n", file);
  }

  if (fflush(file) != 0 || ferror(file)) {
    fprintf(stderr, "AMDGPU completion-carrier trace export failed: %s\n",
            strerror(errno));
  } else {
    fprintf(stderr,
            "AMDGPU completion-carrier trace: %s, packets=%u, epochs=%" PRIu64
            ", carriers=%" PRIu64 ", mismatches=%" PRIu64 ", overflow=%s\n",
            reason, trace->packet_count, trace->assigned_epoch_count,
            trace->carrier_count, trace->carrier_mismatch_count,
            (trace->packet_overflow || trace->pm4_overflow) ? "yes" : "no");
  }
}

iree_status_t iree_hal_amdgpu_host_queue_completion_carrier_trace_initialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!iree_hal_amdgpu_completion_carrier_trace_is_enabled()) {
    return iree_ok_status();
  }
  const char* output_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (!output_dir || output_dir[0] == '\0') {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU completion-carrier trace requires "
                            "TEST_UNDECLARED_OUTPUTS_DIR");
  }

  iree_hal_amdgpu_completion_carrier_trace_t* trace = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*trace), (void**)&trace));
  memset(trace, 0, sizeof(*trace));
  trace->queue = queue;
  trace->allocator = queue->host_allocator;
  trace->kfd_event_fd = -1;
  trace->kfd_stop_fd = -1;
  iree_hal_amdgpu_completion_carrier_trace_sample_positions(
      trace, IREE_HAL_AMDGPU_COMPLETION_CARRIER_POSITION_PHASE_INITIAL);

  iree_status_t status =
      iree_allocator_malloc_array(trace->allocator, queue->aql_ring.mask + 1,
                                  sizeof(*trace->slots), (void**)&trace->slots);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        trace->allocator, IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY,
        sizeof(*trace->packets), (void**)&trace->packets);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        trace->allocator, IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY,
        sizeof(*trace->pm4_records), (void**)&trace->pm4_records);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        trace->allocator, IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_EVENT_CAPACITY,
        sizeof(*trace->kfd_events), (void**)&trace->kfd_events);
  }
  if (iree_status_is_ok(status)) {
    memset(trace->slots, 0, (queue->aql_ring.mask + 1) * sizeof(*trace->slots));
    for (uint32_t i = 0; i <= queue->aql_ring.mask; ++i) {
      trace->slots[i].packet_id = UINT64_MAX;
      trace->slots[i].previous_packet_id = UINT64_MAX;
    }
    memset(trace->packets, 0,
           IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY *
               sizeof(*trace->packets));
    memset(trace->pm4_records, 0,
           IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY *
               sizeof(*trace->pm4_records));
    memset(trace->kfd_events, 0,
           IREE_HAL_AMDGPU_COMPLETION_CARRIER_KFD_EVENT_CAPACITY *
               sizeof(*trace->kfd_events));

    char output_path[IREE_MAX_PATH + 1];
    const int output_path_length = snprintf(
        output_path, sizeof(output_path),
        "%s/amdgpu-completion-carriers-%" PRId64 "-d%" PRIhsz "q%u.jsonl",
        output_dir, iree_time_now(), queue->device_ordinal,
        (uint32_t)queue->physical_queue_ordinal);
    if (output_path_length < 0 ||
        (iree_host_size_t)output_path_length >= sizeof(output_path)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "completion-carrier trace path is too long");
    } else {
      trace->output_file = fopen(output_path, "wb");
      if (!trace->output_file) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "failed to open completion-carrier trace '%s': %s",
                             output_path, strerror(errno));
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_completion_carrier_trace_initialize_kfd_events(trace);
  }

  if (iree_status_is_ok(status)) {
    queue->aql_ring.observer = (iree_hal_amdgpu_aql_ring_observer_t){
        .user_data = trace,
        .reserve = iree_hal_amdgpu_completion_carrier_trace_reserve,
        .before_doorbell =
            iree_hal_amdgpu_completion_carrier_trace_before_doorbell,
        .after_doorbell =
            iree_hal_amdgpu_completion_carrier_trace_after_doorbell,
    };
    return iree_ok_status();
  }

  iree_hal_amdgpu_completion_carrier_trace_stop_kfd_events(trace);
  if (trace->output_file) {
    fclose(trace->output_file);
  }
  iree_allocator_free(trace->allocator, trace->kfd_events);
  iree_allocator_free(trace->allocator, trace->pm4_records);
  iree_allocator_free(trace->allocator, trace->packets);
  iree_allocator_free(trace->allocator, trace->slots);
  iree_allocator_free(trace->allocator, trace);
  return status;
}

void iree_hal_amdgpu_host_queue_completion_carrier_trace_record_overrun(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t completed_epoch,
    uint64_t published_epoch, uint64_t submitted_epoch,
    uint64_t drained_epoch) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)
          queue->aql_ring.observer.user_data;
  if (!trace) {
    return;
  }
  iree_hal_amdgpu_completion_carrier_trace_export(
      trace, "overrun", completed_epoch, published_epoch, submitted_epoch,
      drained_epoch);
}

void iree_hal_amdgpu_host_queue_completion_carrier_trace_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)
          queue->aql_ring.observer.user_data;
  if (!trace) {
    return;
  }
  queue->aql_ring.observer = (iree_hal_amdgpu_aql_ring_observer_t){0};

  if (!trace->has_exported) {
    const iree_hal_amdgpu_notification_epoch_snapshot_t snapshot =
        iree_hal_amdgpu_notification_ring_query_epoch_snapshot(
            &queue->notification_ring);
    const uint64_t drained_epoch =
        (uint64_t)iree_atomic_load(&queue->notification_ring.epoch.last_drained,
                                   iree_memory_order_acquire);
    iree_hal_amdgpu_completion_carrier_trace_export(
        trace, "clean", snapshot.completed_epoch, snapshot.published_epoch,
        queue->notification_ring.epoch.next_submission, drained_epoch);
  }

  fclose(trace->output_file);
  iree_allocator_free(trace->allocator, trace->kfd_events);
  iree_allocator_free(trace->allocator, trace->pm4_records);
  iree_allocator_free(trace->allocator, trace->packets);
  iree_allocator_free(trace->allocator, trace->slots);
  iree_allocator_free(trace->allocator, trace);
}
