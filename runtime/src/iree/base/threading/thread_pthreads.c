// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/threading/thread_impl.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/sysfs.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

struct iree_thread_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[32];
  pthread_t handle;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t suspend_count;
  iree_notification_t suspend_barrier;

  // Thread-safe (has its own synchronization).
  iree_thread_override_list_t qos_override_list;
};

static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

static bool iree_thread_resumed_predicate(void* arg) {
  iree_thread_t* thread = (iree_thread_t*)arg;
  return iree_atomic_load(&thread->suspend_count, iree_memory_order_acquire) ==
         0;
}

typedef int (*pthread_setname_np_fn_t)(pthread_t thread, const char* name);

static pthread_setname_np_fn_t iree_pthread_setname_np_fn = NULL;
static void iree_thread_try_query_setname_fn(void) {
  iree_pthread_setname_np_fn =
      (pthread_setname_np_fn_t)dlsym(RTLD_DEFAULT, "pthread_setname_np");
}

static int iree_thread_set_name(pthread_t handle, const char* name) {
  IREE_TRACE_ZONE_BEGIN(z0);
  static iree_once_flag fn_query_flag = IREE_ONCE_FLAG_INIT;
  iree_call_once(&fn_query_flag, iree_thread_try_query_setname_fn);
  int rc;
  if (iree_pthread_setname_np_fn) {
    rc = iree_pthread_setname_np_fn(handle, name);
  } else {
    rc = EINVAL;
  }
  IREE_TRACE_ZONE_END(z0);
  return rc;
}

static void* iree_thread_start_routine(void* param) {
  iree_thread_t* thread = (iree_thread_t*)param;

  // Wait until we resume if we were created suspended.
  while (iree_atomic_load(&thread->suspend_count, iree_memory_order_acquire) >
         0) {
    iree_notification_await(&thread->suspend_barrier,
                            iree_thread_resumed_predicate, thread,
                            iree_infinite_timeout());
  }

  // Set the thread name used by debuggers and tracy (which must be called on
  // the thread).
  iree_thread_set_name(thread->handle, thread->name);
  IREE_TRACE_SET_THREAD_NAME(thread->name);

  // "Consume" the entry info so that we don't see it again (as we don't own
  // its lifetime).
  iree_thread_entry_t entry = thread->entry;
  void* entry_arg = thread->entry_arg;
  thread->entry = NULL;
  thread->entry_arg = NULL;

  // Call the user thread entry point function.
  // Note that this can be a tail-call which saves a stack frame in all threads
  // (which is really just to make call stacks in debuggers much cleaner).
  // Failed startup resumes the native thread with no user entry to invoke.
  return entry ? (void*)((uintptr_t)entry(entry_arg)) : NULL;
}

iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_thread = NULL;

  // Allocate our thread struct; we'll use it to shuttle params into the thread
  // (including the user-specified entry_arg).
  iree_thread_t* thread = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*thread), (void**)&thread);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_atomic_ref_count_init(&thread->ref_count);
  thread->allocator = allocator;
  thread->entry = entry;
  thread->entry_arg = entry_arg;
  iree_strncpy_s(thread->name, IREE_ARRAYSIZE(thread->name), params.name.data,
                 iree_min(params.name.size, IREE_ARRAYSIZE(thread->name) - 1));
  // Keep the entry behind the barrier until native configuration succeeds.
  thread->suspend_count = IREE_ATOMIC_VAR_INIT(1);
  iree_notification_initialize(&thread->suspend_barrier);
  iree_thread_override_list_initialize(iree_thread_set_priority_class,
                                       params.priority_class, thread->allocator,
                                       &thread->qos_override_list);

  pthread_attr_t thread_attr;
  int rc = pthread_attr_init(&thread_attr);
  bool thread_started = false;
  if (rc == 0) {
    // pthread attributes default to joinable threads.
    if (params.stack_size) {
      rc = pthread_attr_setstacksize(&thread_attr, params.stack_size);
    }
    if (rc == 0) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z1, "pthread_create");
      rc = pthread_create(&thread->handle, &thread_attr,
                          &iree_thread_start_routine, thread);
      thread_started = rc == 0;
      IREE_TRACE_ZONE_END(z1);
    }
    pthread_attr_destroy(&thread_attr);
  }
  if (rc != 0) {
    status = iree_make_status(iree_status_code_from_errno(rc),
                              "thread creation failed: %s", strerror(rc));
  } else {
    if (params.priority_class != IREE_THREAD_PRIORITY_CLASS_NORMAL) {
      iree_thread_set_priority_class(thread, params.priority_class);
    }
    status = iree_thread_request_affinity(thread, params.initial_affinity,
                                          params.initial_processor_set);
  }

  if (iree_status_is_ok(status)) {
    *out_thread = thread;
    if (!params.create_suspended) {
      iree_thread_resume(thread);
    }
  } else if (thread_started) {
    // No user code has run. Let the native thread exit, then join it.
    thread->entry = NULL;
    thread->entry_arg = NULL;
    iree_thread_release(thread);
  } else {
    iree_notification_deinitialize(&thread->suspend_barrier);
    iree_thread_override_list_deinitialize(&thread->qos_override_list);
    iree_allocator_free(allocator, thread);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_thread_delete(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_thread_resume(thread);
  if (pthread_equal(thread->handle, pthread_self())) {
    // Self-releasing threads cannot join themselves. Detach so native thread
    // resources are reclaimed automatically after the entry point returns.
    pthread_detach(thread->handle);
  } else {
    pthread_join(thread->handle, NULL);
  }

  iree_notification_deinitialize(&thread->suspend_barrier);
  iree_thread_override_list_deinitialize(&thread->qos_override_list);
  iree_allocator_free(thread->allocator, thread);

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_retain(iree_thread_t* thread) {
  if (thread) {
    iree_atomic_ref_count_inc(&thread->ref_count);
  }
}

void iree_thread_release(iree_thread_t* thread) {
  if (thread && iree_atomic_ref_count_dec(&thread->ref_count) == 1) {
    iree_thread_delete(thread);
  }
}

uintptr_t iree_thread_id(iree_thread_t* thread) {
  return (uintptr_t)thread->handle;
}

// Maps an IREE iree_thread_priority_class_t value to a pthreads priority param.
// The min/max ranges of the priority are implementation dependent so we need to
// do this at runtime.
static struct sched_param iree_thread_sched_param_for_priority_class(
    int policy, iree_thread_priority_class_t priority_class) {
  struct sched_param param;
  memset(&param, 0, sizeof(param));
  int min_priority = sched_get_priority_min(policy);
  int max_priority = sched_get_priority_max(policy);
  int normal_priority = (max_priority - min_priority) / 2 + min_priority;
  switch (priority_class) {
    case IREE_THREAD_PRIORITY_CLASS_LOWEST:
      param.sched_priority = min_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_LOW:
      param.sched_priority =
          (normal_priority - min_priority) / 2 + min_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_NORMAL:
      param.sched_priority = normal_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      param.sched_priority =
          (max_priority - normal_priority) / 2 + normal_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      param.sched_priority = max_priority;
      break;
  }
  return param;
}

// Sets the thread priority to the given |priority_class|, resetting any
// previous value.
//
// NOTE: this probably doesn't work on Android, because Android.
// They seem to use linux LWPs and setpriority/nice on the tid will actually
// change the priority. It doesn't seem possible to elevate priority above
// normal (without root), but it would at least be useful to be able to
// indicate background threads.
//
// See:
// https://stackoverflow.com/questions/17398075/change-native-thread-priority-on-android-in-c-c
// https://android.googlesource.com/platform/frameworks/native/+/android-4.2.2_r1/include/utils/ThreadDefs.h
//
// TODO(benvanik): try this from filament:
// https://github.com/google/filament/blob/56682794d398236c4caa5be40d80acdb73a13bc8/libs/utils/src/JobSystem.cpp
static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_PLATFORM_ANDROID)
  // TODO(benvanik): Some sort of solution on Android, if possible (see above).
#else
  int policy = 0;
  struct sched_param param;
  pthread_getschedparam(thread->handle, &policy, &param);
  param = iree_thread_sched_param_for_priority_class(policy, priority_class);
  pthread_setschedparam(thread->handle, policy, &param);
#endif  // IREE_PLATFORM_ANDROID

  IREE_TRACE_ZONE_END(z0);
}

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_t* override = iree_thread_override_list_add(
      &thread->qos_override_list, thread, priority_class);
  IREE_TRACE_ZONE_END(z0);
  return override;
}

void iree_thread_override_end(iree_thread_override_t* override) {
  if (!override) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_remove_self(override);
  IREE_TRACE_ZONE_END(z0);
}

typedef struct iree_thread_cpu_set_builder_t {
  // Native mask populated by the sysfs CPU-list parser.
  cpu_set_t* set;
  // Native mask capacity in bytes.
  iree_host_size_t size;
  // Optional application budget bounding the requested node's processors.
  const iree_thread_affinity_set_t* processor_set;
  // Whether the node contains CPUs outside the native mask representation.
  bool overflow;
} iree_thread_cpu_set_builder_t;

static bool iree_thread_cpu_set_add_range(uint32_t start_cpu, uint32_t end_cpu,
                                          void* user_data) {
  iree_thread_cpu_set_builder_t* builder = user_data;
  if (builder->processor_set) {
    end_cpu = iree_min(end_cpu, IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  } else if (end_cpu > builder->size * 8) {
    builder->overflow = true;
    return false;
  }
  for (uint32_t cpu = start_cpu; cpu < end_cpu; ++cpu) {
    if (!builder->processor_set ||
        iree_thread_affinity_set_test(builder->processor_set, cpu)) {
      CPU_SET_S(cpu, builder->size, builder->set);
    }
  }
  return true;
}

// Uses the shared sysfs CPU-list parser so NUMA masks have the same numbering
// as topology discovery. An absent or unrepresentable node is an error.
static iree_status_t iree_thread_make_cpu_set_from_node_id(
    uint32_t node_id, const iree_thread_affinity_set_t* processor_set,
    iree_host_size_t native_size, cpu_set_t* out_set) {
  char path[128];
  iree_snprintf(path, sizeof(path), "%s/node/node%u/cpulist",
                iree_sysfs_get_root_path(), node_id);
  char buffer[4096];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length));
  iree_thread_cpu_set_builder_t builder = {
      .set = out_set,
      .size = native_size,
      .processor_set = processor_set,
  };
  IREE_RETURN_IF_ERROR(
      iree_sysfs_parse_cpu_list(iree_make_string_view(buffer, length),
                                iree_thread_cpu_set_add_range, &builder));
  if (builder.overflow) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node %u CPU mask exceeds %u CPUs", node_id,
                            (unsigned)(native_size * 8));
  }
  return iree_ok_status();
}

#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
// Formats CPU ranges into bounded diagnostic storage. Large sparse sets include
// an explicit ellipsis and total CPU count instead of allocating scratch space.
// Both callers provide at least 256 bytes for ranges and the omission suffix.
static void iree_thread_format_cpu_set(const cpu_set_t* set,
                                       iree_host_size_t native_size,
                                       char* buffer,
                                       iree_host_size_t buffer_capacity) {
  iree_host_size_t length = 0;
  char suffix[48];
  iree_snprintf(suffix, sizeof(suffix), ",... (%d CPUs total)",
                CPU_COUNT_S(native_size, set));
  for (iree_host_size_t cpu = 0; cpu < native_size * 8; ++cpu) {
    if (!CPU_ISSET_S(cpu, native_size, set)) {
      continue;
    }
    const iree_host_size_t start = cpu;
    while (cpu + 1 < native_size * 8 &&
           CPU_ISSET_S(cpu + 1, native_size, set)) {
      ++cpu;
    }
    char range[48];
    iree_host_size_t range_length = 0;
    if (cpu == start) {
      range_length = iree_snprintf(range, sizeof(range), "%s%" PRIhsz,
                                   length ? "," : "", start);
    } else {
      range_length =
          iree_snprintf(range, sizeof(range), "%s%" PRIhsz "-%" PRIhsz,
                        length ? "," : "", start, cpu);
    }
    if (length + range_length + sizeof(suffix) > buffer_capacity) {
      iree_snprintf(buffer + length, buffer_capacity - length, "%s", suffix);
      return;
    }
    memcpy(buffer + length, range, range_length);
    length += range_length;
  }
  if (length) {
    buffer[length] = 0;
  } else {
    iree_snprintf(buffer, buffer_capacity, "none");
  }
}

// Queries only the target thread. Its current affinity may include inherited
// application policy and is not the enclosing cgroup's full CPU allowance.
static void iree_thread_format_observed_affinity(
    iree_thread_t* thread, char* buffer, iree_host_size_t buffer_capacity) {
  const iree_host_size_t native_size =
      CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  cpu_set_t* observed_set = iree_alloca(native_size);
  CPU_ZERO_S(native_size, observed_set);
  int rc = 0;
#if defined(IREE_PLATFORM_ANDROID)
#if __ANDROID_API__ >= 21
  rc = sched_getaffinity(pthread_gettid_np(thread->handle), native_size,
                         observed_set) == 0
           ? 0
           : errno;
#else
  rc = ENOSYS;
#endif  // __ANDROID_API__ >= 21
#else
  rc = pthread_getaffinity_np(thread->handle, native_size, observed_set);
#endif  // IREE_PLATFORM_ANDROID
  if (rc != 0) {
    iree_snprintf(buffer, buffer_capacity, "unavailable (errno %d)", rc);
  } else {
    iree_thread_format_cpu_set(observed_set, native_size, buffer,
                               buffer_capacity);
  }
}
#endif  // IREE_STATUS_FEATURE_ANNOTATIONS

static iree_status_t iree_thread_affinity_failure(
    iree_thread_t* thread, iree_thread_affinity_t affinity,
    const cpu_set_t* requested_set, iree_host_size_t native_size, int error) {
#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
  char requested[256];
  iree_thread_format_cpu_set(requested_set, native_size, requested,
                             sizeof(requested));
  char observed[256];
  iree_thread_format_observed_affinity(thread, observed, sizeof(observed));
  char node[64] = {0};
  if (affinity.group_any) {
    iree_snprintf(node, sizeof(node), "requested NUMA node %u; ",
                  affinity.group);
  }
  return iree_make_status(
      iree_status_code_from_errno(error),
      "cannot apply thread affinity: %s (errno %d); thread '%s'; %s"
      "requested CPUs [%s]; observed thread affinity after rejection [%s]",
      strerror(error), error, thread->name, node, requested, observed);
#else
  return iree_make_status(iree_status_code_from_errno(error),
                          "cannot apply thread affinity: %s (errno %d)",
                          strerror(error), error);
#endif  // IREE_STATUS_FEATURE_ANNOTATIONS
}

iree_status_t iree_thread_request_affinity(
    iree_thread_t* thread, iree_thread_affinity_t affinity,
    const iree_thread_affinity_set_t* processor_set) {
  if (iree_thread_affinity_is_unspecified(affinity) && !processor_set) {
    return iree_ok_status();
  }
  const iree_host_size_t native_size =
      CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  cpu_set_t* native_set = iree_alloca(native_size);
  CPU_ZERO_S(native_size, native_set);
  iree_status_t status = iree_ok_status();
  if (affinity.group_any) {
    status = iree_thread_make_cpu_set_from_node_id(
        affinity.group, processor_set, native_size, native_set);
    if (!iree_status_is_ok(status)) {
      status =
          iree_status_annotate_f(status, "thread '%s'; requested NUMA node %u",
                                 thread->name, affinity.group);
    }
  } else if (affinity.id_assigned) {
    if (!processor_set && affinity.id + affinity.smt >= native_size * 8) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "thread '%s'; CPU %u exceeds native affinity mask",
                           thread->name, affinity.id);
    } else {
      for (uint32_t cpu = affinity.id; cpu <= affinity.id + affinity.smt;
           ++cpu) {
        if (!processor_set ||
            iree_thread_affinity_set_test(processor_set, cpu)) {
          CPU_SET_S(cpu, native_size, native_set);
        }
      }
    }
  } else {
    for (iree_host_size_t cpu =
             iree_thread_affinity_set_find_first_set(processor_set, 0);
         cpu < IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
         cpu =
             iree_thread_affinity_set_find_first_set(processor_set, cpu + 1)) {
      CPU_SET_S(cpu, native_size, native_set);
    }
  }

  if (iree_status_is_ok(status)) {
    int rc = 0;
#if defined(IREE_PLATFORM_ANDROID)
#if __ANDROID_API__ >= 21
    rc = sched_setaffinity(pthread_gettid_np(thread->handle), native_size,
                           native_set) == 0
             ? 0
             : errno;
#else
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "thread affinity requires Android API 21");
#endif  // __ANDROID_API__ >= 21
#else
    rc = pthread_setaffinity_np(thread->handle, native_size, native_set);
#endif  // IREE_PLATFORM_ANDROID
    if (rc != 0) {
      status = iree_thread_affinity_failure(thread, affinity, native_set,
                                            native_size, rc);
    }
  }
  return status;
}

void iree_thread_resume(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_atomic_exchange(&thread->suspend_count, 0,
                           iree_memory_order_acq_rel) == 1) {
    iree_notification_post(&thread->suspend_barrier, IREE_ALL_WAITERS);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_join(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  pthread_join(thread->handle, NULL);
  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_yield(void) { sched_yield(); }

#endif  // IREE_PLATFORM_*
