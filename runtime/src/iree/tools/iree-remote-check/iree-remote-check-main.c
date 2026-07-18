// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/path.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer_transfer.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/transport_factory.h"
#include "iree/net/transport_factory.h"
#include "iree/tools/iree-remote-check/artifact.h"

IREE_FLAG(string, device, "remote-tcp://127.0.0.1:5000",
          "Remote HAL device URI to connect to.");

#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT) && \
    defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_USAGE                               \
  "Embedded artifacts cover the configured AMDGPU targets and Vulkan " \
  "1.3 BDA.\n"                                                         \
  "\n"                                                                 \
  "Alternative Vulkan server:\n"                                       \
  "  iree-serve-device --device=vulkan://\n"                           \
  "\n"                                                                 \
  "Quick start (two terminals):\n"                                     \
  "  iree-serve-device --device=amdgpu://0\n"                          \
  "  iree-remote-check --device=remote-tcp://127.0.0.1:5000\n"
#elif defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_USAGE                      \
  "Embedded artifacts cover the configured AMDGPU targets.\n" \
  "\n"                                                        \
  "Quick start (two terminals):\n"                            \
  "  iree-serve-device --device=amdgpu://0\n"                 \
  "  iree-remote-check --device=remote-tcp://127.0.0.1:5000\n"
#elif defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_USAGE       \
  "Embedded artifacts cover Vulkan 1.3 BDA.\n" \
  "\n"                                         \
  "Quick start (two terminals):\n"             \
  "  iree-serve-device --device=vulkan://\n"   \
  "  iree-remote-check --device=remote-tcp://127.0.0.1:5000\n"
#else
#define IREE_REMOTE_CHECK_ARTIFACT_USAGE \
  "This build contains no backend executable artifacts.\n"
#endif

typedef struct iree_remote_check_connection_state_t {
  // Posted after the connection callback stores its status.
  iree_notification_t notification;
  // Status transferred from the connection callback.
  iree_status_t status;
  // Nonzero after |status| is available.
  iree_atomic_int32_t fired;
} iree_remote_check_connection_state_t;

static bool iree_remote_check_connection_fired(void* user_data) {
  iree_remote_check_connection_state_t* state =
      (iree_remote_check_connection_state_t*)user_data;
  return iree_atomic_load(&state->fired, iree_memory_order_acquire) != 0;
}

static void iree_remote_check_on_connected(void* user_data,
                                           iree_status_t status) {
  iree_remote_check_connection_state_t* state =
      (iree_remote_check_connection_state_t*)user_data;
  state->status = status;
  iree_atomic_store(&state->fired, 1, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static iree_status_t iree_remote_check_connect(iree_hal_device_t* device,
                                               iree_string_view_t device_uri) {
  iree_remote_check_connection_state_t state;
  memset(&state, 0, sizeof(state));
  iree_notification_initialize(&state.notification);
  state.status = iree_ok_status();

  const iree_hal_remote_client_device_connected_callback_t callback = {
      .fn = iree_remote_check_on_connected,
      .user_data = &state,
  };
  iree_status_t status =
      iree_hal_remote_client_device_connect(device, callback);
  if (iree_status_is_ok(status)) {
    const bool connected = iree_notification_await(
        &state.notification, iree_remote_check_connection_fired, &state,
        iree_infinite_timeout());
    if (connected) {
      status = state.status;
      state.status = iree_ok_status();
    } else {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "remote connection did not complete");
    }
  }
  iree_notification_deinitialize(&state.notification);

  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status,
        "connecting to '%.*s'; start iree-serve-device and use the client URI "
        "it reports after binding",
        (int)device_uri.size, device_uri.data);
  }
  return status;
}

typedef struct iree_remote_check_error_state_t {
  // Guards ownership of |status| across the proactor and application threads.
  iree_slim_mutex_t mutex;
  // Joined terminal errors transferred from the remote device callback.
  iree_status_t status;
} iree_remote_check_error_state_t;

static void iree_remote_check_error_state_initialize(
    iree_remote_check_error_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  iree_slim_mutex_initialize(&out_state->mutex);
  out_state->status = iree_ok_status();
}

static iree_status_t iree_remote_check_error_state_take(
    iree_remote_check_error_state_t* state) {
  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = state->status;
  state->status = iree_ok_status();
  iree_slim_mutex_unlock(&state->mutex);
  return status;
}

static void iree_remote_check_error_state_deinitialize(
    iree_remote_check_error_state_t* state) {
  IREE_ASSERT(iree_status_is_ok(state->status));
  iree_slim_mutex_deinitialize(&state->mutex);
}

static void iree_remote_check_on_device_error(void* user_data,
                                              iree_status_t status) {
  iree_remote_check_error_state_t* state =
      (iree_remote_check_error_state_t*)user_data;
  iree_slim_mutex_lock(&state->mutex);
  state->status = iree_status_join(state->status, status);
  iree_slim_mutex_unlock(&state->mutex);
}

typedef struct iree_remote_check_deactivation_state_t {
  // Posted after all remote device callbacks and network operations drain.
  iree_notification_t notification;
  // Nonzero after terminal deactivation completes.
  iree_atomic_int32_t fired;
} iree_remote_check_deactivation_state_t;

static bool iree_remote_check_deactivation_fired(void* user_data) {
  iree_remote_check_deactivation_state_t* state =
      (iree_remote_check_deactivation_state_t*)user_data;
  return iree_atomic_load(&state->fired, iree_memory_order_acquire) != 0;
}

static void iree_remote_check_on_deactivated(void* user_data) {
  iree_remote_check_deactivation_state_t* state =
      (iree_remote_check_deactivation_state_t*)user_data;
  iree_atomic_store(&state->fired, 1, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static iree_status_t iree_remote_check_deactivate(iree_hal_device_t* device) {
  iree_remote_check_deactivation_state_t state;
  memset(&state, 0, sizeof(state));
  iree_notification_initialize(&state.notification);

  const iree_hal_remote_client_device_deactivated_callback_t callback = {
      .fn = iree_remote_check_on_deactivated,
      .user_data = &state,
  };
  iree_status_t status =
      iree_hal_remote_client_device_deactivate(device, callback);
  if (iree_status_is_ok(status)) {
    const bool deactivated = iree_notification_await(
        &state.notification, iree_remote_check_deactivation_fired, &state,
        iree_infinite_timeout());
    if (!deactivated) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "remote device deactivation did not complete");
    }
  }

  iree_notification_deinitialize(&state.notification);
  return status;
}

static const char* iree_remote_check_target_kind_name(
    iree_hal_executable_target_kind_t kind) {
  switch (kind) {
    case IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT:
      return "exact";
    case IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC:
      return "generic";
    default:
      return "unknown";
  }
}

static void iree_remote_check_print_device_spec(
    const iree_hal_device_spec_t* device_spec) {
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  fprintf(stdout, "connected: %.*s\n", (int)identity->display_name.size,
          identity->display_name.data);
  fprintf(stdout, "executable targets:\n");
  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* target = &executable_spec->targets[i];
    fprintf(stdout, "  %.*s / %.*s (%s)\n", (int)target->family.size,
            target->family.data, (int)target->target_key.size,
            target->target_key.data,
            iree_remote_check_target_kind_name(target->kind));
  }
}

static void iree_remote_check_print_values(const char* label,
                                           const int32_t values[4]) {
  fprintf(stdout, "%s: [%" PRId32 ", %" PRId32 ", %" PRId32 ", %" PRId32 "]\n",
          label, values[0], values[1], values[2], values[3]);
}

static iree_status_t iree_remote_check_execute(iree_hal_device_t* device) {
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);
  iree_remote_check_print_device_spec(device_spec);

  iree_remote_check_artifact_t artifact;
  iree_status_t status =
      iree_remote_check_select_artifact(device_spec, &artifact);
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "artifact: %.*s (%.*s / %.*s)\n",
            (int)artifact.file_name.size, artifact.file_name.data,
            (int)artifact.executable_target->family.size,
            artifact.executable_target->family.data,
            (int)artifact.artifact_target_key.size,
            artifact.artifact_target_key.data);
  }

  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data = artifact.executable_data;
    status = iree_status_annotate(
        iree_hal_device_load_executable(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                        artifact.executable_target,
                                        &load_params, &executable),
        IREE_SV("uploading native executable"));
  }

  iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  if (iree_status_is_ok(status)) {
    status =
        iree_status_annotate(iree_hal_executable_lookup_function_by_name(
                                 executable, artifact.entry_point, &function),
                             IREE_SV("looking up diagnostic entry point"));
  }

  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
  iree_hal_buffer_t* input_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                           4 * sizeof(int32_t), &input_buffer),
        IREE_SV("allocating input buffer"));
  }
  iree_hal_buffer_t* output_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                           4 * sizeof(int32_t), &output_buffer),
        IREE_SV("allocating output buffer"));
  }

  const int32_t input[4] = {1, -2, 30, 400};
  const int32_t expected[4] = {8, 5, 37, 407};
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_hal_device_transfer_h2d(
            device, input, input_buffer, /*target_offset=*/0, sizeof(input),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()),
        IREE_SV("uploading input values"));
  }

  iree_hal_semaphore_t* semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }

  iree_hal_buffer_ref_t binding_refs[2];
  memset(binding_refs, 0, sizeof(binding_refs));
  if (iree_status_is_ok(status)) {
    binding_refs[0] = iree_hal_make_buffer_ref(
        input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
    binding_refs[1] =
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer));
    const iree_hal_buffer_ref_list_t bindings = {
        .count = IREE_ARRAYSIZE(binding_refs),
        .values = binding_refs,
    };
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &semaphore,
        .payload_values = &signal_value,
    };
    status = iree_status_annotate(
        iree_hal_device_queue_dispatch(
            device, IREE_HAL_QUEUE_AFFINITY_ANY,
            iree_hal_semaphore_list_empty(), signal_semaphores, executable,
            function, artifact.dispatch_config, artifact.dispatch_constants,
            bindings, IREE_HAL_DISPATCH_FLAG_NONE),
        IREE_SV("dispatching native executable"));
    if (iree_status_is_ok(status)) {
      status = iree_status_annotate(
          iree_hal_semaphore_wait(semaphore, signal_value,
                                  iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE),
          IREE_SV("waiting for remote dispatch"));
    }
  }

  int32_t actual[4] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_hal_device_transfer_d2h(
            device, output_buffer, /*source_offset=*/0, actual, sizeof(actual),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()),
        IREE_SV("downloading output values"));
  }
  if (iree_status_is_ok(status) &&
      memcmp(actual, expected, sizeof(expected)) != 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "remote dispatch returned [%" PRId32 ", %" PRId32
                         ", %" PRId32 ", %" PRId32 "]; expected [%" PRId32
                         ", %" PRId32 ", %" PRId32 ", %" PRId32 "]",
                         actual[0], actual[1], actual[2], actual[3],
                         expected[0], expected[1], expected[2], expected[3]);
  }

  if (iree_status_is_ok(status)) {
    iree_remote_check_print_values("input", input);
    fprintf(stdout, "operation: output[i] = input[i] + 7\n");
    iree_remote_check_print_values("output", actual);
    fprintf(stdout, "PASS: native executable dispatch completed remotely\n");
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
  iree_hal_executable_release(executable);
  return status;
}

static iree_status_t iree_remote_check_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_remote_check_error_state_t error_state;
  iree_remote_check_error_state_initialize(&error_state);

  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &proactor_pool);

  const iree_string_view_t device_uri = iree_make_cstring_view(FLAG_device);
  const iree_string_view_t driver_name = iree_uri_schema(device_uri);
  iree_net_transport_factory_t* transport_factory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_transport_factory_create(
        driver_name, host_allocator, &transport_factory);
  }

  iree_hal_driver_t* driver = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_driver_options_t options;
    iree_hal_remote_client_driver_options_initialize(&options);
    options.transport_factory = transport_factory;
    options.default_device_options.error_callback.fn =
        iree_remote_check_on_device_error;
    options.default_device_options.error_callback.user_data = &error_state;
    status = iree_hal_remote_client_driver_create(driver_name, &options,
                                                  host_allocator, &driver);
  }

  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_device_by_uri(
        driver, device_uri, &create_params, host_allocator, &device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_remote_check_connect(device, device_uri);
  }

  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &frontier_tracker);
  }
  iree_hal_device_group_t* device_group = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, host_allocator, &device_group);
  }
  if (iree_status_is_ok(status)) {
    status = iree_remote_check_execute(device);
  }

  if (device) {
    status = iree_status_join(status, iree_remote_check_deactivate(device));
  }
  iree_hal_device_group_release(device_group);
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_net_transport_factory_release(transport_factory);
  iree_async_proactor_pool_release(proactor_pool);

  status = iree_status_join(status,
                            iree_remote_check_error_state_take(&error_state));
  iree_remote_check_error_state_deinitialize(&error_state);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-remote-check",
      "Uploads and dispatches a backend-native executable through remote HAL.\n"
      "The client links only the remote HAL implementation; native driver "
      "code\n"
      "runs exclusively in "
      "iree-serve-device.\n" IREE_REMOTE_CHECK_ARTIFACT_USAGE);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_remote_check_run();
  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
