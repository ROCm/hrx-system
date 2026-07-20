// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process_platform.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS) || \
    defined(IREE_PLATFORM_ANDROID)

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "loom/target/tool/process_posix.h"

extern char** environ;

typedef struct loom_tool_capture_file_t {
  // Open file descriptor passed to the child process.
  int fd;
} loom_tool_capture_file_t;

static const char* loom_tool_temp_directory(void) {
  const char* temp_directory = getenv("TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  return "/tmp";
}

static iree_status_t loom_tool_posix_status(int error_number,
                                            const char* message) {
  return iree_make_status(iree_status_code_from_errno(error_number),
                          "%s (%d: %s)", message, error_number,
                          strerror(error_number));
}

static iree_status_t loom_tool_posix_set_close_on_exec(int fd) {
  int flags = -1;
  do {
    flags = fcntl(fd, F_GETFD);
  } while (flags < 0 && errno == EINTR);
  if (flags < 0) {
    return loom_tool_posix_status(errno, "failed to query descriptor flags");
  }
  int set_result = -1;
  do {
    set_result = fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  } while (set_result < 0 && errno == EINTR);
  if (set_result < 0) {
    return loom_tool_posix_status(errno,
                                  "failed to set descriptor close-on-exec");
  }
  return iree_ok_status();
}

static iree_status_t loom_tool_capture_file_open(
    const char* stem, loom_tool_capture_file_t* out_file) {
  out_file->fd = -1;
  char template_path[4096] = {0};
  int length =
      iree_snprintf(template_path, sizeof(template_path),
                    "%s/loom_tool_%s_XXXXXX", loom_tool_temp_directory(), stem);
  if (length < 0 || (iree_host_size_t)length >= sizeof(template_path)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary capture path is too long");
  }
  int fd = mkstemp(template_path);
  if (fd < 0) {
    return loom_tool_posix_status(errno, "failed to open capture file");
  }
  if (unlink(template_path) < 0) {
    int error_number = errno;
    close(fd);
    return loom_tool_posix_status(error_number,
                                  "failed to unlink capture file");
  }
  iree_status_t status = loom_tool_posix_set_close_on_exec(fd);
  if (!iree_status_is_ok(status)) {
    close(fd);
    return status;
  }
  out_file->fd = fd;
  return iree_ok_status();
}

static void loom_tool_capture_file_deinitialize(
    loom_tool_capture_file_t* file) {
  if (file->fd >= 0) {
    close(file->fd);
    file->fd = -1;
  }
}

static iree_status_t loom_tool_capture_file_read(
    loom_tool_capture_file_t* file, iree_allocator_t allocator,
    loom_tool_output_t* out_output) {
  *out_output = (loom_tool_output_t){0};
  if (lseek(file->fd, 0, SEEK_SET) < 0) {
    return loom_tool_posix_status(errno, "failed to rewind capture file");
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  char buffer[4096];
  bool reading = true;
  while (reading && iree_status_is_ok(status)) {
    ssize_t bytes_read = read(file->fd, buffer, sizeof(buffer));
    if (bytes_read > 0) {
      status = iree_string_builder_append_string(
          &builder,
          iree_make_string_view(buffer, (iree_host_size_t)bytes_read));
    } else if (bytes_read == 0) {
      reading = false;
    } else if (errno != EINTR) {
      status = loom_tool_posix_status(errno, "failed to read capture file");
    }
  }
  if (iree_status_is_ok(status)) {
    out_output->length = iree_string_builder_size(&builder);
    out_output->data = iree_string_builder_take_storage(&builder);
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(out_output, allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_tool_process_wait(pid_t pid, int* out_exit_code) {
  *out_exit_code = 1;
  int status = 0;
  pid_t wait_result = -1;
  do {
    wait_result = waitpid(pid, &status, 0);
  } while (wait_result < 0 && errno == EINTR);
  if (wait_result < 0) {
    return loom_tool_posix_status(errno, "failed to wait for tool process");
  }
  if (WIFEXITED(status)) {
    *out_exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    *out_exit_code = 128 + WTERMSIG(status);
  } else {
    *out_exit_code = 1;
  }
  return iree_ok_status();
}

typedef struct loom_tool_spawn_options_t {
  // Child descriptor actions.
  posix_spawn_file_actions_t file_actions;
  // Whether |file_actions| must be destroyed.
  bool file_actions_initialized;
  // OS-specific child descriptor inheritance policy.
  loom_tool_posix_spawn_policy_t policy;
} loom_tool_spawn_options_t;

static void loom_tool_spawn_options_deinitialize(
    loom_tool_spawn_options_t* options) {
  if (options->policy.descriptor_directory != NULL) {
    int close_result = closedir(options->policy.descriptor_directory);
    IREE_ASSERT(close_result == 0);
    (void)close_result;
  }
  if (options->policy.attributes_initialized) {
    int destroy_result = posix_spawnattr_destroy(&options->policy.attributes);
    IREE_ASSERT(destroy_result == 0);
    (void)destroy_result;
  }
  if (options->file_actions_initialized) {
    int destroy_result =
        posix_spawn_file_actions_destroy(&options->file_actions);
    IREE_ASSERT(destroy_result == 0);
    (void)destroy_result;
  }
  memset(options, 0, sizeof(*options));
}

static iree_status_t loom_tool_spawn_options_initialize(
    int stdout_fd, int stderr_fd, loom_tool_spawn_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  int spawn_result = posix_spawn_file_actions_init(&out_options->file_actions);
  if (spawn_result != 0) {
    return loom_tool_posix_status(spawn_result,
                                  "failed to initialize process spawn actions");
  }
  out_options->file_actions_initialized = true;

  spawn_result = posix_spawn_file_actions_addopen(
      &out_options->file_actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  if (spawn_result != 0) {
    return loom_tool_posix_status(spawn_result,
                                  "failed to disconnect child stdin");
  }
  spawn_result = posix_spawn_file_actions_adddup2(&out_options->file_actions,
                                                  stdout_fd, STDOUT_FILENO);
  if (spawn_result != 0) {
    return loom_tool_posix_status(spawn_result, "failed to redirect stdout");
  }
  spawn_result = posix_spawn_file_actions_adddup2(&out_options->file_actions,
                                                  stderr_fd, STDERR_FILENO);
  if (spawn_result != 0) {
    return loom_tool_posix_status(spawn_result, "failed to redirect stderr");
  }
  return loom_tool_posix_spawn_policy_initialize(&out_options->file_actions,
                                                 &out_options->policy);
}

iree_status_t loom_tool_process_run_platform(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  *out_result = (loom_tool_process_result_t){0};

  loom_tool_capture_file_t stdout_file;
  loom_tool_capture_file_t stderr_file;
  IREE_RETURN_IF_ERROR(loom_tool_capture_file_open("stdout", &stdout_file));
  iree_status_t status = loom_tool_capture_file_open("stderr", &stderr_file);

  loom_tool_spawn_options_t spawn_options;
  if (iree_status_is_ok(status)) {
    status = loom_tool_spawn_options_initialize(stdout_file.fd, stderr_file.fd,
                                                &spawn_options);
  } else {
    memset(&spawn_options, 0, sizeof(spawn_options));
  }
  pid_t pid = 0;
  if (iree_status_is_ok(status)) {
    const posix_spawnattr_t* attributes =
        spawn_options.policy.attributes_initialized
            ? &spawn_options.policy.attributes
            : NULL;
    int spawn_result = 0;
    if (search_path) {
      spawn_result = posix_spawnp(&pid, argv[0], &spawn_options.file_actions,
                                  attributes, argv, environ);
    } else {
      spawn_result = posix_spawn(&pid, argv[0], &spawn_options.file_actions,
                                 attributes, argv, environ);
    }
    if (spawn_result != 0) {
      status =
          loom_tool_posix_status(spawn_result, "failed to spawn tool process");
    }
  }
  loom_tool_spawn_options_deinitialize(&spawn_options);
  if (iree_status_is_ok(status)) {
    status = loom_tool_process_wait(pid, &out_result->exit_code);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stdout_file, allocator,
                                         &out_result->stdout_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stderr_file, allocator,
                                         &out_result->stderr_bytes);
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_process_result_deinitialize(out_result, allocator);
  }
  loom_tool_capture_file_deinitialize(&stderr_file);
  loom_tool_capture_file_deinitialize(&stdout_file);
  return status;
}

iree_status_t loom_tool_temp_file_initialize_platform(
    const char* stem_buffer, loom_tool_temp_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));

  int length = iree_snprintf(out_file->path, sizeof(out_file->path),
                             "%s/loom_tool_%s_XXXXXX",
                             loom_tool_temp_directory(), stem_buffer);
  if (length < 0 || (iree_host_size_t)length >= sizeof(out_file->path)) {
    memset(out_file, 0, sizeof(*out_file));
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary file path is too long");
  }
  int fd = mkstemp(out_file->path);
  if (fd < 0) {
    int error_number = errno;
    memset(out_file, 0, sizeof(*out_file));
    return loom_tool_posix_status(error_number,
                                  "failed to open temporary file");
  }
  iree_status_t status = loom_tool_posix_set_close_on_exec(fd);
  if (!iree_status_is_ok(status)) {
    close(fd);
    unlink(out_file->path);
    memset(out_file, 0, sizeof(*out_file));
    return status;
  }
  if (close(fd) < 0 && errno != EINTR) {
    int error_number = errno;
    unlink(out_file->path);
    memset(out_file, 0, sizeof(*out_file));
    return loom_tool_posix_status(error_number,
                                  "failed to close temporary file");
  }
  return iree_ok_status();
}

iree_status_t loom_tool_temp_file_deinitialize_platform(
    loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (file->path[0] != '\0') {
    if (unlink(file->path) < 0 && errno != ENOENT) {
      status = loom_tool_posix_status(errno, "failed to delete temporary file");
    }
  }
  memset(file, 0, sizeof(*file));
  return status;
}

#endif  // POSIX platforms
