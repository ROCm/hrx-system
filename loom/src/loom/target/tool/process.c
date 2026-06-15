// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)

#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#else  // POSIX

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_ANDROID)
#define LOOM_TOOL_PROCESS_POSIX 1
#endif

#if LOOM_TOOL_PROCESS_POSIX
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif  // LOOM_TOOL_PROCESS_POSIX

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// String and argv utilities
//===----------------------------------------------------------------------===//

static bool loom_tool_string_view_contains_nul(iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] == '\0') {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_tool_dup_cstring(iree_string_view_t value,
                                           iree_allocator_t allocator,
                                           char** out_string) {
  *out_string = NULL;
  if (loom_tool_string_view_contains_nul(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool arguments cannot contain NUL bytes");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string size overflow");
  }
  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&string));
  if (value.size != 0) {
    memcpy(string, value.data, value.size);
  }
  string[value.size] = '\0';
  *out_string = string;
  return iree_ok_status();
}

static void loom_tool_argv_deinitialize(char** argv,
                                        iree_allocator_t allocator) {
  if (argv == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; argv[i] != NULL; ++i) {
    iree_allocator_free(allocator, argv[i]);
  }
  iree_allocator_free(allocator, argv);
}

static iree_status_t loom_tool_argv_allocate(
    iree_string_view_t executable_path, const iree_string_view_t* arguments,
    iree_host_size_t argument_count, iree_allocator_t allocator,
    char*** out_argv) {
  *out_argv = NULL;
  if (iree_string_view_is_empty(executable_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool executable path is required");
  }

  iree_host_size_t argv_count = 0;
  iree_host_size_t total_count = 0;
  if (!iree_host_size_checked_add(argument_count, 1, &argv_count) ||
      !iree_host_size_checked_add(argv_count, 1, &total_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "argv size overflow");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(total_count, sizeof(char*), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "argv allocation size overflow");
  }

  char** argv = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&argv));
  memset(argv, 0, total_size);

  iree_status_t status =
      loom_tool_dup_cstring(executable_path, allocator, &argv[0]);
  for (iree_host_size_t i = 0; i < argument_count && iree_status_is_ok(status);
       ++i) {
    status = loom_tool_dup_cstring(arguments[i], allocator, &argv[i + 1]);
  }
  if (iree_status_is_ok(status)) {
    *out_argv = argv;
  } else {
    loom_tool_argv_deinitialize(argv, allocator);
  }
  return status;
}

static iree_status_t loom_tool_output_append_text(
    iree_string_builder_t* builder, const char* data, iree_host_size_t length) {
  return iree_string_builder_append_string(builder,
                                           iree_make_string_view(data, length));
}

static iree_status_t loom_tool_output_take(iree_string_builder_t* builder,
                                           loom_tool_output_t* out_output) {
  out_output->length = iree_string_builder_size(builder);
  out_output->data = iree_string_builder_take_storage(builder);
  return iree_ok_status();
}

void loom_tool_output_deinitialize(loom_tool_output_t* output,
                                   iree_allocator_t allocator) {
  if (output == NULL) {
    return;
  }
  iree_allocator_free(allocator, output->data);
  *output = (loom_tool_output_t){0};
}

void loom_tool_process_result_deinitialize(loom_tool_process_result_t* result,
                                           iree_allocator_t allocator) {
  if (result == NULL) {
    return;
  }
  loom_tool_output_deinitialize(&result->stdout_text, allocator);
  loom_tool_output_deinitialize(&result->stderr_text, allocator);
  result->exit_code = 0;
}

bool loom_tool_process_result_succeeded(
    const loom_tool_process_result_t* result) {
  return result != NULL && result->exit_code == 0;
}

//===----------------------------------------------------------------------===//
// Platform process capture
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

typedef struct loom_tool_capture_file_t {
  // Capturing file handle passed to the child process.
  HANDLE handle;
  // Temporary filesystem path backing |handle|.
  char path[4096];
} loom_tool_capture_file_t;

static iree_status_t loom_tool_win32_status(DWORD error, const char* message) {
  return iree_make_status(iree_status_code_from_win32_error(error),
                          "%s (GetLastError=%lu)", message,
                          (unsigned long)error);
}

static iree_status_t loom_tool_capture_file_open(
    const char* prefix, loom_tool_capture_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));
  char temp_directory[4096] = {0};
  DWORD temp_directory_length =
      GetTempPathA(IREE_ARRAYSIZE(temp_directory), temp_directory);
  if (temp_directory_length == 0 ||
      temp_directory_length >= IREE_ARRAYSIZE(temp_directory)) {
    return loom_tool_win32_status(GetLastError(),
                                  "failed to resolve temporary directory");
  }

  if (GetTempFileNameA(temp_directory, prefix, 0, out_file->path) == 0) {
    return loom_tool_win32_status(GetLastError(),
                                  "failed to allocate temporary file path");
  }

  SECURITY_ATTRIBUTES security_attributes;
  memset(&security_attributes, 0, sizeof(security_attributes));
  security_attributes.nLength = sizeof(security_attributes);
  security_attributes.bInheritHandle = TRUE;

  HANDLE handle = CreateFileA(
      out_file->path, GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      &security_attributes, CREATE_ALWAYS,
      FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    DeleteFileA(out_file->path);
    memset(out_file, 0, sizeof(*out_file));
    return loom_tool_win32_status(error, "failed to open capture file");
  }
  out_file->handle = handle;
  return iree_ok_status();
}

static void loom_tool_capture_file_deinitialize(
    loom_tool_capture_file_t* file) {
  if (file->handle != NULL && file->handle != INVALID_HANDLE_VALUE) {
    CloseHandle(file->handle);
  }
  if (file->path[0] != '\0') {
    DeleteFileA(file->path);
  }
  memset(file, 0, sizeof(*file));
}

static iree_status_t loom_tool_capture_file_read(
    loom_tool_capture_file_t* file, iree_allocator_t allocator,
    loom_tool_output_t* out_output) {
  *out_output = (loom_tool_output_t){0};
  if (SetFilePointer(file->handle, 0, NULL, FILE_BEGIN) ==
      INVALID_SET_FILE_POINTER) {
    DWORD error = GetLastError();
    if (error != NO_ERROR) {
      return loom_tool_win32_status(error, "failed to rewind capture file");
    }
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  uint8_t buffer[4096];
  bool reading = true;
  while (reading && iree_status_is_ok(status)) {
    DWORD bytes_read = 0;
    if (!ReadFile(file->handle, buffer, sizeof(buffer), &bytes_read, NULL)) {
      status =
          loom_tool_win32_status(GetLastError(), "failed to read capture file");
    } else if (bytes_read == 0) {
      reading = false;
    } else {
      status = loom_tool_output_append_text(&builder, (const char*)buffer,
                                            (iree_host_size_t)bytes_read);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_output_take(&builder, out_output);
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(out_output, allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static bool loom_tool_win32_arg_needs_quotes(const char* argument) {
  if (argument[0] == '\0') {
    return true;
  }
  for (const char* p = argument; *p != '\0'; ++p) {
    if (*p == ' ' || *p == '\t' || *p == '"') {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_tool_win32_append_char(iree_string_builder_t* builder,
                                                 char value) {
  return iree_string_builder_append_string(builder,
                                           iree_make_string_view(&value, 1));
}

static iree_status_t loom_tool_win32_append_repeated_char(
    iree_string_builder_t* builder, char value, iree_host_size_t count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_tool_win32_append_char(builder, value);
  }
  return status;
}

static iree_status_t loom_tool_win32_append_quoted_arg(
    iree_string_builder_t* builder, const char* argument) {
  if (!loom_tool_win32_arg_needs_quotes(argument)) {
    return iree_string_builder_append_cstring(builder, argument);
  }

  iree_status_t status = loom_tool_win32_append_char(builder, '"');
  iree_host_size_t backslash_count = 0;
  for (const char* p = argument; *p != '\0' && iree_status_is_ok(status); ++p) {
    if (*p == '\\') {
      ++backslash_count;
    } else if (*p == '"') {
      status = loom_tool_win32_append_repeated_char(builder, '\\',
                                                    backslash_count * 2 + 1);
      if (iree_status_is_ok(status)) {
        status = loom_tool_win32_append_char(builder, '"');
      }
      backslash_count = 0;
    } else {
      status =
          loom_tool_win32_append_repeated_char(builder, '\\', backslash_count);
      if (iree_status_is_ok(status)) {
        status = loom_tool_win32_append_char(builder, *p);
      }
      backslash_count = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_win32_append_repeated_char(builder, '\\',
                                                  backslash_count * 2);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_win32_append_char(builder, '"');
  }
  return status;
}

static iree_status_t loom_tool_win32_make_command_line(
    char** argv, iree_allocator_t allocator, char** out_command_line) {
  *out_command_line = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; argv[i] != NULL && iree_status_is_ok(status);
       ++i) {
    if (i != 0) {
      status = loom_tool_win32_append_char(&builder, ' ');
    }
    if (iree_status_is_ok(status)) {
      status = loom_tool_win32_append_quoted_arg(&builder, argv[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_command_line = iree_string_builder_take_storage(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_tool_process_run_argv(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  (void)search_path;
  *out_result = (loom_tool_process_result_t){0};

  loom_tool_capture_file_t stdout_file;
  loom_tool_capture_file_t stderr_file;
  IREE_RETURN_IF_ERROR(loom_tool_capture_file_open("out", &stdout_file));
  iree_status_t status = loom_tool_capture_file_open("err", &stderr_file);

  char* command_line = NULL;
  PROCESS_INFORMATION process_info;
  memset(&process_info, 0, sizeof(process_info));
  if (iree_status_is_ok(status)) {
    status = loom_tool_win32_make_command_line(argv, allocator, &command_line);
  }
  if (iree_status_is_ok(status)) {
    STARTUPINFOA startup_info;
    memset(&startup_info, 0, sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup_info.hStdOutput = stdout_file.handle;
    startup_info.hStdError = stderr_file.handle;

    if (!CreateProcessA(NULL, command_line, NULL, NULL, TRUE, 0, NULL, NULL,
                        &startup_info, &process_info)) {
      status = loom_tool_win32_status(GetLastError(),
                                      "failed to spawn tool process");
    }
  }
  if (iree_status_is_ok(status)) {
    WaitForSingleObject(process_info.hProcess, INFINITE);
    DWORD exit_code = 1;
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
      status = loom_tool_win32_status(GetLastError(),
                                      "failed to query tool process exit code");
    } else {
      out_result->exit_code = (int)exit_code;
    }
  }
  if (process_info.hThread != NULL) {
    CloseHandle(process_info.hThread);
  }
  if (process_info.hProcess != NULL) {
    CloseHandle(process_info.hProcess);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stdout_file, allocator,
                                         &out_result->stdout_text);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stderr_file, allocator,
                                         &out_result->stderr_text);
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_process_result_deinitialize(out_result, allocator);
  }
  iree_allocator_free(allocator, command_line);
  loom_tool_capture_file_deinitialize(&stderr_file);
  loom_tool_capture_file_deinitialize(&stdout_file);
  return status;
}

#elif LOOM_TOOL_PROCESS_POSIX

typedef struct loom_tool_capture_file_t {
  // Open file descriptor passed to the child process.
  int fd;
} loom_tool_capture_file_t;

static const char* loom_tool_temp_directory(void) {
  const char* temp_directory = getenv("TEST_TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  temp_directory = getenv("TMPDIR");
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
  unlink(template_path);
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
      status = loom_tool_output_append_text(&builder, buffer,
                                            (iree_host_size_t)bytes_read);
    } else if (bytes_read == 0) {
      reading = false;
    } else if (errno != EINTR) {
      status = loom_tool_posix_status(errno, "failed to read capture file");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_output_take(&builder, out_output);
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

static iree_status_t loom_tool_process_run_argv(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  *out_result = (loom_tool_process_result_t){0};

  loom_tool_capture_file_t stdout_file;
  loom_tool_capture_file_t stderr_file;
  IREE_RETURN_IF_ERROR(loom_tool_capture_file_open("stdout", &stdout_file));
  iree_status_t status = loom_tool_capture_file_open("stderr", &stderr_file);

  posix_spawn_file_actions_t file_actions;
  bool file_actions_initialized = false;
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_init(&file_actions);
    if (spawn_result == 0) {
      file_actions_initialized = true;
    } else {
      status = loom_tool_posix_status(
          spawn_result, "failed to initialize process spawn actions");
    }
  }
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_adddup2(
        &file_actions, stdout_file.fd, STDOUT_FILENO);
    if (spawn_result != 0) {
      status =
          loom_tool_posix_status(spawn_result, "failed to redirect stdout");
    }
  }
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_adddup2(
        &file_actions, stderr_file.fd, STDERR_FILENO);
    if (spawn_result != 0) {
      status =
          loom_tool_posix_status(spawn_result, "failed to redirect stderr");
    }
  }
  pid_t pid = 0;
  if (iree_status_is_ok(status)) {
    int spawn_result =
        search_path
            ? posix_spawnp(&pid, argv[0], &file_actions, NULL, argv, environ)
            : posix_spawn(&pid, argv[0], &file_actions, NULL, argv, environ);
    if (spawn_result != 0) {
      status =
          loom_tool_posix_status(spawn_result, "failed to spawn tool process");
    }
  }
  if (file_actions_initialized) {
    posix_spawn_file_actions_destroy(&file_actions);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_process_wait(pid, &out_result->exit_code);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stdout_file, allocator,
                                         &out_result->stdout_text);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tool_capture_file_read(&stderr_file, allocator,
                                         &out_result->stderr_text);
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_process_result_deinitialize(out_result, allocator);
  }
  loom_tool_capture_file_deinitialize(&stderr_file);
  loom_tool_capture_file_deinitialize(&stdout_file);
  return status;
}

#else

static iree_status_t loom_tool_process_run_argv(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  (void)argv;
  (void)search_path;
  (void)allocator;
  *out_result = (loom_tool_process_result_t){0};
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "external tool execution is unsupported on this platform");
}

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Public process API
//===----------------------------------------------------------------------===//

iree_status_t loom_tool_process_run(iree_string_view_t executable_path,
                                    bool search_path,
                                    const iree_string_view_t* arguments,
                                    iree_host_size_t argument_count,
                                    iree_allocator_t allocator,
                                    loom_tool_process_result_t* out_result) {
  *out_result = (loom_tool_process_result_t){0};

  char** argv = NULL;
  IREE_RETURN_IF_ERROR(loom_tool_argv_allocate(
      executable_path, arguments, argument_count, allocator, &argv));
  iree_status_t status =
      loom_tool_process_run_argv(argv, search_path, allocator, out_result);
  loom_tool_argv_deinitialize(argv, allocator);
  return status;
}

//===----------------------------------------------------------------------===//
// Temporary files
//===----------------------------------------------------------------------===//

static bool loom_tool_temp_file_stem_char_is_valid(char value) {
  return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
         (value >= '0' && value <= '9') || value == '_' || value == '-';
}

static iree_status_t loom_tool_temp_file_copy_stem(iree_string_view_t stem,
                                                   char* buffer,
                                                   iree_host_size_t capacity) {
  if (iree_string_view_is_empty(stem)) {
    stem = IREE_SV("tmp");
  }
  if (stem.size >= capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary file stem is too long");
  }
  for (iree_host_size_t i = 0; i < stem.size; ++i) {
    if (!loom_tool_temp_file_stem_char_is_valid(stem.data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid temporary file stem character '%c'",
                              stem.data[i]);
    }
    buffer[i] = stem.data[i];
  }
  buffer[stem.size] = '\0';
  return iree_ok_status();
}

iree_string_view_t loom_tool_temp_file_path(const loom_tool_temp_file_t* file) {
  return iree_make_cstring_view(file->path);
}

#if defined(IREE_PLATFORM_WINDOWS)

iree_status_t loom_tool_temp_file_initialize(iree_string_view_t stem,
                                             loom_tool_temp_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));

  char stem_buffer[32] = {0};
  IREE_RETURN_IF_ERROR(loom_tool_temp_file_copy_stem(
      stem, stem_buffer, IREE_ARRAYSIZE(stem_buffer)));

  char temp_directory[4096] = {0};
  DWORD temp_directory_length =
      GetTempPathA(IREE_ARRAYSIZE(temp_directory), temp_directory);
  if (temp_directory_length == 0 ||
      temp_directory_length >= IREE_ARRAYSIZE(temp_directory)) {
    return loom_tool_win32_status(GetLastError(),
                                  "failed to resolve temporary directory");
  }

  if (GetTempFileNameA(temp_directory, stem_buffer, 0, out_file->path) == 0) {
    memset(out_file, 0, sizeof(*out_file));
    return loom_tool_win32_status(GetLastError(),
                                  "failed to allocate temporary file path");
  }
  return iree_ok_status();
}

void loom_tool_temp_file_deinitialize(loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return;
  }
  if (file->path[0] != '\0') {
    DeleteFileA(file->path);
  }
  memset(file, 0, sizeof(*file));
}

#elif LOOM_TOOL_PROCESS_POSIX

iree_status_t loom_tool_temp_file_initialize(iree_string_view_t stem,
                                             loom_tool_temp_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));

  char stem_buffer[32] = {0};
  IREE_RETURN_IF_ERROR(loom_tool_temp_file_copy_stem(
      stem, stem_buffer, IREE_ARRAYSIZE(stem_buffer)));

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
  if (close(fd) < 0 && errno != EINTR) {
    int error_number = errno;
    unlink(out_file->path);
    memset(out_file, 0, sizeof(*out_file));
    return loom_tool_posix_status(error_number,
                                  "failed to close temporary file");
  }
  return iree_ok_status();
}

void loom_tool_temp_file_deinitialize(loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return;
  }
  if (file->path[0] != '\0') {
    unlink(file->path);
  }
  memset(file, 0, sizeof(*file));
}

#else

iree_status_t loom_tool_temp_file_initialize(iree_string_view_t stem,
                                             loom_tool_temp_file_t* out_file) {
  (void)stem;
  memset(out_file, 0, sizeof(*out_file));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "external tool temporary files are unsupported on this platform");
}

void loom_tool_temp_file_deinitialize(loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return;
  }
  memset(file, 0, sizeof(*file));
}

#endif  // IREE_PLATFORM_WINDOWS
