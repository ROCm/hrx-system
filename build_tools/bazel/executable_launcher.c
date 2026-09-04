// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native trampoline for Windows Bazel executable wrappers. It launches the
// original binary through its runfiles path so the Windows loader sees the
// source-adjacent DLLs that a cross-package wrapper symlink would strand.

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <errno.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <windows.h>

#define IREE_BAZEL_EXECUTABLE_RUNFILE_ENV L"IREE_BAZEL_EXECUTABLE_RUNFILE"

static wchar_t* iree_bazel_read_environment(const wchar_t* name) {
  SetLastError(ERROR_SUCCESS);
  DWORD capacity = GetEnvironmentVariableW(name, NULL, 0);
  if (capacity == 0) return NULL;
  wchar_t* value = (wchar_t*)malloc((size_t)capacity * sizeof(wchar_t));
  if (!value) return NULL;
  DWORD length = GetEnvironmentVariableW(name, value, capacity);
  if (length == 0 || length >= capacity) {
    free(value);
    return NULL;
  }
  return value;
}

static wchar_t* iree_bazel_query_module_path(void) {
  DWORD capacity = 256;
  while (capacity <= 32768) {
    wchar_t* path = (wchar_t*)malloc((size_t)capacity * sizeof(wchar_t));
    if (!path) return NULL;
    DWORD length = GetModuleFileNameW(NULL, path, capacity);
    if (length > 0 && length < capacity) return path;
    free(path);
    capacity *= 2;
  }
  return NULL;
}

static wchar_t* iree_bazel_find_runfiles_root(void) {
  const wchar_t* const environment_names[] = {
      L"RUNFILES_DIR",
      L"TEST_SRCDIR",
  };
  for (size_t i = 0;
       i < sizeof(environment_names) / sizeof(environment_names[0]); ++i) {
    wchar_t* value = iree_bazel_read_environment(environment_names[i]);
    if (value) return value;
  }

  wchar_t* module_path = iree_bazel_query_module_path();
  if (!module_path) return NULL;
  const wchar_t suffix[] = L".runfiles";
  size_t module_length = wcslen(module_path);
  size_t suffix_length = wcslen(suffix);
  if (module_length > SIZE_MAX - suffix_length - 1) {
    free(module_path);
    return NULL;
  }
  wchar_t* runfiles_root = (wchar_t*)realloc(
      module_path, (module_length + suffix_length + 1) * sizeof(wchar_t));
  if (!runfiles_root) {
    free(module_path);
    return NULL;
  }
  memcpy(runfiles_root + module_length, suffix,
         (suffix_length + 1) * sizeof(wchar_t));
  return runfiles_root;
}

static wchar_t* iree_bazel_join_path(const wchar_t* root,
                                     const wchar_t* relative_path) {
  size_t root_length = wcslen(root);
  size_t relative_length = wcslen(relative_path);
  if (root_length > SIZE_MAX - relative_length - 2) return NULL;
  wchar_t* path =
      (wchar_t*)malloc((root_length + relative_length + 2) * sizeof(wchar_t));
  if (!path) return NULL;
  memcpy(path, root, root_length * sizeof(wchar_t));
  size_t position = root_length;
  if (position > 0 && path[position - 1] != L'\\' &&
      path[position - 1] != L'/') {
    path[position++] = L'\\';
  }
  for (size_t i = 0; i < relative_length; ++i) {
    path[position++] = relative_path[i] == L'/' ? L'\\' : relative_path[i];
  }
  path[position] = 0;
  return path;
}

int wmain(int argc, wchar_t** argv) {
  wchar_t* target_runfile =
      iree_bazel_read_environment(IREE_BAZEL_EXECUTABLE_RUNFILE_ENV);
  if (!target_runfile) {
    fwprintf(stderr, L"iree-executable-launcher: missing %ls\n",
             IREE_BAZEL_EXECUTABLE_RUNFILE_ENV);
    return 127;
  }
  wchar_t* runfiles_root = iree_bazel_find_runfiles_root();
  if (!runfiles_root) {
    fwprintf(stderr,
             L"iree-executable-launcher: unable to locate the runfiles root\n");
    free(target_runfile);
    return 127;
  }
  wchar_t* target_path = iree_bazel_join_path(runfiles_root, target_runfile);
  free(runfiles_root);
  free(target_runfile);
  DWORD target_attributes =
      target_path ? GetFileAttributesW(target_path) : INVALID_FILE_ATTRIBUTES;
  if (target_attributes == INVALID_FILE_ATTRIBUTES ||
      (target_attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    fwprintf(stderr, L"iree-executable-launcher: missing target %ls\n",
             target_path ? target_path : L"<allocation failed>");
    free(target_path);
    return 127;
  }
  if (!SetEnvironmentVariableW(IREE_BAZEL_EXECUTABLE_RUNFILE_ENV, NULL)) {
    fwprintf(
        stderr,
        L"iree-executable-launcher: unable to clear private environment\n");
    free(target_path);
    return 127;
  }

  if ((size_t)argc > SIZE_MAX / sizeof(wchar_t*) - 1) {
    fwprintf(stderr, L"iree-executable-launcher: argument count overflow\n");
    free(target_path);
    return 127;
  }
  const wchar_t** target_argv =
      (const wchar_t**)malloc(((size_t)argc + 1) * sizeof(wchar_t*));
  if (!target_argv) {
    fwprintf(stderr, L"iree-executable-launcher: argument allocation failed\n");
    free(target_path);
    return 127;
  }
  target_argv[0] = target_path;
  for (int i = 1; i < argc; ++i) target_argv[i] = argv[i];
  target_argv[argc] = NULL;

  intptr_t result = _wspawnv(_P_WAIT, target_path, target_argv);
  if (result == -1) {
    fwprintf(stderr, L"iree-executable-launcher: unable to launch %ls: %d\n",
             target_path, errno);
    result = 127;
  }
  free(target_argv);
  free(target_path);
  return (int)result;
}

#else

#include <stdio.h>

int main(void) {
  fputs("iree-executable-launcher: Windows launcher used on another platform\n",
        stderr);
  return 127;
}

#endif  // _WIN32
