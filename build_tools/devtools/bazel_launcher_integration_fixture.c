// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synchronized process fixture for the host-side Bazel launcher probe.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#define getcwd _getcwd
#else
#include <sys/types.h>
#include <unistd.h>
#endif

static unsigned long process_id(void) {
#if defined(_WIN32)
  return (unsigned long)_getpid();
#else
  return (unsigned long)getpid();
#endif
}

static unsigned long parent_process_id(void) {
#if defined(_WIN32)
  DWORD current_process_id = GetCurrentProcessId();
  DWORD parent_process_id = 0;
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32 entry;
  memset(&entry, 0, sizeof(entry));
  entry.dwSize = sizeof(entry);
  if (Process32First(snapshot, &entry)) {
    do {
      if (entry.th32ProcessID == current_process_id) {
        parent_process_id = entry.th32ParentProcessID;
        break;
      }
    } while (Process32Next(snapshot, &entry));
  }
  CloseHandle(snapshot);
  return (unsigned long)parent_process_id;
#else
  return (unsigned long)getppid();
#endif
}

static int parse_exit_code(const char* value, int* out_exit_code) {
  char* end = NULL;
  long exit_code = strtol(value, &end, 10);
  if (!value[0] || !end || end[0] || exit_code < 0 || exit_code > 255) {
    return 1;
  }
  *out_exit_code = (int)exit_code;
  return 0;
}

int main(int argc, char** argv) {
  const char* ready_file = NULL;
  const char* result_file = NULL;
  int exit_code = 0;
  int argument_count = 0;
  char** arguments = argv + 1;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
      ready_file = argv[++i];
    } else if (strcmp(argv[i], "--result-file") == 0 && i + 1 < argc) {
      result_file = argv[++i];
    } else if (strcmp(argv[i], "--exit-code") == 0 && i + 1 < argc) {
      if (parse_exit_code(argv[++i], &exit_code)) {
        fprintf(stderr, "invalid exit code: %s\n", argv[i]);
        return 126;
      }
    } else {
      arguments[argument_count++] = argv[i];
    }
  }

  if (ready_file) {
    FILE* file = fopen(ready_file, "wb");
    if (!file) {
      perror("unable to create ready file");
      return 126;
    }
    fputs("ready\n", file);
    fclose(file);
    if (getchar() == EOF) {
      fprintf(stderr, "launcher fixture stdin closed before release\n");
      return 126;
    }
  }

  if (result_file) {
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) {
      perror("unable to read fixture cwd");
      return 126;
    }
    FILE* file = fopen(result_file, "wb");
    if (!file) {
      perror("unable to create result file");
      return 126;
    }
    fprintf(file, "%lu\n", process_id());
    fprintf(file, "%lu\n", parent_process_id());
    fprintf(file, "%s\n", cwd);
    fprintf(file, "%d\n", argument_count);
    for (int i = 0; i < argument_count; ++i) {
      fprintf(file, "%s\n", arguments[i]);
    }
    fclose(file);
  }
  return exit_code;
}
