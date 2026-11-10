// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Process policy for explicitly configured rocjitsu invocations. This program
// never loads the simulator; exec transfers all device ownership to the caller.
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

static int option_matches(const char* fixed, const char* argument) {
  size_t length = strcspn(fixed, "=");
  return strncmp(fixed, argument, length) == 0 &&
         (argument[length] == '\0' || argument[length] == '=');
}

int main(int argc, char** argv) {
  // HOST_CPU PROGRAM FIXED_OPTIONS... -- CALLER_ARGUMENTS...
  if (argc < 4) {
    fprintf(stderr, "rocjitsu launcher requires a host CPU, program, and --\n");
    return 2;
  }
  if (strcmp(argv[1], "x86-64-v3") == 0) {
    if (!__builtin_cpu_supports("x86-64-v3")) {
      fprintf(stderr, "rocjitsu runtime requires an x86-64-v3 host CPU\n");
      return 2;
    }
  } else if (strcmp(argv[1], "x86-64") != 0) {
    fprintf(stderr, "unsupported rocjitsu host CPU: %s\n", argv[1]);
    return 2;
  }
  int separator = 3;
  while (separator < argc && strcmp(argv[separator], "--") != 0) {
    ++separator;
  }
  if (separator == argc) {
    fprintf(stderr, "rocjitsu launcher requires -- before caller arguments\n");
    return 2;
  }
  for (int i = separator + 1; i < argc; ++i) {
    for (int j = 3; j < separator; ++j) {
      if (option_matches(argv[j], argv[i])) {
        fprintf(stderr, "cannot override rocjitsu invocation option: %s\n",
                argv[i]);
        return 2;
      }
    }
  }

  struct rlimit descriptors;
  if (getrlimit(RLIMIT_NOFILE, &descriptors) != 0) {
    fprintf(stderr, "getrlimit(RLIMIT_NOFILE): %s\n", strerror(errno));
    return 2;
  }
  // SimulatedKfd raises the process limit on open only when it is below 8192.
  // Prepare that minimum before the consumer starts so embedding changes no
  // process resource policy.
  if (descriptors.rlim_cur < 8192) {
    descriptors.rlim_cur = 8192;
    if (setrlimit(RLIMIT_NOFILE, &descriptors) != 0) {
      fprintf(stderr, "rocjitsu requires RLIMIT_NOFILE >= 8192: %s\n",
              strerror(errno));
      return 2;
    }
  }

  // Remove only our separator, retaining the caller's argument boundaries and
  // terminating null pointer. The consumer keeps its original executable path.
  memmove(argv + separator, argv + separator + 1,
          (size_t)(argc - separator) * sizeof(*argv));
  execv(argv[2], argv + 2);
  fprintf(stderr, "exec %s: %s\n", argv[2], strerror(errno));
  return 126;
}
