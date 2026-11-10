// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char** argv) {
  struct rlimit descriptors;
  struct sigaction action;
  if (getrlimit(RLIMIT_NOFILE, &descriptors) != 0 ||
      sigaction(SIGUSR1, NULL, &action) != 0) {
    return 1;
  }
  printf("%ld %llu %d\n", (long)getpid(),
         (unsigned long long)descriptors.rlim_cur,
         action.sa_handler == SIG_IGN);
  for (int i = 1; i < argc; ++i) {
    printf("%zu:", strlen(argv[i]));
    fputs(argv[i], stdout);
    putchar('\n');
  }
  return 0;
}
