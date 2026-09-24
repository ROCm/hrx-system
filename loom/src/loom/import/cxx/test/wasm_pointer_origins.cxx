// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Pair {
  int first;
  int second;
};

int read(const int* pointer) { return *pointer; }

int read_zero(const int* pointer) { return pointer[0]; }

int read_next(const int* pointer) { return pointer[1]; }

int read_previous(const int* pointer) { return pointer[-1]; }

int read_at(const int* pointer, int index) { return pointer[index]; }

int read_member(const Pair* pointer) { return pointer->second; }

static int load_next(const int* pointer) { return pointer[1]; }

int read_helper(const int* pointer) { return load_next(pointer + 1); }

int exchange(int* pointer, int index, int replacement) {
  int previous = pointer[index];
  pointer[index] = replacement;
  return previous;
}

unsigned read_byte(const unsigned char* buffer, unsigned position) {
  return buffer[position];
}

void write_byte(unsigned char* buffer, unsigned position, unsigned char value) {
  buffer[position] = value;
}
