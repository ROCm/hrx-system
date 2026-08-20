# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core and HAL 0.0 contracts inherited by instruction families."""

from __future__ import annotations

from model.specification import CORE_0, HAL_0, NormativeClause

CORE_MACHINE = NormativeClause(
    entity_id="core.contract.machine",
    since=CORE_0,
    summary="Core frame, record, value, publication, and failure contract.",
    normative_text=(
        "A core invocation owns one checked LIFO byte span containing every "
        "live frame. Each bytecode frame holds its current record, up to 256 "
        "64-bit value registers, 256 typed ref registers, 256 non-owning "
        "16-byte function registers, function-declared overflow/local storage, "
        "and child module frame state while a call is active. Descriptor extents "
        "are architectural; in-memory layout is private runtime state. Each "
        "module receives one max-aligned opaque per-process state slice. No "
        "bytecode field contains a host pointer or process-relative address. "
        "First-byte values 0x01..0xEF form the Core opcode space; 0xF0..0xFD "
        "select architectural extension pages whose second-byte local opcode "
        "is in 0x01..0xFF. First byte 0x00 and canonical-v0 prefixes 0xFE/0xFF "
        "are invalid. "
        "Every instruction begins on a four-byte boundary, has an opcode-known "
        "positive four-byte-multiple size, is decoded exactly once during "
        "verification, requires zero reserved/padding bits, and executes from "
        "the immutable verified image. Unknown opcode, truncation, invalid "
        "selector/ordinal/static range, or nonzero reserved bits rejects the "
        "image; there is no unknown-record skip rule. A value register is one "
        "untyped 64-bit cell whose entry bits are unspecified until written. "
        "Frame-local POD bytes likewise begin unspecified. Ownership-bearing ref "
        "regions and capability-bearing function regions begin canonical null. "
        "A ref register is canonical null, "
        "borrowed typed ref, or owned typed ref; linked descriptor identity is exact and a live "
        "ref may always be retained without querying its count. A function "
        "register is canonical null or one complete non-owning 16-byte, program-"
        "bound function reference and cannot be forged by raw value/byte ops. "
        "Publishing a ref into independently lived storage first validates every "
        "precondition without mutation, promotes a borrow by retaining it, saves "
        "any replaced owner, moves the new owner and clears its source, then "
        "releases the old owner. Each function/import declares conservative "
        "may_yield. False promises suspension is unreachable; true permits it. "
        "An import false cannot bind target true; import true accepts either. "
        "The loader never infers or propagates this fact through a call graph. "
        "A public invocation cannot pass a non-null borrowed ref to a may_yield "
        "function. Each yield-capable operation and call checks the invocation's "
        "borrow fact before arming work or publishing suspension and fails with "
        "failed_precondition without commitment on violation. The invocation wake "
        "callback is an optional level-triggered scheduling hint. Every VM or "
        "provider wake publishes its state first, invokes the callback only when "
        "its function pointer is non-null, and never resumes execution inline. "
        "Driving returns ordinary COMPLETED or SUSPENDED outcomes; status is never coroutine "
        "control flow. Any non-OK status terminates that invocation attempt, "
        "leaves public result storage untouched, unwinds live owners once, and "
        "returns invocation storage to idle. The ISA assigns no process-taint, "
        "retry, or hosting-process policy. Assembly uses %vN/%rN/%fN registers, "
        "@gvN/@grN/@gfN globals, and derived ^bbN labels as a readable projection "
        "only; it is not a second serialized format."
    ),
)

HAL_PAGE = NormativeClause(
    entity_id="hal.contract.page",
    since=HAL_0,
    summary="Optional HAL instruction-page architecture and execution boundary.",
    dependencies=(CORE_MACHINE.entity_id,),
    normative_text=(
        "HAL is optional direct instruction page 0xF0, not an import convention. "
        "Its first record byte selects the page and its nonzero second byte is a "
        "page-local opcode. A core-only runtime need not register the page or HAL "
        "types. Records inherit core framing, little-endian fields, zero padding, "
        "verified immutable image, and terminal status behavior. The page exposes "
        "typed primitives but prescribes no planning, initialization, target "
        "selection, caching, queue topology, or executable policy. There is no "
        "default device/group/queue/frontier/allocator/channel/executable/thread-"
        "local context; each operation names all authority it uses. Native "
        "pointers, function tokens, external semaphore handles, and provider "
        "state never enter value registers or serialized fields. Verification "
        "checks page version, records, local ranges, ordinals, selectors, flags, "
        "and padding, but dynamic non-null refs still require exact linked-"
        "descriptor checks before dereference. Every u64 is checked before C-ABI "
        "narrowing. Ref-producing operations construct one local owner and only "
        "replace the destination after success; failure releases any partial "
        "result and preserves the prior destination. Synchronous calls borrow "
        "device/object pointers only for entry. The host keeps the enclosing "
        "device group/device alive across work; HAL copies/captures or retains "
        "everything else it needs after return and never keeps VM-owned pointer "
        "arrays or aggregate rows borrowed. Adaptation uses the invocation's one "
        "checked LIFO stack with no growth, retry, separate HAL capacity, or per-"
        "record heap allocation. Insufficient space fails resource_exhausted "
        "before adaptation/provider entry. semaphore.await alone may suspend; "
        "its provider state stays in invocation storage and quiesces callbacks "
        "before reuse. Every provider/adapter failure terminates the invocation "
        "through core unwind without assigning retry, process-taint, device-reset, "
        "or host policy. Capabilities absent from the page have no hidden import "
        "fallback."
    ),
)

HAL_ABI = NormativeClause(
    entity_id="hal.contract.abi",
    since=HAL_0,
    summary="Typed HAL refs, packet adaptation, narrowing, and publication.",
    dependencies=(HAL_PAGE.entity_id,),
    normative_text=(
        "Required HAL refs reject canonical null with failed_precondition and a "
        "non-null descriptor mismatch with invalid_argument; optional refs accept "
        "only canonical null or the exact descriptor. All scalar refs validate "
        "before dereference, native-array formation, destination change, or "
        "provider call. Ref-range elements apply the same rule independently and "
        "their VM slots anchor borrowed pointers through the synchronous call. "
        "Dynamic cells checked-narrow exactly to their declared u16/u32/u64, "
        "host-size, device-size, affinity-bitset, or signed-i64 domain; invalid "
        "bits and cross-field violations never truncate, mask, wrap, or gain a "
        "default. Counted function-local packets are struct-of-arrays groups. Ref "
        "bases are slot ordinals, byte bases are offsets, and one u16 count owns "
        "all columns. Nonempty ranges use checked widened subtraction and declared "
        "alignment; zero count requires every base in the group to be zero. "
        "Semaphore/payload groups pair exact non-null semaphore refs with aligned "
        "u64 payloads no greater than 0x000000007FFFFFFE. Empty lists are valid. "
        "Queue order is not FIFO; only explicit semaphore edges and operation "
        "semantics create ordering. A direct-or-slot buffer reference contains "
        "nullable exact hal.buffer, u32 slot, u64 offset, and u64 length. Non-null "
        "buffer requires slot zero and a live checked range; null buffer requires "
        "slot below 2^24 and offset/length not UINT64_MAX. All values fit device "
        "size; UINT64_MAX never means whole-buffer. Static dispatch launch packets "
        "hold u32 count[3], size[3], and dynamic-local-memory lanes; indirect "
        "packets omit counts. All-zero size selects executable defaults, otherwise "
        "all three lanes are nonzero. Static dispatch flags allow only 0x20. "
        "Indirect flags allow 0x20 and require exactly one of dynamic/static "
        "indirect-parameter bits 0/1. Allocation packets checked-narrow usage "
        "against 0x1F0F3F03, access against 0x001F, memory type against 0x7F, "
        "plus affinity, alignment, and size; nonzero alignment is a power of two "
        "and access ANY 0x20 is rejected. Ref result publication performs all VM "
        "checks, calls HAL into one local result, releases any partial result on "
        "failure, wraps a complete exact-type owner, then replaces the prior "
        "destination. Value results likewise publish only after every fallible "
        "step. Provider statuses propagate unchanged and never mean retry."
    ),
)

CONTRACTS = (CORE_MACHINE, HAL_PAGE, HAL_ABI)
ENTITIES = CONTRACTS
