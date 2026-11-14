# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic benchmark workloads.

A benchmark is only comparable to another run of the same harness, so the
workloads are generated: a seeded word stream of ordinary English, in document
sizes a card's prefill rows name. The exact prompt size is never estimated; each
request reports the prompt tokens the server counted.
"""

from __future__ import annotations

HEADER = (
    "Write a long, detailed essay discussing this documentation in depth, "
    "covering its purpose, its structure and its consequences."
)

#: Ordinary words, so the tokenizer sees natural text rather than one repeated
#: token and the reported token count stays close to the intended size.
WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or "
    "his from at which but have an they you were her she there one all we can "
    "her would make like him into time has look two more write go see number no "
    "way could people my than first water been call who oil its now find long "
    "down day did get come made may part over new sound take only little work "
    "know place year live me back give most very after thing our just name good "
    "sentence man think say great where help through much before line right too "
    "mean old any same tell boy follow came want show also around form three "
    "small set put end does another well large must big even such because turn "
    "here why ask went men read need land different home us move try kind hand "
    "picture again change off play spell air away animal house point page letter "
    "mother answer found study still learn should world high every near add food "
    "between own below country plant last school father keep tree never start "
    "city earth eye light thought head under story saw left few while along might "
    "close something seem next hard open example begin life always those both "
    "paper together got group often run important until children side feet car "
    "mile night walk white sea began grow took river four carry state once book "
    "hear stop without second later miss idea enough eat face watch far really "
    "almost let above girl sometimes mountain cut young talk soon list song being "
    "leave family it under young"
)

#: The document identity stored with every record, so a comparison is honest.
WORKLOAD_IDENTITY = "loom-launch/text-v1"


def words_for(count: int, seed: int = 0x5EED) -> str:
    """Generates `count` words from a seeded stream, deterministically.

    Each word is drawn from a xorshift generator, so two runs anywhere produce
    the same bytes and a tokenizer sees the same text.
    """
    if count <= 0:
        return ""
    state = seed & 0xFFFFFFFF or 1
    vocabulary = WORDS.split()
    size = len(vocabulary)
    produced: list[str] = []
    for index in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        produced.append(vocabulary[state % size])
        if index and index % 12 == 0:
            produced[-1] += "."
    return " ".join(produced)


def document(approximate_tokens: int, seed: int = 0x5EED) -> str:
    """A document sized in tokens; English runs about 0.75 words per token."""
    if approximate_tokens <= 0:
        return ""
    return words_for(int(approximate_tokens * 0.75), seed)


def prefill_prompt(approximate_tokens: int, seed: int = 0x5EED) -> str:
    """A document followed by a short instruction, for a prefill measurement."""
    body = document(approximate_tokens, seed)
    return f"{HEADER}\n\n{body}\n\nReply with one short sentence."


def decode_prompt(approximate_tokens: int = 0, seed: int = 0x5EED) -> str:
    """A decode request, optionally behind a document of the given size."""
    if approximate_tokens <= 0:
        return HEADER
    return f"{HEADER}\n\n{document(approximate_tokens, seed)}"
