# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import random
import subprocess
import sys
import unittest

from experimental.loom_serve.tools.packing_policy import pack_shapes, pack_spans


class PackingPolicyTest(unittest.TestCase):
    def test_minimums_precede_rotating_expansion(self):
        ready = [(4, 4), (500, 1), (2, 1), (4, 4)]
        self.assertEqual(
            pack_spans(ready, 32, 4, 512, 0),
            (((0, 4), (1, 23), (2, 1), (3, 4)), 1),
        )
        self.assertEqual(
            pack_spans(ready, 32, 4, 512, 2),
            (((2, 2), (3, 4), (0, 4), (1, 22)), 3),
        )

    def test_small_shape_advances_prompt_without_splitting_verifier(self):
        ready = [(4, 4), (1, 1), (4, 4), (0, 0)]
        self.assertEqual(pack_spans(ready, 2, 4, 1, 0), (((1, 1),), 1))
        self.assertEqual(pack_spans(ready, 4, 4, 1, 0), (((0, 4),), 1))
        self.assertEqual(pack_spans(ready, 2, 1, 1, 2), (((1, 1),), 3))

    def test_native_differential(self):
        cases = []
        states = ((0, 0), (1, 1), (2, 1), (4, 4), (19, 1))
        for count in range(1, 5):
            for ready in itertools.product(states, repeat=count):
                for cursor in range(count):
                    cases.append((ready, ((2, 2), (8, 2), (32, 4)), 7, cursor))
        randomizer = random.Random(59317)
        for _ in range(2048):
            count = randomizer.choice((1, 2, 4, 8, 16, 32, 64))
            ready = []
            for _ in range(count):
                tokens = randomizer.choice((0, 1, 2, 4, 7, 31, 512, 4096))
                minimum = min(tokens, randomizer.choice((1, 4, 8)))
                ready.append((tokens, minimum))
            shapes = [
                (randomizer.choice((1, 2, 8, 32, 128, 512)), randomizer.randint(1, 16))
                for _ in range(randomizer.randint(1, 9))
            ]
            cases.append(
                (
                    ready,
                    shapes,
                    randomizer.choice((1, 3, 128, 1024)),
                    randomizer.randrange(count),
                )
            )

        # One native process receives all independent cases. The normal test
        # harness owns hang detection; no wall-clock guess bounds valid work.
        inputs = []
        expected = []
        for ready, shapes, chunk_size, cursor in cases:
            values = [len(ready), len(shapes), chunk_size, cursor]
            values.extend(itertools.chain.from_iterable(ready))
            values.extend(itertools.chain.from_iterable(shapes))
            inputs.append(" ".join(map(str, values)))
            result = pack_shapes(ready, shapes, chunk_size, cursor)
            expected.append(
                [result.shape, result.cursor, len(result.spans)]
                + list(itertools.chain.from_iterable(result.spans))
            )
        completed = subprocess.run(
            [REFERENCE],
            input="\n".join(inputs) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        actual = [
            list(map(int, line.split())) for line in completed.stdout.splitlines()
        ]
        self.assertEqual(len(actual), len(expected))
        for case, observed, predicted in zip(cases, actual, expected):
            self.assertEqual(observed, predicted, case)


if __name__ == "__main__":
    REFERENCE = sys.argv.pop(1)
    unittest.main()
