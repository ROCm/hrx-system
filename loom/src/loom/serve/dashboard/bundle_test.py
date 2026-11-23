# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks the actual independently negotiable dashboard representations."""

import gzip
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and "src" in attrs:
            self.references.add(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.references.add(attrs["href"])


class DashboardBundleTest(unittest.TestCase):
    def test_embedded_resources(self):
        names = {file.name for file in self.files}
        total = 0
        for file in self.files:
            with self.subTest(asset=file.name):
                encoded = file.read_bytes()
                total += len(encoded)
                self.assertEqual(encoded[:10], bytes.fromhex("1f8b08000000000002ff"))
                source = gzip.decompress(encoded).decode()
                self.assertTrue(source.strip())
                if file.suffix == ".html":
                    resources = Resources()
                    resources.feed(source)
                    self.assertLessEqual(resources.references, names)
        self.assertLess(total, 50_000)


if __name__ == "__main__":
    DashboardBundleTest.files = list(map(Path, sys.argv[1:]))
    unittest.main(argv=sys.argv[:1])
