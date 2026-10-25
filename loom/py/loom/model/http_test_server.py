# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Joined HTTP fixture lifetimes without a polling shutdown delay."""

import socket
import threading
from concurrent.futures import Future
from contextlib import contextmanager


@contextmanager
def serving_http(server):
    """Serves a caller-owned HTTP server until an explicit loopback wakeup."""
    stopped = threading.Event()
    completed = Future()

    def serve():
        try:
            while not stopped.is_set():
                server.handle_request()
        except BaseException as error:
            completed.set_exception(error)
        else:
            completed.set_result(None)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        # EOF wakes the listener without issuing an HTTP request to the fixture.
        # The enclosing server context joins its request handlers after this.
        with socket.create_connection(server.server_address):
            pass
        thread.join()
        completed.result()
