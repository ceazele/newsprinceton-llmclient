"""Failure-path tests for BidirectionalClient (Oct 2026 connection storms). No server: the gRPC stub is faked."""
import asyncio
import os
import unittest
from unittest import mock

import LlmClient.LlmLib as L


class FakeCall:
    """A streaming call: write() answers hello/__initengine__ by resolving the owner's pending future; cancel() is counted."""
    def __init__(self, owner_ref, answer=True, die=None):
        self.owner_ref, self.answer, self.cancelled, self.die = owner_ref, answer, 0, die
        self._msgs = asyncio.Queue()

    async def write(self, msg):
        owner = self.owner_ref()
        if self.answer and owner is not None and owner._msg_future and not owner._msg_future.done():
            owner._msg_future.set_result(L.SimpleMessage(mtype="fresh", payload=b""))

    def cancel(self):
        self.cancelled += 1

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.die is not None:                # a stream that resets once ``die`` is set
            await self.die.wait()
            raise RuntimeError("stream reset")
        return await self._msgs.get()          # never yields unless fed; the reader task just waits


class FakeConnection:
    def __init__(self):
        self.stub = mock.Mock()
        self.calls = []
        self.stub.BidirectionalMessage = self._new_call
        self.owner = lambda: None

    def _new_call(self):
        c = FakeCall(self.owner)
        self.calls.append(c)
        return c


def fast_sleep(*_a, **_k):
    async def _s(): pass
    return _s()


class ConnectStormTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("LLM_USER_CODE", "test")

    def test_failed_connect_cancels_its_streams(self):
        """A: every failed attempt leaves zero live streams behind (the leak that filled the shared channel)."""
        conn = FakeConnection()
        attempts = {"n": 0}
        orig = L.BidirectionalClient._connect

        async def flaky(self):
            conn.owner = lambda: self
            attempts["n"] += 1
            if attempts["n"] <= 3:
                self.call = conn.stub.BidirectionalMessage()         # streams opened, then the handshake fails
                self.heartbeat_call = conn.stub.BidirectionalMessage()
                raise RuntimeError("handshake failed")
            self.is_connected = True

        with mock.patch.object(L.BidirectionalClient, "_connect", flaky), mock.patch.object(L.asyncio, "sleep", fast_sleep):
            client = asyncio.run(L.BidirectionalClient.create("llm", conn))
        self.assertTrue(client.is_connected)
        self.assertEqual(attempts["n"], 4)
        self.assertEqual(len(conn.calls), 6)
        self.assertEqual([c.cancelled for c in conn.calls], [1] * 6)   # 2 streams x 3 failed attempts, each cancelled once

    def test_initengine_handshake_is_bounded(self):
        """B: the __initengine__ exchange runs under a timeout like the hello (an unanswered one no longer hangs)."""
        conn = FakeConnection()
        timeouts = []
        real_wait_for = asyncio.wait_for

        async def recording_wait_for(aw, timeout=None):
            timeouts.append(timeout)
            return await real_wait_for(aw, timeout=timeout)

        with mock.patch.object(L.asyncio, "wait_for", recording_wait_for):
            async def run():
                c = L.BidirectionalClient("llm", conn)
                conn.owner = lambda: c
                await c._connect()
                await c.close()
            asyncio.run(run())
        self.assertEqual(timeouts, [10, 30])

    def test_dead_stream_fails_the_pending_reply_at_once(self):
        """C: when the call stream dies, a waiting send_receive raises now (not after SendSurely's 300 s), and so does
        any later send on that client, whichever happens first."""
        conn = FakeConnection()

        async def run():
            # case 1: the request is pending, then the stream dies
            c = L.BidirectionalClient("llm", conn)
            conn.owner = lambda: c
            gate = asyncio.Event()
            c.call = FakeCall(lambda: c, answer=False, die=gate)
            reader = asyncio.create_task(c._read_call_stream())
            pending = asyncio.create_task(c.send_receive(L.SimpleMessage(mtype="ask", payload=b"")))
            await asyncio.sleep(0.05)
            t0 = asyncio.get_running_loop().time()
            gate.set()
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(pending, timeout=5)
            await reader
            self.assertLess(asyncio.get_running_loop().time() - t0, 1.0)
            self.assertFalse(c.is_connected)
            # case 2: the stream died first; the next send must not wait at all
            with self.assertRaises(ConnectionError):
                await asyncio.wait_for(c.send_receive(L.SimpleMessage(mtype="ask", payload=b"")), timeout=5)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
