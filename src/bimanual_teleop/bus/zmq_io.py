"""Minimal ZeroMQ PUB/SUB helpers with LATEST-VALUE semantics — the load-bearing
rate-decoupling primitive for the multi-process hardware path.

A producer binds one Publisher; consumers connect a LatestSub to the producers
they need and call poll() each tick to drain the socket, keeping only the newest
message PER TOPIC (so a slow/stalled producer can never back up a real-time loop).
This is the localhost, no-broker form; the schema (bus/topics.py) is ROS-portable
if you later migrate.
"""
from __future__ import annotations

import zmq

from . import topics


class Publisher:
    def __init__(self, endpoint: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(endpoint)

    def send(self, topic: str, obj: dict) -> None:
        self.sock.send_multipart([topic.encode(), topics.pack(obj)])

    def close(self) -> None:
        self.sock.close(0)


class LatestSub:
    def __init__(self, endpoints, sub_topics):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        for ep in (endpoints if isinstance(endpoints, (list, tuple)) else [endpoints]):
            self.sock.connect(ep)
        for t in (sub_topics if isinstance(sub_topics, (list, tuple)) else [sub_topics]):
            self.sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        self._latest: dict[str, dict] = {}

    def poll(self) -> None:
        """Drain everything available; keep only the newest msg per topic."""
        while True:
            try:
                topic, buf = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            self._latest[topic.decode()] = topics.unpack(buf)

    def get(self, topic: str) -> dict | None:
        return self._latest.get(topic)

    def close(self) -> None:
        self.sock.close(0)
