"""Kafka/Redpanda transport for session messages.

Accuracy runs never sleep: the producer publishes every session, the consumer reads them
back from the broker by exact offset, and the deterministic barrier (schedule.py) decides
which of the received messages the worker has committed when the query fires.
"""
from __future__ import annotations

import json
import threading

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic


class SessionBus:
    def __init__(self, bootstrap: str, topic: str):
        self.bootstrap = bootstrap
        self.topic = topic
        self._lock = threading.Lock()
        self.producer = Producer({
            "bootstrap.servers": bootstrap, "broker.address.family": "v4", "enable.idempotence": True, "acks": "all",
            "linger.ms": 2, "message.max.bytes": 4 * 1024 * 1024, "compression.type": "lz4",
        })

    def ensure_topic(self) -> None:
        admin = AdminClient({"bootstrap.servers": self.bootstrap, "broker.address.family": "v4"})
        md = admin.list_topics(timeout=10)
        if self.topic in md.topics:
            return
        fut = admin.create_topics([NewTopic(self.topic, num_partitions=1, replication_factor=1)])[self.topic]
        try:
            fut.result(timeout=15)
        except KafkaException as e:  # raced with another creator
            if "TOPIC_ALREADY_EXISTS" not in str(e):
                raise

    def publish(self, ns: str, messages: list[dict]) -> list[int]:
        """Publish in order; returns broker offsets aligned with `messages`."""
        offsets: list[int | None] = [None] * len(messages)
        errors: list[str] = []

        def cb(i):
            def _cb(err, msg):
                if err is not None:
                    errors.append(str(err))
                else:
                    offsets[i] = msg.offset()
            return _cb

        with self._lock:
            for i, m in enumerate(messages):
                self.producer.produce(self.topic, key=ns.encode(), value=json.dumps(m).encode(),
                                      partition=0, on_delivery=cb(i))
                self.producer.poll(0)
            self.producer.flush(30)
        if errors or any(o is None for o in offsets):
            raise RuntimeError(f"publish failed: {errors[:3]}")
        return offsets  # type: ignore[return-value]

    def fetch(self, ns: str, offsets: list[int], timeout_s: float = 60.0) -> list[dict]:
        """Read back exactly the given offsets (any order) from partition 0."""
        want = set(offsets)
        got: dict[int, dict] = {}
        c = Consumer({
            "bootstrap.servers": self.bootstrap, "broker.address.family": "v4", "group.id": f"mem-worker-{ns}",
            "enable.auto.commit": False, "auto.offset.reset": "earliest",
            "fetch.max.bytes": 64 * 1024 * 1024,
        })
        try:
            c.assign([TopicPartition(self.topic, 0, min(offsets))])
            remaining = timeout_s
            while want - got.keys():
                msg = c.poll(1.0)
                remaining -= 1.0
                if msg is None:
                    if remaining <= 0:
                        raise TimeoutError(f"fetch timed out; missing {len(want - got.keys())} offsets")
                    continue
                if msg.error():
                    raise KafkaException(msg.error())
                if msg.offset() in want:
                    got[msg.offset()] = json.loads(msg.value())
                if msg.offset() >= max(offsets) and not (want - got.keys()):
                    break
        finally:
            c.close()
        return [got[o] for o in offsets]
