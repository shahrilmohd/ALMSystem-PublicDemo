"""
Unit tests for worker/job_queue.py.

Coverage
--------
get_redis_connection()
    - returns a Redis client object
    - raises ConnectionError when Redis is unreachable (patched URL)

get_queue()
    - returns an RQ Queue
    - queue is named 'alm_jobs'
    - is_async=False produces a synchronous queue (used in other tests)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from rq import Queue

from worker.job_queue import _QUEUE_NAME, get_queue, get_redis_connection


class TestGetRedisConnection:
    def test_returns_redis_client(self, fake_redis, monkeypatch):
        # Patch get_redis_connection to return the fakeredis instance
        with patch("worker.job_queue.redis.Redis.from_url", return_value=fake_redis):
            conn = get_redis_connection()
        assert conn is fake_redis

    def test_raises_on_unreachable_redis(self):
        import redis as redis_lib
        with patch(
            "worker.job_queue.redis.Redis.from_url",
            side_effect=redis_lib.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(redis_lib.exceptions.ConnectionError):
                get_redis_connection()


class TestGetQueue:
    def test_returns_rq_queue(self, fake_redis):
        with patch("worker.job_queue.get_redis_connection", return_value=fake_redis):
            q = get_queue()
        assert isinstance(q, Queue)

    def test_queue_has_correct_name(self, fake_redis):
        with patch("worker.job_queue.get_redis_connection", return_value=fake_redis):
            q = get_queue()
        assert q.name == _QUEUE_NAME

    def test_is_async_false_produces_synchronous_queue(self, fake_redis):
        with patch("worker.job_queue.get_redis_connection", return_value=fake_redis):
            q = get_queue(is_async=False)
        # A synchronous queue executes jobs immediately in-process
        assert q.is_async is False
