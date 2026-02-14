"""TCP connection pool for peer-to-peer communication.

Each peer gets a small pool of reusable TCP sockets to avoid the
overhead of connect/disconnect on every RPC call.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from typing import Any

from . import rpc_protocol
from ..exceptions import RpcConnectionError, RpcTimeoutError

logger = logging.getLogger(__name__)

# Default settings
_DEFAULT_POOL_SIZE = 4
_DEFAULT_CONNECT_TIMEOUT = 5.0
_DEFAULT_RECV_TIMEOUT = 10.0
_DEFAULT_HEALTH_CHECK_INTERVAL = 30.0

class _PooledConnection:
    """Wrapper around a TCP socket with send/recv helpers."""

    __slots__ = ("sock", "address", "_closed")

    def __init__(self, sock: socket.socket, address: str) -> None:
        self.sock = sock
        self.address = address
        self._closed = False

    def send_message(self, data: dict[str, Any]) -> None:
        """Encode and send a length-prefixed msgpack message."""
        frame = rpc_protocol.encode_message(data)
        self.sock.sendall(frame)

    def recv_message(self) -> dict[str, Any]:
        """Receive a length-prefixed msgpack message."""
        header = self._recv_exact(rpc_protocol.header_size())
        length = rpc_protocol.decode_header(header)
        payload = self._recv_exact(length)
        return rpc_protocol.decode_payload(payload)

    def _recv_exact(self, num_bytes: int) -> bytes:
        """Read exactly *num_bytes* from the socket."""
        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError(
                    f"Connection to {self.address} closed while reading"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()

    @property
    def is_closed(self) -> bool:
        return self._closed


class ConnectionPool:
    """Pool of TCP connections to cluster peers.

    Parameters:
        pool_size_per_peer:
            Maximum number of connections to maintain per peer.
        connect_timeout:
            Timeout in seconds for establishing a new TCP connection.
        recv_timeout:
            Timeout in seconds for receiving an RPC response.
    """

    def __init__(
        self,
        pool_size_per_peer: int = _DEFAULT_POOL_SIZE,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        recv_timeout: float = _DEFAULT_RECV_TIMEOUT,
        health_check_interval: float = _DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> None:
        self._pool_size = pool_size_per_peer
        self._connect_timeout = connect_timeout
        self._recv_timeout = recv_timeout
        self._lock = threading.Lock()
        self._pools: dict[str, queue.Queue[_PooledConnection]] = {}
        self._closed = False

        # ── Health check background thread ────────────────────────
        self._health_check_interval = health_check_interval
        self._health_thread: threading.Thread | None = None
        if health_check_interval > 0:
            self._health_thread = threading.Thread(
                target=self._health_check_loop,
                name="connpool-health",
                daemon=True,
            )
            self._health_thread.start()

    # ── Public API ────────────────────────────────────────────────

    def acquire(self, address: str) -> _PooledConnection:
        """Get a connection to *address* from the pool (or create one).

        Args:
            address: ``host:port`` of the target node.

        Returns:
            A :class:`_PooledConnection` ready for use.

        Raises:
            RpcConnectionError: If the connection cannot be established.
        """
        pool = self._get_or_create_pool(address)

        # Try to grab an existing connection
        while True:
            try:
                conn = pool.get_nowait()
            except queue.Empty:
                break
            if conn.is_closed:
                continue
            # Quick liveness check
            if self._is_alive(conn):
                return conn
            conn.close()

        # No reusable connection — create a new one
        return self._create_connection(address)

    def release(self, conn: _PooledConnection) -> None:
        """Return a connection to the pool for reuse.

        If the pool is full, the connection is closed instead.
        """
        if conn.is_closed or self._closed:
            conn.close()
            return

        pool = self._get_or_create_pool(conn.address)
        try:
            pool.put_nowait(conn)
        except queue.Full:
            conn.close()

    def send_and_receive(
        self,
        address: str,
        message: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a message and wait for the response.

        This is the primary convenience method for RPC calls.

        Args:
            address: Target ``host:port``.
            message: Dict to send as the RPC request.
            timeout: Override recv timeout for this call.

        Returns:
            The response dict.

        Raises:
            RpcConnectionError: On connection failure.
            RpcTimeoutError: If the response is not received in time.
        """
        conn = self.acquire(address)
        try:
            conn.send_message(message)
            if timeout is not None:
                conn.sock.settimeout(timeout)
            else:
                conn.sock.settimeout(self._recv_timeout)
            try:
                response = conn.recv_message()
            except socket.timeout as exc:
                conn.close()
                raise RpcTimeoutError(address, timeout or self._recv_timeout) from exc
            self.release(conn)
            return response
        except (ConnectionError, OSError) as exc:
            conn.close()
            raise RpcConnectionError(address, reason=str(exc)) from exc

    # ── Health Check ──────────────────────────────────────────────

    def _health_check_loop(self) -> None:
        """Periodically sweep all pools and evict dead connections."""
        while not self._closed:
            time.sleep(self._health_check_interval)
            if self._closed:
                break
            self._evict_dead_connections()

    def _evict_dead_connections(self) -> None:
        """Check every pooled connection and close dead ones."""
        with self._lock:
            addresses = list(self._pools.keys())

        evicted = 0
        for address in addresses:
            with self._lock:
                pool = self._pools.get(address)
            if pool is None:
                continue

            # Drain, check, re-add healthy connections
            healthy: list[_PooledConnection] = []
            while True:
                try:
                    conn = pool.get_nowait()
                except queue.Empty:
                    break
                if conn.is_closed or not self._is_alive(conn):
                    conn.close()
                    evicted += 1
                else:
                    healthy.append(conn)

            for conn in healthy:
                try:
                    pool.put_nowait(conn)
                except queue.Full:
                    conn.close()

        if evicted:
            logger.info(
                "Connection pool health check: evicted %d dead connections",
                evicted,
            )

    @property
    def pool_stats(self) -> dict[str, int]:
        """Return a snapshot of pool sizes per peer (for metrics)."""
        with self._lock:
            return {addr: pool.qsize() for addr, pool in self._pools.items()}

    def close_all(self) -> None:
        """Close all pooled connections and prevent new ones."""
        self._closed = True
        if self._health_thread is not None:
            self._health_thread.join(timeout=5.0)
            self._health_thread = None
        with self._lock:
            for pool in self._pools.values():
                while True:
                    try:
                        conn = pool.get_nowait()
                        conn.close()
                    except queue.Empty:
                        break
            self._pools.clear()

    def remove_peer(self, address: str) -> None:
        """Close and remove all connections to a specific peer."""
        with self._lock:
            pool = self._pools.pop(address, None)
        if pool:
            while True:
                try:
                    conn = pool.get_nowait()
                    conn.close()
                except queue.Empty:
                    break

    # ── Internal Helpers ──────────────────────────────────────────

    def _get_or_create_pool(self, address: str) -> queue.Queue[_PooledConnection]:
        with self._lock:
            if address not in self._pools:
                self._pools[address] = queue.Queue(maxsize=self._pool_size)
            return self._pools[address]

    def _create_connection(self, address: str) -> _PooledConnection:
        """Open a new TCP socket to the target address."""
        host, port_str = self._parse_address(address)
        port = int(port_str)
        try:
            sock = socket.create_connection(
                (host, port), timeout=self._connect_timeout
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self._recv_timeout)
            return _PooledConnection(sock, address)
        except (ConnectionRefusedError, OSError, socket.timeout) as exc:
            raise RpcConnectionError(address, reason=str(exc)) from exc

    @staticmethod
    def _parse_address(address: str) -> tuple[str, str]:
        """Split ``host:port`` into components."""
        parts = address.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid address format: {address!r}. Expected host:port")
        return parts[0], parts[1]

    @staticmethod
    def _is_alive(conn: _PooledConnection) -> bool:
        """Non-blocking check if the socket is still connected."""
        try:
            conn.sock.setblocking(False)
            data = conn.sock.recv(1, socket.MSG_PEEK)
            conn.sock.setblocking(True)
            # If recv returns b"", the peer has closed
            return len(data) > 0  # noqa: PLR2004 – data waiting is OK
        except BlockingIOError:
            # No data available, but socket is alive
            conn.sock.setblocking(True)
            return True
        except (ConnectionError, OSError):
            return False