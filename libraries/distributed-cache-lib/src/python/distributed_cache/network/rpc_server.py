"""TCP RPC server — accepts incoming connections and dispatches messages.

Runs a background thread that listens for connections and spawns a
handler thread per accepted client.  The server uses length-prefixed
msgpack framing via :mod:`rpc_protocol`.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any, Callable

from . import rpc_protocol

logger = logging.getLogger(__name__)

# Type alias for the request handler callback
RequestHandler = Callable[[dict[str, Any]], dict[str, Any]]

class RpcServer:
    """Multi-threaded TCP RPC server.

    Parameters:
        host: Bind address (use ``"0.0.0.0"`` to listen on all interfaces).
        port: TCP port to listen on.
        handler: Callback invoked for each incoming message.
            It receives the decoded request dict and must return a
            response dict.
        max_connections: Maximum number of concurrent client threads.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4322,
        handler: RequestHandler | None = None,
        max_connections: int = 64,
    ) -> None:
        self._host = host
        self._port = port
        self._handler = handler
        self._max_connections = max_connections

        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._running = False
        self._active_connections: int = 0
        self._connections_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start listening for incoming connections."""
        if self._running:
            return

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # Allow periodic shutdown checks
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(self._max_connections)
        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="rpc-server-accept",
            daemon=True,
        )
        self._accept_thread.start()
        logger.info("RPC server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if self._accept_thread:
            self._accept_thread.join(timeout=5.0)
            self._accept_thread = None
        logger.info("RPC server stopped")

    def set_handler(self, handler: RequestHandler) -> None:
        """Set or replace the request handler callback."""
        self._handler = handler

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def port(self) -> int:
        return self._port

    # ── Accept Loop ───────────────────────────────────────────────

    def _accept_loop(self) -> None:
        """Main loop that accepts incoming TCP connections."""
        while self._running:
            try:
                assert self._server_socket is not None
                client_sock, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.exception("Accept error")
                break

            with self._connections_lock:
                if self._active_connections >= self._max_connections:
                    logger.warning(
                        "Max connections reached (%d), rejecting %s",
                        self._max_connections,
                        addr,
                    )
                    client_sock.close()
                    continue
                self._active_connections += 1

            client_thread = threading.Thread(
                target=self._handle_client,
                args=(client_sock, addr),
                name=f"rpc-client-{addr[0]}:{addr[1]}",
                daemon=True,
            )
            client_thread.start()

    # ── Client Handler ────────────────────────────────────────────

    def _handle_client(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
    ) -> None:
        """Handle a single client connection (one or more messages)."""
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_sock.settimeout(30.0)

        try:
            while self._running:
                # Read header
                try:
                    header = self._recv_exact(client_sock, rpc_protocol.header_size())
                except (ConnectionError, OSError):
                    break

                if not header:
                    break

                # Read payload
                try:
                    length = rpc_protocol.decode_header(header)
                    payload_bytes = self._recv_exact(client_sock, length)
                    request = rpc_protocol.decode_payload(payload_bytes)
                except (ValueError, ConnectionError, OSError) as exc:
                    logger.warning("Bad message from %s: %s", addr, exc)
                    break

                # Dispatch to handler
                if self._handler is None:
                    response: dict[str, Any] = {
                        "success": False,
                        "error": "No handler registered",
                    }
                else:
                    try:
                        response = self._handler(request)
                    except Exception:
                        logger.exception("Handler error for request from %s", addr)
                        response = {
                            "success": False,
                            "error": "Internal server error",
                        }

                # Send response
                try:
                    response_frame = rpc_protocol.encode_message(response)
                    client_sock.sendall(response_frame)
                except (ConnectionError, OSError):
                    break

        finally:
            client_sock.close()
            with self._connections_lock:
                self._active_connections -= 1

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
        """Read exactly *num_bytes* from a socket."""
        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError("Connection closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)