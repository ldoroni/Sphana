"""File-based cluster nodes provider.

Reads a text file where each non-empty line is a ``host:port`` address
of a cluster node.  The file is re-read on every ``get_nodes()`` call
(which the distributed cache polls every ~5 seconds), so external
processes (sidecars, init-containers, DNS resolvers, scripts) can
update the file at any time to reflect pod scale-up/down.
"""

import logging
from pathlib import Path
from distributed_cache import ClusterNodesProvider


class FileClusterNodesProvider(ClusterNodesProvider):
    """Reads cluster node addresses from a plain-text file.

    Args:
        self_address: This node's ``host:port`` reachable by peers.
        nodes_file: Path to a text file with one ``host:port`` per line.
    """

    def __init__(self, self_address: str, nodes_file: str) -> None:
        self._self_address = self_address
        self._nodes_file = Path(nodes_file)
        self._last_known_nodes: list[str] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_nodes(self) -> list[str]:
        """Read the nodes file and return the list of addresses.

        If the file is missing or unreadable, logs a warning and
        returns the last successfully-read list (graceful degradation).
        """
        try:
            text = self._nodes_file.read_text(encoding="utf-8")
            nodes = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if nodes:
                self._last_known_nodes = nodes
            return nodes
        except FileNotFoundError:
            self._logger.warning(
                "Nodes file not found: %s — using last known list (%d nodes)",
                self._nodes_file,
                len(self._last_known_nodes),
            )
            return self._last_known_nodes
        except Exception:
            self._logger.warning(
                "Failed to read nodes file: %s — using last known list (%d nodes)",
                self._nodes_file,
                len(self._last_known_nodes),
                exc_info=True,
            )
            return self._last_known_nodes

    def get_self_address(self) -> str:
        """Return this node's address."""
        return self._self_address