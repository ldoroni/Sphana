"""Abstract base class for cluster node discovery.

Consumers of this library must implement this interface to provide
the list of peer pod IPs/addresses. The list may change at any time
as pods join or leave the Kubernetes cluster.
"""

from __future__ import annotations

import abc


class ClusterNodesProvider(abc.ABC):
    """Interface that supplies the current set of cluster node addresses.

    Implementations might query the Kubernetes API, read from a headless
    Service DNS, or use any other discovery mechanism.

    Each address must be a string in ``host:port`` format (e.g.
    ``"10.0.0.5:4322"``).  If only an IP is returned, the default
    RPC port will be appended by the caller.
    """

    @abc.abstractmethod
    def get_nodes(self) -> list[str]:
        """Return the current list of peer node addresses.

        This method is called periodically by the cache's membership
        monitor.  It **must** be safe to call from any thread and
        should return promptly (ideally < 1 s).

        Returns:
            A list of ``"host:port"`` strings representing reachable
            cluster members.  The list may include or exclude the
            local node — the cache handles de-duplication internally.
        """
        ...

    @abc.abstractmethod
    def get_self_address(self) -> str:
        """Return the address of the local node.

        Returns:
            The ``"host:port"`` address that other nodes can use to
            reach *this* node.
        """
        ...