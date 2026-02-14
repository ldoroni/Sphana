import logging
import sys
import yaml
from pathlib import Path
from injector import singleton


@singleton
class RagConfig:

    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)

        # Resolve config file path relative to __main__.py entry point
        # From: services/sphana-rag-service/src/python/sphana_rag/__main__.py
        # To:   services/sphana-rag-service/src/resources/configs/default.yaml
        current_file = Path(sys.argv[0]).resolve()
        service_root = current_file.parents[2]
        config_path = service_root / "resources" / "configs" / "default.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        rag_config = raw_config.get("sphana", {}).get("rag", {})

        # Database paths
        # database_config = rag_config.get("database", {})
        # self.database_root_dir: str = database_config.get("root_dir", "./.database")
        # self.database_index_details_dir: str = database_config.get("index_details_dir", "./index_details")
        # self.database_shard_details_dir: str = database_config.get("shard_details_dir", "./shard_details")
        # self.database_index_vectors_dir: str = database_config.get("index_vectors_dir", "./index_vectors")
        # self.database_document_details_dir: str = database_config.get("document_details_dir", "./document_details")
        # self.database_parent_chunk_details_dir: str = database_config.get("parent_chunk_details_dir", "./parent_chunk_details")
        # self.database_child_chunk_details_dir: str = database_config.get("child_chunk_details_dir", "./child_chunk_details")

        # Cluster settings
        cluster_config = rag_config.get("cluster", {})
        self.cluster_self_address: str = cluster_config.get("self_address", "localhost:9100")
        self.cluster_nodes_file: str = cluster_config.get("nodes_file", "./nodes.txt")
        self.cluster_backup_count: int = cluster_config.get("backup_count", 1)
        self.cluster_wal_dir: str | None = cluster_config.get("wal_dir", None)

        self.__logger.info(f"RagConfig loaded from {config_path}")
