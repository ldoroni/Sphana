import logging
import sys
import yaml
from pathlib import Path
from injector import singleton


@singleton
class StoreConfig:

    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)

        # Resolve config file path relative to __main__.py entry point
        # From: services/sphana-store-service/src/python/sphana_store/__main__.py
        # To:   services/sphana-store-service/src/resources/configs/default.yaml
        current_file = Path(sys.argv[0]).resolve()
        service_root = current_file.parents[2]
        config_path = service_root / "resources" / "configs" / "default.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        store_config = raw_config.get("sphana", {}).get("store", {})

        # Database paths
        # database_config = store_config.get("database", {})
        # self.database_root_dir: str = database_config.get("root_dir", "./.database")
        # self.database_index_details_dir: str = database_config.get("index_details_dir", "./index_details")
        # self.database_shard_details_dir: str = database_config.get("shard_details_dir", "./shard_details")
        # self.database_index_vectors_dir: str = database_config.get("index_vectors_dir", "./index_vectors")
        # self.database_document_details_dir: str = database_config.get("document_details_dir", "./document_details")
        # self.database_parent_chunk_details_dir: str = database_config.get("parent_chunk_details_dir", "./parent_chunk_details")
        # self.database_child_chunk_details_dir: str = database_config.get("child_chunk_details_dir", "./child_chunk_details")

        # Cluster nodes
        self.nodes: list[str] = store_config.get("nodes", [])
        
        self.__logger.info(f"StoreConfig loaded from {config_path}")