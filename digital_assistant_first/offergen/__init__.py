import json
from functools import lru_cache
from typing import Optional
import yaml

from pydantic import BaseModel, Field

from digital_assistant_first.utils.paths import ROOT_DIR as root_dir
from digital_assistant_first.utils.custom_logging import setup_logging
from digital_assistant_first.offergen.vector_db import VectorDBService, instantiate_db_service


def load_config_yaml(config_file=root_dir / "config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


offersgen_config = load_config_yaml()["offersgen"]
rag_n_examples = offersgen_config['rag_n_examples']
city_filter = offersgen_config['city_filter']
logger = setup_logging(logging_path=str(root_dir / "logs" / "digital_assistant.log"))


class Offer(BaseModel):
    """Model for default offer data in database"""

    category: str
    name: str
    image_url: str
    image_path: str
    offer_url: str
    short_description: str
    full_description: str


with open(root_dir / offersgen_config["offers_db"], "r", encoding="utf-8") as f:
    offers_db = {int(k): Offer(**v) for k, v in json.load(f).items()}


class DBServiceSingleton:
    _instance: Optional[VectorDBService] = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(
        cls,
        json_path: str,
        chunk_overlap: int = 300,
        chunk_size: int = 700,
        dump_path: str = None,
        persist_directory: str = None,
        collection_name: str = "vtb_family_offers",
    ) -> VectorDBService:
        if cls._instance is None:
            logger.info(
                f"Initializing rag db_service for offers generation with json_path={json_path}, chunk_overlap={chunk_overlap}, chunk_size={chunk_size}, dump_path={dump_path}"
            )
            cls._instance = instantiate_db_service(
                json_path=json_path,
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size,
                dump_path=dump_path,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            logger.info(f"Rag db_service for offers generation initialized")
        return cls._instance


db_service_config = offersgen_config["db_service"]
db_service = DBServiceSingleton.get_instance(
    json_path=root_dir / db_service_config["json_path"],
    chunk_overlap=db_service_config["chunk_overlap"],
    chunk_size=db_service_config["chunk_size"],
    dump_path=root_dir / db_service_config["dump_path"],
    persist_directory=str(root_dir / db_service_config["persist_directory"]),
    collection_name=db_service_config["collection_name"],
)
