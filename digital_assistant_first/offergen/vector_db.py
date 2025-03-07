import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, Field

from digital_assistant_first.utils.custom_logging import setup_logging
from digital_assistant_first.utils.paths import ROOT_DIR as root_dir

logger = setup_logging(logging_path=str(root_dir / "logs" / "digital_assistant.log"))


class Context(BaseModel):
    """Document context with content and metadata."""

    content: str = Field(description="Document content")
    metadata: Dict[str, Any] = Field(description="Document metadata")


class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str = Field(description="Search query")
    k: int = Field(description="Number of results to return", default=5)


class SearchResponse(BaseModel):
    """Search response with documents and relevance scores."""

    documents: List[Context] = Field(
        description="List of documents with context and metadata"
    )
    scores: List[float] = Field(description="List of scores for each document")


class VectorDBService:
    """Vector database service for document storage and semantic search.

    Processes JSON documents into chunks, embeds them using OpenAI,
    and provides semantic search functionality via Chroma vector store.

    If persist_directory is not empty, the vectorstore will be loaded from it.
    If persist_directory is empty, the vectorstore will be created from scratch.
    It takes some time to create the vectorstore from scratch.

    Args:
        json_path: Path to source JSON data
        persist_directory: Path to existing or new vectorstore
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        collection_name: Name of Chroma collection
    """

    def __init__(
        self,
        json_path: str,
        persist_directory: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        collection_name: str = "vtb_docs",
    ) -> None:
        """Initialize the VectorDBService.

        Args:
            json_path: Path to source JSON data
            persist_directory: Path to existing or new vectorstore
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            collection_name: Name of Chroma collection
        """
        load_dotenv(root_dir / ".env")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_api_key,
        )
        self.json_data = self._load_json(json_path)

        # Try to load existing vectorstore if persist_directory exists and is not empty
        if (
            persist_directory
            and os.path.exists(persist_directory)
            and os.listdir(persist_directory)
        ):
            self.documents, self.text_splitter, self.splits = None, None, None
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
        else:
            # Create new vectorstore if no persist_directory or it's empty

            self.documents = self._prepare_documents()

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            self.splits = self.text_splitter.split_documents(self.documents)

            self.vectorstore = Chroma.from_documents(
                documents=self.splits,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )

    def _load_json(self, path: str) -> Dict:
        """Load JSON data from file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded JSON data
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_documents(self) -> List[Document]:
        """Process JSON data into documents.

        Traverses JSON structure and converts entries into Document objects
        with appropriate metadata and formatting.

        Returns:
            List of processed Document objects
        """
        documents: List[Document] = []

        def clean_text(text: str) -> str:
            return (
                text.strip()
                .replace("«", "")
                .replace("»", "")
                .replace("\t", " ")
                .replace("  ", " ")
            )

        def make_document(content: str, path: List[str]) -> None:
            readable_path = " > ".join(path)
            content_clean = clean_text(content)

            definition_match = re.match(r"^([^:]+):\s*(.+)$", content_clean)
            is_definition = bool(definition_match)

            # Extract category and offer name from path
            path_parts = readable_path.split(" > ")
            if len(path_parts) >= 2:
                category = path_parts[0]
                offer_name = path_parts[1]
                # Get offer_url from original data
                offer_url = (
                    self.json_data.get(category, {})
                    .get(offer_name, {})
                    .get("offer_url", "")
                )
            else:
                offer_url = ""

            metadata = {
                "doc_id": str(uuid.uuid4()),
                "path": readable_path,
                "section": path[-1] if path else "root",
                "depth": str(len(path)),
                "is_definition": str(is_definition),
                "content_type": "definition" if is_definition else "content",
                "offer_url": offer_url,  # Add offer_url to metadata
                "image_url": self.json_data.get(category, {})
                .get(offer_name, {})
                .get("image_url", ""),
                "title": self.json_data.get(category, {})
                .get(offer_name, {})
                .get("title", ""),
                "category": category,
            }

            if is_definition:
                term, definition = definition_match.groups()
                term = term.strip()
                definition = definition.strip()
                final_content = f"{term}\n{definition}"
            else:
                final_content = content_clean

            context_lines = []
            if path:
                context_lines.append(f"Контекст: {readable_path}")

            doc_text = "\n".join(filter(None, [*context_lines, final_content]))

            documents.append(Document(page_content=doc_text, metadata=metadata))

        def process_table_row(row: Dict[str, Any], path: List[str]) -> None:
            row_content_lines = []
            for col_key, col_val in row.items():

                if isinstance(col_val, list):
                    items = [str(item).strip() for item in col_val if str(item).strip()]
                    if items:
                        bullet_points = "\n• " + "\n• ".join(items)
                        row_content_lines.append(f"{col_key}:{bullet_points}")
                elif col_val:
                    row_content_lines.append(f"{col_key}: {col_val}")
            if row_content_lines:
                row_content = "\n".join(row_content_lines)
                make_document(row_content, path)

        def process_list_of_strings(lst: List[Any], path: List[str]) -> None:
            items = []
            for idx, item in enumerate(lst, 1):
                if isinstance(item, str) and item.strip():
                    items.append(f"{idx}. {item.strip()}")
                elif isinstance(item, dict):
                    for k, v in item.items():
                        if v:
                            traverse_json(k, v, path)
            if items:
                make_document("\n".join(items), path)

        def traverse_json(key: Optional[str], value: Any, path: List[str]) -> None:
            if not value:
                return
            current_path = path[:] + ([key] if key else [])

            if isinstance(value, dict):

                for k, v in value.items():
                    if not v:
                        continue
                    if k.lower().startswith("таблица") and isinstance(v, list):
                        for row_idx, row in enumerate(v, 1):
                            if isinstance(row, dict):
                                row_path = current_path + [f"{k} строка {row_idx}"]
                                process_table_row(row, row_path)
                            elif row:
                                content = f"{k} строка {row_idx}: {row}"
                                make_document(content, current_path)
                    else:
                        # Pass offer_url down the traversal
                        traverse_json(k, v, current_path)

            elif isinstance(value, list):
                process_list_of_strings(value, current_path)

            elif isinstance(value, (str, int, float)) and key == "content":
                content = f"{key}: {value}" if key else str(value)
                make_document(content, current_path)

        traverse_json(None, self.json_data, [])
        return documents

    def search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Search for documents similar to query.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of (Document, similarity_score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def close(self):
        """Clean up resources."""
        if hasattr(self, "vectorstore"):
            self.vectorstore = None
        if hasattr(self, "embeddings"):
            self.embeddings = None

    def dump_documents(self, output_path: str | Path) -> None:
        """Dump processed documents to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        if self.documents:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            docs_data = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
        else:
            logger.info(
                "Skipping document dump: No documents available in vector DB service. "
                "This is expected when using a pre-processed database from persist_directory."
            )


def instantiate_db_service(
    json_path: str,
    chunk_overlap: int = 300,
    chunk_size: int = 700,
    dump_path: str = None,
    persist_directory: str = None,
    collection_name: str = "vtb_family_offers",
) -> VectorDBService:
    """Instantiate the VectorDBService.

    Args:
        json_path: Path to source JSON data
        chunk_overlap: Overlap between chunks
        chunk_size: Size of text chunks for splitting
        dump_path: Path to output JSON file
        persist_directory: Path to existing or new vectorstore
        collection_name: Name of Chroma collection
    """
    db_service = VectorDBService(
        json_path=json_path,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    if dump_path:
        db_service.dump_documents(dump_path)
    return db_service
