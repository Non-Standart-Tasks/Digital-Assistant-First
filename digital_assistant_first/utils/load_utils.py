from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters.base import TextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import yaml


class AutoChatOpenAI(ChatOpenAI):
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs).content

def load_model(config: dict) -> BaseLLM:
    if config['type'] == 'ollama':
        return Ollama(
            model=config['name'], 
            temperature=config['temperature'], 
            num_predict=config['max_tokens'],
        )
    
    elif config['type'] == 'chat-gpt':
        return AutoChatOpenAI(
            model=config['name'], 
            temperature=config['temperature'], 
            max_tokens=config['max_tokens'], 
        )

def load_text_splitter(config: dict) -> TextSplitter:
    if config['type'] == 'character':
        return CharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=0, separator=config['separator'])
    # TODO: add json

def load_embeddings(config: dict) -> Embeddings:
    return OpenAIEmbeddings(config['name'])
    
def load_vector_store(config: dict, embeddings: Embeddings, path_to_context: str) -> VectorStore:
    text_splitter = load_text_splitter(config['text_splitter'])

    documents = TextLoader(path_to_context).load()
    docs = text_splitter.split_documents(documents)

    if config['type'] == 'default':
        return FAISS.from_documents(docs, embeddings)
    # TODO: add keyValueFAISS

def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml