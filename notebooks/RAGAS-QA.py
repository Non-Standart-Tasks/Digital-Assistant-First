from langchain_community.document_loaders import DirectoryLoader

path = "./notebooks"
loader = DirectoryLoader(path, glob="**/*.txt")
docs = loader.load()

for document in docs:
    document.metadata["filename"] = document.metadata["source"]

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# generator with openai models
generator_llm = Ollama(model="llama3.2", temperature=0.2)
critic_llm = Ollama(model="llama3.2", temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)


# generate testset
testset = generator.generate_with_langchain_docs(
    docs, test_size=1, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

print(testset)
