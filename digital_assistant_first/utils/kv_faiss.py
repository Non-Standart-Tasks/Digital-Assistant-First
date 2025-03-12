from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class KeyValueFAISS(FAISS):
    """
    Modification of langchain FAISS allowes to search by keys and return values.
    For example,

    docs = [Document('Sber'), Document('VTB'), Document('T-Bank')]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    kv_dict = {
        'Sber': 'PJSC Sberbank is a Russian majority state-owned banking and financial services company headquartered in Moscow.',
        'VTB': 'VTB Bank is a Russian majority state-owned bank headquartered in various federal districts of Russia; its legal address is registered in St. Petersburg.',
        'T-Bank': 'T-Bank (Russian: Т-Банк), formerly known as Tinkoff Bank is a Russian commercial bank based in Moscow and founded by Oleg Tinkov in 2006.',
    }

    KeyValueFAISS.from_documents(docs, embeddings) \
        .add_value_documents(kv_dict) \
        .similarity_search_with_score('VTB bank', k=1)

    returns [(Document(page_content='VTB Bank is a Russian majority state-owned bank headquartered in various federal districts of Russia; its legal address is registered in St. Petersburg.'), 
    4.1523235e-12)]
    because 'VTB' in query is equal to 'VTB' in second document's key, so cosine distance between query and key is zero.
    """

    def add_value_documents(self, key_value_dict: dict):
        self.dict = key_value_dict
        return self

    def similarity_search_with_score_by_vector(self, *args, **kwargs):
        unprocessed_docs = super().similarity_search_with_score_by_vector(
            *args, **kwargs
        )
        return [
            (Document(self.dict[doc.page_content]), score)
            for doc, score in unprocessed_docs
        ]
