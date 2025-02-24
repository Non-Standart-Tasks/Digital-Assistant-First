import json
import os
from typing import List, Dict, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class CategoryClassifier:
    def __init__(self):
        self.categories = {
            'restaurants': ['ресторан', 'кафе', 'еда', 'кухня', 'меню'],
            'events': ['событие', 'концерт', 'выставка', 'афиша', 'мероприятие'],
            'travel': ['путешествие', 'отель', 'тур', 'экскурсия'],
            # добавьте другие категории
        }

    def classify_query(self, query: str) -> List[str]:
        query = query.lower()
        relevant_categories = []
        for category, keywords in self.categories.items():
            if any(keyword in query for keyword in keywords):
                relevant_categories.append(category)
        return relevant_categories or list(self.categories.keys())

class EnhancedRAGSystem:
    def __init__(self, data_file: str, index_directory: str):
        self.data_file = data_file
        self.index_directory = index_directory
        self.category_classifier = CategoryClassifier()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
        self.vector_stores = {}
        
        # Создаем директории, если они не существуют
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        os.makedirs(index_directory, exist_ok=True)
        
        self.initialize_vector_stores()

    def initialize_vector_stores(self):
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.categorized_data = json.load(f)

        for category, messages in self.categorized_data.items():
            index_path = os.path.join(self.index_directory, f"{category}.faiss")
            
            # Проверяем существование индекса
            if os.path.exists(index_path):
                # Загружаем существующий индекс с явным разрешением десериализации
                self.vector_stores[category] = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Создаем новый индекс
                texts = [msg['text'] for msg in messages]
                if texts:
                    vector_store = FAISS.from_texts(
                        texts,
                        self.embeddings,
                        metadatas=[{
                            'link': msg['link'],
                            'date': msg['date'],
                            'channel': msg['channel']
                        } for msg in messages]
                    )
                    # Сохраняем индекс
                    vector_store.save_local(index_path)
                    self.vector_stores[category] = vector_store

    def format_context(self, results: List[Dict]) -> str:
        context = []
        for result in results:
            text = result['text']
            link = result['metadata']['link']
            context.append(f"Текст: {text}\nИсточник: {link}\n---")
        return "\n".join(context)

    def query(self, query: str, k: int = 5) -> Tuple[List[Dict], str]:
        relevant_categories = self.category_classifier.classify_query(query)
        all_results = []

        for category in relevant_categories:
            if category in self.vector_stores:
                results = self.vector_stores[category].similarity_search_with_score(
                    query,
                    k=k
                )
               
                for doc, score in results:
                    all_results.append({
                        'category': category,
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_score': float(score)
                    })
                    

        # Сортировка по релевантности
        all_results.sort(key=lambda x: x['relevance_score'])
        top_results = all_results[:k]
        
        # Форматируем контекст
        formatted_context = self.format_context(top_results)
        
        return top_results, formatted_context