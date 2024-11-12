import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from sklearn.neighbors import NearestNeighbors
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, documents):
        self.documents = documents
        self.graph = self._build_graph()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._process_documents()
        self._build_vector_index()

    def _process_documents(self):
        if not self.documents:
            self.tfidf_matrix = None
            return

        # Filter out empty documents
        non_empty_docs = [doc for doc in self.documents if doc.strip()]

        if not non_empty_docs:
            self.tfidf_matrix = None
            return

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(non_empty_docs)
            logger.info("Documents processed successfully")
        except ValueError as e:
            logger.error(f"Error processing documents: {str(e)}")
            self.tfidf_matrix = None

    def _build_graph(self):
        G = nx.Graph()
        for i, doc in enumerate(self.documents):
            G.add_node(i, content=doc)
        return G

    def _add_edges(self, threshold=0.3):
        if self.tfidf_matrix is None:
            return

        for i in range(self.tfidf_matrix.shape[0]):
            for j in range(i + 1, self.tfidf_matrix.shape[0]):
                similarity = cosine_similarity(self.tfidf_matrix[i], self.tfidf_matrix[j])[0][0]
                if similarity > threshold:
                    self.graph.add_edge(i, j, weight=similarity)

    def _build_vector_index(self):
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # or "text-embedding-ada-002"
            )
            embeddings = [self.embeddings.embed_query(doc) for doc in self.documents]
            self.index = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.index.fit(embeddings)
            logger.info("Vector index built successfully")
        except Exception as e:
            logger.error(f"Error building vector index: {str(e)}")
            self.index = None

    def query(self, query):
        if self.tfidf_matrix is None or self.index is None:
            logger.warning("No processable documents available for querying.")
            return "No processable documents available for querying."

        try:
            query_embedding = self.embeddings.embed_query(query)
            distances, indices = self.index.kneighbors([query_embedding], n_neighbors=3)

            response = "Based on the query, here are the most relevant document excerpts:\n\n"
            for i, doc_index in enumerate(indices[0], 1):
                response += f"{i}. {self.documents[doc_index][:200]}...\n\n"

            logger.info("Query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing the query: {str(e)}"

    def visualize_graph(self):
        if len(self.graph.nodes) == 0:
            logger.warning("No documents to visualize")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No documents to visualize", ha='center', va='center', fontsize=12)
            plt.axis('off')
            return plt.gcf()

        try:
            pos = nx.spring_layout(self.graph)
            plt.figure(figsize=(12, 8))
            nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                    node_size=500, font_size=10, font_weight='bold')
            edge_weights = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_weights)
            plt.title("Document Relationship Graph")
            logger.info("Graph visualization created successfully")
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error creating graph visualization: {str(e)}")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error visualizing graph: {str(e)}", ha='center', va='center', fontsize=12)
            plt.axis('off')
            return plt.gcf() 