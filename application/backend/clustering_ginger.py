from abc import ABC
from typing import List, Optional
import logging

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk import word_tokenize  # tokenizing
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)

_DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"

# Global shared embedding model (initialized once, reused across all clustering instances)
_SHARED_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

def get_shared_embedding_model() -> SentenceTransformer:
    """Get or initialize the shared embedding model (singleton pattern)."""
    global _SHARED_EMBEDDING_MODEL
    if _SHARED_EMBEDDING_MODEL is None:
        logger.info(f"Initializing shared embedding model: {_DEFAULT_SENTENCE_TRANSFORMERS_MODEL}")
        _SHARED_EMBEDDING_MODEL = SentenceTransformer(_DEFAULT_SENTENCE_TRANSFORMERS_MODEL)
    return _SHARED_EMBEDDING_MODEL

stop_words = list(nltk.corpus.stopwords.words("english"))


def text_preprocessing(text: str):
    """Preprocess the given text.

    Args:
        text: The text to preprocess.

    Returns:
        The lemmatized text with stopwords removed.
    """
    le = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    tokens = [
        le.lemmatize(w)
        for w in word_tokens
        if w not in stop_words and len(w) > 3
    ]
    cleaned_text = " ".join(tokens)
    return cleaned_text


class Clustering(ABC):
    def __init__(self):
        self._clustering_model = None

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """


class LSAClustering(Clustering):
    def __init__(self):
        super().__init__()

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts using LSA.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """
        processed_texts = [text_preprocessing(text) for text in texts]
        vect = TfidfVectorizer(stop_words=stop_words, max_features=1000)
        vect_text = vect.fit_transform(processed_texts)

        lsa_model = TruncatedSVD(
            n_components=int(len(texts) / 2),
            algorithm="randomized",
            n_iter=10,
            random_state=42,
        )
        lsa_matrix = lsa_model.fit_transform(vect_text)

        topics = []
        probabilities = []

        for doc_stats in lsa_matrix:
            topic = np.argmax(doc_stats)
            topics.append(topic)
            probabilities.append(doc_stats[topic])

        return pd.DataFrame(
            {"Document": texts, "Topic": topics, "Probability": probabilities}
        )


class BERTopicClustering(Clustering):
    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 5,
        min_dist: float = 0.0,
        min_cluster_size: int = 3,
        min_samples: int = 1,
    ):
        """Initialize the BERTopic clustering model.

        Args:
            n_neighbors (int, optional): The number of neighbors to consider. Defaults to 50.
            n_components (int, optional): The number of components. Defaults to 2.
            min_dist (float, optional): The minimum distance. Defaults to 0.01.
            min_cluster_size (int, optional): The minimum cluster size. Defaults to 2.
            min_samples (int, optional): The minimum number of samples. Defaults to 1.
        """
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words="english"
        )
        # Reuse the shared embedding model instead of creating a new one
        embedding_model = get_shared_embedding_model()
        
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,  # 3
            min_dist=min_dist,
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        self._clustering_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=True,
            verbose=True,
        )

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts using BERTopic.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """
        if not texts or len(texts) == 0:
            logger.warning("cluster() called with empty text list")
            raise ValueError("Cannot cluster empty text list")
        
        logger.info(f"Clustering {len(texts)} texts...")
        topics, probs = self._clustering_model.fit_transform(texts)
        
        result = self._clustering_model.get_document_info(texts)
        
        if result is None or result.empty:
            logger.error("BERTopic returned empty document info")
            raise ValueError("BERTopic clustering returned empty results")
        
        logger.info(f"Clustering complete: {len(result)} documents, {len(result['Topic'].unique())} topics")
        return result


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    information_nuggets = [
        "Recommended dosage is standard dose",
        "Recommended dosage is standard estimation",
        "Recommended dosage is standard percentage",
        "Rec dosage is standard dose",
        "Patient presents with symptoms of chronic condition",
        "Treatment protocol includes standard therapy",
        "Diagnosis confirmed through laboratory analysis",
        "Side effects may include adverse reaction",
        "Prognosis is favorable with proper management",
        "Clinical findings indicate positive indicator",
        "Contraindications include known allergy",
    ]

    bertopic = BERTopicClustering(min_cluster_size=5)

    bertopic_topic_clusters_count = []

    bertopic_freq = bertopic.cluster(information_nuggets)
    bertopic_topic_clusters_count.append(len(bertopic_freq))

    print(bertopic_freq)

    clustered_docs = bertopic_freq.groupby('Topic')['Document'].apply(list).to_dict()
    clustered_docs.pop(-1, None)
    print(clustered_docs)
