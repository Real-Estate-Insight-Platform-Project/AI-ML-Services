"""
Advanced Text embeddings and theme mining for review comments.

Implements sophisticated theme mining from your original proposal:
- Sentence transformer embeddings (all-MiniLM-L6-v2) for review comments
- UMAP dimensionality reduction for high-quality clustering space
- HDBSCAN clustering to discover natural review themes
- c-TF-IDF for automatic theme labeling with domain-aware patterns
- Sentiment-weighted Agent × Theme matrix with recency decay
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class AdvancedThemeMiner:
    """Advanced text processing and theme discovery using embeddings + UMAP + HDBSCAN."""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 min_cluster_size: int = 10,
                 min_samples: int = 5,
                 umap_n_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 umap_n_components: int = 50,
                 recency_halflife_days: float = 365.0):
        """
        Initialize advanced theme mining system.
        
        Args:
            embedding_model: Sentence transformer model name
            min_cluster_size: Minimum size for HDBSCAN clusters
            min_samples: Minimum samples for HDBSCAN core points
            umap_n_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            umap_n_components: UMAP output dimensions for clustering
            recency_halflife_days: Half-life for recency decay (days)
        """
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.recency_halflife = recency_halflife_days
        
        # Models (initialized lazily)
        self._embedding_model = None
        self._umap_model = None
        self._cluster_model = None
        
        # Results
        self.embeddings = None
        self.umap_embeddings = None
        self.cluster_labels = None
        self.theme_labels = {}
        self.theme_keywords = {}
    
    @property
    def embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is None:
            logger.info(f"Loading sentence transformer model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Clean and preprocess review comments."""
        logger.info(f"Preprocessing {len(texts)} review comments...")
        
        processed = []
        for text in texts:
            if pd.isna(text) or text == '':
                processed.append("")
                continue
            
            # Basic cleaning
            text = str(text).strip()
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove very short texts (less than 10 characters)
            if len(text) < 10:
                processed.append("")
                continue
            
            processed.append(text)
        
        return processed
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings for review comments."""
        logger.info("Generating sentence embeddings...")
        
        # Filter out empty texts for embedding
        non_empty_indices = [i for i, text in enumerate(texts) if text != ""]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        if len(non_empty_texts) == 0:
            logger.warning("No valid texts for embedding")
            return np.zeros((len(texts), 384))  # Default embedding size
        
        # Generate embeddings
        embeddings_subset = self.embedding_model.encode(
            non_empty_texts,
            batch_size=32,
            show_progress_bar=True
        )
        
        # Create full embedding matrix with zeros for empty texts
        embedding_dim = embeddings_subset.shape[1]
        full_embeddings = np.zeros((len(texts), embedding_dim))
        
        for i, orig_idx in enumerate(non_empty_indices):
            full_embeddings[orig_idx] = embeddings_subset[i]
        
        self.embeddings = full_embeddings
        return full_embeddings
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction for clustering (high-dimensional) and visualization."""
        logger.info("Applying UMAP dimensionality reduction...")
        
        # Filter out zero embeddings (empty texts)
        non_zero_mask = np.any(embeddings != 0, axis=1)
        
        if non_zero_mask.sum() < self.umap_n_neighbors:
            logger.warning(f"Too few non-zero embeddings ({non_zero_mask.sum()}) for UMAP")
            # Return appropriate dimensions for clustering
            umap_embeddings = np.zeros((len(embeddings), self.umap_n_components))
            if non_zero_mask.sum() > 1:
                # Simple PCA-like projection 
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(self.umap_n_components, non_zero_mask.sum() - 1))
                umap_embeddings[non_zero_mask] = pca.fit_transform(embeddings[non_zero_mask])
        else:
            # Apply UMAP for high-dimensional clustering space (not just 2D visualization)
            self._umap_model = umap.UMAP(
                n_neighbors=min(self.umap_n_neighbors, non_zero_mask.sum() - 1),
                min_dist=self.umap_min_dist,
                n_components=self.umap_n_components,  # Use higher dimensions for clustering
                random_state=42
            )
            
            # Fit on non-zero embeddings
            reduced_subset = self._umap_model.fit_transform(embeddings[non_zero_mask])
            
            # Create full result array
            umap_embeddings = np.zeros((len(embeddings), self.umap_n_components))
            umap_embeddings[non_zero_mask] = reduced_subset
        
        self.umap_embeddings = umap_embeddings
        return umap_embeddings
    
    def discover_themes(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering to discover review themes."""
        logger.info("Discovering themes with HDBSCAN clustering...")
        
        # Filter out zero embeddings
        non_zero_mask = np.any(embeddings != 0, axis=1)
        
        if non_zero_mask.sum() < self.min_cluster_size:
            logger.warning(f"Too few valid embeddings ({non_zero_mask.sum()}) for clustering")
            # Assign all to noise cluster (-1)
            cluster_labels = np.full(len(embeddings), -1, dtype=int)
        else:
            # Apply HDBSCAN clustering
            self._cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=min(self.min_cluster_size, non_zero_mask.sum() // 3),
                min_samples=min(self.min_samples, non_zero_mask.sum() // 5),
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            # Fit on non-zero embeddings
            cluster_labels_subset = self._cluster_model.fit_predict(embeddings[non_zero_mask])
            
            # Create full label array
            cluster_labels = np.full(len(embeddings), -1, dtype=int)
            cluster_labels[non_zero_mask] = cluster_labels_subset
        
        self.cluster_labels = cluster_labels
        
        # Log cluster statistics
        unique_labels = np.unique(cluster_labels)
        logger.info(f"Discovered {len(unique_labels)} clusters (including noise)")
        
        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            if label == -1:
                logger.info(f"Noise cluster: {count} reviews")
            else:
                logger.info(f"Cluster {label}: {count} reviews")
        
        return cluster_labels
    
    def label_themes_with_ctfidf(self, texts: List[str], cluster_labels: np.ndarray) -> Dict[int, str]:
        """Automatically label themes using c-TF-IDF analysis."""
        logger.info("Labeling themes with c-TF-IDF...")
        
        theme_labels = {}
        theme_keywords = {}
        
        # Get unique cluster labels (excluding noise)
        unique_labels = [label for label in np.unique(cluster_labels) if label != -1]
        
        if len(unique_labels) == 0:
            logger.warning("No valid clusters found for theme labeling")
            return theme_labels
        
        # Prepare documents for each cluster
        cluster_documents = {}
        for label in unique_labels:
            # Combine all texts in this cluster
            cluster_texts = [texts[i] for i in range(len(texts)) 
                           if cluster_labels[i] == label and texts[i] != ""]
            
            if len(cluster_texts) > 0:
                # Combine all reviews in cluster into single document
                cluster_documents[label] = " ".join(cluster_texts)
        
        if len(cluster_documents) == 0:
            return theme_labels
        
        # Apply TF-IDF
        documents = list(cluster_documents.values())
        labels = list(cluster_documents.keys())
        
        # Adjust min_df/max_df based on available cluster documents to avoid
        # the common error where min_df > number_of_documents or max_df
        # becomes incompatible with min_df for very small corpora.
        n_documents = len(documents)
        min_df_val = 2 if n_documents >= 2 else 1
        # If only one document, allow max_df=1.0 so it doesn't filter everything
        max_df_val = 0.95 if n_documents >= 2 else 1.0

        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=min_df_val,
            max_df=max_df_val
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform(documents)
            feature_names = tfidf.get_feature_names_out()
            
            # Extract top keywords for each cluster
            for i, label in enumerate(labels):
                # Get TF-IDF scores for this cluster
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # Get top keywords
                top_indices = np.argsort(tfidf_scores)[-10:][::-1]
                top_keywords = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
                
                # Generate theme label from keywords
                if len(top_keywords) >= 2:
                    theme_name = self._generate_theme_name(top_keywords[:5])
                    theme_labels[label] = theme_name
                    theme_keywords[label] = top_keywords[:10]
                else:
                    theme_labels[label] = f"Theme_{label}"
                    theme_keywords[label] = top_keywords
        
        except Exception as e:
            logger.error(f"Error in c-TF-IDF analysis: {e}")
            # Fallback to simple labels
            for label in unique_labels:
                theme_labels[label] = f"Theme_{label}"
                theme_keywords[label] = []
        
        self.theme_labels = theme_labels
        self.theme_keywords = theme_keywords
        
        # Log discovered themes
        for label, name in theme_labels.items():
            keywords = ", ".join(theme_keywords.get(label, [])[:5])
            logger.info(f"Theme {label}: '{name}' - Keywords: {keywords}")
        
        return theme_labels
    
    def _generate_theme_name(self, keywords: List[str]) -> str:
        """Generate human-readable theme name from top keywords."""
        # Simple heuristic theme naming
        keywords_lower = [k.lower() for k in keywords]
        
        # Common theme patterns
        theme_patterns = {
            'communication': ['communication', 'responsive', 'response', 'contact', 'call', 'email', 'quick'],
            'negotiation': ['negotiation', 'negotiate', 'deal', 'price', 'offer', 'contract'],
            'professionalism': ['professional', 'knowledgeable', 'expert', 'experienced', 'reliable'],
            'first_time_buyers': ['first time', 'first-time', 'new buyer', 'guidance', 'explained'],
            'market_knowledge': ['market', 'area', 'neighborhood', 'local', 'pricing', 'value'],
            'selling_process': ['sell', 'selling', 'listed', 'marketing', 'showing'],
            'buying_process': ['buy', 'buying', 'purchase', 'found', 'search'],
            'closing': ['closing', 'close', 'paperwork', 'process', 'smooth'],
            'availability': ['available', 'time', 'schedule', 'flexible', 'accommodating']
        }
        
        # Check for pattern matches
        for theme, pattern_words in theme_patterns.items():
            if any(word in ' '.join(keywords_lower) for word in pattern_words):
                return theme.replace('_', ' ').title()
        
        # Fallback: use top 2 keywords
        return f"{keywords[0].title()} & {keywords[1].title()}" if len(keywords) >= 2 else keywords[0].title()
    
    def calculate_agent_theme_matrix(self, 
                                   reviews_df: pd.DataFrame,
                                   cluster_labels: np.ndarray,
                                   agents_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Agent × Theme matrix with sentiment weighting and recency decay."""
        logger.info("Calculating Agent × Theme matrix...")
        
        # Add cluster labels to reviews
        reviews_with_themes = reviews_df.copy()
        reviews_with_themes['theme_cluster'] = cluster_labels
        
        # Get unique themes (excluding noise)
        unique_themes = [label for label in np.unique(cluster_labels) if label != -1]
        theme_names = [self.theme_labels.get(label, f"Theme_{label}") for label in unique_themes]
        
        # Initialize agent theme matrix
        agent_ids = agents_df['advertiser_id'].values
        theme_matrix = pd.DataFrame(
            0.0,
            index=agent_ids,
            columns=[f"theme_{name.lower().replace(' ', '_')}" for name in theme_names]
        )
        
        # Calculate theme strengths for each agent
        for agent_id in agent_ids:
            agent_reviews = reviews_with_themes[
                reviews_with_themes['advertiser_id'] == agent_id
            ]
            
            if len(agent_reviews) == 0:
                continue
            
            # For each theme, calculate sentiment-weighted frequency with recency decay
            for i, theme_label in enumerate(unique_themes):
                theme_reviews = agent_reviews[agent_reviews['theme_cluster'] == theme_label]
                
                if len(theme_reviews) == 0:
                    continue
                
                # Calculate weighted score
                # Sentiment weight: positive reviews get weight 1.0, others get 0.5
                sentiment_weights = theme_reviews['is_positive'].astype(float)
                sentiment_weights = sentiment_weights.where(sentiment_weights == 1.0, 0.5)
                
                # Recency weights (already calculated in data processor)
                recency_weights = theme_reviews['recency_weight']
                
                # Combined weight
                combined_weights = sentiment_weights * recency_weights
                
                # Theme strength = sum of weighted occurrences / total reviews
                theme_strength = combined_weights.sum() / max(len(agent_reviews), 1)
                
                # Store in matrix
                col_name = f"theme_{theme_names[i].lower().replace(' ', '_')}"
                theme_matrix.loc[agent_id, col_name] = theme_strength
        
        logger.info(f"Created theme matrix: {theme_matrix.shape[0]} agents × {theme_matrix.shape[1]} themes")
        
        return theme_matrix
    
    def fit_transform(self, reviews_df: pd.DataFrame, agents_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Complete theme mining pipeline."""
        logger.info("Starting complete theme mining pipeline...")
        
        # Extract and preprocess texts
        texts = reviews_df['review_comment'].fillna('').tolist()
        processed_texts = self.preprocess_text(texts)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(processed_texts)
        
        # Reduce dimensions
        umap_embeddings = self.reduce_dimensions(embeddings)
        
        # Discover themes
        cluster_labels = self.discover_themes(embeddings)
        
        # Label themes
        theme_labels = self.label_themes_with_ctfidf(processed_texts, cluster_labels)
        
        # Calculate agent theme matrix
        agent_theme_matrix = self.calculate_agent_theme_matrix(
            reviews_df, cluster_labels, agents_df
        )
        
        # Return results
        results = {
            'embeddings': embeddings,
            'umap_embeddings': umap_embeddings,
            'cluster_labels': cluster_labels,
            'theme_labels': theme_labels,
            'theme_keywords': self.theme_keywords,
            'n_themes': len([l for l in np.unique(cluster_labels) if l != -1]),
            'noise_ratio': np.mean(cluster_labels == -1)
        }
        
        logger.info("Theme mining pipeline completed successfully")
        logger.info(f"Discovered {results['n_themes']} themes with {results['noise_ratio']:.2%} noise")
        
        return agent_theme_matrix, results