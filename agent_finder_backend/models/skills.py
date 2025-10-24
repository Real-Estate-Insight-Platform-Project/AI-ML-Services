import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import re
from config.settings import settings


class SkillExtractor:
    """Extract skills and qualities from review comments using semantic similarity."""
    
    # Define skill categories (excluding the ones already in sub-scores)
    POSITIVE_SKILLS = {
        'communication': [
            'communicate', 'communication', 'responsive', 'available', 'contact',
            'reach', 'answer', 'reply', 'update', 'inform', 'accessible'
        ],
        'local_knowledge': [
            'local', 'area', 'neighborhood', 'market', 'community', 'schools',
            'location', 'region', 'district', 'familiar', 'knowledge of area'
        ],
        'attention_to_detail': [
            'detail', 'thorough', 'careful', 'meticulous', 'diligent', 
            'organized', 'prepared', 'documentation', 'paperwork'
        ],
        'patience': [
            'patient', 'understanding', 'calm', 'supportive', 'reassuring',
            'time', 'flexible', 'accommodating', 'listened'
        ],
        'honesty': [
            'honest', 'transparent', 'straightforward', 'truthful', 'upfront',
            'candid', 'sincere', 'genuine', 'trust', 'integrity'
        ],
        'problem_solving': [
            'creative', 'solution', 'problem', 'resolve', 'overcome', 'challenge',
            'issue', 'fix', 'handle', 'manage', 'navigate'
        ],
        'dedication': [
            'dedicated', 'committed', 'hard work', 'effort', 'above beyond',
            'extra mile', 'went out', 'helpful', 'service'
        ]
    }
    
    NEGATIVE_SKILLS = {
        'unresponsive': [
            'unresponsive', 'slow', 'delayed', 'hard to reach', 'unavailable',
            'didn\'t return', 'never called', 'poor communication', 'ignored'
        ],
        'pushy': [
            'pushy', 'aggressive', 'pressure', 'forced', 'insistent',
            'pushy', 'demanding', 'rush', 'urgent without reason'
        ],
        'unprofessional': [
            'unprofessional', 'rude', 'disrespectful', 'inappropriate',
            'late', 'missed appointment', 'disorganized', 'messy'
        ],
        'inexperienced': [
            'inexperienced', 'new', 'didn\'t know', 'unfamiliar', 'lacked knowledge',
            'confused', 'uncertain', 'mistakes', 'errors'
        ],
        'dishonest': [
            'dishonest', 'lied', 'misled', 'false', 'deceptive', 'withheld',
            'hidden', 'misleading', 'untruthful'
        ]
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize skill extractor.
        
        Args:
            model_name: Sentence transformer model name
        """
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL
        
        self.model = SentenceTransformer(model_name)
        
        # Create embeddings for skill keywords
        self.positive_skill_embeddings = {}
        self.negative_skill_embeddings = {}
        
        for skill, keywords in self.POSITIVE_SKILLS.items():
            self.positive_skill_embeddings[skill] = self.model.encode(keywords)
        
        for skill, keywords in self.NEGATIVE_SKILLS.items():
            self.negative_skill_embeddings[skill] = self.model.encode(keywords)
    
    def extract_skills_from_comment(
        self, 
        comment: str, 
        threshold: float = 0.4
    ) -> Dict[str, float]:
        """
        Extract skill scores from a single comment using semantic similarity.
        
        Args:
            comment: Review comment text
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            Dictionary of skill scores {skill_name: score}
        """
        if pd.isna(comment) or not comment.strip():
            return {}
        
        # Encode comment
        comment_embedding = self.model.encode([comment])
        
        skill_scores = {}
        
        # Check positive skills
        for skill, skill_embeddings in self.positive_skill_embeddings.items():
            similarities = cosine_similarity(comment_embedding, skill_embeddings)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity >= threshold:
                skill_scores[f'skill_{skill}'] = float(max_similarity)
        
        # Check negative skills (with negative prefix)
        for skill, skill_embeddings in self.negative_skill_embeddings.items():
            similarities = cosine_similarity(comment_embedding, skill_embeddings)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity >= threshold:
                skill_scores[f'negative_{skill}'] = float(max_similarity)
        
        return skill_scores
    
    def aggregate_agent_skills(
        self, 
        reviews_df: pd.DataFrame,
        agent_id_col: str = 'advertiser_id',
        comment_col: str = 'review_comment',
        date_col: str = 'review_created_date',
        apply_recency_weight: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate skill scores for all agents.
        
        Args:
            reviews_df: DataFrame with reviews
            agent_id_col: Agent ID column name
            comment_col: Comment column name
            date_col: Review date column name
            apply_recency_weight: Whether to weight recent reviews higher
        
        Returns:
            DataFrame with agent_id and aggregated skill scores
        """
        # Calculate days since review for recency weighting
        if apply_recency_weight and date_col in reviews_df.columns:
            reviews_df[date_col] = pd.to_datetime(reviews_df[date_col])
            current_date = pd.Timestamp.now()
            reviews_df['days_since_review'] = (
                current_date - reviews_df[date_col]
            ).dt.days
            reviews_df['recency_weight'] = np.exp(
                -settings.REVIEW_RECENCY_DECAY * reviews_df['days_since_review']
            )
        else:
            reviews_df['recency_weight'] = 1.0
        
        # Extract skills from each review
        all_skills = []
        for idx, row in reviews_df.iterrows():
            comment = row[comment_col]
            skills = self.extract_skills_from_comment(comment)
            
            # Add metadata
            skills[agent_id_col] = row[agent_id_col]
            skills['recency_weight'] = row['recency_weight']
            all_skills.append(skills)
        
        skills_df = pd.DataFrame(all_skills)
        
        if skills_df.empty:
            return pd.DataFrame()
        
        # Get all skill columns (excluding metadata)
        skill_columns = [
            col for col in skills_df.columns 
            if col.startswith('skill_') or col.startswith('negative_')
        ]
        
        # Aggregate by agent with recency weighting
        aggregated = []
        
        for agent_id, group in skills_df.groupby(agent_id_col):
            agent_skills = {agent_id_col: agent_id}
            
            for skill_col in skill_columns:
                if skill_col in group.columns:
                    # Weighted average using recency weights
                    values = group[skill_col].fillna(0)
                    weights = group['recency_weight']
                    
                    if weights.sum() > 0:
                        weighted_avg = (values * weights).sum() / weights.sum()
                        agent_skills[skill_col] = weighted_avg
                    else:
                        agent_skills[skill_col] = 0.0
            
            aggregated.append(agent_skills)
        
        return pd.DataFrame(aggregated)


# Global skill extractor instance
skill_extractor = SkillExtractor()