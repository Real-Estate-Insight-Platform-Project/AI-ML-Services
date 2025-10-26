import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from config.settings import settings


class SentimentAnalyzer:
    """Sentiment analysis for review comments using review-focused models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Model name (default from settings)
        """
        if model_name is None:
            model_name = settings.SENTIMENT_MODEL
        
        # Use VADER for fast sentiment analysis (tuned for social media/reviews)
        self.vader = SentimentIntensityAnalyzer()
        
        # Optional: Use transformer model for better accuracy
        self.use_transformer = True
        if self.use_transformer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
            except Exception as e:
                print(f"Could not load transformer model: {e}")
                self.use_transformer = False
    
    def analyze_comment(self, comment: str, rating: float = None) -> Tuple[str, float]:
        """
        Analyze sentiment of a review, falling back to rating if no comment.
        
        Args:
            comment: Review comment text
            rating: Numeric rating (used if comment is empty)
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        # NEW: Check if comment is missing
        if pd.isna(comment) or not comment.strip():
            if rating is not None:
                return self.get_sentiment_from_rating(rating)
            return 'neutral', 0.0
        
        # EXISTING CODE CONTINUES...
        vader_scores = self.vader.polarity_scores(comment)
        compound = vader_scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
            confidence = min(compound, 1.0)
        elif compound <= -0.05:
            sentiment = 'negative'
            confidence = min(abs(compound), 1.0)
        else:
            sentiment = 'neutral'
            confidence = 1.0 - abs(compound)
        
        return sentiment, confidence
    
    def analyze_with_transformer(self, comment: str) -> Tuple[str, float]:
        """
        Analyze sentiment using transformer model (more accurate but slower).
        
        Args:
            comment: Review comment text
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not self.use_transformer:
            return self.analyze_comment(comment)
        
        try:
            inputs = self.tokenizer(
                comment,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Map to sentiment (5-star rating model)
            # 1-2 stars: negative, 3 stars: neutral, 4-5 stars: positive
            probs = predictions[0].cpu().numpy()
            
            negative_prob = probs[0] + probs[1]  # 1-2 stars
            neutral_prob = probs[2]  # 3 stars
            positive_prob = probs[3] + probs[4]  # 4-5 stars
            
            max_prob = max(negative_prob, neutral_prob, positive_prob)
            
            if positive_prob == max_prob:
                return 'positive', positive_prob
            elif negative_prob == max_prob:
                return 'negative', negative_prob
            else:
                return 'neutral', neutral_prob
                
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return self.analyze_comment(comment)
    
    def get_sentiment_from_rating(self, rating: float) -> str:
        """
        Infer sentiment from rating when comment is missing.
        
        Args:
            rating: Review rating (typically 1-5)
        
        Returns:
            Sentiment label
        """
        if pd.isna(rating):
            return 'neutral'
        
        if rating >= 4.0:
            return 'positive'
        elif rating <= 2.5:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_reviews_batch(
        self, 
        reviews_df: pd.DataFrame,
        comment_column: str = 'review_comment',
        rating_column: str = 'review_rating'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all reviews in dataframe.
        
        Args:
            reviews_df: DataFrame with reviews
            comment_column: Name of comment column
            rating_column: Name of rating column
        
        Returns:
            DataFrame with added sentiment columns
        """
        sentiments = []
        confidences = []
        
        for idx, row in reviews_df.iterrows():
            comment = row[comment_column]
            rating = row[rating_column]
            
            # If comment exists, analyze it
            if pd.notna(comment) and str(comment).strip():
                sentiment, confidence = self.analyze_comment(str(comment))
            # Otherwise, infer from rating
            elif pd.notna(rating):
                sentiment = self.get_sentiment_from_rating(rating)
                confidence = 0.8  # Lower confidence for inferred sentiment
            else:
                sentiment = 'neutral'
                confidence = 0.0
            
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        reviews_df['sentiment'] = sentiments
        reviews_df['sentiment_confidence'] = confidences
        
        return reviews_df
    
    def get_sentiment_from_rating(self, rating: float) -> Tuple[str, float]:
        """
        Derive sentiment from rating when no review comment exists.
        
        Args:
            rating: Numeric rating (1-5 scale)
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if pd.isna(rating) or rating == 0:
            return 'neutral', 0.0
        
        # 4-5 stars = positive
        if rating >= 4.0:
            sentiment = 'positive'
            confidence = min(rating / 5.0, 1.0)
        # 1-2 stars = negative
        elif rating <= 2.0:
            sentiment = 'negative'
            confidence = min((5.0 - rating) / 5.0, 1.0)
        # 3 stars = neutral
        else:
            sentiment = 'neutral'
            confidence = 0.7
        
        return sentiment, confidence

# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()