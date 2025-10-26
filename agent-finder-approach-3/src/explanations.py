"""
Comprehensive explanation system for agent recommendations.

Generates:
- Match explanations based on user preferences
- Theme-based praise snippets from reviews
- Confidence intervals and data quality indicators
- Coverage fit descriptions for user requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import re

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """Generates comprehensive explanations for agent recommendations."""
    
    def __init__(self, theme_results: Dict, feature_names: List[str]):
        """
        Initialize explanation generator.
        
        Args:
            theme_results: Results from theme mining with labels and keywords
            feature_names: List of feature names from model
        """
        self.theme_results = theme_results
        self.feature_names = feature_names
        self.theme_labels = theme_results.get('theme_labels', {})
        self.theme_keywords = theme_results.get('theme_keywords', {})
    
    def generate_agent_explanations(self,
                                  recommended_agents: pd.DataFrame,
                                  reviews_df: pd.DataFrame,
                                  user_preferences: Dict[str, float],
                                  fused_weights: np.ndarray,
                                  metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive explanations for each recommended agent.
        
        Args:
            recommended_agents: DataFrame of recommended agents with scores
            reviews_df: Full review data for extracting snippets
            user_preferences: User preference settings
            fused_weights: Final personalized weights used for scoring
            metadata: Scoring metadata from recommendation engine
            
        Returns:
            List of explanation dictionaries for each agent
        """
        logger.info(f"Generating explanations for {len(recommended_agents)} recommendations...")
        
        explanations = []
        
        for _, agent in recommended_agents.iterrows():
            agent_id = agent['advertiser_id']
            
            # Get agent reviews
            agent_reviews = reviews_df[reviews_df['advertiser_id'] == agent_id].copy()
            
            # Generate comprehensive explanation
            explanation = {
                'agent_id': agent_id,
                'agent_name': agent.get('full_name', 'Unknown'),
                'rank': agent.get('rank', 0),
                'utility_score': agent.get('utility_score', 0.0),
                'preference_matches': self._explain_preference_matches(agent, user_preferences),
                'theme_strengths': self._explain_theme_strengths(agent, agent_reviews),
                'confidence_metrics': self._explain_confidence(agent),
                'coverage_fit': self._explain_coverage_fit(agent, metadata['user_filters']),
                'review_highlights': self._extract_review_highlights(agent_reviews),
                'data_quality': self._assess_data_quality(agent, agent_reviews),
                'why_recommended': self._generate_why_recommended(agent, user_preferences, fused_weights)
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _explain_preference_matches(self, 
                                  agent: pd.Series,
                                  user_preferences: Dict[str, float]) -> List[Dict[str, str]]:
        """Explain how agent matches user preferences."""
        
        matches = []
        
        # Sub-score preferences
        preference_mapping = {
            'responsiveness': ('skill_responsiveness', 'Responsiveness'),
            'negotiation': ('skill_negotiation', 'Negotiation Skills'),
            'professionalism': ('skill_professionalism', 'Professionalism'),
            'market_expertise': ('skill_market_expertise', 'Market Expertise')
        }
        
        for pref_name, pref_value in user_preferences.items():
            if pref_name in preference_mapping:
                skill_col, display_name = preference_mapping[pref_name]
                
                if skill_col in agent.index:
                    agent_score = agent[skill_col]
                    
                    # Calculate percentile (rough estimate)
                    if agent_score > 0.5:
                        percentile = "top 25%"
                    elif agent_score > 0.0:
                        percentile = "above average"
                    elif agent_score > -0.5:
                        percentile = "average"
                    else:
                        percentile = "below average"
                    
                    # Determine match quality
                    if pref_value > 0.7:  # High preference
                        if agent_score > 0.3:
                            match_quality = "excellent"
                        elif agent_score > 0.0:
                            match_quality = "good"
                        else:
                            match_quality = "moderate"
                    elif pref_value > 0.3:  # Medium preference
                        if agent_score > 0.0:
                            match_quality = "good"
                        else:
                            match_quality = "adequate"
                    else:  # Low preference
                        match_quality = "adequate"
                    
                    matches.append({
                        'aspect': display_name,
                        'user_preference': self._preference_to_text(pref_value),
                        'agent_performance': percentile,
                        'match_quality': match_quality,
                        'description': f"{display_name}: {match_quality} match ({percentile})"
                    })
        
        return matches
    
    def _explain_theme_strengths(self, 
                               agent: pd.Series,
                               agent_reviews: pd.DataFrame) -> List[Dict[str, Any]]:
        """Explain agent's theme strengths with examples."""
        
        strengths = []
        
        # Get theme columns and scores
        theme_cols = [col for col in agent.index if col.startswith('theme_')]
        
        for theme_col in theme_cols:
            theme_strength = agent[theme_col]
            
            if theme_strength > 0.1:  # Only show meaningful themes
                # Extract theme name
                theme_name = theme_col.replace('theme_', '').replace('_', ' ').title()
                
                # Find reviews that match this theme
                theme_examples = self._find_theme_examples(agent_reviews, theme_name)
                
                strengths.append({
                    'theme_name': theme_name,
                    'strength_score': theme_strength,
                    'strength_level': self._score_to_level(theme_strength),
                    'examples': theme_examples,
                    'description': f"Strong in {theme_name.lower()} (strength: {theme_strength:.2f})"
                })
        
        # Sort by strength
        strengths.sort(key=lambda x: x['strength_score'], reverse=True)
        
        return strengths[:5]  # Top 5 themes
    
    def _explain_confidence(self, agent: pd.Series) -> Dict[str, Any]:
        """Explain confidence metrics for the agent."""
        
        review_count = agent.get('review_count_actual', 0)
        wilson_lb = agent.get('wilson_lower_bound', 0.0)
        confidence_score = agent.get('confidence_score', 0.5)
        recency_score = agent.get('recency_score', 0.0)
        
        # Determine confidence level
        if confidence_score > 0.8:
            confidence_level = "very high"
        elif confidence_score > 0.6:
            confidence_level = "high"
        elif confidence_score > 0.4:
            confidence_level = "moderate"
        else:
            confidence_level = "lower"
        
        # Generate confidence explanation
        if review_count >= 10:
            volume_desc = f"substantial review history ({review_count} reviews)"
        elif review_count >= 3:
            volume_desc = f"moderate review history ({review_count} reviews)"
        elif review_count >= 1:
            volume_desc = f"limited review history ({review_count} reviews)"
        else:
            volume_desc = "no reviews available"
        
        # Recency description
        if recency_score > 0.8:
            recency_desc = "very recent client feedback"
        elif recency_score > 0.5:
            recency_desc = "recent client feedback"
        elif recency_score > 0.2:
            recency_desc = "some recent feedback"
        else:
            recency_desc = "mostly older feedback"
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'review_count': review_count,
            'wilson_lower_bound': wilson_lb,
            'volume_description': volume_desc,
            'recency_description': recency_desc,
            'overall_description': f"{confidence_level.title()} confidence based on {volume_desc} and {recency_desc}"
        }
    
    def _explain_coverage_fit(self, 
                            agent: pd.Series,
                            user_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how agent fits user's coverage requirements."""
        
        fit_aspects = []
        
        # Location fit
        if 'state' in user_filters:
            agent_state = agent.get('state', '')
            if agent_state == user_filters['state']:
                fit_aspects.append(f"Located in {agent_state}")
        
        if 'city' in user_filters:
            service_areas = agent.get('marketing_area_cities', '')
            if user_filters['city'].lower() in service_areas.lower():
                fit_aspects.append(f"Serves {user_filters['city']} area")
        
        # Transaction type fit
        if 'transaction_type' in user_filters:
            transaction_type = user_filters['transaction_type'].lower()
            if transaction_type == 'buying' and agent.get('is_buyer_agent', False):
                fit_aspects.append("Specializes in buyer representation")
            elif transaction_type == 'selling' and agent.get('is_seller_agent', False):
                fit_aspects.append("Specializes in seller representation")
        
        # Language fit
        if 'language' in user_filters and user_filters['language'] != 'English':
            languages = agent.get('languages', 'English')
            if user_filters['language'].lower() in languages.lower():
                fit_aspects.append(f"Speaks {user_filters['language']}")
        
        # Experience fit
        experience = agent.get('experience_years', 0)
        if experience >= 10:
            fit_aspects.append(f"Highly experienced ({int(experience)} years)")
        elif experience >= 5:
            fit_aspects.append(f"Experienced ({int(experience)} years)")
        
        # Activity level
        active_listings = agent.get('active_listings_count', 0)
        recent_sales = agent.get('recently_sold_count', 0)
        
        if active_listings > 10 or recent_sales > 20:
            fit_aspects.append("Very active in current market")
        elif active_listings > 0 or recent_sales > 5:
            fit_aspects.append("Active in current market")
        
        return {
            'fit_aspects': fit_aspects,
            'coverage_description': "; ".join(fit_aspects) if fit_aspects else "Basic coverage match"
        }
    
    def _extract_review_highlights(self, agent_reviews: pd.DataFrame) -> List[Dict[str, str]]:
        """Extract highlighted review snippets."""
        
        if len(agent_reviews) == 0:
            return []
        
        highlights = []
        
        # Get recent positive reviews with comments
        positive_reviews = agent_reviews[
            (agent_reviews['review_rating'] >= 4.0) & 
            (agent_reviews['review_comment'].notna()) &
            (agent_reviews['review_comment'] != '')
        ].copy()
        
        # Sort by recency and rating
        if len(positive_reviews) > 0:
            positive_reviews = positive_reviews.sort_values(
                ['recency_weight', 'review_rating'], 
                ascending=[False, False]
            )
            
            # Extract top snippets
            for _, review in positive_reviews.head(3).iterrows():
                comment = review['review_comment']
                
                # Extract key sentences (simple approach)
                snippet = self._extract_key_snippet(comment)
                
                if snippet:
                    highlights.append({
                        'snippet': snippet,
                        'rating': review['review_rating'],
                        'date': review.get('review_created_date', ''),
                        'reviewer_role': review.get('reviewer_role', 'CLIENT')
                    })
        
        return highlights
    
    def _assess_data_quality(self, 
                           agent: pd.Series,
                           agent_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality and completeness of agent data."""
        
        quality_score = 0.0
        quality_aspects = []
        
        # Review volume and recency
        review_count = agent.get('review_count_actual', 0)
        if review_count >= 10:
            quality_score += 0.4
            quality_aspects.append("Substantial review history")
        elif review_count >= 3:
            quality_score += 0.2
            quality_aspects.append("Adequate review history")
        
        # Profile completeness
        profile_fields = ['agent_bio', 'specializations', 'languages', 'designations']
        complete_fields = sum(1 for field in profile_fields if not pd.isna(agent.get(field, np.nan)))
        
        profile_completeness = complete_fields / len(profile_fields)
        quality_score += profile_completeness * 0.3
        
        if profile_completeness > 0.75:
            quality_aspects.append("Complete profile information")
        elif profile_completeness > 0.5:
            quality_aspects.append("Good profile information")
        
        # Recent activity
        recency = agent.get('recency_score', 0.0)
        if recency > 0.5:
            quality_score += 0.3
            quality_aspects.append("Recent client activity")
        
        # Determine overall quality
        if quality_score > 0.8:
            quality_level = "excellent"
        elif quality_score > 0.6:
            quality_level = "good"
        elif quality_score > 0.4:
            quality_level = "adequate"
        else:
            quality_level = "limited"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'quality_aspects': quality_aspects,
            'description': f"{quality_level.title()} data quality: " + "; ".join(quality_aspects)
        }
    
    def _generate_why_recommended(self,
                                agent: pd.Series,
                                user_preferences: Dict[str, float],
                                fused_weights: np.ndarray) -> str:
        """Generate overall explanation for why agent was recommended."""
        
        reasons = []
        
        # Check top preference matches
        high_prefs = {k: v for k, v in user_preferences.items() if v > 0.7}
        
        for pref_name in high_prefs:
            if pref_name == 'responsiveness' and agent.get('skill_responsiveness', 0) > 0.2:
                reasons.append("excellent responsiveness track record")
            elif pref_name == 'negotiation' and agent.get('skill_negotiation', 0) > 0.2:
                reasons.append("strong negotiation skills")
            elif pref_name == 'professionalism' and agent.get('skill_professionalism', 0) > 0.2:
                reasons.append("high professionalism ratings")
            elif pref_name == 'market_expertise' and agent.get('skill_market_expertise', 0) > 0.2:
                reasons.append("solid market expertise")
        
        # Add confidence factors
        if agent.get('wilson_lower_bound', 0) > 0.7:
            reasons.append("consistently positive client feedback")
        
        if agent.get('experience_years', 0) >= 10:
            reasons.append("extensive experience")
        
        # Combine reasons
        if len(reasons) >= 3:
            why_text = f"Recommended for {reasons[0]}, {reasons[1]}, and {reasons[2]}"
        elif len(reasons) == 2:
            why_text = f"Recommended for {reasons[0]} and {reasons[1]}"
        elif len(reasons) == 1:
            why_text = f"Recommended for {reasons[0]}"
        else:
            why_text = "Recommended based on overall profile match to your preferences"
        
        return why_text
    
    # Helper methods
    
    def _preference_to_text(self, value: float) -> str:
        """Convert preference value to human-readable text."""
        if value > 0.8:
            return "very important"
        elif value > 0.6:
            return "important"
        elif value > 0.4:
            return "somewhat important"
        else:
            return "less important"
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to level description."""
        if score > 0.7:
            return "very strong"
        elif score > 0.4:
            return "strong"
        elif score > 0.2:
            return "moderate"
        else:
            return "some"
    
    def _find_theme_examples(self, reviews: pd.DataFrame, theme_name: str) -> List[str]:
        """Find example review snippets for a theme."""
        # Simple keyword-based matching for now
        # In production, could use the cluster assignments
        
        theme_keywords = {
            'communication': ['responsive', 'communication', 'contact', 'call', 'email'],
            'professionalism': ['professional', 'knowledgeable', 'expert', 'reliable'],
            'negotiation': ['negotiation', 'deal', 'price', 'offer'],
            'market knowledge': ['market', 'area', 'neighborhood', 'local', 'pricing']
        }
        
        keywords = theme_keywords.get(theme_name.lower(), [])
        examples = []
        
        if len(keywords) > 0 and len(reviews) > 0:
            for _, review in reviews.iterrows():
                comment = str(review.get('review_comment', ''))
                if any(keyword.lower() in comment.lower() for keyword in keywords):
                    snippet = self._extract_key_snippet(comment)
                    if snippet and snippet not in examples:
                        examples.append(snippet)
                        if len(examples) >= 2:
                            break
        
        return examples
    
    def _extract_key_snippet(self, comment: str, max_length: int = 150) -> str:
        """Extract a key snippet from a review comment."""
        if not comment or len(comment.strip()) == 0:
            return ""
        
        # Clean and split into sentences
        comment = comment.strip()
        sentences = re.split(r'[.!?]+', comment)
        
        # Find the most informative sentence (longest with good content)
        best_sentence = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > len(best_sentence) and len(sentence) >= 20:
                # Skip sentences that are too short or just punctuation
                if any(word.isalpha() for word in sentence.split()):
                    best_sentence = sentence
        
        # Truncate if too long
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length].rsplit(' ', 1)[0] + "..."
        
        return best_sentence if best_sentence else comment[:max_length]