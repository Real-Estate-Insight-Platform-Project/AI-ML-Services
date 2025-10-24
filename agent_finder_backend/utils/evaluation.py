"""
Evaluation module for agent recommendation system.

Provides metrics and analysis for assessing recommendation quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns


class RecommendationEvaluator:
    """Evaluate recommendation system performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def calculate_diversity_metrics(
        self,
        recommendations: List[Dict],
        features: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations: List of recommended agents
            features: Features to check for diversity
        
        Returns:
            Dictionary of diversity metrics
        """
        if not recommendations:
            return {}
        
        if features is None:
            features = ['experience_years', 'agent_base_city', 'office_name']
        
        df = pd.DataFrame(recommendations)
        
        metrics = {}
        
        # Feature diversity (unique values / total)
        for feature in features:
            if feature in df.columns:
                unique_ratio = df[feature].nunique() / len(df)
                metrics[f'diversity_{feature}'] = unique_ratio
        
        # Score distribution (variance)
        if 'matching_score' in df.columns:
            metrics['score_std'] = df['matching_score'].std()
            metrics['score_range'] = df['matching_score'].max() - df['matching_score'].min()
        
        # Price range diversity
        if 'min_price' in df.columns and 'max_price' in df.columns:
            price_ranges = []
            for _, row in df.iterrows():
                if pd.notna(row.get('min_price')) and pd.notna(row.get('max_price')):
                    price_ranges.append((row['min_price'], row['max_price']))
            
            if price_ranges:
                # Calculate overlap
                overlaps = 0
                total_pairs = len(price_ranges) * (len(price_ranges) - 1) / 2
                
                for i in range(len(price_ranges)):
                    for j in range(i + 1, len(price_ranges)):
                        r1, r2 = price_ranges[i], price_ranges[j]
                        # Check if ranges overlap
                        if not (r1[1] < r2[0] or r2[1] < r1[0]):
                            overlaps += 1
                
                metrics['price_range_overlap_ratio'] = overlaps / total_pairs if total_pairs > 0 else 0
        
        return metrics
    
    def calculate_coverage_metrics(
        self,
        recommendations: List[Dict],
        all_agents: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate coverage of recommendation space.
        
        Args:
            recommendations: List of recommended agents
            all_agents: All available agents
        
        Returns:
            Coverage metrics
        """
        if not recommendations:
            return {}
        
        rec_ids = [r['advertiser_id'] for r in recommendations]
        
        metrics = {
            'coverage_ratio': len(rec_ids) / len(all_agents),
            'unique_offices': pd.Series([r.get('office_name') for r in recommendations]).nunique(),
            'unique_cities': pd.Series([r.get('agent_base_city') for r in recommendations]).nunique()
        }
        
        return metrics
    
    def calculate_relevance_metrics(
        self,
        recommendations: List[Dict],
        ground_truth_scores: Dict[int, float] = None
    ) -> Dict[str, float]:
        """
        Calculate relevance metrics (if ground truth available).
        
        Args:
            recommendations: List of recommended agents
            ground_truth_scores: True relevance scores (agent_id -> score)
        
        Returns:
            Relevance metrics including NDCG
        """
        if not recommendations or not ground_truth_scores:
            return {}
        
        # Get predicted and true scores
        predicted_scores = []
        true_scores = []
        
        for rec in recommendations:
            agent_id = rec['advertiser_id']
            if agent_id in ground_truth_scores:
                predicted_scores.append(rec.get('matching_score', 0))
                true_scores.append(ground_truth_scores[agent_id])
        
        if not predicted_scores:
            return {}
        
        # Calculate NDCG
        try:
            # Reshape for sklearn
            true_scores_arr = np.array([true_scores])
            predicted_scores_arr = np.array([predicted_scores])
            
            ndcg = ndcg_score(true_scores_arr, predicted_scores_arr)
            
            metrics = {
                'ndcg': ndcg,
                'mean_predicted_score': np.mean(predicted_scores),
                'mean_true_score': np.mean(true_scores)
            }
        except:
            metrics = {}
        
        return metrics
    
    def analyze_feature_importance(
        self,
        agents_df: pd.DataFrame,
        score_column: str = 'matching_score'
    ) -> pd.DataFrame:
        """
        Analyze which features correlate with high scores.
        
        Args:
            agents_df: DataFrame with agents and scores
            score_column: Column with matching scores
        
        Returns:
            DataFrame with feature correlations
        """
        numeric_features = agents_df.select_dtypes(include=[np.number]).columns
        
        correlations = []
        
        for feature in numeric_features:
            if feature != score_column and agents_df[feature].notna().sum() > 0:
                corr = agents_df[feature].corr(agents_df[score_column])
                if not np.isnan(corr):
                    correlations.append({
                        'feature': feature,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        return corr_df
    
    def create_evaluation_report(
        self,
        recommendations: List[Dict],
        all_agents: pd.DataFrame,
        search_params: Dict,
        ground_truth_scores: Dict[int, float] = None
    ) -> Dict:
        """
        Create comprehensive evaluation report.
        
        Args:
            recommendations: List of recommended agents
            all_agents: All available agents
            search_params: Search parameters used
            ground_truth_scores: Optional ground truth scores
        
        Returns:
            Evaluation report dictionary
        """
        report = {
            'search_params': search_params,
            'num_recommendations': len(recommendations),
            'diversity_metrics': self.calculate_diversity_metrics(recommendations),
            'coverage_metrics': self.calculate_coverage_metrics(recommendations, all_agents)
        }
        
        if ground_truth_scores:
            report['relevance_metrics'] = self.calculate_relevance_metrics(
                recommendations,
                ground_truth_scores
            )
        
        # Score distribution
        if recommendations:
            scores = [r.get('matching_score', 0) for r in recommendations]
            report['score_distribution'] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        
        return report
    
    def compare_ranking_methods(
        self,
        agents_df: pd.DataFrame,
        ranking_methods: Dict[str, callable]
    ) -> pd.DataFrame:
        """
        Compare different ranking methods.
        
        Args:
            agents_df: DataFrame with agents
            ranking_methods: Dictionary of {method_name: ranking_function}
        
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for method_name, rank_func in ranking_methods.items():
            # Apply ranking
            ranked = rank_func(agents_df.copy())
            
            # Get top 10
            top_10 = ranked.head(10)
            
            results.append({
                'method': method_name,
                'avg_score': ranked['matching_score'].mean(),
                'top_10_avg': top_10['matching_score'].mean(),
                'score_std': ranked['matching_score'].std(),
                'diversity': top_10['office_name'].nunique() / 10
            })
        
        return pd.DataFrame(results)
    
    def plot_score_distribution(
        self,
        recommendations: List[Dict],
        save_path: str = None
    ):
        """
        Plot score distribution.
        
        Args:
            recommendations: List of recommended agents
            save_path: Path to save plot (optional)
        """
        if not recommendations:
            return
        
        scores = [r.get('matching_score', 0) for r in recommendations]
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Matching Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Matching Scores')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_feature_importance(
        self,
        feature_correlations: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ):
        """
        Plot feature importance based on correlations.
        
        Args:
            feature_correlations: DataFrame with feature correlations
            top_n: Number of top features to show
            save_path: Path to save plot (optional)
        """
        top_features = feature_correlations.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['correlation'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Correlation with Matching Score')
        plt.title(f'Top {top_n} Feature Correlations')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# Global evaluator instance
evaluator = RecommendationEvaluator()


def run_evaluation_example():
    """
    Example evaluation run.
    
    Run this to evaluate the recommendation system.
    """
    from utils.database import db_client
    from app.service import recommendation_service
    from app.models import AgentSearchRequest
    
    # Load data
    print("Loading data...")
    agents_df = db_client.get_all_agents()
    
    # Create example request
    request = AgentSearchRequest(
        user_type='buyer',
        state='CA',
        city='Los Angeles',
        min_price=500000,
        max_price=1000000,
        property_type='single_family',
        is_urgent=False,
        language='English',
        max_results=20
    )
    
    # Get recommendations
    print("Getting recommendations...")
    recommendations = recommendation_service.find_agents(request)
    
    # Convert to dict
    recs_dict = [rec.dict() for rec in recommendations]
    
    # Create evaluation report
    print("\nCreating evaluation report...")
    report = evaluator.create_evaluation_report(
        recs_dict,
        agents_df,
        request.dict()
    )
    
    # Print report
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\nNumber of recommendations: {report['num_recommendations']}")
    
    print("\nDiversity Metrics:")
    for key, value in report['diversity_metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nCoverage Metrics:")
    for key, value in report['coverage_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nScore Distribution:")
    for key, value in report['score_distribution'].items():
        print(f"  {key}: {value:.2f}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    agents_scored = pd.DataFrame([rec.dict() for rec in recommendations])
    feature_corr = evaluator.analyze_feature_importance(agents_scored)
    
    print("\nTop 10 Features Correlated with Matching Score:")
    print(feature_corr.head(10))
    
    return report, feature_corr


if __name__ == "__main__":
    run_evaluation_example()