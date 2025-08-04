import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class DataProcessor: 
    _data: pd.DataFrame

    def __init__(self, data_path: str = "data"):
        """
        Initialize with data directory path.

        Args:
            data_path (str): Path to the directory where data files are stored.
        """
        self._data_path = Path(data_path)
        self._data_path.mkdir(exist_ok=True)
        self._data = None
    
    def _create_sample_comment_data(self) -> pd.DataFrame:
        """Create sample customer comment data for the assignment."""
        np.random.seed(42)  # For reproducible results
        
        categories = ["category A", "category B", "category C", "category D"]
        sentiments = ["positive", "negative", "neutral"]
        
        # Sample comment texts with varying sentiments
        comment_samples = {
            "positive": [
                "I absolutely love this category! It exceeded my expectations.",
                "Great quality and fast shipping. Highly recommend!",
                "Excellent customer service and fantastic category quality.",
                "This category changed my life. Worth every penny!",
                "Amazing features and user-friendly interface. Five stars!",
                "Best purchase I've made this year. Outstanding quality!",
                "Impressed with the build quality and performance.",
                "Great value for money. Will definitely buy again.",
            ],
            "negative": [
                "Terrible category. Waste of money. Would not recommend.",
                "Poor quality materials. Broke after one week of use.",
                "Worst customer service ever. Still waiting for my refund.",
                "category doesn't match the description. Very disappointed.",
                "Overpriced for what you get. There are better alternatives.",
                "Shipping took forever and category arrived damaged.",
                "Difficult to use and confusing instructions.",
                "Not worth the money. Cheaply made and unreliable.",
            ],
            "neutral": [
                "It's an okay category. Does what it's supposed to do.",
                "Average quality. Nothing special but gets the job done.",
                "The category works fine. No major complaints.",
                "Decent category for the price. Could be better.",
                "It's alright. Met my basic expectations.",
                "Good enough for what I needed. Standard quality.",
                "The category is functional but not impressive.",
                "Fair price for an average category. Nothing outstanding.",
            ]
        }
        
        data = []
        for i in range(500):  # Generate 500 comment entries
            sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])  # Slightly more positive
            category = np.random.choice(categories)
            comment = np.random.choice(comment_samples[sentiment])
            rating = {
                "positive": np.random.randint(4, 5),
                "negative": np.random.randint(1, 2),
                "neutral": 3
            }[sentiment]
            
            data.append({
                "title": f"Sample Comment {i + 1}",
                "category": category,
                "comments": comment,
                "rating": rating,
                "date": pd.date_range(start="2023-01-01", end="2024-01-01", periods=500)[i]
            })
        
        return pd.DataFrame(data)

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset.
        """
        summary = {
            "total_records": len(df),
            "categories": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            "rating_distribution": df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
            "average_rating": df['rating'].mean().__round__(3) if 'rating' in df.columns else None,
            "date_range": {
                "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                "end": df['date'].max().isoformat() if 'date' in df.columns else None
            } if 'date' in df.columns else None,
            "comment_length_stats": pd.Series(df['comments'].str.len()).describe().to_dict() if 'comments' in df.columns else None,
            }
        
        return summary

    def load_customer_comments(self, filename: str = "customer_comment.csv") -> pd.DataFrame:
        # Trying to read the CSV file with utf-8 encoding first, then falling back to latin-1 if it fails.
        try:
            df = pd.read_csv(self._data_path / filename, encoding='utf-8' )
        except UnicodeDecodeError:
            df = pd.read_csv(self._data_path / filename, encoding='latin-1')
        return df
    
    def _clean_comments_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the comments data.

        This includes:
        - Removing duplicates
        - Handling missing values
        - Cleaning text data
        - Ensuring numeric columns are properly formatted
        - Converting date columns to datetime format if present

        Args:
            df (pd.DataFrame): The DataFrame containing comments data.

        Returns:
            pd.DataFrame: Cleaned DataFrame ready for analysis.
        """

        # Standardizing column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.rename(columns={'review_title': 'title'}, inplace=True)

        # Converting date columns to datetime format
        df['date'] = pd.to_datetime(df['date'], format='on %d %B %Y')

        # Removes duplicates when the same customer comments twice in the same category with the same title
        df.drop_duplicates(inplace=True,subset=['customer_name','category','title'])

        # Parse rating column to int64
        df['rating'] = df['rating'].str.extract('(\d+)').astype("int64")

        # Handling missing values in Useful column
        df['useful'] = df['useful'].replace('', np.nan)
        df['useful'] = df["useful"].fillna('0 people found this helpful yet')

        return df   