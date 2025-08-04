"""
LangChain Tools for Customer Comments Analysis

This module contains all the tools used by the DataAnalysisAgent for analyzing
customer feedback data.
"""
from ..prompts.templates import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts
import logging
import json
import pandas as pd
from typing import Dict, Any, Type, Optional
from ..data import DataProcessor
from ..llm.providers import BaseLLMProvider

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for tool inputs
class DataStatsInput(BaseModel):
    """Input schema for data statistics tool."""
    metric: str = Field(description="The single statistic to calculate. Must be one of: 'count', 'avg_length', 'word_frequency', 'rating_distribution', 'category_breakdown', 'date_stats'")
    column: str = Field(default="comments", description="Column name to analyze for metrics like 'avg_length' or 'word_frequency'. Default is 'comments'.")


class SentimentAggregationInput(BaseModel):
    """Input schema for sentiment aggregation tool."""
    aggregation_type: str = Field(description="Type of aggregation: 'summary', 'distribution', 'trends'")


class InsightGenerationInput(BaseModel):
    """Input schema for insight generation tool."""
    insight_type: str = Field(description="Type of insights: 'recommendations', 'trends', 'priorities'")


class DataStatsTool(BaseTool):
    """Tool for calculating statistical metrics on customer comments data."""
    
    name: str = "calculate_data_stats"
    description: str = """Calculate statistical metrics for customer comments dataset.
    Available metrics: count, avg_length, word_frequency, rating_distribution, category_breakdown, date_stats.
    Use metric parameter to specify which statistic to calculate.
    Use column parameter to specify which column to analyze (default: comments)."""
    args_schema: Type[BaseModel] = DataStatsInput
    data_processor: DataProcessor 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_data = None
    
    def set_data(self, data):
        """Set the current dataset for analysis."""
        self._current_data = processor.py data
    
    def _run(self, metric: str, column: str = "comments", **kwargs) -> Dict[str, Any]:
        """
        Calculate statistical metrics for customer comments data.
        
        Args:
            metric: The type of metric to calculate
            column: Column to analyze
            
        Returns:
            Dict with statistical results
        """
        if self._current_data is None:
            return {"error": "No data available for analysis"}
        
        if metric.startswith('{') and metric.endswith('}'):
            try:
                # Parse the JSON string to extract metric and column
                metric_data = json.loads(metric)
                metric = metric_data.get("metric", metric)
                column = metric_data.get("column", column)
            except json.JSONDecodeError:
                pass
        logger.debug(f"Final metric: '{metric}', column: '{column}'")

        try:
            if metric == "count":
                return {
                    "count": len(self._current_data), 
                    "metric": metric,
                    "message": f"Dataset contains {len(self._current_data)} customer comments"
                }
            elif metric == "avg_length":
                avg_len = self._current_data[column].str.len().mean()
                return {
                    "avg_length": round(avg_len, 2),
                    "median_length": self._current_data[column].str.len().median(),
                    "metric": metric,
                    "message": f"Average comment length is {avg_len:.2f} characters"
                }
            elif metric == "word_frequency":
                # Simple word frequency analysis
                from collections import Counter
                all_text = " ".join(self._current_data[column].astype(str))
                words = all_text.lower().split()
                word_freq = Counter(words).most_common(20)
                                
                # Get top 20 words
                return {
                    "word_frequency": dict(word_freq),
                    "metric": metric,
                    "message": "Top 20 most frequent words in customer comments"
                }

            elif metric == "rating_distribution":
                # Simple rating distribution analysis
                if "rating" not in self._current_data.columns:
                    return {"error": "No rating data available for analysis"}
                rating_counts = self._current_data["rating"].value_counts().to_dict()
                return {
                    "rating_distribution": rating_counts,
                    "metric": metric,
                    "message": "Distribution of customer ratings"
                }
            
            elif metric == "category_breakdown":
                # Category breakdown analysis
                if "category" not in self._current_data.columns:
                    return {"error": "No category data available for analysis"}
                category_counts = self._current_data["category"].value_counts().to_dict()
                return {
                    "category_breakdown": category_counts,
                    "metric": metric,
                    "message": "Breakdown of customer comments by category"
                }
            elif metric == "date_stats":
                # Date range and most active day information
                if "date" not in self._current_data.columns:
                    return {"error": "No date data available for analysis"}
                date_counts = self._current_data["date"].value_counts()
                return {
                    "start_date": self._current_data["date"].min().isoformat(),
                    "end_date": self._current_data["date"].max().isoformat(),
                    "most_active_day": {"Date": date_counts.idxmax(),
                                        "Count": date_counts.max()},
                    "metric": metric,
                }
            else:
                return {"error": f"Unknown metric: {metric}"}
        except Exception as e:
            return {"error": f"Error calculating {metric}: {str(e)}"}
    
    async def _arun(self, metric: str, column: str = "comments") -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(metric, column)


class SentimentAggregationTool(BaseTool):
    """Tool for aggregating sentiment analysis results."""
    
    name: str = "aggregate_sentiment" 
    description: str = """Aggregate sentiment analysis results across all comments.
    Use this tool to summarize sentiment patterns and distributions."""
    args_schema: Type[BaseModel] = SentimentAggregationInput
    llm_provider: BaseLLMProvider
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sentiment_results = None
    
    def set_sentiment_data(self, sentiment_data):
        """Set sentiment analysis results."""
        self._sentiment_results = sentiment_data
    
    def _run(self, aggregation_type: str) -> Dict[str, Any]:
        """
        TODO: Implement sentiment aggregation logic.
        
        The intern should implement:
        1. Process individual sentiment scores/labels
        2. Calculate overall sentiment distribution  
        3. Identify sentiment trends or patterns
        4. Generate summary statistics
        
        Args:
            aggregation_type: How to aggregate the data
            
        Returns:
            Dict with aggregated sentiment insights
        """
        if self._sentiment_results is None:
            return {"error": "No sentiment data available"}
        
        # TODO: Intern must implement sentiment aggregation
        if aggregation_type == "summary":
            formatted_data = json.dumps(self._sentiment_results, indent=2, default=str)
            formatted_prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(
                feedback=formatted_data
            )

            # Calling the LLM with the formatted prompt and getting infos about the response
            llm_response = self.llm_provider.generate(formatted_prompt)
            llm_content = llm_response.content
            llm_response_time = llm_response.response_time
            llm_response_tokens_used = llm_response.tokens_used

            try:
                # Parsing the JSON response from the LLM
                if "```json" in llm_content:
                    json_start = llm_content.find("```json") + len("```json")
                    json_end = llm_content.find("```", json_start)
                    json_str = llm_content[json_start:json_end].strip()
                else:
                    start = llm_content.find("{")
                    end = llm_content.rfind("}") + 1
                    json_str = llm_content[start:end].strip() if start != -1 and end > start else llm_content.strip()

                parsed_response = json.loads(json_str)
                return {
                    "aggregation_type": aggregation_type,
                    "prompt_used": formatted_prompt,
                    "parsed_response": parsed_response,
                    "model_used": llm_response.model,
                    "response_time": llm_response_time,
                    "tokens_used": llm_response_tokens_used,
                }

            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response from LLM",
                        "raw_response": llm_response}

            
        elif aggregation_type == "distribution":

            # Verifying that sentiment results are in a list format
            if not isinstance(self._sentiment_results, list):
                return {"error": "Distribution requires a list of sentiment results."}

            # Getting sentiment distribution from the list of JSON results
            sentiment_list = [result['sentiment'] for result in self._sentiment_results]
            sentiment_counts = pd.Series(sentiment_list).value_counts().to_dict()

            return {
                "sentiment_distribution": sentiment_counts,
                "aggregation_type": aggregation_type,
                "message": "Distribution of sentiments across all comments"
            }
        
        elif aggregation_type == "trends":
            # Verifying that sentiment results are in a list format
            if not isinstance(self._sentiment_results, list):
                return {"error": "Trends require a list of sentiment results."}
            
            # Getting sentiment trends from the list of JSON results
            try:
                df = pd.DataFrame(self._sentiment_results)
                if 'date' not in df.columns or 'sentiment' not in df.columns:
                    return {"error": "Each sentiment entry must include 'date' and 'sentiment'"}

                ## Creating a weekly trend analysis ##

                # Creating a new column for week since start date
                df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

                # Grouping by week and sentiment, then calculating mean sentiment score by week
                weekly_trends = df.groupby('week')['sentiment'].value_counts().unstack(fill_value=0)
                weekly_trends = weekly_trends.to_dict(orient='index')


                return {
                    "aggregation_type": aggregation_type,
                    "weekly_sentiment_trends": weekly_trends
                }

            except Exception as e:
                return {"error": f"Error processing sentiment data: {str(e)}"}
        else:
            return {"error": f"Unknown aggregation type: {aggregation_type}"}
    
    async def _arun(self, aggregation_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(aggregation_type)


class InsightGenerationTool(BaseTool):
    """Tool for generating business insights from analysis results."""
    
    name: str = "generate_insights"
    description: str = """Generate actionable business insights from customer feedback analysis.
    Use this tool to create recommendations and identify key business opportunities."""
    args_schema: Type[BaseModel] = InsightGenerationInput
    llm_provider: BaseLLMProvider

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._analysis_results = None
    
    def set_analysis_results(self, results):
        """Set the analysis results for insight generation."""
        self._analysis_results = results
    
    def _run(self,insight_type: str) -> Dict[str, Any]:
        """
        TODO: Implement insight generation logic.
        
        The intern should implement:
        1. Analyze patterns across sentiment, topics, and statistics
        2. Identify key business opportunities or issues
        3. Generate actionable recommendations
        4. Prioritize findings by impact/importance
        
        Args:
            insight_type: Type of insights to focus on
            
        Returns:
            Dict with structured business insights
        """
        if self._analysis_results is None:
            return {"error": "No analysis results available"}
        
        # TODO: Intern must implement insight generation
        if insight_type in ["recommendations", "trends", "priorities"]:
            prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
            data_summary=self._analysis_results.get("data_summary", {}),
            sentiment_results=self._analysis_results.get("sentiment_analysis", {}),
            topic_results=self._analysis_results.get("topic_extraction", {})
        )
            
            llm_response = self.llm_provider.generate(prompt)
            llm_content = llm_response.content
            llm_response_time = llm_response.response_time
            llm_response_tokens_used = llm_response.tokens_used
            return{
                "insight_type": insight_type,
                "prompt_used": prompt,
                "model_used": llm_response.model,
                "business_report": llm_content,
                "response_time": llm_response_time,
                "tokens_used": llm_response_tokens_used
            }

        else:
            return {"error": f"Unknown insight type: {insight_type}"}
    
    async def _arun(self, insight_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(insight_type)
    
