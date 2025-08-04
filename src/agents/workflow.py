"""
Workflow nodes for the DataAnalysisAgent.

This module contains all the individual workflow steps (nodes) that are
executed as part of the LangGraph workflow.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from .state import AgentState, update_state_with_error
from .tools import DataStatsTool, SentimentAggregationTool, InsightGenerationTool
from ..data import DataProcessor
from ..prompts.templates import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Container class for all workflow node implementations."""
    
    def __init__(self, data_processor: DataProcessor, llm_providers: Dict[str, Any], llm_to_use: str):
        """Initialize workflow nodes with required dependencies."""
        self.data_processor = data_processor
        self.llm_providers = llm_providers
        self.llm_to_use = llm_to_use
        
        # Initialize tools
        self.data_stats_tool = DataStatsTool(data_processor)
        self.sentiment_aggregation_tool = SentimentAggregationTool(llm_provider=llm_providers[llm_to_use])
        self.insight_generation_tool = InsightGenerationTool(llm_provider=llm_providers[llm_to_use])

    def load_data(self, state: AgentState) -> AgentState:
        """Load and preprocess customer comments data."""
        try:
            df = self.data_processor.load_customer_comments()
            state["data"] = df
            state["current_step"] = "data_loaded"
            state["analysis_results"]["data_summary"] = self.data_processor.get_data_summary(df)
            state["messages"] = []
            state["tool_calls_made"] = []
            
            # Set data in tools
            self.data_stats_tool.set_data(df)
            
            logger.info(f"Loaded {len(df)} comments records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            state = update_state_with_error(state, str(e), "load_data")
        
        return state
    
    def agent_with_tools(self, state: AgentState) -> AgentState:
        """
        Main agent logic with tool calling capabilities.
        """
        try:
            messages = state["messages"]
            current_step = state.get("current_step", "starting")
            
            if current_step == "data_loaded":
                # Create the prompt for the agent
                prompt = """You are a data analysis expert. Your task is to analyze a dataset of customer comments.

You must proceed in the following order:
1.  First, get the total number of comments by calling the `calculate_data_stats` tool with the metric 'count'.
2.  Next, get the distribution of ratings by calling the `calculate_data_stats` tool with the metric 'rating_distribution'.
3.  After gathering these statistics, you must perform sentiment analysis on the entire dataset.
4.  Finally, after all analysis is complete, call the `generate_insights` tool with the insight_type 'recommendations' to create a final business summary.

The data is already loaded. Begin by calling the `calculate_data_stats` tool as described in step 1."""

                # Append the initial human message to the conversation
                message = HumanMessage(content=prompt)
                messages.append(message)

                # Get the LLM provider based on the configured one
                llm = self.llm_providers.get(self.llm_to_use)
                if not llm:
                    raise ValueError(f"LLM provider '{self.llm_to_use}' not found.")

                # Setting tools for the agent
                tools = [
                    self.data_stats_tool,
                    self.sentiment_aggregation_tool,
                    self.insight_generation_tool
                ]
                llm_with_tools = llm.bind_tools(tools)
                
                # Run the agent with the prompt
                agent_response = llm_with_tools.invoke(messages)

                # Append the agent response to messages
                messages.append(agent_response)

                # Update the state
                state["current_step"] = "analyzing"
                state["messages"] = messages
            
        except Exception as e:
            logger.error(f"Error in agent with tools: {e}")
            state = update_state_with_error(state, str(e), "agent_with_tools")
        
        return state
    
    def generate_final_summary(self, state: AgentState) -> AgentState:
        """
        Generate final summary of all analysis results.
        """
        summary_prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
        data_summary=state["analysis_results"]["data_summary"],
        sentiment_results=state["analysis_results"]["sentiment_analysis"],
        topic_results=state["analysis_results"]["topic_extraction"]
    )
        
        try:

            # Get the current llm provider
            llm = self.llm_providers.get(self.llm_to_use)
            if not llm:
                raise ValueError(f"LLM provider '{self.llm_to_use}' not found.")

            # Call the llm with the final prompt
            final_summary = llm.invoke(summary_prompt).content

            # Store the genereated report in the state
            state["analysis_results"]["final_summary"] = {
                "status": "completed",
                "report": final_summary
            }
            state["current_step"] = "completed"
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            state = update_state_with_error(state, str(e), "generate_final_summary")
        
        return state
    
    def should_continue_with_tools(self, state: AgentState) -> str:
        """
        Determine if we should continue with tool calling or end the workflow.
        """
        messages = state["messages"]
        
        MAX_TOOL_CALLS = 5

        if not messages or len(state.get("tool_calls_made", [])) >= MAX_TOOL_CALLS:
            # End the loop if there are no messages or max tool calls reached
            return "end"
        
        last_message = messages[-1]
        
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "end"
