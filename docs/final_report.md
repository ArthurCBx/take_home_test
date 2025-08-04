# Results Comparison
After several hours of running and debugging the code, I exhausted my token quota for both the OpenAI and Gemini free tiers. Due to this limitation, I was unable to generate a full set of results to compare them. However, with the final implementations it is possible to get the data stats with the LLM and generate a final analysis. The sentiment analysis function was not fully tested because an error with the data was occurring in the `SentimentAggregationTool`.

# Challenges Encountered
As I didn't have much experience with langchain and its usages understanding the initial code structure was hard. But after some time researching and experimenting with AI assistants I became more familiar with it and gained a comprehension of the project.

# Areas of Strength
The data manipulation in processor.py were the most straightforward for me as i had already worked with pandas in my studies. Consequently, preparing the data for the LLM didn't take as much time as implementing the agent's logic in the tools.py, workflow.py and core.py files.

# Comments 
* I had to delete the emojis in main.py to resolve console encoding issues.
* I have removed the API keys from .env file as they have reached their usage limits.
* The prompts given in templates.py were not changed.
* The file test_llm had a test changed to support response times equal to 0.
