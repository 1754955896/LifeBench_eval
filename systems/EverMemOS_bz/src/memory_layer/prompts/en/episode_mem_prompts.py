DEFAULT_CUSTOM_INSTRUCTIONS = """
When generating episodic memories, please follow these principles:
1. Each episode should be a complete, independent story or event
2. Preserve all important information, including names, times, places, emotions, etc.
3. Use descriptive language to describe the episode, not dialogue format
4. Highlight key information and emotional changes
5. Ensure the episode content is conducive to subsequent retrieval
"""

EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following content into an episodic memory.

Content start time: {conversation_start_time}
Content:
{conversation}

Please generate a structured episodic memory and return only a JSON object containing the following three fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "summary": "A brief and clear summary of the main points and outcomes (50-100 words). This should be more concise and easier to understand than the detailed content, focusing on key takeaways and essential information.",
    "content": "A detailed factual record of the content in third-person narrative. It must include all important information: what happened, when it happened, who was involved, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account of what actually happened, focusing on observable actions and direct statements rather than interpretive conclusions. Use the provided content start time as the base time for this episode."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The summary should be concise, clear, and capture the essence more effectively than the detailed content.
3. The content must include all important information from the content.
4. Convert any dialogue format into a narrative description.
5. Maintain chronological order and causal relationships.
6. Use third-person unless explicitly first-person.
7. Include specific details that aid keyword search, especially concrete activities, places, and objects.
8. For time references, use the dual format: "relative time (absolute date)" to support different question types.
9. When describing decisions or actions, naturally include the reasoning or motivation behind them.
10. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
11. The content must include all important information from the input content, and its language must match the input content language.

Return only the JSON object, do not add any other text:
"""

GROUP_EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Please generate a structured episodic memory and return only a JSON object containing the following three fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "summary": "A brief and clear summary of the main points and outcomes from the conversation (50-100 words). This should be more concise and easier to understand than the detailed content, focusing on key takeaways and essential information.",
    "content": "A detailed factual record of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account of what actually happened, focusing on observable actions and direct statements rather than interpretive conclusions. Use the provided conversation start time as the base time for this episode."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The summary should be concise, clear, and capture the essence more effectively than the detailed content.
3. The content must include all important information from the conversation.
4. Convert the dialogue format into a narrative description.
5. Maintain chronological order and causal relationships.
6. Use third-person unless explicitly first-person.
7. Include specific details that aid keyword search, especially concrete activities, places, and objects.
8. For time references, use the dual format: "relative time (absolute date)" to support different question types.
9. When describing decisions or actions, naturally include the reasoning or motivation behind them.
10. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
11. The content must include all important information from the conversation, and the language of the content must be chinese.

Return only the JSON object, do not add any other text:
"""
