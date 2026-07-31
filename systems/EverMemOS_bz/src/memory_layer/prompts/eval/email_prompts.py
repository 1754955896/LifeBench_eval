"""
Email-related prompts for LLM processing.

This module contains all prompts used for email analysis and extraction.
"""

EMAIL_INSIGHTS_EXTRACTION_PROMPT = """
Please analyze the following email content and extract three key pieces of information:

Original Email Subject: {original_subject}

Email Body Content:
{clean_text}

Please return the results in the following JSON format:
{{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "summary": "Core content summary of the email, similar to a paper's abstract, concisely summarizing the main information and purpose of the email",
    "subject": "Core theme of the email, similar to a paper's title, more concise than summary, highlighting the most important information"
}}

Requirements:
1. keywords: Extract 5 keywords that best represent the email type and content, should reflect the semantic type of the email
2. summary: Summarize the core content and purpose of the email in 2-3 sentences, should be concise but complete
3. subject: Summarize the most core theme of the email in one sentence, should be more concise than summary, like a title
4. Please ensure the returned content is in valid JSON format
5. IMPORTANT: The language of the extracted content should match the language of the original email content. If the email is in Chinese, return Chinese; if in English, return English; if in other languages, return in that language.

Please analyze the language of the original email and respond in the same language.
"""

EMAIL_SHOULD_CREATE_MEMCELL_PROMPT = """
Please analyze the following email content and determine if it should be stored as a memory cell.

Email Subject: {subject}
Sender Address: {sender_address}
Email Body Content:
{clean_text}

Please return the result in the following JSON format:
{{
    "should_create": true/false,
    "reasoning": "Brief explanation for the decision"
}}

Criteria for creating a memory cell:

1. ALWAYS CREATE (high priority) if the email contains:
   - Business communications with meaningful content
   - Meeting requests, scheduling, or calendar events
   - Project updates, decisions, or work-related discussions
   - Personal communications with significant emotional or factual content
   - Action items, tasks, or follow-up requirements
   - Important announcements or notifications
   - App Store/platform review results or status updates
   - User replies with substantial content (not just "thanks" or "ok")
   - Account setup, access grants, or system onboarding
   - Invoice, payment, or transaction confirmations

2. SPECIAL CONSIDERATIONS for system emails:
   - Apple/App Store notifications: CREATE if about app reviews, submissions, or account status
   - Platform notifications (GitHub, Slack, etc.): CREATE if about important state changes
   - SaaS service emails: CREATE if about account changes, billing, or feature updates
   - Automated replies with user content: CREATE if the user wrote meaningful content in their reply

3. DO NOT CREATE if the email is:
   - Pure marketing newsletters with no personal relevance
   - Obvious spam or junk mail
   - Generic promotional content without specific relevance
   - Empty or very short acknowledgments ("Thanks", "OK", "Got it")
   - Unsubscribe confirmations
   - Pure advertising with no business relationship

4. REPLY EMAIL ANALYSIS:
   - For emails starting with "Re:" or "回复:", analyze the NEW content the user wrote
   - Ignore quoted/forwarded content, focus on the fresh user input
   - CREATE if user added meaningful thoughts, questions, or responses
   - DO NOT CREATE if it's just forwarding without commentary

5. SENDER AUTHORITY CONSIDERATION:
   - Emails from official platforms (apple.com, github.com, etc.) are more likely important
   - Emails from business domains related to user's work should be preserved
   - Personal emails from known contacts deserve careful content analysis

IMPORTANT: Focus on whether this email contains information that would be useful to remember for future reference, personal history, or business continuity. When in doubt about business-related content, lean towards CREATING the memory cell.

Analyze the content carefully and make a decision:
"""
