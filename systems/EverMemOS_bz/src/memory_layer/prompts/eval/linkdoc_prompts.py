# Prompts for LinkDoc processing
LINKDOC_ANALYSIS_PROMPT = """
You are an expert document analyzer. Please analyze the following document and extract summary, subject, and keywords.

IMPORTANT: All output must be in the SAME LANGUAGE as the input document content.

Document Title: {title}
Document Content:
{content}

Please provide your analysis in the following JSON format:
{{
    "summary": "A comprehensive but concise summary (2-3 sentences) in the SAME LANGUAGE as the document",
    "subject": "A clear, descriptive subject/topic that captures the main theme of the document",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

Requirements:
- **Summary**: Should help someone quickly understand what this document is about and why it might be relevant. Include main topics, key decisions, important information, and overall purpose. Be concise but comprehensive.
- **Subject**: A clear, descriptive subject that captures the main theme/topic of the document. This should be better than the original title if the title is unclear or generic.
- **Keywords**: Extract 3-5 most important keywords that represent the core concepts, topics, or themes in the document. The extracted keywords must be in the original document.

LANGUAGE RULE: If document is in Chinese, all outputs (summary, subject, keywords) must be in Chinese. If document is in English, all must be in English.

Return only the JSON object, no additional text or formatting.
"""

CONTENT_MERGE_PROMPT = """
You are an expert document analyzer. Please merge multiple partial analysis results from different sections of a long document into one comprehensive analysis.

Document Title: {title}
Partial Analysis Results:
{partial_results}

Please create a comprehensive analysis that merges all the partial results into:
{{
    "summary": "A comprehensive summary that integrates all key information, removes redundancy, and maintains logical flow (3-5 sentences)",
    "subject": "A unified subject/topic that best represents the entire document",
    "keywords": ["merged list of most important keywords (3-5 total)"]
}}

Requirements:
1. **Summary**: Integrate all key information from partial summaries, remove redundant information while preserving important details, maintain logical flow and coherence
2. **Subject**: Choose or synthesize the most representative subject that captures the main theme of the entire document
3. **Keywords**: Merge and deduplicate keywords from all sections, keeping only the 3-5 most important ones that represent the entire document. The extracted keywords must be in the original document.

LANGUAGE RULE: Use the same language as the input partial results.

Return only the JSON object, no additional text or formatting.
"""

DOCUMENT_FILTER_PROMPT = """
You are a document relevance classifier for a professional knowledge management system. Please analyze the following document preview and determine if it should be processed for knowledge extraction.

Document Title: {title}
Document Preview (first {preview_length} characters):
{content_preview}

Please classify this document based on its relevance to professional/work contexts:

INCLUDE (should process):
- Meeting notes, work discussions, project documentation
- Technical documentation, tutorials, how-to guides
- Business plans, reports, analysis documents
- Research notes, academic papers, professional articles
- Product specifications, requirements documents
- Team communications about work topics
- Planning documents, strategy papers
- Learning materials, training content

EXCLUDE (should skip):
- Personal diary entries, private thoughts
- Shopping lists, personal todo items
- Entertainment content (movies, games, fiction)
- Pure data files (logs, dumps, raw data)
- Spam, advertisements, promotional content
- Very short fragments with no meaningful content
- Duplicate or template content
- Social media posts, casual conversations

Return your decision in the following JSON format:
{{
    "should_process": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of the decision",
    "category": "work/personal/technical/entertainment/data/other"
}}

Be conservative - when in doubt, lean towards including the document rather than excluding it.

Return only the JSON object, no additional text or formatting.
"""
