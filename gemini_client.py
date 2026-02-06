"""
Gemini AI Client for Chat Search
Uses Gemini 1.5 Flash to summarize search results and answer questions.
"""

import os
import json
from typing import List, Dict, Optional

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiClient:
    """Client for Gemini AI API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.model = None
        self._initialized = False

    def _initialize(self):
        """Initialize the Gemini client."""
        if self._initialized:
            return True

        if not HAS_GEMINI:
            print("google-generativeai not installed")
            return False

        if not self.api_key:
            print("GEMINI_API_KEY not set")
            return False

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self._initialized = True
            print("Gemini client initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            return False

    def answer_from_context(self, query: str, search_results: List[Dict],
                           max_results: int = 5) -> Dict:
        """
        Generate an answer based on search results.

        Args:
            query: User's question
            search_results: List of search results with context
            max_results: Max results to include in context

        Returns:
            Dict with 'answer', 'sources', and 'success'
        """
        if not self._initialize():
            return {
                'success': False,
                'error': 'Gemini not available',
                'answer': None
            }

        # Build context from search results
        context_parts = []
        sources = []

        for i, result in enumerate(search_results[:max_results]):
            # Handle different result formats
            if 'message' in result:
                # search_with_context format
                msg = result['message']
                context_parts.append(f"""
--- תוצאה {i+1} (ציון: {result.get('score', 0):.2f}) ---
מאת: {msg.get('from_name', 'לא ידוע')}
תאריך: {msg.get('date', 'לא ידוע')}
הודעה: {msg.get('text', '')}
""")
                sources.append({
                    'from_name': msg.get('from_name'),
                    'date': msg.get('date'),
                    'message_id': result.get('message_id')
                })

                # Add context if available
                if result.get('context_before'):
                    context_parts.append("הקשר לפני:")
                    for ctx in result['context_before']:
                        context_parts.append(f"  [{ctx.get('from_name', '?')}] {ctx.get('text_plain', '')[:100]}")

                if result.get('context_after'):
                    context_parts.append("הקשר אחרי:")
                    for ctx in result['context_after']:
                        context_parts.append(f"  [{ctx.get('from_name', '?')}] {ctx.get('text_plain', '')[:100]}")

            elif 'chunk_text' in result:
                # hybrid_search format
                context_parts.append(f"""
--- תוצאה {i+1} (ציון: {result.get('score', 0):.2f}) ---
{result.get('chunk_text', '')}
""")
                sources.append({
                    'message_id': result.get('message_id'),
                    'score': result.get('score')
                })

        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""אתה עוזר שמנתח שיחות מקבוצת טלגרם ועונה על שאלות.

השאלה: {query}

להלן תוצאות חיפוש רלוונטיות מהשיחות:

{context}

הנחיות:
1. ענה בעברית
2. תן תשובה קצרה וממוקדת (1-3 משפטים)
3. אם המידע לא ברור או לא קיים בתוצאות, אמור "לא מצאתי מידע ברור"
4. ציין את המקור (שם השולח והתאריך) אם רלוונטי
5. אל תמציא מידע שלא מופיע בתוצאות

התשובה:"""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()

            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'query': query,
                'results_used': len(context_parts)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': None
            }

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self._initialize()


# Singleton instance
_gemini_client = None


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def ai_search(query: str, limit: int = 5) -> Dict:
    """
    Perform AI-powered search: hybrid search + Gemini summarization.

    Args:
        query: Search query
        limit: Max results to use

    Returns:
        Dict with answer and metadata
    """
    from hybrid_search import get_hybrid_search

    # Get hybrid search results
    hs = get_hybrid_search()
    results = hs.search_with_context(query, limit=limit)

    if not results:
        return {
            'success': False,
            'error': 'No search results found',
            'answer': 'לא נמצאו תוצאות לחיפוש זה',
            'query': query
        }

    # Get AI answer
    client = get_gemini_client()
    response = client.answer_from_context(query, results, max_results=limit)

    # Add raw results for transparency
    response['search_results'] = results

    return response


# CLI for testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_client.py 'search query'")
        print("\nChecking Gemini availability...")
        client = get_gemini_client()
        if client.is_available():
            print("Gemini is available!")
        else:
            print("Gemini is NOT available. Set GEMINI_API_KEY environment variable.")
        sys.exit(0)

    query = ' '.join(sys.argv[1:])
    print(f"\n=== AI Search: {query} ===\n")

    result = ai_search(query)

    if result['success']:
        print(f"Answer: {result['answer']}")
        print(f"\nSources: {len(result.get('sources', []))} results used")
    else:
        print(f"Error: {result.get('error')}")
