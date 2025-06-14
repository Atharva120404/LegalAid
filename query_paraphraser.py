import os
import logging
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryParaphraser:
    """
    Paraphrases user queries into legally structured language using Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini model.
        
        Args:
            api_key (str): API key for Gemini. Uses environment variable GEMINI_API_KEY if not provided.
            model_name (str): Gemini model version to use.
        """
        self.model = None
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.error("Gemini API key not provided or found in environment.")
            return
        
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.exception("Error initializing Gemini model")

    def paraphrase_query(self, query: str) -> str:
        """
        Paraphrases a query into legal language.

        Args:
            query (str): User's original layman query.
        
        Returns:
            str: Paraphrased legal query.
        """
        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query provided.")
            return query

        if not self.model:
            logger.warning("Gemini model not initialized. Returning original query.")
            return query

        prompt = (
            f'Paraphrase the following query into a clear, legally structured question. '
            f'Preserve the original meaning and include appropriate legal terms:\n"{query}"\n'
            f'Return only the paraphrased query without extra explanation.'
        )

        try:
            response = self.model.generate_content(prompt)
            paraphrased = response.text.strip()
            if paraphrased:
                logger.info(f"Paraphrased query: {paraphrased}")
                return paraphrased
            else:
                logger.warning("Empty response from Gemini. Returning original query.")
                return query
        except Exception as e:
            logger.exception("Failed to generate paraphrased content.")
            return query

if __name__ == "__main__":
    paraphraser = QueryParaphraser()
    test_query = "What are legal drink age india."
    result = paraphraser.paraphrase_query(test_query)
    print(f"\nOriginal   : {test_query}")
    print(f"Paraphrased: {result}")
