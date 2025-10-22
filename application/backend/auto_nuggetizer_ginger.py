"""Class for detecting information nuggets in documents given a query."""

import ast
import re
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from google import genai
from google.genai import types
from backend.utils import load_api_key, load_sample_pdf_files


class NuggetDetector(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect_nuggets(self, query: str, document: bytes,) -> List[str]:
        """Detects information nuggets in a Document given a query.

        Args:
            query: Query to answer.
            document: Document to detect information nuggets in.

        Returns:
            List of information nuggets extracted from the document.
        """
        raise NotImplementedError


class GPTNuggetDetector(NuggetDetector):
    def __init__(
        self, api_key: str, gemini_model: str = 'gemini-2.5-flash'
    ) -> None:
        """Instantiates a nugget detector using a Google Gemini model.

        Args:
            api_key: Google GenAI API key.
            gemini_model (optional): Gemini model version.
        """  # noqa
        self._gemini_client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

    def detect_nuggets(
        self, query: str, document: bytes, thinking_budget: int = 24576, num_retries: int = 3
    ) -> List[str]:
        """Detects information nuggets in a document given a query.

        Args:
            query: Query to answer.
            document: Document to detect information nuggets in.
            thinking_budget (optional): Thinking budget for Gemini model.
            num_retries (optional): Number of retries for Gemini API call.

        Returns:
            List of information nuggets extracted from the document.
        """
        contents = [
            types.Part.from_bytes(
                data=document,
                mime_type='application/pdf',
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Question: {}".format(query)),
                ],
            )
        ]
        for _ in range(num_retries):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=[
                            "Given a query and a document, annotate information nuggets that contain the key information answering the query. Copy the text of the document and put each annotated information nugget "
                            "between <IN> and </IN>. Do NOT modify the content of the passage. Do NOT add additional symbols, spaces, etc. to the text.",
                        ],
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_budget,
                            include_thoughts=True
                        )
                    ),
                )
                return re.findall("<IN>(.*?)</IN>", response.text.strip().replace('\n', ' '))
            except Exception as e:
                print(f"Error calling Gemini API: {e}. Retrying...")
        return []


if __name__ == "__main__":
    # Example usage
    nugget_detector = GPTNuggetDetector(api_key=load_api_key())

    query = "What are the recommendations regarding antibiotic prophylaxis for transrectal or transperineal prostate biopsy?"
    document = load_sample_pdf_files()['EAU-Guideline_Urological_Infections_Paper.pdf']

    information_nuggets = nugget_detector.detect_nuggets(query, document)

    for information_nugget in information_nuggets:
        print(information_nugget)
