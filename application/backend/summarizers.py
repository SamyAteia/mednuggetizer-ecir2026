"""Text summarizers."""

from abc import ABC, abstractmethod
from typing import List

from google import genai
from google.genai import types
from backend.utils import load_api_key


class Summarizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def summarize_info_nuggets(self, text: str, max_length: int) -> str:
        """Summarize text.

        Args:
            text: Text to summarize.
            max_length: Maximum number of tokens in the summary.

        Returns:
            Summary of text.
        """
        raise NotImplementedError


class GPTSummarizerIndividualNuggets(Summarizer):
    def __init__(
        self, api_key: str, gemini_model: str = 'gemini-2.5-flash'
    ) -> None:
        """Instantiates a summarizer based on Google's Gemini model.

        Args:
            api_key: Google GenAI API key.
            gemini_model (optional): Gemini model version.
        """  # noqa
        self._gemini_client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

    def _summarize_single_info_nugget(
            self, info_nugget: str, query: str, thinking_budget: int) -> str:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Information Nugget: {}, Query: {}".format(info_nugget, query)),
                ],
            )
        ]
        response = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=[
                    "Create a short description of the information nugget given a query. Do not change the information itself included in the information nugget. Do not add information not mentioned in the information nugget.",
                ],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True
                )
            ),
        )
        return response.text

    def summarize_info_nuggets(
        self, info_nuggets: List[str], query: str, thinking_budget: int = 24576,
            max_retries: int = 3
    ) -> List[str]:
        """Summarizes a list of information nuggets.

        Args:
            info_nuggets: List of information nuggets to summarize.
            query: Query to answer.
            thinking_budget (optional): Thinking budget for Gemini model.
            max_retries (optional): Maximum number of retries for API calls.

        Returns:
            Summarized information nuggets.
        """
        summarized_info_nuggets = []
        for info_nugget in info_nuggets:
            for _ in range(max_retries):
                try:
                    summarized_info_nuggets.append(
                        self._summarize_single_info_nugget(
                            info_nugget=info_nugget,
                            query=query,
                            thinking_budget=thinking_budget
                        )
                    )
                    break
                except Exception as e:
                    if _ == max_retries - 1:
                        raise e
        return summarized_info_nuggets


class GPTSummarizerAllNuggets(Summarizer):
    def __init__(
        self, api_key: str, gemini_model: str = 'gemini-2.5-flash'
    ) -> None:
        """Instantiates a summarizer based on Google's Gemini model.

        Args:
            api_key: Google GenAI API key.
            gemini_model (optional): Gemini model version.
        """  # noqa
        self._gemini_client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

    def summarize_info_nuggets(
        self, info_nuggets: List[str], query: str, thinking_budget: int = 24576,
            max_retries: int = 3
    ) -> str:
        """Summarizes a list of information nuggets.

        Args:
            info_nuggets: List of information nuggets to summarize.
            query: Query to answer.
            thinking_budget (optional): Thinking budget for Gemini model.
            max_retries (optional): Maximum number of retries for API calls.

        Returns:
            Summary of information nuggets.
        """
        info_nuggets = '\n'.join(info_nuggets)
        for _ in range(max_retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=f"""Given a user query and multiple overlapping nuggets derived from that query, create a single unified nugget that:
                                        * Accurately reflects the content of all nuggets.
                                        * Keeps every factual detail mentioned.
                                        * Avoids repetition and irrelevant phrasing.
                                        * Does not add new facts or infer beyond the nuggets.
                                        
                                        Query:
                                        {query}
                                        
                                        Nuggets:
                                        {info_nuggets}
                                        
                                        Output:
                                        Concise merged nugget (max 1–2 sentences)."""
                            ),

                        ],
                    )
                ]
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_budget,
                            include_thoughts=True
                        )
                    ),
                )
                return response.text
            except Exception as e:
                if _ == max_retries - 1:
                    raise e
        return ''


class SummarizerHeadings(Summarizer):
    def __init__(
            self, api_key: str, gemini_model: str = 'gemini-2.5-flash'
    ) -> None:
        """Instantiates a summarizer based on Google's Gemini model.

        Args:
            api_key: Google GenAI API key.
            gemini_model (optional): Gemini model version.
        """  # noqa
        self._gemini_client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

    def summarize_info_nuggets(
            self, info_nuggets: List[str],
            thinking_budget: int = 24576,
            max_retries: int = 3
    ) -> str:
        """Creates a heading as asummary for a list of information nuggets.

        Args:
            info_nuggets: List of information nuggets to summarize.
            thinking_budget (optional): Thinking budget for Gemini model.
            max_retries (optional): Maximum number of retries for API calls.

        Returns:
            Heading summarizing the list of information nuggets.
        """
        info_nuggets = '\n'.join(info_nuggets)
        for _ in range(max_retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=f"""
                                      Given the following list of information nuggets, generate a short, concise heading (no more than 10 words) that captures the main idea or common theme of the list.
                                        
                                      Information Nuggets:
                                      {info_nuggets}
                                      """
                            ),

                        ],
                    )
                ]
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_budget,
                            include_thoughts=True
                        )
                    ),
                )
                return response.text
            except Exception as e:
                if _ == max_retries - 1:
                    raise e
        return ''


if __name__ == '__main__':
    query = ''
    information_nuggets = [
        "Table 19 - Suggested antimicrobial prophylaxis regimens before urological procedures Procedure Prophylaxis recommended Antimicrobial ... Transrectal prostate biopsy Yes 1. Targeted prophylaxis on the basis of a rectal swab or stool culture. 2. Augmented prophylaxis with two or more different antibiotic classes.a 3. Alternative antibiotics • Fosfomycin trometamolb (eg, 3 g before and 3 g 24-48 h after biopsy) • Cephalosporin (eg, ceftriaxone 1 g i.m.; cefixime 400 mg p.o. for 3 d start-ing 24 h before biopsy) • Aminoglycoside (eg, gentamicin 3mg/kg i.v.; amikacin 15mg/kg i.m.)",
        "b The indication for fosfomycin trometamol for prostate biopsy has been withdrawn in Germany as the manufacturers did not submit the necessary pharmacokinetic data in support of this indication. Urologists are advised to check their local guidance in relation to the use of fosfomycin trometamol for prostate biopsy."
    ]

    # Example usage individual nuggets
    summarizer = GPTSummarizerIndividualNuggets(api_key=load_api_key())

    summarized_information_nuggets = summarizer.summarize_info_nuggets(
        info_nuggets=information_nuggets,
        query=query
    )

    for summarized_nugget in summarized_information_nuggets:
        print(summarized_nugget)
        print(30*'#')


    # Example usage nugget list
    summarizer = GPTSummarizerAllNuggets(api_key=load_api_key())

    summarized_information_nuggets = summarizer.summarize_info_nuggets(
        info_nuggets=summarized_information_nuggets,
        query=query
    )

    print(summarized_information_nuggets)

    heading_summarizer = SummarizerHeadings(api_key=load_api_key())
    heading = heading_summarizer.summarize_info_nuggets(
        info_nuggets=information_nuggets
    )
    print(heading)
