"""RAG-based Q&A flow and CLI entrypoint.

This module defines the `RagFlow` conversation flow that validates a user
question against a main topic about Calls for Proposals, performs retrieval
augmented generation via `RagCrew`, optionally validates web sources, and then
produces a final answer via `AnswerWriter`.

The flow is designed to be documented with Sphinx and follows Google-style
docstrings for classes and functions.
"""

import json
from typing import Any, Dict, List, Optional

import opik
import yaml
from crewai import CrewOutput, LLM
from crewai.flow import Flow, listen, or_, router, start
from opik.integrations.crewai import track_crewai
from pydantic import BaseModel, Field, ValidationError

from progetto.crews.answer_writer.answer_writer import AnswerWriter
from progetto.crews.rag_crew.rag_crew import RagCrew

opik.configure(use_local=True)

# Global Settings
WHITE_LIST_PATH = "utils/white_list.yaml"
MAIN_TOPIC = "Calls for Proposals or Calls for Action"
ALLOWED_TOPICS = ["Culture", "ICT", "Green Economy"]
# MAIN_TOPIC = "python"
# ALLOWED_TOPICS = ["python"]

class RagCrewResponseItem(BaseModel):
    """Single result item returned by the RAG pipeline.

    Attributes:
        origin: Origin label (e.g., ``"RAG"`` or ``"WEB"``).
        title: Title of the source.
        source: URL or identifier of the source.
        content: Content excerpt from the source.
        similarity: Similarity score from the retriever.
        is_trusted: Whether the source is on the whitelist.
    """
    origin: str = Field(description="The origin of the source, e.g., 'RAG' or 'WEB'")
    title: str = Field(description="The title of the source")
    source: str = Field(description="The URL or identifier of the source")
    content: str = Field(description="The content of the source used as a knowledge")
    similarity: float = Field(description="A similarity value comes from RAG")
    is_trusted: bool = Field(description="Whether the source is trusted based on white list")

class RagCrewResponseList(BaseModel):
    """Validation model for a list of `RagCrewResponseItem` objects.

    Attributes:
        results: List of validated retrieval results.
    """
    results: List[RagCrewResponseItem] = Field(description="List of RAG Crew search results")

class FlowState(BaseModel):
    """State carried across the flow steps.

    Attributes:
        topic: The sector/topic used to scope valid questions.
        is_topic_related: Whether the question is related to ``MAIN_TOPIC``.
        user_question: The user-provided question.
        rag_crew_raw_response: Raw JSON/string response returned by the RAG crew.
        rag_crew_validated_response: Parsed and validated RAG results.
        web_validation_results: Web results annotated with trust flags.
        answer_writer_raw_response: Raw response returned by the AnswerWriter crew.
        rag_crew: Crew handle for the executed RAG crew.
        answer_writer: Crew handle for the executed AnswerWriter crew.
    """
    topic: str = Field(default="", description="The topic for scoping questions")
    is_topic_related: bool = Field(default=False, description="Whether the question is about the topic")
    user_question: str = Field(default="", description="The question provided by the user")
    rag_crew_raw_response: str = Field(default="", description="The raw response from the RagCrew")
    rag_crew_validated_response: Optional[RagCrewResponseList] = Field(default=None, description="Validated search results")
    web_validation_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Results after web source validation")
    answer_writer_raw_response: str = Field(default="", description="The raw response from the AnswerWriter")
    rag_crew: Optional[CrewOutput] = Field(default=None, description="To hold the RagCrew instance")
    answer_writer: Optional[CrewOutput] = Field(default=None, description="To hold the AnswerWriter instance")

class RagFlow(Flow[FlowState]):
    """Conversation flow to validate and answer topic-specific questions.

    Steps:
        - ``start``: Initialize the flow.
        - ``ask_question``: Prompt the user for a sector and question.
        - ``process_topic``: Validate topic relevance using an LLM.
        - ``validate_question``: Route to retry or proceed.
        - ``process_rag``: Run the RAG crew.
        - ``process_web_site_validation``: Mark trusted web sources.
        - ``process_answer_writing``: Generate a final answer.
        - ``finalize``: Return the aggregated payload.
    """

    @start()
    def start(self):
        """The first step in the flow."""
        return
    
    @listen(or_(start, "Retry question"))
    def ask_question(self):
        """Prompt the user to provide a sector and a question relevant to the main topic."""
        print(f"Provide the {MAIN_TOPIC} sector: ")
        self.state.topic = input("sector: ")
        print("Provide a question: ")
        self.state.user_question = input("question: ")

    @listen(ask_question)
    def process_topic(self):
        """Evaluate topic relevance using an LLM and update state.

        Side Effects:
            Updates ``self.state.is_topic_related`` based on the model output.

        Returns:
            dict: A payload summarizing the validation inputs and outcome.
        """
        print("Validating the question and the sector.")

        # inline Pydantic class for LLM response validation
        class TopicValidation(BaseModel):
            """Response schema produced by the LLM for topic validation."""
            is_topic_related: bool = Field(description="Whether the question and topic are relevant")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=TopicValidation)

        # Create the messages for the outline
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant designed to evaluate whether a topic and a "
                    "user question are relevant to "
                    f"{MAIN_TOPIC}, and provide an answer in a JSON format with a "
                    "boolean field called is_topic_related."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate if: \n"
                    f"- the question \"{self.state.user_question}\" and the topic "
                    f"\"{self.state.topic}\" are relevant to {MAIN_TOPIC}. \n"
                    f"- the topic \"{self.state.topic}\" is related to one of these "
                    f"topics: {', '.join(ALLOWED_TOPICS)}.\n"
                    "Italian translations are accepted.\n"
                    "Respond strictly with a JSON string containing only the JSON."
                ),
            },
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        print("Raw response: ", response)
        # Parse the JSON response
        llm_response = json.loads(response)
        self.state.is_topic_related = llm_response.get("is_topic_related", False)

        payload = {
            "main_topic": MAIN_TOPIC,
            "allowed_topics": ALLOWED_TOPICS,
            "user_topic": self.state.topic,
            "user_question": self.state.user_question,
            "is_topic_related": self.state.is_topic_related
            }
        return payload

    @router(process_topic)
    def validate_question(self):
        """Route based on validation outcome.

        Returns:
            str: ``"Question valid"`` if relevant, otherwise ``"Retry question"``.
        """
        if self.state.is_topic_related:
            print("The question is relevant to the topic")
            return "Question valid"
        print(
            (
                f"The question '{self.state.user_question}' is not about the topic "
                f"'{self.state.topic}'. Please ask a relevant question."
            )
        )
        return "Retry question"

    @listen(or_("Question valid", "JSON not valid"))
    def process_rag(self):
        """Execute the RAG crew after a valid question is confirmed.

        Returns:
            dict: Payload including crew handle and raw response.
        """
        rag_crew = (
            RagCrew()
            .crew()
            .kickoff(inputs={
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "main_topic": MAIN_TOPIC
                })
        )

        self.state.rag_crew = rag_crew
        self.state.rag_crew_raw_response = rag_crew.raw
        
        payload = {
            "rag_crew": rag_crew,
            "user_question": self.state.user_question,
            "topic": self.state.topic,
            "is_about_topic": self.state.is_topic_related,
            "response_rag_crew": rag_crew.raw
        }
        
        return payload

    @router(process_rag)
    def validate_rag_crew_results(self):
        """Validate RAG results using Pydantic models.

        Returns:
            str: ``"JSON valid"`` on success, otherwise ``"JSON not valid"``.
        """
        try:
            # Parse the JSON string first, then validate
            raw_response = self.state.rag_crew_raw_response
            if isinstance(raw_response, str):
                parsed_response = json.loads(raw_response)
            else:
                parsed_response = raw_response

            # validate using Pydantic
            validated_results = RagCrewResponseList(results=parsed_response)

            # Salva nello state per uso successivo
            self.state.rag_crew_validated_response = validated_results
            print(f"✅ Validation successful! Found {len(validated_results.results)} results")
            return "JSON valid"
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as err:
            print(f"❌ Validation failed: {err}")
            # Puoi decidere se continuare con dati parziali o fermare il flow
            print("Invalid JSON structure from RagCrew, repeat the flow.")
            return "JSON not valid"

    @listen("JSON valid")
    def process_web_site_validation(self):
        """Mark web results as trusted based on the whitelist.

        Reads ``WHITE_LIST_PATH`` and annotates each WEB-origin result with an
        ``is_trusted`` flag.
        """
        # Usa i dati già validati
        validated_results = self.state.rag_crew_validated_response
        # Carica whitelist per validazione URL
        with open(WHITE_LIST_PATH, 'r', encoding='utf-8') as file:
            white_list_data = yaml.safe_load(file)
            trusted_domains = list(white_list_data.get("white_list", {}).values())
        # Aggiorna is_trusted per risultati web
        updated_results = []
        for result in validated_results.results:
            if result.origin == "WEB":
                # Verifica se l'URL è nella whitelist
                is_trusted = any(
                    domain in result.source for domain in trusted_domains
                )
                # Crea nuovo oggetto con is_trusted aggiornato
                updated_result = result.copy(update={"is_trusted": is_trusted})
                updated_results.append(updated_result.dict())
            else:
                updated_results.append(result.dict())

        self.state.web_validation_results = updated_results

    @listen(process_web_site_validation)
    def process_answer_writing(self):
        """Execute the AnswerWriter crew using validated web results.

        Returns:
            dict: Aggregated payload ready for downstream consumption.
        """
        answer_writer = (
            AnswerWriter()
            .crew()
            .kickoff(inputs={
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "response_rag_crew": self.state.web_validation_results
                })
        )

        self.state.answer_writer = answer_writer
        self.state.answer_writer_raw_response = answer_writer.raw

        payload = {
            "rag_crew": self.state.rag_crew,
            "answer_writer": self.state.answer_writer,
            "user_question": self.state.user_question,
            "topic": self.state.topic,
            "is_about_topic": self.state.is_topic_related,
            "response_rag_crew": self.state.rag_crew_raw_response,
            "validated_rag_crew": self.state.rag_crew_validated_response,
            "web_validation_results": self.state.web_validation_results,
            "answer_writer_raw_response": self.state.answer_writer_raw_response
        }
        return payload

    @listen(process_answer_writing)
    def finalize(self, payload: Dict[str, Any]):
        """Return the aggregated payload at the end of the flow.

        Args:
            payload (dict): Aggregated results from all previous steps.

        Returns:
            dict: The same payload passed in, for convenience.
        """
        return payload

def kickoff():
    """Start the ``RagFlow`` from the command line.

    This helper sets up telemetry and kicks off the interactive flow.
    """
    track_crewai(project_name="crewai-integration-rag_crew")
    rag_flow = RagFlow()
    rag_flow.kickoff()
    # rag_flow.plot()

if __name__ == "__main__":
    kickoff()
