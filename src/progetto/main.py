"""High-level orchestration for the RAG demo application.

This module defines a Flow-based interaction where a user is prompted for a
question about a predefined topic, the question is validated by an LLM, and
valid questions are executed through a `RagCrew` to produce answers.

It exposes the `kickoff` function to start the flow from CLI and the
`RagFlow` class implementing the flow logic.
"""
import json
from random import randint
from typing import Any, List, Dict, Union
from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start, or_, router

from progetto.crews.rag_crew.rag_crew import RagCrew
from progetto.crews.answer_writer.answer_writer import AnswerWriter
from crewai import LLM
import yaml 

import opik
opik.configure(use_local=True)
from opik.integrations.crewai import track_crewai
track_crewai(project_name="crewai-rag-crew-progetto")

# Settings
white_list_path = "utils/white_list.yaml"

class JSONResponse(BaseModel):
    """Schema for the JSON response returned by the validation LLM.

    Attributes:
        is_about_topic (bool): Whether the question refers to the configured topic.
    """

    is_about_topic: bool = Field(description="Whether the question is about the topic")

class RagState(BaseModel):
    """State carried across the flow steps.

    Attributes:
        topic (str): The subject used to scope valid questions.
        user_question (str): The last question provided by the user.
        is_about_topic (bool): Result of the LLM validation step.
        response_rag_crew (List[Dict[str, Any]]): The parsed response from the RagCrew.
    """

    topic: str = Field(default="Python programming", description="The topic to research about")
    user_question: str = Field(default="", description="The question provided by the user")
    is_about_topic: bool = Field(default=False, description="Whether the question is about the topic")
    response_rag_crew: List[Dict[str, Any]] = Field(default_factory=list, description="The response from the RagCrew")
    

class RagFlow(Flow[RagState]):
    """Conversation flow to validate and answer topic-specific questions.

    The flow consists of:
    - start: initialize the flow
    - ask_question: prompt the user for a question
    - process_topic: validate the question using an LLM with JSON schema
    - validate_question: route to either retry or proceed to RAG execution
    - process_rag: run the `RagCrew` to answer the question
    """

    @start()
    def start(self):
        """Initialize the flow state and start the run.

        Returns:
            None: This method is an entry point that triggers the flow.

        Examples:
            >>> flow = RagFlow()
            >>> result = flow.start()  # returns None, begins the flow
        """
        return 
    
    @listen(or_(start, "Retry question"))
    def ask_question(self):
        """Prompt the user to input a question about the configured topic.

        Side Effects:
            Stores the user's input in ``self.state.user_question``.

        Returns:
            None

        Examples:
            >>> flow = RagFlow()
            >>> flow.state.topic = "Python programming"
            >>> # This call prompts on stdin; for tests, patch input()
            >>> # flow.ask_question()
        """
        print("Make a question about the Python programming topic")
        self.state.user_question = input("Your question: ")
        

    @listen(ask_question)
    def process_topic(self):
        """Evaluate if the question is about the current topic using an LLM.

        On success, sets ``self.state.is_about_topic`` by parsing the LLM JSON
        response.

        Returns:
            None

        Raises:
            json.JSONDecodeError: If the LLM output is not valid JSON.
            Exception: Propagated from the underlying LLM client on failure.

        Examples:
            >>> flow = RagFlow()
            >>> flow.state.topic = "Python programming"
            >>> flow.state.user_question = "What is a list in Python?"
            >>> # In tests, mock LLM.call to return '{"is_about_topic": true}'
            >>> # flow.process_topic()
        """
        print("Evaluating if the question is about the topic...")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=JSONResponse)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "You are an assistant designed to evaluate whether the question is about the topic, and provide an answer in a json format with a boolean field called is_about_topic."},
            {"role": "user", "content": f"""
            Evaluate if the question "{self.state.user_question}" is about the topic "{self.state.topic}". Respond strictly with a JSON string in the style of provided JSONResponse format without any kind of other extra text rather than the json.
            """}
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        print("Raw response: ", response)
        # Parse the JSON response
        outline_dict = json.loads(response)
        self.state.is_about_topic = outline_dict.get("is_about_topic", False)
    
    @router(process_topic)
    def validate_question(self):
        """Route based on validation outcome.

        Returns:
            str: The next route label ("Question valid" or "Retry question").

        Examples:
            >>> flow = RagFlow()
            >>> flow.state.is_about_topic = True
            >>> flow.validate_question()
            'Question valid'
            >>> flow.state.is_about_topic = False
            >>> flow.validate_question()
            'Retry question'
        """
        print("Validate if the question is about the topic")
        if self.state.is_about_topic:
            return "Question valid"
        else:
            print(f"The question '{self.state.user_question}' is not about the topic '{self.state.topic}'. Please ask a relevant question.")
            return "Retry question"
        
    @listen("Question valid")
    def process_rag(self):
        """Execute the RAG crew to answer a validated question.

        Returns:
            None

        Examples:
            >>> flow = RagFlow()
            >>> flow.state.user_question = "What is a list in Python?"
            >>> flow.state.topic = "Python programming"
            >>> # This triggers the configured crew; in tests, mock RagCrew.crew()
            >>> # flow.process_rag()
        """
        with open('utils/white_list.yaml', 'r') as file:
            white_list = yaml.safe_load(file)
        rag_crew = (
            RagCrew()
            .crew()
            .kickoff(inputs={
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "white_list": white_list
                })
        )
        # Parse the raw JSON response from rag_crew
        response_rag_crew_parsed = json.loads(rag_crew.raw)
        
        payload = {"rag_crew": rag_crew,
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "is_about_topic": self.state.is_about_topic,
                "response_rag_crew": response_rag_crew_parsed
            }
        
        return payload
    
    @listen(process_rag)
    def process_web_site_validation(self, payload: Dict[str, Any]):
        """ Process the web site validation step. Check if the url is in white list"""
        print("=======PROCESS WEB SITE=========")
        print(payload["response_rag_crew"])
        sources = json.loads(payload["response_rag_crew"])
        with open(white_list_path, 'r') as file:
            white_list_data = yaml.safe_load(file)
            # Now white_list is directly a dict with name->url mapping
            urls = list(white_list_data.get("white_list", {}).values())
        for item in sources:
            if item.get("origin") == "WEB":
                item["is_trusted"] = any(url in item.get("source", "") for url in urls)
        # payload["response_rag_crew"] = sources
        return sources
        


    @listen(process_web_site_validation)
    def process_answer_writing(self, sources: Dict[str, Any]):
        # sources = json.loads(payload["response_rag_crew"])
        answer_writer = (
            AnswerWriter()
            .crew()
            .kickoff(inputs={
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "response_rag_crew": sources
                })
        )
        # sources.update({"answer_writer_crew": answer_writer,
        #         "final_answer_crew": answer_writer.raw
        #     })
        # return sources

    # @listen(process_answer_writing)
    # def finalize(self, payload: Dict[str, Any]):
    #     return payload

def kickoff():
    """Start the ``RagFlow`` from the command line.

    Returns:
        None

    Examples:
        >>> # From Python
        >>> kickoff()
        >>> # From CLI
        >>> # python -m rag_crew.main
    """
    track_crewai(project_name="crewai-integration-rag_crew")
    rag_flow = RagFlow()
    rag_flow.kickoff()
    # rag_flow.plot()

if __name__ == "__main__":
    kickoff()

#Peppa pig è un cane? 
#Come definisco una lista? What are the application fields of Python?