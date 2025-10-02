import json
from random import randint
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field, ValidationError

from crewai.flow import Flow, listen, start, or_, router

from progetto.crews.rag_crew.rag_crew import RagCrew
from progetto.crews.answer_writer.answer_writer import AnswerWriter
from crewai import LLM
import yaml 

# Opik tracking
import opik
from opik.integrations.crewai import track_crewai
opik.configure(use_local=True)
track_crewai(project_name="crewai-rag-crew-progetto")

# Global Settings
white_list_path = "utils/white_list.yaml"
main_topic = "Calls for Proposals or Calls for Action"
allowed_topics = ["Culture", "ICT", "Green Economy"]

class RagCrewResponseItem(BaseModel):
    """Schema for the JSON response returned by the RagCrew.
    The JSON response is expected to be similar to this structure:
    [
        {
            "origin": "RAG",
            "title": "01_intro_python.md",
            "similarity": 0.82157665,
            "source": "01_intro_python.md",
            "content": "## Campi di applicazione\n- **Web Development**: framework come Django e Flask.\n- **Data Science**: librerie come NumPy, Pandas, Matplotlib.\n- **Automazione**: scripting e automazione di processi.\n- **Intelligenza Artificiale**: TensorFlow, PyTorch.\n- **Applicazioni Desktop**: interfacce grafiche con Tkinter, PyQt.\n\n## Esempio semplice\n```python\nprint(\"Ciao, mondo!\")\n```\n\nPython non accetta assolutamente bottle.",
            "is_trusted": true
        },
        {
            "origin": "WEB",
            "title": "Python - Application Areas - Tutorials Point",
            "similarity": 1.0,
            "source": "https://www.tutorialspoint.com/python/python_application_areas.htm",
            "content": "Python is used in Data Science, Machine Learning, Web Development, Computer Vision, Embedded Systems, Job Scheduling, GUI, Console, CAD, and Game Development.",
            "is_trusted": false
        }
    ]
    """
    origin: str = Field(description="The origin of the source, e.g., 'RAG' or 'WEB'")
    title: str = Field(description="The title of the source")
    source: str = Field(description="The URL or identifier of the source")
    content: str = Field(description="The content of the source used as a knowledge")
    similarity: float = Field(description="A similarity value comes from RAG")
    is_trusted: bool = Field(description="Whether the source is trusted based on white list")

class RagCrewResponseList(BaseModel):
    """Validation model for a list of RagCrewResponseItem."""
    results: List[RagCrewResponseItem] = Field(description="List of RAG Crew search results")

class FlowState(BaseModel):
    """State carried across the flow steps.

    Attributes:
        topic (str): The subject used to scope valid questions.
        user_question (str): The last question provided by the user.
        is_about_topic (bool): Result of the LLM validation step.
        response_rag_crew (List[Dict[str, Any]]): The parsed response from the RagCrew.
    """
    topic: str = Field(default="", description="The topic for scoping questions")
    is_topic_related: bool = Field(default=False, description="Whether the question is about the topic")
    user_question: str = Field(default="", description="The question provided by the user")
    rag_crew_raw_response: str = Field(default="", description="The raw response from the RagCrew")
    rag_crew_validated_response: Optional[RagCrewResponseList] = Field(default=None, description="Validated search results")


class RagFlow(Flow[FlowState]):
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
        """The first step in the flow."""
        return 
    
    @listen(or_(start, "Retry question"))
    def ask_question(self):
        """Prompt the user to provide a sector and a question relevant to the main topic."""
        print(f"Provide the {main_topic} sector: ")
        self.state.topic = input("sector: ")
        print("Provide a question: ")
        self.state.user_question = input("question: ")
        

    @listen(ask_question)
    def process_topic(self):
        """Evaluate if the sector and the question are relevant to the current topic using an LLM.
        On success, sets ``self.state.is_topic_related`` by parsing the LLM JSON
        response.
        """
        print("Validating the question and the sector.")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=FlowState)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": f"""
             You are an assistant designed to evaluate whether a topic and a user question are relevant to 
             {main_topic}, and provide an answer in a json format with a boolean field called 
             is_topic_related.
             """},
            {"role": "user", "content": f"""
            Evaluate if:
            - the question "{self.state.user_question}" and the topic "{self.state.topic}" are relevant to {main_topic}. 
            - the topic "{self.state.topic}" is related to one of these topics: {', '.join(allowed_topics)}.
            Italian translations are accepted.
            Respond strictly with a JSON string in the style of provided JSONResponse format without any kind of other extra text rather than the json.
            """}
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        print("Raw response: ", response)
        # Parse the JSON response
        llm_response = json.loads(response)
        self.state.is_topic_related = llm_response.get("is_topic_related", False)

        payload = {
            "main_topic": main_topic,
            "allowed_topics": allowed_topics,
            "user_topic": self.state.topic,
            "user_question": self.state.user_question,
            "is_topic_related": self.state.is_topic_related,}
        return payload

    @router(process_topic)
    def validate_question(self, payload: Dict[str, Any]):
        """Route based on validation outcome."""
        if self.state.is_topic_related:
            print(f"The question is relevant to the topic")
            return "Question valid"
        else:
            print(f"The question '{self.state.user_question}' is not about the topic '{self.state.topic}'. Please ask a relevant question.")
            return "Retry question"

    @listen(or_("Question valid", "JSON not valid"))
    def process_rag(self):
        """Execute the RAG crew to answer a validated question.
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
        """Valida i risultati della ricerca usando Pydantic."""
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
        except ValidationError as e:
            print(f"❌ Validation failed: {e}")
            # Puoi decidere se continuare con dati parziali o fermare il flow
            print("Invalid JSON structure from RagCrew, repeat the flow.")
            return "JSON not valid"

    @listen("JSON valid")
    def process_web_site_validation(self, payload: Dict[str, Any]):
        """Process the web site validation step with validated data."""
        # Usa i dati già validati
        validated_results = self.state.rag_crew_validated_response
        # Carica whitelist per validazione URL
        with open(white_list_path, 'r') as file:
            white_list_data = yaml.safe_load(file)
            trusted_domains = list(white_list_data.get("white_list", {}).values())
        # Aggiorna is_trusted per risultati web
        updated_results = []
        for result in validated_results.results:
            if result.origin == "WEB":
                # Verifica se l'URL è nella whitelist
                is_trusted = any(domain in result.source for domain in trusted_domains)
                # Crea nuovo oggetto con is_trusted aggiornato
                updated_result = result.copy(update={"is_trusted": is_trusted})
                updated_results.append(updated_result.dict())
            else:
                updated_results.append(result.dict())
        return updated_results
       

    
    # @listen("JSON valid")
    # def process_web_site_validation(self, payload: Dict[str, Any]):
    #     """ Process the web site validation step. Check if the url is in white list"""
    #     print("=======PROCESS WEB SITE=========")
    #     print(payload["response_rag_crew"])
    #     sources = json.loads(payload["response_rag_crew"])
    #     with open(white_list_path, 'r') as file:
    #         white_list_data = yaml.safe_load(file)
    #         # Now white_list is directly a dict with name->url mapping
    #         urls = list(white_list_data.get("white_list", {}).values())
    #     for item in sources:
    #         if item.get("origin") == "WEB":
    #             item["is_trusted"] = any(url in item.get("source", "") for url in urls)
    #     # payload["response_rag_crew"] = sources
    #     return sources
        


    @listen(process_web_site_validation)
    def process_answer_writing(self, updated_results: List[Dict[str, Any]]):
        # sources = json.loads(payload["response_rag_crew"])
        answer_writer = (
            AnswerWriter()
            .crew()
            .kickoff(inputs={
                "user_question": self.state.user_question,
                "topic": self.state.topic,
                "response_rag_crew": updated_results
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