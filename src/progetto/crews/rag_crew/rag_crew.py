from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from progetto.tools.custom_tool import rag_tool_qdrant
from crewai_tools import SerperDevTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Define an agent with the RagTool

@CrewBase
class RagCrew():
    """RagCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def rag_retriever(self) -> Agent:
        '''
        This agent uses the RagTool to answer questions about the knowledge base.
        '''
        return Agent(
            config=self.agents_config["rag_retriever"],
            allow_delegation=False,
            tools=[rag_tool_qdrant]  # Chiamiamo la funzione per ottenere l'istanza di RagTool
        )

    @agent
    def web_researcher(self) -> Agent:
        '''
        This agent uses the RagTool to answer questions about the knowledge base.
        '''
        return Agent(
            config=self.agents_config["web_researcher"],
            allow_delegation=False,
            tools=[SerperDevTool()]  # Chiamiamo la funzione per ottenere l'istanza di RagTool
        )
    
    # @agent
    # def validator(self) -> Agent:
    #     '''
    #     This agent uses the RagTool to answer questions about the knowledge base.
    #     '''
    #     return Agent(
    #         config=self.agents_config["validator"],
    #         allow_delegation=False,
    #         tools=[] 
    #     )
        
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def rag_retriever_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_retriever_task'], # type: ignore[index]
        )
    @task
    def web_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_research_task'],
            context=[self.rag_retriever_task()]  # type: ignore[index]
        )

    # @task
    # def validator_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['validator_task'],
    #         context=[self.web_research_task()]  # type: ignore[index]
    #     )


    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

