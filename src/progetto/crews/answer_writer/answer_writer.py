"""Answer writer crew definition.

This module defines the `AnswerWriter`, a crew that validates inputs and writes
the final report-style answer based on retrieved and validated contexts.
"""
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AnswerWriter():
    """Crew that validates inputs and writes the final answer.

    Attributes:
        agents: Agents automatically created from YAML configuration.
        tasks: Tasks automatically created from YAML configuration.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def validator(self) -> Agent:
        """Agent that validates inputs and constraints before writing.

        Returns:
            Agent: Configured validation agent.
        """
        return Agent(
            config=self.agents_config['validator'], # type: ignore[index]
            verbose=True
        )
    @agent
    def answer_writer(self) -> Agent:
        """Agent that drafts the final answer/report for the user.

        Returns:
            Agent: Configured authoring agent.
        """
        return Agent(
            config=self.agents_config['answer_writer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    @task
    def validator_task(self) -> Task:
        """Task to validate inputs and intermediate content.

        Returns:
            Task: Configured validation task.
        """
        return Task(
            config=self.tasks_config['validator_task'], # type: ignore[index]
        )
    
    @task
    def writing_task(self) -> Task:
        """Task to write the final answer using validated context.

        Returns:
            Task: Configured writing task that depends on validation.
        """
        return Task(
            config=self.tasks_config['writing_task'], # type: ignore[index]
            context=[self.validator_task()]  # ← Usa output del task precedente
        )


    @crew
    def crew(self) -> Crew:
        """Create the `AnswerWriter` crew.

        Returns:
            Crew: Configured sequential crew with validation and writing tasks.
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
