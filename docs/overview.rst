Project overview
================

This project builds a question–answering workflow to generate a report based on
user questions about Calls for Proposals. It uses a Retrieval Augmented
Generation (RAG) approach backed by local knowledge and optional web sources,
orchestrated by crews.

Main components
---------------

- ``progetto.main``: Conversation flow that validates topic relevance, runs RAG,
  validates web sources, and produces an answer.
- ``progetto.crews.rag_crew``: Crew that retrieves relevant knowledge from the
  dataset and optionally queries the web.
- ``progetto.crews.answer_writer``: Crew that validates inputs and composes the
  final answer/report.
- ``utils``: Utilities for prompt-injection detection and RAG helpers.


