# Progetto Academy Gruppo 4

## Overview

This repository contains the source code and configuration for a modular Python system designed to automate the retrieval, validation, and synthesis of information about official calls for proposals, funding opportunities, and related documentation.  
The architecture leverages CrewAI for agent orchestration and supports advanced retrieval-augmented generation (RAG) workflows, including web search integration and document validation.

## Architecture

The project is organized around two main "crews", each responsible for a distinct stage in the pipeline:

### 1. RAG Crew (`src/progetto/crews/rag_crew/`)

- **Purpose:**  
  Retrieves relevant knowledge chunks from a Qdrant vector database and supplements document-based answers with targeted web research.
- **Agents:**  
  - `rag_retriever`: Specialized in extracting high-similarity document chunks using `rag_tool_qdrant`.
  - `web_researcher`: Conducts authoritative web searches via SerperDevTool, filters and merges with internal results.
- **Tasks:**  
  - `rag_retriever_task`: Executes vector retrieval and formats results as structured JSON.
  - `web_research_task`: Performs web queries, extracts high-quality snippets, and merges them with RAG results.
- **Configuration:**  
  All agent roles, goals, and output schemas are declared in YAML files for maintainability and clarity.
- **Extensibility:**  
  Custom tools (e.g., `rag_tool_qdrant`) are defined in `src/progetto/tools/custom_tool.py` and can be easily extended.

### 2. Answer Writer Crew (`src/progetto/crews/answer_writer/`)

- **Purpose:**  
  Validates retrieved information against a strict source priority hierarchy, resolves factual conflicts, and synthesizes answers into clear, actionable Markdown reports.
- **Agents:**  
  - `validator`: Detects and resolves discrepancies between RAG and web sources, enforcing trust policies.
  - `answer_writer`: Synthesizes validated context into professional, structured answers with explicit source citation.
- **Tasks:**  
  - `validator_task`: Group-by-source, detect conflicts, resolve via priority (RAG > trusted web > untrusted web), and annotate JSON validity.
  - `writing_task`: Generates user-facing Markdown answers, citing only validated sources and explaining any resolved conflicts.
- **Output:**  
  Structured Markdown (see `outputs/answer_writer/complete_answer.md`), with inline citations and source transparency.

## File Structure

```
src/progetto/
│
├── crews/
│   ├── rag_crew/
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   └── tasks.yaml
│   │   ├── rag_crew.py
│   │   └── __init__.py
│   └── answer_writer/
│       ├── config/
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       └── answer_writer.py
│
├── tools/
│   └── custom_tool.py
│
├── documents/
│   └── [Official PDF documents, calls, notices...]
│
└── outputs/
    └── [Generated answers, validation logs...]
```

## Key Technologies

- **Python 3.10+**
- **CrewAI** (agent orchestration, process management)
- **Qdrant** (vector database for semantic retrieval)
- **SerperDevTool** (web search integration)
- **Structured YAML configuration** (agent/task definitions)
- **Custom tool integration** for extensible retrieval pipelines

## Usage

1. **Install dependencies:**  
   (Ensure Python virtual environment is activated)
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure agents and tasks:**  
   Edit YAML files under `src/progetto/crews/*/config/` to define agent behaviors and task output formats.

3. **Add knowledge sources:**  
   Place official documents in `documents/` and ensure Qdrant is populated with chunked embeddings.

4. **Run crews:**  
   Instantiate and execute crews from Python:
   ```python
   from src.progetto.crews.rag_crew.rag_crew import RagCrew
   from src.progetto.crews.answer_writer.answer_writer import AnswerWriter
   # crew = RagCrew().crew()
   # crew.kickoff()
   ```

5. **Review outputs:**  
   Generated answers and validation logs are saved in `outputs/`.

## Extending the System

- **Add new agents:**  
  Define roles and goals in YAML, implement Python logic as needed.
- **Integrate new tools:**  
  Add tool definitions to `src/progetto/tools/custom_tool.py` and reference in agent configs.
- **Customize validation logic:**  
  Modify the priority hierarchy or conflict resolution strategy in validator agent YAML/task definitions.

## Example Workflow

1. User submits a query about a funding call.
2. RAG Crew retrieves relevant document chunks and supplements with web research.
3. Answer Writer Crew validates all sources, resolves conflicts, and synthesizes a Markdown answer with source citations.

## License

MIT License (see LICENSE file).

## Contributors

- Group 4, Academy Program
- Eleonora Amico, Fabrizio Corda, Adriano Neroni, Alessandro Venanzi 

## References

- [CrewAI documentation](https://docs.crewai.com/)
- [Qdrant Vector DB](https://qdrant.tech/)
- [SerperDevTool](https://serper.dev/)

---

*For further details, consult the code and configuration in the respective `src/progetto/crews/` folders.*
