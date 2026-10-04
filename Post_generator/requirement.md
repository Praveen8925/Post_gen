# Project Requirements: Content Generator AI

## 1. Project Overview
**Content Generator AI** is an automated content repurposing system designed to transform raw information into high-engagement social media content. The system takes an article, a document, a URL, or a general topic and converts it into professionally crafted posts for **LinkedIn** and **Instagram**, accompanied by AI-generated visual assets.

### Goal
To eliminate the manual effort of reading a long article and manually drafting multiple social media versions by using a structured AI pipeline that ensures consistency in tone, target audience, and visual branding.

---

## 2. Core AI Components

### 2.1 LLM (Large Language Model)
- **Model**: `llama3.2:3b` (via Ollama).
- **Role**: The "brain" of the operation. It is used for:
    - Analyzing raw text to extract key themes.
    - Generating a structured summary (JSON).
    - Drafting platform-specific copy (LinkedIn vs. Instagram).
    - Deciding which tools to call when running in "Agent Mode".
    - Creating prompts for the image generation tool.

### 2.2 AI Agent
The system implements a hybrid agent architecture that can operate in two distinct modes:

#### A. Pipeline Mode (Deterministic)
- **How it works**: Follows a fixed, rule-based sequence of tool calls.
- **Purpose**: Maximum reliability and speed.
- **Flow**: Input $\to$ Scraping/RAG $\to$ Summarization $\to$ Post Generation $\to$ Media Generation.

#### B. LLM Agent Mode (Autonomous)
- **How it works**: The LLM is given a set of tool definitions and a goal. It decides autonomously which tool to call, analyzes the output, and decides the next step.
- **Purpose**: Greater flexibility. It can handle complex or ambiguous requests where a fixed pipeline might fail.

---

## 3. Toolset Specification

The project employs a "Tool-Use" architecture where the agent delegates specific tasks to specialized Python modules.

| Tool Name | Purpose | Working Mechanism |
| :--- | :--- | :--- |
| **Web Scraper** | Extract content from a website. | Uses `Playwright` to render JavaScript-heavy pages and `BeautifulSoup4` to parse and clean the HTML content, returning only the main article text. |
| **Web Search** | Gather info on a specific topic. | Interfaces with the `DuckDuckGo Search` API to find the most relevant current web pages and summarizes the top results to create a knowledge base. |
| **RAG Tool** | Process PDF/DOCX files. | **R**etrieval-**A**ugmented **G**eneration. It parses documents $\to$ splits text into chunks $\to$ generates embeddings (Ollama) $\to$ stores in `ChromaDB` $\to$ retrieves relevant chunks based on a query. |
| **Summarizer** | Create a "Content Blueprint". | Processes raw text and converts it into a structured JSON format containing: Topic, Summary, Key Points, a "Hook" line, Tone, and Target Audience. |
| **Post Generator** | Draft social media copy. | Takes the structured summary and applies platform-specific templates (e.g., professional/bold for LinkedIn, emoji-rich/casual for Instagram). |
| **Media Tool** | Generate visual assets. | Analyzes the final post content to create a visual prompt, then calls the `NVIDIA NIM` API to generate images like Infographics, Diagrams, or Theme images. |

---

## 4. Architecture Pipeline

The system follows a linear data-flow pipeline to ensure the output is grounded in the input data.

### The Logical Flow:
1. **Input Layer**: 
   - User provides a URL, Topic, Text, or File.
   - The system identifies the input type.
2. **Acquisition Layer (The "Gatherer")**:
   - If URL $\to$ `Web Scraper`.
   - If Topic $\to$ `Web Search`.
   - If File $\to$ `RAG Tool`.
   - If Text $\to$ Direct pass.
3. **Processing Layer (The "Analyzer")**:
   - The raw text is passed to the `Summarizer`.
   - Output: A structured JSON "Blueprint".
4. **Generation Layer (The "Writer")**:
   - The Blueprint is passed to the `Post Generator`.
   - Output: Platform-specific copy for LinkedIn and Instagram.
5. **Visual Layer (The "Artist")**:
   - The generated posts are analyzed by the `Media Tool`.
   - Output: An AI-generated image saved to `media_output/`.
6. **Delivery Layer (The "UI")**:
   - The final posts, structured summary, and images are rendered in the Streamlit dashboard.
