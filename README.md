# <h1 align="center">Chefmate AI: Conversational AI Cooking Assistant (Backend)</h1>

**An AI-driven cooking assistant API that recommends recipes, answers culinary questions, and helps you cook creatively with what's already in your kitchen.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) 
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)
![AI-Powered](https://img.shields.io/badge/AI-enabled-6f42c1.svg)

---

## Project Description

**Chefmate AI** is an AI-driven cooking assistant backend designed to answer fundamental questions like:

> _“What can I cook with what I have right now?”_

This project reimagines how users interact with recipe databases by transforming traditional keyword searches into a **context-aware, conversational recommendation system**. The backend executes a highly modular natural language processing (NLP) pipeline powered by local inference and vector-based retrieval (RAG).

Chefmate AI goes beyond static recipes; it understands ingredient substitutions, cooking methods, and dietary constraints through an **LLM-enhanced recipe reasoning engine** enhanced using **Retrieval-Augmented Generation (RAG)**. This enables the system to generate custom cooking instructions and recommendations in real-time, fostering an interactive dialogue that adapts to user needs and preferences.

---

## Table of Contents
- [Project Title and Overview](#project-title-and-overview)
- [Project Description](#project-description)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Installation](#installation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contribution](#contribution)
- [License](#license)
- [FAQs](#faqs)
- [Contact Information](#contact-information)

---

## Features

### Conversational Recipe Retrieval
- Accepts natural-language queries (e.g., _“What can I make with mushrooms and garlic?”_)
- Retrieves relevant recipes using semantic understanding, not keyword matching
- Maintains conversational context over multiple turns

### Ingredient-Based Search with Substitution
- Analyzes available ingredients and dietary preferences
- Suggests recipes based on pantry inventory

### Remote LLM via OpenRouter
- Uses OpenRouter API for high-quality conversational responses
- Supports models like `openai/gpt-5.4-mini` and others
- No local GPU or large model files required

### Vector-Based Semantic Search
- Embeds both queries and recipes using MiniLM transformers
- Powered by FAISS for high-speed approximate nearest neighbor search
- Smart fallback logic ensures query satisfaction

### Modular NLP Pipeline
- Heuristic intent detection categorizes user messages (e.g., find, refine, clarify)
- Dynamic prompt construction using system template + retrieved context + chat history
- Output is sanitized and structured in standard JSON

### Structured Recipe Output
- JSON format includes: title, ingredient list, method, and optional tips

---

## Technology Stack

- **Python** *(core backend language)*
- **FastAPI** *(high-performance API framework for routing and inference orchestration)*
- **Uvicorn** *(ASGI server for high-speed FastAPI hosting)*
- **FAISS** *(Facebook AI Similarity Search for efficient vector retrieval)*
- **Sentence Transformers**: `all-MiniLM-L6-v2` *(for embedding user queries and recipe corpus)*
- **Prompt Construction Engine** *(Jinja2 templating or dynamic string formatting for building LLM prompts)*
- **Custom Heuristic Engine** *(for intent detection from user message)*

### LLM & Inference Engine

- **OpenRouter API** – Unified gateway for accessing multiple LLM providers (OpenAI, Anthropic, and more).
- **GPT-5.4-mini (via OpenRouter)** – Default model for fast, high-quality conversational responses.
- No local model files or GPU required.

### NLP & Processing Pipeline

- **Heuristic-based Intent Detection** *(determine user goal: search, refine, clarify, etc.)*
- **Embedding Pipeline using Sentence Transformers** *(MiniLM)*
- **Semantic Retrieval using FAISS** *(fallback logic on score thresholds)*
- **Post-processing module** *(cleans and formats LLM output into structured JSON)*

---

## Project Structure

```
backend/
├── main.py              # FastAPI entrypoint, CORS setup, router registration.
├── requirements.txt     # All Python dependencies.
└── app/
    ├── api/             # FastAPI routers for chat and data preparation.
    ├── core/            # Startup logic, dependency initialization.
    └── utils/           # Core logic for embeddings, FAISS, LLM, intent detection,
                         # prompt engineering, and recipe preprocessing.
```

---

## Documentation

### Backend API and Flow

- [**Backend API Documentation**](./docs/backend_api_doc.pdf)  
  Comprehensive reference for the `/chat` endpoint and RAG + LLM response pipeline.

- [**RAG Flow Diagram**](./docs/rag_llm_backend_flow.png)  
  *High-level flowchart of the backend request pipeline (RAG + LLM).*

- [**Backend Flow Diagram**](./docs/backend_api_flow_diagram.png)  
  Visual walkthrough of the backend logic per user request.

- [**Chefmate Pipeline Architecture (Text)**](./docs/chefmate_pipeline_architechture_diagram.txt)  
  Textual breakdown of the processing pipeline logic for easy version tracking and discussion.

---

## Installation

### Prerequisites

- **Python** (3.10+)
- **pip** (latest)
- **Git** (Optional)
- **virtualenv** (for Python isolation)

### 1. Clone the Repository

```bash
git clone https://github.com/ThakkarVidhi/chefmate-ai.git
cd chefmate-ai
```

### 2. Backend Setup

#### a. Create and Activate a Virtual Environment (Recommended)

```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

#### b. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### c. Configure Environment

- Copy `.env.example` to `.env` and fill in your **OpenRouter API key**:
  ```bash
  cp .env.example .env
  ```

- The `.env` file controls all paths, model names, and API keys. No `config.yml` is required.

#### d. Prepare Recipe Data and Indexes

1. Download the raw recipe dataset from Kaggle: [Kaggle Food Recipes Dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data?select=recipes.csv)

2. Save the raw CSV in:
    ```bash
    data/raw/recipes.csv
    ```

3. Run the standalone data preparation script:
    ```bash
    python scripts/prepare_data.py
    ```

    This cleans the CSV, generates embeddings, and builds FAISS indexes. The outputs are stored in:
    - `data/processed/` (cleaned CSV and pickle)
    - `data/indexes/` (FAISS indexes for titles, ingredients, and quantities)

#### e. Run the Backend Server
    
```bash
uvicorn main:app --reload
```

- The API will be available at [http://localhost:8000](http://localhost:8000).

---

## Configuration

Configuration is handled entirely through **environment variables** (via a `.env` file or your shell).

Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
```

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | *(optional)* | Kaggle credentials for automatic dataset download |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `EMBEDDING_BATCH_SIZE` | `128` | Batch size for embedding generation |
| `OPENROUTER_MODEL` | `openai/gpt-5.4-mini` | LLM model accessed via OpenRouter |

All file paths (recipe data, processed data, FAISS indexes) can also be overridden via environment variables. See `.env.example` for the full list.

> **Note:** A legacy `config.yml` is still supported for local development, but **`.env` is the preferred and recommended method** — especially for Docker deployments.

---

## Docker Deployment

The easiest way to run Chefmate AI in production or development is with Docker Compose.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start

1. **Clone the repository and enter the project folder:**
   ```bash
   git clone https://github.com/ThakkarVidhi/chefmate-ai.git
   cd chefmate-ai
   ```

2. **Create your environment file:**
   ```bash
   cp .env.example .env
   # Edit .env and set at least OPENROUTER_API_KEY and KAGGLE credentials
   ```

3. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

   The container expects data artifacts to already exist in the `data/` volume. If they are missing, the container will exit with an error.

4. **(First time only) Prepare data inside the container:**
   If you haven't transferred pre-built artifacts, place `recipes.csv` in `data/raw/` and run:
   ```bash
   docker-compose exec chefmate python scripts/prepare_data.py
   ```
   Then restart the container:
   ```bash
   docker-compose restart
   ```

5. **Access the API:**
   - API docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health check: [http://localhost:8000/](http://localhost:8000/)

### Useful Commands

| Command | Description |
|---------|-------------|
| `docker-compose up --build` | Build image and start container |
| `docker-compose up -d` | Start in detached (background) mode |
| `docker-compose down` | Stop and remove container |
| `docker-compose down -v` | Stop and **remove data volume** (wipe all indexes) |
| `docker logs -f chefmate-ai` | View live logs |

### Resource Limits

The `docker-compose.yml` is pre-configured for a **1 CPU / 4 GB RAM** machine:

```yaml
deploy:
  resources:
    limits:
      cpus: "1.0"
      memory: 4G
    reservations:
      cpus: "0.5"
      memory: 2G
```

Adjust these values in `docker-compose.yml` if your host has different specs.

### Security Notes

- The container runs as a **non-root user** (`appuser`, UID 1001).
- The `OPENROUTER_API_KEY` is injected via the `.env` file and is **never baked into the image**.
- Kaggle credentials are also read from environment variables at runtime.

---

## Testing

To ensure Chefmate AI backend is working correctly, follow these steps for basic validation.

- Start the backend server:
    ```bash
    uvicorn main:app --reload
    ```

- Try endpoint: `/chat`

- Test a basic ingredients POST request:
    ```bash
    curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" --data-raw '{"chat_history":[{"role":"user","content":"What can I cook with flour, eggs, salt, onion and garlic"}]}'
    ```

> Full automated tests will be added in future version.

---

## Contribution

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make changes & commit (`git commit -m 'Add a feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).  
![License](https://img.shields.io/badge/license-MIT-blue)

---

## FAQs

**Q: Where can I get the recipe dataset?**  
**A:** The dataset is sourced from Kaggle. Search for "Food Recipes Dataset" on Kaggle, download the CSV file, and place it at the path defined in `config.yml`, usually `data/raw/recipes.csv`.

--

**Q: Do I still need `config.yml`?**  
**A:** No. Configuration is now handled through environment variables (`.env` file). A legacy `config.yml` is still read if present, but all settings can be overridden via environment variables. For Docker deployments, `.env` is the only file you need.

--

**Q: How is recipe data processed and indexed?**  
**A:** 
- Raw recipe data is cleaned and stored in `data/processed/`.
- Embeddings are generated and stored in the same folder.
- FAISS indexes for title, ingredient, and ingredient_with_quantity are saved in `data/indexes/`.

You can update the configuration paths for these in `config.yml`.

Refer to the Installation and Configuration sections above for detailed setup.

---

## Contact Information

For questions or feedback, feel free to reach out:

- **Email**: [vidhithakkar.ca@gmail.com](mailto:vidhithakkar.ca@gmail.com)
- **LinkedIn**: [Vidhi Thakkar](https://www.linkedin.com/in/vidhi-thakkar-0b509724a/)
