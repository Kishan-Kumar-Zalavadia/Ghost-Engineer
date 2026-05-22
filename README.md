# Ghost Engineer

AI-powered engineering assistant built with FastAPI, MongoDB, and Google Cloud Vertex AI.

## Project Structure

```
ghost-engineer/
├── backend/        # FastAPI application
├── frontend/       # Frontend application
├── agent/          # AI agent logic
├── tests/          # Test suite
├── docs/           # Documentation
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Google Cloud SDK
- MongoDB (or use Docker Compose)

## Setup

1. Clone the repository and navigate to the project root.

2. Copy the environment file and fill in your values:
   ```bash
   cp .env.example .env
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

5. Or run the backend locally:
   ```bash
   uvicorn backend.main:app --reload
   ```

## Development

Run tests:
```bash
pytest
```

## Services

| Service   | Port  | Description           |
|-----------|-------|-----------------------|
| Backend   | 8000  | FastAPI REST API      |
| Frontend  | 3000  | Web UI                |
| MongoDB   | 27017 | Database              |
