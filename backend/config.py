from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "ghost_engineer"
    mongodb_max_pool_size: int = 20
    mongodb_min_pool_size: int = 2

    gitlab_webhook_token: str = "change-me"
    gitlab_url: str = "https://gitlab.com"
    gitlab_token: str = ""
    gitlab_project_id: int = 0

    google_cloud_project: str = ""
    vertex_ai_location: str = "us-central1"


settings = Settings()
