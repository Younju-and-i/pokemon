from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  neo4j_uri: str
  neo4j_user: str
  neo4j_password: str
  ollama_host: str

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()
