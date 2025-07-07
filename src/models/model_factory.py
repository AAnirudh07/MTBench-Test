from base_model import BaseModel


class ModelFactory:
    def __init__(self, config: dict):
        self.config = config

    def get_model(self, model_type: str) -> BaseModel:
        if model_type == "deepseek":
            pass
        elif model_type == "llama":
            pass
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
