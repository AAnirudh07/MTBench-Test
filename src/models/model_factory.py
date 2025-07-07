from base_model import BaseModel
from llama_model import LLaMAModel
from deepseek_model import DeepSeekModel

class ModelFactory:
    def __init__(self, config: dict):
        self.config = config

    def get_model(self, model_type: str, model_name: str, **kwargs) -> BaseModel:
        if model_type == "deepseek":
            return DeepSeekModel(model_name=model_name, **kwargs)
        elif model_type == "llama":
            return LLaMAModel(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
