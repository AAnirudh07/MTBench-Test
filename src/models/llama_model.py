from transformers import pipeline
from base_model import BaseModel

class LLaMAModel(BaseModel):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", **kwargs):
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
            **kwargs
        )

    def inference(self, content: str) -> str:
        messages = [{"role": "user", "content": content}]
        outputs = self.pipeline(messages, max_new_tokens=1024)
        return outputs[0]["generated_text"][-1]["content"]
