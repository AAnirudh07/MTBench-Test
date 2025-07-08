from transformers import AutoModelForCausalLM, AutoTokenizer
from base_model import BaseModel

class DeepSeekModel(BaseModel):
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", **kwargs):
        
        # The model is set in eval mode by default by using eval()
        # See: https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto", 
            device_map="auto",
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, content: str) -> str:
        messages = [{"role": "user", "content": content}]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_input = self.tokenizer([chat_prompt], return_tensors="pt").to(self.model.device)
        generated_output = self.model.generate(
            **tokenized_input,
            max_new_tokens=4096,
        )
        new_tokens = [
            output[len(input_ids):]
            for input_ids, output in zip(tokenized_input.input_ids, generated_output)
        ]
        outputs = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return outputs