from transformers import AutoModelForCausalLM, AutoTokenizer
from base_model import BaseModel

class DeepSeekModel(BaseModel):
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", **kwargs):
        
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
            add_generation_prompt=True,
            enable_thinking=False
        )
        tokenized_input = self.tokenizer([chat_prompt], return_tensors="pt").to(self.model.device)
        # https://huggingface.co/Qwen/Qwen3-1.7B#switching-between-thinking-and-non-thinking-mode
        generated_output = self.model.generate(
            **tokenized_input,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )

        output_ids = generated_output[0][len(tokenized_input.input_ids[0]):].tolist() 
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        outputs = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return outputs 