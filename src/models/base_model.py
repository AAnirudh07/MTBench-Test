from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def inference(self, content: str) -> str:
        """
        Run inference on a given input prompt and return the generated output.
        """
        pass
