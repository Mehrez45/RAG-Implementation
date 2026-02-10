from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "llama.cpp/build/models/qwen2.5-7b-instruct-q5_k_m.gguf"

class LocalLLM:
    def __init__(self):
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 512,
                temperature: float = 0.3, repeat_penalty: float =1.2) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["END", "</s>"],
            repeat_penalty=repeat_penalty,
            
        )

        return output["choices"][0]["text"].strip()

    