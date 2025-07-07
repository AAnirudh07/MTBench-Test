# MTBench-Test

- [Paper Summary](#paper-summary)
- [Rationale Behind Choices](#rationale-behind-choices)
- [Code Structure](#code-structure)
- [Experiment Results](#experiment-results)


## Paper Summary

<details>
    <summary>1. What are the authors trying to do (no jargon)?</summary>
    A: The authors created a new test suite to benchmark how LLMs understand and reason when given both time‑series data (like stock prices or temperature readings) and the corresponding textual reports (financial news or weather summaries). They want to see if models can answer questions by jointly looking at numbers and words, rather than treating each separately. This is important as real-world events in text can influence these numerical trends and vice versa.
</details>


## Rationale Behind Choices
This section explains why specific models were chosen for this project.

Two open-source models were evaluated in the [MTBench](https://arxiv.org/pdf/2503.16858) paper: LLaMA 3.1 8B and DeepSeek-V3 (used in DeepSeek-Chat).
1. _LLaMA 3.1 8B_: The paper used LLaMA in the bf16 format which is similar to fp32 in terms of dynamic range. I initially attempted to run the model in fp32 as the compute resources I had access to did not support bf16. Unfortunately, this exceeded the available memory. Switching to fp16 allowed the model to load, but resulted in a very low tokens per second which meant I would not be able to complete the experiments within my remaining compute time.
2. _DeepSeek-V3_: This model has 671B parameters which makes it difficult to load on a non-GPU cluster.


As a result, I decided to use the following variants of the models:
1. **LlaMA 3.2 1B**: LLaMA 3.2 is built on the same core architecture as LLaMA 3.1 (difference is the addition of a vision adapter in LLaMA 3.2 for multimodal tasks). Thus, opted to use the 1B variant.
2. **DeepSeek-R1 Distill Qwen-1.5B**: Unlike V3, it does not use a Mixture of Experts (MoE), but still shows strong reasoning and logical thinking ability.

In effect, this project evalues This project evaluates smaller models on MTBench that claim to be similar in performance to their larger counterparts.

## Code Structure
The files in the `src/` directory are a minimal rewrite of the original [MTBench codebase](https://github.com/Graph-and-Geometric-Learning/MTBench/tree/mainline), but for the financial domain.

- `finance`
    - `correlation_prediction.py`: Similar to the original code but includes extra arguments specific to the model (see `models/` for details).
    - `mcqa.py`: Same as above.
    - `meta_prompt.py`: Contains input prompts that are used only for financial tasks.
    - `trend_classification.py`: Similar to the original code, with added model-specific arguments and modified data loading to support Parquet files.
    - `value_prediction.py`: Same as above.
- `models`
    - `base_model.py`: Base class for all models.
    - `deepseek_model.py`: DeepSeek model implementation.
    - `llama_model.py`: LLaMA model implementation.
    - `model_factory.py`: Factory class to create instances of models.
- `utils.py`: Utility functions for data loading and evaluation.

**NOTE**: The notebooks used for execution contain the code from `src/` pasted into a single file. On VMs, I would have run the code using the `src/` directory structure.



## Experiment Results