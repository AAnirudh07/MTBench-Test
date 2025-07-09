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

<details>
    <summary>2. How was it done before their work, and what were the limits of prior work?</summary>
    A: 
    <ul>
        <li>Many benchmarks focused only on numerical forecasting but ignored any accompanying text.</li>
        <li>Other datasets (e.g. FinanceBench, FinDABench) tested models on textual financial questions or news summarization without using the underlying numerical data.</li>
        <li> A few multimodal efforts paired text and numbers (e.g. Time‑MMD, ForecastBench), but they either had very limited time‑series length, few data points, or were designed only for simple forecasting, not deeperreasoning tasks like causal inference or QA.</li>
        <li>Existing benchmarks typically fixed the time‑series window and task complexity.</li>
    </ul>

</details>

<details>
    <summary>3. What is new in their approach?</summary>
    A: MTBench introduces a novel approach by comprising paired time-series and textual data across financial and weather domains:
    <ul>
        <li>Financial: Scraping over 200,000 financial news articles, curating a subset of 20,000 articles paired with corresponding stock price movements.</li>
        <li>Weather: Using data from 50 US airports, with historical temperature records from 2003-2020 from the GHCN-H dataset, aligned with the Storm Events Database.</li>
    </ul>
    Beyond forecasting, it introduces semantic trend analysis, technical indicator prediction (e.g. MACD values), and news‑driven QA. The temporal granularity varies, with short-term (1-day) and long-term (7-day) intervals.

</details>

<details>
    <summary>4. What impact does the new approach have?</summary>
    A: The paper suggests that suggests current AI models perform better when using both text and time-series data together. However, they face challenges such as struggling with long-term temporal dependencies, causal reasoning, and always assuming a mildly positive correlation between text and time-series data.  
</details>

<details>
    <summary>5. How do the authors evaluate their proposal? 
</summary>
    A: The authors run a suite of off-the-shelf LLMs measuring:
    <ul>
        <li><i>Forecasting</i>: MAE and MAPE of predicted vs. actual time‑series values, comparing TS‑only vs. TS + Text inputs.</li>
        <li><i>Trend Prediction</i>: Accuracy in assigning correct trend bins (e.g., "–2 % to +2 %"), comparing TS-only vs. TS + Text inputs.</li>
        <li><i>Techincal Indicator Prediction</i>: MSE of predicted vs. actual technical metrics like MACD, comparing TS-only vs. TS + Text inputs.</li>
        <li><i>News-driven QA</i>: Multiple‑choice accuracy and correlation prediction scores on questions that require linking textual claims to numeric evidence.</li>
    </ul>


</details>


## Rationale Behind Choices

**TL;DR**: The open-source models used in MTBench (LLaMA 3.1 8B and DeepSeek-V3) were too large to run within available compute limits.  Even smaller reasoning models like DeepSeek-R1 Distill Qwen-1.5B require generating many tokens to produce meaningful outputs. 

To stay as close as possible to the original setup, used LLaMA 3.2 1B (which shares the same core architecture as LLaMA 3.1) and DeepSeek-R1 1.5B. This setup limited the evaluation to a subset of tasks due to compute time constraints.

---

Two open-source models were evaluated in the [MTBench](https://arxiv.org/pdf/2503.16858) paper: LLaMA 3.1 8B and DeepSeek-V3 (used in DeepSeek-Chat).
1. _LLaMA 3.1 8B_: The paper used LLaMA in the bf16 format which is similar to fp32 in terms of dynamic range. I initially attempted to run the model in fp32 as the compute resources I had access to did not support bf16. Unfortunately, this exceeded the available memory. Switching to fp16 allowed the model to load, but resulted in a very low tokens per second which meant I would not be able to complete the experiments within my remaining compute time.
2. _DeepSeek-V3_: This model has 671B parameters which makes it difficult to load on a non-GPU cluster.

As a result, I decided to use the following variants of the models:
1. **LlaMA 3.2 1B Instruct**: LLaMA 3.2 is built on the same core architecture as LLaMA 3.1 (difference is the addition of a vision adapter in LLaMA 3.2 for multimodal tasks). Thus, opted to use the 1B variant. The paper requires models to follow instructions and use a chat template which is more attuned to the instruct variant.
2. **DeepSeek-R1 Distill Qwen-1.5B**: Unlike V3, it does not use a Mixture of Experts (MoE), but still shows strong reasoning and logical thinking ability.

Both models also have long context lengths.

In effect, this project evalues This project evaluates smaller models on MTBench that claim to be similar in performance to their larger counterparts.

## Code Structure
The files in the `src/` directory are a minimal rewrite of the original [MTBench codebase](https://github.com/Graph-and-Geometric-Learning/MTBench/tree/mainline), but for the financial domain.

- `finance`
    - `correlation_prediction.py`: Similar to the original code but includes extra arguments specific to the model (see `models/` for details).
    - `mcqa.py`: Same as above.
    - `meta_prompt.py`: Contains input prompts that are used only for financial tasks.
    - `trend_classification.py`: Similar to the original code, with added model-specific arguments and modified data loading to support Parquet files and converting data to the required format.
    - `value_prediction.py`: Same as above.
- `models`
    - `base_model.py`: Base class for all models.
    - `deepseek_model.py`: DeepSeek model implementation. Added post-processing to parse out the `<think>` tags.
    - `llama_model.py`: LLaMA model implementation.
    - `qwen_model.py`: Qwen model implementation.
    - `model_factory.py`: Factory class to create instances of models.
- `finetune_llama_tsforecast.py`: A basic first version of an untested script for finetuning LLaMA 3.2 1B on the time series forecasting task.
- `utils.py`: Utility functions for data loading and evaluation.


**NOTE**: The notebooks used for execution contain the code from `src/` pasted into a single file. On VMs, I would have run the code using the `src/` directory structure.



## Experiment Results
As discussed in the `Rationale Behind Choices` section, the DeepSeek model had a very slow inference speed. To stay within compute constraints, the number of evaluation samples for DeepSeek was reduced accordingly.

The datasets and model checkpoints may be accessed [here](https://drive.google.com/drive/folders/1PeoN0CXGaLTgwNMd_Uv1W0JLfplZ6tJl?usp=sharing).

The experiment runs may be accessed at `notebooks/`. If the "Invalid Notebook" error occurs on the GitHub UI, please access the notebooks [here](https://drive.google.com/drive/folders/18_XUCpVM6aUFtBaKmB7YcokjBveq21v8).

#### News-stock Correlation
All numbers are percentages. 
- LlaMA 3.2 1B: Full dataset used.
- DeepSeek-R1 Distill Qwen-1.5B: 100 samples for 7-day and 150 samples for 30-day datasets.

| Model     | 7-Day brief | 7-Day exact | 30-Day brief | 30-Day exact |
|-----------|-------------|-------------|--------------|--------------|
| LLaMA 3.2 1B | 9.0            |   3.69          | 6.51            |  3.68            |
| DeepSeek-R1 Distill Qwen-1.5B |  **34.41**          |  **10.75**          | **31.91**             | **17.02**             |


_Observation_: DeepSeek performs better than LLaMA likely due to stronger instruction tuning and reasoning capabilities, which help it better interpret the relationship between news content and stock movement. As in the paper, the performance slightly degraded on the 30-day dataset.

#### News-driven MCQA
All numbers are percentages. 
- LlaMA 3.2 1B: Full dataset used.
- DeepSeek-R1 Distill Qwen-1.5B: 100 samples for 7-day and 150 samples for 30-day datasets.

| Model     | 7-Day Acc | 30-Day Acc |  
|-----------|----------------|----------------
| LLaMA 3.2 1B |  33.83              | 31.08               |                  
| DeepSeek-R1 Distill Qwen-1.5B  |  **47.31**              |     **36.88**           |                  

_Observation_: DeepSeek shows higher accuracy than LLaMA on both 7-day and 30-day MCQA tasks. This is likely due to the same reasons mentioned in the previous section.

#### Time-Series Forecasting
- LlaMA 3.2 1B: 150 samples for time-series only. 100 samples for time-series + text (both 7-day and 30-day). 
- DeepSeek-R1 Distill Qwen-1.5B: Processing each sample took over 6 minutes on average. Running the full dataset would have required reducing the sample size significantly which would affect evaluation quality. Additionally, only had very little compute time left.


Table: Forecast with LlaMA 3.2 1B (TS vs. w/ Text)

|                        | MAE (TS) | MAE (w/ Text) | MAPE (TS) | MAPE (w/ Text) |
|-----------------------------|----------|---------------|-----------|----------------|
|     7-Day| 2.40         |    **2.23**           | 6.01              |     **5.88**               |
| 30-Day | **3.02**        |  3.09              | 12.12              |  **8.21**          |     


_Observation_: One thing to note is that a significat portion of samples produced very high MSE values and were excluded from the cumulative results.