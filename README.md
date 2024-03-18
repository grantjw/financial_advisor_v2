# Financial Advisory with Large Language Models

## Project Background and Objective

The landscape of financial advising has seen a remarkable evolution from traditional methodologies to cutting-edge AI-driven solutions. This transformation is largely attributed to advancements in technology and the advent of large language models (LLMs). Our project seeks to capitalize on this technological progression to empower users in managing their financial portfolios more effectively. By integrating diverse Retrieval-Augmented Generation (RAG) architectures with human-aligned, fine-tuned LLMs, we propose a sophisticated solution aimed at redefining financial advising. This endeavor is not just about leveraging the computational prowess of LLMs but also about aligning these models with the nuanced needs of financial advisory services to deliver personalized, insightful, and actionable advice.

## Introduction to Large Language Models in Finance

Large Language Models (LLMs) like LLaMA2 and Mistral-7B have shown remarkable capabilities in understanding and generating human-like text. Our project aims to harness these capabilities to solve complex problems in the financial domain. By focusing on a model-agnostic framework, we introduce methodologies such as Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning with Human Feedback (RLHF) to enhance the performance of LLMs on finance-related tasks. Our approach is rooted in the belief that LLMs can become invaluable tools for financial advising, provided they are finely tuned and optimized to understand and anticipate the financial queries and needs of users.

## Literature Review and References

In our exploration of aligning LLMs to financial advising tasks, we draw upon significant works like Cheng et al.'s "Black-Box Prompt Optimization" and Rafailov et al.'s insights on Direct Preference Optimization (DPO). These studies provide a foundation for our methodologies, particularly in human-alignment strategies and efficiency in model optimization without extensive retraining. Our project stands at the intersection of cutting-edge research and practical financial advisory solutions, embodying the latest advancements in AI and machine learning within the financial context.

## Data Science Capstone Project Partnership

Our project, in collaboration with Accenture, represents a pivotal step in the Data Science Capstone Project. This partnership underscores our commitment to integrating academic rigor with industry-relevant applications, particularly in the realm of generative AI LLMs for financial advising.

## Methodologies and Technologies

### Auto Retrieval-Augmented Generation (AutoRAG)

AutoRAG epitomizes our model-agnostic approach, facilitating dynamic queries for financial advice. By incorporating intent classification, we ensure that user queries are efficiently routed to either structured or unstructured RAG based on the nature of the information sought. This dual-pathway ensures that whether the query requires insights from textual content or data analytics, the system is poised to generate accurate and relevant responses.

### Prompt Optimization and Fine-Tuning

Our project employs a sophisticated prompt optimization strategy, enhancing user prompts through a seq-to-seq model trained on optimized and original prompt-result pairs. This refinement process ensures that user intentions are precisely captured and effectively communicated to the LLM. Moreover, our Supervised Fine-tuning (SFT) methodology further trains pre-existing LLMs on specific tasks relevant to financial advising, significantly boosting their performance and relevance in this domain.

### Direct Preference Optimization (DPO)

DPO represents a streamlined approach to human-alignment, leveraging the logits from SFT LLMs to compute rewards directly. This method stands out for its efficiency, bypassing the need for a separate reward model and instead utilizing the LLM itself for reward calculation.

## Data Utilization

Our project leverages an extensive dataset encompassing a subset of Fortune 500 companies, cryptocurrencies, and commodities. This dataset includes structured data such as user portfolios and historical stock prices from sources like Yahoo Finance, as well as unstructured data from news articles to capture market sentiments. This diverse data foundation enables our LLMs to generate informed, context-aware financial advice.

## User Interface and Interaction

The culmination of our project is a user-friendly interface that allows for seamless interaction with our financial advising tool. Users can engage with various models, including Llama2-7B, Mistral-7B, ChatGPT-3.5, and their fine-tuned versions, to perform Retrieval-augmented Generation tailored to their financial queries. The interface not only facilitates direct interaction with the models but also captures user activity for analytics, enhancing the user experience through personalized insights.

## Results and Evaluation

Our evaluation framework assesses the performance of different models on finance-related queries using metrics such as sentence similarity, cosine similarity, and perplexity. Through rigorous testing across a series of queries, we provide a comprehensive view of each model's capabilities, laying the groundwork for further optimizations and improvements.
### Evaluation Metrics Chart

```markdown
| Metric \ Model           | Base Mistral | Mistral SFT | Mistral DPO | GPT-3.5 Turbo |
|--------------------------|--------------|-------------|-------------|----------------|
| **Sentence Similarity**  | 0.842        | 0.848       | 0.872       | 0.877          |
| **Cosine Similarity**    | 0.890        | 0.846       | 0.849       | 0.893          |
| **Perplexity**           | 10.76        | 7.89        | 8.24        | -              |
| **Time (s)**             | 15.20        | 18.87       | 19.09       | 18.11          |

## Conclusion and Future Directions

This project represents a significant step forward in the application of LLMs to financial advising. By combining advanced AI


# fin-rlhf

Run SFT with: 
```
accelerate launch sft.py
```

Merge SFT LORA weights with 
```
python merge.py --checkpoint_path="results/sft-finqa/checkpoint-550" --merged_path="results/sft-finqa/final_model
```

Generate dataset with 
```
accelerate launch feedback/generate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1" --dataset_name="gbharti/finance-alpaca" --save_path="feedback/finance-alpaca-unlabeled.csv" --num_steps=100
```

Annotate dataset with 
```
python feedback/annotate.py --unlabeled_path="feedback/finance-alpaca-unlabeled.csv" --labels_path="feedback/finance-alpaca-labels.csv"
```

Merge generations with annotations and upload to huggingface with 
```
python feedback/merge_labels.py --hf_repo="danyoung/finance-feedback"
```

Run DPO with
```
accelerate launch dpo.py
```
(Or, you can run "reward_model.py" and "ppo_finqa.ipynb". We met CUDA out of memory issues. The code was not runnable on TR4 8 vCPU 30 GB RAM 16GB VRAM)

Upload model with 
```
huggingface-cli upload danyoung/finance-qa results/sft-finqa/final_model
```

Evaluate a model with
```
accelerate launch evaluation/evaluate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1"
```
#### Python environment
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

#### Data
```
python3 ingest.py
```
#### Environment Variables
```
cp example.env .env # update necessary fields
```

#### Run streamlit app
```
streamlit run app.py
```
