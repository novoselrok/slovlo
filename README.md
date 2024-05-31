# SloVlo Project

The SloVlo V1 (**Slo**venske **Vlo**žitve) project brings purposefully built embeddings and semantic search capabilities to the Slovenian language.

This repository contains the necessary code to produce the slovlo-v1 dataset, and the slovlo-v1 embedding model (both available on HuggingFace).

You can read more about the motivation behind the project, development, and evaluation [here](#).

## Usage

For a quick usage example see the [find_trip.py](examples/find_trip.py) script:

```py
import sys
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# We will read the query from the command line arguments.
query = sys.argv[1]

# First, we define the documents we want to search over.
# In our case, that is a list of destination descriptions.
documents = [
    "Triglav je najvišja gora v Sloveniji (2864 m) in simbol slovenske narodne identitete. Pohod je zahteven in običajno traja dva dni. Potrebna je dobra fizična pripravljenost in osnovno znanje plezanja. Priporočena je tudi uporaba vodnika za manj izkušene pohodnike.",
    "Velika Planina je zelo priljubljena pohodniška destinacija z značilnimi pastirskimi kočami. Pohod je primeren za vse starosti in ponuja čudovite razglede na okoliške gore. Na vrh se lahko povzpnete peš ali z nihalko iz Kamniške Bistrice.",
    "Bled je znan po kremnih rezinah. Če vas zanima pohod, so pa zraven še Ojstrica, ter Mala in Velika Osojnica.",
    "Golica je znana po neskončnih poljih narcis v maju. Pohod se začne iz vasi Planina pod Golico in traja približno 2-3 ure. Pot je primerna za vse pohodnike in ponuja lepe razglede na Julijske Alpe in Avstrijo.",
    "Šmarna Gora je najbolj priljubljena pohodniška destinacija v bližini Ljubljane. Pohod traja približno 1 uro iz Tacna. Na vrhu je koča, kjer lahko uživate v tradicionalni slovenski hrani in lepih razgledih na Ljubljansko kotlino.",
    "Pohorje je pohodniško območje z različnimi potmi, primernimi za vse starosti in pripravljenosti. Posebej priljubljena je pot do Črnega jezera in Slivniškega jezera. Pozimi je Pohorje tudi priljubljena smučarska destinacija.",
]

# Load the model and the tokenizer.
slovlo_model = AutoModel.from_pretrained("rokn/slovlo-v1").eval().to(device)
slovlo_tokenizer = AutoTokenizer.from_pretrained("rokn/slovlo-v1")


def get_embeddings(texts: List[str], prefix: str):
    def mean_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    prefixed_texts = [f"{prefix}{text}" for text in texts]
    inputs = slovlo_tokenizer(
        prefixed_texts, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    with torch.no_grad():
        model_output = slovlo_model(**inputs)

    embeddings = mean_pool(model_output.last_hidden_state, inputs["attention_mask"])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


# Embed the documents (destinations).
document_embeddings = get_embeddings(documents, "document: ")

# Embed the user query.
query_embedding = get_embeddings([query], "query: ")

# Compute dot product between the query and each document.
similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()

# Find the nearest neighbor.
nearest_index = torch.argmax(similarities).item()

print("Predlog za tvojo naslednjo avanturo:", documents[nearest_index])
```

## Requirements

```sh
pip install numpy torch transformers sentencepiece protobuf
```

```sh
# For evaluation only
pip install faiss-cpu elasticsearch
```

## Data collection

### rtvslo.si

```sh
python slovlo/scrape/gather_rtvslo_articles.py --output_dir ../data/rtvslo
python slovlo/scrape/combine_rtvslo_urls.py --dir ../data/rtvslo --output_path ../data/rtvslo_urls.json
python slovlo/scrape/scrape_rtvslo_urls.py --urls_path ../data/rtvslo_urls.json --output_dir ../data/rtvslo_content
python slovlo/scrape/create_rtvslo_pairs.py --dir ../data/rtvslo_content --output_path ../data/rtvslo_pairs.jsonl
```

### r/slovenia

```sh
python slovlo/scrape/scrape_reddit.py --dir ../data/reddit --output_path ../data/reddit_pairs.jsonl
```

### Wikipedia

```sh
python slovlo/scrape/create_wikipedia_pairs.py --dir ../data/wiki/dump/A --output_path ../data/wikipedia_pairs.jsonl --min_paragraph_length 64
```

## Training the model

```sh
./slovlo/scripts/train_embedding_model.sh slovlo-dataset-v1 slovlo-v1 train.log
```

## Running the evaluation

```sh
# multilingual-e5-base
python slovlo/embedding_evaluation/mrr_with_dense_models.py --dataset_path slovlo-dataset-v1/test.jsonl --model_path intfloat/e5-base-v2

# multilingual-e5-base
python slovlo/embedding_evaluation/mrr_with_dense_models.py --dataset_path slovlo-dataset-v1/test.jsonl --model_path intfloat/multilingual-e5-base

# BGE-M3
python slovlo/embedding_evaluation/mrr_with_dense_models.py --dataset_path slovlo-dataset-v1/test.jsonl --model_path BAAI/bge-m3

# slovlo-v1
python slovlo/embedding_evaluation/mrr_with_dense_models.py --dataset_path slovlo-dataset-v1/test.jsonl --model_path slovlo-v1

# Elasticsearch
python slovlo/embedding_evaluation/mrr_with_bm25.py --dataset_path slovlo-dataset-v1/test.jsonl
```
