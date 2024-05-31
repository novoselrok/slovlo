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
