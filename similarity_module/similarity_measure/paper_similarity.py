import numpy as np
from copy import deepcopy as cp

# измеритель схожести 1. берт + cosine sim
## метод векторизации 1.
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

# модели для векторизации
e_tokenizer = AutoTokenizer.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny3")
e_model = AutoModel.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny3")
# tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
# model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embedding(sentence, model = e_model, tokenizer = e_tokenizer, max_length=None):
    # Tokenize sentences
    sentence = sentence
    encoded_input = tokenizer(
        [sentence], padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(model.device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().detach().numpy()[0]

from sklearn.metrics.pairwise import cosine_similarity
def be_text_sim_func(lhs_texts, rhs_texts):
    lhs_vectors = np.array([get_sentence_embedding(text) for text in lhs_texts])
    rhs_vectors = np.array([get_sentence_embedding(text) for text in rhs_texts])
    return cosine_similarity(lhs_vectors, rhs_vectors)

available_tags = ['Goal','Obl','Resl','Algo','None']

def calculate_text_similarities(lhs_text : list, rhs_text : list, 
                                compare_sentences = True, # сравнивать сегменты, или предложения по-отдельности
                                selection_method = 'max', # только при compare_sentences, как обрабатывать полученную матрицу (max, mean, nmax)
                                comparation_method = be_text_sim_func,
                               ):
    if isinstance(selection_method, int):
        selection_n = cp(selection_method)
        selection_method = 'nmax'
    tags_sims = {tag : None for tag in available_tags}
    for tag in available_tags[:-1]: # не сравниваем по none
        lhs_text_fragment = lhs_text[tag] if tag in lhs_text else None
        rhs_text_fragment = rhs_text[tag] if tag in rhs_text else None
        if lhs_text_fragment is None or rhs_text_fragment is None:
            continue
        if compare_sentences:
            tags_sims[tag] = comparation_method(lhs_text_fragment, rhs_text_fragment)
            if selection_method == 'max':
                tags_sims[tag] = np.max(tags_sims[tag])
            if selection_method == 'mean':
                tags_sims[tag] = np.mean(tags_sims[tag])
            if selection_method == 'nmax':
                f_tags = tags_sims[tag].flatten()
                tags_sims[tag] = np.mean(np.sort(f_tags)[-(min(len(f_tags), selection_n)) : ])
        else:
            tags_sims[tag] = comparation_method(['. '.join(lhs_text_fragment)], ['. '.join(rhs_text_fragment)])[0][0]
    return tags_sims
