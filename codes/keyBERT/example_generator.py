from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from sentence_transformers import SentenceTransformer

from icecream import ic
from tqdm import tqdm
import os

from fileParser import *

def get_unsupervised_keywords(doc, model, n_gram_range=(1,1), top_n=10):
    # extract all possible keyphrase candidates (n_grams)
    stop_words="english"
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()

    # convert candidates to embeddings
    doc_emb = model.encode([doc])
    candidate_embs = model.encode(candidates)

    # get the closest embeddings (lowest dist -> most closely related embeddings)
    distances = cosine_distances(doc_emb, candidate_embs)
    keywords = [candidates[index].split(' ') for index in distances.argsort()[0][:top_n]]

    distances.sort()
    dists = distances[0, :top_n]

    # for i, j in zip(keywords, dists):
    #     print(i,j)

    return keywords, candidates

model = SentenceTransformer('all-MiniLM-L12-v2')
doc = "For the reverse current analysis, for both scenarios (shading and short circuits) were tested on two systems, one system using standard silicon modules and another system using high efficiency modules. For the standard silicon system, a power of 50kWp was considered, with a system composed by 10 strings of 24 modules per string and an approximate system Voc of 864 [VDC]. For the high efficiency system, a power of 40kWp was considered, with a system composed by 10 strings of 18 modules per string and an approximate system Voc of 873 [VDC]. Fig. 5(a) shows the reverse current present in one string when different numbers of modules in the string are shaded by 90%. Fig. 5(b) shows the reverse current present in one string when different numbers of modules of the string are short-circuited. For both figures the continuous lines are for the standard silicon system and the dashed lines are for the high efficiency system."

p, _ = get_unsupervised_keywords(doc, model, (3,3), 10)
print(p)