from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from sentence_transformers import SentenceTransformer

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
    keywords = [candidates[index] for index in distances.argsort()[0][:top_n]]

    distances.sort()
    dists = distances[0, :top_n]

    for i, j in zip(keywords, dists):
        print(i,j)

    return keywords


"""
istilbert — base-nli-stsb-mean-tokens performs well for semantic similarity
xlm-r-distilroberta-base-paraphase-v1 performs well for paraphrase identification
"""
model = SentenceTransformer('all-MiniLM-L12-v2')

doc = "Poor oxidation behavior is the major barrier to the increased use of Ti-based alloys in high-temperature structural applications. The demand to increase the service temperature of these alloys beyond 550°C (the typical temperature limit) requires careful study to understand the role that composition has on the oxidation behavior of Ti-based alloys [1–3]. The attempt to overcome this limitation in Ti-based alloys has led to the production of alloys with substantially improved oxidation resistance such as β-21S and also development of coatings and pre-oxidation techniques [1,4–6]. While it is tempting to extrapolate the oxidation behavior (e.g. oxidation rate law, depth of oxygen ingress and scale thickness) observed for a limited number of compositions under a certain oxidation condition to a broader compositional range, there are numerous examples in the literature where deviations from the expected relations are observed [7,8]."
# doc = "The Age of Enlightenment or the Enlightenment,[note 2] also known as the Age of Reason, was an intellectual and philosophical movement that occurred in Europe in the 17th and 18th centuries, with global influences and effects.[2][3] The Enlightenment included a range of ideas centered on the value of human happiness, the pursuit of knowledge obtained by means of reason and the evidence of the senses, and ideals such as natural law, liberty, progress, toleration, fraternity, constitutional government, and separation of church and state.[4][5] The Enlightenment was preceded by the Scientific Revolution and the work of Francis Bacon, John Locke, among others. Some date the beginning of the Enlightenment to the publication of René Descartes' Discourse on the Method in 1637, featuring his famous dictum, Cogito, ergo sum (\"I think, therefore I am\"). Others cite the publication of Isaac Newton's Principia Mathematica (1687) as the culmination of the Scientific Revolution and the beginning of the Enlightenment. European historians traditionally date its beginning with the death of Louis XIV of France in 1715 and its end with the 1789 outbreak of the French Revolution. Many historians now date the end of the Enlightenment as the start of the 19th century, with the latest proposed year being the death of Immanuel Kant in 1804. Philosophers and scientists of the period widely circulated their ideas through meetings at scientific academies, Masonic lodges, literary salons, coffeehouses and in printed books, journals,[7] and pamphlets. The ideas of the Enlightenment undermined the authority of the monarchy and the Catholic Church and paved the way for the political revolutions of the 18th and 19th centuries. A variety of 19th century movements including liberalism, communism, and neoclassicism trace their intellectual heritage to the Enlightenment. The central doctrines of the Enlightenment were individual liberty and religious tolerance, in opposition to an absolute monarchy and the fixed dogmas of the Church. The concepts of utility and sociability were also crucial in the dissemination of information that would better society as a whole. The Enlightenment was marked by an increasing awareness of the relationship between the mind and the everyday media of the world,[9] and by an emphasis on the scientific method and reductionism, along with increased questioning of religious orthodoxy—an attitude captured by Kant's essay Answering the Question: What is Enlightenment, where the phrase Sapere aude (Dare to know) can be found.[10] "

pred_kws = get_unsupervised_keywords(doc, model, (1,1), 7)
print(pred_kws)

# print()

gt_kws = parseKeywords('/home/aneesh/UbuntuStorage/Homework/INLP/SciTraction/SemEvalData/scienceie2017_train/train2/S0010938X1500195X.ann')
single = []
for k in gt_kws:
    if len(k) == 1:
        single.append(k)
print(single)