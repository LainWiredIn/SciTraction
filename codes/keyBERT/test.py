from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from sentence_transformers import SentenceTransformer

from icecream import ic
from tqdm import tqdm
import os

from fileParser import *

def get_file_names(dir_path):
    file_names = [file.split('.')[0] for file in os.listdir(dir_path)]
    file_names = list(set(file_names))

    pairs = []
    for name in file_names:
        pairs.append([os.path.join(dir_path, name + ".txt"), os.path.join(dir_path, name + ".ann")])
    
    return pairs

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



"""
istilbert — base-nli-stsb-mean-tokens performs well for semantic similarity
xlm-r-distilroberta-base-paraphase-v1 performs well for paraphrase identification

all-MiniLM-L12-v2 -- modern, recommended version of the above
"""
model = SentenceTransformer('all-MiniLM-L12-v2')
file_pairs = get_file_names("/home/aneesh/UbuntuStorage/Homework/INLP/SciTraction/SemEvalData/scienceie2017_dev/dev")   # [text, annotation]

# doc = "Poor oxidation behavior is the major barrier to the increased use of Ti-based alloys in high-temperature structural applications. The demand to increase the service temperature of these alloys beyond 550°C (the typical temperature limit) requires careful study to understand the role that composition has on the oxidation behavior of Ti-based alloys [1–3]. The attempt to overcome this limitation in Ti-based alloys has led to the production of alloys with substantially improved oxidation resistance such as β-21S and also development of coatings and pre-oxidation techniques [1,4–6]. While it is tempting to extrapolate the oxidation behavior (e.g. oxidation rate law, depth of oxygen ingress and scale thickness) observed for a limited number of compositions under a certain oxidation condition to a broader compositional range, there are numerous examples in the literature where deviations from the expected relations are observed [7,8]."
# doc = "The Age of Enlightenment or the Enlightenment,[note 2] also known as the Age of Reason, was an intellectual and philosophical movement that occurred in Europe in the 17th and 18th centuries, with global influences and effects.[2][3] The Enlightenment included a range of ideas centered on the value of human happiness, the pursuit of knowledge obtained by means of reason and the evidence of the senses, and ideals such as natural law, liberty, progress, toleration, fraternity, constitutional government, and separation of church and state.[4][5] The Enlightenment was preceded by the Scientific Revolution and the work of Francis Bacon, John Locke, among others. Some date the beginning of the Enlightenment to the publication of René Descartes' Discourse on the Method in 1637, featuring his famous dictum, Cogito, ergo sum (\"I think, therefore I am\"). Others cite the publication of Isaac Newton's Principia Mathematica (1687) as the culmination of the Scientific Revolution and the beginning of the Enlightenment. European historians traditionally date its beginning with the death of Louis XIV of France in 1715 and its end with the 1789 outbreak of the French Revolution. Many historians now date the end of the Enlightenment as the start of the 19th century, with the latest proposed year being the death of Immanuel Kant in 1804. Philosophers and scientists of the period widely circulated their ideas through meetings at scientific academies, Masonic lodges, literary salons, coffeehouses and in printed books, journals,[7] and pamphlets. The ideas of the Enlightenment undermined the authority of the monarchy and the Catholic Church and paved the way for the political revolutions of the 18th and 19th centuries. A variety of 19th century movements including liberalism, communism, and neoclassicism trace their intellectual heritage to the Enlightenment. The central doctrines of the Enlightenment were individual liberty and religious tolerance, in opposition to an absolute monarchy and the fixed dogmas of the Church. The concepts of utility and sociability were also crucial in the dissemination of information that would better society as a whole. The Enlightenment was marked by an increasing awareness of the relationship between the mind and the everyday media of the world,[9] and by an emphasis on the scientific method and reductionism, along with increased questioning of religious orthodoxy—an attitude captured by Kant's essay Answering the Question: What is Enlightenment, where the phrase Sapere aude (Dare to know) can be found.[10] "

# init recall and precision values for keyphrases until a cutoff length
cutoff_len = 5
all_stats = {}

quiet = True

for pair in tqdm(file_pairs):
    # parse ground truth keywords and text
    with open(pair[0], 'r') as f:
        doc = f.readline()

    gt_kws = parseKeywords(pair[1])
    
    # sort keywords acc to length
    kws_len_dict = {}
    for kw in gt_kws:
        if len(kw) not in kws_len_dict.keys():
            kws_len_dict[len(kw)] = [kw]
        else:
            kws_len_dict[len(kw)].append(kw)

    if not quiet:
        ic(gt_kws)
        print(kws_len_dict)
        print()

    for i in range(1, cutoff_len):
        if i in kws_len_dict.keys():
            pred_kws, candidates = get_unsupervised_keywords(doc, model, (i,i), len(kws_len_dict[i]))       # generate as many keywords based on no. of gt keywords of that length

            # check if each ground truth kw was generated
            correctly_predicted = 0
            incorrectly_predicted = 0
            missed_kws = 0
            total = len(candidates)

            if not quiet:
                print("Length: %d" % i)

            for kw in kws_len_dict[i]:

                if not quiet:
                    print("\t\t\tkw: %s" % kw)
                    print("\t\t\tkw: %s" % pred_kws)

                if kw in pred_kws:
                    correctly_predicted += 1
                else:
                    missed_kws += 1

            for pred_kw in pred_kws:
                if pred_kws not in kws_len_dict[i]:
                    incorrectly_predicted += 1

            discarded = total - len(pred_kws)

            
            prec = correctly_predicted/(correctly_predicted + incorrectly_predicted)
            reca = correctly_predicted/(correctly_predicted + missed_kws)
            
            if prec == 0 and reca == 0:
                fsco = 0
            else:
                fsco = 2 * (prec * reca)/(prec + reca)
            
            if not quiet:
                ic(correctly_predicted, incorrectly_predicted, missed_kws, discarded)
                print()
                ic(prec, reca, fsco)
                print()
                print()

            if i in all_stats.keys():
                all_stats[i]["tp"] += correctly_predicted
                all_stats[i]["tn"] += discarded
                all_stats[i]["fp"] += incorrectly_predicted
                all_stats[i]["fn"] += missed_kws
            else:
                all_stats[i] = {"tp": correctly_predicted, "tn": discarded, "fp": incorrectly_predicted, "fn": missed_kws}

print("-"*100)

print("Final results:")
valid_lengths = list(all_stats.keys())
valid_lengths.sort()
for length in valid_lengths:
    print("Keyphrase length: %d" % length)

    if all_stats[length]["tp"] != 0:
        precision = all_stats[length]["tp"]/(all_stats[length]["tp"] + all_stats[length]["fp"])
        recall = all_stats[length]["tp"]/(all_stats[length]["tp"] + all_stats[length]["fn"])
    else:
        precision = 0
        recall = 0

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall)/(precision + recall)

    ic(precision, recall, f1_score)
    print()



