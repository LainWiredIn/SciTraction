import re
from transformers import AutoTokenizer


def parseKeywords(fileName):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    
    all_kws = []
    with open(fileName, 'r') as f:
        lines = f.readlines()

        for line in lines:
            words = line.split('\t')
            if words[0][0] != 'R':
                kw_tag = re.sub('\n', '', words[-1])
                kws = re.split('[-.:\/\' ]', kw_tag.lower())

            all_kws.append(kws)
    return all_kws

# all_kws = parseKeywords('/home/aneesh/UbuntuStorage/Homework/INLP/SciTraction/SemEvalData/scienceie2017_train/train2/S0010938X1500195X.ann')
# print(all_kws)