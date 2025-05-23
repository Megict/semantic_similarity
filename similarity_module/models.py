from similarity_measure.paper_similarity import calculate_text_similarities
from segmentation_models.paper_segmentation_model import text_segment_classifier
from similarity_measure.paper_similarity import get_sentence_embedding


from tqdm import tqdm
import numpy as np
import json
import pylab
import re

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import pylab

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import pylab

def create_tsne_for_plot(data, labels, highlight = None, size = None, nc = 10, color_map = cm.cool ):
    np.random.seed(0)
    max_label = max(labels)
    # max_items = np.random.choice(range(data.shape[0]), size=len(labels), replace=False)
    
    # pca = PCA(random_state = 1, n_components=2).fit_transform(np.asarray(data))
    tsne = TSNE(random_state = 1, n_iter = 5000).fit_transform(PCA(n_components=nc).fit_transform(np.asarray(data)))
    
    label_colors = [color_map(i/max_label) for i in labels]

    return tsne[:,0], tsne[:,1], label_colors

## Предобработчик
def preprocess(texts):
    # 1. базовая обработка на уровне символов. удаление ненужных символов.
    red_texts = []
    for i in tqdm(range(len(texts))):
        rt = texts[i].replace('-\n', '')
        rt = rt.replace('\n', ' ')
        rt = re.sub('[^а-яё.,:()0-9А-Я–\-\[\]]', ' ', rt)
        
        rt = rt.replace(' . ', ' ')
        rt = re.sub('\s+', ' ', rt)
        rt = rt.split('.')
        rt = '. '.join([elm.strip() for elm in rt if elm.strip() !=''])
        
        red_texts.append(rt)
    # 2. Обработка на уровне предложений. очистка текста от мусорных предложений.
    red2_texts = []
    for text in tqdm(red_texts):
        sents = []
        for sent in text.split('. '):
            sent = sent.strip('0123456789 ,')
            if len(sent) == 0:
                continue
            sent_only_letters = re.sub('[^а-яёА-Я]', '', sent)
            sent_only_lower = re.sub('[^а-яё]', '', sent)
            sent_split = sent.lower().split()
            if len(sent_only_letters) / len(sent) < 0.8 or len(sent_only_lower) / len(sent) < 0.7:
                # удаление предложений, содержащих мало букв                
                continue
            if len(sent_split) < 4: 
                # удаление предложений, содержащих мало слов                
                continue
            if 'Список' in sent_split and 'литературы' in sent_split:
                # отсечение списка литературы
                break
            if 'Список' in sent_split and 'источников' in sent_split:
                break
            sents.append(sent)
        text_red = '. '.join(sents)
        red2_texts.append(text_red)
    # финальная обработка текста. Удаление слишком длинных предложений и слияние разбитых и слишком коротких предложений. 
    # разделение текста на отдельные предложения.
    split_texts = []
    for cur_text in tqdm(red2_texts):
        split_text = []
        join_next = False
        for sent in cur_text.split('. '):
            if len(sent) > 1500:
                continue
            # просто по точкам делить не стоит.
            if join_next:
                split_text[-1] = split_text[-1] + " " + sent
                join_next = False
                continue
            # принимаем решение, добавлять ли предложение или нет.
            if len(sent) < 30:
                continue        
            # про несправедливые разбиения
            if (not re.search("рис$",sent.lower()) is None) or (not re.search("табл$",sent.lower()) is None):
                # print("will join")
                join_next = True
                split_text.append(sent)
                continue
            split_text.append(sent)
        split_texts.append(split_text)
    return split_texts


class similarity_base:
    def __init__(self):
        self.segment_model = text_segment_classifier("segmentation_models/full_model_segmentation_0.74_.wt", only_cpu= True)
        self.text_library = []
        self.available_tags = ['Goal','Obl','Resl','Algo','None']

    def add_texts(self, texts, titles = []):
        # предобработка текстов
        preproc_texts = preprocess(texts)
        # сегметация текстов
        segmented_texts = []
        for ind in tqdm(range(len(preproc_texts))):
            try:
                if titles != []:
                    this_title = titles[ind]
                else:
                    this_title = ''
                text = preproc_texts[ind]
                preds = self.segment_model.mass_predict_alter(preproc_texts[ind], verbal = False)
                text_dict = {}
                vect_dict = {}
                for i in range(len(preds)):
                    text_dict.setdefault(preds[i], [])
                    vect_dict.setdefault(preds[i], [])
                    text_dict[preds[i]].append(text[i].strip())
                    vect_dict[preds[i]].append(get_sentence_embedding(text[i].strip()))
                segmented_texts.append({'title' : this_title, 'full' : preproc_texts[ind], 'segmented' : text_dict, 
                                        'vectorized' : {i[0]:np.array(i[1]) for i in vect_dict.items()}})
            except IndexError as err:
                print(err)
                continue
        # добавление
        self.text_library += segmented_texts

    def vector_similarity(self, lhs_vectors, rhs_vectors):
        return cosine_similarity(lhs_vectors, rhs_vectors)

    def find_similar(self, text, tag_to_measure_on = None):
        # предобработка текстоа
        preproc_text = preprocess([text])[0]
        # сегметация текста
        preds = self.segment_model.mass_predict_alter(preproc_text, verbal = False)
        text_dict = {}
        vect_dict = {}
        for i in range(len(preds)):
            text_dict.setdefault(preds[i], [])
            vect_dict.setdefault(preds[i], [])
            text_dict[preds[i]].append(preproc_text[i].strip())
            vect_dict[preds[i]].append(get_sentence_embedding(preproc_text[i].strip()))
        vect_dict = {i[0]:np.array(i[1]) for i in vect_dict.items()}
        # теперь проведем сравнение
        similarities = {tag : {} for tag in vect_dict if tag != 'None'}
        for tag in similarities:
            for other_text_id in tqdm(range(len(self.text_library))):
                other_text = self.text_library[other_text_id]
                # if tag in other_text['vectorized']:True
                #     simil = self.vector_similarity(vect_dict[tag], other_text['vectorized'][tag])
                #     similarities[tag][other_text_id] = [simil, other_text['full'], other_text['segmented'][tag]]
                simils = calculate_text_similarities(vect_dict, 
                                                     other_text['vectorized'], 
                                                     compare_sentences = True, selection_method = 5, comparation_method=self.vector_similarity)
                for tag in simils:
                    if tag in other_text['segmented'] and tag != 'None':
                        if simils[tag] is not None:
                            similarities[tag][other_text_id] = [simils[tag], other_text['title'], other_text['full'], other_text['segmented'][tag]]

        top_similarities = {tag : {} for tag in vect_dict}
        for tag in similarities:
            top_similarities[tag] = dict(sorted(similarities[tag].items(), key = lambda x: x[1][0], reverse = True)[0:5])
        if tag_to_measure_on is None:
            return text_dict, top_similarities
        else:
            return text_dict, top_similarities[tag_to_measure_on]

    def make_display(self, tag = 'Algo', highlight_text_id = [], n_clusters = 5):
        text_vector_dim = get_sentence_embedding('a').shape[0]
        single_vector_representations = []
        all_text_vectors = []
        opacity_param = []
        cnt = 0
        for text in self.text_library:
            opacity_param.append(1 if cnt in highlight_text_id else 0.2)
            text_repr = np.zeros(text_vector_dim)
            all_text_vectors.append(text['vectorized'])
            if tag in text['vectorized']:
                text_repr = np.mean(np.array(text['vectorized'][tag]), axis = 0)
            single_vector_representations.append(text_repr)
            cnt += 1
        # делаем кластеризацию для разметки по цветам
        text_similarities_matrix = np.zeros((len(all_text_vectors), len(all_text_vectors)))
        for i in tqdm(range(len(all_text_vectors))):
            for j in range(i, len(all_text_vectors)):
                t_sims = calculate_text_similarities(all_text_vectors[i], all_text_vectors[j], compare_sentences = True,
                                                     selection_method = 5, comparation_method=self.vector_similarity)
                text_similarities_matrix[i][j] = t_sims[tag]
        for i in range(len(text_similarities_matrix)):
            for j in range(0,i):
                text_similarities_matrix[i,j] = text_similarities_matrix[j,i]
        text_similarities_matrix = np.nan_to_num(text_similarities_matrix, nan = 1e-3)
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state = 0).fit(text_similarities_matrix)
        coloring = clustering.labels_
        # само размещение
        _plot_data = create_tsne_for_plot(data = single_vector_representations, 
                                              labels = coloring, color_map = cm.twilight)
    
        f, ax = pylab.subplots(1, 1, figsize=(7, 7))
        if highlight_text_id != []:
            ax.scatter(_plot_data[0], _plot_data[1], c = _plot_data[2], marker = '+', alpha = opacity_param)   
        else:
            ax.scatter(_plot_data[0], _plot_data[1], c = _plot_data[2], marker = '+')   
        pylab.show()

    def save_text_library(self, lib_fpath = "similarity_base_segment_lib.dump"):
        for i in range(len(self.text_library)):
            for cm in self.text_library[i]['vectorized']:
                for j in range(len(self.text_library[i]['vectorized'][cm])):
                    self.text_library[i]['vectorized'][cm][j] = self.text_library[i]['vectorized'][cm][j].tolist()
                
        with open(lib_fpath, "w") as fout:
            json.dump(self.text_library, fout)

    def load_text_library(self, lib_fpath = "similarity_base_segment_lib.dump"):
        with open(lib_fpath, "r") as fin:
            self.text_library = json.load(fin)
        for i in range(len(self.text_library)):
            for cm in self.text_library[i]['vectorized']:
                for j in range(len(self.text_library[i]['vectorized'][cm])):
                    self.text_library[i]['vectorized'][cm][j] = np.array(self.text_library[i]['vectorized'][cm][j])
                    

    def compare(self, lhs_text, rhs_text):
        # предобработка текстоа
        lhs_preproc_, rhs_preproc_ = preprocess([lhs_text, rhs_text])
        # сегметация текста
        lhs_text_dict = {}
        rhs_text_dict = {}
        preds = self.segment_model.mass_predict_alter(lhs_preproc_, verbal = False)
        for i in range(len(preds)):
            lhs_text_dict.setdefault(preds[i],[])
            lhs_text_dict[preds[i]].append(lhs_preproc_[i])
        preds = self.segment_model.mass_predict_alter(rhs_preproc_, verbal = False)
        for i in range(len(preds)):
            rhs_text_dict.setdefault(preds[i],[])
            rhs_text_dict[preds[i]].append(rhs_preproc_[i])
        
        simils = calculate_text_similarities(lhs_text_dict, 
                                             rhs_text_dict, 
                                             compare_sentences = True, selection_method = 5)
        return simils