import pymorphy3
import pickle
from tqdm import tqdm

morph = pymorphy3.MorphAnalyzer()
def lemmatize_word(word: str) -> str:
    """Лемматизирует одно слово."""
    parsed = morph.parse(word)[0]  # берем первый (наиболее вероятный) разбор
    return parsed.normal_form
def lemmatize_text(text: str) -> str:
    """Лемматизирует текст (разбивает на слова и обрабатывает каждое)."""
    words = text.split()  # простейшее разбиение по пробелам
    lemmas = [lemmatize_word(word) for word in words]
    return " ".join(lemmas)

def simple_segmentation_sentences(split_text : list):
    return split_text

def simple_segmentation_single(split_text : list):
    return '. '.join(split_text)

#---- text tiling ----
with open('experimental/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

from nltk.tokenize.texttiling import TextTilingTokenizer
tt = TextTilingTokenizer(demo_mode=True, stopwords= stopwords, smoothing_width = 10)

def text_tiling_segmentation(text, n_text): # text - список предложений
    n_text = ' \n\n\n\t '.join(n_text)
    text = ' \n\n\n\t '.join(text)

    gaps = tt._smooth_scores(tt._block_comparison(tt._divide_to_tokensequences(n_text), tt._create_token_table(tt._divide_to_tokensequences(n_text), tt._mark_paragraph_breaks(n_text))))

    depth_scores = tt._depth_scores(gaps)
    segment_boundaries = tt._identify_boundaries(depth_scores)

    normalized_boundaries = tt._normalize_boundaries(
        n_text, segment_boundaries, tt._mark_paragraph_breaks(n_text)
    )
    # End of Boundary Identification
    segmented_text = []
    prevb = 0

    for b in normalized_boundaries:
        if b == 0:
            continue
        segmented_text.append(text[prevb:b])
        prevb = b

    if prevb < len(text):  # append any text that may be remaining
        segmented_text.append(text[prevb:])

    if not segmented_text:
        segmented_text = [text]

    for segment_pos in range(len(segmented_text)):
        segmented_text[segment_pos] = segmented_text[segment_pos].replace(' \n\n\n\t ', ' . ').strip()
    return segmented_text        

from experimental.graph_segmentation.main import PartitionSolver
ps = PartitionSolver('experimental/graph_segmentation/data/adjacency_table.npy', 'experimental/graph_segmentation/data/table.pkl')

def segment_with_graphs(text, normalized_text):
    seg_str, gaps, threshold = ps.solve_partition(normalized_text)
    segmented_text = ['']
    for i in range(len(text)):
        segmented_text[-1] += text[i] + ' . '
        if seg_str[i] == '1':
            segmented_text.append('')
    if segmented_text[-1] == '':
        segmented_text = segmented_text[:-1]
    return segmented_text