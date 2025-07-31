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


with open('stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

from nltk.tokenize.texttiling import TextTilingTokenizer
tt = TextTilingTokenizer(demo_mode=True, stopwords= stopwords, smoothing_width = 10)

def segment_with_tt(text): # text - список предложений
    text = ' \n\n\n\t '.join(text)
    n_text = lemmatize_text(text)

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

def segment_texts(texts, segmentation_method = segment_with_tt, segmentation_method_params = {}, verbouse = True):
    # на вход идет список с текстами, на выход – слоарь сегментов и словарь текстов (с сегментами)
    segmented_texts = {}
    all_segments_separated = {}
    seg_ind = 0
    if verbouse:
        it = tqdm(range(len(texts)))
    else:
        it = range(len(texts))
    for i in it:
        clean_text, norm_text, text_ind = texts[i]['clean_text'], texts[i]['tokenized_text'], texts[i]['id']
        try:# сегментированный текст по идее тоже надо инвертировать к ненормализованному формату
            segmented_text_norm = segmentation_method(norm_text, **segmentation_method_params)# нужно еще добавить полный текст без сегментов и нормализации
            segmented_texts[text_ind] = segmented_text_norm
            
            for i in range(len(segmented_text_norm)):
                all_segments_separated[seg_ind] = [text_ind, segmented_text_norm[i]] # записываем, к какому тексту относится сегмент.
                seg_ind += 1
        except ValueError:
            segmented_texts[text_ind] = [[''],['']]
            pass
    return segmented_texts