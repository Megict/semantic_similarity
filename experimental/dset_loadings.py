import pandas as pd
def create_and_save_dataframe(split_texts, split_texts_normalized, all_udcc, filename='text_data.csv'):
    assert len(split_texts) == len(split_texts_normalized) == len(all_udcc), \
        "Все списки должны быть одинаковой длины"
    df = pd.DataFrame({
        'original_text': split_texts,
        'normalized_text': split_texts_normalized,
        'class': all_udcc
    })
    df.to_csv(filename, index=False)
    
def load_and_parse_dataframe(filename='text_data.csv'):
    df = pd.read_csv(filename)
    split_texts_ = df['original_text'].tolist()
    split_texts = []
    for text in split_texts_:
        s_text = text.strip("[]'")
        split_texts.append(s_text.split("', '"))

    split_texts_normalized_ = df['normalized_text'].tolist()
    split_texts_normalized = []
    for text in split_texts_normalized_:
        s_text = text.strip("[]'")
        split_texts_normalized.append(s_text.split("', '"))
    all_udcc = df['class'].tolist()
    return split_texts, split_texts_normalized, all_udcc