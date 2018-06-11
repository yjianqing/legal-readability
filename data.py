import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.parse.corenlp import CoreNLPParser
from xml.etree import ElementTree as ET


data_home = 'data'
glove_home = '../cs224u/vsmdata/glove.6B'
wiki_normal_file = 'normal.aligned'
wiki_simple_file = 'simple.aligned'
scraper_file = 'scraper_combined.tsv'
uscode_file = 'uscode_combined.tsv'


def load_wiki_data(mode=''):
    """
    Load normal and simple Wikipedia datasets and perform additional
    processing: already tokenized similar to CoreNLP format, but some tokens
    need to be replaced, lowercasing (for using uncased GloVe), and removing
    rows where the parallel sentences are the same in both datasets.
    
    Splits the dataset into either for cross-validation or fixed split,
    including keeping the sentence pairs in the same splits, which also
    balances the labels for each class in every split. For cross-validation,
    use scikit-learn's GroupShuffleSplit.
    """
    reps = {'â': '-', '-LRB-': '(', '-RRB-': ')'}
    df_normal = pd.read_csv(os.path.join(data_home, wiki_normal_file), sep='\t', header=None)
    df_normal = df_normal.drop([0, 1], axis=1)
    df_normal = df_normal.rename({2: 0}, axis=1)
    for key in reps:
        df_normal = df_normal[0].apply(lambda x: x.replace(key, reps[key])).to_frame()
    df_normal = df_normal[0].apply(lambda x: x.lower()).to_frame()
    df_normal[1] = 1
    df_simple = pd.read_csv(os.path.join(data_home, wiki_simple_file), sep='\t', header=None)
    df_simple = df_simple.drop([0, 1], axis=1)
    for key in reps:
        df_simple = df_simple[2].apply(lambda x: x.replace(key, reps[key])).to_frame()
    df_simple = df_simple[2].apply(lambda x: x.lower()).to_frame()
    df_simple[3] = 0
    df = pd.concat([df_normal, df_simple], axis=1)
    df = df[df[0] != df[2]] #remove same-string rows
    
    if mode == 'stats':
        calc_stats(df) #normal
        df[0] = df[2]
        calc_stats(df) #simple
        return None
    elif mode == 'cv':
        df = df.reset_index().drop(['index'], axis=1)
        X = pd.concat([df[0], df[2]])
        y = pd.concat([df[1], df[3]])
        groups = df.index.append(df.index)
        return X, y, groups
    else:
        train_dev_set, test_set = train_test_split(df, train_size=0.9) #keep aligned pairs in split
        train_set, dev_set = train_test_split(train_dev_set, train_size=0.778)
        train_set_temp = pd.DataFrame(data={0: train_set[2], 1:train_set[3]})
        train_set = train_set.drop([2, 3], axis=1)
        train_set = pd.concat([train_set, train_set_temp], ignore_index=True)
        dev_set_temp = pd.DataFrame(data={0: dev_set[2], 1:dev_set[3]})
        dev_set = dev_set.drop([2, 3], axis=1)
        dev_set = pd.concat([dev_set, dev_set_temp], ignore_index=True)
        test_set_temp = pd.DataFrame(data={0: test_set[2], 1:test_set[3]})
        test_set = test_set.drop([2, 3], axis=1)
        test_set = pd.concat([test_set, test_set_temp], ignore_index=True)
        
        return train_set, dev_set, test_set

def preprocess_scraper_data(save=False):
    """
    Formats Wise Scraper exported data: requires CoreNLP server to tokenize in
    format compatible with pre-trained GloVe vectors (e.g. punctuations, ``,
    's, etc), lowercasing (for using uncased GloVe) and removing empty rows.
    Meant for pre-processing and saving (CoreNLP tokenization is slow).
    """
    parser = CoreNLPParser()
    df_scraper = pd.DataFrame()
    for file in os.listdir(data_home):
        if file.startswith('scraper_') and file.endswith('.json'):
            data = pd.read_json(os.path.join(data_home, file))
            data = data['text'].apply(lambda x: x.strip()).to_frame()
            data = data.drop(data[data['text'] == ''].index)
            data = data['text'].apply(lambda x: x.lower()).to_frame()
            data = data['text'].apply(lambda x: ' '.join(parser.tokenize(x))).to_frame()
            df_scraper = pd.concat([df_scraper, data], ignore_index=True)
    df_scraper[1] = 0 #19561 rows
    if save:
        df_scraper.to_csv(os.path.join(data_home, scraper_file), sep='\t', index=False, header=None)
    return df_scraper

def load_scraper_data():
    """
    Load saved Wise Scraper data extracted by preprocess_scraper_data
    """
    df_scraper = pd.read_csv(os.path.join(data_home, scraper_file), sep='\t', header=None)
    return df_scraper

def preprocess_uscode_data(max_rows=19561, save=False):
    """
    Formats US code XML data (http://uscode.house.gov/download/download.shtml),
    with manually selected titles and chapters to match most of the legal aid
    topics, while omitting repealed or omitted sections' content. Requires
    CoreNLP server to tokenize in format compatible with pre-trained GloVe
    vectors (e.g. punctuations, ``, 's, etc), lowercasing (for using uncased
    GloVe). Randomly samples from extracted content to match a maximum number
    of data rows, to balance amount of data against the legal aid scrapings.
    Meant for pre-processing and saving (CoreNLP tokenization is slow).
    """
    df_uscode = pd.DataFrame()
    prefix = '{http://xml.house.gov/schemas/uslm/1.0}'
    file_chapters = [('usc08.xml', ['11', '12', '14', '15']),
                     ('usc29.xml', ['5', '6', '8', '14', '20', '28']),
                     ('usc38.xml', ['11', '13', '15', '17', '18', '19', '20', '21', '31', '41', '42', '43']),
                     ('usc42.xml', ['5', '7', '7A', '8', '21', '32', '33', '34', '35', '45', '67', '75', '76', '94', '110', '114', '119', '126', '127', '130', '132', '135', '136', '144', '151']),
                     ('usc52.xml', ['103', '107', '201', '205'])]
    for file_chapter in file_chapters:
        doc = ET.parse(os.path.join(data_home, file_chapter[0])).getroot()
        title = doc.find(prefix + 'main').find(prefix + 'title')
        chapters = title.findall(prefix + 'chapter')
        for chapter in chapters:
            if chapter.find(prefix + 'num').get('value') not in file_chapter[1]:
                title.remove(chapter)
            else:
                subchapters = chapter.findall(prefix + 'subchapter')
                for subchapter in subchapters:
                    sections = subchapter.findall(prefix + 'section')
                    for section in sections:
                        if section.get('status') != None: #remove repealed and omitted sections
                            subchapter.remove(section)
        contents = title.iter(prefix + 'content')
        for content in contents:
            if content.find(prefix + 'p') is not None:
                df_uscode = pd.concat([df_uscode, pd.DataFrame([content.find(prefix + 'p').text])], ignore_index=True)
            elif content.text not in [None, '']:
                df_uscode = pd.concat([df_uscode, pd.DataFrame([content.text])], ignore_index=True)
    df_uscode = df_uscode.drop_duplicates()
    df_uscode = df_uscode.dropna()
    df_uscode = df_uscode.sample(n=max_rows, random_state=0)
    parser = CoreNLPParser()
    df_uscode = df_uscode[0].apply(lambda x: x.lower()).to_frame()
    df_uscode = df_uscode[0].apply(lambda x: ' '.join(parser.tokenize(x))).to_frame()
    df_uscode[1] = 1
    if save:
        df_uscode.to_csv(os.path.join(data_home, uscode_file), sep='\t', index=False, header=None)
    return df_uscode

def load_uscode_data():
    """
    Load saved US code data extracted by preprocess_uscode_data
    """
    df_uscode = pd.read_csv(os.path.join(data_home, uscode_file), sep='\t', header=None)
    return df_uscode

def load_legal_data(mode=''):
    """
    Processes the scraper and US Code data as a combined dataset. Splits the
    dataset into either for cross-validation or fixed split, including
    balancing the labels for each class in every split. For cross-validation,
    use scikit-learn's StratifiedShuffleSplit.
    """
    df_scraper = load_scraper_data()
    df_uscode = load_uscode_data()
    if mode == 'cv':
        X = pd.concat([df_scraper[0], df_uscode[0]], ignore_index=True)
        y = pd.concat([df_scraper[1], df_uscode[1]], ignore_index=True)
        return X, y
    else:
        sc_train_dev_set, sc_test_set = train_test_split(df_scraper, train_size=0.9)
        sc_train_set, sc_dev_set = train_test_split(sc_train_dev_set, train_size=0.778)
        us_train_dev_set, us_test_set = train_test_split(df_uscode, train_size=0.9)
        us_train_set, us_dev_set = train_test_split(us_train_dev_set, train_size=0.778)
        train_set = pd.concat([sc_train_set, us_train_set], ignore_index=True)
        dev_set = pd.concat([sc_dev_set, us_dev_set], ignore_index=True)
        test_set = pd.concat([sc_test_set, us_test_set], ignore_index=True)
        return train_set, dev_set, test_set

def load_glove(size):
    """
    Load pre-trained GloVe vectors and add tokens for OOV and padding.
    """
    word2id = {'$PAD':0, '$UNK':1}
    embed_matrix = []
    embed_matrix.append(np.zeros((size)))
    embed_matrix.append(np.random.randn(size))
    id_counter = len(word2id)
    with open(os.path.join(glove_home, 'glove.6B.' + str(size) + 'd.txt')) as fin:
        while True:
            try:
                line = next(fin)
                line = line.strip().split()
                word2id[line[0]] = id_counter
                embed_matrix.append(np.array(list(map(float, line[1:]))))
                id_counter += 1
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return word2id, np.array(embed_matrix)

def calc_stats(df):
    """
    Get statistics for number of tokens and characters for a dataframe.
    Assumes text column name is 0.
    """
    stats = pd.DataFrame()
    stats['tokens'] = df[0].apply(lambda x: len(str(x).split()))
    stats['chars'] = df[0].apply(lambda x: len(str(x)))
    print(stats.describe())

def get_minibatches(data, minibatch_size, shuffle=True, random_seed=None):
    """
    NOTE: minibatch functions code adapted from CS224N provided pset starter code
    
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True, random_seed=None):
    batches = [data[col].values for col in data]
    return get_minibatches(batches, batch_size, shuffle, random_seed)


#preprocess_scraper_data(save=True)
#preprocess_uscode_data(save=True)

#calc_stats(load_scraper_data())
#calc_stats(load_uscode_data())