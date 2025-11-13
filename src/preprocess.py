import nltk
from nltk.tokenize import word_tokenize
import torch 
from torch.utils.data import DataLoader, random_split
from config import seed, batch_size, vocab_size
nltk.download('punkt_tab', quiet=True)

def tokenize_df(df):
    '''
    tokenizes entire dataframe and returns vocab
    '''
    df = df.to_list()
    vocab = {}
    tokenized_sentences = []
    for sentence in df:
        words = word_tokenize(sentence)
        for word in words:
            vocab[word] = vocab.get(word, 0) + 1
        tokenized_sentences.append(words)
    return tokenized_sentences, vocab

def preprocess(data):
    # lowercase each_review
    data = data.str.lower()
    # remove punctuation and special characters
    data = data.str.replace(r'[^\w\s]', '', regex=True)
    # take all sentences and form vocabulary and corresponding ids
    tokenized_data,vocab  = tokenize_df(data)   
    # keeping only 10k most freq words
    top_vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)[:vocab_size]
    # extracting only words from words and frequencies
    top_words_list = [x[0] for x in top_vocab]
    top_words = set(top_words_list)
    # forming a word to index dictionary
    word_to_id = {word: idx + 1 for idx, word in enumerate(top_words)}
    # filter the words with only  top 10k words
    filtered_sentences = []
    for tokenized_sentence in tokenized_data:
        filtered_words = [word for word in tokenized_sentence if word in top_words]
        filtered_sentences.append(filtered_words)
    # replace with ids
    sentence_ids = []
    for sentence in filtered_sentences:
        ids = [word_to_id[word] for word in sentence]
        sentence_ids.append(ids)
    return sentence_ids, word_to_id    

def pad_seq(sequences, maxlen):
    """
    Pads sequences to the same length.
    Args:
        sequences (list of list of int): A list of sequences of word IDs.
        maxlen (int): The maximum length to pad sequences to.
    Returns:
        list of list of int: Padded sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            # Pad with zeros at the end
            padded_seq = seq + [0] * (maxlen - len(seq))
        else:
            # Truncate if longer than maxlen
            padded_seq = seq[:maxlen]
        padded_sequences.append(padded_seq)
    return padded_sequences

def split_data(ds):
    # 50/50 split as per requirements (25k training, 25k testing)
    train_size = int(0.5 * len(ds))
    val_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


class IMDBDataset(torch.utils.data.Dataset):
    '''
    returns tensors as per calls
    '''
    def __init__(self, data_X, data_y):
        super().__init__()
        self.X = torch.load(data_X, weights_only = False)
        self.y = torch.load(data_y, weights_only = False)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]