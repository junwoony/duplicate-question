from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import pickle

# Export Training Data and dropout NaN rows
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
df = pd.read_csv('questions.csv')
df = df.dropna()


# Data Processing
# 1) Take unique words from questions
results = set()
df['question1'].str.lower().str.split().apply(results.update)
df['question2'].str.lower().str.split().apply(results.update)
# Now results is a bag of unique words
# 2) Store indices of unique words
idx2word = dict(enumerate(results))
word2idx = {word: idx for idx, word in idx2word.items()}

# Encode questions to index
def encode_question(question):
    words = question.split()
    encoded = []
    for word in words:
        encoded.append(word2idx[word])
    return encoded 

df['enc_question1'] = df.question1.str.lower().apply(encode_question)
df['enc_question2'] = df.question2.str.lower().apply(encode_question)


embedding_dim = 300
embeddings = 1 * np.random.randn(len(results) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in word2idx.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

filename = 'embeddings'
outfile = open(filename,'wb')
pickle.dump(embeddings,outfile)
outfile.close()

filename = 'df'
outfile = open(filename,'wb')
pickle.dump(df,outfile)
outfile.close()