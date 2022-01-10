import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

def clean_text(text):
    # set text to lowercase
    text = str(text).lower()
    # remove url
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove html
    text = re.sub('<.*?>+', '', text)
    # remove other unnecessary text
    text = re.sub('@\S+|https?:\S+|http?:\S', ' ', text)
    # remove newline
    text = re.sub('\n', '', text)
    # remove whitespace and non-words
    text =  re.sub('[^\w\s]','',text)
    # remove Reuters keyword
    text = re.sub('Reuters','',text)
    return text


def remove_stopwords(text):
    stopword_set = set(stopwords.words('english'))
    text = [word.lower() for word in text.split() if word.lower() not in stopword_set]

    return " ".join(text)

def lemmatize_words(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lem = ' '.join([wnl.lemmatize(word) for word in text.split()])
    return lem

def main():
    # nltk.download('wordnet')
    
    # read in files
    fake_df = pd.read_csv('./fake.csv')
    real_df = pd.read_csv('./true.csv')

    # concatenate title and text into one string
    fake_df['text'] = fake_df['title'] + fake_df['text']
    real_df['text'] = real_df['title'] + real_df['text']

    # assign labels (0 for fake, 1 for real) to data
    fake_df['label'] = 0
    real_df['label'] = 1

    # remove unnecessary columns
    fake_df.drop(['date', 'subject', 'title'], axis=1, inplace=True)
    real_df.drop(['date', 'subject', 'title'], axis=1, inplace=True)

    # merge the two datasets into one
    df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)

    # clean text
    df['text'] = df['text'].apply(lambda x:clean_text(x))

    # remove stopwords from text
    df['text'] = df['text'].apply(remove_stopwords)

    # lemmatize words in text
    df['text'] = df['text'].apply(lemmatize_words)

    # drop NaN rows
    # df.dropna(axis=0, how='any', inplace=True)

    dfX = df['text']
    dfY = df['label']

    # export datasets to csv
    df.to_csv('input_label.csv')
    dfX.to_csv('input.csv')
    dfY.to_csv('label.csv')

if __name__ == "__main__":
    main()