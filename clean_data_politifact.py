import pandas as pd
from clean_data import *

def main():
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # read from the csv file
    df = pd.read_csv('data_from_kaggle/politifact.csv')

    # df['fact'].unique() gives:
    # ['false', 'pants-fire', 'true', 'barely-true', 'half-true',
    # 'mostly-true', 'full-flop', 'no-flip', 'half-flip']
    # we choose 'true' and 'mostly-true' to label as 1 (true),
    # and 'false', 'pants-fire' and 'barely-true' as 0 (false).
    df_true = df[(df['fact'] == 'true') | (df['fact'] == 'mostly-true')]
    df_false = df[(df['fact'] == 'false') | (df['fact'] == 'pants-fire') | (df['fact'] == 'barely-true')]
    df_true['label'] = 1
    df_false['label'] = 0

    # drop unnecessary columns
    df_true.drop(['Unnamed: 0', 'sources', 'sources_dates',
                  'sources_post_location', 'curator_name',
                  'curated_date', 'fact', 'sources_url',
                  'curators_article_title', 'curator_complete_article',
                  'curator_tags'],
                  axis=1, inplace=True)
    df_false.drop(['Unnamed: 0', 'sources', 'sources_dates',
                   'sources_post_location', 'curator_name',
                   'curated_date', 'fact', 'sources_url',
                   'curators_article_title', 'curator_complete_article',
                   'curator_tags'],
                   axis=1, inplace=True)

    # concatenate the two df
    df_out = pd.concat([df_true, df_false], ignore_index=True, sort=False)

    # clean text
    df_out['sources_quote'] = df_out['sources_quote'].apply(lambda x:clean_text(x))

    # remove stopwords from text
    df_out['sources_quote'] = df_out['sources_quote'].apply(remove_stopwords)

    # lemmatize words in text
    df_out['sources_quote'] = df_out['sources_quote'].apply(lemmatize_words)

    # drop NaN rows
    df_out.dropna(axis=0, how='any', inplace=True)

    # output to a csv file
    df_out.to_csv('politifact_labeled.csv')

if __name__ == "__main__":
    main()