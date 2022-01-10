import argparse
from numpy.lib.type_check import real
import pandas as pd

def label(fstring):
    inputDf = pd.read_csv(fstring)

    if str.startswith(fstring, 't'):
        label = 1
    else:
        label = 0
    
    labels = []
    for t in inputDf['title']:
        labels.append(label)
    
    inputDf['label'] = labels

    newDf = inputDf[['title','text','label']]
    ofstring = fstring.split('.')[0] + '_labeled.csv'
    newDf.to_csv(ofstring)
    print(newDf)

def clean(fstring):
    inputDf = pd.read_csv(fstring)

    inputDf['label'].replace({'REAL':'1', 'FAKE':'0'}, inplace=True)

    newDf = inputDf[['title','text','label']]

    ofstring = fstring.split('.')[0] + '_labeled.csv'
    newDf.to_csv(ofstring)
    print(newDf)

def merge(files):
    df1 = pd.read_csv(files[0])
    df2 = pd.read_csv(files[1])
    df3 = pd.read_csv(files[2])
    df4 = pd.read_csv(files[3])

    frames = [df1, df2, df3, df4]
    result = pd.concat(frames)
    result = result[['title', 'text', 'label']]
    result.to_csv('merged_labeled.csv')
    print(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='input file name')
    parser.add_argument('--files', help='merge input files', nargs='*')
    parser.add_argument('-l', help='add label to file', action='store_true')
    parser.add_argument('-c', help='keep just title, text, label', action='store_true')
    
    args = parser.parse_args()

    to_label = args.l
    to_clean = args.c
    to_merge = args.files
    fstring = args.file
    if to_label:
        label(fstring)
    elif to_clean:
        clean(fstring)
    else:
        merge(to_merge)

    # first run $python mergeData.py --file fake.csv -l
    # and $python mergeData.py --file true.csv -l
    # to label these articles
    
    # then run $python mergeData.py --file fake_news.csv -c
    # and $python mergeData.py --file news.csv -c
    # to keep just the correct columns and ensure labels are 0 or 1
    
    # lastly run 
    # $python mergeData.py --files fake_labeled.csv fake_news_labeled.csv news_labeled.csv true_labeled.csv
    # to merge all the files
    

if __name__ == '__main__':
    main()