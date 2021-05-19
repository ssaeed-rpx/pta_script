print('Prediction Under Progress... \n')

from time import time 
import click
import pandas as pd
import pickle

import nltk # please download wordnet and stopwords if not already downloaded 

import pandas as pd
import numpy as np

from nltk.corpus import wordnet
#from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

import pickle
import dill
import re
from nltk import word_tokenize


import warnings
warnings.filterwarnings("ignore")


lemma=WordNetLemmatizer()




    
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as pl
from sklearn.svm import SVC

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def keyword_func(input_series):
    
    return np.array(input_series.tolist()).reshape(-1,1)


import nltk

@click.command()
@click.argument("prediction_data", type=str)
@click.argument("keywords", type=str,default='keywords.csv')
@click.argument("model_file", type=str, default="model.pkl")
@click.argument("pipeline_file", type=str, default="pipeline.pkl")
@click.option("--version", type=str, default="1")
@click.option("--prediction-file", type=str, default="predictions.csv")

def main(
    prediction_data,
    keywords,
    model_file,
    pipeline_file,
    version,
    prediction_file):

    
    import nltk


    stopwords2=['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herse"', 'him', 'himse"', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itse"', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myse"', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']
    stopwords_extra=['.',',',';',":",")","(",'#','%','$','receiving','comprising','least','processor','method','system','includes','may','user','wherein','utilizing','whether','said','used','using']
    stopwords_2 = nltk.corpus.stopwords.words('english')
    stopwords2.extend(stopwords_extra)
    stopwords2.extend(stopwords_2)


    
    
    text_ensemble_lemmatized2 = pickle.load(open(model_file, 'rb'))
    feature_pipeline = dill.load(open(pipeline_file, 'rb'))

    if version=='1':
        df = pd.read_csv(prediction_data)
    else:
        df=pd.read_csv(prediction_file)
        
    keywords=pd.read_csv(keywords)
    List=list(keywords['keywords'])
    List=[x.strip() for x in List]

    import nltk

        
    def split_it(line):
    
        line2=re.sub('\b\w{1,3}\b|<.*?>|((\d+)[\.])|\((.+?)\)|[0-9]+'," ",str(line))
    
        return line2

    df['clean_descr']=df['descr'].apply(split_it)
    df['descr_300']=df.clean_descr.apply(lambda x:  ' '.join(str(x).split()[0:300]))
   
    
    df['concat2']=10*df['title'].str.pad(width=150,side='both')+' '+df['abstract_text']+' '+df['claim_text']+' '+1*( df['descr_300']+ ' ')
    df['concat3']=df['concat2'].apply(lambda x: str(x).lower())
    
    df['concat']=df['title']#+' '+df['abstract_text']+' '+df['claim_text']


    

    def t(row):

        l=re.compile('|'.join(List))

        if (len(l.findall(row.concat.lower()))==0):
            return False

        else:
            return True

    
    
    df['keyword_occur']=df.apply(t,1)

    

    time0=time()
    df_data=feature_pipeline.transform(df)

  
    predicted6=text_ensemble_lemmatized2.predict_proba(df_data)
    predicted=np.where(predicted6[:,1]>0.5,1,0)
    y_score=predicted6[:,1]

    df['pred_'+version]=predicted
    df['pp_'+version]=y_score

    Features=feature_pipeline.transformer_list[0][1].named_steps['vectorizer'].get_feature_names()


    def top_tfidf_feats(row, features, top_n=3):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return top_feats


    df_data2=df_data[:,:-1]


    def top_feats_in_doc(df):
    
        az=[]
        for row in range(len(df)):
            row = np.squeeze(df_data2[row].toarray())
            az.append(top_tfidf_feats(row, Features, 3))
        
        return az

    if version=='1':

        df['top_feats']=top_feats_in_doc(df)


    

    print('Prediction Successfully Completed in ', str(round(time()-time0,1))+' Seconds.', 'Please view predictions.csv file for predictions on provided set')
    print("{} Positive Predictions in set out of total {}".format(len(df[df['pred_'+version]==1]),len(df)))
    

    if version=='1':

        df.drop('issue_date',1,inplace=True)

    df.drop(['clean_descr','descr_300','concat2','concat3','concat','keyword_occur'],1,inplace=True)

    new_cols = [col for col in df.columns if col != 'top_feats'] + ['top_feats']
    df = df[new_cols]

   
    
    df.to_csv(prediction_file, index=False)

    #df[['title','abstract_text','claim_text','pred_'+version,'pp_'+version,'top_feats']].to_csv(prediction_file, index=False)

    
if __name__ == "__main__":
   # A little disconcerting, but click injects the arguments for you.
    main()






