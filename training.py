print('Training Under Progress... \n')

from time import time 
import click
import pandas as pd
import pickle

import nltk # please download wordnet and stopwords if not already downloaded 

import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

import joblib
import dill


import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
import time

#Lemmatization


lemma=WordNetLemmatizer()


class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)]) 

    
stopwords_extra=['include','example','may','be','first','second']
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(stopwords_extra)

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re 
from bs4 import BeautifulSoup
from html import unescape

from sklearn.preprocessing import FunctionTransformer

lemma=WordNetLemmatizer()


class lemmatokenizer(object):
    def __call__(self, text):
        return [lemma.lemmatize(t) for t in word_tokenize(text) if t not in stopwords]
    
    
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




@click.command()
@click.argument("training_data", type=str)
@click.option("--keywords", type=str,default='keywords.csv')
@click.option("--model-file", type=str, default="model.pkl")
@click.option("--prediction-file", type=str, default="test_predictions.csv")

def main(
    training_data,
    keywords,
    model_file,
    prediction_file):
    
    df = pd.read_csv(training_data)
    keywords=pd.read_csv(keywords)
    List=list(keywords['keywords'])
    List=[x.strip() for x in List]

    import nltk

    stopwords=['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herse"', 'him', 'himse"', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itse"', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myse"', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']
    stopwords_extra=['.',',',';',":",")","(",'#','%','$','receiving','comprising','least','processor','method','system','includes','may','user','wherein','utilizing','whether','said','used','using']
    stopwords_2 = nltk.corpus.stopwords.words('english')
    stopwords.extend(stopwords_extra)
    stopwords.extend(stopwords_2)


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

    

    class ItemSelector(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, x, y=None):
            return self

        def transform(self, data_dict):
            return data_dict[self.key]


    def keyword_func(input_series):
    
        return np.array(input_series.tolist()).reshape(-1,1)

    keyword_transform = FunctionTransformer(keyword_func, validate=False)

    keyword_transformer = pl(steps=[
        ('selector', ItemSelector(key='keyword_occur')),
        ('converter', keyword_transform), 
        #('tfidf', TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True)),
    ])




    LCV= CountVectorizer(stop_words=stopwords,ngram_range=(1,3),max_df=0.5,min_df=2,max_features=200000, 
                             tokenizer=None)


    concat3_transformer = pl([
        ('selector', ItemSelector(key='concat3')),
        ('vectorizer', LCV), 
        ('tfidf', TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True)),
    ])




    feature_pipeline = FeatureUnion(
        transformer_list=[
     
        ('concat3_transformer', concat3_transformer),
        ('keyword_transformer', keyword_transformer),
       
         
    ])


    train,test=train_test_split(df,test_size=0.2,random_state=5)

    from time import time
   

    
    time0=time()
    train_data=feature_pipeline.fit_transform(train,train.label)
    test_data=feature_pipeline.transform(test)

    X_train3=train_data
    y_train3=train.label
    X_test3=test_data
    y_test3=test.label

    from sklearn.metrics import confusion_matrix


    text_ensemble_lemmatized2= Pipeline([#('vectorizer', LCV), ('tfidf', TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True)),
                                     #('RUS', SMOTE(k_neighbors=10,sampling_strategy='minority',random_state=1)),
                                     ('lr',LogisticRegression(C=1,penalty='l2',class_weight={0:1,1:1}))])

    #print('Training Under Progress \n')
   
    text_ensemble_lemmatized2.fit(X_train3,y_train3)

    predicted6=text_ensemble_lemmatized2.predict_proba(X_train3)
    predicted=np.where(predicted6[:,1]>0.5,1,0)
    y_score=predicted6[:,1]

    train['pred']=predicted
    train['pp']=y_score



    predicted6=text_ensemble_lemmatized2.predict_proba(X_test3)
    predicted=np.where(predicted6[:,1]>0.5,1,0)
    y_score=predicted6[:,1]

    test['pred']=predicted
    test['pp']=y_score

    print("F1:",round(f1_score(y_test3,predicted,average='binary'),3))
    print("Precision:",round(precision_score(y_test3,predicted,average='binary'),3))
    print("Recall:",round(recall_score(y_test3,predicted,average='binary'),3))
    print("ROC SCORE=",round(roc_auc_score(y_test3,y_score),3))

    print('\n')



    cm=confusion_matrix(y_test3,predicted)
    tn,fp,fn,tp=confusion_matrix(y_test3,predicted).ravel()
    spec=tn/(tn+fp)
    sens=tp/(tp+fn)
    prec= tp/(tp+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    f1=2/((1/sens)+(1/prec))


    print('TN',tn,'FP',fp)
    print('FN',fn, 'TP', tp)
    print('Training Successfully Completed in ', str(round(time()-time0,1))+' Seconds.', 'Please view test_predictions.csv file for predictions on Test Set')
    
    test[['title','abstract_text','claim_text','keyword_occur','label','pred','pp']].to_csv(prediction_file, index=False)

    filename = 'model.pkl'
    pickle.dump(text_ensemble_lemmatized2, open(filename, 'wb'))

    dill.dump(feature_pipeline, open( 'pipeline.pkl', 'wb'))
if __name__ == "__main__":
   # A little disconcerting, but click injects the arguments for you.
    main()

