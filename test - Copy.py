# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:44:01 2020

@author: Bilal Ahmad Amiri
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:32:33 2020

@author: Bilal Ahmad Amiri
"""
#%% Loading Libraries

import re
import nltk
import string
import warnings
import wordninja
import seaborn           as sns
import pandas            as pd
import matplotlib.pyplot as plt
import tweepy            as tw
import numpy             as np

from nltk.stem                       import WordNetLemmatizer
from nltk                            import pos_tag
from nltk.corpus                     import wordnet
from nltk.sentiment.vader            import SentimentIntensityAnalyzer
from wordcloud                       import WordCloud
from afinn                           import Afinn
from sklearn                         import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors               import KNeighborsClassifier 
from sklearn.naive_bayes             import GaussianNB
from dmba                            import regressionSummary, classificationSummary
from sklearn.linear_model            import LinearRegression
from sklearn.neural_network          import MLPClassifier
from sklearn.metrics                 import roc_curve, auc, roc_auc_score
from sklearn.metrics                 import precision_recall_curve


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download("stopwords")

warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

stop_words = set(nltk.corpus.stopwords.words('english'))
exclude_words = set(("no", "out", "not", "don't", "doesn't", "should'nt", "won't", "what", "will", "how"))
new_stop_words = stop_words.difference(exclude_words)
#%% Defining the functions for clearing the data

# Parts of Speech (PoS) tagging
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
#Cleaning the text
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = new_stop_words
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# wordcloud function
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
#%% Authorizing Twitter API to download tweets
#Giving the consumer key and consumer secret key for Twitter API
consumer_key          = 'hpYJO2RLJcp8Un04vrnvnRVfr'
consumer_secret       = 'CIQSmlGjvZd78yeRXhJP4x6FC9EpkNpqgRNvq1fSWpP7VwFGd5'
access_token          = '349627256-IxC0GzMi5vlz8Y8J7w9yGDLLWLz8TkUYluews4MT'
access_token_secret   = 'dQITZ3VZHT5mEG88gUu1V4jyQdgZNcH2dCTGRUsmN4e1e'
auth                  = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api                   = tw.API(auth, wait_on_rate_limit=True)

#%% Gathering the Data
tweets = tw.Cursor(api.search,
              q="#CovidVaccine -filter:retweets",
              lang="en",
              since="2020-11-03",
              tweet_mode="extended").items(5000) #.will return 5,000 latest tweets

list_of_tweets = []
for tweet in tweets:
	list_of_tweets.append(tweet.full_text)
list_of_tweets[:5]

#%%  1: Data Cleaning
#Removing urls
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

#Removing the urls from the tweets
list_of_tweets_2 = [remove_url(tweet) for tweet in list_of_tweets]
list_of_tweets_2[:5]

#Creating dataframe of the cleanest data
clean_tweets = pd.DataFrame(list_of_tweets_2, columns=['Tweets'])

#Splitting the words with no space in-between into two words
clean_tweets["Tweets_2"] = clean_tweets["Tweets"].apply(lambda x: wordninja.split(x))
separator = ' '
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].apply(lambda x: separator.join(x))

#Rejoining the words split accidentally
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('C ovid','Covid')
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('C OVID 19','COVID19')
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('c ovid','covid')
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('c ovid 19','covid19')
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('Covid','')
clean_tweets["Tweets_2"] = clean_tweets["Tweets_2"].str.replace('covid','')

#Adding the new column to existing data
clean_tweets["Tweets_Clean"] = clean_tweets["Tweets_2"].apply(lambda x: clean_text(x))

#Giving sentiments using nltk analyzer
sid = SentimentIntensityAnalyzer()
clean_tweets["sentiments"] = clean_tweets["Tweets_2"].apply(lambda x: sid.polarity_scores(x))
clean_tweets = pd.concat([clean_tweets.drop(['sentiments'], axis=1), clean_tweets['sentiments'].apply(pd.Series)], axis=1)

# Sentiment analysis using AFINN
afn = Afinn(emoticons=True)
clean_tweets["afinn"] = clean_tweets["Tweets_2"].apply(lambda x: afn.score(x))
clean_tweets["label_afinn"] = clean_tweets["afinn"].apply(lambda x: 'negative' if x < 0  else ('neutral' if x == 0 else 'positive'))
clean_tweets["label_vs_affin"] = clean_tweets["label"]== clean_tweets["label_afinn"]

#Flagging the negative reviews
clean_tweets["label"] = clean_tweets["compound"].apply(lambda x: 'negative' if x < 0  else ('neutral' if x == 0 else 'positive'))

# add number of characters column
clean_tweets["nb_chars"] = clean_tweets["Tweets_2"].apply(lambda x: len(x))

# add number of words column
clean_tweets["nb_words"] = clean_tweets["Tweets_2"].apply(lambda x: len(x.split(" ")))

#Flagging the negative reviews
clean_tweets["is_negative"] = clean_tweets["label"].apply(lambda x: 1 if x == 'negative' else 0)
clean_tweets["is_negative_afinn"] = clean_tweets["label_afinn"].apply(lambda x: 1 if x == 'negative' else 0)

#deleting tweets with less than three words
clean_tweets.drop(clean_tweets.index[clean_tweets['nb_words'] < 3], inplace = True)

#removing unwanted tweets
clean_tweets = clean_tweets[~clean_tweets["Tweets_Clean"].str.contains('pakistan|india|cricket|minister|modi')]

#exporting the data
clean_tweets.to_csv(r'D:\3- Bilal Ahmad MBA\4 - Fall 2020\Marketing Analysis\Final Project\clean_tweets_2.csv', index = False)

#exporting the data
clean_tweets.to_csv(r'D:\3- Bilal Ahmad MBA\4 - Fall 2020\Marketing Analysis\Final Project\df_2.csv', index = False)

#loading data
clean_tweets = pd.read_csv(r'D:\3- Bilal Ahmad MBA\4 - Fall 2020\Marketing Analysis\Final Project\df.csv')

#%%
# print wordcloud
show_wordcloud(clean_tweets["Tweets_Clean"])

# highest positive sentiment reviews (with more than 5 words)
clean_tweets[clean_tweets["nb_words"] >= 5].sort_values("pos", ascending = False)[["Tweets_2", "pos"]].head(10)

# lowest negative sentiment reviews (with more than 5 words)
clean_tweets[clean_tweets["nb_words"] >= 5].sort_values("neg", ascending = False)[["Tweets_2", "neg"]].head(10)

#Finding the number of tweets flagged negative by each algorithm
clean_tweets["is_negative"].value_counts(normalize = True)
clean_tweets["is_negative_afinn"].value_counts(normalize = True)

#Further cleaning the data by removing unwanted and duplicated tweets
clean_tweets = clean_tweets[~clean_tweets["Tweets_Clean"].str.contains('ministry|modi')]
clean_tweets = clean_tweets.drop_duplicates()

#Getting the Y-Variable
clean_tweets["is_negative_y"] = clean_tweets["sentiment"].apply(lambda x: 1 if x == 'negative' else 0)

#Removing the unneeded columns
clean_tweets = clean_tweets.drop(['neg', 'neu', 'pos', 'compound', 'label', 'label_afinn', 'label_vs_affin', 'afinn', 'Tweets',
                                  'nb_chars', 'nb_words', 'is_negative_afinn', 'is_negative', 'sentiment'], axis = 1)


# %%% 2. Data splitting
clean_tweets['ML_group']   = np.random.randint(100,size = clean_tweets.shape[0])
clean_tweets               = clean_tweets.sort_values(by='ML_group')
inx_train                  = clean_tweets.ML_group<80                     
inx_valid                  = (clean_tweets.ML_group>=80)&(clean_tweets.ML_group<90)
inx_test                   = (clean_tweets.ML_group>=90)
   

# %%% 3. Putting structure in the text
corpus          = clean_tweets.Tweets_Clean.to_list()
ngram_range     = (1,1)
max_df          = 0.85
min_df          = 0.01
vectorizer      = CountVectorizer(lowercase   = True,
                                  ngram_range = ngram_range,
                                  max_df      = max_df     ,
                                  min_df      = min_df     );
                                  
X               = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())

# %%% 4. Performing the TVT - SPLIT
Y_train   = clean_tweets.is_negative_y[inx_train].to_list()
Y_valid   = clean_tweets.is_negative_y[inx_valid].to_list()
Y_test    = clean_tweets.is_negative_y[inx_test].to_list()

X_train   = X[np.where(inx_train)[0],:]
X_valid   = X[np.where(inx_valid)[0],:]
X_test    = X[np.where(inx_test) [0],:]

X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]

print('Training   : ',   X_train.shape)
print('Validation : ',   X_valid.shape)
print('Test       : ',    X_test.shape)

#%%
# %%% 5. Sentiment analysis using linear regression
clf_linear  = LinearRegression().fit(X_train, Y_train)

clean_tweets['is_negative_reg'] = np.concatenate(
        [
                clf_linear.predict(X_train),
                clf_linear.predict(X_valid),
                clf_linear.predict(X_test )
        ]
        ).round().astype(int)

#printing the results
regressionSummary(Y_train, clf_linear.predict(X_train))
regressionSummary(Y_valid, clf_linear.predict(X_valid))
regressionSummary(Y_test,  clf_linear.predict (X_test))

#%%%% Modelling as KNN

k            = 4;
results_list = [];
max_k_nn     = 8
for k in range(1,max_k_nn):
    clf_knn      = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    results_list.append(
            np.concatenate(
                    [
                            clf_knn.predict(X_train),
                            clf_knn.predict(X_valid),
                            clf_knn.predict(X_test)
                    ])
    )
    print('one more done')
    
#Transforming teh result into a dataframe 
dta_results_knn                 = pd.DataFrame(results_list).transpose()
dta_results_knn['is_negative']  = clean_tweets.is_negative_y.copy()

#Printing the confusion Matrix for KNN training and validating datasets
classificationSummary(Y_train, clf_knn.predict(X_train))
classificationSummary(Y_valid, clf_knn.predict(X_valid))
classificationSummary(Y_test,  clf_knn.predict (X_test))

# %%% Modeling using Naive Bayes Classification
clf_nb                       = GaussianNB().fit(X_train.toarray(), Y_train)
result_list                  = []
result_list.append(
    np.concatenate(
        [
                clf_nb.predict(X_train.toarray()),
                clf_nb.predict(X_valid.toarray()),
                clf_nb.predict (X_test.toarray())
        ]).round().astype(int)
)
dta_results_nb                 = pd.DataFrame(result_list).transpose()
dta_results_nb['is_negative']  = clean_tweets.is_negative_y.copy()

#printing the confusion matrix
classificationSummary(Y_train, clf_nb.predict(X_train.toarray()))
classificationSummary(Y_valid, clf_nb.predict(X_valid.toarray()))
classificationSummary(Y_test,  clf_nb.predict (X_test.toarray()))

# %%% 8. Sentiment analysis using trees
criterion_chosen     = ['entropy','gini'][1]
random_state         = 96
max_depth            = 10
results_list         = []
for depth in range(2,max_depth):
    clf_tree    = tree.DecisionTreeClassifier(
            criterion    = criterion_chosen, 
            max_depth    = depth,
            random_state = 96).fit(X_train.toarray(), Y_train)

    results_list.append(
            np.concatenate(
                    [
                            clf_tree.predict(X_train.toarray()),
                            clf_tree.predict(X_valid.toarray()),
                            clf_tree.predict(X_test.toarray( ))
                    ]).round().astype(int)
            )
    
tree.plot_tree(clf_tree) 

dta_results_tree              = pd.DataFrame(results_list).transpose()
dta_results_tree['inx_train'] = inx_train.to_list()
dta_results_tree['inx_valid'] = inx_valid.to_list()
dta_results_tree['inx_test']  = inx_test.to_list()

#printing the confusion matrix
classificationSummary(Y_train, clf_tree.predict(X_train.toarray()))
classificationSummary(Y_valid, clf_tree.predict(X_valid.toarray()))
classificationSummary(Y_test,  clf_tree.predict (X_test.toarray()))

# %%% 9. Sentiment analysis using neural networks
# train neural network with 2 hidden nodes
from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train, with_mean = False)
X_valid_scaled = preprocessing.scale(X_valid, with_mean = False)
X_test_scaled  = preprocessing.scale (X_test, with_mean = False)

clf_nn = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs',
                    random_state=1)

clf_nn.fit(X_train_scaled, Y_train)

result_list                  = []
result_list.append(
    np.concatenate(
        [
                clf_nn.predict(X_train_scaled),
                clf_nn.predict(X_valid_scaled),
                clf_nn.predict (X_test_scaled)
        ]).round().astype(int)
)
dta_results_nn                 = pd.DataFrame(result_list).transpose()
dta_results_nn['is_negative']  = clean_tweets.is_negative_y.copy()

# training performance (use idxmax to revert the one-hot-encoding)
classificationSummary(Y_train, clf_nn.predict(X_train_scaled))
classificationSummary(Y_valid, clf_nn.predict(X_valid_scaled))
classificationSummary(Y_test, clf_nn.predict  (X_test_scaled))

#%%Visulaization
df = pd.read_csv(r'D:\3- Bilal Ahmad MBA\4 - Fall 2020\Marketing Analysis\Final Project\df.csv')
for x in [0, 1]:
    subset = df[df['is_negative'] == x]
    # Draw the density plot
    if x == 0:
        label = "Good reviews"
    else:
        label = "Bad reviews"
    sns.distplot(subset['compound'], hist = False, label = label)

#ROC for models
y_pred_knn = [x[1] for x in clf_knn.predict_proba(X_test)]
y_pred_nb = [x[1] for x in clf_nb.predict_proba(X_test.toarray())]
y_pred_tree = [x[1] for x in clf_tree.predict_proba(X_test)]
y_pred_nn = [x[1] for x in clf_nn.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_nn, pos_label = 1)
roc_auc = auc(fpr, tpr)
plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label= 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Neural Nets')
plt.legend(loc="lower right")
plt.show()

#ROC AUC Score for all models
roc_auc_knn = roc_auc_score(Y_test, y_pred_knn)
roc_auc_nb = roc_auc_score(Y_test, y_pred_nb)
roc_auc_tree = roc_auc_score(Y_test, y_pred_tree)
roc_auc_nn = roc_auc_score(Y_test, y_pred_nn)
print('KNN ROC AUC %.3f' % roc_auc_knn)
print('Naive Bayes ROC AUC %.3f' % roc_auc_nb)
print('D-Tree ROC AUC %.3f' % roc_auc_tree)
print('Neural Nets ROC AUC %.3f' % roc_auc_nn)

#Precision and Recalls
precision_knn, recall_knn, _ = precision_recall_curve(Y_test, y_pred_knn)
auc_score_knn = auc(recall_knn, precision_knn)
precision_nb, recall_nb, _ = precision_recall_curve(Y_test, y_pred_nb)
auc_score_nb = auc(recall_nb, precision_nb)
precision_tree, recall_tree, _ = precision_recall_curve(Y_test, y_pred_tree)
auc_score_tree = auc(recall_tree, precision_tree)
precision_nn, recall_nn, _ = precision_recall_curve(Y_test, y_pred_nn)
auc_score_nn = auc(recall_nn, precision_nn)
print('KNN ROC AUC %.3f' % auc_score_knn)
print('Naive Bayes ROC AUC %.3f' % auc_score_nb)
print('D-Tree ROC AUC %.3f' % auc_score_tree)
print('Neural Nets ROC AUC %.3f' % auc_score_nn)





