# Random Forest implementation to match and compare results in R
# Is MultiThread or MultiCore? Yes
# Is GPGPU? No
from sklearn.metrics import classification_report
from sklearn import ensemble, feature_extraction, preprocessing
import pandas as pd
import numpy as np
import time, scipy.sparse
import yaml
import sys
import psutil
import random

startTime = time.time()
def shuffle(df, n=1, axis=0):     
        df = df.copy()
       	for _ in range(n):
            df.apply(np.random.shuffle, axis=axis)
        return df
        
# load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('submissions/sampleSubmission.csv')

# shuffle before continuing
shuffle(train)

labels = train.target.values
ids = train.id.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
train_orig = train
test = test.drop('id', axis=1)
# TODO: use arguments to configure
# TODO: transfer shared code to initialization pythons script

# configure
# check the models configuration required parameters and set default parameters
# ideally dont use defaults, always send through args
class Foo(object):
    pass


config = Foo()
config.estimators = 800
config.cores = psutil.cpu_count()
config.pc_owner = 'jd'
config.pc_location = 'office'
config.os = sys.platform
config.pc_cores = psutil.cpu_count()
config.shffle = 'true'
config.learning_rate = 0.05
config.max_depth = 6
config.verbose = 1
config.comment ='Increased Max depth from last model'
config.name = './submissions/gradient-classifier-'

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# Dont shuffle without shuffling labels
# random.shuffle(train)

#

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
print('starting training ... ')
clf = ensemble.GradientBoostingClassifier( n_estimators=config.estimators, learning_rate=config.learning_rate,
max_depth=config.max_depth, verbose=config.verbose )
clf.fit(train, labels)

# predict on test set
print('starting prediction ... ')
preds = clf.predict_proba(test)
train_pred = clf.predict(tfidf.transform(train_orig).toarray())
config.score = classification_report(train_pred, labels)

# create submission file
timestr = time.strftime("%m%d-%H%M%S")
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv(config.name + timestr + '.csv', index_label='id')

# Write config data
data = dict(
    start_time=time.strftime("%b %d %Y %H:%M:%S", time.gmtime(startTime)),
    end_time=time.strftime("%b %d %Y %H:%M:%S"),
    elapsedSeconds=time.time() - startTime,
    config=config.__dict__
)
with open(config.name + timestr + '.yaml', 'w') as outfile:
    outfile.write(yaml.dump(data, default_flow_style=False))

print yaml.dump(data, allow_unicode=True, default_flow_style=False)

# use boto api to upload results, model, running configg and accuracy to S3 OR 
# push to git    