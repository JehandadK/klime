# Random Forest implementation to match and compare results in R
# Is MultiThread or MultiCore? No
# Is GPGPU? No

from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble, feature_extraction, preprocessing
import pandas as pd
import numpy as np
import time
import yaml
startTime = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('submissions/sampleSubmission.csv')
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)
 

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
print('starting classification ... ')
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
clf.fit(train, labels)

# predict on test set
preds = clf.predict_proba(test)

# create submission file
timestr = time.strftime("%m%d-%H%M%S")
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('./submissions/random-forest-'+timestr+'.csv', index_label='id')

# Write config data
data = dict(
	start_time = time.strftime("%b %d %Y %H:%M:%S", startTime),
    end_time = time.strftime("%b %d %Y %H:%M:%S"),
    elapsedSeconds = time.time() - startTime
    
)
with open('./submissions/random-forest-'+timestr+'.yaml', 'w') as outfile:
    outfile.write( yaml.dump(data, default_flow_style=True) )