# Random Forest implementation to match and compare results in R
# Is MultiThread or MultiCore? Yes
# Is GPGPU? No
from sklearn.metrics import classification_report
from sklearn import ensemble, feature_extraction, preprocessing
import pandas as pd
import time
import yaml
import sys
import psutil

startTime = time.time()

# load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('submissions/sampleSubmission.csv')
labels = train.target.values
ids = train.id.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
train_orig = train
test = test.drop('id', axis=1)


# configure
# check the models configuration required parameters and set default parameters
# ideally dont use defaults, always send through args
class Foo(object):
    pass


config = Foo()
config.estimators = 100
config.cores = psutil.cpu_count()
config.pc_owner = 'jd'
config.pc_type = 'mac'
config.pc_location = 'office'
config.os = sys.platform
config.pc_cores = psutil.cpu_count()

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
print('starting classification ... ')
clf = ensemble.RandomForestClassifier(n_jobs=config.cores, n_estimators=config.estimators)
clf.fit(train, labels)

# predict on test set
preds = clf.predict_proba(test)
train_pred = clf.predict(tfidf.transform(train_orig))
config.score = classification_report(train_pred, labels)

# create submission file
timestr = time.strftime("%m%d-%H%M%S")
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('./submissions/random-forest-' + timestr + '.csv', index_label='id')

# Write config data
data = dict(
    start_time=time.strftime("%b %d %Y %H:%M:%S", time.gmtime(startTime)),
    end_time=time.strftime("%b %d %Y %H:%M:%S"),
    elapsedSeconds=time.time() - startTime,
    config=config.__dict__
)
with open('./submissions/random-forest-' + timestr + '.yaml', 'w') as outfile:
    outfile.write(yaml.dump(data, default_flow_style=False))

print yaml.dump(data, allow_unicode=True, default_flow_style=False)

# use boto api to upload results, model, running configg and accuracy to S3 OR 
# push to git    