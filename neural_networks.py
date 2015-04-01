# Random Forest implementation to match and compare results in R
# Is MultiThread or MultiCore? No
# Is GPGPU? No

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
 
train = pd.DataFrame.from_csv('data/train.csv')
test = pd.DataFrame.from_csv('data/test.csv')
target = [x[94] for x in train]
train = [x[1:93] for x in train] # Remove the id and target column from features
 
def make_submission(m, test, filename):
    preds = m.predict_porba(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')
    preds.save(filename)
 
 
rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
# fit the training data
print('fitting the model')
rf.fit(train, target)
# timestr = time.strftime("%m%d-%H%M%S")
# make_submission(m, test, './submissions/boosted-trees-'+timestr+'.csv')