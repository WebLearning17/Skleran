import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

data_path = '../data/'

y = np.load(data_path + 'label.npy')
X_egemaps = np.load(data_path + 'audio_features_egemaps_statistic.npy')
X_mfcc = np.load(data_path + 'audio_features_mfcc_statistic.npy')
X_xbow = np.load(data_path + 'audio_features_xbow_statistic.npy')
X_statistics = np.load(data_path + 'opensmile_statistics_norm.npy')

mapping = {'anxious': 0, 'worried': 1, 'angry': 2, 'surprise': 3, 'happy': 4, 'sad': 5, 'disgust': 6, 'neutral': 7}

# Params
# features = (X_egemaps, X_mfcc, X_xbow, X_statistics)
features = (X_egemaps, X_mfcc, X_xbow, X_statistics)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 20]},
                    {'kernel': ['linear'], 'C': [1, 10, 20]}]


X = np.concatenate(features, axis=1)
print(X.shape, y.shape)

X_train = X[:4917, :]
y_train = y[:4917]
X_test = X[4917:, :]
y_test = y[4917:]

clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring='precision_macro', n_jobs=20)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
predict = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predict))
print("Precision:", precision_score(y_test, predict, average='macro'))
print("Recall:", recall_score(y_test, predict, average='macro'))
print("F1 Score:", f1_score(y_test, predict, average='macro'))
print(confusion_matrix(y_test, predict))
