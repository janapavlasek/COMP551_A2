from sklearn import svm
from sklearn.model_selection import cross_val_score
from clean_data import CleanData
import numpy as np
import csv

cd = CleanData(tfidf=True,max_features=20000)

print 'Getting Training data.'
training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")
print 'Done collecting data.'

X = [x[1] for x in training_data]
y = [y[2] for y in training_data]

print 'Training the model.'
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
print 'Done training.'

print 'Cross Validation'
c_validation = cross_val_score(lin_clf, X, y, scoring='accuracy')
print c_validation.mean()


print 'Collecting test data.'
test_data = cd.get_x_in()
test = [x[1] for x in test_data]
print 'Done collecting data.'


print 'Predicting results.'
out = lin_clf.predict(test)
print 'Done predicitng.'

out = np.array(out).tolist()

for i in range(0,len(out)):
     if out[i] == 0:
            out[i] = 'hockey'
     if out[i] == 1:
             out[i] = 'movies'
     if out[i] == 2:
             out[i] = 'nba'
     if out[i] == 3:
             out[i] = 'news'
     if out[i] == 4:
             out[i] = 'nfl'
     if out[i] == 5:
             out[i] = 'politics'
     if out[i] == 6:
             out[i] = 'soccer'
     if out[i] == 7:
             out[i] = 'worldnews'

print out[0:5]

print 'Writing results into CSV file'
with open('svm_predictions.csv','wb') as result:
     wr = csv.writer(result)
     for item in out:
         wr.writerow([item])

print 'Completed successfully'
