# importing the Dataset
import pandas as pd
import joblib

messages =  pd.read_csv("Data/SMSSpamCollection",sep = '\t',names=['label','message'])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
""" here we have taken max_features=2500 because all the words are not that frequently
    repeated ,hence we have taken top 2500 words which needs to be convert into vector."""

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
Y = pd.get_dummies(messages['label'],drop_first = True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8)


from sklearn.naive_bayes import MultinomialNB
Spam_detect_model = MultinomialNB().fit(x_train,y_train)

y_pred = Spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
metrics = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


joblib.dump(Spam_detect_model,"model.pkl")
joblib.dump(cv,"Vectorizer.pkl")
