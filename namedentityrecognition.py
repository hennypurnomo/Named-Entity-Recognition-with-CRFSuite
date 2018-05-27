# This script is based on  the sklearn_crfsuite tutorial by Mikhail Korobov
# Loading the wikiner dataset is inspired and print the detail precision and recall metric by http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
# Removing any blank lines and https://codereview.stackexchange.com/questions/145126/open-a-text-file-and-remove-any-blank-lines
# The main code is from lab 6 - Named Entity Recognition, Text Analytics couse from University od Essex
# The loading data on testing part is from Rabia Yasa 1700421


import nltk
import sklearn
import sklearn_crfsuite

from nltk import pos_tag, word_tokenize
from sklearn.metrics import make_scorer
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# read data from wikiner, storing each sentences including 
# its element (word, pos tag, label) into a list
with open('aij-wikiner-en-wp2.txt', encoding='utf-8') as text:
    training = []
    data = text.read()

    # split sentence based on \n character
    for line in data.split("\n"):
        sents = []

        # skip for any blank lines
        if line.strip():

            # split each annotated word by space
            for w in line.split(" "):
                word = []

                # split word, posttag and label
                for n in w.split("|"):
                    n.replace("|",",")
                    word.append(n)
                sents.append(word)
            training.append(sents)

# read data from wikigold, storing each pair of word and label
# append pos tag for each element and storing into a list
with open('wikigold.conll.txt', encoding='utf-8') as test:
    all_x = []
    test_data = []
    
    # produce pairs of word and its label
    # skipping some unnecessary lines in the beggining of file
    for line in test:
        #read and append elements per line
        element = line.strip().split(' ')
        test_data.append(element)
        
        #removing unnecessary parts
        if line == '\n':
	    #append except the last character
            all_x.append(test_data[:-1])
            test_data = []
    test_data = test_data[:-1]
    
    # produce list of words with its label and pos tag
    testing=[]
    temp_list=[]
    
    for i in test_data:

        # append the elements per sentence into list
        if i[0]=="\\":
            testing.append(temp_list)
            temp_list = []
            continue

        # add pos tag and replace the \\ in every element in list
        else:
            word=(i[0])
            postag = pos_tag(word_tokenize(word))
            lab=str(i[1])
            label=(lab.replace("\\", ""))
            temp=str(word)+" "+str(postag[0][1]) + " " + str(label)
        temp_list.append(tuple(temp.split()))

# store lists into training set and testing set
train_sents = (training)
test_sents = (testing)

# define the features
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    # define features based on current word
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        }

    # define features based on previous word,
    # if it is not the beginning of a sentence
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            })
    else:
        # indicate it is the beginning of sentence
        features['BOS'] = True

    # define features based on next word,
    # if it is not the last of a sentence
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            })
    else:
        # indicate it is the beginning of sentence
        features['EOS'] = True

    return features

# extract sentence into features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# extract sentence into label
def sent2labels(sent):
    return [label for token, postag, label in sent]

# extract sentence into token or word
def sent2tokens(sent):
    return [token for token, postag, label in sent]

# extract features from the training set as well as testing
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# training a model
crf = sklearn_crfsuite.CRF(
    algorithm='pa',
    c=1,
    error_sensitive=True,
    averaging=True,
    max_iterations=100,
    epsilon=1e-5
)
crf.fit(X_train, y_train)

# predict the data on model (after removing O label)
labels = list(crf.classes_)
labels.remove('O')
y_pred = crf.predict(X_test)

# print f1 on the model
print("F1 score: ", metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))
# print accuracy on the model
print("Accuracy:", metrics.flat_accuracy_score(y_test, y_pred))


#print classification report
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

