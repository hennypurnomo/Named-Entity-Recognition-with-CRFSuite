# Named Entity Recognition with Distant Learning

Unlike supervised and unsupervised learning, distant learning (distant supervision) is acquiring training data by extracting examples either positives or negatives rather than have exact label like supervised learning. 

Several advantages of distant supervision:
1. Leverage rich, reliable and unlimited hand-created (bootstrap) knowledge rather than have exact label for each data in supervised learning
2. Rich features (such as syntactic feature) can be used 
3. Does not sensitive to training corpus
4. Reduce the heavy cost such as time and effort to annotate the data

Therefore, this project employs distant supervision method with Wikipedia articles. In term of dataset for training, the system uses wikiner while for testing using wikigold. 

