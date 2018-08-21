# import machine learning module and relevant algorithm
from sklearn import tree

# collect training data
features = [ [140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"] ]
labels = ["apple", "apple", "orange", "orange"]

# clean training data
modified_features = [ [140, 1], [130, 1], [150, 0], [170, 0] ] # 0 = bumpy, 1 = smooth
modified_labels = [0, 0, 1, 1] # 0 = apple, 1 = orange

# train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(modified_features, modified_labels)

# make predictions
print(clf.predict([[160, 0]]))


