#-------------------------------------------------------------------------
# AUTHOR: Sanat Vankayalapati
# FILENAME: decision_tree.py
# SPECIFICATION: This program reads a CSV file (contact_lens.csv), encodes categorical data into numerical values,
# and generates a decision tree using the ID3 algorithm.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 days
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays.

# Importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

# Initialize lists to store data
db = []
X = []  # Features
Y = []  # Target labels

# Dictionary for encoding categorical values
age_map = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle_map = {"Myope": 1, "Hypermetrope": 2}
astigmatism_map = {"Yes": 1, "No": 2}
tear_map = {"Normal": 1, "Reduced": 2}
lens_map = {"Yes": 1, "No": 2}  # Target variable

# Reading the data from the CSV file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)

# Transform categorical features into numerical values and add them to X
for row in db:
    X.append([
        age_map[row[0]],
        spectacle_map[row[1]],
        astigmatism_map[row[2]],
        tear_map[row[3]]
    ])
    Y.append(lens_map[row[4]])  # Transforming class labels

# Fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# Plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
               class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
