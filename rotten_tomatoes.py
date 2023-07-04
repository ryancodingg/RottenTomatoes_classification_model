import pandas as pd
import numpy as np 
import csv
import itertools
import matplotlib_inline
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import compute_class_weight
from sklearn.tree import DecisionTreeClassifier


df_review = pd.read_csv("C:/code/dataset/rotten_tomatoes_critic_reviews.csv")
df_movie = pd.read_csv("C:/code/dataset/rotten_tomatoes_movies.csv")
df_movie.head()


#Content rating visualization
print(f'Content Rating Category:{df_movie.content_rating.unique()}')
ax = df_movie.content_rating.value_counts().plot(kind ='bar', color ='orange', figsize = (9,6))
ax.bar_label(ax.containers[0], color ='white')
content_rating = pd.get_dummies(df_movie.content_rating, dtype=int)

print(f'Audience status category:{df_movie.audience_status.unique()}')
ax = df_movie.audience_status.value_counts().plot(kind='barh',color = 'orange', fontsize=12)
plt.show()


#Encoding audience status variable with ordinal encoding
audience_status = pd.DataFrame(df_movie.audience_status.replace(['Spilled', 'Upright'], ['0', '1']))
audience_status.head()

#Encode tomatometer status variable with ordinal encoding
tomatometer_status = pd.DataFrame(df_movie.tomatometer_status.replace(['Rotten', 'Fresh', 'Certified-Fresh'], ['0', '1', '2']))
tomatometer_status.head()

df_feature = pd.concat([df_movie[['runtime', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 'audience_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count']], content_rating, audience_status, tomatometer_status], axis=1).dropna()
df_feature.head()
df_feature.describe()
len(df_feature)

#Audience status visualizing
ax = df_feature.tomatometer_status.value_counts().plot(kind = 'bar', color = 'orange', figsize = (9,6))
ax.bar_label(ax.containers[0], color = 'white')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(df_feature.drop(['tomatometer_status'], axis=1), df_feature.tomatometer_status, test_size = 0.2, random_state = 42)
print(f'Size of training data is {len(x_train)} and the size of test data is {len(x_test)}')
tree_3_leaf = DecisionTreeClassifier(max_leaf_nodes =3, random_state=2)
tree_3_leaf.fit(x_train, y_train)
y_predict = tree_3_leaf.predict(x_test)
rint(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

y_pred = tree_3_leaf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(12, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_3_leaf.classes_)
disp.plot(ax= ax, cmap='cividis')
plt.show()

# the decision tree classifier stored in tree_3_leaf
fig, ax = plt.subplots(figsize=(12, 9))
tree.plot_tree(tree_3_leaf, ax=ax)
plt.show()

# Instantiate DecisionTreeClassifier with default hyperparameter settings
dt_classifier = DecisionTreeClassifier(random_state=2)

# Train the classifier on the training data
dt_classifier.fit(x_train, y_train)

# Predict the test data with the trained classifier
y_predict = dt_classifier.predict(x_test)

# Calculate accuracy and print classification report on the test data
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_test, y_predict)
class_labels = sorted(set(y_test) | set(y_predict))

fig, ax = plt.subplots(figsize=(12, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(ax=ax, cmap='cividis')
plt.show()

#Merge review dataframe and movie dataframe
df_merged = df_review.merge(df_movie, how='inner', on=['rotten_tomatoes_link'])
df_merged.head(5)
df_merged = df_merged[['rotten_tomatoes_link', 'movie_title', 'review_type', 'review_score', 'review_content', 'tomatometer_status']]
df_merged.head()

#Drop entries with missing reviews
df_merged = df_merged.dropna(subset=['review_content'])
#Plot distibution of the review
ax = df_merged.review_type.value_counts().plot(kind = 'bar',color ='orange' ,figsize=(9,6))
ax.bar_label(ax.containers[0], color ='white')

#Pick only 5000 entries from the original dataset
df_sub = df_merged[0:5000]
#Encode the label 
review_type = pd.DataFrame(df_sub.review_type.replace(['Rotten', 'Fresh'], ['0', '1']))
#Build final dataframe
df_feature_critics = pd.concat([df_sub['review_content'], review_type], axis =1).dropna()

#Split data into training data and testing data
x_train, x_test, y_train, y_test  = train_test_split(df_feature_critics['review_content'], df_feature_critics['review_type'], test_size=0.2, random_state = 42)
#We need to convert these strings into a format that can be used by a machine learning algorithm. This process is called tokenization in natural leanguage processing (NLP)

#Instantiatiate vectorizer class
vectorizer = CountVectorizer(min_df =1)
#Transform the text data into vector
x_train_vec = vectorizer.fit_transform(x_train).toarray()
#Initialize random forest and train it
rf = RandomForestClassifier(random_state =2)
rf.fit(x_train_vec, y_train)
#Predict and output classification report 
y_predicted = rf.predict(vectorizer.transform(x_test).toarray())

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

fig,ax = plt.subplots(figsize=(12,9))
im = ax.imshow(cm, interpolation='nearest', cmap='cividis')
ax.figure.colorbar(im, ax=ax)
classes = np.unique(np.concatenate((y_test, y_predicted), axis=None))


# Add labels to the plot
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes, rotation=45)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes)

# Add text annotations
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="black" if cm[i, j] > thresh else "white")

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()

#Improve model performance by including class weights using the compute_class_weight function from scikit-learn
#Calculate class weight

class_weight = compute_class_weight(class_weight = 'balanced', classes =np.unique(df_feature_critics.review_type), y = df_feature_critics.review_type.values)
class_weight_dict = dict(zip(range(len(class_weight.tolist())), class_weight.tolist()))
class_weight_dict

label_mapping = {label: i for i, label in enumerate(['0', '1'])}
y_train_int = np.array([label_mapping[label] for label in y_train])
y_test_int = np.array([label_mapping[label] for label in y_test])

# Instantiate vectorizer class
vectorizer = CountVectorizer(min_df=1)

# Transform the text data into vectors
x_train_vec = vectorizer.fit_transform(x_train).toarray()

# Get unique class labels from training data
class_labels = np.unique(y_train_int)

# Calculate class weights based on class distribution
class_counts = np.bincount(y_train_int)
total_samples = np.sum(class_counts)
class_weights = {label: total_samples / (len(class_labels) * count) for label, count in enumerate(class_counts)}

# Initialize random forest and train it
rf_weighted = RandomForestClassifier(random_state=2, class_weight=class_weights)
rf_weighted.fit(x_train_vec, y_train_int)

# Predict and output classification report
x_test_vec = vectorizer.transform(x_test).toarray()
y_predicted_int = rf_weighted.predict(x_test_vec)
y_predicted = np.array([str(label) for label in y_predicted_int])

print(classification_report(y_test, y_predicted))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(12, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_weighted.classes_)
disp.plot(ax=ax, cmap='cividis')
plt.show()
