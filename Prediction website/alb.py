import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_excel('18) Glucose, serum.xlsx')


df.shape

df.head()

df.dtypes

df.isna().sum()

df = df.dropna()

df.shape

df.isna().sum()

print(df.columns)


plt.scatter(df['class'],df['glucose'])
plt.show()

idx, c = np.unique(df['class'],return_counts=True)
sns.barplot(x=idx,y=c)

label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

df['class'].value_counts()

# Separate majority and minority classes
majority_class = df[df['class'] == 'Healthy']
minority_class_Excess = df[df['class'] == 'Excess']

# Calculate the difference in counts between classes
minority_class_Excess_count = minority_class_Excess.shape[0]
majority_class_count = majority_class.shape[0]
difference = majority_class_count - minority_class_Excess_count

# Duplicate minority class samples to balance the dataset
oversampled_minority_Excess = minority_class_Excess.sample(n=difference+1630, replace=True)

# Concatenate the oversampled minority class with the original majority class
balanced_df = pd.concat([majority_class, oversampled_minority_Excess])

# Separate majority and minority classes
majority_class = df[df['class'] == 'Healthy']
minority_class_Deficiency = df[df['class'] == 'Deficiency']

# Calculate the difference in counts between classes
minority_class_Deficiency_count = minority_class_Deficiency.shape[0]
majority_class_count = majority_class.shape[0]
difference = majority_class_count - minority_class_Deficiency_count

# Duplicate minority class samples to balance the dataset
oversampled_minority_Deficiency = minority_class_Deficiency.sample(n=difference+155, replace=True)

# Concatenate the oversampled minority class with the original majority class
balanced_df = pd.concat([balanced_df, oversampled_minority_Deficiency])

balanced_df.shape

idx, c = np.unique(balanced_df['class'],return_counts=True)
sns.barplot(x=idx,y=c)

balanced_df['class'].value_counts()

plt.scatter(balanced_df['class'],balanced_df['glucose'])
plt.show()

x_train,x_test,y_train,y_test = train_test_split(balanced_df.iloc[:,[0,1]],balanced_df.iloc[:,2],test_size=0.2,random_state=42)

model_glucose	 = DecisionTreeClassifier()
model_glucose.fit(x_train,y_train)

y_pred = model_glucose.predict(x_test)

accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

Report = classification_report(y_test,y_pred)
print(Report)

balanced_df.head(13000)

balanced_df.reset_index(drop=True,inplace=True)

balanced_df.head(13000)

balanced_df.iloc[11137:11138,2]

model_glucose.predict(balanced_df.iloc[11137:11138,[0,1]])





import pickle
pickle.dump(model_glucose,open('model_glucose.pkl','wb'))