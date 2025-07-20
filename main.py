# manipulation of data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import chardet
from colorama import Fore, Style, init
from sklearn.metrics import classification_report

# To automatically reset colour after printing each statement
init(autoreset=True)

# This opens the CSV file in binary mode, reads first 100000 bytes and use chardet to detect file encoding
with open('Suicide_Detection.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))

# converting csv into a panda dataframe.
data = pd.read_csv('Suicide_Detection.csv', encoding=result['encoding'], usecols=['text', 'class'])


print(Fore.GREEN + Style.BRIGHT + "Sample Data:")
# it will print the header along with the data of first 5 row(by default)
# data.head(10)
print(data.head())


def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""


# convert the text to lower case
data['text'] = data['text'].apply(preprocess_text)
# fills the missing values with "" string
data['text'] = data['text'].fillna("")

# to filter out rows with invalid class labels, keeping only the rows with 'non-suicide', 'suicide' labels
valid_classes = ['non-suicide', 'suicide']
data = data[data['class'].isin(valid_classes)]


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])


y = data['class'].map({'non-suicide': 0, 'suicide': 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'suicide_predictor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


y_pred = model.predict(X_test)


print(Fore.BLUE + Style.BRIGHT + "\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(Fore.MAGENTA + Style.BRIGHT + f'Accuracy: {accuracy:.2f}')
print(Fore.CYAN + Style.BRIGHT + "\nClassification Report:\n")
print(Fore.YELLOW + classification_report(y_test, y_pred, target_names=['Non-Suicide', 'Suicide']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


actual_counts = y_test.value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()

plt.figure(figsize=(10, 5))
bar_width = 0.35
index = np.arange(len(valid_classes))

bar1 = plt.bar(index, actual_counts, bar_width, label='Actual')
bar2 = plt.bar(index + bar_width, predicted_counts, bar_width, label='Predicted')

plt.xlabel('Class')
plt.ylabel('Counts')
plt.title('Actual vs Predicted Counts')
plt.xticks(index + bar_width / 2, ['Non-Suicide', 'Suicide'])
plt.legend()

plt.show()


test_examples = [
    "I feel so hopeless and can't see a way out.",
    "Life is beautiful, I am happy and excited for the future.",
    "Nobody cares about me, I'm all alone.",
    "I'm just having a bad day, things will get better."
]


test_examples_preprocessed = [preprocess_text(text) for text in test_examples]
test_examples_vectorized = vectorizer.transform(test_examples_preprocessed)

example_predictions = model.predict(test_examples_vectorized)


print(Fore.YELLOW + Style.BRIGHT + "\nTest Examples and Predictions:\n")
for text, prediction in zip(test_examples, example_predictions):
    print(f'{Fore.CYAN + Style.BRIGHT}Text: {Fore.RESET}"{text}"')
    print(f'{Fore.GREEN + Style.BRIGHT}Predicted Risk: '
          f'{Fore.RED + "High Risk" if prediction == 1 else Fore.GREEN + "Low Risk"}')
    print('---')


sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
sample_texts = data.iloc[sample_indices]['text'].values
sample_actual = y_test.iloc[sample_indices].values
sample_predicted = y_pred[sample_indices]

print(Fore.YELLOW + Style.BRIGHT + "\nSample Actual vs Predicted:\n")
for text, actual, predicted in zip(sample_texts, sample_actual, sample_predicted):
    print(f'{Fore.CYAN + Style.BRIGHT}Text: {Fore.RESET}"{text}"')
    print(f'{Fore.GREEN + Style.BRIGHT}Actual Risk: '
          f'{Fore.RED + "High Risk" if actual == 1 else Fore.GREEN + "Low Risk"}')
    print(f'{Fore.GREEN + Style.BRIGHT}Predicted Risk: '
          f'{Fore.RED + "High Risk" if predicted == 1 else Fore.GREEN + "Low Risk"}')
    print('---')


plt.figure(figsize=(8, 5))
sns.countplot(x='class', hue='class', data=data, palette='Set2', legend=False)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


def predict_risk(input_text):
    # Preprocess and vectorize input text
    input_text_preprocessed = preprocess_text(input_text)
    input_text_vectorized = vectorizer.transform([input_text_preprocessed])

    # Predict risk
    prediction1 = model.predict(input_text_vectorized)[0]

    # Return prediction
    return "High Risk" if prediction1 == 1 else "Low Risk"


exit_key = "exit"
while True:
    user_input = input("Enter someone's thought (type 'exit' to quit): ")

    if user_input.lower() == exit_key:
        print("Exiting the program...")
        break

    risk_prediction = predict_risk(user_input)

    print(f"\nBased on the input, the thought is classified as: {risk_prediction}\n")

actual_counts = y_test.value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()
