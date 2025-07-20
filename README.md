# 🧠 Suicide Prediction using Machine Learning

This project applies machine learning to detect suicidal tendencies based on text inputs. It uses Natural Language Processing (NLP) with TF-IDF vectorization and a Logistic Regression classifier to predict the risk of suicide.

## 🔍 Problem Statement

Given a dataset of texts labeled as `suicide` or `non-suicide`, the goal is to build a model that can analyze new textual data and determine whether it reflects suicidal thoughts.

## 📁 Project Structure

suicide-prediction-ml/
├── data/
│ └── Suicide_Detection.csv # Dataset
├── model/
│ ├── suicide_predictor_model.pkl
│ └── vectorizer.pkl
├── main.py # Core training & evaluation script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

## ⚙️ Technologies Used

- **Python 3.x**
- **scikit-learn** for machine learning
- **pandas**, **numpy** for data manipulation
- **matplotlib**, **seaborn** for visualization
- **joblib** for model serialization
- **TF-IDF Vectorization** for text processing

## 🧪 Key Features

- Preprocessing and cleaning of text data
- Train-test split using `sklearn`
- Model training with `LogisticRegression`
- Accuracy and classification report evaluation
- Visualization: Confusion Matrix and class distribution
- Save and reuse model with `joblib`
- Real-time risk prediction from user input

## 📊 Model Performance

- Achieved **high accuracy** on test data
- Visual evaluation using **confusion matrix**
- Tested on **realistic suicidal vs non-suicidal texts**

## 💡 Sample Input/Output

```bash
Enter someone's thought: I feel so hopeless and can't go on.
Prediction: High Risk

Enter someone's thought: Life is beautiful and I'm looking forward to tomorrow.
Prediction: Low Risk

🚀 How to Run
Clone the repository
git clone https://github.com/Adrija-16/suicide-prediction-ml.git

Navigate to the project directory
cd suicide-prediction-ml

(Optional) Create a virtual environment
python -m venv venv && source venv/bin/activate (or use venv\Scripts\activate on Windows)

Install dependencies
pip install -r requirements.txt

Run the script
python main.py

## Model Evaluation

### Sample Data Examples

| Text Sample                                                                                | Class       |
|--------------------------------------------------------------------------------------------|-------------|
| Ex Wife Threatening SuicideRecently I left my ...                                          | suicide     |
| Am I weird I don't get affected by compliments...                                         | non-suicide |
| Finally 2020 is almost over... So I can never ...                                         | non-suicide |
| i need helpjust help me im crying so hard                                                 | suicide     |
| I’m so lostHello, my name is Adam (16) and I...                                          | suicide     |

---

### Performance Metrics

- **Accuracy:** 88%

#### Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Non-Suicide | 0.84      | 0.96   | 0.90     | 27      |
| Suicide     | 0.94      | 0.76   | 0.84     | 21      |
| **Accuracy**|           |        | **0.88** | 48      |
| Macro Avg   | 0.89      | 0.86   | 0.87     | 48      |
| Weighted Avg| 0.88      | 0.88   | 0.87     | 48      |

---

### Test Examples and Predictions

| Text                                                              | Predicted Risk |
|-------------------------------------------------------------------|----------------|
| I feel so hopeless and can't see a way out.                       | Low Risk       |
| Life is beautiful, I am happy and excited for the future.        | Low Risk       |
| Nobody cares about me, I'm all alone.                             | Low Risk       |
| I'm just having a bad day, things will get better.                | Low Risk       |

---

### Sample Actual vs Predicted

| Text                                                                                      | Actual Risk | Predicted Risk |
|-------------------------------------------------------------------------------------------|-------------|----------------|
| fuck the verizon smart family app i can’t even watch porn privately anymore wtf why...    | Low Risk    | Low Risk       |
| well, im screwed. i locked myself in the school toilet, and can't get out. for now...     | High Risk   | Low Risk       |
| i am ending my life today, goodbye everyone...                                            | High Risk   | High Risk      |
| guys i want friends that’s it, i’m alone and don’t talk to anyone...                      | Low Risk    | Low Risk       |
| finally 2020 is almost over... so i can never hear "2020 has been a bad year" ever again. | Low Risk    | Low Risk       |

---
### Confusion Matrix

[Confusion Matrix](CONFUSION_MATRIX.png)

### Class Distribution

[Class Distribution](Class_distribution.png)

### Actual V/s Predicted

[Actual V/s Predicted](Actual_Predicted.png)


📌 Note
This project is for educational purposes only.
If you're experiencing suicidal thoughts, please seek professional help immediately.

📬 Author
👩‍💻 Adrija Halder
📧 adrijahalder838@gmail.com
