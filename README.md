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

📌 Note
This project is for educational purposes only.
If you're experiencing suicidal thoughts, please seek professional help immediately.

📬 Author
👩‍💻 Adrija Halder
📧 adrijahalder838@gmail.com
