# 🎓 Student Academic Performance Prediction

A **Machine Learning project** that predicts a student’s academic performance based on various academic, behavioral, and demographic factors.
This project demonstrates the **end-to-end ML workflow** — from data preprocessing and model training to deployment readiness.

---

## 📌 Project Overview

Educational institutions often need early insights into student performance to provide timely support.
This project uses **machine learning algorithms** to analyze student data and predict academic outcomes.

🔹 **Goal:** Predict student performance
🔹 **Domain:** Education + Machine Learning
🔹 **Type:** Supervised Learning (Regression / Classification – based on model used)

---

## 🧠 Machine Learning Workflow

1. **Data Collection**
2. **Data Cleaning & Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Selection & Engineering**
5. **Model Training**
6. **Model Evaluation**
7. **Model Serialization (joblib)**
8. **Deployment-ready Application**

---

## 🛠️ Tech Stack Used

* **Programming Language:** Python
* **Libraries & Tools:**

  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * Scikit-learn
  * Joblib
  * Streamlit / Gradio (for UI – deployment stage)

---

## 📊 Features Used for Prediction

Some of the common features include (example):

* Study hours
* Attendance
* Previous grades
* Assignment performance
* Internal assessments
* Other academic indicators

> ⚠️ Exact features depend on the dataset used in the project.

---

## 📈 Model Used

* Machine Learning algorithms such as:

  * Linear Regression / Logistic Regression
  * Random Forest
  * Decision Tree
    *(final model depends on best evaluation score)*

Model performance is evaluated using metrics like:

* Accuracy / R² Score
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ragini-Roy7/Student_Academic_Performance_Prediction.git
cd Student_Academic_Performance_Prediction
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # For Linux / Mac
venv\Scripts\activate         # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

*(or streamlit/gradio command based on deployment file)*

---

## 🌐 Deployment

The project is **deployment-ready** and can be hosted on platforms like:

* Hugging Face Spaces (Gradio)
* Render / Railway (FastAPI backend)
* Streamlit Cloud (version-compatible setup)

> Deployment issues related to library version conflicts are handled using proper dependency management.

---

## 📁 Project Structure

```
│   └── dataset.csv
├── model/
│   └── trained_model.joblib
├── app.py
├── requirements.txt
├── README.md
```

---

## 🎯 Learning Outcomes

Through this project, you will understand:

* Practical ML pipeline implementation
* Feature engineering & evaluation
* Model serialization using joblib
* Real-world deployment challenges
* Version compatibility & dependency management

---

## 👩‍💻 Author

**Ragini Roy**
🎓 MCA (AI & ML) Student
💡 Passionate about Machine Learning, Backend Development & Problem Solving

🔗 GitHub: [https://github.com/Ragini-Roy7](https://github.com/Ragini-Roy7)

---

## ⭐ Future Improvements

* Add more features for better accuracy
* Improve UI/UX
* Deploy using containerization (Docker)
* Add REST API support

---

