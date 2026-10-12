# AI Diabetes Risk Assessment

## Project Overview

AI Diabetes Risk Assessment is a Python-based web application developed using Streamlit.

The application uses a trained machine learning model to assess the likelihood of diabetes based on health measurements entered by the user.

This project demonstrates how a machine learning model can be integrated into a simple and user-friendly frontend application.

---

## Project Objective

The objective of this project is to develop a frontend application that uses the provided Diabetes Risk Assessment machine learning model to classify whether a user's health measurements indicate a lower or higher predicted likelihood of diabetes.

The application focuses on providing a simple interface, input validation, and clear prediction results.

---

## Health Measurements

The current model uses four health measurements:

- Glucose
- Insulin
- Body Mass Index (BMI)
- Age

These values are entered by the user through the Streamlit application and processed before being passed to the machine learning model.

---

## Machine Learning Model

The application uses a K-Nearest Neighbors (KNN) classification model.

The trained model is stored in:

`Diabetesmodel.pkl`

The model produces a binary classification result:

- Lower predicted likelihood of diabetes
- Higher predicted likelihood of diabetes

The prediction is intended for educational screening purposes only and should not be considered a medical diagnosis.

---

## Data Preprocessing

The original dataset is loaded from:

`diabetes.csv`

The application follows the preprocessing approach used in the supplied model development notebook.

The original dataset treats zero values in selected health measurements as missing values. These missing values are replaced using the mean of the relevant feature.

The selected model inputs are then scaled using MinMax scaling before being passed to the trained machine learning model.

---

## Application Features

The application includes:

- A user-friendly diabetes risk assessment interface
- Four health measurement input fields
- Input validation for missing or invalid values
- Validation for numeric values
- Whole-number validation for age
- Validation to prevent zero or negative values
- Automatic clearing of incorrect input fields
- Machine learning prediction
- Lower or higher predicted likelihood results
- Automatic scrolling to the prediction result
- Summary of the measurements used in the assessment
- Next-step guidance
- General health information
- Explanations of the health measurements
- Technical information about the prediction model
- Clear form function
- Medical disclaimer

---

## Technologies Used

The project uses the following technologies:

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Pickle
- HTML
- CSS

---

## Project Files

### `diabetes_app.py`

The main Streamlit application.

It contains the user interface, input validation, data preprocessing, machine learning prediction, and result display.

### `Diabetesmodel.pkl`

The trained K-Nearest Neighbors machine learning model used by the application.

### `diabetes.csv`

The dataset used for the original model development and preprocessing.

### `Diabetes_prediction (1).ipynb`

The original notebook containing the model development and training process.

### `DIABETIC.py`

Legacy/reference script generated from the original notebook.  
The final application is `diabetes_app.py`.

### `requirements.txt`

Contains the Python packages required to run the application.

### `README.md`

Contains the project description, setup instructions, and application information.

---

## Installation

Before running the application, ensure that Python is installed on the computer.

Open a terminal in the project folder.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Application

After installing the required packages, run the following command:

```bash
python -m streamlit run diabetes_app.py
```

Streamlit will start the application and provide a local address in the terminal.

For example:

```text
Local URL: http://localhost:8501
```

Open the provided address in a web browser to use the application.

---

## How to Use the Application

### Step 1: Prepare the Measurements

Prepare the following four health measurements:

- Glucose
- Insulin
- BMI
- Age

### Step 2: Enter the Measurements

Enter all four measurements into the assessment form.

### Step 3: Check Diabetes Risk

Select:

**Check My Diabetes Risk**

The application will validate the information before sending it to the machine learning model.

### Step 4: View the Result

The application will automatically move to the result section.

The prediction will be displayed as either:

- Lower predicted likelihood
- Higher predicted likelihood

The measurements used for the assessment will also be displayed.

---

## Input Validation

The application validates user input before making a prediction.

### Missing Values

All four fields are required.

If a field is empty, the application displays an error message.

### Numeric Validation

Glucose, Insulin, and BMI must contain numeric values.

For example:

Valid:

```text
120
80.5
25
```

Invalid:

```text
abc
hello
```

### Age Validation

Age must be entered as a whole number.

Valid:

```text
30
```

Invalid:

```text
30.5
```

### Positive Values

All measurements must be greater than zero.

Values such as the following are rejected:

```text
0
-10
```

If an incorrect value is entered, only the incorrect input field is cleared. Correct values entered in the other fields remain unchanged.

---

## Prediction Process

The application follows the following process:

1. The user enters the four required health measurements.
2. The application validates the input.
3. The health measurements are converted into numerical values.
4. The values are scaled using the same MinMax approach used during model development.
5. The processed values are passed to the trained KNN model.
6. The model generates a binary classification.
7. The result is displayed to the user in a clear and user-friendly format.

---

## Prediction Output

The machine learning model produces two possible classifications.

### Lower Predicted Likelihood

This means that the model classified the submitted measurements into the lower-likelihood category.

### Higher Predicted Likelihood

This means that the model classified the submitted measurements into the higher-likelihood category.

These results are machine learning predictions and are not medical diagnoses.

---

## User Interface

The application was designed to provide a simple and clear user experience.

The interface contains:

- A diabetes risk screening introduction
- A preparation guide
- The main assessment form
- Short explanations of each health measurement
- Prediction results
- A summary of entered measurements
- Next-step information
- General health information
- Input explanations
- Technical notes

The layout is designed to make the application easy to understand and use.

---

## Technical Information

### Machine Learning Algorithm

K-Nearest Neighbors (KNN)

### Prediction Type

Binary classification

### Current Model Inputs

- Glucose
- Insulin
- BMI
- Age

### Preprocessing Method

MinMax scaling

### Frontend Framework

Streamlit

### Programming Language

Python

---

## Error Handling

The application includes error handling for:

- Missing model files
- Missing dataset files
- Empty input fields
- Non-numeric values
- Invalid age values
- Zero values
- Negative values

The application prevents prediction when invalid input is detected.

---

## Important Disclaimer

This application was developed for educational purposes as part of an AI-driven healthcare classroom project.

The output is generated by a machine learning model.

The application does not provide a medical diagnosis and should not be used as a replacement for professional medical advice, diagnosis, or treatment.

Users who have concerns about their health should consult a qualified healthcare professional.

---

## Project Title

**AI Diabetes Risk Assessment**