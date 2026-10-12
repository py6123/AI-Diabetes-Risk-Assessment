import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import textwrap

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# HTML HELPER
# =========================================================

def render_html(content):
    cleaned_html = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

.stApp {
    background:
        radial-gradient(circle at 0% 0%, #eef4ff 0%, transparent 25%),
        radial-gradient(circle at 100% 5%, #ecfeff 0%, transparent 22%),
        #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =====================================================
   NAVIGATION
===================================================== */

.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 4px 22px 4px;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo {
    width: 43px;
    height: 43px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 13px;

    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );

    color: white;

    font-size: 14px;
    font-weight: 850;

    box-shadow:
        0 8px 24px rgba(79, 70, 229, 0.23);
}

.nav-title {
    color: #0f172a;
    font-size: 17px;
    font-weight: 850;
}

.nav-subtitle {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 1px;
}


/* =====================================================
   HERO
===================================================== */

.hero {
    position: relative;
    overflow: hidden;

    border-radius: 29px;
    padding: 50px 54px;

    background: linear-gradient(
        125deg,
        #0f172a 0%,
        #1e3a8a 50%,
        #4338ca 100%
    );

    box-shadow:
        0 24px 60px rgba(15, 23, 42, 0.18);

    margin-bottom: 38px;
}

.hero::after {
    content: "";

    position: absolute;

    width: 360px;
    height: 360px;

    right: -105px;
    top: -145px;

    border-radius: 50%;

    background: rgba(255, 255, 255, 0.07);
}

.hero::before {
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    right: 200px;
    bottom: -185px;

    border-radius: 50%;

    background: rgba(125, 211, 252, 0.10);
}

.hero-label {
    position: relative;
    z-index: 2;

    color: #bfdbfe;

    font-size: 11px;
    font-weight: 850;

    letter-spacing: 1.3px;
    text-transform: uppercase;

    margin-bottom: 16px;
}

.hero-title {
    position: relative;
    z-index: 2;

    color: white;

    font-size: 3.2rem;
    font-weight: 850;

    letter-spacing: -1.3px;
    line-height: 1.07;

    max-width: 760px;
}

.hero-text {
    position: relative;
    z-index: 2;

    color: #dbeafe;

    max-width: 690px;

    font-size: 1.02rem;
    line-height: 1.7;

    margin-top: 18px;
}

.hero-pill {
    position: relative;
    z-index: 2;

    display: inline-block;

    margin-top: 24px;

    padding: 9px 14px;

    border-radius: 999px;

    background: rgba(255, 255, 255, 0.10);

    border: 1px solid rgba(255, 255, 255, 0.17);

    color: #e0e7ff;

    font-size: 12px;
    font-weight: 650;
}


/* =====================================================
   SECTION TITLES
===================================================== */

.section-label {
    color: #4f46e5;

    font-size: 11px;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 1.1px;
}

.section-title {
    color: #0f172a;

    font-size: 1.85rem;
    font-weight: 850;

    margin-top: 4px;
    margin-bottom: 5px;
}

.section-description {
    color: #64748b;

    font-size: 0.92rem;
    line-height: 1.6;

    margin-bottom: 22px;
}


/* =====================================================
   LEFT GUIDE CARD
===================================================== */

.guide-card {
    height: 100%;

    background: linear-gradient(
        145deg,
        #eef2ff,
        #ffffff
    );

    border: 1px solid #dbe4f2;
    border-radius: 22px;

    padding: 27px;

    box-shadow:
        0 14px 38px rgba(15, 23, 42, 0.045);
}

.guide-icon {
    width: 46px;
    height: 46px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background: #e0e7ff;

    font-size: 21px;

    margin-bottom: 18px;
}

.card-title {
    color: #0f172a;

    font-size: 1.13rem;
    font-weight: 850;

    margin-bottom: 8px;
}

.card-text {
    color: #64748b;

    font-size: 0.87rem;
    line-height: 1.65;
}

.small-heading {
    color: #475569;

    font-size: 0.72rem;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 0.8px;

    margin-top: 22px;
    margin-bottom: 11px;
}

.check-item {
    display: flex;
    align-items: center;

    gap: 9px;

    margin-bottom: 10px;

    color: #475569;

    font-size: 0.84rem;
}

.check-circle {
    width: 22px;
    height: 22px;

    display: flex;
    align-items: center;
    justify-content: center;

    flex-shrink: 0;

    border-radius: 7px;

    background: #ecfdf5;

    color: #059669;

    font-size: 11px;
    font-weight: 900;
}


/* =====================================================
   RIGHT QUICK GUIDE
===================================================== */

.quick-card {
    height: 100%;

    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 22px;

    padding: 27px;

    box-shadow:
        0 14px 38px rgba(15, 23, 42, 0.045);
}

.quick-item {
    padding: 13px 0;

    border-bottom: 1px solid #f1f5f9;
}

.quick-item:last-child {
    border-bottom: none;
}

.quick-name {
    color: #0f172a;

    font-size: 0.86rem;
    font-weight: 800;
}

.quick-description {
    color: #94a3b8;

    font-size: 0.76rem;
    line-height: 1.45;

    margin-top: 3px;
}


/* =====================================================
   FORM
===================================================== */

div[data-testid="stForm"] {
    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 23px;

    padding: 29px;

    box-shadow:
        0 17px 48px rgba(15, 23, 42, 0.06);
}

div[data-testid="stTextInput"] label {
    color: #334155;

    font-size: 0.87rem;
    font-weight: 750;
}

div[data-testid="stTextInput"] input {
    min-height: 50px;

    border-radius: 12px;

    border: 1px solid #dbe3ef;

    background: #f8fafc;

    color: #0f172a;

    transition: 0.15s ease;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1;

    background: white;

    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.10);
}


/* =====================================================
   PRIMARY BUTTON
===================================================== */

div[data-testid="stFormSubmitButton"] button {
    width: 100%;

    min-height: 53px;

    border: none;
    border-radius: 13px;

    margin-top: 11px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    );

    color: white;

    font-size: 15px;
    font-weight: 750;

    box-shadow:
        0 11px 25px rgba(79, 70, 229, 0.20);

    transition: 0.15s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: none;

    transform: translateY(-1px);

    box-shadow:
        0 15px 30px rgba(79, 70, 229, 0.27);
}


/* =====================================================
   SECONDARY BUTTON
===================================================== */

div[data-testid="stButton"] button {
    border-radius: 11px;

    border: 1px solid #dbe3ef;

    background: white;

    color: #475569;

    font-weight: 650;
}


/* =====================================================
   RESULT
===================================================== */

.result-card {
    border-radius: 24px;

    padding: 31px;

    margin-top: 10px;

    box-shadow:
        0 16px 45px rgba(15, 23, 42, 0.055);
}

.result-low {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #ffffff
    );

    border: 1px solid #a7f3d0;
}

.result-high {
    background: linear-gradient(
        135deg,
        #fff7ed,
        #ffffff
    );

    border: 1px solid #fed7aa;
}

.result-badge {
    display: inline-block;

    padding: 7px 12px;

    border-radius: 999px;

    font-size: 10px;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 0.7px;

    margin-bottom: 13px;
}

.badge-low {
    background: #d1fae5;
    color: #047857;
}

.badge-high {
    background: #ffedd5;
    color: #c2410c;
}

.result-title {
    color: #0f172a;

    font-size: 1.65rem;
    font-weight: 850;

    line-height: 1.25;
}

.result-text {
    color: #64748b;

    font-size: 0.91rem;
    line-height: 1.65;

    margin-top: 10px;

    max-width: 850px;
}

.result-footer {
    color: #475569;

    font-size: 0.82rem;

    margin-top: 18px;
    padding-top: 15px;

    border-top:
        1px solid rgba(148, 163, 184, 0.25);
}


/* =====================================================
   MEASUREMENT CARDS
===================================================== */

.measurement-card {
    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 18px;

    padding: 21px;

    min-height: 145px;

    box-shadow:
        0 10px 32px rgba(15, 23, 42, 0.04);
}

.measurement-name {
    color: #64748b;

    font-size: 0.74rem;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.measurement-value {
    color: #0f172a;

    font-size: 1.65rem;
    font-weight: 850;

    margin-top: 8px;
}

.measurement-help {
    color: #94a3b8;

    font-size: 0.76rem;
    line-height: 1.45;

    margin-top: 8px;
}


/* =====================================================
   WELLBEING CARDS
===================================================== */

.wellbeing-card {
    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 18px;

    padding: 22px;

    min-height: 180px;
}

.wellbeing-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.wellbeing-title {
    color: #0f172a;

    font-size: 1rem;
    font-weight: 800;

    margin-bottom: 7px;
}

.wellbeing-text {
    color: #64748b;

    font-size: 0.82rem;
    line-height: 1.55;
}


/* =====================================================
   LEARN CARDS
===================================================== */

.learn-card {
    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 17px;

    padding: 21px;

    min-height: 170px;
}

.learn-title {
    color: #0f172a;

    font-size: 1rem;
    font-weight: 800;

    margin-bottom: 7px;
}

.learn-text {
    color: #64748b;

    font-size: 0.82rem;
    line-height: 1.6;
}


/* =====================================================
   DISCLAIMER
===================================================== */

.disclaimer {
    margin-top: 28px;

    padding: 17px 20px;

    border-radius: 15px;

    background: #f8fafc;

    border: 1px solid #e2e8f0;

    color: #64748b;

    font-size: 0.79rem;
    line-height: 1.6;
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    text-align: center;

    color: #94a3b8;

    font-size: 0.75rem;

    margin-top: 35px;
}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 800px) {

    .hero {
        padding: 34px 27px;
    }

    .hero-title {
        font-size: 2.2rem;
    }
}

</style>
""")


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "Diabetesmodel.pkl"
DATA_PATH = BASE_DIR / "diabetes.csv"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()

except Exception as error:

    st.error(
        f"Unable to load the diabetes assessment model: {error}"
    )

    st.stop()


# =========================================================
# REBUILD ORIGINAL DATA SCALING
# =========================================================

@st.cache_resource
def prepare_scaling():

    dataset = pd.read_csv(DATA_PATH)

    cleaned = dataset.copy()

    zero_missing_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    cleaned[
        zero_missing_columns
    ] = cleaned[
        zero_missing_columns
    ].replace(
        0,
        np.nan
    )

    for column in zero_missing_columns:

        cleaned[column] = (
            cleaned[column]
            .fillna(
                cleaned[column].mean()
            )
        )

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(
        cleaned
    )

    # Original Kaggle notebook features:
    # 1 = Glucose
    # 4 = Insulin
    # 5 = BMI
    # 7 = Age

    selected_indices = [
        1,
        4,
        5,
        7
    ]

    minimum_values = scaler.data_min_[
        selected_indices
    ]

    value_ranges = scaler.data_range_[
        selected_indices
    ]

    return minimum_values, value_ranges


try:

    FEATURE_MINS, FEATURE_RANGES = (
        prepare_scaling()
    )

except Exception as error:

    st.error(
        f"Unable to prepare the dataset: {error}"
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "form_version": 0,
    "validation_message": "",
    "prediction_result": None,
    "submitted_values": None,
    "scroll_to_result": False
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CURRENT INPUT KEYS
# =========================================================

def get_input_keys():

    version = st.session_state.form_version

    return {
        "glucose": f"glucose_{version}",
        "insulin": f"insulin_{version}",
        "bmi": f"bmi_{version}",
        "age": f"age_{version}"
    }


# =========================================================
# FORMAT FIELD NAMES
# =========================================================

def join_names(items):

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return (
        ", ".join(items[:-1])
        + f", and {items[-1]}"
    )


# =========================================================
# PREDICTION
# =========================================================

def assess_diabetes():

    keys = get_input_keys()

    # Reset previous output
    st.session_state.validation_message = ""
    st.session_state.prediction_result = None
    st.session_state.submitted_values = None

    values = {}

    missing_fields = []
    invalid_numeric_fields = []
    non_positive_fields = []

    age_invalid = False


    # =====================================================
    # GLUCOSE, INSULIN AND BMI
    # =====================================================

    numeric_fields = {
        "glucose": "Glucose",
        "insulin": "Insulin",
        "bmi": "BMI"
    }


    for field, label in numeric_fields.items():

        key = keys[field]

        value_text = str(
            st.session_state[key]
        ).strip()


        # Empty input
        if value_text == "":

            missing_fields.append(label)

            continue


        # Numeric validation
        try:

            value = float(
                value_text
            )

            # Reject NaN and infinity
            if not np.isfinite(value):

                raise ValueError


        except ValueError:

            invalid_numeric_fields.append(
                label
            )

            # Only clear the incorrect field
            st.session_state[key] = ""

            continue


        # Positive-value validation
        if value <= 0:

            non_positive_fields.append(
                label
            )

            # Only clear the incorrect field
            st.session_state[key] = ""

            continue


        values[label] = value


    # =====================================================
    # AGE
    # =====================================================

    age_key = keys["age"]

    age_text = str(
        st.session_state[age_key]
    ).strip()


    if age_text == "":

        missing_fields.append(
            "Age"
        )


    else:

        try:

            age = int(
                age_text
            )


        except ValueError:

            age_invalid = True

            # Only clear Age
            st.session_state[
                age_key
            ] = ""


        else:

            if age <= 0:

                non_positive_fields.append(
                    "Age"
                )

                st.session_state[
                    age_key
                ] = ""


            else:

                values["Age"] = age


    # =====================================================
    # ERROR MESSAGES
    # =====================================================

    messages = []


    # Empty fields
    if missing_fields:

        names = join_names(
            missing_fields
        )

        if len(missing_fields) == 1:

            messages.append(
                f"{names} is required."
            )

        else:

            messages.append(
                f"{names} are required."
            )


    # Invalid numeric fields
    if invalid_numeric_fields:

        names = join_names(
            invalid_numeric_fields
        )

        if len(
            invalid_numeric_fields
        ) == 1:

            messages.append(
                f"{names} must be a numeric value."
            )

        else:

            messages.append(
                f"{names} must be numeric values."
            )


    # Age must be integer
    if age_invalid:

        messages.append(
            "Age must be entered as a whole number."
        )


    # Zero or negative values
    if non_positive_fields:

        names = join_names(
            non_positive_fields
        )

        messages.append(
            f"{names} must be greater than 0."
        )


    # Stop prediction if validation fails
    if messages:

        st.session_state.validation_message = (
            " ".join(messages)
        )

        return


    # =====================================================
    # SCALE USER INPUT
    # =====================================================

    raw_input = np.array(
        [
            values["Glucose"],
            values["Insulin"],
            values["BMI"],
            values["Age"]
        ],
        dtype=float
    )


    scaled_input = (
        raw_input
        - FEATURE_MINS
    ) / FEATURE_RANGES


    scaled_input = (
        scaled_input.reshape(
            1,
            -1
        )
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        scaled_input
    )


    prediction_value = int(
        prediction[0]
    )


    st.session_state.prediction_result = (
        prediction_value
    )


    # Save the values used for the result cards
    st.session_state.submitted_values = {
        "Glucose": values["Glucose"],
        "Insulin": values["Insulin"],
        "BMI": values["BMI"],
        "Age": values["Age"]
    }


    # Automatically move the user to the result
    st.session_state.scroll_to_result = True


# =========================================================
# CLEAR FORM
# =========================================================

def clear_everything():

    # New widget keys force all input boxes to reset
    st.session_state.form_version += 1

    st.session_state.validation_message = ""

    st.session_state.prediction_result = None

    st.session_state.submitted_values = None

    st.session_state.scroll_to_result = False


# =========================================================
# NAVIGATION
# =========================================================

render_html("""
<div class="nav">

<div class="nav-brand">

<div class="nav-logo">
AI
</div>

<div>

<div class="nav-title">
AI Diabetes Risk Assessment
</div>

<div class="nav-subtitle">
Diabetes screening application
</div>

</div>

</div>

</div>
""")


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">

<div class="hero-label">
Diabetes Risk Screening
</div>

<div class="hero-title">
Check your diabetes risk in under a minute
</div>

<div class="hero-text">
Enter four health measurements to receive an instant
AI-assisted screening result and easy-to-understand guidance.
</div>

<div class="hero-pill">
Simple • Fast • Educational
</div>

</div>
""")


# =========================================================
# ASSESSMENT HEADING
# =========================================================

render_html("""
<div class="section-label">
Your Assessment
</div>

<div class="section-title">
Enter your health measurements
</div>

<div class="section-description">
Complete all four fields below, then select the Check My Diabetes Risk button.
</div>
""")


# =========================================================
# THREE-COLUMN ASSESSMENT LAYOUT
# =========================================================

left_guide, main_form, right_guide = st.columns(
    [0.75, 1.55, 0.75],
    gap="large"
)


# =========================================================
# LEFT: BEFORE YOU BEGIN
# =========================================================

with left_guide:

    render_html("""
    <div class="guide-card">

    <div class="guide-icon">
    🩺
    </div>

    <div class="card-title">
    Before you begin
    </div>

    <div class="card-text">
    Have these four measurements ready.
    Enter them exactly as recorded before starting your assessment.
    </div>

    <div class="small-heading">
    You'll need
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Glucose level</div>
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Insulin level</div>
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Body Mass Index</div>
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Age</div>
    </div>

    </div>
    """)


# =========================================================
# CENTER: INPUT FORM
# =========================================================

with main_form:

    input_keys = get_input_keys()


    with st.form(
        f"assessment_form_{st.session_state.form_version}"
    ):

        first_left, first_right = st.columns(2)


        with first_left:

            st.text_input(
                "Glucose",
                placeholder="Enter glucose level",
                key=input_keys["glucose"],
                help=(
                    "Enter your recorded glucose "
                    "measurement as a number."
                )
            )


        with first_right:

            st.text_input(
                "Insulin",
                placeholder="Enter insulin level",
                key=input_keys["insulin"],
                help=(
                    "Enter your recorded insulin "
                    "measurement as a number."
                )
            )


        second_left, second_right = st.columns(2)


        with second_left:

            st.text_input(
                "BMI",
                placeholder="e.g. 25.0",
                key=input_keys["bmi"],
                help=(
                    "Enter Body Mass Index "
                    "as a numeric value."
                )
            )


        with second_right:

            st.text_input(
                "Age",
                placeholder="e.g. 30",
                key=input_keys["age"],
                help=(
                    "Age must be entered "
                    "as a whole number."
                )
            )


        st.form_submit_button(
            "Check My Diabetes Risk",
            use_container_width=True,
            on_click=assess_diabetes
        )


    # Validation error
    if st.session_state.validation_message:

        st.error(
            st.session_state.validation_message
        )


    # Clear all values and previous result
    st.button(
        "Clear form",
        on_click=clear_everything,
        key="clear_all_button"
    )


# =========================================================
# RIGHT: QUICK GUIDE
# =========================================================

with right_guide:

    render_html("""
    <div class="quick-card">

    <div class="card-title">
    What do these mean?
    </div>

    <div class="card-text">
    A quick guide to the information used in your assessment.
    </div>

    <div class="quick-item">

    <div class="quick-name">
    Glucose
    </div>

    <div class="quick-description">
    A measurement related to blood sugar level.
    </div>

    </div>


    <div class="quick-item">

    <div class="quick-name">
    Insulin
    </div>

    <div class="quick-description">
    A hormone involved in controlling blood glucose.
    </div>

    </div>


    <div class="quick-item">

    <div class="quick-name">
    BMI
    </div>

    <div class="quick-description">
    A screening measure based on height and weight.
    </div>

    </div>


    <div class="quick-item">

    <div class="quick-name">
    Age
    </div>

    <div class="quick-description">
    Used as additional context in the assessment.
    </div>

    </div>

    </div>
    """)


# =========================================================
# RESULT SECTION
# =========================================================

if st.session_state.prediction_result is not None:

    st.markdown("")

    st.divider()


    # Anchor for automatic scrolling
    render_html("""
    <div id="assessment-result"></div>
    """)


    render_html("""
    <div class="section-label">
    Your Result
    </div>

    <div class="section-title">
    Diabetes risk assessment
    </div>
    """)


    # =====================================================
    # RESULT CLASSIFICATION
    # =====================================================

    if st.session_state.prediction_result == 0:

        render_html("""
        <div class="result-card result-low">

        <div class="result-badge badge-low">
        Lower predicted likelihood
        </div>

        <div class="result-title">
        ✓ Your assessment indicates a lower likelihood of diabetes.
        </div>

        <div class="result-text">
        Based on the measurements you entered,
        the prediction model classified this assessment
        in the lower-likelihood category.
        </div>

        <div class="result-footer">
        This is a screening result rather than a medical diagnosis.
        Continue regular health checks as appropriate.
        </div>

        </div>
        """)


    else:

        render_html("""
        <div class="result-card result-high">

        <div class="result-badge badge-high">
        Higher predicted likelihood
        </div>

        <div class="result-title">
        ⚠ Your assessment indicates a higher likelihood of diabetes.
        </div>

        <div class="result-text">
        Based on the measurements you entered,
        the prediction model classified this assessment
        in the higher-likelihood category.
        </div>

        <div class="result-footer">
        Consider discussing your measurements
        with a qualified healthcare professional.
        </div>

        </div>
        """)


    # =====================================================
    # VALUES ENTERED
    # =====================================================

    st.markdown("")

    render_html("""
    <div class="section-label">
    Your Information
    </div>

    <div class="section-title">
    Measurements used in this assessment
    </div>
    """)


    values = st.session_state.submitted_values


    card1, card2, card3, card4 = st.columns(4)


    with card1:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Glucose
        </div>

        <div class="measurement-value">
        {values["Glucose"]:g}
        </div>

        <div class="measurement-help">
        The glucose value entered for this assessment.
        </div>

        </div>
        """)


    with card2:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Insulin
        </div>

        <div class="measurement-value">
        {values["Insulin"]:g}
        </div>

        <div class="measurement-help">
        The insulin value entered for this assessment.
        </div>

        </div>
        """)


    with card3:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        BMI
        </div>

        <div class="measurement-value">
        {values["BMI"]:g}
        </div>

        <div class="measurement-help">
        Body Mass Index used as an assessment input.
        </div>

        </div>
        """)


    with card4:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Age
        </div>

        <div class="measurement-value">
        {values["Age"]:g}
        </div>

        <div class="measurement-help">
        Age used as additional assessment context.
        </div>

        </div>
        """)


    # =====================================================
    # EXTRA USER FEATURES
    # =====================================================

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Next steps",
            "Everyday health",
            "Understand your inputs",
            "Technical notes"
        ]
    )


    # -----------------------------------------------------
    # NEXT STEPS
    # -----------------------------------------------------

    with tab1:

        if st.session_state.prediction_result == 1:

            st.markdown(
                """
                ### What can you do next?

                Your assessment returned a **higher predicted likelihood**.

                - Keep a record of the measurements used in this assessment.
                - If you are concerned about the result, consider discussing it with a qualified healthcare professional.
                - Do not use this screening result as a substitute for a medical diagnosis.
                """
            )


        else:

            st.markdown(
                """
                ### What can you do next?

                Your assessment returned a **lower predicted likelihood**.

                - Continue maintaining regular health checks.
                - Keep track of important health measurements over time.
                - Remember that a lower predicted likelihood does not rule out diabetes or other health conditions.
                """
            )


    # -----------------------------------------------------
    # EVERYDAY HEALTH
    # -----------------------------------------------------

    with tab2:

        wellness1, wellness2, wellness3 = st.columns(3)


        with wellness1:

            render_html("""
            <div class="wellbeing-card">

            <div class="wellbeing-icon">
            🥗
            </div>

            <div class="wellbeing-title">
            Build steady eating habits
            </div>

            <div class="wellbeing-text">
            A varied and balanced eating pattern can support
            general health and long-term wellbeing.
            </div>

            </div>
            """)


        with wellness2:

            render_html("""
            <div class="wellbeing-card">

            <div class="wellbeing-icon">
            🚶
            </div>

            <div class="wellbeing-title">
            Keep moving regularly
            </div>

            <div class="wellbeing-text">
            Regular physical activity can support overall
            health, mobility and weight management.
            </div>

            </div>
            """)


        with wellness3:

            render_html("""
            <div class="wellbeing-card">

            <div class="wellbeing-icon">
            📋
            </div>

            <div class="wellbeing-title">
            Keep track of changes
            </div>

            <div class="wellbeing-text">
            Regular health checks can help you notice
            important changes in your measurements over time.
            </div>

            </div>
            """)


    # -----------------------------------------------------
    # UNDERSTAND INPUTS
    # -----------------------------------------------------

    with tab3:

        learn1, learn2 = st.columns(2)


        with learn1:

            render_html("""
            <div class="learn-card">

            <div class="learn-title">
            Glucose
            </div>

            <div class="learn-text">
            Glucose refers to sugar circulating in the blood.
            Its interpretation can depend on when and how the
            measurement was taken.
            </div>

            </div>
            """)


            render_html("""
            <div class="learn-card" style="margin-top:15px;">

            <div class="learn-title">
            Insulin
            </div>

            <div class="learn-text">
            Insulin is a hormone involved in regulating blood glucose.
            Individual readings should be interpreted in the correct
            clinical and laboratory context.
            </div>

            </div>
            """)


        with learn2:

            render_html("""
            <div class="learn-card">

            <div class="learn-title">
            BMI
            </div>

            <div class="learn-text">
            Body Mass Index is calculated using height and weight
            and is commonly used as a general screening measure.
            </div>

            </div>
            """)


            render_html("""
            <div class="learn-card" style="margin-top:15px;">

            <div class="learn-title">
            Age
            </div>

            <div class="learn-text">
            Age is included as contextual information used by the
            prediction model alongside the other measurements.
            </div>

            </div>
            """)


    # -----------------------------------------------------
    # TECHNICAL NOTES
    # -----------------------------------------------------

    with tab4:

        st.markdown(
            """
            ### Behind the assessment

            **Model:**  
            K-Nearest Neighbors (KNN)

            **Model inputs:**  
            Glucose, Insulin, BMI and Age

            **Prediction type:**  
            Binary classification

            **Preprocessing:**  
            Input values are scaled using the same MinMax approach used with the supplied training dataset.

            **Technology:**  
            Python, Scikit-learn and Streamlit

            **Model source:**  
            Diabetes Risk Assessment model provided for the classroom project.
            """
        )


    # =====================================================
    # AUTO-SCROLL TO RESULT
    # =====================================================

    if st.session_state.scroll_to_result:

        components.html(
            """
            <script>
            setTimeout(function () {

                const element =
                    window.parent.document.getElementById(
                        "assessment-result"
                    );

                if (element) {

                    element.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }

            }, 200);
            </script>
            """,
            height=0
        )

        st.session_state.scroll_to_result = False


# =========================================================
# DISCLAIMER
# =========================================================

render_html("""
<div class="disclaimer">

<strong>Important:</strong>

This application is an educational diabetes screening project.
Its output is generated by a machine learning model and does not
provide a medical diagnosis or replace professional medical advice.

</div>
""")


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">
AI Diabetes Risk Assessment
</div>
""")