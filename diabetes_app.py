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

    max-width: 800px;
}

.hero-text {
    position: relative;
    z-index: 2;

    color: #dbeafe;

    max-width: 720px;

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
   GUIDE CARDS
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
   QUICK GUIDE
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
    padding: 12px 0;

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

    max-width: 900px;
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
   INFO CARDS
===================================================== */

.wellbeing-card,
.learn-card,
.dataset-card {
    background: white;

    border: 1px solid #e2e8f0;
    border-radius: 18px;

    padding: 22px;
}

.wellbeing-card {
    min-height: 180px;
}

.learn-card {
    min-height: 170px;
}

.wellbeing-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.wellbeing-title,
.learn-title {
    color: #0f172a;

    font-size: 1rem;
    font-weight: 800;

    margin-bottom: 7px;
}

.wellbeing-text,
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
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    dataset = load_dataset()

except Exception as error:

    st.error(
        f"Unable to load the diabetes dataset: {error}"
    )

    st.stop()


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
# PREPARE ORIGINAL DATA SCALING
# =========================================================

@st.cache_resource
def prepare_scaling():

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


    # Original supplied notebook:
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


    training_ranges = {
        "Glucose": (
            float(cleaned["Glucose"].min()),
            float(cleaned["Glucose"].max())
        ),

        "Insulin": (
            float(cleaned["Insulin"].min()),
            float(cleaned["Insulin"].max())
        ),

        "BMI": (
            float(cleaned["BMI"].min()),
            float(cleaned["BMI"].max())
        ),

        "Age": (
            float(cleaned["Age"].min()),
            float(cleaned["Age"].max())
        )
    }


    return (
        minimum_values,
        value_ranges,
        training_ranges
    )


try:

    (
        FEATURE_MINS,
        FEATURE_RANGES,
        TRAINING_RANGES
    ) = prepare_scaling()


except Exception as error:

    st.error(
        f"Unable to prepare the model data: {error}"
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "form_version": 0,
    "validation_message": "",
    "training_warnings": [],
    "prediction_result": None,
    "submitted_values": None,
    "scroll_to_result": False,

    "scenario_result": None,
    "scenario_values": None,
    "scenario_warnings": []
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def get_input_keys():

    version = st.session_state.form_version

    return {
        "glucose": f"glucose_{version}",
        "insulin": f"insulin_{version}",
        "weight": f"weight_{version}",
        "height": f"height_{version}",
        "age": f"age_{version}"
    }


def join_names(items):

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return (
        ", ".join(items[:-1])
        + f", and {items[-1]}"
    )


def format_number(value):

    if float(value).is_integer():
        return str(int(value))

    return f"{value:.2f}"


def calculate_bmi(weight_kg, height_cm):

    height_m = height_cm / 100

    return weight_kg / (height_m ** 2)


def build_scaled_input(
    glucose,
    insulin,
    bmi,
    age
):

    raw_input = np.array(
        [
            glucose,
            insulin,
            bmi,
            age
        ],
        dtype=float
    )

    scaled_input = (
        raw_input
        - FEATURE_MINS
    ) / FEATURE_RANGES

    return scaled_input.reshape(
        1,
        -1
    )


def run_prediction(
    glucose,
    insulin,
    bmi,
    age
):

    scaled_input = build_scaled_input(
        glucose,
        insulin,
        bmi,
        age
    )

    prediction = model.predict(
        scaled_input
    )

    return int(
        prediction[0]
    )


def get_training_warnings(
    glucose,
    insulin,
    bmi,
    age
):

    warnings = []

    values = {
        "Glucose": glucose,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age
    }


    for feature, value in values.items():

        minimum, maximum = (
            TRAINING_RANGES[feature]
        )

        if value < minimum or value > maximum:

            warnings.append(
                f"{feature} is outside the range represented "
                f"in the training dataset "
                f"({format_number(minimum)}–"
                f"{format_number(maximum)}). "
                f"The prediction may be less reliable."
            )


    return warnings


def result_label(result):

    if result == 0:
        return "Lower predicted likelihood"

    return "Higher predicted likelihood"


# =========================================================
# MAIN ASSESSMENT
# =========================================================

def assess_diabetes():

    keys = get_input_keys()

    # Reset previous output
    st.session_state.validation_message = ""
    st.session_state.training_warnings = []
    st.session_state.prediction_result = None
    st.session_state.submitted_values = None

    st.session_state.scenario_result = None
    st.session_state.scenario_values = None
    st.session_state.scenario_warnings = []


    values = {}

    missing_fields = []
    invalid_numeric_fields = []
    non_positive_fields = []
    plausibility_errors = []

    age_invalid = False


    # =====================================================
    # NUMERIC FIELDS
    # =====================================================

    numeric_fields = {
        "glucose": "Glucose",
        "insulin": "Insulin",
        "weight": "Weight",
        "height": "Height"
    }


    for field, label in numeric_fields.items():

        key = keys[field]

        value_text = str(
            st.session_state[key]
        ).strip()


        # Empty
        if value_text == "":

            missing_fields.append(label)

            continue


        # Numeric
        try:

            value = float(
                value_text
            )

            if not np.isfinite(value):
                raise ValueError


        except ValueError:

            invalid_numeric_fields.append(
                label
            )

            st.session_state[key] = ""

            continue


        # Positive
        if value <= 0:

            non_positive_fields.append(
                label
            )

            st.session_state[key] = ""

            continue


        # Basic plausibility checks
        if label == "Weight":

            if value < 10 or value > 500:

                plausibility_errors.append(
                    "Weight must be between 10 and 500 kg."
                )

                st.session_state[key] = ""

                continue


        if label == "Height":

            if value < 50 or value > 250:

                plausibility_errors.append(
                    "Height must be between 50 and 250 cm."
                )

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


            elif age > 120:

                plausibility_errors.append(
                    "Age must be between 1 and 120 years."
                )

                st.session_state[
                    age_key
                ] = ""


            else:

                values["Age"] = age


    # =====================================================
    # BUILD ERROR MESSAGES
    # =====================================================

    messages = []


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


    if age_invalid:

        messages.append(
            "Age must be entered as a whole number."
        )


    if non_positive_fields:

        names = join_names(
            non_positive_fields
        )

        messages.append(
            f"{names} must be greater than 0."
        )


    messages.extend(
        plausibility_errors
    )


    # Stop if hard validation fails
    if messages:

        st.session_state.validation_message = (
            " ".join(messages)
        )

        return


    # =====================================================
    # BACKEND BMI CALCULATION
    # =====================================================

    bmi = calculate_bmi(
        values["Weight"],
        values["Height"]
    )


    # =====================================================
    # TRAINING-DATA RANGE WARNINGS
    # =====================================================

    warnings = get_training_warnings(
        values["Glucose"],
        values["Insulin"],
        bmi,
        values["Age"]
    )


    st.session_state.training_warnings = (
        warnings
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction_value = run_prediction(
        values["Glucose"],
        values["Insulin"],
        bmi,
        values["Age"]
    )


    st.session_state.prediction_result = (
        prediction_value
    )


    # Save all user-entered and calculated values
    st.session_state.submitted_values = {
        "Glucose": values["Glucose"],
        "Insulin": values["Insulin"],
        "Weight": values["Weight"],
        "Height": values["Height"],
        "BMI": bmi,
        "Age": values["Age"]
    }


    st.session_state.scroll_to_result = True


# =========================================================
# CLEAR FORM
# =========================================================

def clear_everything():

    # Force new widget keys
    st.session_state.form_version += 1

    st.session_state.validation_message = ""
    st.session_state.training_warnings = []

    st.session_state.prediction_result = None
    st.session_state.submitted_values = None

    st.session_state.scroll_to_result = False

    st.session_state.scenario_result = None
    st.session_state.scenario_values = None
    st.session_state.scenario_warnings = []


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
Enter your health measurements to receive an
AI-assisted screening result, calculated BMI,
and easy-to-understand guidance.
</div>

<div class="hero-pill">
Simple • Interactive • Educational
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
Complete all fields below. BMI will be calculated automatically
from your weight and height before the values are sent to the model.
</div>
""")


# =========================================================
# THREE-COLUMN LAYOUT
# =========================================================

left_guide, main_form, right_guide = st.columns(
    [0.75, 1.55, 0.75],
    gap="large"
)


# =========================================================
# LEFT GUIDE
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
    Have your health measurements ready.
    Weight and height are used to calculate BMI automatically.
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
    <div>Weight in kilograms</div>
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Height in centimetres</div>
    </div>

    <div class="check-item">
    <div class="check-circle">✓</div>
    <div>Age</div>
    </div>

    </div>
    """)


# =========================================================
# MAIN FORM
# =========================================================

with main_form:

    input_keys = get_input_keys()


    with st.form(
        f"assessment_form_{st.session_state.form_version}"
    ):

        row1_left, row1_right = (
            st.columns(2)
        )


        with row1_left:

            st.text_input(
                "Glucose",
                placeholder="Enter glucose level",
                key=input_keys["glucose"],
                help="Enter your recorded glucose value."
            )


        with row1_right:

            st.text_input(
                "Insulin",
                placeholder="Enter insulin level",
                key=input_keys["insulin"],
                help="Enter your recorded insulin value."
            )


        row2_left, row2_right = (
            st.columns(2)
        )


        with row2_left:

            st.text_input(
                "Weight (kg)",
                placeholder="e.g. 65",
                key=input_keys["weight"],
                help="Enter your body weight in kilograms."
            )


        with row2_right:

            st.text_input(
                "Height (cm)",
                placeholder="e.g. 165",
                key=input_keys["height"],
                help="Enter your height in centimetres."
            )


        row3_left, row3_right = (
            st.columns(2)
        )


        with row3_left:

            st.text_input(
                "Age",
                placeholder="e.g. 30",
                key=input_keys["age"],
                help="Age must be entered as a whole number."
            )


        with row3_right:

            st.markdown(
                """
                **BMI calculation**

                BMI is calculated automatically using:

                `weight ÷ height²`
                """
            )


        st.form_submit_button(
            "Check My Diabetes Risk",
            use_container_width=True,
            on_click=assess_diabetes
        )


    # Hard validation errors
    if st.session_state.validation_message:

        st.error(
            st.session_state.validation_message
        )


    # Soft training-data warnings
    for warning in (
        st.session_state.training_warnings
    ):

        st.warning(
            warning
        )


    st.button(
        "Clear form",
        on_click=clear_everything,
        key="clear_all_button"
    )


# =========================================================
# RIGHT GUIDE
# =========================================================

with right_guide:

    render_html("""
    <div class="quick-card">

    <div class="card-title">
    What happens in the app?
    </div>

    <div class="card-text">
    Your information is checked and prepared before prediction.
    </div>

    <div class="quick-item">
    <div class="quick-name">1. Validate</div>
    <div class="quick-description">
    The app checks for missing or invalid values.
    </div>
    </div>

    <div class="quick-item">
    <div class="quick-name">2. Calculate BMI</div>
    <div class="quick-description">
    Weight and height are converted into BMI in the backend.
    </div>
    </div>

    <div class="quick-item">
    <div class="quick-name">3. Prepare data</div>
    <div class="quick-description">
    Model inputs are scaled using the original preprocessing approach.
    </div>
    </div>

    <div class="quick-item">
    <div class="quick-name">4. Predict</div>
    <div class="quick-description">
    The trained KNN model returns a diabetes-risk classification.
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


    render_html("""
    <div id="assessment-result"></div>
    """)


    render_html("""
    <div class="section-label">
    Original Assessment
    </div>

    <div class="section-title">
    Your diabetes risk result
    </div>

    <div class="section-description">
    This result is your original assessment.
    You can compare it with an adjusted scenario below.
    </div>
    """)


    # =====================================================
    # RESULT CARD
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
        This is a machine-learning screening result
        and is not a medical diagnosis.
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
    # MEASUREMENTS USED
    # =====================================================

    st.markdown("")

    render_html("""
    <div class="section-label">
    Your Information
    </div>

    <div class="section-title">
    Measurements used in the assessment
    </div>
    """)


    values = (
        st.session_state.submitted_values
    )


    row1 = st.columns(3)


    with row1[0]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Glucose
        </div>

        <div class="measurement-value">
        {values["Glucose"]:g}
        </div>

        <div class="measurement-help">
        User-entered glucose value.
        </div>

        </div>
        """)


    with row1[1]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Insulin
        </div>

        <div class="measurement-value">
        {values["Insulin"]:g}
        </div>

        <div class="measurement-help">
        User-entered insulin value.
        </div>

        </div>
        """)


    with row1[2]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Calculated BMI
        </div>

        <div class="measurement-value">
        {values["BMI"]:.2f}
        </div>

        <div class="measurement-help">
        Calculated automatically from weight and height.
        </div>

        </div>
        """)


    st.markdown("")


    row2 = st.columns(3)


    with row2[0]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Weight
        </div>

        <div class="measurement-value">
        {values["Weight"]:g} kg
        </div>

        <div class="measurement-help">
        Used to calculate BMI.
        </div>

        </div>
        """)


    with row2[1]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Height
        </div>

        <div class="measurement-value">
        {values["Height"]:g} cm
        </div>

        <div class="measurement-help">
        Used to calculate BMI.
        </div>

        </div>
        """)


    with row2[2]:

        render_html(f"""
        <div class="measurement-card">

        <div class="measurement-name">
        Age
        </div>

        <div class="measurement-value">
        {values["Age"]:g}
        </div>

        <div class="measurement-help">
        Included as a model input.
        </div>

        </div>
        """)


    # =====================================================
    # SCENARIO EXPLORER
    # =====================================================

    st.markdown("")

    st.divider()


    render_html("""
    <div class="section-label">
    Scenario Explorer
    </div>

    <div class="section-title">
    Compare your original assessment with an adjusted scenario
    </div>

    <div class="section-description">
    Select one measurement and increase it to see whether
    the model's classification changes. This feature is
    for educational model exploration only.
    </div>
    """)


    with st.form(
        "scenario_form"
    ):

        scenario_left, scenario_right = (
            st.columns(2)
        )


        with scenario_left:

            scenario_variable = st.selectbox(
                "Measurement to increase",
                [
                    "Glucose",
                    "Insulin",
                    "Weight",
                    "Height"
                ]
            )


        with scenario_right:

            increase_percentage = st.slider(
                "Increase by",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                format="%d%%"
            )


        compare_scenario = (
            st.form_submit_button(
                "Compare Scenario",
                use_container_width=True
            )
        )


    if compare_scenario:

        adjusted = values.copy()

        adjusted[
            scenario_variable
        ] = (
            adjusted[
                scenario_variable
            ]
            * (
                1
                + increase_percentage / 100
            )
        )


        scenario_error = None


        if (
            adjusted["Weight"] > 500
        ):

            scenario_error = (
                "The adjusted weight exceeds the "
                "supported plausibility limit of 500 kg."
            )


        if (
            adjusted["Height"] > 250
        ):

            scenario_error = (
                "The adjusted height exceeds the "
                "supported plausibility limit of 250 cm."
            )


        if scenario_error:

            st.session_state.scenario_result = None
            st.session_state.scenario_values = None
            st.session_state.scenario_warnings = []

            st.warning(
                scenario_error
            )


        else:

            adjusted_bmi = calculate_bmi(
                adjusted["Weight"],
                adjusted["Height"]
            )

            adjusted["BMI"] = (
                adjusted_bmi
            )


            adjusted_prediction = (
                run_prediction(
                    adjusted["Glucose"],
                    adjusted["Insulin"],
                    adjusted["BMI"],
                    adjusted["Age"]
                )
            )


            scenario_warnings = (
                get_training_warnings(
                    adjusted["Glucose"],
                    adjusted["Insulin"],
                    adjusted["BMI"],
                    adjusted["Age"]
                )
            )


            st.session_state.scenario_result = (
                adjusted_prediction
            )

            st.session_state.scenario_values = (
                adjusted
            )

            st.session_state.scenario_warnings = (
                scenario_warnings
            )


    # =====================================================
    # SHOW SCENARIO COMPARISON
    # =====================================================

    if (
        st.session_state.scenario_result
        is not None
    ):

        adjusted = (
            st.session_state.scenario_values
        )

        baseline_prediction = (
            st.session_state.prediction_result
        )

        adjusted_prediction = (
            st.session_state.scenario_result
        )


        comparison1, comparison2 = (
            st.columns(2)
        )


        with comparison1:

            st.markdown(
                "### Original Assessment"
            )

            st.metric(
                "Model classification",
                result_label(
                    baseline_prediction
                )
            )


        with comparison2:

            st.markdown(
                "### Adjusted scenario"
            )

            st.metric(
                "Model classification",
                result_label(
                    adjusted_prediction
                )
            )


        comparison_table = pd.DataFrame(
            {
                "Measurement": [
                    "Glucose",
                    "Insulin",
                    "Weight (kg)",
                    "Height (cm)",
                    "Calculated BMI",
                    "Age"
                ],

                "Original Assessment": [
                    values["Glucose"],
                    values["Insulin"],
                    values["Weight"],
                    values["Height"],
                    round(
                        values["BMI"],
                        2
                    ),
                    values["Age"]
                ],

                "Adjusted scenario": [
                    round(
                        adjusted["Glucose"],
                        2
                    ),
                    round(
                        adjusted["Insulin"],
                        2
                    ),
                    round(
                        adjusted["Weight"],
                        2
                    ),
                    round(
                        adjusted["Height"],
                        2
                    ),
                    round(
                        adjusted["BMI"],
                        2
                    ),
                    adjusted["Age"]
                ]
            }
        )


        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True
        )


        if (
            baseline_prediction
            != adjusted_prediction
        ):

            st.info(
                "The model classification changed "
                "after the selected measurement was increased."
            )


        else:

            st.info(
                "The model classification remained the same "
                "for this adjusted scenario."
            )


        for warning in (
            st.session_state.scenario_warnings
        ):

            st.warning(
                warning
            )


        st.caption(
            "Scenario results demonstrate model behaviour only. "
            "They should not be interpreted as a clinical forecast."
        )


    # =====================================================
    # RESULT INFORMATION TABS
    # =====================================================

    st.markdown("")

    tab1, tab2, tab3, tab4 = (
        st.tabs(
            [
                "Next steps",
                "Everyday health",
                "Understand your inputs",
                "Technical notes"
            ]
        )
    )


    # -----------------------------------------------------
    # NEXT STEPS
    # -----------------------------------------------------

    with tab1:

        if (
            st.session_state.prediction_result
            == 1
        ):

            st.markdown(
                """
                ### What can you do next?

                Your original assessment returned a
                **higher predicted likelihood**.

                - Keep a record of the measurements used.
                - If you are concerned about the result, consider speaking with a qualified healthcare professional.
                - Do not use this screening result as a medical diagnosis.
                """
            )


        else:

            st.markdown(
                """
                ### What can you do next?

                Your original assessment returned a
                **lower predicted likelihood**.

                - Continue regular health checks where appropriate.
                - Keep track of important health measurements over time.
                - A lower predicted likelihood does not rule out diabetes or other health conditions.
                """
            )


    # -----------------------------------------------------
    # EVERYDAY HEALTH
    # -----------------------------------------------------

    with tab2:

        wellness1, wellness2, wellness3 = (
            st.columns(3)
        )


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
            Regular physical activity can support
            overall health, mobility and weight management.
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

        learn1, learn2 = (
            st.columns(2)
        )


        with learn1:

            render_html("""
            <div class="learn-card">

            <div class="learn-title">
            Glucose
            </div>

            <div class="learn-text">
            Glucose refers to sugar circulating in the blood.
            Its interpretation can depend on when and how
            the measurement was taken.
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
            A single reading should be interpreted in its correct
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
            Body Mass Index is calculated using weight and height.
            In this application, BMI is calculated automatically
            in the backend before prediction.
            </div>

            </div>
            """)


            render_html("""
            <div class="learn-card" style="margin-top:15px;">

            <div class="learn-title">
            Age
            </div>

            <div class="learn-text">
            Age is included as contextual information used by
            the prediction model alongside the other model inputs.
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

            **User-entered information:**  
            Glucose, Insulin, Weight, Height and Age

            **Model inputs:**  
            Glucose, Insulin, calculated BMI and Age

            **BMI calculation:**  
            BMI is calculated automatically from weight and height before prediction.

            **Prediction type:**  
            Binary classification

            **Validation:**  
            The application checks required fields, numeric values, positive values, age format and basic human plausibility.

            **Training-data warning:**  
            Values outside ranges represented in the supplied training dataset are allowed, but the user is warned that the prediction may be less reliable.

            **Preprocessing:**  
            Zero values in selected medical fields are handled as missing values, missing values are filled using the feature mean, and model inputs are scaled using the same MinMax approach used in the supplied model development notebook.

            **Technology:**  
            Python, Scikit-learn, Pandas, NumPy and Streamlit
            """
        )


    # =====================================================
    # AUTO-SCROLL
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
# DATASET EXPLORER
# =========================================================

st.markdown("")

st.divider()


render_html("""
<div class="section-label">
Dataset Explorer
</div>

<div class="section-title">
Explore the data behind the model
</div>

<div class="section-description">
The application uses the supplied diabetes dataset for preprocessing
and model context. You can inspect the original records,
summary statistics and target distribution below.
</div>
""")


dataset_metric1, dataset_metric2, dataset_metric3, dataset_metric4 = (
    st.columns(4)
)


with dataset_metric1:

    st.metric(
        "Records",
        f"{len(dataset):,}"
    )


with dataset_metric2:

    st.metric(
        "Columns",
        len(dataset.columns)
    )


with dataset_metric3:

    diabetes_count = int(
        (
            dataset["Outcome"]
            == 1
        ).sum()
    )

    st.metric(
        "Outcome = 1",
        diabetes_count
    )


with dataset_metric4:

    non_diabetes_count = int(
        (
            dataset["Outcome"]
            == 0
        ).sum()
    )

    st.metric(
        "Outcome = 0",
        non_diabetes_count
    )


# =========================================================
# FULL DATASET
# =========================================================

with st.expander(
    "View the full diabetes dataset",
    expanded=False
):

    st.caption(
        "Scroll vertically or horizontally to inspect the original records."
    )

    st.dataframe(
        dataset,
        use_container_width=True,
        height=420
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

with st.expander(
    "View summary statistics",
    expanded=False
):

    summary = (
        dataset
        .describe()
        .T
        .round(2)
    )

    st.dataframe(
        summary,
        use_container_width=True
    )


# =========================================================
# ZERO / MISSING-VALUE CONTEXT
# =========================================================

with st.expander(
    "View preprocessing context",
    expanded=False
):

    zero_missing_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    zero_summary = pd.DataFrame(
        {
            "Feature": zero_missing_columns,

            "Zero values in original dataset": [
                int(
                    (
                        dataset[column]
                        == 0
                    ).sum()
                )

                for column
                in zero_missing_columns
            ]
        }
    )


    st.markdown(
        """
        In the supplied model-development workflow,
        zero values in selected medical measurements
        are treated as missing values before preprocessing.
        """
    )


    st.dataframe(
        zero_summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# OUTCOME DISTRIBUTION
# =========================================================

with st.expander(
    "View diabetes outcome distribution",
    expanded=False
):

    outcome_counts = (
        dataset["Outcome"]
        .value_counts()
        .sort_index()
    )


    outcome_table = pd.DataFrame(
        {
            "Outcome": [
                "0 - No diabetes",
                "1 - Diabetes"
            ],

            "Number of records": [
                int(
                    outcome_counts.get(
                        0,
                        0
                    )
                ),

                int(
                    outcome_counts.get(
                        1,
                        0
                    )
                )
            ]
        }
    )


    st.dataframe(
        outcome_table,
        use_container_width=True,
        hide_index=True
    )


    chart_data = pd.DataFrame(
        {
            "Records": [
                int(
                    outcome_counts.get(
                        0,
                        0
                    )
                ),

                int(
                    outcome_counts.get(
                        1,
                        0
                    )
                )
            ]
        },

        index=[
            "No diabetes",
            "Diabetes"
        ]
    )


    st.bar_chart(
        chart_data
    )


# =========================================================
# DISCLAIMER
# =========================================================

render_html("""
<div class="disclaimer">

<strong>Important:</strong>

This application was developed for educational purposes.
Its output is generated by a machine learning model and does not
provide a medical diagnosis or replace professional medical advice.

The Scenario Explorer demonstrates how model output may change
when selected inputs are adjusted. It should not be interpreted
as a prediction of how a person's health will change.

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