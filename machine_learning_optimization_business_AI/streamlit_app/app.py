"""Menolyte Core — churn risk platform for mid-sized ISPs.

Loads the pipeline and results exported by telco_churn_optimization_xai.ipynb. Nothing is retrained here.
Run locally from the project folder:  streamlit run streamlit_app/app.py
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIRECTORY / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIRECTORY = PROJECT_DIRECTORY / "models"

HIGHLIGHT_COLOR = "#d62728"
NEUTRAL_COLOR = "#b0b7c3"
RETENTION_COLOR = "#1f77b4"
RISK_SEGMENT_NAMES = ["Low risk", "Medium risk", "High risk"]

st.set_page_config(page_title="Menolyte Core", page_icon="📡", layout="wide")


# ----------------------------------------------------------------------------------------------------------------------
# Loading (cached: the model and results are read once per session)
# ----------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_results():
    final_pipeline = joblib.load(MODELS_DIRECTORY / "final_svm_pipeline.joblib")
    app_results = joblib.load(MODELS_DIRECTORY / "app_results.joblib")
    return final_pipeline, app_results


@st.cache_data
def load_customer_data():
    customer_data = pd.read_csv(DATA_PATH)
    customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"].str.strip().replace("", "0"))
    customer_data["churned"] = customer_data["Churn"].map({"Yes": 1, "No": 0})
    return customer_data


final_pipeline, app_results = load_model_and_results()
customer_data = load_customer_data()
metadata = app_results["metadata"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
category_values = metadata["category_values"]
feature_columns = list(app_results["features_train"].columns)
risk_segment_edges = metadata["risk_segment_edges"]

fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
fitted_svm = final_pipeline.named_steps["model"]
transformed_feature_names = fitted_preprocessor.get_feature_names_out()


def original_variable_of(transformed_name):
    without_prefix = transformed_name.split("__", 1)[1]
    return next(column for column in feature_columns if without_prefix == column or without_prefix.startswith(column + "_"))


transformed_to_original = [original_variable_of(name) for name in transformed_feature_names]


def risk_segment_of(probability):
    for segment_name, upper_edge in zip(RISK_SEGMENT_NAMES, risk_segment_edges[1:]):
        if probability < upper_edge:
            return segment_name
    return RISK_SEGMENT_NAMES[-1]


@st.cache_resource
def build_shap_explainer():
    return shap.KernelExplainer(lambda transformed_rows: fitted_svm.predict_proba(transformed_rows)[:, 1],
                                app_results["shap_background"])


def encode_for_lime(original_rows):
    encoded_rows = original_rows[feature_columns].copy()
    for column in categorical_features:
        encoded_rows[column] = encoded_rows[column].map({value: code for code, value in enumerate(category_values[column])})
    return encoded_rows.to_numpy(dtype=float)


def churn_probability_from_lime_rows(encoded_rows):
    decoded_rows = pd.DataFrame(encoded_rows, columns=feature_columns)
    for column in categorical_features:
        decoded_rows[column] = [category_values[column][int(round(code))] for code in decoded_rows[column]]
    for column in numeric_features:
        decoded_rows[column] = decoded_rows[column].astype(float)
    return final_pipeline.predict_proba(decoded_rows)


@st.cache_resource
def build_lime_explainer():
    categorical_indices = [feature_columns.index(column) for column in categorical_features]
    return LimeTabularExplainer(
        training_data=encode_for_lime(app_results["features_train"]),
        feature_names=feature_columns,
        categorical_features=categorical_indices,
        categorical_names={index: category_values[column] for index, column in zip(categorical_indices, categorical_features)},
        class_names=["Stays", "Churns"],
        discretize_continuous=True,
        mode="classification",
        random_state=42,
    )


PLAIN_LANGUAGE_LABELS = {
    "tenure": "Time as a customer", "Contract": "Contract type", "MonthlyCharges": "Monthly fee",
    "TotalCharges": "Total paid so far", "PaymentMethod": "Payment method", "OnlineSecurity": "Online security",
    "TechSupport": "Technical support", "InternetService": "Internet type", "SeniorCitizen": "Senior citizen",
    "PaperlessBilling": "Paperless billing", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "MultipleLines": "Multiple lines",
    "PhoneService": "Phone service", "Dependents": "Dependents", "Partner": "Partner", "gender": "Gender",
}


def describe_value(variable, value):
    if variable == "tenure":
        return f"{int(value)} month" + ("" if int(value) == 1 else "s")
    if variable in ["MonthlyCharges", "TotalCharges"]:
        return f"${value:,.0f}"
    if variable == "SeniorCitizen":
        return "Yes" if value == 1 else "No"
    return str(value)


def suggested_action(variable, value):
    """Retention action for a risk driver (hypotheses to validate with an A/B test, not causal guarantees)."""
    if variable == "Contract" and value == "Month-to-month":
        return "Offer a discounted one- or two-year contract."
    if variable == "tenure" and value <= 12:
        return "Enroll in the first-months onboarding program (welcome call, usage check-in)."
    if variable == "MonthlyCharges" and value >= 70:
        return "Review the plan: suggest a cheaper plan that fits real usage, or a loyalty discount."
    if variable in ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"] and value == "No":
        return f"Offer a free trial of {PLAIN_LANGUAGE_LABELS[variable].lower()} for a few months."
    if variable == "PaymentMethod" and value == "Electronic check":
        return "Offer a small bill credit for switching to automatic payment."
    if variable == "InternetService" and value == "Fiber optic":
        return "Check fiber service quality and price competitiveness for this customer."
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------------------------------------
st.title("📡 Menolyte Core")
st.caption("Churn risk prediction and explainable retention for mid-sized internet service providers")

eda_tab, performance_tab, prediction_tab, business_tab = st.tabs(
    ["📊 Exploratory Data Analysis", "🧪 Model Performance", "🔍 Prediction & Explainability", "💼 Business Application"])

# ----------------------------------------------------------------------------------------------------------------------
# A. Exploratory Data Analysis
# ----------------------------------------------------------------------------------------------------------------------
with eda_tab:
    st.subheader("Explore the customer base")
    filter_columns = st.columns(4)
    selected_contracts = filter_columns[0].multiselect("Contract", category_values["Contract"], default=category_values["Contract"])
    selected_internet = filter_columns[1].multiselect("Internet service", category_values["InternetService"],
                                                      default=category_values["InternetService"])
    selected_payment = filter_columns[2].multiselect("Payment method", category_values["PaymentMethod"],
                                                     default=category_values["PaymentMethod"])
    minimum_tenure, maximum_tenure = int(customer_data["tenure"].min()), int(customer_data["tenure"].max())
    selected_tenure = filter_columns[3].slider("Months as a customer", minimum_tenure, maximum_tenure,
                                               (minimum_tenure, maximum_tenure))

    segment_data = customer_data[customer_data["Contract"].isin(selected_contracts)
                                 & customer_data["InternetService"].isin(selected_internet)
                                 & customer_data["PaymentMethod"].isin(selected_payment)
                                 & customer_data["tenure"].between(*selected_tenure)]

    if segment_data.empty:
        st.warning("No customers match these filters.")
    else:
        overall_churn_rate = customer_data["churned"].mean()
        segment_churn_rate = segment_data["churned"].mean()
        kpi_columns = st.columns(4)
        kpi_columns[0].metric("Customers in segment", f"{len(segment_data):,}")
        kpi_columns[1].metric("Churn rate", f"{segment_churn_rate:.1%}",
                              delta=f"{(segment_churn_rate - overall_churn_rate) * 100:+.1f} pts vs all customers",
                              delta_color="inverse")
        kpi_columns[2].metric("Average monthly fee", f"${segment_data['MonthlyCharges'].mean():,.2f}")
        kpi_columns[3].metric("Monthly revenue lost to churn",
                              f"${segment_data.loc[segment_data['churned'] == 1, 'MonthlyCharges'].sum():,.0f}")

        chart_columns = st.columns(2)
        with chart_columns[0]:
            breakdown_variable = st.selectbox("Churn rate by", categorical_features,
                                              index=categorical_features.index("Contract"))
            churn_by_category = (segment_data.groupby(breakdown_variable)["churned"].agg(["mean", "size"])
                                 .reset_index().rename(columns={"mean": "churn_rate", "size": "customers"}))
            churn_by_category[breakdown_variable] = churn_by_category[breakdown_variable].astype(str)
            figure = px.bar(churn_by_category, x=breakdown_variable, y="churn_rate", text_auto=".0%",
                            hover_data=["customers"], color_discrete_sequence=[RETENTION_COLOR])
            figure.add_hline(y=overall_churn_rate, line_dash="dash", line_color=HIGHLIGHT_COLOR,
                             annotation_text="All customers")
            figure.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", title=f"Churn rate by {breakdown_variable}")
            st.plotly_chart(figure, width="stretch")
        with chart_columns[1]:
            figure = px.histogram(segment_data, x="tenure", color="Churn", barmode="overlay", histnorm="percent", nbins=36,
                                  color_discrete_map={"Yes": HIGHLIGHT_COLOR, "No": NEUTRAL_COLOR},
                                  title="Months as a customer: who stays vs who leaves")
            figure.update_layout(xaxis_title="Months as a customer", yaxis_title="% of group")
            st.plotly_chart(figure, width="stretch")

        figure = px.box(segment_data, x="Churn", y="MonthlyCharges", color="Churn",
                        color_discrete_map={"Yes": HIGHLIGHT_COLOR, "No": NEUTRAL_COLOR},
                        title="Monthly fee of customers who stay vs leave")
        figure.update_layout(yaxis_title="Monthly fee (USD)", showlegend=False)
        st.plotly_chart(figure, width="stretch")

# ----------------------------------------------------------------------------------------------------------------------
# B. Model Performance
# ----------------------------------------------------------------------------------------------------------------------
with performance_tab:
    st.subheader("Final model: SVM (RBF kernel) tuned with Grid Search")
    hyperparameters = metadata["hyperparameters"]
    st.caption(f"C = {hyperparameters['C']:.3f}, gamma = {hyperparameters['gamma']:.4f}, class_weight = balanced. "
               "Primary metric: F2 (recall weighs twice as much as precision, because missing a churner is the costliest error).")

    metric_columns = st.columns(5)
    for column, metric_name in zip(metric_columns, ["f2", "recall", "precision", "f1", "roc_auc"]):
        test_value = metadata["test_metrics"][metric_name]
        cross_validation_value = metadata["cross_validation_metrics"][metric_name]
        column.metric(f"{metric_name.upper()} (test)", f"{test_value:.3f}",
                      delta=f"{test_value - cross_validation_value:+.3f} vs cross-validation")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        confusion_matrix = app_results["test_confusion_matrix"].to_numpy()
        figure = px.imshow(confusion_matrix, text_auto=True, color_continuous_scale="Blues",
                           x=["Predicted: stays", "Predicted: churns"], y=["Actual: stays", "Actual: churns"],
                           title="Confusion matrix on the test set (1,409 unseen customers)")
        figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure, width="stretch")
    with chart_columns[1]:
        model_comparison = app_results["model_comparison_cv"]
        metric_means = (model_comparison[[f"{metric}_mean" for metric in ["f2", "recall", "precision", "f1", "roc_auc"]]]
                        .rename(columns=lambda column: column.replace("_mean", "").upper())
                        .reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="score"))
        figure = px.bar(metric_means, x="metric", y="score", color="model", barmode="group",
                        title="Model comparison (5-fold cross-validation, default hyperparameters)")
        figure.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(figure, width="stretch")

    st.markdown("#### Hyperparameter optimization (same model, folds and budget of 36 configurations)")
    strategies = app_results["optimization_strategies"].reset_index(names="strategy")
    st.dataframe(strategies[["strategy", "configurations_evaluated", "optimization_seconds", "best_C", "best_gamma",
                             "cv_f2", "cv_recall", "cv_precision", "f2_gain_vs_default_svm"]].round(4),
                 hide_index=True, width="stretch")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        figure = px.bar(strategies, x="strategy", y="cv_f2", text_auto=".4f", title="Best cross-validated F2",
                        color_discrete_sequence=[RETENTION_COLOR])
        figure.update_layout(yaxis_range=[strategies["cv_f2"].min() - 0.02, strategies["cv_f2"].max() + 0.01])
        st.plotly_chart(figure, width="stretch")
    with chart_columns[1]:
        figure = px.bar(strategies, x="strategy", y="optimization_seconds", text_auto=".1f",
                        title="Optimization time (seconds)", color_discrete_sequence=["darkorange"])
        st.plotly_chart(figure, width="stretch")

    st.markdown("#### Multi-objective optimization (NSGA-II): F2 vs inference time")
    nsga2_trials = app_results["nsga2_trials"]
    pareto_front = nsga2_trials[nsga2_trials["is_pareto_optimal"]].sort_values("ms_per_1000_customers")
    grid_best = app_results["grid_best_speed"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=nsga2_trials.loc[~nsga2_trials["is_pareto_optimal"], "ms_per_1000_customers"],
                                y=nsga2_trials.loc[~nsga2_trials["is_pareto_optimal"], "cv_f2"], mode="markers",
                                marker={"color": "lightgray", "size": 9}, name="Dominated configurations"))
    figure.add_trace(go.Scatter(x=pareto_front["ms_per_1000_customers"], y=pareto_front["cv_f2"], mode="lines+markers",
                                marker={"color": HIGHLIGHT_COLOR, "size": 11}, line={"color": HIGHLIGHT_COLOR},
                                name="Pareto front", customdata=pareto_front[["C", "gamma"]],
                                hovertemplate="F2 %{y:.4f}<br>%{x:.0f} ms<br>C %{customdata[0]:.3f}, gamma %{customdata[1]:.4f}"))
    figure.add_trace(go.Scatter(x=[grid_best["ms_per_1000_customers"]], y=[grid_best["cv_f2"]], mode="markers",
                                marker={"symbol": "star", "size": 22, "color": RETENTION_COLOR, "line": {"color": "black", "width": 1}},
                                name="Selected model (Grid Search winner)"))
    figure.update_layout(xaxis_title="Inference time (ms per 1,000 customers) — lower is better",
                         yaxis_title="Cross-validated F2 — higher is better", height=450)
    st.plotly_chart(figure, width="stretch")
    st.caption("The selected model is not dominated: it sits at the high-F2 end of the front. The faster 'knee' "
               "solutions save a few hundredths of a millisecond per customer, irrelevant at this scale.")

    st.markdown("#### What drives the predictions: SHAP (selected SVM) vs Random Forest importance")
    importance = app_results["importance_comparison"][["SHAP (SVM)", "Random Forest MDI"]].iloc[::-1].reset_index(names="variable")
    figure = px.bar(importance.melt(id_vars="variable", var_name="method", value_name="normalized importance"),
                    x="normalized importance", y="variable", color="method", barmode="group", orientation="h", height=600)
    st.plotly_chart(figure, width="stretch")

# ----------------------------------------------------------------------------------------------------------------------
# C. Prediction & Explainability
# ----------------------------------------------------------------------------------------------------------------------
with prediction_tab:
    st.subheader("Score a customer and understand why")

    test_predictions = app_results["test_predictions"]
    example_customers = {
        "Typical new customer (manual input)": None,
        "Example: highest-risk test customer": test_predictions.iloc[0]["customer_index"],
        "Example: lowest-risk test customer": test_predictions.iloc[-1]["customer_index"],
    }
    example_choice = st.selectbox("Start from", list(example_customers))
    example_index = example_customers[example_choice]
    if example_index is None:
        default_values = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
                          "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                          "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                          "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
                          "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
                          "TotalCharges": 255.0}
    else:
        default_values = customer_data.loc[int(example_index), feature_columns].to_dict()

    with st.form("customer_form"):
        input_values = {}
        form_columns = st.columns(3)
        column_groups = {
            "Profile": ["gender", "SeniorCitizen", "Partner", "Dependents"],
            "Account": ["tenure", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
            "Services": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                         "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
        }
        for form_column, (group_title, group_variables) in zip(form_columns, column_groups.items()):
            form_column.markdown(f"**{group_title}**")
            for variable in group_variables:
                widget_key = f"{example_choice}_{variable}"
                label = PLAIN_LANGUAGE_LABELS[variable]
                if variable == "tenure":
                    input_values[variable] = form_column.slider(label + " (months)", 0, 72, int(default_values[variable]), key=widget_key)
                elif variable in ["MonthlyCharges", "TotalCharges"]:
                    input_values[variable] = form_column.number_input(label + " (USD)", 0.0, 10000.0,
                                                                      float(default_values[variable]), key=widget_key)
                else:
                    options = category_values[variable]
                    input_values[variable] = form_column.selectbox(label, options, index=options.index(default_values[variable]),
                                                                   key=widget_key)
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        customer_row = pd.DataFrame([input_values])[feature_columns]
        churn_probability = final_pipeline.predict_proba(customer_row)[0, 1]
        predicted_churn = final_pipeline.predict(customer_row)[0]
        risk_segment = risk_segment_of(churn_probability)

        result_columns = st.columns(3)
        result_columns[0].metric("Churn probability", f"{churn_probability:.0%}")
        result_columns[1].metric("Risk segment", risk_segment)
        result_columns[2].metric("Model decision", "Likely to leave" if predicted_churn == 1 else "Likely to stay")
        st.progress(float(churn_probability))

        with st.spinner("Explaining the prediction with SHAP and LIME..."):
            transformed_customer = fitted_preprocessor.transform(customer_row)
            transformed_shap_values = build_shap_explainer().shap_values(transformed_customer, nsamples=300, silent=True)[0]
            shap_by_variable = (pd.Series(transformed_shap_values, index=transformed_feature_names)
                                .groupby(transformed_to_original).sum())
            lime_explanation = build_lime_explainer().explain_instance(
                encode_for_lime(customer_row)[0], churn_probability_from_lime_rows, num_features=8, num_samples=1500, labels=(1,))

        top_drivers = shap_by_variable.reindex(shap_by_variable.abs().sort_values(ascending=False).index).head(8)

        st.markdown("#### Why? The main factors behind this prediction")
        explanation_columns = st.columns(2)
        with explanation_columns[0]:
            shap_chart_data = pd.DataFrame({
                "factor": [f"{PLAIN_LANGUAGE_LABELS[variable]}: {describe_value(variable, input_values[variable])}"
                           for variable in top_drivers.index],
                "risk points": top_drivers.values * 100,
            }).iloc[::-1]
            figure = px.bar(shap_chart_data, x="risk points", y="factor", orientation="h", text_auto=".1f",
                            color=np.where(shap_chart_data["risk points"] > 0, "Raises risk", "Lowers risk"),
                            color_discrete_map={"Raises risk": HIGHLIGHT_COLOR, "Lowers risk": RETENTION_COLOR},
                            title=f"SHAP: points of risk added to the average customer ({metadata['shap_base_value']:.0%})")
            figure.update_layout(xaxis_title="Points of churn risk", yaxis_title="", legend_title="")
            st.plotly_chart(figure, width="stretch")
        with explanation_columns[1]:
            lime_weights = pd.DataFrame(lime_explanation.as_list(label=1), columns=["condition", "weight"]).iloc[::-1]
            figure = px.bar(lime_weights, x="weight", y="condition", orientation="h", text_auto=".2f",
                            color=np.where(lime_weights["weight"] > 0, "Raises risk", "Lowers risk"),
                            color_discrete_map={"Raises risk": HIGHLIGHT_COLOR, "Lowers risk": RETENTION_COLOR},
                            title="LIME: local linear approximation around this customer")
            figure.update_layout(xaxis_title="Effect on churn probability", yaxis_title="", legend_title="")
            st.plotly_chart(figure, width="stretch")

        risk_raising_drivers = [variable for variable in top_drivers.index if top_drivers[variable] > 0]
        summary_factors = ", ".join(f"{PLAIN_LANGUAGE_LABELS[variable].lower()} ({describe_value(variable, input_values[variable])})"
                                    for variable in risk_raising_drivers[:3])
        st.markdown("#### In plain words")
        if risk_raising_drivers:
            st.info(f"This customer has a **{churn_probability:.0%}** probability of leaving. "
                    f"The factors that raise the risk the most are: {summary_factors}.")
        else:
            st.success(f"This customer has a **{churn_probability:.0%}** probability of leaving; no factor raises the risk.")

        actions = [action for action in (suggested_action(variable, input_values[variable]) for variable in risk_raising_drivers) if action]
        if actions and churn_probability >= risk_segment_edges[1]:
            st.markdown("#### Suggested retention actions")
            for action in dict.fromkeys(actions):
                st.markdown(f"- {action}")
        st.caption("Explanations show what the model associates with churn, not proven causes. "
                   "Suggested actions are hypotheses to validate with a controlled test (contacted vs control group).")

# ----------------------------------------------------------------------------------------------------------------------
# D. Business Application
# ----------------------------------------------------------------------------------------------------------------------
with business_tab:
    st.subheader("Menolyte Core for retention managers")
    st.markdown("Every month, Menolyte Core scores the whole customer base, ranks customers by churn risk, explains "
                "each case and estimates the economic impact of a retention campaign. "
                "Below: results on **1,409 test customers** the model never saw during training.")

    ranked_customers = app_results["test_predictions"].copy()
    ranked_customers["risk_segment"] = ranked_customers["churn_probability"].map(risk_segment_of)
    high_risk_customers = ranked_customers[ranked_customers["risk_segment"] == "High risk"]

    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Customers scored", f"{len(ranked_customers):,}")
    kpi_columns[1].metric("High-risk customers", f"{len(high_risk_customers):,}",
                          help=f"Churn probability ≥ {risk_segment_edges[2]:.0%}")
    kpi_columns[2].metric("Monthly revenue in high risk", f"${high_risk_customers['MonthlyCharges'].sum():,.0f}")
    kpi_columns[3].metric("Actual churn in high-risk segment", f"{high_risk_customers['actual_churn'].mean():.0%}",
                          help="Share of the high-risk customers who actually left (test set)")

    st.markdown("#### How many customers can your team contact?")
    contact_share = st.slider("Share of customers contacted (riskiest first)", 5, 100, 20, step=5, format="%d%%") / 100
    contacted_customers = ranked_customers[ranked_customers["share_of_customers_contacted"] <= contact_share]
    churners_reached = contacted_customers["actual_churn"].sum() / ranked_customers["actual_churn"].sum()
    reach_columns = st.columns(3)
    reach_columns[0].metric("Customers to contact", f"{len(contacted_customers):,}")
    reach_columns[1].metric("Churners reached", f"{churners_reached:.0%}",
                            delta=f"{(churners_reached - contact_share) * 100:+.0f} pts vs random selection")
    reach_columns[2].metric("Monthly revenue of churners reached",
                            f"${contacted_customers.loc[contacted_customers['actual_churn'] == 1, 'MonthlyCharges'].sum():,.0f}")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=ranked_customers["share_of_customers_contacted"], y=ranked_customers["share_of_churners_reached"],
                                    mode="lines", line={"color": HIGHLIGHT_COLOR, "width": 3}, name="Menolyte Core ranking"))
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": NEUTRAL_COLOR, "dash": "dash"},
                                    name="Random selection"))
        figure.add_vline(x=contact_share, line_dash="dot")
        figure.update_layout(title="Churners reached vs customers contacted", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                             xaxis_title="Customers contacted", yaxis_title="Churners reached")
        st.plotly_chart(figure, width="stretch")
    with chart_columns[1]:
        segment_summary = (ranked_customers.groupby("risk_segment")
                           .agg(customers=("actual_churn", "size"), actual_churn_rate=("actual_churn", "mean"),
                                monthly_revenue=("MonthlyCharges", "sum"))
                           .reindex(RISK_SEGMENT_NAMES).reset_index())
        figure = px.bar(segment_summary, x="risk_segment", y="actual_churn_rate", text_auto=".0%", color="risk_segment",
                        color_discrete_map={"Low risk": NEUTRAL_COLOR, "Medium risk": "#f2a65a", "High risk": HIGHLIGHT_COLOR},
                        hover_data=["customers", "monthly_revenue"], title="Actual churn rate by risk segment")
        figure.update_layout(yaxis_tickformat=".0%", showlegend=False, xaxis_title="", yaxis_title="Customers who actually left")
        st.plotly_chart(figure, width="stretch")

    st.markdown("#### Priority list for the retention team")
    priority_list = (contacted_customers.merge(customer_data[["customerID"]], left_on="customer_index", right_index=True)
                     [["customerID", "churn_probability", "risk_segment", "MonthlyCharges", "tenure", "Contract"]]
                     .rename(columns={"churn_probability": "churn risk", "MonthlyCharges": "monthly fee",
                                      "tenure": "months as customer"}))
    st.dataframe(priority_list.style.format({"churn risk": "{:.0%}", "monthly fee": "${:,.2f}"}),
                 hide_index=True, width="stretch", height=300)
    st.download_button("Download priority list (CSV)", priority_list.to_csv(index=False), "menolyte_priority_list.csv", "text/csv")

    st.markdown("#### Coming soon: Menolyte multi-agent system")
    roadmap_columns = st.columns(3)
    roadmap_columns[0].info("**🧭 Orchestrator agent**\n\nCoordinates the monthly cycle: scores the base, triggers the "
                            "analyst and the campaign agent, and collects results.")
    roadmap_columns[1].info("**📈 Analyst agent**\n\nReads predictions, segments and SHAP explanations and writes "
                            "recommendations for the manager in plain language.")
    roadmap_columns[2].info("**💬 Campaign agent (WhatsApp)**\n\nContacts approved high-risk customers with "
                            "personalized offers and records the outcome.")
    st.caption("Human in the loop: the manager approves every campaign. Customers are contacted only with prior consent, "
               "in line with Ecuador's Personal Data Protection Law (LOPDP).")
