import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

st.set_page_config(page_title="Real Estate Regression Dashboard", layout="wide")
st.title("🏡 Real Estate Regression Analysis")

# Load dataset directly from local file
estate = pd.read_csv("real_estate_dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(estate.head(), height=200)

# Tabs for compact layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Correlation", "📈 Regression", "📉 Residuals", "🔮 Predictions"])

# --- Tab 1: Correlation ---
with tab1:
    st.subheader("Correlation Heatmap")
    corr = estate.corr()
    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Tab 2: Regression ---
with tab2:
    st.subheader("Regression Modeling")
    predictors = st.multiselect(
        "Select predictors for regression:",
        [c for c in estate.columns if c not in ["Price"]]
    )

    model = None
    if predictors:
        X = estate[predictors]
        X = sm.add_constant(X)
        y = np.log(estate["Price"])
        model = sm.OLS(y, X).fit()

        with st.expander("See full regression summary"):
            st.text(model.summary())

        st.write("### Coefficients")
        coef_table = pd.DataFrame({
            "Predictor": model.params.index,
            "Estimate": model.params.values,
            "p-value": model.pvalues.values
        })
        st.dataframe(coef_table)

# --- Tab 3: Residuals ---
with tab3:
    st.subheader("Residual Diagnostics")
    if model is not None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=150)
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax,
                      line_kws={"color":"red"})
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(5,4), dpi=150)
        sm.qqplot(model.resid, line='45', ax=ax)
        ax.set_title("Q-Q Plot of Residuals")
        st.pyplot(fig)

# --- Tab 4: Predictions ---
with tab4:
    st.subheader("Make Predictions")
    if model is not None:
        st.write("Enter values for predictors to estimate Price:")

        input_data = {}
        for pred in predictors:
            val = st.number_input(f"{pred}", value=float(estate[pred].mean()))
            input_data[pred] = val

        new_X = pd.DataFrame([input_data])
        new_X = sm.add_constant(new_X)
        new_X = new_X.reindex(columns=model.model.exog_names, fill_value=0)

        log_pred = model.predict(new_X)[0]
        price_pred = np.exp(log_pred)

        st.write(f"**Predicted logPrice:** {log_pred:.4f}")
        st.write(f"**Predicted Price:** {price_pred:.2f}")

#streamlit run housePrice.py