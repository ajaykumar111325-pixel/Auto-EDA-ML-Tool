import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Advanced AI Data Analyst", layout="wide")
st.title("🤖 Advanced AI Data Analyst + ML Tool (Enhanced Version)")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    # Load Data
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📌 Data Preview")
    st.write(df.head())

    # ---------------- BASIC INFO ----------------
    st.subheader("📊 Basic Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data Types:")
    st.write(df.dtypes)

    # ---------------- MISSING VALUES ----------------
    st.subheader("⚠️ Missing Values")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_percent})
    st.write(missing_df)

    # ---------------- DUPLICATES ----------------
    st.subheader("🧾 Duplicate Records")
    duplicates = df.duplicated().sum()
    st.write(f"Total Duplicate Rows: {duplicates}")

    if duplicates > 0:
        dup_percent = (duplicates / len(df)) * 100
        st.write(f"Duplicate %: {dup_percent:.2f}%")
        st.warning("⚠️ Dataset contains duplicate records")
    else:
        st.success("✅ No duplicate records found")

    # ---------------- SMART COLUMN INSIGHTS ----------------
    st.subheader("🔹 Smart Column Insights")

    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in df.columns:
        st.write(f"### {col}")
        st.write(f"Unique Values: {df[col].nunique()}")

        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(f"Mean: {df[col].mean():.2f}")
            st.write(f"Median: {df[col].median():.2f}")
            st.write(f"Max: {df[col].max()}")
            st.write(f"Min: {df[col].min()}")
            st.write(f"Std: {df[col].std():.2f}")
            st.write(f"Skewness: {df[col].skew():.2f}")
            st.write(f"Kurtosis: {df[col].kurt():.2f}")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            st.write(f"Start Date: {df[col].min()}")
            st.write(f"End Date: {df[col].max()}")

        else:
            st.write("Top 5 frequent values:")
            st.write(df[col].value_counts().head())

    # ---------------- OUTLIER DETECTION ----------------
    st.subheader("🚨 Outlier Detection")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        st.write(f"{col}: {len(outliers)} outliers")

    # ---------------- AUTO CLEAN ----------------
    if st.button("🧹 Auto Clean Data"):
        df = df.drop_duplicates()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_val = df[col].mode()
                df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)

        st.success("✅ Data Cleaned")
        st.write(df.head())

    # ---------------- VISUALIZATION ----------------
    st.subheader("📈 Visualizations")

    if len(numeric_cols) > 0:
        col = st.selectbox("Select Column", numeric_cols)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df[col], kde=True, ax=ax[0], color="skyblue")
        ax[0].set_title(f"Histogram of {col}")

        sns.boxplot(y=df[col], ax=ax[1], color="lightgreen")
        ax[1].set_title(f"Boxplot of {col}")

        st.pyplot(fig)

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

    # ---------------- MACHINE LEARNING ----------------
    st.subheader("🤖 Machine Learning")

    target = st.selectbox("Select Target Column", df.columns)

    if target:
        df_ml = pd.get_dummies(df, drop_first=True)

        target_cols = [col for col in df_ml.columns if target in col]

        if len(target_cols) > 0:
            y = df_ml[target_cols[0]]
            X = df_ml.drop(columns=target_cols)

            if len(X.columns) > 0:
                test_size = st.slider("Test Size (%)", 10, 50, 20)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

                if y.dtype == 'object' or y.nunique() < 10:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                st.write(f"🎯 Model Score: {score:.2f}")

                # Feature Importance
                st.subheader("📊 Feature Importance (Top 10)")
                importance = pd.Series(model.feature_importances_, index=X.columns)
                importance = importance.sort_values(ascending=False)
                st.bar_chart(importance.head(10))

    # ---------------- AI SUGGESTIONS ----------------
    st.subheader("💡 AI Suggestions")

    suggestions = []

    if (missing_percent > 20).any():
        suggestions.append("High missing values → consider removing/imputing")

    if duplicates > 0:
        suggestions.append("Remove duplicate rows")

    for col in df.columns:
        if df[col].nunique() > 100:
            suggestions.append(f"{col}: High cardinality → use encoding")

    for col in numeric_cols:
        if abs(df[col].skew()) > 1:
            suggestions.append(f"{col}: Highly skewed → apply log transform")

    if suggestions:
        for s in suggestions:
            st.write(f"👉 {s}")
    else:
        st.success("✅ Dataset looks clean and balanced!")

    # ---------------- REPORT DOWNLOAD ----------------
    st.subheader("📥 Download Report")

    report = f"""
DATASET REPORT
==============
Shape: {df.shape}

Columns:
{df.dtypes}

Missing Values:
{missing_df}

Duplicates: {duplicates}
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="data_report.txt",
        mime="text/plain"
    )

    # ---------------- FINAL SUMMARY ----------------
    st.subheader("📢 Final Summary")

    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(f"Numeric Columns: {len(numeric_cols)}")
    st.write(f"Categorical Columns: {df.select_dtypes(include=['object']).shape[1]}")

    high_missing = missing_df[missing_df["Missing %"] > 10]

    if not high_missing.empty:
        st.warning("Columns with high missing values:")
        st.write(high_missing)
    else:
        st.success("No major missing value issues")

    st.info("💡 Tip: Use highly correlated features for better ML performance")