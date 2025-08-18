import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a style for the plots for better aesthetics
sns.set_style('whitegrid')

# --- Streamlit Dashboard Title ---
st.title("HR Employee Attrition Analysis Dashboard")
st.write("Explore key factors influencing employee attrition with interactive charts.")
st.markdown("---")


# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_clean_data(file_path):
    """
    Loads the dataset, handles missing/duplicate values,
    and preprocesses columns for analysis.
    This function is cached by Streamlit to prevent re-running on every interaction.
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}. Please ensure the CSV is inside the 'data/' folder.")
        return None

    df = pd.read_csv(file_path)

    # Sidebar Data Info
    st.sidebar.subheader("Data Info")
    st.sidebar.write(f"Shape of dataset: {df.shape}")

    # Missing values
    missing_values = df.isnull().sum().to_frame("Missing Values").sort_values("Missing Values", ascending=False)
    st.sidebar.write("Missing values per column:")
    st.sidebar.dataframe(missing_values)

    # Duplicates
    duplicates = df.duplicated().sum()
    st.sidebar.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.sidebar.write("Duplicate rows have been removed.")

    # Drop irrelevant columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Convert 'Attrition' to numeric
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    st.sidebar.markdown("---")

    return df


# --- Visualization Functions ---
def plot_categorical_attrition(df, column):
    attrition_rate = df.groupby(column)['Attrition'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='viridis', ax=ax)
    ax.set_title(f'Attrition Rate by {column}', fontsize=16)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Attrition Rate', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def plot_numerical_attrition(df, column):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Attrition', y=column, data=df, palette='Set2', ax=ax)
    ax.set_title(f'{column} vs Attrition', fontsize=16)
    ax.set_xlabel('Attrition (0: No, 1: Yes)', fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    if 'Income' in column:  # log scale for skewed data
        plt.yscale('log')
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(df, features):
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
                mask=np.triu(np.ones_like(corr, dtype=bool)), ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


# --- Main App ---
def main():
    data_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df_original = load_and_clean_data(data_path)

    if df_original is not None:
        categorical_features = ['Department', 'JobRole', 'MaritalStatus', 'Gender', 'EducationField']
        numerical_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                              'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
                              'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                              'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction',
                              'WorkLifeBalance']

        # Summary Stats
        st.header("Summary Statistics")
        st.write(df_original.describe().T)
        st.markdown("---")

        # Sidebar Filter
        st.sidebar.subheader("Dashboard Filters")
        selected_department = st.sidebar.selectbox(
            "Filter by Department:",
            options=['All'] + list(df_original['Department'].unique())
        )

        df = df_original.copy()
        if selected_department != 'All':
            df = df[df['Department'] == selected_department]
            st.write(f"Displaying data for the **{selected_department}** Department.")
            st.write(f"**Total Employees:** {df.shape[0]}")
            st.write(f"**Attrition Rate:** {df['Attrition'].mean():.2%}")

        st.markdown("---")

        # Layout
        st.header("Interactive Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Categorical Feature Analysis")
            selected_cat_feature = st.selectbox("Choose a categorical feature:", categorical_features)
            df['JobRole'] = df['JobRole'].replace({
                'Sales Representative': 'Sales Rep',
                'Research Scientist': 'Res Scientist',
                'Laboratory Technician': 'Lab Tech',
                'Manufacturing Director': 'Mfg Director',
                'Healthcare Representative': 'Health Rep',
                'Human Resources': 'HR',
                'Manager': 'Mgr',
                'Research Director': 'Res Director',
                'Sales Executive': 'Sales Exec'
            })
            plot_categorical_attrition(df, selected_cat_feature)

        with col2:
            st.subheader("Numerical Feature Analysis")
            selected_num_feature = st.selectbox("Choose a numerical feature:", numerical_features)
            plot_numerical_attrition(df, selected_num_feature)

        # Heatmap
        st.markdown("---")
        st.header("Correlation Analysis")
        st.write("Below is a heatmap showing the correlation between various numerical features.")
        all_numerical_features = numerical_features + ['Attrition']
        plot_correlation_heatmap(df_original, all_numerical_features)


if __name__ == "__main__":
    main()
