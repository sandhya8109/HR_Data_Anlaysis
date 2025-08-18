import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please make sure the CSV is in the same directory.")
        return None

    # Initial data info
    st.sidebar.subheader("Data Info")
    st.sidebar.write(f"Shape of dataset: {df.shape}")

    # Check for missing values
    missing_values = df.isnull().sum().to_frame("Missing Values").sort_values("Missing Values", ascending=False)
    st.sidebar.write("Missing values per column:")
    st.sidebar.dataframe(missing_values)

    # Check for duplicate values
    duplicates = df.duplicated().sum()
    st.sidebar.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.sidebar.write("Duplicate rows have been removed.")

    # Drop columns that have no predictive value
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df.drop(columns=drop_cols, inplace=True)

    # Convert 'Attrition' to a numerical value
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    st.sidebar.markdown("---")

    return df


# --- Exploratory Data Analysis (EDA) and Visualization Functions ---

def plot_categorical_attrition(df, column):
    """
    Calculates and plots the attrition rate for a given categorical column.
    """
    # Group by the column and calculate the mean attrition rate
    attrition_rate = df.groupby(column)['Attrition'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='viridis', ax=ax)
    ax.set_title(f'Attrition Rate by {column}', fontsize=16)
    ax.set_xlabel(column.replace('JobRole', 'Job Role').replace('Department', 'Department'), fontsize=12)
    ax.set_ylabel('Attrition Rate', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def plot_numerical_attrition(df, column):
    """
    Plots a boxplot to show the distribution of a numerical column by attrition.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Attrition', y=column, data=df, palette='Set2', ax=ax)
    ax.set_title(f'{column} vs Attrition', fontsize=16)
    ax.set_xlabel('Attrition (0: No, 1: Yes)', fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    # Use log scale for highly skewed data like income
    if 'Income' in column:
        plt.yscale('log')
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(df, features):
    """
    Generates a heatmap to visualize the correlation between numerical features.
    """
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
                mask=np.triu(np.ones_like(corr, dtype=bool)), ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


# --- Main Application Logic ---
def main():
    df_original = load_and_clean_data("WA_Fn-UseC_-HR-Employee-Attrition.csv")

    if df_original is not None:
        # Separate features for interactive plotting
        categorical_features = ['Department', 'JobRole', 'MaritalStatus', 'Gender', 'EducationField']
        numerical_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                              'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
                              'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                              'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction',
                              'WorkLifeBalance']

        # --- Display Summary Statistics ---
        st.header("Summary Statistics")
        st.write(df_original.describe().T)
        st.markdown("---")

        # --- Interactive Sidebar for Filtering ---
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

        # --- Dashboard Layout and Plotting ---
        st.header("Interactive Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Categorical Feature Analysis")
            selected_cat_feature = st.selectbox("Choose a categorical feature:", categorical_features)
            # Renaming job roles for better plot readability
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

        # Plotting the correlation heatmap
        st.markdown("---")
        st.header("Correlation Analysis")
        st.write("Below is a heatmap showing the correlation between various numerical features.")
        all_numerical_features = numerical_features + ['Attrition']
        plot_correlation_heatmap(df_original, all_numerical_features)

if __name__ == "__main__":
    main()
