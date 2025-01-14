import streamlit as st
import pandas as pd
import numpy as np
import warnings
import dtale
import dtale.app as dtale_app
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Comprehensive EDA Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def detect_data_type(series):
    """
    Detailed data type detection for each column
    """
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return "Discrete Numeric"
        elif series.dtype in ['int32', 'int64']:
            return "Integer"
        else:
            return "Continuous Numeric"
    elif pd.api.types.is_string_dtype(series):
        if series.nunique() <= 10:
            return "Categorical"
        else:
            return "Text"
    elif pd.api.types.is_datetime64_dtype(series):
        return "DateTime"
    elif pd.api.types.is_bool_dtype(series):
        return "Boolean"
    return "Other"

def calculate_descriptive_stats(series, data_type):
    """
    Calculate comprehensive descriptive statistics based on data type
    """
    stats_dict = {
        "count": len(series),
        "unique_values": series.nunique(),
        "missing_values": series.isnull().sum(),
        "missing_percentage": (series.isnull().sum() / len(series)) * 100,
    }

    if data_type in ["Integer", "Continuous Numeric", "Discrete Numeric"]:
        numeric_stats = {
            "mean": series.mean(),
            "median": series.median(),
            "mode": series.mode().iloc[0] if not series.mode().empty else None,
            "std": series.std(),
            "variance": series.var(),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "min": series.min(),
            "max": series.max(),
            "range": series.max() - series.min(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
            "iqr": series.quantile(0.75) - series.quantile(0.25),
            "coefficient_of_variation": (series.std() / series.mean()) * 100 if series.mean() != 0 else None
        }
        stats_dict.update(numeric_stats)

        # Add outlier detection
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        stats_dict["outliers_count"] = len(outliers)
        stats_dict["outliers_percentage"] = (len(outliers) / len(series)) * 100

    elif data_type in ["Categorical", "Text", "Boolean"]:
        categorical_stats = {
            "most_common_value": series.mode().iloc[0] if not series.mode().empty else None,
            "most_common_count": series.value_counts().iloc[0] if not series.value_counts().empty else None,
            "most_common_percentage": (series.value_counts().iloc[0] / len(series)) * 100 if not series.value_counts().empty else None,
            "least_common_value": series.value_counts().index[-1] if not series.value_counts().empty else None,
            "least_common_count": series.value_counts().iloc[-1] if not series.value_counts().empty else None,
            "least_common_percentage": (series.value_counts().iloc[-1] / len(series)) * 100 if not series.value_counts().empty else None,
        }
        stats_dict.update(categorical_stats)

        # Add frequency distribution
        value_counts = series.value_counts()
        stats_dict["value_distribution"] = value_counts.to_dict()
        stats_dict["entropy"] = stats.entropy(value_counts) if len(value_counts) > 1 else 0

    elif data_type == "DateTime":
        temporal_stats = {
            "earliest_date": series.min(),
            "latest_date": series.max(),
            "date_range": (series.max() - series.min()).days,
            "most_common_year": series.dt.year.mode().iloc[0] if not series.dt.year.mode().empty else None,
            "most_common_month": series.dt.month.mode().iloc[0] if not series.dt.month.mode().empty else None,
            "most_common_weekday": series.dt.dayofweek.mode().iloc[0] if not series.dt.dayofweek.mode().empty else None,
        }
        stats_dict.update(temporal_stats)

    return stats_dict

def analyze_relationships(df):
    """
    Analyze relationships between variables and generate heatmap
    """
    relationships = {}
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Correlation analysis
        correlations = df[numeric_cols].corr()

        # Generate Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        plt.title("Correlation Heatmap")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels from being cut off

        relationships['heatmap'] = fig

        # Find strong correlations (absolute value > 0.7)
        strong_correlations = []
        for i in range(len(correlations.columns)):
            for j in range(i):
                if abs(correlations.iloc[i, j]) > 0.7:
                    strong_correlations.append({
                        'variables': (correlations.columns[i], correlations.columns[j]),
                        'correlation': correlations.iloc[i, j]
                    })
        
        relationships['strong_correlations'] = strong_correlations
        
    return relationships

def analyze_data_quality(df):
    """
    Comprehensive data quality analysis
    """
    quality_report = {
        "completeness": {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "columns_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        },
        "uniqueness": {
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
        },
        "consistency": {
            "mixed_dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
    }
    
    # Check for potential data quality issues
    quality_report["potential_issues"] = []
    
    # Check for suspicious patterns
    for column in df.columns:
        # Check for columns with too many missing values
        missing_pct = (df[column].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            quality_report["potential_issues"].append(
                f"Column '{column}' has {missing_pct:.2f}% missing values"
            )
            
        # Check for columns with too many unique values
        unique_pct = (df[column].nunique() / len(df)) * 100
        if unique_pct > 95 and len(df) > 100:
            quality_report["potential_issues"].append(
                f"Column '{column}' has {unique_pct:.2f}% unique values"
            )
            
    return quality_report

def main():
    st.title("📊 Comprehensive EDA Tool")
    st.write("Upload your data for detailed exploratory data analysis")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Checkboxes for analysis sections
            show_overview = st.checkbox("Show Dataset Overview", value=True)
            show_quality = st.checkbox("Show Data Quality Analysis", value=True)
            show_variable = st.checkbox("Show Variable Analysis", value=True)
            show_relationships = st.checkbox("Show Variable Relationships", value=True)
            show_dtale = st.checkbox("Launch Interactive EDA with D-Tale", value=False)
            show_pandas_profiling = st.checkbox("Generate Pandas Profiling Report", value=False)

            if show_overview:
                # Dataset Overview
                st.header("📋 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                     st.metric("Total Missing Values", df.isnull().sum().sum())
    

            if show_quality:
                # Data Quality Analysis
                st.header("🔍 Data Quality Analysis")
                quality_report = analyze_data_quality(df)
                
                # Display data quality metrics
                st.subheader("Completeness")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Missing Values", quality_report["completeness"]["total_missing"])
                with col2:
                    st.metric("Missing Percentage", f"{quality_report['completeness']['missing_percentage']:.2f}%")

                st.subheader("Duplicates")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duplicate Rows", quality_report["uniqueness"]["duplicate_rows"])
                with col2:
                    st.metric("Duplicate Percentage", f"{quality_report['uniqueness']['duplicate_percentage']:.2f}%")
            
            if show_variable:
                # Variable Analysis
                st.header("🔢 Variable Analysis")
                
                # Allow user to select variable
                selected_column = st.selectbox("Select a variable to analyze:", df.columns)
                
                if selected_column:
                    # Detect data type
                    data_type = detect_data_type(df[selected_column])
                    
                    # Calculate statistics
                    stats = calculate_descriptive_stats(df[selected_column], data_type)
                    
                    # Display variable information
                    st.subheader(f"Analysis of {selected_column}")
                    
                    # Basic Information
                    st.write("**Basic Information**")
                    cols = st.columns(3)
                    with cols[0]:
                        st.write(f"Data Type: {data_type}")
                        st.write(f"Count: {stats['count']}")
                        st.write(f"Unique Values: {stats['unique_values']}")
                    with cols[1]:
                        st.write(f"Missing Values: {stats['missing_values']}")
                        st.write(f"Missing Percentage: {stats['missing_percentage']:.2f}%")
                    with cols[2]:
                        st.write(f"Memory Usage: N/A")

                    # Display type-specific statistics
                    if data_type in ["Integer", "Continuous Numeric", "Discrete Numeric"]:
                        st.write("**Descriptive Statistics**")
                        cols = st.columns(3)
                        with cols[0]:
                            st.write(f"Mean: {stats['mean']:.2f}")
                            st.write(f"Median: {stats['median']:.2f}")
                            st.write(f"Mode: {stats['mode']:.2f}")
                        with cols[1]:
                            st.write(f"Standard Deviation: {stats['std']:.2f}")
                            st.write(f"Variance: {stats['variance']:.2f}")
                            st.write(f"Coefficient of Variation: {stats['coefficient_of_variation']:.2f}%")
                        with cols[2]:
                            st.write(f"Skewness: {stats['skewness']:.2f}")
                            st.write(f"Kurtosis: {stats['kurtosis']:.2f}")

                        st.write("**Range Information**")
                        cols = st.columns(3)
                        with cols[0]:
                            st.write(f"Minimum: {stats['min']:.2f}")
                            st.write(f"Maximum: {stats['max']:.2f}")
                        with cols[1]:
                            st.write(f"Range: {stats['range']:.2f}")
                            st.write(f"IQR: {stats['iqr']:.2f}")
                        with cols[2]:
                            st.write(f"Outliers Count: {stats['outliers_count']}")
                            st.write(f"Outliers Percentage: {stats['outliers_percentage']:.2f}%")

                    elif data_type in ["Categorical", "Text", "Boolean"]:
                        st.write("**Category Statistics**")
                        cols = st.columns(2)
                        with cols[0]:
                            st.write(f"Most Common: {stats['most_common_value']}")
                            st.write(f"Most Common Count: {stats['most_common_count']}")
                            st.write(f"Most Common Percentage: {stats['most_common_percentage']:.2f}%")
                        with cols[1]:
                            st.write(f"Least Common: {stats['least_common_value']}")
                            st.write(f"Least Common Count: {stats['least_common_count']}")
                            st.write(f"Least Common Percentage: {stats['least_common_percentage']:.2f}%")

                    elif data_type == "DateTime":
                        st.write("**Temporal Statistics**")
                        cols = st.columns(2)
                        with cols[0]:
                            st.write(f"Earliest Date: {stats['earliest_date']}")
                            st.write(f"Latest Date: {stats['latest_date']}")
                            st.write(f"Date Range (days): {stats['date_range']}")
                        with cols[1]:
                            st.write(f"Most Common Year: {stats['most_common_year']}")
                            st.write(f"Most Common Month: {stats['most_common_month']}")
                            st.write(f"Most Common Weekday: {stats['most_common_weekday']}")

            if show_relationships:
                # Variable Relationships
                st.header("🔗 Variable Relationships")
                relationships = analyze_relationships(df)
                if 'heatmap' in relationships:
                    st.pyplot(relationships['heatmap']) # Display the heatmap
                
                if relationships.get('strong_correlations'):
                    st.subheader("Strong Correlations (|r| > 0.7)")
                    for corr in relationships['strong_correlations']:
                        st.write(f"{corr['variables'][0]} vs {corr['variables'][1]}: {corr['correlation']:.3f}")
                else:
                    st.write("No strong correlations found between numeric variables.")

            # Data Quality Issues
            if quality_report["potential_issues"] and show_quality:
                st.header("⚠️ Potential Data Quality Issues")
                for issue in quality_report["potential_issues"]:
                    st.warning(issue)
            
            if show_dtale:
                # Run D-Tale
                d = dtale.app.get_instance(data=df)
                d.open_browser()
                dtale_url = d._url
                st.write(f"D-Tale is running at: {dtale_url}")

                #Embed D-Tale in the app
                st.markdown(
                    f'<iframe src="{dtale_url}" width="100%" height="800px"></iframe>',
                    unsafe_allow_html=True,
                )
            
            if show_pandas_profiling:
                st.header("📊 Pandas Profiling Report")
                profile = ProfileReport(df, title="Pandas Profiling Report")
                st_profile_report(profile)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()