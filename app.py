"""Streamlit Demo App with Charts and Data Analysis Examples"""

from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Streamlit Demo App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Streamlit Demo: Charts & Data Analysis")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
demo_mode = st.sidebar.radio(
    "Choose Mode:",
    ["Demo with Sample Data", "Upload Your Own CSV"]
)

if demo_mode == "Demo with Sample Data":
    st.header("📈 Interactive Data Visualization Demo")

    # Create sample datasets
    @st.cache_data
    def generate_sample_data():
        """Generate various sample datasets for demonstration"""

        # 1. Sales data (time series)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        sales_data = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.randint(1000, 5000, len(dates)) +
                     np.sin(np.arange(len(dates)) / 30) * 500,
            'Customers': np.random.randint(50, 200, len(dates)),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
        })

        # 2. Product performance data
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        product_data = pd.DataFrame({
            'Product': products,
            'Revenue': [45000, 38000, 52000, 41000, 35000],
            'Units Sold': [1200, 950, 1450, 1100, 890],
            'Customer Rating': [4.5, 4.2, 4.8, 4.3, 4.1],
            'Market Share': [22, 19, 26, 20, 13]
        })

        # 3. Customer demographics
        np.random.seed(42)
        customer_data = pd.DataFrame({
            'Age': np.random.randint(18, 70, 500),
            'Income': np.random.randint(25000, 150000, 500),
            'Purchases': np.random.randint(1, 50, 500),
            'Satisfaction': np.random.uniform(1, 5, 500)
        })

        return sales_data, product_data, customer_data

    sales_df, products_df, customers_df = generate_sample_data()

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Time Series", "📈 Product Analysis", "👥 Customer Insights", "📋 Data Tables"]
    )

    with tab1:
        st.subheader("Sales Performance Over Time")

        col1, col2, col3 = st.columns(3)
        with col1:
            total_sales = sales_df['Sales'].sum()
            st.metric("Total Sales", f"${total_sales:,.0f}")
        with col2:
            avg_daily_sales = sales_df['Sales'].mean()
            st.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
        with col3:
            total_customers = sales_df['Customers'].sum()
            st.metric("Total Customers", f"{total_customers:,}")

        st.markdown("### Daily Sales Trend")
        st.line_chart(sales_df.set_index('Date')['Sales'])

        st.markdown("### Sales by Region")
        region_sales = sales_df.groupby('Region')['Sales'].sum().reset_index()
        st.bar_chart(region_sales.set_index('Region'))

        # Interactive filter
        st.markdown("### Filter by Date Range")
        date_range = st.date_input(
            "Select date range:",
            value=(sales_df['Date'].min(), sales_df['Date'].max()),
            min_value=sales_df['Date'].min(),
            max_value=sales_df['Date'].max()
        )

        if len(date_range) == 2:
            filtered_sales = sales_df[
                (sales_df['Date'] >= pd.Timestamp(date_range[0])) &
                (sales_df['Date'] <= pd.Timestamp(date_range[1]))
            ]
            st.write(f"Filtered data: {len(filtered_sales)} days")
            st.area_chart(filtered_sales.set_index('Date')['Sales'])

    with tab2:
        st.subheader("Product Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Revenue by Product")
            st.bar_chart(products_df.set_index('Product')['Revenue'])

            st.markdown("### Market Share Distribution")
            st.write(products_df[['Product', 'Market Share']])

        with col2:
            st.markdown("### Product Comparison")
            st.dataframe(
                products_df.style.highlight_max(axis=0, subset=['Revenue', 'Units Sold', 'Customer Rating']),
                use_container_width=True
            )

            st.markdown("### Units Sold vs Revenue")
            chart_data = products_df[['Product', 'Units Sold', 'Revenue']].set_index('Product')
            st.line_chart(chart_data)

        # Best performing product
        best_product = products_df.loc[products_df['Revenue'].idxmax()]
        st.success(f"🏆 Top Product: **{best_product['Product']}** with ${best_product['Revenue']:,} in revenue")

    with tab3:
        st.subheader("Customer Demographics & Behavior")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Customer Age Distribution")
            age_hist = np.histogram(customers_df['Age'], bins=20)
            st.bar_chart(pd.DataFrame(age_hist[0], columns=['Count']))

            st.markdown("### Purchase Distribution")
            purchase_hist = np.histogram(customers_df['Purchases'], bins=20)
            st.bar_chart(pd.DataFrame(purchase_hist[0], columns=['Count']))

        with col2:
            st.markdown("### Income vs Purchases")
            scatter_data = customers_df[['Income', 'Purchases']].sample(100)
            st.scatter_chart(scatter_data.set_index('Income'))

            st.markdown("### Customer Statistics")
            st.dataframe({
                'Metric': ['Avg Age', 'Avg Income', 'Avg Purchases', 'Avg Satisfaction'],
                'Value': [
                    f"{customers_df['Age'].mean():.1f} years",
                    f"${customers_df['Income'].mean():,.0f}",
                    f"{customers_df['Purchases'].mean():.1f}",
                    f"{customers_df['Satisfaction'].mean():.2f}/5.0"
                ]
            }, use_container_width=True, hide_index=True)

        # Correlation analysis
        st.markdown("### Correlation Matrix")
        corr_matrix = customers_df.corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None),
                    use_container_width=True)

    with tab4:
        st.subheader("Raw Data Tables")

        data_choice = st.selectbox(
            "Select dataset to view:",
            ["Sales Data", "Product Data", "Customer Data"]
        )

        if data_choice == "Sales Data":
            st.markdown("### Sales Data (First 100 rows)")
            st.dataframe(sales_df.head(100), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(sales_df))
            with col2:
                st.metric("Columns", len(sales_df.columns))
            with col3:
                st.metric("Memory Usage", f"{sales_df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        elif data_choice == "Product Data":
            st.markdown("### Product Performance Data")
            st.dataframe(products_df, use_container_width=True)

        else:
            st.markdown("### Customer Demographics (Sample of 50)")
            st.dataframe(customers_df.sample(50), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Summary Statistics")
                st.dataframe(customers_df.describe(), use_container_width=True)
            with col2:
                st.markdown("#### Data Info")
                st.write(f"- Total Customers: {len(customers_df)}")
                st.write(f"- Features: {len(customers_df.columns)}")
                st.write(f"- Memory: {customers_df.memory_usage(deep=True).sum() / 1024:.1f} KB")

else:  # Upload Your Own CSV mode
    st.header("📁 Upload & Analyze Your CSV Files")

    MULTI_FILE = True

    uploaded_file = st.file_uploader(
        "Pick a CSV to analyze",
        type="csv",
        accept_multiple_files=MULTI_FILE
    )

    if uploaded_file:
        st.metric("Number of files", len(uploaded_file))

        for file in uploaded_file:
            bytes_data = file.getvalue()
            name = file.name
            st.header(f"📄 {name}")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Overview", "📈 Visualizations", "🔢 Raw Data", "💾 File Content"]
            )

            dataframe = pd.read_csv(file)

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Columns", len(dataframe.columns))
                with col2:
                    st.metric("Number of Rows", len(dataframe))
                with col3:
                    st.metric("Memory Usage", f"{dataframe.memory_usage(deep=True).sum() / 1024:.1f} KB")

                st.markdown("### Data Preview")
                st.dataframe(dataframe.head(10), use_container_width=True)

                st.markdown("### Summary Statistics")
                st.dataframe(dataframe.describe(), use_container_width=True)

            with tab2:
                st.markdown("### Visualizations")

                # Let user select columns for visualization
                numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    col1, col2 = st.columns(2)

                    with col1:
                        selected_col = st.selectbox("Select column for histogram:", numeric_cols)
                        if selected_col:
                            st.markdown(f"#### Distribution of {selected_col}")
                            st.bar_chart(dataframe[selected_col].value_counts().head(20))

                    with col2:
                        if len(numeric_cols) >= 2:
                            x_col = st.selectbox("Select X axis:", numeric_cols)
                            y_col = st.selectbox("Select Y axis:", [col for col in numeric_cols if col != x_col])

                            if x_col and y_col:
                                st.markdown(f"#### {x_col} vs {y_col}")
                                chart_data = dataframe[[x_col, y_col]].dropna()
                                st.scatter_chart(chart_data.set_index(x_col))
                else:
                    st.info("No numeric columns found for visualization")

            with tab3:
                st.markdown("### Complete Dataset")
                st.dataframe(dataframe, use_container_width=True)

                st.markdown("### Column Information")
                col_info = pd.DataFrame({
                    'Column': dataframe.columns,
                    'Type': dataframe.dtypes.values,
                    'Non-Null Count': dataframe.count().values,
                    'Null Count': dataframe.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)

            with tab4:
                st.markdown("### File Content (String)")
                stringio = StringIO(file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                st.text_area("Raw CSV Content", string_data, height=300)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit • Demo App with Sample Data
    </div>
    """,
    unsafe_allow_html=True
)
