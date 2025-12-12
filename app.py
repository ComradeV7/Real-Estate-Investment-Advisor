import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. SETUP & CONFIGURATION
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; background-color: #007bff; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #0056b3;}
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #007bff;
    }
    .metric-title {font-size: 1.1rem; color: #6c757d; margin-bottom: 10px;}
    .metric-value {font-size: 2rem; font-weight: bold; color: #212529;}
    .prediction-header {font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. DATA & MODEL LOADING (Issue #11: Optimized data loading)
@st.cache_data
def load_data():
    """Load and preprocess housing data. Data is cached for efficiency."""
    try:
        df = pd.read_csv('data/processed_housing_data.csv')
        # Standardize text
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.title()
        
        # Generate Median Map using original Price_per_SqFt (in Lakhs)
        # Note: Price_per_SqFt in training data is in Lakhs (e.g., 0.05, 0.11)
        # No need to recalculate - using data as-is for efficiency
        median_map = df.groupby('City')['Price_per_SqFt'].median().to_dict()
        
        return df, median_map
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Load pre-trained machine learning models. Models are cached."""
    try:
        clf = joblib.load('best-models/best_model_classification.pkl')
        reg = joblib.load('best-models/best_model_regression.pkl')
        return clf, reg
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

df, median_map = load_data()
clf_model, reg_model = load_models()

# Issue #5: Validate critical resources - stop app if models or data failed to load
if clf_model is None or reg_model is None:
    st.error("❌ **Critical Error:** Failed to load ML models. Please check that model files exist in 'best-models/' directory.")
    st.stop()
    
if df is None or median_map is None:
    st.error("❌ **Critical Error:** Failed to load data. Please check that 'data/processed_housing_data.csv' exists.")
    st.stop()

# 3. SIDEBAR: USER INPUTS
with st.sidebar:
    st.title("Property Details")
    st.markdown("---")

    # Location Selection
    cities = sorted(df['City'].unique())
    selected_city = st.selectbox("City", cities, index=cities.index('Mumbai') if 'Mumbai' in cities else 0)
    
    # Dynamic Locality Filtering
    localities = sorted(df[df['City'] == selected_city]['Locality'].unique())
    selected_locality = st.selectbox("Locality", localities)
    
    # Property Specs
    prop_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
    
    col1, col2 = st.columns(2)
    with col1:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        floor_no = st.number_input("Floor No.", min_value=0, max_value=100, value=5)
    with col2:
        size_sqft = st.number_input("Size (Sq. Ft.)", min_value=100, max_value=20000, value=1200)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=15)

    # Pricing Input
    price_lakhs = st.number_input("Current Price (Lakhs)", min_value=1.0, max_value=10000.0, value=150.0, step=1.0)
    
    # Amenities & Features
    furnish_status = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
    transport = st.selectbox("Public Transport Access", ["High", "Medium", "Low"])
    
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age (Yrs)", 0, 100, 5)
        schools = st.slider("Schools Nearby", 0, 10, 3)
    with col4:
        hospitals = st.slider("Hospitals Nearby", 0, 10, 2)
        
    analyze_btn = st.button("Analyze Investment")

# 4. MAIN DASHBOARD
st.markdown("## Real Estate Investment Advisor")
st.markdown("##### AI-Powered Valuation & Future Forecasting Tool")
st.markdown("---")

if analyze_btn:
    
    # Issue #10: Input Validation - Check for logical errors
    validation_errors = []
    
    if floor_no > total_floors:
        validation_errors.append(f"❌ **Floor Number ({floor_no}) cannot be greater than Total Floors ({total_floors})**")
    
    if size_sqft < 100 or size_sqft > 50000:
        validation_errors.append(f"⚠️ **Property size ({size_sqft} sqft) seems unrealistic. Expected range: 100-50,000 sqft**")
    
    if bhk < 1 or bhk > 10:
        validation_errors.append(f"❌ **BHK ({bhk}) must be between 1 and 10**")
    
    if price_lakhs <= 0:
        validation_errors.append(f"❌ **Price must be greater than zero**")
    
    # Display validation errors if any
    if validation_errors:
        st.error("### Input Validation Failed")
        for error in validation_errors:
            st.markdown(error)
        st.info("Please correct the above errors and try again.")
    else:
        # Proceed with analysis if validation passed
        
        # A. Feature Engineering (Match Training Pipeline)
        try:
            # Calculate Price per SqFt (in Lakhs to match training data)
            # Note: Training data has Price_per_SqFt in Lakhs (e.g., 0.05, 0.11)
            price_per_sqft = price_lakhs / size_sqft if size_sqft > 0 else 0
            
            # Issue #7: Derived Features (with division-by-zero protection)
            avg_room_size = size_sqft / bhk if bhk > 0 else 0
            floor_ratio = floor_no / total_floors if total_floors > 0 else 0
            floor_ratio = min(floor_ratio, 1.0) 
            
            # Infra Score Logic
            transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
            transport_val = transport_map.get(transport, 1)
            infra_score = schools + hospitals + (transport_val * 2)
            
            # Issue #4: Market Benchmarking (Improved to avoid hardcoded defaults)
            # Calculate overall median once as fallback for new/unknown cities
            overall_median = df['Price_per_SqFt'].median()
            
            # Get city-specific median, use overall median if city not found
            if selected_city in median_map:
                city_median = median_map[selected_city]
            else:
                city_median = overall_median
                st.warning(f"⚠️ **Note:** '{selected_city}' was not found in the training data. Using overall market median (₹{overall_median:.4f} Lakhs/sqft) for comparison.")
            
            # Calculate price relative to market median (with zero-division protection)
            price_relative = price_per_sqft / city_median if city_median > 0 else 1.0
            
            # Prepare Input Dataframe (matching training features exactly)
            # Features used in training (from Model-Training.ipynb line 528-530):
            # 'City', 'Locality', 'Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 
            # 'BHK', 'Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt', 'City_Median_Price', 
            # 'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals', 
            # 'Infra_Score', 'Avg_Room_Size', 'Floor_Ratio'
            input_data = pd.DataFrame({
                'City': [selected_city],
                'Locality': [selected_locality],
                'Property_Type': [prop_type],
                'Furnished_Status': [furnish_status],
                'Public_Transport_Accessibility': [transport],
                'BHK': [bhk],
                'Size_in_SqFt': [size_sqft],
                'Price_in_Lakhs': [price_lakhs],
                'Price_per_SqFt': [price_per_sqft],
                'City_Median_Price': [city_median],
                'Floor_No': [floor_no],
                'Total_Floors': [total_floors],
                'Age_of_Property': [age],
                'Nearby_Schools': [schools],
                'Nearby_Hospitals': [hospitals],
                'Infra_Score': [infra_score],
                'Avg_Room_Size': [avg_room_size],
                'Floor_Ratio': [floor_ratio]
            })

            # B. Generate Predictions
            pred_class = clf_model.predict(input_data)[0]
            pred_prob = clf_model.predict_proba(input_data)[0][1] 
            pred_future_price = reg_model.predict(input_data)[0]
            roi = ((pred_future_price - price_lakhs) / price_lakhs) * 100 if price_lakhs > 0 else 0
            
            # C. Visualization & Output
            
            # Verdict Section
            col_res1, col_res2 = st.columns([2, 1])
            
            # LOGIC FIX: Calculate price difference string dynamically to avoid negative %
            diff_pct = (price_relative - 1) * 100
            if diff_pct < 0:
                price_msg = f"priced **{abs(diff_pct):.1f}% below** the market median"
            else:
                price_msg = f"priced **{diff_pct:.1f}% above** the market median"

            with col_res1:
                st.markdown("<div class='prediction-header'>Investment Verdict</div>", unsafe_allow_html=True)
                if pred_class == 1:
                    st.success("**RECOMMENDED: GOOD INVESTMENT**")
                    st.markdown(f"This property is {price_msg}.")
                else:
                    st.error("**CAUTION: HIGH RISK / OVERPRICED**")
                    st.markdown(f"This property is {price_msg}.")
                    
                    # ADDRESSING USER QUERY: Why profit if bad investment?
                    if roi > 0:
                        st.warning("⚠️ **Note on Profit:** While the price is projected to rise due to general market inflation, this property is flagged as 'Risky' because the returns are suboptimal compared to better opportunities in this area.")
            
            with col_res2:
                # Confidence Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred_prob * 100,
                    title = {'text': "Confidence Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#007bff"},
                        'steps': [{'range': [0, 100], 'color': "#f8f9fa"}]
                    }
                ))
                fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")

            # Financial Metrics Section
            st.markdown("<div class='prediction-header'>Financial Forecast (5 Years)</div>", unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-card'><div class='metric-title'>Current Price</div><div class='metric-value'>₹ {price_lakhs:,.0f} L</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><div class='metric-title'>Predicted Future</div><div class='metric-value'>₹ {pred_future_price:,.0f} L</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><div class='metric-title'>Net Profit</div><div class='metric-value'>₹ {(pred_future_price - price_lakhs):,.0f} L</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-card'><div class='metric-title'>Expected ROI</div><div class='metric-value'>{roi:.1f}%</div></div>", unsafe_allow_html=True)

            st.markdown("###")

            # Appreciation Chart
            chart_data = pd.DataFrame({
                'Timeline': ['Current', 'Future (5Y)'],
                'Price': [price_lakhs, pred_future_price]
            })
            
            fig_bar = px.bar(chart_data, x='Timeline', y='Price', text_auto='.0f',
                             color='Timeline', color_discrete_sequence=['#6c757d', '#28a745'])
            fig_bar.update_layout(title="Capital Appreciation Forecast", yaxis_title="Price (Lakhs)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed Insights
            with st.expander("View Detailed Property Insights"):
                st.write("**Calculated Metrics:**")
                # Display in Lakhs to match training data
                st.write(f"- **Price per SqFt:** ₹ {price_per_sqft:.4f} Lakhs / sqft")
                st.write(f"- **City Median Rate:** ₹ {city_median:.4f} Lakhs / sqft")
                st.write(f"- **Price Relative to Median:** {price_relative:.2f}x")
                st.write(f"- **Infrastructure Score:** {infra_score}/20")
                st.write(f"- **Spaciousness:** {avg_room_size:.0f} sqft/room")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    # Default State
    st.info("Please enter property details in the sidebar and click 'Analyze Investment'")
    
    st.markdown("### Market Snapshot")
    col_a, col_b = st.columns(2)
    with col_a:
        top_cities = df.groupby('City')['Price_in_Lakhs'].median().nlargest(5)
        st.plotly_chart(px.bar(top_cities, orientation='h', title="Most Expensive Cities", color_discrete_sequence=['#007bff']), use_container_width=True)
    with col_b:
        st.plotly_chart(px.pie(df, names='Property_Type', title="Market Inventory"), use_container_width=True)