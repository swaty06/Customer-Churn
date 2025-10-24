import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_input

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Title and Description
# ------------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("---")
st.write("Enter customer details below to predict churn probability.")

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'churn_model.pkl' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ------------------------------
# Sidebar - User Input
# ------------------------------
st.sidebar.header("🔧 Customer Information")

# Numerical inputs
st.sidebar.subheader("📊 Numerical Features")
col1, col2 = st.sidebar.columns(2)

with col1:
    account_age = st.number_input("Account Age (months)", min_value=0, max_value=120, value=12, step=1)
    viewing_hours = st.number_input("Viewing Hours/Week", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
    content_downloads = st.number_input("Content Downloads/Month", min_value=0, max_value=100, value=2, step=1)
    support_tickets = st.number_input("Support Tickets/Month", min_value=0, max_value=50, value=1, step=1)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=120.0, step=10.0)
    avg_duration = st.number_input("Avg Viewing Duration (min)", min_value=0.0, max_value=300.0, value=45.0, step=5.0)
    user_rating = st.number_input("User Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    watchlist_size = st.number_input("Watchlist Size", min_value=0, max_value=200, value=20, step=1)

st.sidebar.markdown("---")

# Categorical inputs
st.sidebar.subheader("📝 Categorical Features")
subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
payment_method = st.sidebar.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
paperless_billing = st.sidebar.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
content_type = st.sidebar.selectbox("Content Type", ["Movies", "TV Shows", "Both"])
multi_device = st.sidebar.radio("Multi-Device Access", ["Yes", "No"], horizontal=True)
device_registered = st.sidebar.selectbox("Device Registered", ["1", "2", "3", "4+"])
genre_preference = st.sidebar.selectbox("Genre Preference", ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance"])
gender = st.sidebar.selectbox("Gender", ["M", "F", "Other"])
parental_control = st.sidebar.radio("Parental Control", ["Yes", "No"], horizontal=True)
subtitles_enabled = st.sidebar.radio("Subtitles Enabled", ["Yes", "No"], horizontal=True)

# ------------------------------
# Main Area - Display Input Summary
# ------------------------------
st.subheader("📋 Customer Profile Summary")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Account Age", f"{account_age} months")
    st.metric("Viewing Hours/Week", f"{viewing_hours} hrs")

with col_b:
    st.metric("Total Charges", f"${total_charges:.2f}")
    st.metric("User Rating", f"{user_rating}/5.0")

with col_c:
    st.metric("Subscription", subscription_type)
    st.metric("Payment", payment_method)

with col_d:
    st.metric("Content Type", content_type)
    st.metric("Multi-Device", multi_device)

st.markdown("---")

# ------------------------------
# Create input DataFrame
# ------------------------------
input_data = pd.DataFrame({
    "AccountAge": [account_age],
    "TotalCharges": [total_charges],
    "ViewingHoursPerWeek": [viewing_hours],
    "AverageViewingDuration": [avg_duration],
    "ContentDownloadsPerMonth": [content_downloads],
    "UserRating": [user_rating],
    "SupportTicketsPerMonth": [support_tickets],
    "WatchlistSize": [watchlist_size],
    "SubscriptionType": [subscription_type],
    "PaymentMethod": [payment_method],
    "PaperlessBilling": [paperless_billing],
    "ContentType": [content_type],
    "MultiDeviceAccess": [multi_device],
    "DeviceRegistered": [device_registered],
    "GenrePreference": [genre_preference],
    "Gender": [gender],
    "ParentalControl": [parental_control],
    "SubtitlesEnabled": [subtitles_enabled]
})

# ------------------------------
# Predict Button
# ------------------------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ------------------------------
# Prediction Logic
# ------------------------------
if predict_button:
    try:
        # Preprocess input
        with st.spinner("🔄 Processing data..."):
            input_final = preprocess_input(input_data)
        
        # Make prediction
        with st.spinner("🤖 Making prediction..."):
            prediction = model.predict(input_final)[0]
            prob = model.predict_proba(input_final)[0]
            churn_prob = prob[1]
            no_churn_prob = prob[0]
        
        st.markdown("---")
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH CHURN RISK")
                st.write(f"**Churn Probability:** {churn_prob:.2%}")
                st.progress(churn_prob)
                
                # Risk indicator
                if churn_prob > 0.8:
                    st.write("🔴 **Critical Risk Level**")
                elif churn_prob > 0.6:
                    st.write("🟠 **High Risk Level**")
                else:
                    st.write("🟡 **Moderate Risk Level**")
            else:
                st.success("### ✅ LOW CHURN RISK")
                st.write(f"**Retention Probability:** {no_churn_prob:.2%}")
                st.progress(no_churn_prob)
                
                if no_churn_prob > 0.9:
                    st.write("🟢 **Excellent Retention**")
                elif no_churn_prob > 0.7:
                    st.write("🟢 **Good Retention**")
                else:
                    st.write("🟡 **Moderate Retention**")
        
        with result_col2:
            st.write("### 📊 Probability Breakdown")
            
            # Create a nice probability display
            prob_data = {
                "Outcome": ["🔴 Will Churn", "🟢 Will Stay"],
                "Probability": [f"{churn_prob:.2%}", f"{no_churn_prob:.2%}"],
                "Confidence": [f"{churn_prob:.4f}", f"{no_churn_prob:.4f}"]
            }
            prob_df = pd.DataFrame(prob_data)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Visual bar chart
            st.write("**Visual Comparison:**")
            chart_data = pd.DataFrame({
                'Probability': [churn_prob, no_churn_prob]
            }, index=['Churn', 'Stay'])
            st.bar_chart(chart_data)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Actionable Recommendations")
        
        if prediction == 1:
            # High churn risk recommendations
            st.warning("**⚠️ Immediate Actions Required:**")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("""
                **Retention Strategies:**
                - 🎁 Offer exclusive loyalty discount (10-20%)
                - 📞 Schedule priority customer success call
                - 🎯 Provide personalized content bundle
                - ⭐ Address low satisfaction scores
                """)
            
            with rec_col2:
                st.markdown("""
                **Investigation Points:**
                - 💳 Review billing/payment concerns
                - 📺 Analyze viewing pattern changes
                - 🎬 Check content availability issues
                - 🛠️ Review support ticket history
                """)
            
            # Specific recommendations based on input
            st.info("**📌 Personalized Insights:**")
            insights = []
            
            if support_tickets > 3:
                insights.append(f"⚠️ High support tickets ({support_tickets}/month) - Address service quality issues")
            if user_rating < 3:
                insights.append(f"⚠️ Low user rating ({user_rating}/5) - Urgent satisfaction improvement needed")
            if viewing_hours < 5:
                insights.append(f"⚠️ Low engagement ({viewing_hours} hrs/week) - Boost content recommendations")
            if subscription_type == "Basic":
                insights.append("💡 Consider offering Premium trial to increase engagement")
            
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.write("🔍 Conduct detailed customer interview to identify pain points")
                
        else:
            # Low churn risk recommendations
            st.success("**✅ Customer Retention Tips:**")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("""
                **Engagement Strategies:**
                - 🌟 Continue excellent service delivery
                - 📧 Send personalized recommendations
                - 🎬 Early access to new content
                - 🎉 Loyalty rewards program
                """)
            
            with rec_col2:
                st.markdown("""
                **Growth Opportunities:**
                - 📈 Consider upselling to premium tier
                - 👥 Encourage referral program participation
                - 🎯 Gather feedback for improvements
                - ⭐ Request testimonials/reviews
                """)
            
            # Upsell opportunities
            if subscription_type == "Basic" and user_rating >= 4:
                st.info("💡 **Upsell Opportunity:** High satisfaction + Basic plan → Offer Standard/Premium upgrade")
            if viewing_hours > 20:
                st.info("💡 **Engagement Insight:** High usage detected → Perfect candidate for premium features")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        
        with st.expander("🐛 Debug Information"):
            st.write("**Error Type:**", type(e).__name__)
            st.write("**Error Message:**", str(e))
            st.write("**Input Data Shape:**", input_data.shape)
            st.write("**Input Data Columns:**", list(input_data.columns))
            
            # Show sample of input data
            st.write("**Sample Input Data:**")
            st.dataframe(input_data)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>Customer Churn Prediction System</strong></p>
        <p>Built with ❤️ using Streamlit | Powered by Machine Learning</p>
        <p style='font-size: 12px;'>Version 1.0 | Last Updated: 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar footer with info
st.sidebar.markdown("---")
st.sidebar.info("""
**ℹ️ How to Use:**
1. Enter customer details
2. Click 'Predict Churn'
3. Review risk assessment
4. Follow recommendations
""")

st.sidebar.success("""
**✅ Files Required:**
- churn_model.pkl ✓
- scaler.pkl ✓
- training_columns.pkl ✓
""")