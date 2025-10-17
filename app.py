import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
import uuid # Used for generating unique booking IDs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 0. Configuration and Constants ---

# YOUR REAL API KEY AND LOCATION (Replace this with your actual details)
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY_HERE" # <<<<< REPLACE THIS
DEFAULT_CITY = "Mumbai"
WEATHER_URL = f"http://api.openweathermap.org/data/2.5/weather?q={DEFAULT_CITY}&appid={WEATHER_API_KEY}&units=metric"

# Persistent Data Files
USERS_FILE = 'users.json'
PREDICTION_DATA_FILE = 'prediction_data.csv'
EQUIPMENT_FILE = 'equipment_inventory.json' # NEW FILE
BOOKING_FILE = 'booking_records.json'     # NEW FILE

COMMODITY_OPTIONS = ["Wheat", "Rice", "Cotton", "Maize"]
PREDICTION_TARGETS = {
    "Wheat": "Price (USD/ton)",
    "Rice": "Price (USD/ton)",
    "Cotton": "Production (bales)",
    "Maize": "Yield (ton/hectare)"
}

# Mock data for Crop Recommendation (based on simple logic)
CROP_RECOMMENDATIONS = {
    "High Rainfall": "Rice",
    "Moderate Temperature": "Wheat",
    "High Temperature": "Cotton",
    "High Humidity": "Maize"
}

# --- 1. Utility Functions for Persistence ---

def load_json(file_path):
    """Load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {} if file_path != EQUIPMENT_FILE else []
    except json.JSONDecodeError:
        return {} if file_path != EQUIPMENT_FILE else []

def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_users(): return load_json(USERS_FILE)
def save_users(users): save_json(USERS_FILE, users)

def load_equipment(): return load_json(EQUIPMENT_FILE)
def save_equipment(equipment): save_json(EQUIPMENT_FILE, equipment)

def load_bookings(): return load_json(BOOKING_FILE)
def save_bookings(bookings): save_json(BOOKING_FILE, bookings)

# (load_prediction_data, train_mock_model, get_real_time_weather, and get_live_data functions remain unchanged)

def load_prediction_data():
    """Load mock data for prediction models."""
    try:
        data = pd.read_csv(PREDICTION_DATA_FILE)
    except FileNotFoundError:
        st.error(f"Prediction data file '{PREDICTION_DATA_FILE}' not found. Generating mock data.")
        n_samples = 100
        data = pd.DataFrame({
            'Temperature': np.random.uniform(15, 40, n_samples),
            'Humidity': np.random.uniform(50, 95, n_samples),
            'Rainfall': np.random.uniform(0, 50, n_samples),
            'Soil_Type_A': np.random.randint(0, 2, n_samples),
            'Soil_Type_B': np.random.randint(0, 2, n_samples),
            'Fertilizer_Usage': np.random.uniform(10, 50, n_samples),
            'Price (USD/ton)': (np.random.rand(n_samples) * 100 + 300) * (1 + np.random.uniform(-0.1, 0.1, n_samples)),
            'Production (bales)': (np.random.rand(n_samples) * 500 + 1000) * (1 + np.random.uniform(-0.1, 0.1, n_samples)),
            'Yield (ton/hectare)': (np.random.rand(n_samples) * 3 + 5) * (1 + np.random.uniform(-0.05, 0.05, n_samples))
        })
        data.to_csv(PREDICTION_DATA_FILE, index=False)
    return data

def train_mock_model(data, target_col):
    """Trains a simple Linear Regression model on mock data."""
    features = ['Temperature', 'Humidity', 'Rainfall', 'Fertilizer_Usage']
    for col in features:
        if col not in data.columns: return None
    
    X = data[features]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.session_state['model_mse'] = mse
    return model

def get_real_time_weather():
    """Fetches real-time weather data from OpenWeatherMap API."""
    try:
        response = requests.get(WEATHER_URL, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            temp = np.round(data['main']['temp'], 1)
            humidity = data['main']['humidity']
            rainfall = np.round(np.random.uniform(0, 5), 1)
            market_status = np.random.choice(["Stable", "Volatile", "Upward Trend"])
            return temp, humidity, rainfall, market_status
        else:
            temp = np.round(np.random.uniform(28, 35), 1)
            humidity = np.round(np.random.uniform(60, 95), 1)
            return temp, humidity, 0, "API Failed (Mock)"

    except requests.exceptions.RequestException:
        temp = np.round(np.random.uniform(28, 35), 1)
        humidity = np.round(np.random.uniform(60, 95), 1)
        return temp, humidity, 0, "Connection Failed (Mock)"

def get_live_data(): return get_real_time_weather()


# --- 2. Streamlit Page Functions ---

def login_page():
    """Handles user login and registration."""
    users = load_users()
    st.title("AgriSmart: Login")
    # ... (Login and Registration forms as before)
    st.subheader("Existing User Login")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in users and users[username]['password'] == password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

    st.subheader("New User Registration")
    with st.form("register_form"):
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        register_submitted = st.form_submit_button("Register")

        if register_submitted:
            if new_username in users:
                st.error("Username already exists.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_username) < 3 or len(new_password) < 6:
                st.error("Username must be at least 3 characters and password 6 characters.")
            else:
                users[new_username] = {'password': new_password}
                save_users(users)
                st.success("Registration successful! You can now log in.")

def dashboard_page():
    """Main page dispatcher."""
    st.title(f"Welcome, {st.session_state['username']}! 🌾")
    
    st.sidebar.header("Navigation")
    feature = st.sidebar.radio("Go to", [
        "Dashboard", 
        "Prediction", 
        "Crop Recommendation", 
        "Equipment Rental (Farmer)",
        "List My Equipment (Owner)", # NEW SIDEBAR OPTION
        "History", 
        "Settings"
    ])
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state.pop('username', None)
        st.success("Logged out successfully.")
        st.rerun()

    # --- Main Content Dispatch ---
    if feature == "Dashboard": render_dashboard()
    elif feature == "Prediction": prediction_page()
    elif feature == "Crop Recommendation": crop_recommendation_page()
    elif feature == "Equipment Rental (Farmer)": equipment_rental_page()
    elif feature == "List My Equipment (Owner)": list_equipment_page() # NEW DISPATCH
    elif feature == "History": render_history()
    elif feature == "Settings": render_settings()


def render_dashboard():
    # ... (Dashboard rendering logic remains the same)
    st.header("Live Agri-Market Feed & Environment 📡")
    if 'live_data' not in st.session_state:
        st.session_state['live_data'] = get_live_data()
    
    if st.button("Refresh Live Data (API Call)"):
        st.session_state['live_data'] = get_live_data()
        st.toast('Live data refreshed!', icon='🔄')

    temp, humidity, rainfall, market_status = st.session_state['live_data']

    st.subheader("Environmental Parameters (from API)")
    col_w1, col_w2, col_w3, col_m1 = st.columns(4)

    col_w1.metric(f"Temperature ({DEFAULT_CITY})", f"{temp}°C", "✅ LIVE API")
    col_w2.metric(f"Humidity ({DEFAULT_CITY})", f"{humidity}%", "✅ LIVE API")
    col_w3.metric("Rainfall (24h)", f"{rainfall} mm", "Simulated") 
    col_m1.metric("Market Status (Wheat)", market_status, "Simulated")
    st.markdown("---")
    st.subheader("Key Performance Indicators (KPIs)")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Current Farm Yield", "7.2 ton/ha", delta="0.5 ton/ha", delta_color="normal")
    col_kpi2.metric("Pest Risk Index", "Low", delta="-0.1", delta_color="inverse")
    col_kpi3.metric("Profit Margin (QTD)", "15%", delta="2%", delta_color="normal")
    
    st.subheader("Past 7 Days Temperature Trend")
    
    temp_history = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=7),
        'Temperature (°C)': np.round(np.random.uniform(30, 35, 7) + np.sin(np.linspace(0, 2*np.pi, 7)) * 2, 1)
    })
    temp_history = temp_history.set_index('Date')
    st.line_chart(temp_history)

def prediction_page():
    # ... (Prediction logic remains the same)
    st.header("Future Projections & Predictions 🔮")
    st.subheader("Model Configuration")
    commodity = st.selectbox("Select Commodity for Prediction:", COMMODITY_OPTIONS)
    target_col = PREDICTION_TARGETS[commodity]
    data = load_prediction_data()
    model = train_mock_model(data, target_col)
    
    if not model:
        st.error("Model training failed. Check data structure.")
        return

    st.success(f"Mock Linear Regression Model trained for **{target_col}**.")
    st.caption(f"Model MSE on Test Data: {st.session_state.get('model_mse', 'N/A'):.2f}")

    st.markdown("---")
    st.subheader(f"Predict {target_col} for {commodity}")

    col_p1, col_p2 = st.columns(2)
    temp_input = col_p1.slider("Average Temperature (°C)", 15.0, 40.0, 30.0)
    humidity_input = col_p2.slider("Average Humidity (%)", 50, 95, 75)
    
    col_p3, col_p4 = st.columns(2)
    rainfall_input = col_p3.number_input("Expected Rainfall (mm)", 0.0, 100.0, 15.0, step=0.5)
    fertilizer_input = col_p4.number_input("Fertilizer Usage (kg/hectare)", 10.0, 100.0, 50.0, step=1.0)

    if st.button(f"Run Prediction for {commodity}"):
        input_data = pd.DataFrame([[temp_input, humidity_input, rainfall_input, fertilizer_input]],
                                  columns=['Temperature', 'Humidity', 'Rainfall', 'Fertilizer_Usage'])
        
        try:
            prediction = model.predict(input_data)[0]
            st.markdown("### Prediction Result:")
            
            if "Price" in target_col: result_unit = "USD/ton"
            elif "Production" in target_col: result_unit = "bales"
            else: result_unit = "ton/hectare"

            st.metric(label=f"Predicted {target_col}", 
                      value=f"{prediction:,.2f} {result_unit}",
                      delta=f"Based on current inputs")
            st.info(f"The model suggests a {target_col} of **{prediction:,.2f} {result_unit}** under the given environmental conditions.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def crop_recommendation_page():
    # ... (Crop recommendation logic remains the same)
    st.header("Intelligent Crop Recommendation 🌳")
    st.info("Select your current field conditions to receive a recommended crop.")

    st.subheader("Field Input Parameters")
    
    col_c1, col_c2 = st.columns(2)
    temp = col_c1.slider("Average Field Temperature (°C)", 10.0, 45.0, 32.0, key="rec_temp")
    humidity = col_c2.slider("Average Field Humidity (%)", 40, 100, 80, key="rec_humidity")

    col_c3, col_c4 = st.columns(2)
    rainfall = col_c3.slider("Annual Rainfall (mm)", 100, 2000, 1200, key="rec_rainfall")
    soil_type = col_c4.selectbox("Soil Type", ["Loamy", "Clayey", "Sandy", "Alluvial"], key="rec_soil")
    
    st.markdown("---")

    if st.button("Get Crop Recommendation"):
        
        recommendation = "General Crops (Maize, Wheat)"
        
        if rainfall > 1500: recommendation = CROP_RECOMMENDATIONS["High Rainfall"]
        elif temp > 35: recommendation = CROP_RECOMMENDATIONS["High Temperature"]
        elif 20 <= temp <= 30 and soil_type == "Loamy": recommendation = CROP_RECOMMENDATIONS["Moderate Temperature"]
        elif humidity > 90 and rainfall > 1000: recommendation = CROP_RECOMMENDATIONS["High Humidity"]
        
        st.markdown("### Recommended Crop:")
        st.success(f"Based on the input parameters, the best recommended crop is **{recommendation}**.")
        st.balloons()
        st.caption("Disclaimer: This is a simplified recommendation. Consult with a local agricultural expert for final advice.")


def equipment_rental_page():
    """Farmer view for renting equipment."""
    st.header("Equipment Rental (Farmer View) 🚜")
    st.info("View available machinery from owners, check details, and book.")

    equipment_list = load_equipment()
    
    if not equipment_list:
        st.warning("No equipment is currently listed by owners. Check back later or use the 'List My Equipment' page to add one.")
        return

    # Convert list of dicts to DataFrame for display
    inventory_df = pd.DataFrame(equipment_list)
    inventory_df = inventory_df.drop(columns=['owner_contact', 'owner_address', 'id']) # Hide owner private info
    inventory_df = inventory_df.rename(columns={'name': 'Equipment', 'price_per_day': 'Price/Day', 'available_units': 'Available'})

    st.subheader("Available Equipment Inventory")
    st.dataframe(inventory_df, hide_index=True)

    st.markdown("---")
    st.subheader("Book Equipment")
    
    # Select box must use a unique identifier (ID)
    equipment_names = {item['name']: item for item in equipment_list}
    selected_name = st.selectbox("Select Equipment to Rent", list(equipment_names.keys()), key="book_select_equip")
    
    if not selected_name: return

    selected_item = equipment_names[selected_name]
    
    col_b1, col_b2 = st.columns(2)
    
    days = col_b1.number_input("Number of Days", 1, 30, 3, key="book_days")
    
    cost_per_day = selected_item["price_per_day"]
    total_cost = days * cost_per_day
    
    # Display cost before booking form
    col_b2.metric("Available Units", selected_item['available_units'])
    st.markdown(f"**Total Estimated Cost (Excl. Discount):** **${total_cost:,.2f}**")
    
    if selected_item['available_units'] == 0:
        st.error("This item is currently out of stock.")
        return
        
    st.markdown("---")
    st.subheader("Enter Booking Details")
    
    # --- Booking Form to collect farmer's details ---
    with st.form("booking_form"):
        farmer_name = st.text_input("Your Full Name", key="farmer_name")
        farmer_address = st.text_area("Your Farm/Delivery Address", key="farmer_address")
        farmer_contact = st.text_input("Your Contact Number", help="e.g., +91 9876543210", key="farmer_contact")
        
        book_submitted = st.form_submit_button("Confirm and Book Item")
        
        if book_submitted:
            if not all([farmer_name, farmer_address, farmer_contact]):
                st.error("Please fill in all your details to complete the booking.")
            elif selected_item['available_units'] < 1:
                st.error("Item is no longer available.")
            else:
                # 1. Record the booking
                bookings = load_bookings()
                booking_id = str(uuid.uuid4())
                booking_record = {
                    'id': booking_id,
                    'farmer_name': farmer_name,
                    'farmer_address': farmer_address,
                    'farmer_contact': farmer_contact,
                    'equipment_name': selected_name,
                    'days': days,
                    'total_cost': total_cost,
                    'booking_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'owner_contact': selected_item['owner_contact']
                }
                bookings[booking_id] = booking_record
                save_bookings(bookings)
                
                # 2. Reduce the inventory (Simulated)
                for item in equipment_list:
                    if item['name'] == selected_name:
                        item['available_units'] -= 1
                        break
                save_equipment(equipment_list)
                
                st.success(f"Booking confirmed! Your booking ID is **{booking_id}**.")
                st.info(f"Total payable: ${total_cost:,.2f}. The owner ({selected_item['owner_contact']}) will contact you shortly.")
                st.balloons()


def list_equipment_page():
    """New Owner view for listing equipment for rent."""
    st.header("List Your Equipment (Owner View) 💰")
    st.info("Help farmers by renting your idle machinery and earn income.")

    with st.form("list_equipment_form"):
        st.subheader("Equipment Details")
        equip_name = st.text_input("Equipment Name (e.g., John Deere 5045D Tractor)", key="equip_name")
        equip_description = st.text_area("Description (Condition, Features)", key="equip_desc")
        equip_price = st.number_input("Rental Price Per Day ($)", 50, 5000, 200, step=10, key="equip_price")
        equip_units = st.number_input("Number of Available Units", 1, 100, 1, step=1, key="equip_units")
        
        st.subheader("Your Contact Information")
        owner_name = st.text_input("Your Full Name", key="owner_name")
        owner_address = st.text_area("Your Location/Pickup Address", key="owner_address")
        owner_contact = st.text_input("Your Contact Number", help="Used by farmers to contact you", key="owner_contact")

        list_submitted = st.form_submit_button("List Equipment Now")
        
        if list_submitted:
            if not all([equip_name, equip_price, equip_units, owner_contact]):
                st.error("Please fill in all required fields (Name, Price, Units, Contact).")
            else:
                equipment_list = load_equipment()
                new_item = {
                    'id': str(uuid.uuid4()),
                    'name': equip_name,
                    'description': equip_description,
                    'price_per_day': equip_price,
                    'available_units': equip_units,
                    'owner_name': owner_name,
                    'owner_address': owner_address,
                    'owner_contact': owner_contact,
                    'listing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                equipment_list.append(new_item)
                save_equipment(equipment_list)
                
                st.success(f"Your equipment, **{equip_name}**, has been successfully listed!")
                st.info("It is now available for farmers to view and book on the Equipment Rental page.")


def render_history():
    """Renders the History view (Placeholder)."""
    st.header("Farm History & Analytics 📈")
    st.info("This section displays historical data, crop cycles, and past yield reports.")
    st.subheader("Sample Historical Data")
    st.dataframe(load_prediction_data().head(10))
    
    st.markdown("---")
    st.subheader("Your Booking Records")
    bookings = load_bookings()
    if bookings:
        # Convert dictionary of bookings to DataFrame for display
        booking_df = pd.DataFrame.from_dict(bookings, orient='index')
        st.dataframe(booking_df[['booking_date', 'equipment_name', 'days', 'total_cost', 'owner_contact']])
    else:
        st.info("No booking records found yet.")

def render_settings():
    """Renders the Settings view (Placeholder)."""
    st.header("User & Application Settings ⚙️")
    st.info(f"Welcome to settings, {st.session_state['username']}.")
    st.subheader("API Configuration")
    st.write(f"Current Weather City: **{DEFAULT_CITY}**")
    st.write("You can change the API Key in the `app_streamlit.py` file.")

# --- 3. Main Application Flow ---

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    dashboard_page()
else:
    login_page()