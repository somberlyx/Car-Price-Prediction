import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",  # Streamlit requires emoji or image for page icon
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS (modern card UI)
# ----------------------------
st.markdown("""
<style>
/* Import Font Awesome */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* ---------- Theme-aware tokens ---------- */
:root{
  --card-bg: rgba(255,255,255,.06);
  --card-border: rgba(255,255,255,.10);
  --muted: rgba(255,255,255,.70);
  --muted2: rgba(255,255,255,.55);
  --shadow: 0 12px 30px rgba(0,0,0,.35);
  --radius: 18px;
}

/* Light mode fallback */
@media (prefers-color-scheme: light) {
  :root{
    --card-bg: rgba(0,0,0,.04);
    --card-border: rgba(0,0,0,.08);
    --muted: rgba(0,0,0,.65);
    --muted2: rgba(0,0,0,.50);
    --shadow: 0 12px 28px rgba(0,0,0,.10);
  }
}

/* ---------- Layout spacing ---------- */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
hr { border: none; height: 1px; background: rgba(255,255,255,.08); }

/* ---------- Hero ---------- */
.hero {
  border: 1px solid var(--card-border);
  border-radius: calc(var(--radius) + 6px);
  padding: 1.25rem 1.35rem;
  background: radial-gradient(1200px 400px at 10% 10%, rgba(255,75,75,.22), transparent 55%),
              radial-gradient(900px 380px at 90% 30%, rgba(102,126,234,.22), transparent 60%),
              rgba(255,255,255,.03);
  box-shadow: var(--shadow);
}
.hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: .2px; }
.hero p { margin: .35rem 0 0 0; color: var(--muted); }

/* ---------- Badges ---------- */
.badge {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding: .28rem .65rem;
  margin-right: .4rem;
  border-radius: 999px;
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,.05);
  color: var(--muted);
  font-size: .85rem;
}

/* ---------- Cards ---------- */
.card {
  border: 1px solid var(--card-border);
  border-radius: var(--radius);
  padding: 1rem 1rem;
  background: var(--card-bg);
  box-shadow: 0 10px 24px rgba(0,0,0,.18);
}
.card-title { font-weight: 800; margin-bottom: .55rem; }
.muted { color: var(--muted); }
.small { font-size: .92rem; color: var(--muted2); }

/* Sticky panel */
.sticky { position: sticky; top: 1.05rem; }

/* Make Streamlit metric text visible on dark cards */
div[data-testid="stMetricValue"] { color: inherit; }
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricDelta"] { color: var(--muted2); }

/* ---------- Button ---------- */
.stButton>button {
  width: 100%;
  border-radius: 16px;
  padding: .95rem 1rem;
  font-weight: 900;
  font-size: 1.02rem;
  background: linear-gradient(135deg, #FF4B4B, #ff2e88);
  color: white;
  border: 0;
}
.stButton>button:hover { filter: brightness(.98); }

/* ---------- Tabs polish ---------- */
button[data-baseweb="tab"] {
  font-weight: 700 !important;
  letter-spacing: .2px;
}

/* ---------- Result hero ---------- */
.result {
  border-radius: calc(var(--radius) + 6px);
  padding: 1.2rem 1.3rem;
  color: white;
  background: radial-gradient(900px 260px at 20% 10%, rgba(255,255,255,.14), transparent 55%),
              linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: 1px solid rgba(255,255,255,.18);
  box-shadow: var(--shadow);
}
.result .big {
  font-size: 3rem;
  font-weight: 950;
  margin: .15rem 0;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_car_price_model.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.stop()

model, scaler = load_models()

# ----------------------------
# Header / Hero
# ----------------------------
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
    <div>
      <h1><i class="fas fa-car"></i> Car Price Predictor</h1>
      <p>Enter your car details and get an instant market estimate + what's driving the price.</p>
      <div style="margin-top:.5rem;">
        <span class="badge"><i class="fas fa-brain"></i> ML model</span>
        <span class="badge"><i class="fas fa-bolt"></i> Instant estimate</span>
        <span class="badge"><i class="fas fa-sliders-h"></i> Scenario explorer</span>
      </div>
    </div>
    <div style="display:flex; gap:.75rem; flex-wrap:wrap;">
      <div class="card" style="min-width:170px;">
        <div class="small">Model Accuracy (R²)</div>
        <div style="font-size:1.35rem; font-weight:900;">0.91</div>
      </div>
      <div class="card" style="min-width:170px;">
        <div class="small">Typical Error</div>
        <div style="font-size:1.35rem; font-weight:900;">±5%</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Inputs + Sticky Estimate Panel
# ----------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown('<div class="card-title">Step 1–3: Tell us about the car</div>', unsafe_allow_html=True)

    tabs = st.tabs(["① Basics", "② Specs", "③ Ownership"])

    # ---- Basics
    with tabs[0]:
        brands = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford',
                  'Renault', 'Chevrolet', 'Volkswagen', 'Skoda', 'Nissan', 'Datsun',
                  'BMW', 'Audi', 'Mercedes-Benz', 'Jaguar', 'Land', 'Lexus', 'Volvo',
                  'Fiat', 'Jeep', 'Kia', 'MG', 'Mitsubishi', 'Isuzu', 'Force',
                  'Daewoo', 'Opel', 'Ashok', 'Peugeot']

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-industry"></i> Brand</label>', unsafe_allow_html=True)
            brand = st.selectbox("Brand", sorted(brands), label_visibility="collapsed")
        with c2:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-gas-pump"></i> Fuel Type</label>', unsafe_allow_html=True)
            fuel_type = st.selectbox("Fuel", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], label_visibility="collapsed")

        st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-calendar-alt"></i> Year of Manufacture</label>', unsafe_allow_html=True)
        current_year = datetime.now().year
        year = st.slider("Year", 1995, current_year, 2015, label_visibility="collapsed")
        
        st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-road"></i> Kilometers Driven</label>', unsafe_allow_html=True)
        km_driven = st.number_input("KM", 0, 500000, 50000, 1000, label_visibility="collapsed")

        st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-cog"></i> Transmission</label>', unsafe_allow_html=True)
        transmission = st.selectbox("Trans", ['Manual', 'Automatic'], label_visibility="collapsed")

    # ---- Specs
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-cogs"></i> Engine (CC)</label>', unsafe_allow_html=True)
            engine_cc = st.number_input("Engine", 500, 5000, 1200, 100, label_visibility="collapsed")
        with c2:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-bolt"></i> Max Power (bhp)</label>', unsafe_allow_html=True)
            max_power = st.number_input("Power", 30.0, 500.0, 80.0, 5.0, label_visibility="collapsed")
        with c3:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-tachometer-alt"></i> Mileage (kmpl)</label>', unsafe_allow_html=True)
            mileage = st.number_input("Mileage", 5.0, 40.0, 18.0, 0.5, label_visibility="collapsed")

        st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-chair"></i> Number of Seats</label>', unsafe_allow_html=True)
        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2, label_visibility="collapsed")

    # ---- Ownership
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-user"></i> Owner Type</label>', unsafe_allow_html=True)
            owner = st.selectbox("Owner", [
                'First Owner', 'Second Owner', 'Third Owner',
                'Fourth & Above Owner', 'Test Drive Car'
            ], label_visibility="collapsed")
        with c2:
            st.markdown('<label style="font-size:0.9rem; color:var(--muted);"><i class="fas fa-store"></i> Seller Type</label>', unsafe_allow_html=True)
            seller_type = st.selectbox("Seller", ['Dealer', 'Individual', 'Trustmark Dealer'], label_visibility="collapsed")

        st.info("💡 Tip: owner history + annual mileage usually has a noticeable impact.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Derived age based on training assumption (2020)
    car_age = 2020 - year

# ----------------------------
# Feature engineering
# ----------------------------
def create_feature_vector(brand, car_age, km_driven, fuel_type, transmission,
                          engine_cc, max_power, mileage, seats, owner, seller_type):

    owner_map = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 0
    }
    owner_num = owner_map[owner]

    km_per_year = km_driven / (car_age + 1)
    power_to_engine = max_power / (engine_cc + 1)

    luxury_brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Jaguar', 'Lexus', 'Volvo', 'Land']
    is_luxury = 1 if brand in luxury_brands else 0

    power_age_interaction = max_power * car_age
    mileage_power_interaction = mileage * max_power

    if car_age <= 3:
        age_cat = 'new'
    elif car_age <= 7:
        age_cat = 'recent'
    elif car_age <= 15:
        age_cat = 'used'
    else:
        age_cat = 'old'

    if km_per_year <= 5000:
        km_cat = 'low'
    elif km_per_year <= 15000:
        km_cat = 'medium'
    elif km_per_year <= 30000:
        km_cat = 'high'
    else:
        km_cat = 'very_high'

    features = {
        'seats': seats,
        'mileage_num': mileage,
        'max_power_bhp': max_power,
        'car_age': car_age,
        'km_per_year': km_per_year,
        'power_to_engine': power_to_engine,
        'owner_num': owner_num,
        'is_luxury': is_luxury,
        'power_age_interaction': power_age_interaction,
        'mileage_power_interaction': mileage_power_interaction,
        'is_automatic': 1 if transmission == 'Automatic' else 0
    }

    all_brands = ['Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat',
                  'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep',
                  'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
                  'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata',
                  'Toyota', 'Volkswagen', 'Volvo']

    for b in all_brands:
        features[f'brand_{b}'] = 1 if brand == b else 0

    age_categories = ['recent', 'used', 'old']
    for cat in age_categories:
        features[f'age_category_{cat}'] = 1 if age_cat == cat else 0

    km_categories = ['medium', 'high', 'very_high']
    for cat in km_categories:
        features[f'km_category_{cat}'] = 1 if km_cat == cat else 0

    fuel_types = ['Diesel', 'LPG', 'Petrol']
    for ft in fuel_types:
        features[f'fuel_{ft}'] = 1 if fuel_type == ft else 0

    seller_types = ['Individual', 'Trustmark Dealer']
    for st_type in seller_types:
        features[f'seller_type_{st_type}'] = 1 if seller_type == st_type else 0

    return features

# quick preview metrics for sticky panel
owner_map = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0
}
owner_num = owner_map[owner]
km_per_year = km_driven / (car_age + 1)

with right:
    st.markdown('<div class="card-title">Live Preview</div>', unsafe_allow_html=True)

    st.write(f"**{brand}** • {year} • {fuel_type} • {transmission}")
    st.caption(f"{km_driven:,.0f} km • {engine_cc}cc • {max_power:.0f} bhp • {mileage:.1f} kmpl")

    c1, c2, c3 = st.columns(3)
    c1.metric("Age", f"{car_age}y")
    c2.metric("KM/yr", f"{km_per_year:,.0f}")
    c3.metric("Owners", f"{owner_num}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ---- Currency controls (display only) ----
    st.markdown('<div class="card-title">Currency</div>', unsafe_allow_html=True)

    currency_options = {
        "INR (₹)": {"code": "INR", "symbol": "₹", "per_inr_default": 1.0},
        "USD ($)": {"code": "USD", "symbol": "$", "per_inr_default": 0.012},
        "EUR (€)": {"code": "EUR", "symbol": "€", "per_inr_default": 0.011},
        "GBP (£)": {"code": "GBP", "symbol": "£", "per_inr_default": 0.0095},
        "AED (د.إ)": {"code": "AED", "symbol": "د.إ", "per_inr_default": 0.044},
        "PKR (₨)": {"code": "PKR", "symbol": "₨", "per_inr_default": 3.7},
    }

    currency_label = st.selectbox("Display currency", list(currency_options.keys()), index=0)
    currency = currency_options[currency_label]

    st.caption("Exchange rate is editable (approx defaults).")
    rate_per_inr = st.number_input(
        f"1 INR equals how many {currency['code']}?",
        min_value=0.000001,
        value=float(currency["per_inr_default"]),
        step=float(currency["per_inr_default"]) * 0.05 if currency["per_inr_default"] > 0 else 0.01,
        format="%.6f"
    )

    # helper
    def convert_from_inr(amount_in_inr: float) -> float:
        return amount_in_inr * rate_per_inr

    def fmt_money(amount: float) -> str:
        return f"{currency['symbol']}{amount:,.0f}"


    predict = st.button("🔮 Get Estimate")

    st.markdown("<div class='small muted'>We'll also show scenario comparisons and price drivers.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction + Results Layout
if predict:
    with st.spinner("Calculating your car's value..."):
        try:
            features = create_feature_vector(
                brand, car_age, km_driven, fuel_type, transmission,
                engine_cc, max_power, mileage, seats, owner, seller_type
            )
            feature_df = pd.DataFrame([features])

            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
            else:
                raise ValueError(
                    "Model missing feature_names_in_. Retrain or save training columns."
                )

            feature_df = feature_df.reindex(columns=expected_cols, fill_value=0)

            predicted_log_price = model.predict(feature_df)[0]
            predicted_price_lakhs = float(np.expm1(predicted_log_price))  # model output shown as "lakhs"
            predicted_price_inr = predicted_price_lakhs * 100000          # ✅ lakhs -> INR

            lower_inr = predicted_price_inr * 0.95
            upper_inr = predicted_price_inr * 1.05

            # convert to chosen currency for display
            predicted_display = convert_from_inr(predicted_price_inr)
            lower_display = convert_from_inr(lower_inr)
            upper_display = convert_from_inr(upper_inr)


            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            # Result hero row
            r1, r2 = st.columns([1.2, 1], gap="large")

            with r1:
                st.markdown(f"""
                <div class="big">{fmt_money(predicted_display)}</div>
                <div style="opacity:.9;">Range (±5%): {fmt_money(lower_display)} – {fmt_money(upper_display)}</div>
                <div class="small" style="opacity:.85; margin-top:.35rem;">
                Note: Model output is treated as <b>lakhs</b> (×100,000 INR) before conversion.
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown('<div class="card-title">Quick Insights</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("Condition Score*", f"{max(30, 100 - car_age*5):.0f}/100")
                c2.metric("Luxury Brand", "Yes" if brand in ['Audi','BMW','Mercedes-Benz','Jaguar','Lexus','Volvo','Land'] else "No")
                st.caption("*Simple heuristic (not the ML feature importance).")
                st.markdown('</div>', unsafe_allow_html=True)

            # Price drivers
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-lightbulb"></i> What\'s likely moving your price</h3>', unsafe_allow_html=True)

            p1, p2 = st.columns(2, gap="large")

            with p1:
                st.markdown('<div class="card-title"><i class="fas fa-check-circle" style="color:#10b981;"></i> Value boosters</div>', unsafe_allow_html=True)
                bullets = []
                if owner == "First Owner": bullets.append("First owner history")
                if car_age < 5: bullets.append("Relatively newer vehicle age")
                if km_per_year < 10000: bullets.append("Lower annual mileage")
                if brand in ['Audi','BMW','Mercedes-Benz','Toyota','Honda']: bullets.append("Stronger brand perception / resale")
                if transmission == "Automatic": bullets.append("Automatic transmission demand")
                if not bullets: bullets = ["No strong boosters detected — price driven mostly by baseline specs."]
                st.write("\n".join([f"- {b}" for b in bullets]))
                st.markdown("</div>", unsafe_allow_html=True)

            with p2:
                st.markdown('<div class="card-title"><i class="fas fa-exclamation-triangle" style="color:#f59e0b;"></i> Headwinds</div>', unsafe_allow_html=True)
                bullets = []
                if car_age > 10: bullets.append("Older vehicle age increases depreciation")
                if km_per_year > 20000: bullets.append("Higher annual mileage")
                if owner in ["Third Owner","Fourth & Above Owner"]: bullets.append("More ownership transfers")
                if fuel_type == "Diesel": bullets.append("Diesel resale can vary by region/regulation")
                if not bullets: bullets = ["No major red flags detected."]
                st.write("\n".join([f"- {b}" for b in bullets]))
                st.markdown("</div>", unsafe_allow_html=True)

            # Scenario explorer
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-chart-bar"></i> Scenario Explorer</h3>', unsafe_allow_html=True)

            # scenarios computed in INR first (since your multipliers are “price logic”)
            scenarios_inr = {
                "Your Car": predicted_price_inr,
                "If First Owner": predicted_price_inr * 1.1 if owner != "First Owner" else predicted_price_inr,
                "If 5 Years Newer": predicted_price_inr * 1.3 if car_age > 5 else predicted_price_inr,
                "If Lower Mileage": predicted_price_inr * 1.08 if km_per_year > 15000 else predicted_price_inr,
            }

            # convert for chart display
            scenarios_display = {k: convert_from_inr(v) for k, v in scenarios_inr.items()}

            fig = go.Figure(data=[
                go.Bar(
                    x=list(scenarios_display.keys()),
                    y=list(scenarios_display.values()),
                    text=[fmt_money(v) for v in scenarios_display.values()],
                    textposition="auto",
                )
            ])

            fig.update_layout(
                height=380,
                yaxis_title=f"Price ({currency['code']})",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Footer note
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
              <div class="card-title"><i class="fas fa-info-circle"></i> Notes</div>
              <div class="small muted">
                This is a model-based estimate. Actual price varies by condition, location, negotiation, listing quality, and market demand.
              </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly and try again.")