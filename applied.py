import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy, norm, gaussian_kde
from scipy.interpolate import interp1d
import io

# ===================================================================
# === 1. Hodgkin-Huxley Model (The "Engine")
# =This code is the core logic from your script.
# ===================================================================

# === Parameters ===
Cm = 1
g_Na, g_L = 120, 0.3
E_Na, E_L = 55, -49

# === Gating kinetics ===
def alpha_n(V): return 0.01*(V+55)/(1 - np.exp(-(V+55)/10) + 1e-9)
def beta_n(V): return 0.125*np.exp(-(V+65)/80)
def alpha_m(V): return 0.1*(V+40)/(1 - np.exp(-(V+40)/10) + 1e-9)
def beta_m(V): return 4*np.exp(-(V+65)/18)
def alpha_h(V): return 0.07*np.exp(-(V+65)/20)
def beta_h(V): return 1 / (1 + np.exp(-(V+35)/10))

# === Ionic currents ===
def I_Na(V, m, h): return g_Na * m**3 * h * (V - E_Na)
def I_K(V, n, gK, EK): return gK * n**4 * (V - EK)
def I_L(V): return g_L * (V - E_L)

# === Hodgkin-Huxley differential equations ===
def dALLdt(X, t, I_inj_func, gK, EK):
    V, m, h, n = X
    I_inj = I_inj_func(t)
    dVdt = (I_inj - I_Na(V, m, h) - I_K(V, n, gK, EK) - I_L(V)) / Cm
    dmdt = alpha_m(V)*(1 - m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1 - h) - beta_h(V)*h
    dndt = alpha_n(V)*(1 - n) - beta_n(V)*n
    return [dVdt, dmdt, dhdt, dndt]

# ===================================================================
# === 2. Analysis Functions (The "Engine")
# ===================================================================

@st.cache_data # Cache this function for performance
def solve_gating_vars(V_trace, t_trace):
    def dGatingsdt(X, t):
        m, h, n = X
        V = np.interp(t, t_trace, V_trace) 
        dmdt = alpha_m(V)*(1 - m) - beta_m(V)*m
        dhdt = alpha_h(V)*(1 - h) - beta_h(V)*h
        dndt = alpha_n(V)*(1 - n) - beta_n(V)*n
        return [dmdt, dhdt, dndt]
    X0 = [0.05, 0.6, 0.32] 
    X = odeint(dGatingsdt, X0, t_trace)
    return X[:,0], X[:,1], X[:,2] # m, h, n

def estimate_params_windowed(t_sec, V, I_func, window_size_sec, step_size_sec):
    estimated_times = []
    estimated_gK = []
    estimated_EK = []
    # Check for sufficient data points
    if len(t_sec) < 2:
        return [], [], []
    dt = t_sec[1] - t_sec[0]
    # Handle zero dt
    if dt == 0:
        return [], [], []
        
    dVdt = np.gradient(V, dt)
    m, h, n = solve_gating_vars(V, t_sec) # Using the cached function
    I_inj_trace = I_func(t_sec)
    I_Na_trace = I_Na(V, m, h)
    I_L_trace = I_L(V)
    Y_target = (Cm * dVdt) - I_inj_trace + I_Na_trace + I_L_trace
    X1_feature = - (n**4) * V
    X2_feature = n**4
    X_features = np.vstack([X1_feature, X2_feature]).T
    n_points = len(t_sec)
    window_pts = int(window_size_sec / dt)
    step_pts = int(step_size_sec / dt)
    
    # Ensure window and step are valid
    if window_pts <= 0 or step_pts <= 0 or window_pts > n_points:
        st.error("Invalid window/step size for the given data length.")
        return [], [], []

    model = LinearRegression()
    
    for i in range(0, n_points - window_pts, step_pts):
        win_start = i
        win_end = i + window_pts
        t_window_mid = t_sec[win_start + window_pts // 2]
        Y_win = Y_target[win_start:win_end]
        X_win = X_features[win_start:win_end, :]
        try:
            model.fit(X_win, Y_win)
            C1 = model.coef_[0]  
            C2 = model.coef_[1]  
            if abs(C1) > 1e-3:
                gK_est = C1
                EK_est = C2 / C1
                if gK_est > 0 and gK_est < 100 and EK_est < 0 and EK_est > -150:
                    estimated_times.append(t_window_mid)
                    estimated_gK.append(gK_est)
                    estimated_EK.append(EK_est)
        except Exception as e:
            pass # Skip windows that fail
            
    return estimated_times, estimated_gK, estimated_EK

def run_simulation(params, t_short, I_inj_short, X0):
    gK, EK = params
    try:
        X = odeint(dALLdt, X0, t_short, args=(I_inj_short, gK, EK))
        V_sim = X[:,0]
        if not np.all(np.isfinite(V_sim)):
            return None 
        return V_sim
    except Exception:
        return None 

def objective_rmse(V_short, V_sim):
    if V_sim is None:
        return 1e9
    return np.sqrt(np.mean((V_short - V_sim)**2))

def objective_kl(V_short, V_sim):
    if V_sim is None:
        return 1e9 
    v_range = (-90, 60)
    v_bins = 100
    hist_exp, _ = np.histogram(V_short, bins=v_bins, range=v_range, density=False)
    hist_sim, _ = np.histogram(V_sim, bins=v_bins, range=v_range, density=False)
    if np.sum(hist_sim) == 0:
        return 1e9 
    hist_exp_s = hist_exp + 1e-9
    hist_sim_s = hist_sim + 1e-9
    P = hist_exp_s / np.sum(hist_exp_s)
    Q = hist_sim_s / np.sum(hist_sim_s)
    return entropy(P, Q)

# ===================================================================
# === 3. Streamlit GUI Application
# ===================================================================

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("🔬 Hodgkin-Huxley Parameter Estimation Framework")

# --- Sidebar (User Inputs) ---
st.sidebar.header("Analysis Controls")
uploaded_file = st.sidebar.file_uploader("Upload Voltage Data (Excel/CSV)", type=["xlsx", "csv"])

st.sidebar.subheader("Data & Model Parameters")
dt = st.sidebar.number_input("Time Step (dt) (s)", value=0.01, format="%.3f", step=0.001)
column_number = st.sidebar.number_input("Column Number (1-5)", value=1, min_value=1, max_value=5, step=1)

st.sidebar.subheader("Dynamic Regression Parameters")
win_size = st.sidebar.number_input("Window Size (s)", value=0.1, format="%.2f", step=0.01)
step_size = st.sidebar.number_input("Step Size (s)", value=0.02, format="%.2f", step=0.01)

run_button = st.sidebar.button("Run Analysis", type="primary")

# --- Main Page (Results) ---
if not uploaded_file:
    st.info("Please upload a voltage data file (Excel or CSV) via the sidebar to begin.")
else:
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, header=None)
        else:
            data = pd.read_excel(uploaded_file, header=None, skiprows=3)
        # Get the selected column using formula: actual_column = 2*n - 2 (where n is 1-indexed user input)
        col_idx = 2 * int(column_number) - 2
        if col_idx >= len(data.columns):
            st.error(f"Column {column_number} does not exist in the data. Maximum column: {(len(data.columns) + 2) // 2}")
            st.stop()
        V_exp = data.iloc[:, col_idx].values
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if run_button:
        # --- Create data from inputs ---
        n_points = len(V_exp)
        t_exp_sec = np.arange(0, n_points * dt, dt)
        I_ext = np.zeros_like(V_exp)
        I_inj_func = interp1d(t_exp_sec, I_ext, fill_value="extrapolate")

        # --- Section 1: Introduction ---
        st.header("Introduction")
        st.write("""
        The Hodgkin-Huxley (HH) model is a Nobel Prize-winning mathematical model that describes how action potentials in neurons are initiated and propagated. It models the neuron's membrane as a circuit with a capacitor (the membrane) and variable resistors (the ion channels).
        
        This application uses an experimental voltage trace to perform a "dynamic regression" to estimate the time-varying parameters of the potassium channel ($g_K$ and $E_K$), testing the hypothesis that these parameters are not constant.
        """)
        st.divider()

        # --- Section 2: Dynamic Parameter Estimation ---
        st.header("Dynamic Parameter Estimation Results")
        with st.spinner("Running dynamic parameter estimation..."):
            est_t, est_gK, est_EK = estimate_params_windowed(t_exp_sec, V_exp, I_inj_func, win_size, step_size)
            
            # --- MODIFIED: Reduced plot size ---
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True) # Was (12, 10)
            
            if est_t: # Only plot if results were found
                ax1.plot(est_t, est_gK, 'r.-', label=f'Estimated $g_K$ ({win_size*1000}ms window)')
                ax2.plot(est_t, est_EK, 'b.-', label=f'Estimated $E_K$ ({step_size*1000}ms step)')
            else:
                st.warning("No valid parameter estimates found for the dynamic regression.")

            ax1.set_ylabel('$g_K$ (mS/cm²)')
            ax1.legend()
            ax1.set_title('Dynamic Parameter Estimation Results')
            
            ax2.set_xlabel('Time (sec)')
            ax2.set_ylabel('$E_K$ (mV)')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig1)
        st.divider()

        # --- Section 3: Voltage Distribution Plots ---
        st.header("Visualization of Experimental Voltage Distribution")
        
        V_exp_no_outliers = V_exp[np.abs(V_exp - np.mean(V_exp)) < 3 * np.std(V_exp)]
        
        # --- MODIFIED: Added crash protection ---
        if V_exp_no_outliers.size == 0:
            st.warning("Could not generate distribution plots. Input data might be empty, invalid, or have no variance.")
        else:
            x_range = np.linspace(np.min(V_exp_no_outliers), np.max(V_exp_no_outliers), 500)

            # --- Display all three plots in columns with reduced size ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
                ax_hist.hist(V_exp_no_outliers, bins=100, density=True, histtype='stepfilled',
                         alpha=0.7, label='Discrete (Histogram)')
                ax_hist.set_title('1. Discrete Distribution (Histogram)')
                ax_hist.set_xlabel('Voltage (mV)')
                ax_hist.set_ylabel('Probability Density')
                ax_hist.legend()
                plt.tight_layout()
                st.pyplot(fig_hist)

            with col2:
                fig_kde, ax_kde = plt.subplots(figsize=(5, 3))
                kde = gaussian_kde(V_exp_no_outliers)
                p_kde = kde(x_range)
                ax_kde.plot(x_range, p_kde, 'k', linewidth=2, label='Continuous (KDE)')
                ax_kde.set_title('2. Continuous Distribution (Kernel Density Estimate)')
                ax_kde.set_xlabel('Voltage (mV)')
                ax_kde.set_ylabel('Probability Density')
                ax_kde.legend()
                plt.tight_layout()
                st.pyplot(fig_kde)

            with col3:
                fig_norm, ax_norm = plt.subplots(figsize=(5, 3))
                mu, std = norm.fit(V_exp_no_outliers)
                p_norm = norm.pdf(x_range, mu, std)
                ax_norm.plot(x_range, p_norm, 'r--', linewidth=2, 
                         label=rf'Fitted Normal (Gaussian)\n$\mu={mu:.2f}, \sigma={std:.2f}$')
                ax_norm.set_title('3. Fitted Normal (Gaussian) Distribution')
                ax_norm.set_xlabel('Voltage (mV)')
                ax_norm.set_ylabel('Probability Density')
                ax_norm.legend()
                plt.tight_layout()
                st.pyplot(fig_norm)
        st.divider()

        # --- Section 4: Theory about Continuity ---
        st.header("Theory: Continuity of the KL Divergence Integrand")
        st.write("""
        The Kullback-Leibler (KL) divergence for two continuous probability distributions, $P(x)$ (experimental) and $Q(x)$ (simulated), is defined as:
        """)
        st.latex(r"D_{KL}(P || Q) = \int_{-\infty}^{\infty} P(x) \log\left(\frac{P(x)}{Q(x)}\right) dx")
        st.write("""
        The expression to be integrated is $f(x) = P(x) \log(P(x) / Q(x))$. Its continuity relies on:
        
        1.  **Continuity of P and Q:** We model our data (e.g., with a Kernel Density Estimate) to get continuous functions $P(x)$ and $Q(x)$.
        2.  **Continuity of log(x):** The $\log$ function is continuous for all positive inputs.
        3.  **Absolute Continuity:** For $D_{KL}$ to be defined, we must ensure that $Q(x) > 0$ wherever $P(x) > 0$. In our discrete histogram model, we achieve this by adding a tiny value (smoothing) to all bins, which prevents division by zero and ensures a well-defined, continuous integrand.
        """)
        st.divider()

        # --- Section 5: RMSE and KL Divergence Visualization ---
        st.header("Objective Function Visualization")
        with st.spinner("Calculating objective function surfaces... This may take a minute."):
            t_analysis_end = 1.0
            idx_end = np.where(t_exp_sec >= t_analysis_end)[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = len(t_exp_sec) 
            
            t_short = t_exp_sec[:idx_end]
            V_short = V_exp[:idx_end]
            I_inj_short = interp1d(t_short, I_ext[:idx_end], fill_value="extrapolate")
            X0 = [-65, 0.05, 0.6, 0.32] 
            
            gK_range = np.linspace(30, 45, 10) 
            EK_range = np.linspace(-80, -65, 10) 
            
            rmse_surface = np.zeros((len(gK_range), len(EK_range)))
            kl_surface = np.zeros((len(gK_range), len(EK_range)))

            for i, gK in enumerate(gK_range):
                for j, EK in enumerate(EK_range):
                    params = [gK, EK]
                    V_sim = run_simulation(params, t_short, I_inj_short, X0) 
                    rmse_surface[i, j] = objective_rmse(V_short, V_sim)
                    kl_surface[i, j] = objective_kl(V_short, V_sim)
            
            # --- MODIFIED: Reduced plot size ---
            fig_surf, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # Was (16, 7)
            [G, E] = np.meshgrid(EK_range, gK_range)
            
            cp1 = ax1.contourf(E, G, rmse_surface, levels=20)
            fig_surf.colorbar(cp1, ax=ax1, label='RMSE')
            ax1.set_xlabel('$g_K$ (mS/cm²)')
            ax1.set_ylabel('$E_K$ (mV)')
            ax1.set_title('RMSE Objective Function Surface')

            cp2 = ax2.contourf(E, G, np.log(kl_surface + 1e-9), levels=20)
            fig_surf.colorbar(cp2, ax=ax2, label='log(KL Divergence)')
            ax2.set_xlabel('$g_K$ (mS/cm²)')
            ax2.set_ylabel('$E_K$ (mV)')
            ax2.set_title('KL Divergence Objective Function Surface')
            
            plt.tight_layout()
            st.pyplot(fig_surf)
        
        st.success("Analysis Complete!")