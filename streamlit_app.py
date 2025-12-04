import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from Calculator_DOS_lib import (
    numerical_dos_free_particles,
    dos_1d_chain,
    dos_2d_square_lattice,
    dos_1d_phonons
)

def intro():
    import streamlit as st

    st.write("# Phys 459: Project")
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        This 'app' was made for the PHYS 459: Solid States Physics class autumn 2025.
        As it is mentioned in the report send to the professor (hi Dr. Maiti!!).

        In the dropdown on the left you will be able to:\\
            1. Compute the Density of State (DOS) for a given energy.\\
            2. Look at different DOS spectrum.\\
            3. Input a csv file containing energies values to plot the spectrum with an overlay
                of the expected spectrum.\\
            4. Maybe I'll add some stuffs later. Like lattice!!!!
        
    """
    )

def background():
    import streamlit as st
    st.markdown("Add some stuffs to explain DOS")


def  dos_spectrum():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from Calculator_DOS_lib import (
        numerical_dos_free_particles,
        dos_1d_chain,
        dos_2d_square_lattice,
        dos_1d_phonons
    )


    st.title("Spectrum")
    st.write("Note on the spectrum, we use a Monte-Carlo method to add some randomness to the data points. " \
    "It is the reason why the curve is not smooth.")

    model = st.selectbox("Model", 
                        ["Free electrons", "1D chain", "2D square lattice", "1D phonons"])

    dim = st.selectbox("Dimension", [1,2,3]) if model == "Free electrons" else None

    Nk = st.slider("Number of k-points", 1_000, 30_000_000, 50_000)


    if st.button("Compute DOS"):
        # ---- Compute model ----
        if model == "Free electrons":
            with st.spinner("Computing DOS..."):
                E, gE = numerical_dos_free_particles(dim, num_k_points=Nk)
                df = pd.DataFrame({"Energy": E, "DOS": gE})
            
        elif model == "1D chain":
            with st.spinner("Computing DOS..."):
                E, gE = dos_1d_chain(Nk)
                df = pd.DataFrame({"Energy": E, "DOS": gE})
            
        elif model == "2D square lattice":
            with st.spinner("Computing DOS..."):
                E, gE = dos_2d_square_lattice(Nk)
                df = pd.DataFrame({"Energy": E, "DOS": gE})
           
        else:  # phonons
            with st.spinner("Computing DOS..."):
                E, gE = dos_1d_phonons(Nk)
                df = pd.DataFrame({"Energy": E, "DOS": gE})
        

        import plotly.express as px
        df_plot = df.copy()
        scale = 1e-12
        df_plot["DOS_scaled"] = df_plot["DOS"] * scale

        fig = px.line(df_plot, x="Energy", y="DOS_scaled")
        fig.update_yaxes(title=f"DOS g(E)  [×10^{int(np.log10(1/scale))}]")
        fig.update_xaxes(title="Energy (eV)")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button( "Download DOS data as CSV", csv, "dos_data.csv", "text/csv")




def fast_dos():
    import streamlit as st
    import numpy as np
    import pandas as pd

    from Calculator_DOS_lib import (
        numerical_dos_free_particles,
        dos_1d_chain,
        dos_2d_square_lattice,
        dos_1d_phonons
    )

    st.title("Density of States Calculator")
    st.caption("A minimal, fast DOS calculator for PHYS 459.")

    st.markdown("---")

    # ------- INPUT PANEL -------
    st.subheader("Configuration")

    model = st.pills(
        "Model",  
        ["Free electrons", "1D chain", "2D square lattice", "1D phonons"],
        selection_mode="single"
    )

    col_left, col_right = st.columns([0.32, 0.68])

    with col_left:
        if model == "Free electrons":
            dim = st.radio("Dimension", [1, 2, 3], horizontal=True)
        else:
            dim = None

        Nk = st.slider(
            "Number of k-points",
            1_000, 10_000_000, 
            100_000,
            help="More points = smoother DOS but slower computation."
        )

    with col_right:
        target_energy = st.number_input(
            "Target Energy (eV)",
            placeholder="Enter energy...",
            format="%.5f"
        )

    st.markdown("---")

    
    compute = st.button("Compute DOS", use_container_width=True)

    if compute:
        with st.spinner("Fetching the corresponding bits... be right back"):
            
            # ---- Compute model ----
            if model == "Free electrons":
                E, gE = numerical_dos_free_particles(dim, num_k_points=Nk)
            elif model == "1D chain":
                E, gE = dos_1d_chain(Nk)
            elif model == "2D square lattice":
                E, gE = dos_2d_square_lattice(Nk)
            else:
                E, gE = dos_1d_phonons(Nk)

            # ---- Interpolate result ----
            try:
                corr_dos = np.interp(target_energy, E, gE)
            except Exception:
                corr_dos = None

        
        # ------- RESULT CARD -------
        st.subheader("Result")

        if corr_dos is None:
            st.error("Invalid target energy or computation failed.")
        else:
            st.markdown(
                f"""
                <div style="
                    padding: 18px;
                    border-radius: 12px;
                    background-color: #0e1117;
                    border: 1px solid #333;
                    color: #7fe9ff;
                    font-size: 18px;
                ">
                    <strong>DOS(E = {target_energy:.3f} eV):</strong><br>
                    <span style="font-size: 26px; font-weight: bold; color: #91ffff;">
                        {corr_dos:.3e}
                    </span>
                    <br><span style="font-size: 15px; opacity: 0.7;">units: states/eV/unit</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------- OPTIONAL PLOT -------
        with st.expander("View DOS spectrum plot"):
            import plotly.express as px
            idx = np.searchsorted(E, target_energy)
            lo = max(idx-200,0)
            hi = min(idx + 200, len(E))
            df = pd.DataFrame({"Energy": E, "DOS": gE})
            scale = 1e-12
            df["DOS_scaled"] = df["DOS"] * scale

            fig = px.line(
                df, x="Energy", y="DOS_scaled",
                title=f"{model} – DOS Spectrum",
                labels={"DOS_scaled": f"DOS × {1/scale:.0e}", "Energy": "Energy (eV)"},
            )
            fig.update_layout(height=450)

            st.plotly_chart(fig, use_container_width=True)
                    
        

def user_data():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    from scipy.constants import hbar, m_e
    import matplotlib.pyplot as plt
    

    from Calculator_DOS_lib import (dos_from_user_data, numerical_dos_free_particles,
        dos_1d_chain,
        dos_2d_square_lattice,
        dos_1d_phonons)


    st.set_page_config(page_title="User DOS Analyzer", layout="centered")
    st.title("Free-Particle Density of States from Experimental/Data Points")

    st.write("""
    Upload a CSV file containing at least an **Energy** column.  
    Optionally include an **Intensity** column if you want to compare against theoretical free-particle models (1D/2D/3D).
    """)

    # -----------------------------
    # 1. File Upload & Validation
    # -----------------------------
    with st.container(border=True):
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    df = None
    energy_col = None
    intensity_col = None

    if uploaded_file is not None:
        delimiter = st.radio(
            "Delimiter",
            ["comma", "semicolon", "tab", "space"],
            horizontal=True
        )
        sep = {
            "comma": ",",
            "semicolon": ";",
            "tab": "\t",
            "space": " "
        }[delimiter]

        try:
            # Preview header
            header_df = pd.read_csv(uploaded_file, nrows=0)
            columns = header_df.columns.tolist()

            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.write("**Detected columns:**")
                st.write(",".join(columns))
            with col2:
                if st.checkbox("Header looks correct", value=True):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, on_bad_lines="skip")
                    
                    st.success(f"Loaded {len(df):,} rows with delimiter '{sep}'")
                    st.dataframe(df.head(), use_container_width=True)
                else:
                    st.warning("Please fix and re-upload the file.")
                    st.stop()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    st.markdown("---")

    # -----------------------------
    # 2. Configuration
    # -----------------------------
    if df is not None:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("DOS Computation Settings")
            bins = st.select_slider(
                "Number of energy bins (resolution)",
                options=[100, 200, 300, 400, 500, 750, 1000],
                value=400
            )
            try:
                bins = int(bins)
            except:
                bins = 400

            if bins is None:
                bins = 400

        with col_right:
            st.subheader("Column Selection")
            energy_col = st.selectbox("Select **Energy** column", df.columns)
            intensity_col = st.selectbox(
                "Select **Intensity** column (optional for model fitting)",
                ["None"] + df.columns.tolist(),
                index=0
            )
            if intensity_col == "None":
                intensity_col = None

        st.markdown("---")
        st.write(bins)
        # -----------------------------
        # 3. Compute DOS Button
        # -----------------------------
        if st.button("Compute Density of States", type="primary"):
        
            if df is None or energy_col is None :
                st.error("Please upload data and select an energy column.")
                st.stop()
            
            E_data = pd.to_numeric(df[energy_col], errors='coerce').dropna().values
            
            if len(E_data) == 0:
                st.error("No valid numeric energy values found.")
                st.stop()
            

            with st.spinner("Computing density of states from your data..."):
                try:
                    if bins is None:
                        st.error("Internal error: bins is None")
                        st.stop()

                    if energy_col is None:
                        st.error("No energy column selected.")
                        st.stop()

            

                    if len(E_data) == 0:
                        st.error("Energy column contains no numeric data.")
                        st.stop()

                    energies, dos = dos_from_user_data(E_data, bins=bins, spin_degeneracy=2)
                except Exception as e:
                    st.error(f"Error computing DOS: {e}")
                    st.stop()

            # -----------------------------
            # 4. Display Results
            # -----------------------------
            st.success("Density of States computed successfully!")

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(energies, dos, color="#1f77b4", lw=2)
            ax.set_xlabel("Energy (a.u.)")
            ax.set_ylabel("DOS (states / energy / unit cell)")
            ax.set_title("Computed Density of States")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Data table & download
            result_df = pd.DataFrame({"Energy": energies, "DOS": dos})
            st.download_button(
                label="Download DOS as CSV",
                data=result_df.to_csv(index=False).encode(),
                file_name="computed_DOS.csv",
                mime="text/csv"
            )

            # -----------------------------
            # 5. Model Comparison (1D/2D/3D free particles)
            # -----------------------------
            if intensity_col is not None:
                st.markdown("### Model Comparison (1D / 2D / 3D Free Particles)")

                intensity = pd.to_numeric(df[intensity_col], errors='coerce').dropna().values
                if len(intensity) != len(E_data):
                    st.warning("Intensity column length doesn't match Energy → skipping model fit.")
                else:
                    with st.spinner("Comparing with theoretical free-particle models..."):
                        errors = []
                        models = ["1D", "2D", "3D"]

                        for dim in [1, 2, 3]:
                            E_ref, g_ref = numerical_dos_free_particles(dim, num_k_points=200_000)
                            # Interpolate reference DOS onto user energies
                            g_interp = np.interp(E_data, E_ref, g_ref, left=0, right=0)
                            # Normalize both to same area for fair comparison
                            g_interp /= np.trapezoid(g_interp, E_data)
                            intensity_norm = intensity / np.trapezoid(intensity, E_data)
                            mse = np.mean((intensity_norm - g_interp)**2)
                            errors.append(mse)

                        best_idx = int(np.argmin(errors))
                        best_model = models[best_idx]
                        best_error = errors[best_idx]

                        col1, col2, col3 = st.columns(3)
                        for i, (model, err) in enumerate(zip(models, errors)):
                            with [col1, col2, col3][i]:
                                delta = " Best match" if i == best_idx else ""
                                st.metric(f"{model} Model Error", f"{err:.2e}", delta)

                        st.success(f"**Best matching model: {best_model} free-particle gas**")

                        # Optional: overlay best model
                        if st.checkbox("Show best theoretical model on plot"):
                            E_ref, g_ref = numerical_dos_free_particles(best_idx + 1, num_k_points=500_000)
                            g_interp = np.interp(energies, E_ref, g_ref, left=0, right=0)
                            g_interp /= np.trapezoid(g_interp, energies)
                            dos_norm = dos / np.trapezoid(dos, energies)

                            fig2, ax2 = plt.subplots()
                            ax2.plot(energies, dos_norm, label="Your DOS (normalized)", lw=2)
                            ax2.plot(energies, g_interp, '--', label=f"{best_model} Theoretical (normalized)", lw=2)
                            ax2.legend()
                            ax2.set_xlabel("Energy")
                            ax2.set_ylabel("Normalized DOS")
                            ax2.set_title("Comparison with Best Model")
                            st.pyplot(fig2)

    else:
        st.info("Please upload a CSV file to begin.")
    
page_names_to_funcs = {
    "Intro Page": intro,
    "Notes on DOS": background,
    "Spectrum DOS": dos_spectrum,
    "DOS Calulator": fast_dos,
    "Data Analyser": user_data
    
}

demo_name = st.sidebar.selectbox("Navigator", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()