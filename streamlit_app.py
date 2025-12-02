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

    st.write("# Phys 456: Project")
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
    

    from Calculator_DOS_lib import (dos_from_user_data, numerical_dos_free_particles,
        dos_1d_chain,
        dos_2d_square_lattice,
        dos_1d_phonons)

    st.write("""Uplaod a CSV containing **Energy** and **Intensity** (optional) columns."""
            "lol")
    with st.container(border=True):
        st.subheader("Upload CSV")
        uploaded = st.file_uploader("Chose your file", type=["csv"])
        df_csv = None
        headers_col = None

    if uploaded:
        try:
            # Read only header row
            headers = uploaded.getvalue().decode("utf-8").split("\n")[0]
            headers_col = [h.strip() for h in headers.split(",")]
            col_left, col_right = st.columns([0.65, 0.35], vertical_alignment='center', gap="medium", border=True)
            with col_left:
                st.subheader("Detected Header")
                st.dataframe(pd.DataFrame([headers_col], columns=headers_col))
            with col_right:
                st.subheader("Validation")
                confirm = st.radio(
                    "Does this look correct?",
                    ["Yes", "No"],
                    horizontal=True,
                )

            if confirm == "Yes":
                df_csv = pd.read_csv(uploaded)

                st.success("CSV loaded successfully!")
                st.write("Preview:")
                st.dataframe(df_csv.head(), use_container_width=True)

            else:
                st.warning("Please upload a corrected CSV file.")

        except Exception as e:
            st.error(f"Error reading file: {e}")
        
    st.markdown("---")        
    st.subheader("Configuration")
    col_left, col_right = st.columns([1, 0.50], gap="large", border=True)

    with col_left:
        st.subheader("DOS Resolution")
        bins = st.pills(label="Number of bins (resolution)", options=[
            100, 200, 300, 400, 500], selection_mode="single" )
    if df_csv is not None:
        with col_right:

            column_choice = st.selectbox(
                "Select energy column:",
                df_csv.columns.tolist()
            )
    
            
    E1, g1 = numerical_dos_free_particles(1, num_k_points=200_000)
    E2, g2 = numerical_dos_free_particles(2, num_k_points=200_000)
    E3, g3 = numerical_dos_free_particles(3, num_k_points=200_000)

    st.markdown("---")

    if st.button("Compute DOS") and df_csv is not None:
        E_exp = df_csv[column_choice].values

        with st.spinner("Fetching the corresponding bits... be right back"):

            # ---- Compute model ----
            energies, hist = dos_from_user_data(E= E_exp, bins=bins)
            dos = hist
            st.caption("Density of States")
            df_plot = pd.DataFrame({f"Energy": energies,
                                    "DOS": dos}).sort_values("Energy")
            st.line_chart(df_plot, x=f"Energy", y="DOS")
            g1_interp = np.interp(E_exp, E1, g1)
            g2_interp = np.interp(E_exp, E2, g2)
            g3_interp = np.interp(E_exp, E3, g3)

            err1 = np.mean((df_csv["Intensity"] - g1_interp)**2)
            err2 = np.mean((df_csv["Intensity"] - g2_interp)**2)
            err3 = np.mean((df_csv["Intensity"] - g3_interp)**2)

            best = np.argmin([err1, err2, err3])
            models = ["1D Free Electrons", "2D Free Electrons", "3D Free Electrons"]

            best_model = models[best]
            
            st.success(f"DOS computed successfully for {len(E_exp)} energy values.\n \n Best matching model: **{best_model}**")
            
            with st.container(border=True):
                st.write("### Data summary")
                st.metric("Entries", len(E_exp))
                st.metric("Min Energy", f"{E_exp.min():.3f}")
                st.metric("Max Energy", f"{E_exp.max():.3f}")
                st.metric(f"Error on {best_model} model: ", f"{best:.2f}")

            csv = df_plot.to_csv(index=False).encode("utf-8")
            st.download_button("Download DOS CSV", csv, "DOS_output.csv", "text/csv")
        
    
page_names_to_funcs = {
    "Intro Page": intro,
    "Notes on DOS": background,
    "Spectrum DOS": dos_spectrum,
    "DOS Calulator": fast_dos,
    "Data Analyser": user_data
    
}

demo_name = st.sidebar.selectbox("Navigator", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()