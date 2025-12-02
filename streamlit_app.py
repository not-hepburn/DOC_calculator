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
    

    from Calculator_DOS_lib import (
        numerical_dos_free_particles,
        dos_1d_chain,
        dos_2d_square_lattice,
        dos_1d_phonons,
        WangLandau
    )
    st.write("I am not a lab team. Therefore, the file must only contain different energy values.\\" \
        "To compute the DOS we will use the Wang and Landau algorithm\n 2D Ising model"
            "\nSource:\n 1) https://www.physics.rutgers.edu/grad/509/Wang%20Landau.html \n 2) https://nationalmaglab.org/media/4q5govkf/landau_1.pdf")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

    df_csv = None
    model = st.selectbox("Model", 
                        ["Free electrons", "1D chain", "2D square lattice", "1D phonons"])

    dim = st.selectbox("Dimension", [1,2,3]) if model == "Free electrons" else None

    Nk = st.slider("Number of k-points", 1_000, 30_000_000, 50_000)
    
    if uploaded is not None:
        try:
            df_csv = uploaded
            st.success("CSV loaded successfully!")

            st.write("Preview:")
            st.dataframe(df_csv.head())

            # Validate columns
            if not {"Energy", "Intensity"}.issubset(df_csv.columns):
                st.error("CSV must contain 'Energy' and 'Intensity' columns.")
                df_csv = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
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
            "Flatness",
            0, 1, 
            0.05,
            help="Write something after..."
        )

    with col_right:
        Nitt = st.number_input(
            "Number of Itteration (e8)",
            placeholder="Enter the n-itteration...",
            format= "int"
        )

    st.markdown("---")

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

    analytical = st.checkbox("Plot analytical DOS")
    if analytical and model == "Free electrons":
        if dim == 1:
            dos_anal = (1/np.pi)*np.sqrt(m_e/(2*df_plot["Energy"])) / hbar
        elif dim == 2:
            dos_anal = np.full_like(df_plot["Energy"], m_e/(np.pi*hbar**2))
        elif dim == 3:
            dos_anal = (1/(2*np.pi**2)) * (2*m_e/hbar**2)**1.5 * np.sqrt(df_plot["Energy"])

        fig.add_scatter(x=df_plot["Energy"], y=dos_anal*scale, mode="lines",
                        name="Analytical DOS", line=dict(color="red", dash="dash"))
    pass
    
page_names_to_funcs = {
    "Intro Page": intro,
    "Notes on DOS": background,
    "Spectrum DOS": dos_spectrum,
    "DOS Calulator": fast_dos,
    "Data Analyser": user_data
    
}

demo_name = st.sidebar.selectbox("Navigator", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()