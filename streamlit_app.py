import streamlit as st

st.title("DOS Calculator")

model = st.selectbox("Model", ["Free electron", "1D Chain", "2D square Lattice", "1D phonons"])
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
