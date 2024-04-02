"""app.py"""

from io import StringIO

import pandas as pd
import streamlit as st

st.title("Streamlit Example App")

MULTI_FILE = True

uploaded_file = st.file_uploader(
    "Pick a CSV to analyze", type="csv", accept_multiple_files=MULTI_FILE
)

st.metric("Number of files", len(uploaded_file))

if uploaded_file != []:
    for file in uploaded_file:
        # To read file as bytes:
        bytes_data = file.getvalue()
        name = file.name
        st.header(name)
        tab1, tab2, tab3 = st.tabs(["Info", "StringIO", "Dataframe"])
        dataframe = pd.read_csv(file)

        with tab1:
            st.metric("No of Cols", len(dataframe.columns))
            st.metric("No of Rows", len(dataframe))
            st.subheader("Table")
            st.table(dataframe)

        with tab2:
            # To read file as string:
            stringio = StringIO(file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            st.write(string_data)

        with tab3:
            # Can be used wherever a "file-like" object is accepted:
            st.subheader("Dataframe")
            st.write(dataframe.columns)
            st.dataframe(dataframe)
