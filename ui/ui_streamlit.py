import streamlit as st
import pandas as pd
import requests
from io import BytesIO


file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        df = None

    st.write(df)


    url = "https://api-customer-charm-1027262266945.europe-west10.run.app" # URL of the FastAPI endpoint
    response = requests.post(f"{url}/predict", json={"data":df.to_json(orient='records')})
    if response.status_code == 200:
        df["answer of predictions"] = pd.Series(response.json()['predictions'])
    else:
        st.error("Error in prediction: " + response.text)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

    with BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        excel_bytes = buffer.getvalue()
        st.download_button(
            label="Download predictions as Excel",
            data=excel_bytes,
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


