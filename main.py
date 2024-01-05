import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import json

st.title("MLOps Dashboard App")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

st.sidebar.title("All your experiments will appear here")
experiment_id = st.sidebar.selectbox("Select Experiment:", [run.experiment_id for run in client.search_experiments()])

runs = mlflow.search_runs([experiment_id])
params_col = [col.split(".")[-1] for col in runs.columns if col.startswith("params")]

for run_id in range(len(runs)):
    with st.expander(f"Run Details for Tracker #{runs['run_id'][run_id]}", expanded=False):
        st.subheader(f"Run Details")
        
        col_1, col_2 = st.columns(2)
        
        with col_1:
            st.write(f"Run Name: {runs['tags.mlflow.runName'][run_id]}")
            st.write(f"Status: {runs['status'][run_id]}")
        
        with col_2:
            st.write(f"Start Time: {runs['start_time'][run_id]}")
            st.write(f"End Time: {runs['end_time'][run_id]}")

        st.subheader("Track Parameters")
        col_3, col_4 = st.columns(2)
        for idx, param in enumerate(params_col):
            if idx % 2 == 0:
                with col_3:
                    st.write(f"{param}: {runs['params.' + param][run_id]}")
            else:
                with col_4:
                    st.write(f"{param}: {runs['params.' + param][run_id]}")
            
        st.subheader("Model Artifacts")
        artifacts_data = json.loads(runs['tags.mlflow.log-model.history'][run_id])
        artifacts_path = artifacts_data[0]['artifact_path']
        model_path = artifacts_data[0]['flavors']['python_function']['model_path']
        
        st.write(f"Artifact Path: {runs['artifact_uri'][run_id]}" + "/" + artifacts_path)
        st.write("Model Name: " + model_path)
