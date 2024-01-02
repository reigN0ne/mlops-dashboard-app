import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

# Initialize Streamlit app
st.title("MLflow MLOps App")

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Sidebar for selecting runs
st.sidebar.title("Select Run")
print("****************")
print(client.search_experiments())
print("****************")
run_id = st.sidebar.selectbox("Select run:", [run.experiment_id for run in client.search_experiments()])
print(run_id)

# Display run details
run = client.get_run(run_id)
st.subheader(f"Run Details for {run.info.run_id}")
st.write(f"Run Name: {run.data.tags['mlflow.runName']}")
st.write(f"Status: {run.info.status}")
st.write(f"Start Time: {run.info.start_time}")
st.write(f"End Time: {run.info.end_time}")
st.write(f"Artifact URI: {run.info.artifact_uri}")

# Display metrics and parameters
st.subheader("Metrics")
metrics = client.get_metric_history(run.info.run_id)
for metric in metrics:
    st.write(f"{metric.key}: {metric.value}")

st.subheader("Parameters")
params = client.get_run(run.info.run_id).data.params
for param_key, param_value in params.items():
    st.write(f"{param_key}: {param_value}")

# Display model artifacts
st.subheader("Model Artifacts")
artifacts = mlflow.search_runs(f"run_id='{run.info.run_id}'")
for _, artifact in artifacts.iterrows():
    artifact_path = artifact['artifact_uri'] + artifact['path']
    st.write(f"- {artifact_path}")

# Add more sections as needed, such as deployment options, retraining, etc.
