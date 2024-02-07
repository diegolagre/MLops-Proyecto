# %%
import mlflow
from baseline_model import MLF_RS_baseline_usr_mov

# %%
mlflow.set_experiment('RS_baselines')
mlflow.start_run()
# %%
model_checkpoint = "baseline_p_0.5.pkl"
mlflow.log_param('model_checkpoint', model_checkpoint)
# %%
model = MLF_RS_baseline_usr_mov(model_checkpoint)
# %%
# %%
# %%
mlflow.pyfunc.log_model(
    artifact_path='model',
    python_model=model,
    registered_model_name='MLF_RS_baseline_usr_mov',
    code_path=['model_src'],
    #conda_env='conda.yaml',
)
# %%
mlflow.end_run()
# %% Test model

model.load_context(None)
# %%
import numpy as np

X = np.array([[4,5],[2,4]])
X

model.predict(None, X)
# %%
