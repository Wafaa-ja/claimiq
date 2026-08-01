import pickle
from pathlib import Path

model_paths = [
    Path("models/poisson_model.pkl"),
    Path("models/negative_binomial_model.pkl"),
]

for path in model_paths:
    with path.open("rb") as file:
        model = pickle.load(file)

    # Statsmodels stores the entire training dataset unless removed.
    if hasattr(model, "remove_data"):
        model.remove_data()
    elif hasattr(model, "_results") and hasattr(model._results, "remove_data"):
        model._results.remove_data()

    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{path}: {path.stat().st_size / 1024 / 1024:.2f} MB")