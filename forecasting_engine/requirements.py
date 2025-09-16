import importlib
import sys

modules = [
    "pandas",
    "numpy",
    "sklearn",      # PyPI package name is scikit-learn
    "lightgbm",
    "xgboost",
    "catboost",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "dotenv",       # PyPI package name is python-dotenv
    "supabase",      # PyPI package name is supabase
    "requests",
    "darts",
    "mlflow"
]

package_name_map = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv"
}

requirements = []

print(f"Python version: {sys.version}\n")

for module in modules:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", None)
        pkg_name = package_name_map.get(module, module)

        if version:
            print(f"{module}: {version}")
            requirements.append(f"{pkg_name}=={version}")
        else:
            print(f"{module}: Installed (no __version__ attribute)")
            requirements.append(pkg_name)
    except ImportError:
        print(f"{module}: Not installed")

# Write requirements.txt
with open("forecasting_engine/requirements.txt", "w") as f:
    f.write("\n".join(requirements))

print("\nrequirements.txt has been created.")
