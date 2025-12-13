import pytest
import pandas as pd
from src.evaluation import evaluate_models

# -----------------------
# Fixtures pour les données
# -----------------------

@pytest.fixture
def simple_model_data():
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({
        "RealizedVol": [0.1, 0.2, 0.15],
        "RF_Forecast": [0.11, 0.19, 0.14]
    }, index=dates)
    return [df]  # Toujours mettre dans une liste pour evaluate_models

@pytest.fixture
def multi_model_data():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "RealizedVol": [0.1, 0.2, 0.15, 0.18, 0.12],
        "RF_Forecast": [0.11, 0.19, 0.14, 0.17, 0.13],
        "GARCH_Forecast": [0.09, 0.21, 0.16, 0.19, 0.11]
    }, index=dates)
    return [df]

# -----------------------
# Tests
# -----------------------

def test_evaluate_single_model(simple_model_data):
    metrics_df = evaluate_models(simple_model_data)
    # Vérifier que le modèle RF est présent
    assert "RF" in metrics_df.index
    # Vérifier les colonnes de métriques
    assert all(col in metrics_df.columns for col in ["RMSE", "MAE", "Correlation"])
    # RMSE et MAE doivent être positifs
    assert (metrics_df[["RMSE", "MAE"]] >= 0).all().all()
    # Corrélation doit être entre -1 et 1
    corr = metrics_df.loc["RF", "Correlation"]
    assert -1 <= corr <= 1

def test_evaluate_multiple_models(multi_model_data):
    metrics_df = evaluate_models(multi_model_data)
    # Vérifier que les deux modèles sont présents
    assert set(metrics_df.index) == {"RF", "GARCH"}
    # RMSE doit être trié par ordre croissant
    rmse_values = metrics_df["RMSE"].values
    assert all(rmse_values[i] <= rmse_values[i+1] for i in range(len(rmse_values)-1))

def test_evaluate_empty_list():
    metrics_df = evaluate_models([])
    # Retourne un DataFrame vide
    assert metrics_df.empty

def test_evaluate_missing_forecast():
    df = pd.DataFrame({"RealizedVol": [0.1, 0.2]})
    metrics_df = evaluate_models([df])
    # Pas de modèle à évaluer, DataFrame vide attendu
    assert metrics_df.empty
