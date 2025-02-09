import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from darts import TimeSeries
from darts.models import (
    XGBModel,
)
from darts.metrics import rmse, r2_score, mape, mae

from prophet import Prophet  # Librairie officielle Facebook Prophet


def define_features(df):
    """
    Définit les features numériques et catégorielles pour le modèle
    """
    numeric_features = [
        'heure_journee', 
        'Mois', 
        'Année',
        'Superficie (m²)',
        'CA Mensuel TTC N'
    ]
    
    categorical_features = [
        'Site',
        'Zone',
        'jour_semaine',
        'periode_jour',
        'saison',
        'Famille enseigne',
        'Galerie/Pac/Hyper'
    ]
    
    return numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    """Version corrigée qui ignore les colonnes non features"""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ), categorical_features)
        ],
        remainder='drop',  # Supprime toutes les colonnes non listées
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

# def evaluate_regression_model(model, X, y, preprocessor, cv_splits=5):
#     """
#     Évalue le modèle de régression avec validation croisée temporelle
#     """
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
    
#     pipeline = make_pipeline(preprocessor, model)
    
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     cv_scores = {
#         'rmse': [],
#         'r2': []
#     }
    
#     for train_idx, val_idx in tscv.split(X_train):
#         X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
#         pipeline.fit(X_train_cv, y_train_cv)
#         y_pred_cv = pipeline.predict(X_val_cv)
        
#         # Utilisation des métriques sklearn au lieu de darts
#         cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))
#         cv_scores['r2'].append(sklearn.metrics.r2_score(y_val_cv, y_pred_cv))  # Utilisation explicite de sklearn.metrics
    
#     pipeline.fit(X_train, y_train)
#     y_pred_test = pipeline.predict(X_test)
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
#     test_r2 = sklearn.metrics.r2_score(y_test, y_pred_test)
    
#     return {
#         'cv_rmse_mean': np.mean(cv_scores['rmse']),
#         'cv_rmse_std': np.std(cv_scores['rmse']),
#         'cv_r2_mean': np.mean(cv_scores['r2']),
#         'cv_r2_std': np.std(cv_scores['r2']),
#         'test_rmse': test_rmse,
#         'test_r2': test_r2,
#         'model': pipeline
#     }

# def first_baseline(X, y, preprocessor):
#     print("\nDébut de l'évaluation de la baseline")
#     print(f"Nombre d'observations: {len(X)}")
    
#     # 1. Création du modèle simple
#     model = LinearRegression()
    
#     # 2. Évaluation du modèle
#     results = evaluate_regression_model(
#         model=model,
#         X=X,
#         y=y,
#         preprocessor=preprocessor,
#         cv_splits=3
#     )
    
#     # 3. Affichage détaillé des métriques
#     print("\n=== Résultats de la baseline (Régression Linéaire) ===")
#     print(f"CV RMSE: {results['cv_rmse_mean']:.2f} (±{results['cv_rmse_std']:.2f})")
#     print(f"CV R2: {results['cv_r2_mean']:.2f} (±{results['cv_r2_std']:.2f})")
#     print(f"Test RMSE: {results['test_rmse']:.2f}")
#     print(f"Test R2: {results['test_r2']:.2f}")
    
#     return results

# def run_first_baseline():
#     df = pd.read_csv("cleaned_data/merged_data.csv")
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df = df.sort_values('datetime')
        
#     # Features étendues pour la baseline
#     selected_numeric = [
#         'heure_journee',
#         'Mois',
#         'Année',
#         'Superficie (m²)',
#         'CA Mensuel TTC N'
#     ]
    
#     selected_categorical = [
#         'Site',
#         'Zone',
#         'jour_semaine',
#         'periode_jour',
#         'saison',
#         'Famille enseigne',
#         'Galerie/Pac/Hyper'
#     ]
    
#     X = df[selected_numeric + selected_categorical]
#     y = df['Entrées']
    
#     print("\nFeatures utilisées:")
#     print(f"Numériques: {selected_numeric}")
#     print(f"Catégorielles: {selected_categorical}")
    
#     # 3. Création du preprocessor et exécution
#     preprocessor = create_preprocessor(selected_numeric, selected_categorical)
#     results = first_baseline(X, y, preprocessor)
    
#     return results


# def create_darts_models():
#     """
#     Crée une liste de modèles Darts optimisés
#     """
#     return [
#         ('Prophet', Prophet(
#             seasonality_mode='additive',
#             yearly_seasonality=5,
#             weekly_seasonality=True,
#             daily_seasonality=False,
#             changepoint_prior_scale=0.001,
#             # holidays=holidays_df
#         ))
#     ]

# def evaluate_time_series_models(X, y, preprocessor, cv_splits=5):
#     test_size = 24
#     tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)
#     results = {}
#     models = create_darts_models()
    
#     for name, model in models:
#         print(f"\nÉvaluation du modèle: {name}")
#         cv_scores = {'rmse': [], 'r2': []}
        
#         for train_idx, val_idx in tscv.split(X):
#             try:
#                 val_idx_extended = list(val_idx) + list(range(val_idx[-1] + 1, val_idx[-1] + 25))
#                 val_idx_extended = [i for i in val_idx_extended if i < len(X)]
                
#                 X_train, X_val = X.iloc[train_idx], X.iloc[val_idx_extended]
#                 y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
#                 train_series = TimeSeries.from_dataframe(
#                     pd.DataFrame({'Entrées': y_train}),
#                     freq='h'
#                 )
#                 val_series = TimeSeries.from_dataframe(
#                     pd.DataFrame({'Entrées': y_val}),
#                     freq='h'
#                 )
                
#                 pipeline = make_pipeline(preprocessor)
#                 X_train_transformed = pipeline.fit_transform(X_train)
#                 X_val_transformed = pipeline.transform(X_val)
                
#                 train_covariates = TimeSeries.from_dataframe(
#                     pd.DataFrame(X_train_transformed, index=X_train.index),
#                     freq='h'
#                 )
#                 val_covariates = TimeSeries.from_dataframe(
#                     pd.DataFrame(X_val_transformed, index=X_val.index),
#                     freq='h'
#                 )
                
#                 if name == 'Prophet':
#                     model.fit(train_series)
#                     pred_series = model.predict(len(val_series))
#                 else:
#                     model.fit(train_series, future_covariates=train_covariates)
#                     pred_series = model.predict(
#                         n=test_size,
#                         future_covariates=val_covariates
#                     )
                
#                 cv_scores['rmse'].append(rmse(val_series, pred_series))
#                 cv_scores['r2'].append(r2_score(val_series, pred_series))
                
#             except Exception as e:
#                 print(f"Erreur avec {name} sur un fold: {str(e)}")
#                 continue
        
#         if cv_scores['rmse']:
#             results[name] = {
#                 'rmse_mean': np.mean(cv_scores['rmse']),
#                 'rmse_std': np.std(cv_scores['rmse']),
#                 'r2_mean': np.mean(cv_scores['r2']),
#                 'r2_std': np.std(cv_scores['r2'])
#             }
            
#             print(f"RMSE: {results[name]['rmse_mean']:.2f} (±{results[name]['rmse_std']:.2f})")
#             print(f"R2: {results[name]['r2_mean']:.2f} (±{results[name]['r2_std']:.2f})")
    
#     return results

# def run_model_comparison():
#     df = pd.read_csv("cleaned_data/merged_data.csv")
#     df['datetime'] = pd.to_datetime(df['datetime'])
    
#     print(f"\nAnalyse des doublons:")
#     print(df.groupby('datetime').size().describe())
    
#     df = df.groupby('datetime').agg({
#         'Entrées': 'sum',
#         'CA Mensuel TTC N': 'mean',
#         'Superficie (m²)': 'mean',
#         'Site': 'first',
#         'Zone': 'first',
#         'jour_semaine': 'first',
#         'periode_jour': 'first',
#         'saison': 'first',
#         'Famille enseigne': lambda x: x.mode()[0],
#         'Galerie/Pac/Hyper': 'first',
#         'heure_journee': 'first',
#         'Mois': 'first',
#         'Année': 'first'
#     }).reset_index()
    
#     df = df.sort_values('datetime').set_index('datetime')
    
#     print("\nVérification des trous dans les données:")
#     print(df.index.to_series().diff().value_counts().head())
    
#     numeric_features, categorical_features = define_features(df)
#     X = df[numeric_features + categorical_features]
#     y = df['Entrées']
    
#     preprocessor = create_preprocessor(numeric_features, categorical_features)
#     results = evaluate_time_series_models(X, y, preprocessor)
    
#     return results

def load_exception_dates(closure_path, opening_path):
    """Version corrigée pour Excel"""
    try:
        # Lire les fichiers Excel sans paramètre encoding
        closures = pd.read_excel(closure_path, engine='openpyxl')
        openings = pd.read_excel(opening_path, engine='openpyxl')
        
    except UnicodeDecodeError:
        # Solution alternative pour les encodages spéciaux
        closures = pd.read_excel(
            closure_path, 
            engine='openpyxl',
            encoding_override='latin1'  # Paramètre correct pour Excel
        )
        openings = pd.read_excel(
            opening_path,
            engine='openpyxl',
            encoding_override='latin1'
        )
    
    # 2. Nettoyage des noms de colonnes
    closures.columns = closures.columns.str.strip().str.lower()
    openings.columns = openings.columns.str.strip().str.lower()

    # 3. Traitement des dates
    closure_dates = []
    for _, row in closures.iterrows():
        mall_id = row['mall id']
        for col in [c for c in row.index if c not in ['centre', 'mall id', 'clef hyper carrefour']]:
            date_str = str(row[col]).strip()
            if date_str not in ['nan', 'NaT', '']:
                try:
                    # Gestion du format jour/mois/année
                    date = pd.to_datetime(date_str, dayfirst=True)
                    closure_dates.append((mall_id, date))
                except:
                    continue

    # 4. Traitement des ouvertures exceptionnelles
    opening_data = []
    for _, row in openings.iterrows():
        mall_id = row['mall id']
        for col in [c for c in row.index if ':' in c]:
            try:
                date_part = col.split()[-1]
                date = pd.to_datetime(date_part, dayfirst=True)
                opening_data.append({
                    'mall id': mall_id,
                    'date': date,
                    'heures': row[col]
                })
            except:
                continue

    return (
        pd.DataFrame(closure_dates, columns=['mall id', 'date']),
        pd.DataFrame(opening_data)
    )

def adjust_predictions_with_exceptions(predictions, mall_id, closure_df, opening_df):
    mall_closures = closure_df[closure_df['mall id'] == mall_id]['date']
    predictions.loc[predictions['ds'].isin(mall_closures), 'yhat'] = 0
    
    mall_openings = opening_df[opening_df['mall id'] == mall_id]
    for _, row in mall_openings.iterrows():
        mask = predictions['ds'] == row['date']
        if mask.any():
            opening_hours = row['heures']
            if '20:00' in opening_hours:
                predictions.loc[mask, 'yhat'] *= 1.3  # +30% pour ouvertures tardives
            elif '17:00' in opening_hours:
                predictions.loc[mask, 'yhat'] *= 0.7  # -30% pour fermetures anticipées
    
    return predictions

def preprocess_data(df, preprocessor):
    """Prétraitement sécurisé avec conservation des métadonnées"""
    # Copie des métadonnées critiques
    metadata = df[['ID mall', 'datetime', 'Entrées']].copy()
    
    # Suppression des colonnes non features
    df = df.drop(columns=['ID mall', 'datetime', 'Entrées'])
    
    # Application du préprocesseur
    processed = preprocessor.fit_transform(df)
    
    # Réintégration des métadonnées
    return pd.concat([processed, metadata], axis=1)

def create_model(preprocessor):
    """Version améliorée avec paramètres optimisés"""
    return Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=10,  # Augmenté pour capturer plus de saisonnalité
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # Meilleure détection des changements
    )

def evaluate_model(forecast, test_data):
    """Version finale validée avec gestion des divisions par zéro"""
    merged = forecast.set_index('ds')[['yhat']].join(test_data.set_index('datetime'))
    merged = merged[merged['Entrées'] > 0]  # Éviter les divisions par zéro
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(merged['yhat'], merged['Entrées'])),
        'MAE': mean_absolute_error(merged['yhat'], merged['Entrées']),
    }
    
    # Calcul MAPE uniquement si des données valides existent
    if len(merged) > 0:
        metrics['MAPE'] = np.mean(np.abs((merged['Entrées'] - merged['yhat'])/merged['Entrées'])) * 100
    else:
        metrics['MAPE'] = np.nan
    
    return metrics

def load_data():
    df = pd.read_csv("cleaned_data/merged_data.csv")
    
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    
    # Validation des colonnes critiques
    required_cols = ['ID mall', 'datetime', 'Entrées']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes: {missing}")
    
    return df

def visualize_all_predictions(df):
    numeric = ['CA Mensuel TTC N', 'Superficie (m²)', 'heure_journee']
    categorical = ['jour_semaine', 'saison', 'Galerie/Pac/Hyper']
    
    preprocessor = create_preprocessor(numeric, categorical)
    processed_df = preprocess_data(df, preprocessor)

    results = {}
    for mall_id in processed_df['ID mall'].unique():
        mall_data = processed_df[processed_df['ID mall'] == mall_id]
        
        try:
            train_df = mall_data.rename(columns={
                'datetime': 'ds',
                'Entrées': 'y'
            }).drop(columns=['ID mall'])
            
            model = create_model(preprocessor)
            model.fit(train_df)
            
            future = model.make_future_dataframe(periods=7, freq='D')
            for col in preprocessor.get_feature_names_out():
                future[col] = mall_data[col].iloc[-1]
                
            forecast = model.predict(future)
            results[mall_id] = forecast[['ds', 'yhat']].tail(7)
            
        except Exception as e:
            print(f"Erreur mall {mall_id}: {str(e)}")
    
    return results

def display_predictions(forecast, test_data):
    forecast['ds'] = pd.to_datetime(forecast['ds'], unit='ns')
    
    preds = forecast.tail(7)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prédictions'})
    preds['Date'] = preds['Date'].dt.strftime('%Y-%m-%d')
    preds['Prédictions'] = preds['Prédictions'].round().astype(int)
    
    print(f"\nPrédictions pour les 7 prochains jours :")
    print(preds.to_string(index=False))


if __name__ == "__main__":
    df = pd.read_csv("cleaned_data/merged_data.csv")
    print("Colonnes originales :", df.columns.tolist())
    results = visualize_all_predictions(df)
    
    for mall_id, preds in results.items():
        print(f"\nPrédictions pour le mall {mall_id}:")
        print(preds.round().astype(int))

    print(df['datetime'].dtype)d