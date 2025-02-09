import pandas as pd
import numpy as np
from datetime import datetime
import os

def convert_datetime_features(df):
    try:
        if 'Jour' in df.columns and 'heure' in df.columns:
            df['heure_clean'] = df['heure'].str.split('-').str[0].str.strip()
            df['datetime'] = pd.to_datetime(df['Jour'] + ' ' + df['heure_clean'])
            
            df['jour_semaine'] = df['datetime'].dt.day_name()
            df['heure_journee'] = df['datetime'].dt.hour
            df['semaine_annee'] = df['datetime'].dt.isocalendar().week
            df['mois'] = df['datetime'].dt.month
            df['weekend'] = df['datetime'].dt.weekday.isin([5, 6]).astype(int)
            
            df = df.drop(['Jour', 'heure', 'heure_clean'], axis=1)
            
    except Exception as e:
        print(f"Erreur lors de la conversion des dates: {str(e)}")
    
    return df

def handle_missing_values(df, threshold=0.9):
    null_percentages = df.isnull().sum() / len(df)
    
    columns_to_drop = null_percentages[null_percentages > threshold].index
    df = df.drop(columns=columns_to_drop)
    
    essential_columns = ['Entrées', 'CA Mensuel TTC N', 'Superficie (m²)']
    
    for col in essential_columns:
        if col in df.columns and df[col].isnull().any():
            skewness = df[col].skew()
            
            if abs(skewness) > 1:  # Distribution asymétrique
                df[col] = df[col].fillna(df[col].median())
            else:  # Distribution plus ou moins normale
                df[col] = df[col].fillna(df[col].mean())
    
    return df

def clean_mall_data(df):
    sites_mapping = {
        'Antibes': '53',
        'Bab 2': '38',
        'Bay 2': '37',
        'Cesson': '5',
        'Cholet': '20',
        'Laon': '71',
        'Montesson': '10',
        'Nice Lingostiere': '11',
        'Perpignan Claira': '47',
        'Saint Brieuc': '19'
    }
    
    columns_to_drop = [
        'URL', 'Description SEO', 'Description Actus Shopping',
        'Description Offres Emploi', 'Zone push 1', 'Zone push 2',
        'Minisite', 'Couleur', 'Hyper Key', 'Logo', 'Aperçu logo',
        'Import', 'Statut'
    ]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns, errors='ignore')
    
    if 'Site' in df.columns:
        print("Sites présents:", df['Site'].unique())
        print("Sites manquants:", set(sites_mapping.keys()) - set(df['Site'].unique()))
        # Ajout de l'ID Mall correspondant au site
        df['ID_MALL'] = df['Site'].map(sites_mapping)
        
    if 'Code ensemble immobilier' in df.columns:
        print("\nCodes ensemble immobilier uniques:", df['Code ensemble immobilier'].unique())
    
    df = df.loc[:, ~df.columns.str.contains('_x|_y')]
        
    if 'datetime' in df.columns:
        df['periode_jour'] = pd.cut(
            df['heure_journee'],
            bins=[0, 6, 11, 14, 18, 23],
            labels=['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir']
        )
        
        df['saison'] = pd.cut(
            df['Mois'],
            bins=[0, 3, 6, 9, 12],
            labels=['Hiver', 'Printemps', 'Été', 'Automne']
        )
    
    return df

def save_cleaned_data(df, filename, output_dir='cleaned_data'):
    """
    Sauvegarde les données nettoyées
    """
    # Créer le dossier cleaned_data s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Créer le chemin complet du fichier
    output_path = os.path.join(output_dir, f'clean_{filename}')
    
    # Sauvegarder en CSV
    df.to_csv(output_path, index=False)
    print(f"Données nettoyées sauvegardées dans: {output_path}")
    
    return output_path

def load_and_clean_data(file_path):
    """
    Charge, nettoie et sauvegarde les données
    """
    df = pd.read_excel(file_path)
    print(f"\nTraitement du fichier: {file_path}")
    print("Colonnes présentes:", df.columns.tolist())
    
    df = convert_datetime_features(df)  # Conversion des dates
    df = clean_mall_data(df)           # Nettoyage général
    df = handle_missing_values(df)      # Gestion des valeurs manquantes
    
    output_path = file_path.replace('.xlsx', '_cleaned.csv')
    df.to_csv(f"cleaned_data/clean_{output_path.split('/')[-1]}", index=False)
    print(f"Données nettoyées sauvegardées dans: cleaned_data/clean_{output_path.split('/')[-1]}")
    
    return df