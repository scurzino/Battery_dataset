import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

def addestra_xgboost_soh():
    print("⏳ Caricamento del dataset in memoria...")
    df = pd.read_csv('dataset_addestramento_brand3.csv')
    
    # 1. Definiamo quali colonne sono gli Input (Feature) e quale l'Output (Target)
    colonne_input = [
        "Mileage_km", "SoC_Start_%", "SoC_End_%", "Delta_SoC_%",
        "Volt_Start_V", "Volt_End_V", "Temp_Start_C", "Temp_Max_C", 
        "Delta_Temp_C", "Duration_s"
    ]
    colonna_target = "Target_Capacity_Ah"
    
    X = df[colonne_input]
    y = df[colonna_target]
    
    # 2. Dividiamo i dati: 80% Studio, 20% Esame
    print("✂️ Suddivisione dati (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Creazione e Addestramento dell'Algoritmo
    # Usiamo iperparametri classici per la regressione
    print("🚀 Addestramento XGBoost in corso (sfruttando tutti i core della CPU)...")
    modello = xgb.XGBRegressor(
        n_estimators=200,      # Numero di "alberi" decisionali
        learning_rate=0.1,     # Quanto impara velocemente
        max_depth=6,           # Profondità del ragionamento
        n_jobs=-1,             # Usa tutti i thread del tuo processore
        random_state=42
    )
    
    modello.fit(X_train, y_train)
    
    # 4. L'Esame Finale sulle auto sconosciute (X_test)
    print("\n📝 Esecuzione del Test e calcolo dell'errore...")
    previsioni = modello.predict(X_test)
    
    # Calcolo metriche di precisione
    rmse = np.sqrt(mean_squared_error(y_test, previsioni))
    mae = mean_absolute_error(y_test, previsioni)
    r2 = r2_score(y_test, previsioni)
    
    print("\n" + "="*40)
    print("🎯 RISULTATI DEL MODELLO SUL TEST SET")
    print("="*40)
    print(f"👉 Errore Medio Assoluto (MAE): ±{mae:.2f} Ah")
    print(f"👉 Errore Quadratico (RMSE): {rmse:.2f} Ah")
    print(f"👉 Precisione Complessiva (R²): {r2*100:.2f}%")
    print("="*40)
    
    # 5. Esportazione del "Cervello" per la tua piattaforma Web
    nome_modello = 'xgboost_soh_nmc.pkl'
    with open(nome_modello, 'wb') as f:
        pickle.dump(modello, f)
    print(f"\n💾 Modello salvato con successo come '{nome_modello}'! Pronto per il backend.")
    
    # 6. Spiegabilità: Grafico dell'importanza delle Feature
    print("\n📊 Generazione grafico di Feature Importance...")
    # Creiamo esplicitamente la Figura e l'Asse
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # Passiamo l'asse (ax=ax) direttamente a XGBoost
    xgb.plot_importance(modello, ax=ax, max_num_features=10, importance_type='gain', 
                        title='Cosa causa il degrado? (Importanza delle Variabili)',
                        xlabel='Peso Matematico', ylabel='Feature')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    addestra_xgboost_soh()