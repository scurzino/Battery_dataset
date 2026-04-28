import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

def addestra_xgboost_alta_risoluzione():
    print("⏳ Caricamento del dataset ad alta risoluzione in memoria...")
    df = pd.read_csv('dataset_griglia_brand3.csv')
    
    # 1. Separazione automatica di Input (Feature) e Output (Target)
    # Rimuoviamo il Target e l'ID (l'ID non è una grandezza fisica, confonderebbe il modello)
    X = df.drop(columns=['Target_Capacity_Ah', 'Car_ID'])
    y = df['Target_Capacity_Ah']
    
    print(f"✅ Trovate {len(X.columns)} Feature per ogni ricarica!")
    
    # 2. Dividiamo i dati: 80% Studio, 20% Esame
    print("✂️ Suddivisione dati (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Creazione e Addestramento dell'Algoritmo
    print("🚀 Addestramento XGBoost V2 in corso (sfruttando tutti i core)...")
    # Aumentiamo leggermente il numero di alberi (n_estimators) perché ora 
    # il modello ha molte più sfumature da imparare
    modello = xgb.XGBRegressor(
        n_estimators=350,      
        learning_rate=0.05,    
        max_depth=7,           
        n_jobs=-1,             
        random_state=42,
        missing=np.nan         # Diciamo esplicitamente a XGBoost come gestire le celle vuote
    )
    
    modello.fit(X_train, y_train)
    
    # 4. L'Esame Finale
    print("\n📝 Esecuzione del Test e calcolo dell'errore...")
    previsioni = modello.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, previsioni))
    mae = mean_absolute_error(y_test, previsioni)
    r2 = r2_score(y_test, previsioni)
    
    print("\n" + "="*45)
    print("🎯 RISULTATI DEL MODELLO V2 (GRIGLIA 5%)")
    print("="*45)
    print(f"👉 Errore Medio Assoluto (MAE): ±{mae:.2f} Ah")
    print(f"👉 Errore Quadratico (RMSE): {rmse:.2f} Ah")
    print(f"👉 Precisione Complessiva (R²): {r2*100:.2f}%")
    print("="*45)
    
    # 5. Esportazione
    nome_modello = 'xgboost_soh_nmc_v2.pkl'
    with open(nome_modello, 'wb') as f:
        pickle.dump(modello, f)
    print(f"\n💾 Modello salvato con successo come '{nome_modello}'!")
    
    # 6. Spiegabilità: Quali punti della curva sono più critici?
    print("\n📊 Generazione grafico di Feature Importance...")
    fig, ax = plt.subplots(figsize=(12, 8)) 
    
    # Mostriamo le prime 15 feature più importanti
    xgb.plot_importance(modello, ax=ax, max_num_features=15, importance_type='gain', 
                        title='I punti critici del degrado (Top 15 Feature)',
                        xlabel='Peso Matematico', ylabel='Feature (Punto della Curva)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    addestra_xgboost_alta_risoluzione()