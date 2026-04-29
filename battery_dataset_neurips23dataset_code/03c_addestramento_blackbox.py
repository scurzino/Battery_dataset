import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings

# Ignoriamo i warning per i calcoli dei NaN per non sporcare il terminale
warnings.filterwarnings('ignore')

def addestra_metodo_accademico():
    dict_path = 'all_car_dict_brand3.npy'
    all_car_dict = np.load(dict_path, allow_pickle=True).item()
    
    matrice_X = []
    vettore_y = []
    
    # 1. Normalizziamo TUTTE le ricariche a esattamente 100 step temporali
    # Questo è il trucco (PreprocessNormalizer) usato dai ricercatori
    STEPS = 100 
    
    totale_file = sum(len(paths) for paths in all_car_dict.values())
    print(f"🚀 Creazione della Matrice 'Black Box' su {totale_file} file...")

    with tqdm(total=totale_file, desc="Flattening", unit="file") as pbar:
        for car_id, file_paths in all_car_dict.items():
            for file_path in file_paths:
                try:
                    with open(file_path, 'rb') as f:
                        _ = pickle.load(f)
                        _ = pickle.load(f)
                        _ = pickle.load(f)
                        payload = pickle.load(f)
                    
                    telemetria = payload[0]
                    meta = payload[1]
                    
                    capacity = meta.get('capacity', np.nan)
                    if np.isnan(capacity) or capacity == 0:
                        continue
                        
                    # 0: Volt, 1: Current, 5: Temp, 7: Time
                    time_raw = telemetria[:, 7]
                    volt_raw = telemetria[:, 0]
                    curr_raw = telemetria[:, 1]
                    temp_raw = telemetria[:, 5]
                    
                    if len(time_raw) < 10:
                        continue
                        
                    # Rimuoviamo i duplicati temporali
                    _, unique_indices = np.unique(time_raw, return_index=True)
                    time_raw = time_raw[unique_indices]
                    volt_raw = volt_raw[unique_indices]
                    curr_raw = curr_raw[unique_indices]
                    temp_raw = temp_raw[unique_indices]
                    
                    # Creiamo un asse del tempo normalizzato da 0.0 a 1.0
                    time_norm = (time_raw - time_raw[0]) / (time_raw[-1] - time_raw[0])
                    
                    # Intercette standard fisse
                    time_grid = np.linspace(0, 1, STEPS)
                    
                    # Interpoliamo le tre curve
                    f_volt = interp1d(time_norm, volt_raw, kind='linear', fill_value="extrapolate")
                    f_curr = interp1d(time_norm, curr_raw, kind='linear', fill_value="extrapolate")
                    f_temp = interp1d(time_norm, temp_raw, kind='linear', fill_value="extrapolate")
                    
                    volt_100 = f_volt(time_grid)
                    curr_100 = f_curr(time_grid)
                    temp_100 = f_temp(time_grid)
                    
                    # IL FLATTENING: Uniamo tutto in un'unica riga da 300 numeri + Mileage
                    riga_piatta = np.concatenate([volt_100, curr_100, temp_100, [meta.get('mileage', 0)]])
                    
                    matrice_X.append(riga_piatta)
                    vettore_y.append(capacity)
                    
                except Exception:
                    pass
                finally:
                    pbar.update(1)

    X = np.array(matrice_X)
    y = np.array(vettore_y)
    
    print(f"\n✅ Matrice completata: {X.shape[0]} ricariche, {X.shape[1]} colonne (Feature incomprensibili all'uomo).")
    print("✂️ Suddivisione dati (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🚀 Addestramento XGBoost 'Black Box' in corso...")
    modello = xgb.XGBRegressor(
        n_estimators=300,      
        learning_rate=0.05,    
        max_depth=6,           
        n_jobs=-1,             
        random_state=42
    )
    modello.fit(X_train, y_train)
    
    previsioni = modello.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, previsioni))
    mae = mean_absolute_error(y_test, previsioni)
    r2 = r2_score(y_test, previsioni)
    
    print("\n" + "="*45)
    print("🎯 RISULTATI METODO ACCADEMICO (FLATTENING)")
    print("="*45)
    print(f"👉 Errore Medio Assoluto (MAE): ±{mae:.2f} Ah")
    print(f"👉 Errore Quadratico (RMSE): {rmse:.2f} Ah")
    print(f"👉 Precisione Complessiva (R²): {r2*100:.2f}%")
    print("="*45)

# Esportazione del Modello Black Box
    import pickle
    with open('xgboost_blackbox.pkl', 'wb') as f:
        pickle.dump(modello, f)
    print("💾 Modello salvato come 'xgboost_blackbox.pkl'")

if __name__ == "__main__":
    addestra_metodo_accademico()