import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def prepara_o_carica_matrice():
    file_X = 'X_blackbox_brand3.npy'
    file_y = 'y_blackbox_brand3.npy'
    
    # SE I DATI SONO GIA' STATI APPIATTITI IN PASSATO, LI CARICA IN 3 SECONDI
    if os.path.exists(file_X) and os.path.exists(file_y):
        print("⚡ Trovata matrice Flattening salvata! Caricamento fulmineo in corso...")
        X = np.load(file_X)
        y = np.load(file_y)
        return X, y

    # ALTRIMENTI, FA IL LAVORO SPORCO (E POI SALVA I RISULTATI)
    print("⏳ Nessuna matrice trovata. Inizio Flattening profondo (ci vorrà un po')...")
    dict_path = 'all_car_dict_brand3.npy'
    all_car_dict = np.load(dict_path, allow_pickle=True).item()
    
    matrice_X = []
    vettore_y = []
    STEPS = 100 
    
    totale_file = sum(len(paths) for paths in all_car_dict.values())

    with tqdm(total=totale_file, desc="Flattening", unit="file") as pbar:
        for car_id, file_paths in all_car_dict.items():
            for file_path in file_paths:
                try:
                    with open(file_path, 'rb') as f:
                        _ = pickle.load(f); _ = pickle.load(f); _ = pickle.load(f)
                        payload = pickle.load(f)
                    
                    telemetria = payload[0]
                    meta = payload[1]
                    capacity = meta.get('capacity', np.nan)
                    
                    if np.isnan(capacity) or capacity == 0:
                        continue
                        
                    time_raw = telemetria[:, 7]
                    volt_raw = telemetria[:, 0]
                    curr_raw = telemetria[:, 1]
                    temp_raw = telemetria[:, 5]
                    
                    if len(time_raw) < 10:
                        continue
                        
                    _, unique_indices = np.unique(time_raw, return_index=True)
                    time_raw = time_raw[unique_indices]
                    volt_raw = volt_raw[unique_indices]
                    curr_raw = curr_raw[unique_indices]
                    temp_raw = temp_raw[unique_indices]
                    
                    time_norm = (time_raw - time_raw[0]) / (time_raw[-1] - time_raw[0])
                    time_grid = np.linspace(0, 1, STEPS)
                    
                    f_volt = interp1d(time_norm, volt_raw, kind='linear', fill_value="extrapolate")
                    f_curr = interp1d(time_norm, curr_raw, kind='linear', fill_value="extrapolate")
                    f_temp = interp1d(time_norm, temp_raw, kind='linear', fill_value="extrapolate")
                    
                    volt_100 = f_volt(time_grid)
                    curr_100 = f_curr(time_grid)
                    temp_100 = f_temp(time_grid)
                    
                    riga_piatta = np.concatenate([volt_100, curr_100, temp_100, [meta.get('mileage', 0)]])
                    
                    matrice_X.append(riga_piatta)
                    vettore_y.append(capacity)
                    
                except Exception:
                    pass
                finally:
                    pbar.update(1)

    X = np.array(matrice_X)
    y = np.array(vettore_y)
    
    # ECCO IL SALVATAGGIO CHE TI CAMBIERÀ LA VITA!
    print("\n💾 Salvataggio della matrice estratta su disco...")
    np.save(file_X, X)
    np.save(file_y, y)
    print("✅ File 'X_blackbox_brand3.npy' e 'y_blackbox_brand3.npy' salvati! Mai più attese infinite.")
    
    return X, y

def addestra_metodo_accademico():
    X, y = prepara_o_carica_matrice()
    
    print("\n✂️ Suddivisione dati (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🚀 Addestramento XGBoost 'Black Box' in corso (durerà pochissimo)...")
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
    print(f"👉 Precisione Complessiva (R²): {r2*100:.2f}%")
    print("="*45)

    # SALVATAGGIO DEL MODELLO
    nome_modello = 'xgboost_blackbox.pkl'
    with open(nome_modello, 'wb') as f:
        pickle.dump(modello, f)
    print(f"\n💾 CERVELLO AI SALVATO: Il file '{nome_modello}' è pronto per il tuo sito web!")

if __name__ == "__main__":
    addestra_metodo_accademico()