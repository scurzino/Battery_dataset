import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

def estrai_features_griglia_soc():
    dict_path = 'all_car_dict_brand3.npy'
    if not os.path.exists(dict_path):
        print("❌ Errore: File di mappatura non trovato.")
        return

    all_car_dict = np.load(dict_path, allow_pickle=True).item()
    dataset_rows = []
    
    # Definiamo la nostra "Griglia" di analisi (0%, 5%, 10% ... 100%)
    soc_grid = np.arange(0, 105, 5)
    
    totale_file = sum(len(paths) for paths in all_car_dict.values())
    print(f"🚀 Inizio estrazione ad alta risoluzione (Griglia SoC 5%) su {totale_file} file...")

    with tqdm(total=totale_file, desc="Estrazione Griglia", unit="file") as pbar:
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
                    
                    # Indici: 0: Volt_Avg, 1: Current, 2: SoC, 5: Temp_Max, 7: Time
                    df = pd.DataFrame(telemetria)
                    
                    # Pulizia essenziale per l'interpolazione matematica
                    # Rimuoviamo eventuali doppioni di SoC e ordiniamo in modo crescente
                    df = df.sort_values(by=2).drop_duplicates(subset=2)
                    
                    if len(df) < 10: 
                        continue
                        
                    soc_reale = df[2].values
                    
                    # Inizializziamo la riga con le Feature Globali (sempre utilissime)
                    riga = {
                        "Car_ID": car_id,
                        "Mileage_km": meta.get('mileage', 0),
                        "SoC_Start_%": soc_reale[0],
                        "SoC_End_%": soc_reale[-1],
                        "Delta_SoC_%": soc_reale[-1] - soc_reale[0],
                        "Temp_Max_C": df[5].max(),
                        "Duration_s": df[7].iloc[-1] - df[7].iloc[0],
                        "Target_Capacity_Ah": meta.get('capacity', np.nan)
                    }
                    
                    # --- LA MAGIA DELL'INTERPOLAZIONE ---
                    # Creiamo delle funzioni matematiche che tracciano le curve reali
                    # bounds_error=False e fill_value=np.nan faranno in modo che se la ricarica 
                    # è avvenuta tra il 40% e l'80%, le colonne <40 e >80 resteranno felicemente vuote (NaN)
                    
                    f_volt = interp1d(soc_reale, df[0].values, bounds_error=False, fill_value=np.nan)
                    f_curr = interp1d(soc_reale, df[1].values, bounds_error=False, fill_value=np.nan)
                    f_temp = interp1d(soc_reale, df[5].values, bounds_error=False, fill_value=np.nan)
                    
                    # Calcoliamo i valori esatti sulla nostra griglia del 5%
                    volt_grid = f_volt(soc_grid)
                    curr_grid = f_curr(soc_grid)
                    temp_grid = f_temp(soc_grid)
                    
                    # Aggiungiamo tutte queste nuove misurazioni alla nostra riga del CSV
                    for i, soc_val in enumerate(soc_grid):
                        riga[f"Volt_a_{soc_val}%_V"] = volt_grid[i]
                        riga[f"Curr_a_{soc_val}%_A"] = curr_grid[i]
                        riga[f"Temp_a_{soc_val}%_C"] = temp_grid[i]
                    
                    dataset_rows.append(riga)
                    
                except Exception:
                    pass 
                finally:
                    pbar.update(1)

    df_finale = pd.DataFrame(dataset_rows)
    df_finale.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_finale.dropna(subset=['Target_Capacity_Ah'], inplace=True) 
    
    file_output = 'dataset_griglia_brand3.csv'
    df_finale.to_csv(file_output, index=False)
    
    print(f"\n✅ Fatto! Generato '{file_output}'.")
    print(f"📊 Il nuovo dataset ha {len(df_finale.columns)} colonne (Features) per ogni ricarica!")

if __name__ == "__main__":
    estrai_features_griglia_soc()