import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def estrai_features_tabellari():
    # 1. Carichiamo la mappa che abbiamo generato con lo script 01
    dict_path = 'all_car_dict_brand3.npy'
    if not os.path.exists(dict_path):
        print("❌ Errore: File di mappatura non trovato. Esegui prima lo script 01.")
        return

    # allow_pickle=True è obbligatorio per caricare dizionari numpy
    all_car_dict = np.load(dict_path, allow_pickle=True).item()
    
    dataset_rows = []
    
    # Contiamo i file totali per la barra di caricamento
    totale_file = sum(len(paths) for paths in all_car_dict.values())
    print(f"🚀 Inizio estrazione features da {totale_file} sessioni di ricarica...")

    # Barra di caricamento unica
    with tqdm(total=totale_file, desc="Estrazione Tabellare", unit="file") as pbar:
        for car_id, file_paths in all_car_dict.items():
            for file_path in file_paths:
                try:
                    # Lettura del file .pkl
                    with open(file_path, 'rb') as f:
                        _ = pickle.load(f) # Hash
                        _ = pickle.load(f) # ID Veicolo
                        _ = pickle.load(f) # Protocollo
                        payload = pickle.load(f) # Dati e Meta
                    
                    telemetria = payload[0]
                    meta = payload[1]
                    
                    # Convertiamo in DataFrame. Indici dedotti in precedenza:
                    # 0: Volt_Avg, 1: Current, 2: SoC, 5: Temp_Max, 7: Time
                    df = pd.DataFrame(telemetria)
                    
                    # Filtro sicurezza: scartiamo ricariche troppo brevi (es. micro-rabbocchi o log interrotti)
                    if len(df) < 10: 
                        pbar.update(1)
                        continue
                        
                    # --- ESTRAZIONE DELLE FEATURE ---
                    v_start = df[0].iloc[0]
                    v_end = df[0].iloc[-1]
                    
                    t_start = df[5].iloc[0]
                    t_max = df[5].max()
                    delta_t = t_max - t_start
                    
                    soc_start = df[2].iloc[0]
                    soc_end = df[2].iloc[-1]
                    delta_soc = soc_end - soc_start
                    
                    durata_sec = df[7].iloc[-1] - df[7].iloc[0]
                    
                    # Dati target dai metadati
                    mileage = meta.get('mileage', 0)
                    capacity_target = meta.get('capacity', np.nan) 
                    
                    # Costruiamo la riga per la nostra tabella
                    riga = {
                        "Car_ID": car_id,
                        "Mileage_km": mileage,
                        "SoC_Start_%": soc_start,
                        "SoC_End_%": soc_end,
                        "Delta_SoC_%": delta_soc,
                        "Volt_Start_V": v_start,
                        "Volt_End_V": v_end,
                        "Temp_Start_C": t_start,
                        "Temp_Max_C": t_max,
                        "Delta_Temp_C": delta_t,
                        "Duration_s": durata_sec,
                        "Target_Capacity_Ah": capacity_target
                    }
                    
                    dataset_rows.append(riga)
                    
                except Exception:
                    pass # Ignoriamo silenziosamente file corrotti per non fermare il loop
                
                finally:
                    pbar.update(1)

    # 3. Creazione e Pulizia del Dataset Finale
    df_finale = pd.DataFrame(dataset_rows)
    
    # Pulizia ingegneristica: rimuoviamo valori infiniti e le righe senza il Target (se non c'è la capacità, non possiamo addestrare)
    df_finale.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_finale.dropna(subset=['Target_Capacity_Ah'], inplace=True) 
    
    # Salvataggio in formato CSV
    file_output = 'dataset_addestramento_brand3.csv'
    df_finale.to_csv(file_output, index=False)
    
    print(f"\n✅ Estrazione completata!")
    print(f"📊 Generato il file '{file_output}' con {len(df_finale)} ricariche valide pronte per XGBoost.")

if __name__ == "__main__":
    estrai_features_tabellari()