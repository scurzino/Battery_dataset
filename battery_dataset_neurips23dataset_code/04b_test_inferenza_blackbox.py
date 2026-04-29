import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
import warnings

# Ignoriamo i warning per non sporcare il terminale
warnings.filterwarnings('ignore')

def simula_backend_piattaforma():
    # 1. CARICAMENTO DEL MOTORE AI
    nome_modello = 'xgboost_blackbox.pkl'
    try:
        with open(nome_modello, 'rb') as f:
            modello = pickle.load(f)
        print("✅ Motore AI (XGBoost Black Box) caricato in memoria!")
    except FileNotFoundError:
        print(f"❌ Errore: File '{nome_modello}' non trovato. Lancia prima lo script 03c per salvarlo.")
        return

    # 2. SIMULAZIONE UTENTE: Peschiamo un file a caso dal dataset
    # (In produzione, questo sarà il file caricato tramite interfaccia web)
    print("\n📩 Attesa caricamento file utente...")
    all_car_dict = np.load('all_car_dict_brand3.npy', allow_pickle=True).item()
    
    # Prendiamo un ID auto a caso e il suo primo file di ricarica
    import random
    car_id_casuale = random.choice(list(all_car_dict.keys()))
    file_utente = random.choice(all_car_dict[car_id_casuale])
    
    print(f"✅ Utente connesso. Auto ID: {car_id_casuale}")
    print(f"✅ File elaborato: {os.path.basename(file_utente)}")

    # 3. IL BACKEND (Data Preparation per l'Inferenza)
    # Leggiamo i dati esattamente come farebbe il tuo server dopo aver letto i DBC
    with open(file_utente, 'rb') as f:
        _ = pickle.load(f); _ = pickle.load(f); _ = pickle.load(f)
        payload = pickle.load(f)
    
    telemetria = payload[0]
    meta = payload[1]

    time_raw = telemetria[:, 7]
    volt_raw = telemetria[:, 0]
    curr_raw = telemetria[:, 1]
    temp_raw = telemetria[:, 5]

    # Sicurezza: controlliamo i duplicati temporali
    _, unique_indices = np.unique(time_raw, return_index=True)
    time_raw = time_raw[unique_indices]
    volt_raw = volt_raw[unique_indices]
    curr_raw = curr_raw[unique_indices]
    temp_raw = temp_raw[unique_indices]

    # IL FLATTENING SUL SINGOLO FILE (Lo stesso fatto in addestramento)
    time_norm = (time_raw - time_raw[0]) / (time_raw[-1] - time_raw[0])
    time_grid = np.linspace(0, 1, 100)

    f_volt = interp1d(time_norm, volt_raw, kind='linear', fill_value="extrapolate")
    f_curr = interp1d(time_norm, curr_raw, kind='linear', fill_value="extrapolate")
    f_temp = interp1d(time_norm, temp_raw, kind='linear', fill_value="extrapolate")

    volt_100 = f_volt(time_grid)
    curr_100 = f_curr(time_grid)
    temp_100 = f_temp(time_grid)

    riga_piatta = np.concatenate([volt_100, curr_100, temp_100, [meta.get('mileage', 0)]])
    
    # Trasformiamo la singola riga in una "Matrice" da una sola riga, perché XGBoost lo richiede
    X_utente = np.array([riga_piatta])

    # 4. LA DIAGNOSI
    print("\n🧠 Calcolo della salute (SoH) in corso...")
    # predict() restituisce un array, noi prendiamo il primo (e unico) elemento [0]
    capacita_stimata = modello.predict(X_utente)[0]
    
    # Visto che stiamo usando un file del dataset, conosciamo anche la risposta vera!
    capacita_reale = meta.get('capacity', 0)

    # 5. OUTPUT AL CLIENTE
    print("\n" + "="*50)
    print("🔋 REPORT DIAGNOSTICO BATTERIA")
    print("="*50)
    print(f"👉 Capacità STIMATA dal Server : {capacita_stimata:.2f} Ah")
    print(f"👉 Capacità REALE in Laboratorio: {capacita_reale:.2f} Ah")
    print("-" * 50)
    errore = abs(capacita_stimata - capacita_reale)
    print(f"⚖️ Margine di errore           : ± {errore:.2f} Ah")
    print("="*50)

if __name__ == "__main__":
    simula_backend_piattaforma()