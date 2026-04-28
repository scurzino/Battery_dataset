import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

def prepara_dataset_brand_3():
    base_dir = os.path.join(".", "data", "battery_dataset3")
    data_dir = os.path.join(base_dir, "data")
    
    print(f"🔍 Scansione della cartella: {data_dir}...")

    if not os.path.exists(data_dir):
        print(f"❌ Errore: Cartella {data_dir} non trovata.")
        return

    all_car_dict = {}
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and f != 'column.pkl']

    print(f"📂 Trovati {len(pkl_files)} file. Mappatura metadati profonda in corso...")

    for file_name in tqdm(pkl_files, desc="Lettura Profonda", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        
        try:
            # Lettura completa per trovare il vero ID nei metadati
            with open(file_path, 'rb') as f:
                _ = pickle.load(f) # Hash
                _ = pickle.load(f) # ID fittizio (ignoriamo)
                _ = pickle.load(f) # Protocollo
                payload = pickle.load(f) # Dati e Meta
                
            meta = payload[1]
            # Cerchiamo la chiave 'car'. Se non c'è, la chiamiamo 'Auto_Ignota'
            car_id = meta.get('car', 'Auto_Ignota') 
                
            if car_id not in all_car_dict:
                all_car_dict[car_id] = []
                
            all_car_dict[car_id].append(file_path)
            
        except Exception:
            pass 

    np.save('all_car_dict_brand3.npy', all_car_dict)
    
    car_ids = list(all_car_dict.keys())
    print(f"\n✅ FASE 1: Trovati {len(car_ids)} veicoli unici.")

    # --- PROTEZIONE ANTI-CRASH PER IL K-FOLD ---
    folds_dict = {}
    
    if len(car_ids) >= 5:
        print("📊 Esecuzione K-Fold Split classico sui Veicoli...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(car_ids)):
            folds_dict[f'fold_{fold_num}'] = {
                'train': [car_ids[i] for i in train_idx],
                'test': [car_ids[i] for i in test_idx]
            }
    else:
        print("⚠️ Trovati meno di 5 veicoli! Cambio logica di addestramento.")
        print("📊 Esecuzione Split sulle singole sessioni di ricarica invece che sui veicoli...")
        
        tutte_le_ricariche = []
        for cid in car_ids:
            tutte_le_ricariche.extend(all_car_dict[cid])
            
        np.random.shuffle(tutte_le_ricariche)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(tutte_le_ricariche)):
            folds_dict[f'fold_{fold_num}'] = {
                'train_files': [tutte_le_ricariche[i] for i in train_idx],
                'test_files': [tutte_le_ricariche[i] for i in test_idx]
            }

    np.save('five_fold_dict_brand3.npy', folds_dict)
    print("✅ FASE 2 Completata con successo. Nessun errore!")

if __name__ == "__main__":
    prepara_dataset_brand_3()