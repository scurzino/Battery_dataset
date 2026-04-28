import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def verifica_chimica_500_sample():
    dict_path = 'all_car_dict_brand3.npy'
    if not os.path.exists(dict_path):
        print("❌ File di mappatura non trovato.")
        return

    all_car_dict = np.load(dict_path, allow_pickle=True).item()
    car_ids = list(all_car_dict.keys())
    
    num_samples = min(500, len(car_ids))
    print(f"📊 Estrazione di {num_samples} ricariche casuali per il mega-plot...")
    auto_campione = random.sample(car_ids, num_samples)
    
    plt.figure(figsize=(12, 7))
    
    base_color = 'indigo'
    
    for car_id in tqdm(auto_campione, desc="Estrazione Curve", unit="file"):
        if len(all_car_dict[car_id]) > 0:
            file_path = random.choice(all_car_dict[car_id])
            
            try:
                with open(file_path, 'rb') as f:
                    _ = pickle.load(f)
                    _ = pickle.load(f)
                    _ = pickle.load(f)
                    payload = pickle.load(f)
                    
                telemetria = payload[0]
                
                voltaggio = [riga[0] for riga in telemetria]
                soc = [riga[2] for riga in telemetria]
                
                # Trasparenza al 20% e linea più marcata
                plt.plot(soc, voltaggio, color=base_color, alpha=0.2, linewidth=1.0)
                
            except Exception:
                pass 

    plt.title(f'Firma Chimica: Voltaggio vs SoC (Densità su {num_samples} Ricariche)', fontsize=14)
    plt.xlabel('State of Charge (SoC %)', fontsize=12)
    plt.ylabel('Voltaggio (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.axhspan(3.2, 3.4, color='green', alpha=0.15, label='Zona Plateau LFP (~3.3V)')
    plt.legend(loc='lower right')
    
    print("\n📈 Rendering del grafico in corso... controlla la finestra!")
    plt.show()

if __name__ == "__main__":
    verifica_chimica_500_sample()