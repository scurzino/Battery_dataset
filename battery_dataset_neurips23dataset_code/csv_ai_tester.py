"""
inferenza_da_csv.py
-------------------
Testa il modello XGBoost (xgboost_blackbox.pkl) su un file CSV di sessione
di ricarica con colonne: Current [A] ; Voltage [V] ; Temp [°C] ; Age [km]

La corrente è in convenzione ricarica negativa (come nelle sessioni di training).
Il modello si aspetta una riga piatta di 301 feature:
  [volt_0..volt_99, curr_0..curr_99, temp_0..temp_99, mileage]
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "xgboost_blackbox.pkl"
CSV_PATH   = "test_modello.csv"      # <-- cambia se necessario
N_POINTS   = 100                     # punti griglia interpolazione

COLONNE_ATTESE = {
    "current": ["Current", "current", "Current [A]", "I"],
    "voltage": ["Voltage", "voltage", "Voltage [V]", "V"],
    "temp"   : ["Temp", "temp", "Temp [°C]", "Temperature", "T"],
    "age"    : ["Age", "age", "Age [km]", "Mileage", "mileage", "km"],
}


def trova_colonna(df: pd.DataFrame, candidati: list) -> str:
    """Trova il nome reale della colonna nel DataFrame."""
    for c in candidati:
        if c in df.columns:
            return c
    raise KeyError(
        f"Nessuna delle colonne {candidati} trovata nel CSV. "
        f"Colonne disponibili: {list(df.columns)}"
    )


def carica_modello(path: str):
    if not os.path.isfile(path):
        sys.exit(f"❌ Modello non trovato: '{path}'\n"
                 f"   Assicurati che il file .pkl sia nella stessa cartella dello script.")
    with open(path, "rb") as f:
        modello = pickle.load(f)
    print(f"✅ Modello caricato: {path}")
    return modello


def carica_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        sys.exit(f"❌ CSV non trovato: '{path}'")

    # Auto-detect separatore (virgola o punto-e-virgola)
    with open(path, "r") as f:
        prima_riga = f.readline()
    sep = ";" if prima_riga.count(";") > prima_riga.count(",") else ","

    df = pd.read_csv(path, sep=sep)
    print(f"✅ CSV caricato: {path}  →  {len(df)} righe, {len(df.columns)} colonne")
    return df


def filtra_ricarica(df: pd.DataFrame, col_curr: str) -> pd.DataFrame:
    """
    Mantiene solo le righe in cui la corrente è NEGATIVA (ricarica).
    Stampa un avviso se ci sono tratti in scarica.
    """
    n_tot = len(df)
    df_ric = df[df[col_curr] < 0].copy().reset_index(drop=True)
    n_disc = n_tot - len(df_ric)
    if n_disc > 0:
        print(f"   ⚠️  Scartate {n_disc} righe con corrente ≥ 0 (non-ricarica).")
    print(f"   ✅  Righe di ricarica usate per l'inferenza: {len(df_ric)}")
    return df_ric


def prepara_feature_vector(
    curr: np.ndarray,
    volt: np.ndarray,
    temp: np.ndarray,
    mileage: float,
    n_points: int = N_POINTS,
) -> np.ndarray:
    """
    Interpola le tre serie su una griglia uniforme [0,1] di n_points passi
    e concatena con il chilometraggio → vettore di (3*n_points + 1) feature.
    """
    n = len(volt)
    if n < 2:
        raise ValueError("Servono almeno 2 campioni per l'interpolazione.")

    # Asse temporale normalizzato (indice progressivo)
    t_raw  = np.arange(n, dtype=float)
    t_norm = t_raw / t_raw[-1]
    t_grid = np.linspace(0, 1, n_points)

    def interpola(segnale):
        # Rimuoviamo eventuali duplicati sull'asse t
        _, idx = np.unique(t_norm, return_index=True)
        f = interp1d(t_norm[idx], segnale[idx], kind="linear", fill_value="extrapolate")
        return f(t_grid)

    volt_i = interpola(volt)
    curr_i = interpola(curr)
    temp_i = interpola(temp)

    return np.concatenate([volt_i, curr_i, temp_i, [mileage]])


def stampa_report(
    soh_stimato: float,
    mileage: float,
    volt: np.ndarray,
    curr: np.ndarray,
    temp: np.ndarray,
):
    """Stampa un report diagnostico ricco di metriche."""

    separatore = "=" * 55

    # Statistiche segnali grezzi
    v_min, v_max, v_mean = volt.min(), volt.max(), volt.mean()
    i_min, i_max, i_mean = curr.min(), curr.max(), curr.mean()
    t_min, t_max, t_mean = temp.min(), temp.max(), temp.mean()

    # Energia approssimativa (trapezoide su indice, non tempo reale)
    # Corrente in ricarica è negativa → usiamo il valore assoluto
    energia_relativa_wh = np.trapz(np.abs(curr) * volt) / 3600  # in Wh (se dati a 1 Hz)

    print()
    print(separatore)
    print("         🔋 REPORT DIAGNOSTICO BATTERIA")
    print(separatore)
    print(f"  Chilometraggio (Age)    : {mileage:>10.0f}  km")
    print()
    print("  ── SEGNALI RICARICA ──────────────────────────")
    print(f"  Tensione   [V]   min / media / max : "
          f"{v_min:6.1f} / {v_mean:6.1f} / {v_max:6.1f}")
    print(f"  Corrente   [A]   min / media / max : "
          f"{i_min:6.1f} / {i_mean:6.1f} / {i_max:6.1f}")
    print(f"  Temperatura[°C]  min / media / max : "
          f"{t_min:6.1f} / {t_mean:6.1f} / {t_max:6.1f}")
    print()
    print(f"  Energia ricarica (≈, @1Hz)         : {energia_relativa_wh:8.1f}  Wh")
    print()
    print("  ── DIAGNOSI AI ───────────────────────────────")
    print(f"  ⚡ Capacità STIMATA (SoH proxy)    : {soh_stimato:8.2f}  Ah")
    print()

    # Interpretazione qualitativa (adattare alle soglie del tuo dataset)
    if soh_stimato >= 45:
        stato = "🟢 OTTIMO"
        nota  = "Batteria in ottima salute."
    elif soh_stimato >= 38:
        stato = "🟡 BUONO"
        nota  = "Qualche degradazione, ma ancora nella norma."
    elif soh_stimato >= 30:
        stato = "🟠 DA MONITORARE"
        nota  = "Capacità ridotta — valutare diagnostica approfondita."
    else:
        stato = "🔴 CRITICO"
        nota  = "Capacità molto bassa — sostituzione consigliata."

    print(f"  Stato batteria : {stato}")
    print(f"  Note           : {nota}")
    print(separatore)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "─" * 55)
    print("  INFERENZA MODELLO AI SU FILE CSV PERSONALIZZATO")
    print("─" * 55)

    # 1. Carica modello e dati
    modello = carica_modello(MODEL_PATH)
    df      = carica_csv(CSV_PATH)

    # 2. Identifica colonne (flessibile sui nomi)
    col_curr = trova_colonna(df, COLONNE_ATTESE["current"])
    col_volt = trova_colonna(df, COLONNE_ATTESE["voltage"])
    col_temp = trova_colonna(df, COLONNE_ATTESE["temp"])
    col_age  = trova_colonna(df, COLONNE_ATTESE["age"])
    print(f"   Colonne usate → corrente='{col_curr}', "
          f"tensione='{col_volt}', temp='{col_temp}', età='{col_age}'")

    # 3. Filtra solo le righe di ricarica (corrente negativa)
    df = filtra_ricarica(df, col_curr)

    if len(df) < 2:
        sys.exit("❌ Troppo pochi campioni di ricarica per procedere (< 2).")

    # 4. Estrai array numpy
    curr    = df[col_curr].to_numpy(dtype=float)
    volt    = df[col_volt].to_numpy(dtype=float)
    temp    = df[col_temp].to_numpy(dtype=float)
    mileage = df[col_age].iloc[0]  # il chilometraggio è costante nel file

    # 5. Prepara il feature vector (stessa pipeline del training)
    print("\n🔄 Interpolazione e flattening in corso...")
    X = prepara_feature_vector(curr, volt, temp, mileage)
    X_input = X.reshape(1, -1)  # shape (1, 301)

    print(f"   Feature vector: {X_input.shape[1]} feature "
          f"({N_POINTS} V + {N_POINTS} I + {N_POINTS} T + 1 km)")

    # 6. Inferenza
    print("🧠 Inferenza modello XGBoost...")
    capacita_stimata = modello.predict(X_input)[0]

    # 7. Report
    stampa_report(capacita_stimata, mileage, volt, curr, temp)


if __name__ == "__main__":
    main()