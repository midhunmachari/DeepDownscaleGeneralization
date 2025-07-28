#!/usr/bin/env python3
"""
Noise-sensitivity sweep for eight precipitation downscaling models.

• Adds Gaussian noise (σ ∈ [0.1, 1.0], 20 linearly-spaced steps) to channel-1.
• Computes MSE (vs each model’s own control prediction) and KGE (vs HR reference).
• Saves a tidy results file:  noise_sensitivity_summary.csv
"""

import os, time
import numpy as np
import pandas as pd
import hydroeval as he
from tensorflow.keras.models import load_model
from ai4klima.tensorflow.utils import (
    make_predictions, negtozero, r_logtrans
)

def main(tag, noise_grid):
    # ──────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────────
    def log(msg):
        """Timestamped console print."""
        print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)

    # Define evaluation metrics functions
    def kge(y_true, y_pred):
        evaluations = y_true[~np.isnan(y_true)].flatten()
        simulations = y_pred[~np.isnan(y_pred)].flatten()
        kge, r, alpha, beta = he.evaluator(he.kgeprime, simulations, evaluations)
        return kge[0]
        
    def clean(arr):
        """Inverse-log-transform then set negatives → 0."""
        return negtozero(r_logtrans(arr))

    # ──────────────────────────────────────────────────────────────────────────────
    # Data loader
    # ──────────────────────────────────────────────────────────────────────────────
    def loadstack_inputs(bounds=(6209, 7305), *,
                        add_noise_ch1=False, add_noise_ch2=False,
                        noise_stddev=0.1):
        """Return [X_dyn, X_stat] ready for predict()  +  y_ref."""
        DATA_PATH = ("/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/"
                    "ANALYSE/PLTDATA/FM09_InputNoiseSensitivity/data")

        log("Loading dynamic channels …")
        ch1 = np.load(f"{DATA_PATH}/C01_080_IMRG_PREC_2001_2023_GMEAN_LOG.npy")
        ch2 = np.load(f"{DATA_PATH}/C01_080_IMRG_DCLM_2001_2023_GMEAN_LOG.npy")

        if add_noise_ch1:
            log(f"Adding Gaussian noise to channel-1 (σ={noise_stddev:.3f})")
            ch1 += np.random.normal(0.0, noise_stddev, ch1.shape)
        if add_noise_ch2:
            log(f"Adding Gaussian noise to channel-2 (σ={noise_stddev:.3f})")
            ch2 += np.random.normal(0.0, noise_stddev, ch2.shape)

        log("Loading static orography channel …")
        static = np.load(f"{DATA_PATH}/C01_080_GTOP_ELEV_2001_2023_GMEAN_CLOG.npy")
        static = np.expand_dims(static, axis=-1)

        log("Loading reference HR precipitation …")
        y_ref = np.load(f"{DATA_PATH}/C01_010_IMRG_PREC_2001_2023.npy")

        sl = slice(bounds[0], bounds[1])
        ch1, ch2  = ch1[sl], ch2[sl]
        static    = static[sl]
        y_ref     = y_ref[sl]

        X_dyn  = np.stack([ch1, ch2], axis=-1)   # (t, H, W, 2)
        X_stat = static                          # (t, H, W, 1)

        log(f"Shapes → dyn:{X_dyn.shape}, stat:{X_stat.shape}, y:{y_ref.shape}")
        return [X_dyn, X_stat], y_ref

    # ──────────────────────────────────────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────────────────────────────────────
    MPATH = ("/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/"
            "ANALYSE/PLTDATA/FM09_InputNoiseSensitivity/model_files")

    models_dict = {
        'SRCNN' : "p02a_hp-b32-r7e4-wmae_e01_m00_b13_ckpt_best_gen.keras",
        'FSRCNN': "p02a_hp-b32-r7e4-wmae_e01_m01_b13_ckpt_best_gen.keras",
        'EDRN'  : "p02a_hp-b32-r7e4-wmae_e01_m02_b13_ckpt_best_gen.keras",
        'SRDRN' : "p02a_hp-b32-r7e4-wmae_e01_m03_b13_ckpt_best_gen.keras",
        'U-NET' : "p02a_hp-b32-r7e4-wmae_e01_m04_b13_ckpt_best_gen.keras",
        'AU-NET': "p02a_hp-b32-r7e4-wmae_e01_m05_b13_ckpt_best_gen.keras",
        'U-GAN' : "p02a_hp-b32-r7e4-wmae_e01_m07_b13_ckpt_best_gen.keras",
        'AU-GAN': "p02a_hp-b32-r7e4-wmae_e01_m08_b13_ckpt_best_gen.keras",
    }
    models_dict = {k: os.path.join(MPATH, v) for k, v in models_dict.items()}


    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Baseline predictions (σ = 0)
    # ──────────────────────────────────────────────────────────────────────────────
    log("════════════════════════  BASELINE RUN  ════════════════════════")
    X_base, y_ref = loadstack_inputs(add_noise_ch1=False)
    baseline_preds = {}

    for name, path in models_dict.items():
        log(f"Loading model {name} …")
        mdl = load_model(path, compile=False)
        y_raw = make_predictions(mdl, X_base)   # log-space
        baseline_preds[name] = clean(y_raw)     # physical units
        log(f"{name} baseline prediction complete.")

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. Noise sweep
    # ──────────────────────────────────────────────────────────────────────────────
    results = []
    for i, sigma in enumerate(noise_grid, 1):
        print('\n')
        print('*'*100)
        log(f"════════════════════════  σ = {sigma:.3f}  "
            f"({i}/{len(noise_grid)})  ═══════════════════════")
        X_noisy, _ = loadstack_inputs(add_noise_ch1=True, noise_stddev=float(sigma))

        for name, path in models_dict.items():
            log(f"→ {name}")
            mdl = load_model(path, compile=False)
            y_hat = clean(make_predictions(mdl, X_noisy))

            mse_val = np.mean((y_hat - baseline_preds[name])**2)
            kge_val = kge(y_hat.ravel(), y_ref.ravel())
            log(f"   MSE={mse_val:.4e}   KGE={kge_val:+.3f}")

            results.append({
                "model"          : name,
                "sigma"          : round(float(sigma), 3),
                "MSE_vs_control" : float(mse_val),
                "KGE_vs_ref"     : float(kge_val),
            })

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Save results
    # ──────────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values(["model", "sigma"])

    save_path = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM09_InputNoiseSensitivity"
    out_csv = f"noise_sensitivity_summary_{tag}.csv"
    df.to_csv(f"{save_path}/{out_csv}", index=False)

    log("══════════════  sweep finished — results written to "
        f"{out_csv}  ══════════════")

if __name__== "__main__":

    noise_dict = {
        'noise_opt01': np.arange(0.0, 1.1, 0.1),
        'noise_opt02': np.arange(0.0, 0.21, 0.02),
    }

    for tag, noise_grid in noise_dict.items():
        main(tag, noise_grid)

