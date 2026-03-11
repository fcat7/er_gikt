import pandas as pd
import os
import json

input_dir = r"h:\er_gikt\temp\超参数调优"
output_dir = r"h:\er_gikt\back\kt\config\best_params"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        if 'model_name' not in df.columns or 'dataset_name' not in df.columns:
            continue
            
        model_name = str(df['model_name'].iloc[0]).lower()
        datasets = df['dataset_name'].unique()
        
        # Columns to ignore
        ignore_cols = ['Unnamed: 0', 'model_name', 'dataset_name', 'fold', 'seed', 'emb_type']
        param_cols = [c for c in df.columns if c not in ignore_cols]
        
        best_params = {}
        for ds in datasets:
            ds_df = df[df['dataset_name'] == ds]
            # Get the most common configuration
            mode_params = ds_df[param_cols].mode().iloc[0].to_dict()
            # Convert types to python native types
            clean_params = {}
            for k, v in mode_params.items():
                if pd.isna(v): continue
                # Handle numpy types by converting via float or int
                if "learning_rate" in k or "dropout" in k or "l2" in k or "weight_decay" in k:
                    clean_params[k] = float(v)
                else:
                    if isinstance(v, float) and v.is_integer():
                        clean_params[k] = int(v)
                    elif isinstance(v, (int, float)):
                        clean_params[k] = int(v) if float(v).is_integer() else float(v)
                    else:
                        clean_params[k] = str(v)
            best_params[str(ds)] = clean_params
            
        out_file = os.path.join(output_dir, f"{model_name}_best_params.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4)
        print(f"Saved {out_file} with datasets: {list(best_params.keys())}")
