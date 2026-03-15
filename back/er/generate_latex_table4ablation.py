import pandas as pd

def generate_latex_table_ablation(csv_path):
    df = pd.read_csv(csv_path)

    metrics = ['dm', 'wkc', 'ild', 'skill_hit_rate', 'novelty']
    mode_order = ['full', 'no_pid', 'no_mopso', 'no_f2']
    mode_name_map = {
        'full': r'\textbf{CCS-MOPSO-ER (完全版)}',
        'no_pid': 'w/o 动态 ZPD (固定 $\\tau=0.55$)',
        'no_mopso': 'w/o MOPSO (退化为贪心)',
        'no_f2': 'w/o 弱点覆盖目标 (移除 $F_{2}$)'
    }

    agg_df = df.groupby('mode')[metrics].agg(['mean', 'std'])

    latex_code = []
    latex_code.append(r'\begin{table}[htbp]')
    latex_code.append(r'    \centering')
    latex_code.append(r'    \caption{CCS-MOPSO-ER 消融实验评估结果 (Mean $\pm$ SD)}')
    latex_code.append(r'    \label{tab:ablation_results}')
    latex_code.append(r'    \resizebox{\textwidth}{!}{')
    latex_code.append(r'    \begin{tabular}{l|ccccc}')
    latex_code.append(r'        \toprule')
    latex_code.append(r'        \textbf{消融配置} & \textbf{难度适配度(DM) $\uparrow$} & \textbf{弱点知识覆盖率(WKC) $\uparrow$} & \textbf{列表内多样性(ILD) $\uparrow$} & \textbf{知识命中率(SHR) $\uparrow$} & \textbf{新颖性(Novelty) $\uparrow$} \\')
    latex_code.append(r'        \midrule')

    for mode in mode_order:
        if mode not in agg_df.index: continue
        row_cells = [mode_name_map[mode]]
        
        for m in metrics:
            if m not in agg_df.columns.levels[0]:
                row_cells.append('N/A')
                continue
                
            m_val = float(agg_df.loc[mode, (m, 'mean')])
            s_val = float(agg_df.loc[mode, (m, 'std')])
            
            # Get all means for this metric to find best/second best
            all_means = agg_df[(m, 'mean')].dropna()
            
            if pd.isna(m_val) or (m_val == 0.0 and s_val == 0.0) and m != 'wkc':
                row_cells.append('N/A')
            else:
                formatted_mean = f"{m_val:.4f}"
                formatted_std = f"{s_val:.4f}"
                
                # logic for bold (best) and underline (second best) on MEAN ONLY
                if len(all_means) > 0:
                    sorted_vals = sorted(all_means.unique(), reverse=True)
                    if abs(m_val - max(all_means)) < 1e-7:
                        # Bold the mean part only
                        val_str = r'$\mathbf{' + formatted_mean + r'} \pm ' + formatted_std + r'$'
                    elif len(sorted_vals) >= 2 and abs(m_val - sorted_vals[1]) < 1e-7:
                        # Underline the mean part only
                        val_str = r'$\underline{' + formatted_mean + r'} \pm ' + formatted_std + r'$'
                    else:
                        val_str = r'$' + formatted_mean + r' \pm ' + formatted_std + r'$'
                else:
                    val_str = r'$' + formatted_mean + r' \pm ' + formatted_std + r'$'
                    
                row_cells.append(val_str)
                
        latex_code.append('        ' + ' & '.join(row_cells) + r' \\')

    latex_code.append(r'        \bottomrule')
    latex_code.append(r'    \end{tabular}')
    latex_code.append(r'    }')
    latex_code.append(r'    \vspace{2pt}')
    latex_code.append(r'    \begin{flushleft}')
    latex_code.append(r'    {\footnotesize 注: $\uparrow$ 表示该指标数值越大性能越优。粗体表示最优结果，下划线表示次优结果。 }')
    latex_code.append(r'    \end{flushleft}')
    latex_code.append(r'\end{table}')
    
    return '\n'.join(latex_code)

if __name__ == "__main__":
    csv_path = r'H:\er_gikt\back\er\output\ablation_er\ablation_eval_full.csv'
    try:
        latex_table = generate_latex_table_ablation(csv_path)
        print(latex_table)
    except FileNotFoundError:
        print(f"File not found: {csv_path}. Please run run_ablation_real.py first.")
