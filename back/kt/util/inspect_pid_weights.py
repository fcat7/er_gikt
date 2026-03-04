import torch
import os
import sys

# 将项目根目录添加到路径中，以便能够导入 kt 包
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def inspect_pid_weights(model_path):
    print(f"Loading model from {model_path}...")
    
    # 加载模型
    # 注意：由于 torch.save(model) 包含了类定义信息，
    # 加载时需要确保 GIKT 类在命名空间中可见（通常需要导入定义该类的模块）
    try:
        # 使用 weights_only=False 因为它是保存的整个对象，且我们信任这个文件
        model = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not hasattr(model, 'use_pid') or not model.use_pid:
        print("This model does not use PID.")
        return

    print(f"PID Mode: {model.pid_mode}")
    
    if hasattr(model, 'w_pid_i'):
        wi = model.w_pid_i.data
        wd = model.w_pid_d.data
        
        print("\n--- PID Weights (Integral $w_i$) ---")
        print(wi.numpy())
        
        print("\n--- PID Weights (Derivative $w_d$) ---")
        print(wd.numpy())
        
        if model.pid_mode == 'domain':
            # 尝试查看每个域的大小
            domain_map_path = os.path.join(os.path.dirname(model_path), '..', 'data', 'assist09-sample_10%', 'skill_domain_map.npy')
            if os.path.exists(domain_map_path):
                import numpy as np
                skill_domain_map = np.load(domain_map_path)
                unique, counts = np.unique(skill_domain_map, return_counts=True)
                domain_counts = dict(zip(unique, counts))
                
                print("\n--- Domain Distribution (Skill Counts per Domain) ---")
                for d_id, count in domain_counts.items():
                    print(f"Domain {int(d_id):2d}: {count:3d} skills | w_i: {wi[int(d_id)]:6.4f} | w_d: {wd[int(d_id)]:6.4f}")
            else:
                print(f"\nSkill domain map not found at {domain_map_path}, skipping domain distribution details.")
    else:
        print("PID weights not found in model parameters.")

# python util/inspect_pid_weights.py h:/er_gikt/back/kt/model/20260205_1241.pt
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # 默认找最新的模型文件
        model_dir = os.path.join(project_root, 'model')
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if not pt_files:
            print("No .pt files found in model directory.")
            sys.exit(1)
        # 按修改时间排序
        pt_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        path = os.path.join(model_dir, pt_files[0])
        print(f"No path provided, using latest model: {path}")

    inspect_pid_weights(path)
