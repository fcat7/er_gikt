import pandas as pd
import os

# 定义相对路径
file1_path = '../data/assist09/assist09_standard.csv'     # 新的处理方法 (4234 users)
file2_path = '../data/assist09-all/assist09_standard.csv' # 原来的 (4233 users)

def load_users(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return set()
    print(f"Loading {path}...")
    try:
        df = pd.read_csv(path, usecols=['user_id'])
        users = set(df['user_id'].unique())
        print(f" -> Found {len(users)} unique users.")
        return users
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return set()

def main():
    print("=== Comparing User Sets ===")
    users_new = load_users(file1_path)
    users_old = load_users(file2_path)
    
    if not users_new or not users_old:
        print("One or both datasets failed to load. Exiting.")
        return

    # 找出差异
    new_only = users_new - users_old
    old_only = users_old - users_new
    
    print("\n=== Comparison Results ===")
    
    if new_only:
        print(f"Users present only in NEW dataset ({file1_path}):")
        print(f"Count: {len(new_only)}")
        print(f"User IDs: {sorted(list(new_only))}")
    else:
        print(f"No extra users in NEW dataset.")
        
    if old_only:
        print(f"Users present only in OLD dataset ({file2_path}):")
        print(f"Count: {len(old_only)}")
        print(f"User IDs: {sorted(list(old_only))}")
    else:
        print(f"No extra users in OLD dataset.")
        
    if not new_only and not old_only:
        print("The user sets are identical.")

if __name__ == "__main__":
    main()
