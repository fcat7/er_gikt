import pandas as pd
import os

def preprocess_nips2020():
    # 定义原始文件路径
    train_path = r"H:\\dataset\\NIPS2020\\public_data\\train_data\\train_task_3_4.csv"
    answer_meta_path = r"H:\\dataset\\NIPS2020\\public_data\\metadata\\answer_metadata_task_3_4.csv"
    question_meta_path = r"H:\\dataset\\NIPS2020\\public_data\\metadata\\question_metadata_task_3_4.csv"
    output_path = r"H:\\dataset\\NIPS2020\\public_data\\processed_nips2020.csv"

    print("正在加载数据...")
    df_train = pd.read_csv(train_path)
    df_ans_meta = pd.read_csv(answer_meta_path)
    df_q_meta = pd.read_csv(question_meta_path)

    print("正在关联表数据...")
    # 用 AnswerId 关联获取答题时间 (DateAnswered)
    if 'AnswerId' in df_train.columns and 'AnswerId' in df_ans_meta.columns:
        df = pd.merge(df_train, df_ans_meta[['AnswerId', 'DateAnswered']], on='AnswerId', how='left')
    else:
        df = df_train

    # 用 QuestionId 关联获取技能/知识点 (SubjectId)
    if 'QuestionId' in df.columns and 'QuestionId' in df_q_meta.columns:
        df = pd.merge(df, df_q_meta[['QuestionId', 'SubjectId']], on='QuestionId', how='left')

    print("正在清洗清洗技能ID并重命名列...")
    # NIPS2020 的 SubjectId 通常呈这种格式 "[3, 71, 115]"，转换为逗号分隔 "3,71,115"
    if 'SubjectId' in df.columns:
        df['SubjectId'] = df['SubjectId'].astype(str).str.replace(r'\[|\]|\s', '', regex=True)

    # 定义列名映射
    rename_mapping = {
        'UserId': 'user_id',
        'QuestionId': 'problem_id',
        'IsCorrect': 'correct',
        'SubjectId': 'skill_id',
        'DateAnswered': 'timestamp'
    }
    
    # 执行重命名
    df.rename(columns=rename_mapping, inplace=True)
    
    # 填补可能缺失的 response_time 列 (如果该数据集中没有相关的作答时长记录，用空值或者0占位以免后续报错)
    if 'response_time' not in df.columns:
        df['response_time'] = 0

    # 提取我们需要的核心列
    final_columns = ['user_id', 'problem_id', 'correct', 'skill_id', 'timestamp', 'response_time']
    # 兼容处理：确保要取的列都在 dataframe 中
    final_columns = [col for col in final_columns if col in df.columns]
    df_final = df[final_columns]

    print(f"正在保存最终文件至: {output_path}")
    df_final.to_csv(output_path, index=False)
    print("处理完成！总数据量:", len(df_final))

if __name__ == "__main__":
    preprocess_nips2020()
