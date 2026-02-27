class StandardColumns:
    USER_ID        = "user_id"           # 用户ID
    QUESTION_ID    = "question_id"           # 题目ID
    SKILL_IDS      = "skill_ids"           # 技能ID列表（可为单个或多个）
    LABEL          = "label"         # 答题标签（对错/分数）
    TIMESTAMP      = "timestamp"     # 时间戳
    RESPONSE_TIME  = "response_time"            # 答题时长（响应时间）

    ALL = [USER_ID, QUESTION_ID, SKILL_IDS, LABEL, TIMESTAMP, RESPONSE_TIME]
    SKILL_IDS_STD_SEP = "_"  # 标准分隔符