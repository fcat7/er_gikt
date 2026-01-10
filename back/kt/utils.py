"""
工具类
"""
import os
import numpy as np
from scipy import sparse

def build_adj_list(data_dir='data'):
    # 返回每个问题的所有邻居, 每个技能的所有邻居
    qs_table = sparse.load_npz(os.path.join(data_dir, 'qs_table.npz')).toarray() # get qs_table ==> tensor(num_q, num_s)
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    q_neighbors_list = [[] for _ in range(num_question)] # 每个问题的邻居(技能)
    s_neighbors_list = [[] for _ in range(num_skill)] # 每个技能的邻居(问题)
    for q_id in range(num_question):
        s_ids = np.reshape(np.argwhere(qs_table[q_id] > 0), [-1]).tolist() # 每个问题涉及的技能id
        q_neighbors_list[q_id] += s_ids
        for s_id in s_ids:
            s_neighbors_list[s_id].append(q_id)
    return q_neighbors_list, s_neighbors_list

def gen_gikt_graph(q_neighbors_list, s_neighbors_list, q_neighbor_size=4, s_neighbor_size=10):
    """
    为问题和技能生成固定数量的邻居矩阵。

    Args:
        q_neighbors_list (list): 每个问题的邻居列表，列表的每个元素是一个邻居ID的列表。
        s_neighbors_list (list): 每个技能的邻居列表，列表的每个元素是一个邻居ID的列表。
        q_neighbor_size (int, optional): 每个问题需要选取的邻居数量，默认为4。
        s_neighbor_size (int, optional): 每个技能需要选取的邻居数量，默认为10。

    Returns:
        tuple: 包含两个numpy数组的元组 (q_neighbors, s_neighbors)：
            - q_neighbors (numpy.ndarray): 形状为(num_question, q_neighbor_size)的二维数组，包含每个问题选取的邻居ID。
            - s_neighbors (numpy.ndarray): 形状为(num_skill, s_neighbor_size)的二维数组，包含每个技能选取的邻居ID。

    Note:
        - 如果某个问题/技能的邻居数量不足指定的数量，会进行有放回的随机采样。
        - 如果邻居数量超过或等于指定数量，会进行无放回的随机采样。
        - 如果某个问题/技能没有邻居，对应的行将保持为0。
        - 这种做法最核心的目的就是为了保证计算的统一性，从而能进行高效的批量（batch）张量运算。
        - 思考是否能用更智能的方式（如基于重要性排序选择Top-K邻居，而不是随机选）来构建这个跳转表，以减轻信息损失。
        - 但无论如何，最终的张量形状必须是固定的，这是无法绕开的工程前提。
    """
    # 每个问题的固定数量的邻居(随机挑选)构成的矩阵, 每个技能也是同理
    num_question = len(q_neighbors_list) #  获取问题数量和技能数量
    num_skill = len(s_neighbors_list)
    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32) # 初始化问题的邻居矩阵
    s_neighbors = np.zeros([num_skill, s_neighbor_size], dtype=np.int32) # 技能的邻居矩阵
    for i, neighbors in enumerate(q_neighbors_list): #  遍历处理每个问题的邻居
        if len(neighbors) == 0: # 没有邻居的情况，跳过
            continue
        if len(neighbors) >= q_neighbor_size: # 邻居数量超过或等于最大限制,随机选取一些, 但[无重复]
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0: # 邻居数量不够, [有重复]地选取
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for i, neighbors in enumerate(s_neighbors_list): #  遍历处理每个技能的邻居
        if len(neighbors) == 0: #  没有邻居的情况，跳过
            continue
        if len(neighbors) >= s_neighbor_size: #  邻居数量超过或等于最大限制,随机选取一些, 但[无重复]
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0: #  邻居数量不够, [有重复]地选取
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=True)
    return q_neighbors, s_neighbors #  返回处理后的邻居矩阵
