from scipy.spatial.distance import cosine
import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import os
from torch_geometric.data import Data
import jieba
import torch

jieba.set_dictionary(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict.txt.big")
)


w2v_dim = 300
max_len = 50

dic = {
    "non-rumor": 0,  # Non-rumor   NR
    "false": 1,  # false rumor    FR
    "unverified": 2,  # unverified tweet  UR
    "true": 3,  # debunk rumor  TR
}


def clean_str_cut(string, task="twitter"):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = (
        list(jieba.cut(string.strip().lower()))
        if task == "weibo"
        else string.strip().lower().split()
    )
    return words


def read_replies(filepath, tweet_id, task, max_replies=30):
    filepath1 = filepath + "replies/" + tweet_id + ".txt"
    # print('filepath1',filepath1)
    replies = []
    if os.path.exists(filepath1):
        with open(filepath1, "r", encoding="utf-8") as fin:
            for line in fin:
                replies.append(clean_str_cut(line, task)[:max_len])
    return replies[:max_replies]


import json
import os


def read_replies_weibo(filepath, tweet_id, task, max_replies=30):
    filepath1 = filepath + "replies/" + tweet_id + ".json"
    # print('filepath1',filepath1)
    replies = []
    if os.path.exists(filepath1):
        with open(filepath1, "r", encoding="utf-8") as fin:
            for line in fin:
                replies.append(clean_str_cut(line, task)[:max_len])
    else:
        replies.append([])
    return replies[:max_replies]


def read_train_dev_test(root_path, file_name, appendix, X_all_tids):
    filepath = root_path + file_name + appendix
    with open(filepath, "r", encoding="utf-8") as fin:
        X_tid, X_content, X_replies, y_ = [], [], [], []
        for line in fin.readlines():
            tid, content, label = line.strip().split("\t")
            X_all_tids.append(tid)
            X_tid.append(tid)
            if file_name == "weibo":
                replies = read_replies_weibo(root_path, tid, file_name)
                # print('weibo')
            else:
                replies = read_replies(root_path, tid, file_name)
            # print('read_train_dev_test replies:',replies)
            X_replies.append(replies)
            X_content.append(clean_str_cut(content, file_name)[:max_len])
            y_.append(dic[label])
    print("read_train_dev_test:", len(X_tid))
    print("read_train_dev_test:", len(X_content))
    print("read_train_dev_test:", len(X_replies))
    print("read_train_dev_test:", len(y_))
    return X_tid, X_content, X_replies, y_


def read_dataset(root_path, file_name):
    X_all_tids = []
    X_all_uids = []

    X_train_tid, X_train_content, X_train_replies, y_train = read_train_dev_test(
        root_path, file_name, ".train", X_all_tids
    )
    X_dev_tid, X_dev_content, X_dev_replies, y_dev = read_train_dev_test(
        root_path, file_name, ".dev", X_all_tids
    )
    X_test_tid, X_test_content, X_test_replies, y_test = read_train_dev_test(
        root_path, file_name, ".test", X_all_tids
    )

    with open(root_path + file_name + "_graph.txt", "r", encoding="utf-8") as input:
        edge_index, edges_weight = [], []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]

            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_all_uids.append(dst)
                # edge_index.append([src, dst])
                edge_index.append([dst, src])
                # edges_weight.append(float(w))
                edges_weight.append(float(w))

    X_id = list(set(X_all_tids + X_all_uids))
    num_node = len(X_id)
    # print(num_node)
    X_id_dic = {id: i + 1 for i, id in enumerate(X_id)}

    edges_list = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index]
    edges_list = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight)
    data = Data(edge_index=edges_list, edge_weight=edges_weight)

    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])

    return (
        X_train_tid,
        X_train_content,
        X_train_replies,
        y_train,
        X_dev_tid,
        X_dev_content,
        X_dev_replies,
        y_dev,
        X_test_tid,
        X_test_content,
        X_test_replies,
        y_test,
        data,
    )


def read_dataset_weibo(root_path, file_name):
    X_all_uids = []  # 仅保存用户ID，Weibo数据集仅包含用户之间的关系

    # 读取训练集、开发集和测试集（根据需要调整读取函数）
    X_train_tid, X_train_content, X_train_replies, y_train = read_train_dev_test(
        root_path, file_name, ".train", X_all_uids
    )
    print("read_train_dev_test:", X_train_replies)
    X_dev_tid, X_dev_content, X_dev_replies, y_dev = read_train_dev_test(
        root_path, file_name, ".dev", X_all_uids
    )
    X_test_tid, X_test_content, X_test_replies, y_test = read_train_dev_test(
        root_path, file_name, ".test", X_all_uids
    )

    # 读取 Weibo 数据集的网络关系文件
    with open(root_path + file_name + "_graph.txt", "r", encoding="utf-8") as input:
        edge_index, edges_weight = [], []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]

            # 假设 Weibo 数据集中的每一行表示用户关系
            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_all_uids.append(dst)  # 仅保存用户ID
                # 添加边和权重
                edge_index.append([src, dst])
                edges_weight.append(float(w))

    # 去重，生成唯一用户ID列表
    X_id = list(set(X_all_uids))
    X_id_dic = {id: i + 1 for i, id in enumerate(X_id)}

    # 构建边索引的张量
    edges_list = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index]
    edges_list = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight)
    data = Data(edge_index=edges_list, edge_weight=edges_weight)

    # 将训练集、开发集和测试集的 Tweet ID 转换为用户索引
    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid if tid in X_id_dic])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid if tid in X_id_dic])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid if tid in X_id_dic])

    print(
        "read_dataset_weibo:",
        len(X_train_tid),
        len(X_train_content),
        len(X_train_replies),
        len(y_train),
    )

    return (
        X_train_tid,
        X_train_content,
        X_train_replies,
        y_train,
        X_dev_tid,
        X_dev_content,
        X_dev_replies,
        y_dev,
        X_test_tid,
        X_test_content,
        X_test_replies,
        y_test,
        data,
    )


def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            # add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing) + " words found in word2vec.")
    print(str(count_missing) + " words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path="numberbatch-en.txt"):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] >= 2]
    vocabulary_inv = vocabulary_inv[1:]  # don't need <PAD>
    # Mapping from word to index
    word_to_ix = {x: i + 1 for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, word_to_ix)  #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return word_to_ix, embedding_weights


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size + 1, w2v_dim), dtype="float32")
    # initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,))

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size " + str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, word_to_ix, is_replies=False, max_replies=30):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    if not is_replies:
        # 对内容数据进行填充
        X = [
            [0] * (max_len - len(sentence))
            + [word_to_ix.get(word, 0) for word in sentence]
            for sentence in X
        ]
    else:
        # 对回复数据进行填充
        X = [
            [[0] * max_len] * (max_replies - len(replies))
            + [
                [0] * (max_len - len(doc)) + [word_to_ix.get(word, 0) for word in doc]
                for doc in replies
            ]
            for replies in X
        ]
        # 确保所有回复列表的长度一致
        X = [replies[:max_replies] for replies in X]
    return X


# def build_input_data(X, word_to_ix, is_replies=False, max_replies=30, max_len=50):  # #Assuming max_len is 50 as an example
#    """
#    Maps sentences and labels to vectors based on a vocabulary.
#    """
#    if not is_replies:
#        # For content data
#        print("Processing content data...")
#        X = [[0] * (max_len - len(sentence)) + [word_to_ix.get(word, 0) for word in sentence] for sentence in X]
#        # Debug: Print transformed content data
#        print("Transformed content data sample:", X[:2])  # Print first two samples for inspection
#    else:
#        # For reply data
#        print("Processing replies data...")
#        X = [
#            [[0] * max_len] * (max_replies - len(replies)) +
#            [[0] * (max_len - len(doc)) + [word_to_ix.get(word, 0) for word in doc] for doc in replies]
#            for replies in X
#        ]
# Ensure all reply lists have the same length (max_replies)
#        X = [replies[:max_replies] for replies in X]
# Debug: Print transformed replies data
#        print("Transformed replies data sample:", X[:2])  # Print first two samples for inspection
#    return X


def compute_cosine_similarity(
    X_content, X_replies, word_embeddings, threshold=0.4, max_replies=30
):  # 30
    # print(threshold)
    all_cosine_similarities = []
    filtered_replies = []
    print("compute_cosine_similarity", len(X_content))
    print("compute_cosine_similarity", len(X_replies))
    for content, replies in zip(X_content, X_replies):
        # 计算 content_embedding 时，确保非空
        # print(replies)
        content_embeddings = [
            word_embeddings[word_id] for word_id in content if word_id != 0
        ]
        # print(content_embeddings)
        valid_replies = []
        cosine_similarities = []

        if not content_embeddings:
            # valid_replies.append([])
            # cosine_similarities.append([])
            filtered_replies.append([[0] * max_len] * max_replies)
            continue  # 如果没有有效的 embedding，跳过计算

        # 将 content_embeddings 转换为 numpy 数组并计算均值
        content_embeddings = np.array(content_embeddings, dtype=object)
        content_embedding = np.mean(content_embeddings, axis=0)

        for reply in replies:
            # print(reply)
            # 计算 reply_embedding 时，确保非空
            reply_embeddings = [
                word_embeddings[word_id] for word_id in reply if word_id != 0
            ]
            # print(reply_embeddings)
            if not reply_embeddings:
                valid_replies.append([0] * max_len)
                # print('jump calculate')
                continue  # 如果没有有效的 embedding，跳过计算

            # 将 reply_embeddings 转换为 numpy 数组并计算均值
            reply_embeddings = np.array(reply_embeddings, dtype=object)
            reply_embedding = np.mean(reply_embeddings, axis=0)

            similarity = 1 - cosine(content_embedding, reply_embedding)
            # print(similarity)
            cosine_similarities.append(similarity)

            if similarity >= threshold:
                valid_replies.append(reply)
            else:
                # 用全 0 向量替代不符合条件的回复
                valid_replies.append([0] * max_len)

        all_cosine_similarities.extend(cosine_similarities)
        # rint(all_cosine_similarities)
        # 确保回复列表的长度与最大回复数一致
        valid_replies = valid_replies[:max_replies]
        valid_replies.extend([[0] * max_len] * (max_replies - len(valid_replies)))

        filtered_replies.append(valid_replies)

    avg_similarity = np.mean(all_cosine_similarities) if all_cosine_similarities else 0
    return filtered_replies, avg_similarity


def w2v_feature_extract(root_path, filename, w2v_path, threshold=0.4):
    if filename == "weibo":
        (
            X_train_tid,
            X_train_content,
            X_train_replies,
            y_train,
            X_dev_tid,
            X_dev_content,
            X_dev_replies,
            y_dev,
            X_test_tid,
            X_test_content,
            X_test_replies,
            y_test,
            Adj,
        ) = read_dataset_weibo(root_path, filename)

    else:
        (
            X_train_tid,
            X_train_content,
            X_train_replies,
            y_train,
            X_dev_tid,
            X_dev_content,
            X_dev_replies,
            y_dev,
            X_test_tid,
            X_test_content,
            X_test_replies,
            y_test,
            Adj,
        ) = read_dataset(root_path, filename)
    # print('read_dataset(root_path, filename):',X_train_replies)
    print("Text word2vec generation.......")
    text_data = X_train_content + X_dev_content  # + X_test_content
    vocabulary, word_embeddings = build_vocab_word2vec(text_data, w2v_path=w2v_path)
    pickle.dump(vocabulary, open(root_path + "/vocab.pkl", "wb"))
    print("Vocabulary size: " + str(len(vocabulary)))

    print("Building input data.......")
    X_train_content = build_input_data(X_train_content, vocabulary)
    X_dev_content = build_input_data(X_dev_content, vocabulary)
    X_test_content = build_input_data(X_test_content, vocabulary)

    X_train_replies = build_input_data(X_train_replies, vocabulary, True)
    X_dev_replies = build_input_data(X_dev_replies, vocabulary, True)
    X_test_replies = build_input_data(X_test_replies, vocabulary, True)
    print("X_train_content:", len(X_train_content))
    print("X_train_replies:", len(X_train_replies))
    # print('X_train_replies:',X_train_replies)

    print("Calculating cosine similarity and filtering replies...")
    X_train_replies_filtered, avg_train_similarity = compute_cosine_similarity(
        X_train_content, X_train_replies, word_embeddings, threshold
    )
    X_dev_replies_filtered, avg_dev_similarity = compute_cosine_similarity(
        X_dev_content, X_dev_replies, word_embeddings, threshold
    )
    X_test_replies_filtered, avg_test_similarity = compute_cosine_similarity(
        X_test_content, X_test_replies, word_embeddings, threshold
    )
    print("X_train_replies_filtered:", len(X_train_replies_filtered))
    # print()
    print("Average cosine similarity for train set: ", avg_train_similarity)
    print("Average cosine similarity for dev set: ", avg_dev_similarity)
    print("Average cosine similarity for test set: ", avg_test_similarity)

    pickle.dump(
        [
            X_train_tid,
            X_train_content,
            X_train_replies_filtered,
            y_train,
            word_embeddings,
            Adj,
        ],
        open(root_path + "/train.pkl", "wb"),
    )
    pickle.dump(
        [X_dev_tid, X_dev_content, X_dev_replies_filtered, y_dev],
        open(root_path + "/dev.pkl", "wb"),
    )
    pickle.dump(
        [X_test_tid, X_test_content, X_test_replies_filtered, y_test],
        open(root_path + "/test.pkl", "wb"),
    )

    return [
        [
            X_train_tid,
            X_train_content,
            X_train_replies_filtered,
            y_train,
            word_embeddings,
            Adj,
        ],
        [X_dev_tid, X_dev_content, X_dev_replies_filtered, y_dev],
        [X_test_tid, X_test_content, X_test_replies_filtered, y_test],
    ]


if __name__ == "__main__":
    # w2v_feature_extract('./twitter15/', "twitter15", "twitter_w2v.bin")
    w2v_feature_extract("./twitter16/", "twitter16", "twitter_w2v.bin")
    # w2v_feature_extract('./weibo/', "weibo", "weibo_w2v.bin")
