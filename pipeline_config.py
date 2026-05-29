from enum import Enum
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class Method(Enum):
    COS = "cos"
    THRED = "thred"


def glan_module_name(method):
    return "GLAN_cos" if method == Method.COS else "GLAN"

def glan_dropS_module_name(method):
    return "GLAN_cos_dropS" if method == Method.COS else "GLAN_dropS"
def glan_dropC_module_name(method):
    return "GLAN_cos_dropC" if method == Method.COS else "GLAN_dropC"

def _tag(method):
    return method.value


def path_glan_train_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_train_outputs{rep}.npy"
    )


def path_glan_val_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_val_outputs{rep}.npy"
    )


def path_glan_test_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_test_outputs{rep}.npy"
    )

def path_glan_train_outputs_dropS(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropS_train_outputs{rep}.npy"
    )


def path_glan_val_outputs_dropS(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropS_val_outputs{rep}.npy"
    )


def path_glan_test_outputs_dropS(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropS_test_outputs{rep}.npy"
    )

def path_glan_train_outputs_dropC(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropC_train_outputs{rep}.npy"
    )


def path_glan_val_outputs_dropC(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropC_val_outputs{rep}.npy"
    )


def path_glan_test_outputs_dropC(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_GLAN{_tag(method)}_dropC_test_outputs{rep}.npy"
    )

def path_glan_log(method, data, rep, threshold=0.9):
    return os.path.join(LOG_DIR, f"{data}_GLAN{_tag(method)}{rep}_{threshold}.txt")

def path_glan_dropS_log(method, data, rep):
    return os.path.join(LOG_DIR, f"{data}_GLAN{_tag(method)}_dropS{rep}.txt")
def path_glan_dropC_log(method, data, rep):
    return os.path.join(LOG_DIR, f"{data}_GLAN{_tag(method)}_dropC{rep}.txt")

def path_glan_best_model(method, data, rep):
    return os.path.join(CHECKPOINT_DIR, f"{data}bestGLAN{_tag(method)}model{rep}.pth")

def path_glan_dropS_best_model(method, data, rep):
    return os.path.join(CHECKPOINT_DIR, f"{data}bestGLAN{_tag(method)}model_dropS{rep}.pth")
def path_glan_dropC_best_model(method, data, rep):
    return os.path.join(CHECKPOINT_DIR, f"{data}bestGLAN{_tag(method)}model_dropC{rep}.pth")

def path_bigcn_train_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_BiGCN{_tag(method)}_train_outputs{rep}.npy"
    )


def path_bigcn_val_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_BiGCN{_tag(method)}_val_outputs{rep}.npy"
    )


def path_bigcn_test_outputs(method, data, rep):
    return os.path.join(
        CHECKPOINT_DIR, f"{data}_BiGCN{_tag(method)}_test_outputs{rep}.npy"
    )


def path_bigcn_log(method, data, rep):
    return os.path.join(LOG_DIR, f"{data}_BiGCN{_tag(method)}{rep}.txt")


def path_bigcn_best_model(method, data, rep):
    return os.path.join(CHECKPOINT_DIR, f"{data}best{_tag(method)}BiGCNmodel{rep}.pth")


def path_combined_model(method, data, rep):
    return os.path.join(CHECKPOINT_DIR, f"newmodel_{_tag(method)}_{data}{rep}.pth")


def write_timing(seed, dataset, method, pipeline, model, time_seconds):
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(LOG_DIR, "timing.txt")
    with open(filepath, "a+", encoding="utf-8") as f:
        f.write(
            f"[{now}] seed={seed} | dataset={dataset} | method={method}"
            f" | pipeline={pipeline} | model={model}"
            f" | time={time_seconds:.2f}s\n"
        )


def write_flops(seed, dataset, method, pipeline, model, flops, params):
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(LOG_DIR, "flops.txt")
    with open(filepath, "a+", encoding="utf-8") as f:
        f.write(
            f"[{now}] seed={seed} | dataset={dataset} | method={method}"
            f" | pipeline={pipeline} | model={model}"
            f" | FLOPs={flops:.0f} | Params={params}\n"
        )

def write_timing_for_drop(seed, dataset, method, pipeline, model, time_seconds):
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(LOG_DIR, "timing.txt")
    with open(filepath, "a+", encoding="utf-8") as f:
        f.write(
            f"[{now}] seed={seed} | dataset={dataset} | method={method}"
            f" | pipeline={pipeline} | model={model}"
            f" | time={time_seconds:.2f}s\n"
        )

# def write_countdataset(data, y_train, y_dev, y_test, stats):
#     with open(os.path.join(LOG_DIR, f"{data}_statistics.txt"), 'w') as f:
#         f.write(f"Dataset: {data}\n")
#         f.write(f"Total source posts: {len(y_train) + len(y_dev) + len(y_test)}\n")
#         f.write(f"Training set: {len(y_train)}\n")
#         f.write(f"Validation set: {len(y_dev)}\n")
#         f.write(f"Test set: {len(y_test)}\n")
#         f.write("\nClass distribution:\n")
#         f.write(f"Non-rumor (N): {stats['train']['Non-rumor (N)'] + stats['val']['Non-rumor (N)'] + stats['test']['Non-rumor (N)']}\n")
#         f.write(f"False rumor (F): {stats['train']['False rumor (F)'] + stats['val']['False rumor (F)'] + stats['test']['False rumor (F)']}\n")
#         f.write(f"True rumor (T): {stats['train']['True rumor (T)'] + stats['val']['True rumor (T)'] + stats['test']['True rumor (T)']}\n")
#         f.write(f"Unverified rumor (U): {stats['train']['Unverified rumor (U)'] + stats['val']['Unverified rumor (U)'] + stats['test']['Unverified rumor (U)']}\n")
def write_countdataset(data, y_train, y_dev, y_test, stats, cleaned_total_users, cleaned_total_comments):
    """
    输出数据集统计信息到文件
    
    Args:
        data: 数据集名称
        y_train, y_dev, y_test: 标签列表
        stats: 统计字典，包含各类别分布
        cleaned_total_users: 清洗后的用户总数
        cleaned_total_comments: 清洗后的评论总数
    """
    DATASET_ORIGINAL_STATS = {
        'twitter15': {
            'source_posts': 1490,
            'users': 276663,
            'comments': 330122
        },
        'twitter16': {
            'source_posts': 818,  # Twitter16 原始源推文数（需要确认）
            'users': 173487,
            'comments': 204002
        },
        'weibo': {
            'source_posts': 4664,  # Weibo 原始源推文数（需要确认）
            'users': 2746818,
            'comments': 3800992
        }}
    
    cleaned_source_posts = len(y_train) + len(y_dev) + len(y_test)
    if data in DATASET_ORIGINAL_STATS:
        original = DATASET_ORIGINAL_STATS[data]
        original_source_posts = original['source_posts']
        original_users = original['users']
        original_comments = original['comments']
        
        retention_ratio = cleaned_source_posts / original_source_posts
        estimated_users = int(original_users * retention_ratio)
        estimated_comments = int(original_comments * retention_ratio)
        
    
    with open(os.path.join(LOG_DIR, f"{data}_statistics.txt"), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"数据集统计报告 - {data}\n")
        f.write("="*60 + "\n\n")
        
        # 基本统计
        f.write(f"源推文总数 (Source Posts): {cleaned_source_posts}\n")
        f.write(f"用户总数 (Users): {estimated_users}\n")
        f.write(f"评论总数 (Comments): {estimated_comments}\n")
        
        # 数据划分
        f.write(f"训练集大小: {len(y_train)}\n")
        f.write(f"验证集大小: {len(y_dev)}\n")
        f.write(f"测试集大小: {len(y_test)}\n\n")
        
        # 类别分布
        f.write("类别分布 (Class Distribution):\n")
        f.write("-"*40 + "\n")
        f.write(f"Non-rumor (N): {stats['train']['Non-rumor (N)'] + stats['val']['Non-rumor (N)'] + stats['test']['Non-rumor (N)']}\n")
        f.write(f"False rumor (F): {stats['train']['False rumor (F)'] + stats['val']['False rumor (F)'] + stats['test']['False rumor (F)']}\n")
        f.write(f"True rumor (T): {stats['train']['True rumor (T)'] + stats['val']['True rumor (T)'] + stats['test']['True rumor (T)']}\n")
        f.write(f"Unverified rumor (U): {stats['train']['Unverified rumor (U)'] + stats['val']['Unverified rumor (U)'] + stats['test']['Unverified rumor (U)']}\n")
        f.write("-"*40 + "\n")
        f.write(f"总计: {cleaned_source_posts}\n")
        f.write("="*60 + "\n")
    
    # 同时在控制台打印
    print("\n" + "="*60)
    print(f"数据集统计已保存到: {os.path.join(LOG_DIR, f'{data}_statistics.txt')}")
    print("="*60)
    print(f"源推文总数 (Source Posts): {cleaned_source_posts}")
    print(f"用户总数 (Users): {cleaned_total_users}")
    print(f"评论总数 (Comments): {cleaned_total_comments}")
    print("="*60)