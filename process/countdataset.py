# 在 cleandata 函数调用完成后，添加以下统计输出代码

def print_dataset_statistics(data_name, y_train, y_dev, y_test):
    import numpy as np
    """
    打印数据集的详细统计信息
    """
    # 定义类别映射（根据你的数据集调整）
    # 对于 Twitter15，通常是: 0=Non-rumor, 1=False rumor, 2=True rumor, 3=Unverified rumor
    class_names = ['Non-rumor (N)', 'False rumor (F)', 'True rumor (T)', 'Unverified rumor (U)']
    
    # 统计训练集
    train_counts = {}
    for i in range(len(class_names)):
        train_counts[class_names[i]] = np.sum(np.array(y_train) == i)
    
    # 统计验证集
    val_counts = {}
    for i in range(len(class_names)):
        val_counts[class_names[i]] = np.sum(np.array(y_dev) == i)
    
    # 统计测试集
    test_counts = {}
    for i in range(len(class_names)):
        test_counts[class_names[i]] = np.sum(np.array(y_test) == i)
    
    # 计算总数
    total_posts = len(y_train) + len(y_dev) + len(y_test)
    
    # 输出表格
    print("\n" + "="*80)
    print(f"数据集统计: {data_name}")
    print("="*80)
    print(f"{'类别':<25} {'训练集':<12} {'验证集':<12} {'测试集':<12} {'总计':<12}")
    print("-"*80)
    
    for class_name in class_names:
        train_count = train_counts[class_name]
        val_count = val_counts[class_name]
        test_count = test_counts[class_name]
        total = train_count + val_count + test_count
        print(f"{class_name:<25} {train_count:<12} {val_count:<12} {test_count:<12} {total:<12}")
    
    print("-"*80)
    print(f"{'总计 (总事件数)':<25} {len(y_train):<12} {len(y_dev):<12} {len(y_test):<12} {total_posts:<12}")
    print("="*80)
    
    # 额外统计信息（如果需要）
    print(f"\n补充统计信息:")
    print(f"源推文总数 (Source Posts): {total_posts}")
    print(f"训练集占比: {len(y_train)/total_posts*100:.1f}%")
    print(f"验证集占比: {len(y_dev)/total_posts*100:.1f}%")
    print(f"测试集占比: {len(y_test)/total_posts*100:.1f}%")
    
    return {
        'train': train_counts,
        'val': val_counts,
        'test': test_counts,
        'total': total_posts
    }

# # 在你的 cleandata 调用之后，添加这一行
# # 注意：你需要确保 y_train, y_dev, y_test 都已经转换为 numpy 数组或列表
# stats = print_dataset_statistics(data, y_train, y_dev, y_test)

# # 也可以将统计结果保存到文件
# with open(os.path.join(CHECKPOINT_DIR, f"{data}_statistics.txt"), 'w') as f:
#     f.write(f"Dataset: {data}\n")
#     f.write(f"Total source posts: {len(y_train) + len(y_dev) + len(y_test)}\n")
#     f.write(f"Training set: {len(y_train)}\n")
#     f.write(f"Validation set: {len(y_dev)}\n")
#     f.write(f"Test set: {len(y_test)}\n")
#     f.write("\nClass distribution:\n")
#     f.write(f"Non-rumor (N): {stats['train']['Non-rumor (N)'] + stats['val']['Non-rumor (N)'] + stats['test']['Non-rumor (N)']}\n")
#     f.write(f"False rumor (F): {stats['train']['False rumor (F)'] + stats['val']['False rumor (F)'] + stats['test']['False rumor (F)']}\n")
#     f.write(f"True rumor (T): {stats['train']['True rumor (T)'] + stats['val']['True rumor (T)'] + stats['test']['True rumor (T)']}\n")
#     f.write(f"Unverified rumor (U): {stats['train']['Unverified rumor (U)'] + stats['val']['Unverified rumor (U)'] + stats['test']['Unverified rumor (U)']}\n")