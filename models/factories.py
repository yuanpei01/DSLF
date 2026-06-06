import importlib
from dataset_config import DATASET_CONFIG
from pipeline_config import Method, glan_module_name,glan_dropS_module_name,glan_dropC_module_name


def create_glan(method, config, graph):
    _method = Method.COS if method == "cos" else Method.THRED
    _glan_mod = importlib.import_module(f"Stream1.model.GGLAN.{glan_module_name(_method)}") #这里是重点 从哪个文件加载的模型先看thred版本的
    GLAN = _glan_mod.GLAN
    return GLAN(config, graph)

def create_glan_dropS(method, config, graph):
    _method = Method.COS if method == "cos" else Method.THRED
    _glan_mod = importlib.import_module(f"Stream1.model.GGLAN.{glan_dropS_module_name(_method)}") #这里是重点 从哪个文件加载的模型先看thred版本的
    GLAN = _glan_mod.GLAN
    return GLAN(config, graph)

def create_glan_dropC(method, config, graph):
    _method = Method.COS if method == "cos" else Method.THRED
    _glan_mod = importlib.import_module(f"Stream1.model.GGLAN.{glan_dropC_module_name(_method)}") #这里是重点 从哪个文件加载的模型先看thred版本的
    GLAN = _glan_mod.GLAN
    return GLAN(config, graph)

# def create_glan_and_module(method, config, graph):
#     _method = Method.COS if method == "cos" else Method.THRED
#     _glan_mod = importlib.import_module(f"Stream1.model.GGLAN.{glan_module_name(_method)}")
#     GLAN = _glan_mod.GLAN
#     return GLAN(config, graph), _method


def create_bigcn(data, in_feats=5000, hid_feats=64, out_feats=64):
    cfg = DATASET_CONFIG[data]
    mod = importlib.import_module(cfg["bigcn_model"])
    Net = getattr(mod, "Net")
    return Net(in_feats, hid_feats, out_feats)


def load_glan_checkpoint(model, checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if "user_tweet_embedding.weight" in state_dict:
        del state_dict["user_tweet_embedding.weight"]
    if "word_embedding.weight" in state_dict:
        del state_dict["word_embedding.weight"]
    model.load_state_dict(state_dict, strict=False)
    return model

def load_glan_dropS_checkpoint(model, checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # 删除不匹配的键
    if "user_tweet_embedding.weight" in state_dict:
        del state_dict["user_tweet_embedding.weight"]
    if "word_embedding.weight" in state_dict:
        del state_dict["word_embedding.weight"]
    
    # 处理 fc_out.0.weight 维度不匹配
    if "fc_out.0.weight" in state_dict:
        old_weight = state_dict["fc_out.0.weight"]  # shape: [300, 600]
        # 只取前300维（对应原来的 X_local 部分）或后300维（对应 X_global 部分）
        # 这里选择取后300维（X_global）
        new_weight = old_weight[:, 300:]  # [300, 300]
        state_dict["fc_out.0.weight"] = new_weight
        
        # 注意：bias 维度不变，直接保留
        # "fc_out.0.bias" 依然是 [300]
    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_glan_dropC_checkpoint(model, checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # 删除不匹配的键
    if "user_tweet_embedding.weight" in state_dict:
        del state_dict["user_tweet_embedding.weight"]
    if "word_embedding.weight" in state_dict:
        del state_dict["word_embedding.weight"]
    
    # 处理 fc_out.0.weight 维度不匹配
    if "fc_out.0.weight" in state_dict:
        old_weight = state_dict["fc_out.0.weight"]  # shape: [300, 600]
        # 只取前300维（对应原来的 X_local 部分）或后300维（对应 X_global 部分）
        # 这里选择取前300维（X_local）
        new_weight = old_weight[:, :300]  # [300, 300]
        state_dict["fc_out.0.weight"] = new_weight
        
        # 注意：bias 维度不变，直接保留
        # "fc_out.0.bias" 依然是 [300]
    
    model.load_state_dict(state_dict, strict=False)
    return model