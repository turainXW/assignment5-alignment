import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    符合题目要求的实现
    """
    all_input_ids = []
    all_response_masks = []
    
    # 1. 分别编码并手动拼接 (Separate tokenization)
    for p_str, o_str in zip(prompt_strs, output_strs):
        # prompt 加上特殊符号，output 不加
        p_ids = tokenizer.encode(p_str, add_special_tokens=True)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        
        full_ids = p_ids + o_ids
        # 0 代表 prompt, 1 代表 output
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        
        all_input_ids.append(full_ids)
        all_response_masks.append(mask)

    # 2. 动态 Padding (补齐到 Batch 内最大长度)
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_ids = []
    padded_masks = []
    for ids, mask in zip(all_input_ids, all_response_masks):
        curr_len = len(ids)
        # 右填充
        padded_ids.append(ids + [pad_id] * (max_len - curr_len))
        padded_masks.append(mask + [0] * (max_len - curr_len))
    
    # 转为 tensor
    input_ids_tensor = torch.tensor(padded_ids)
    masks_tensor = torch.tensor(padded_masks)

    # 3. 按照题目要求进行 Shift 和 切片 (Crucial!)
    # 长度变为 max_len - 1
    # input_ids: 丢弃最后一个词
    # labels: 偏移 1 位，即丢弃第一个词
    # response_mask: 对应 labels，所以也丢弃第一个
    
    return {
        "input_ids": input_ids_tensor[:, :-1],
        "labels": input_ids_tensor[:, 1:],
        "response_mask": masks_tensor[:, 1:]
    }
