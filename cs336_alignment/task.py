
import torch
import torch.nn.functional as F
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p=F.softmax(logits,dim=-1)
    h=-torch.sum(p*torch.log(p+1e-10),dim=-1)
    return h

import torch

def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    # 1. 获取 Logits (必须加 .logits)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # 形状: (batch, seq, vocab)

    # 2. 手动实现数值稳定的 LogSumExp
    # 减去最大值防止 exp 爆炸
    m = torch.max(logits, dim=-1, keepdim=True).values
    sum_exp = torch.sum(torch.exp(logits - m), dim=-1, keepdim=True)
    lse = m + torch.log(sum_exp) # 形状: (batch, seq, 1)

    # 3. 计算【全词表】的对数概率
    # 公式: log(P) = logits - LogSumExp(logits)
    all_log_probs = logits - lse  # 形状: (batch, seq, vocab)

    # 4. 【核心步骤】提取 labels 对应的对数概率
    # 使用 gather 函数：在最后一个维度上，根据 labels 提供的索引取值
    # labels: (B, S) -> unsqueeze -> (B, S, 1)
    # 提取后的形状是 (B, S, 1)，再 squeeze 变成 (B, S)
    log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    res = {"log_probs": log_probs}

    # 5. 计算熵 (使用计算好的 all_log_probs 最稳)
    if return_token_entropy:
        probs = torch.exp(all_log_probs)
        # H = -sum(p * log_p)
        token_entropy = -torch.sum(probs * all_log_probs, dim=-1)
        res["token_entropy"] = token_entropy

    return res


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor,normalize_constant: float, dim: int | None= None
)->torch.Tensor:
    masked_tensor=tensor.masked_fill(~mask, 0)
    if dim is not None:
        masked_sum = masked_tensor.sum(dim=dim)
    else:
        masked_sum = masked_tensor.sum()
    return masked_sum/normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    raw_loss=masked_normalize(-policy_log_probs,response_mask,normalize_constant,dim=1)
    loss=raw_loss.mean()/gradient_accumulation_steps
    loss.backward()
    metadata = {
        "loss": raw_loss.detach() 
    }

    return loss, metadata