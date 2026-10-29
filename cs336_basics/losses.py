import torch



def cross_entropy(logits:torch.Tensor, targets:torch.Tensor) ->torch.Tensor :
    shifted = logits - logits.max(dim=-1,keepdim=True).values
    targets_scores = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    # shifted是减去最大值的logits targets：形状 (2,)  → [1, 2]
    #                          index：形状 (2, 1)  → [[1],
    #                                               [2]]
    #targets.unsqueeze(-1),增加一个维度
    #gather按照targets沿着-1维度选出类别
    log_denominator = torch.log(torch.exp(shifted).sum(dim=-1))
    losses = log_denominator - targets_scores
    return losses.mean()
