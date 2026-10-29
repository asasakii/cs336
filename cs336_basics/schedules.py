import math

#动态学习率
def get_lr_cosine_schedule(
        it,
        max_learning_rate,
        min_learning_rate,
        warmup_iters,
        cosine_cycle_iters,
    ):
        if it < warmup_iters:
            return max_learning_rate * it /warmup_iters
        elif it <=cosine_cycle_iters:
            progress = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
            cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
            return min_learning_rate + cosine_weight*(max_learning_rate - min_learning_rate)
        else:
            return min_learning_rate