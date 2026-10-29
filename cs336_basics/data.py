import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):

    if len(dataset) <= context_length:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens"
        )
    #randint不包含上界
    starts = np.random.randint(0,len(dataset)-context_length,size=batch_size)
    input_sequences = [dataset[start:start+context_length] for start in starts]
    target_sequences = [dataset[start+1:start+context_length+1] for start in starts]
    inputs_array = np.stack(input_sequences, axis=0)
    targets_array = np.stack(target_sequences, axis=0)
    # input_sequences = [
    #     np.array([10, 11, 12]),
    #     np.array([20, 21, 22]),
    # ]
    inputs = torch.tensor(
        inputs_array,
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        targets_array,
        dtype=torch.long,
        device=device,
    )
    return inputs,targets