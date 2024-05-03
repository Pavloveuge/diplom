import torch


def custom_collate_function(original_batch):
    max_trans_len = max([len(i['text_inds']) for i in original_batch])

    padded_emdedding_samples = []
    padded_text_samples = []
    for sample in original_batch:
        padded_text_samples.append(
            torch.nn.functional.pad(sample['text_inds'], (0, max_trans_len - len(sample['text_inds']))).reshape((1, -1))
        )
    
    return {
        "audio": [i['audio'] for i in original_batch], 
        "text_embedding": torch.cat([i['text_embedding'].unsqueeze(0) for i in original_batch], axis=0), 
        "text_inds": torch.cat(padded_text_samples, axis=0)
        }
        

