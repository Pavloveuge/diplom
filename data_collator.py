import torch


def custom_collate_function(original_batch):
    max_emb_len = max([len(i['embedding']) for i in original_batch])
    max_trans_len = max([len(i['text_inds']) for i in original_batch])

    padded_emdedding_samples = []
    padded_text_samples = []
    for sample in original_batch:
        padded_emdedding_samples.append(
            torch.nn.functional.pad(sample['embedding'], (0, max_emb_len - len(sample['embedding']))).reshape((1, -1))
        )
        padded_text_samples.append(
            torch.nn.functional.pad(sample['text_inds'], (0, max_trans_len - len(sample['text_inds']))).reshape((1, -1))
        )

    return torch.cat(padded_emdedding_samples, axis=0), torch.cat(padded_text_samples, axis=0)
        

