import torch


def custom_collate_function(original_batch):
    max_emb_len = max([len(i['audio_embedding']) for i in original_batch])
    max_trans_len = max([len(i['text_inds']) for i in original_batch])

    padded_emdedding_samples = []
    padded_text_samples = []
    cond_padding_mask = []
    for sample in original_batch:
        padded_emdedding_samples.append(
            torch.nn.functional.pad(sample['audio_embedding'], (0, max_emb_len - len(sample['audio_embedding']), 0, 0))
        )
        cond_padding_mask.append(torch.cat([
            torch.ones(len(sample['audio_embedding'])),
            torch.zeros(max_emb_len - len(sample['audio_embedding']))
        ]))
        padded_text_samples.append(
            torch.nn.functional.pad(sample['text_inds'], (0, max_trans_len - len(sample['text_inds']))).reshape((1, -1))
        )
    cond_padding_mask = torch.vstack(cond_padding_mask)
    
    return {
        "audio_embedding": torch.cat([i.unsqueeze(0) for i in padded_emdedding_samples], axis=0), 
        "text_embedding": torch.cat([i['text_embedding'].unsqueeze(0) for i in original_batch], axis=0), 
        "text_inds": torch.cat(padded_text_samples, axis=0),
        "cond_padding_mask": cond_padding_mask
        }
        

