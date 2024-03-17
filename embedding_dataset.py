import json
import torch

from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, text_processor, path_to_manifest: str):
        self.text_processor = text_processor
        self.manifest = json.load(open(path_to_manifest, "r"))

    def __getitem__(self, ind):
        # непонятно, что будет, со скоростью, если каждый раз читать с диска, может быть больно
        audio_embedding = torch.tensor(torch.load(self.manifest[ind]['audio_emb_path']))
        text_embedding = torch.tensor(torch.load(self.manifest[ind]['text_emb_path']))

        text_inds = torch.tensor(self.text_processor.encode(self.manifest[ind]['text']))


        return {
            "audio_embedding": audio_embedding,
            "text_embedding": text_embedding,
            "text_inds": text_inds
        }

    def __len__(self):
        return len(self.manifest)