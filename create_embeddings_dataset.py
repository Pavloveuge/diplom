import argparse 
from transformers import Wav2Vec2FeatureExtractor
import json
import pandas as pd
import soundfile as sf
from datasets import Dataset
import torch
from tqdm import tqdm




def main(args):
    with open(args.path_to_manifest) as f:
        manifest = [json.loads(line) for line in f]

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    print(feature_extractor)
    
    # дописать построение эмбеддингов для текста
    for ind_start in tqdm(range(0, len(manifest), args.batch_size)):
        batch = manifest[ind_start: ind_start + args.batch_size]
        raw_speech = [sf.read(sample["audio_filepath"])[0] for sample in batch]
        features = feature_extractor(raw_speech=raw_speech, padding="do_not_pad", sampling_rate=16000)['input_values']
        
        for ind in range(len(batch)):
            torch.save(features[ind], batch[ind]["audio_filepath"].replace(".flac", ".pt"))
        
    new_data = [
        {
            "audio_emb_path": sample['audio_filepath'].replace(".flac", ".pt"),
            "duration": sample['duration'],
            "text": sample['text']
        }
        for sample in manifest
    ]

    json_object = json.dumps(new_data, indent=4)
    with open(args.path_to_new_manifest, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My example explanation')
    parser.add_argument(
        '--path_to_manifest',
        type=str,
        help='path to mainfest in NEMO style'
    )
    parser.add_argument(
        '--path_to_new_manifest',
        type=str,
        help='path to manifest with results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="batch size for embedding model"
    )
    args = parser.parse_args()

    main(args)