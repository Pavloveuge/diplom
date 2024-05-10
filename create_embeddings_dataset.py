import argparse 
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import json
import pandas as pd
import soundfile as sf
from datasets import Dataset
import torch
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from text_vae import BARTForConditionalGenerationLatent
from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from text_vae import BARTForConditionalGenerationLatent
from tqdm import tqdm




def main(args):
    with open(args.path_to_manifest) as f:
        manifest = [json.loads(line) for line in f]
    model_name = "facebook/wav2vec2-large"
    device = "cuda:" + args.device_number
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    
    for ind_start in tqdm(range(0, len(manifest), args.batch_size)):
        batch = manifest[ind_start: ind_start + args.batch_size]
        raw_speech = [sf.read(sample["audio_filepath"])[0] for sample in batch]
        features = feature_extractor(raw_speech=raw_speech, padding="longest", sampling_rate=16000, return_tensors="pt")['input_values']
        features = features.to(device)
        with torch.no_grad():
            embs = model(features).last_hidden_state.detach().cpu()
        for ind in range(len(batch)):
            torch.save(embs[ind].clone(), batch[ind]["audio_filepath"].replace(".flac", ".pt"))
    
    del feature_extractor
    del model

    new_data = [
        {
            "audio_emb_path": sample['audio_filepath'].replace(".flac", ".pt"),
            "duration": sample['duration'],
            "text": sample['text']
        }
        for sample in manifest
    ]

    # дописать построение эмбеддингов для текста

    text_vae_args = json.load(open(args.path_to_dir_with_text_vae + "/args.json"))
    config = BartForConditionalGeneration.from_pretrained(
                text_vae_args['enc_dec_model']).config

    lm = BARTForConditionalGenerationLatent.from_pretrained(
        text_vae_args['enc_dec_model'], config=config, num_encoder_latents=text_vae_args['num_encoder_latents'], 
        num_decoder_latents=text_vae_args['num_decoder_latents'], dim_ae=text_vae_args['dim_ae'], num_layers=text_vae_args['num_layers'], 
        l2_normalize_latents=text_vae_args['l2_normalize_latents'], _fast_init=False)

    model_weights = torch.load(args.path_to_dir_with_text_vae + "/model.pt")
    lm.load_state_dict(model_weights['model'])

    lm = lm.to(device)
    lm.eval()

    tokenizer = AutoTokenizer.from_pretrained(text_vae_args['enc_dec_model'])

    for ind_start in tqdm(range(0, len(manifest), args.batch_size)):
        batch = manifest[ind_start: ind_start + args.batch_size]
        batch_refs = [sample['text'].lower().strip() for sample in batch]
        data = tokenizer(batch_refs, return_tensors="pt", max_length=text_vae_args['max_seq_len'], padding=True, truncation=True)
        data = data.to(device)


        with torch.inference_mode():
            encoder_outputs = lm.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
            results = lm.get_diffusion_latent(encoder_outputs, data['attention_mask']).detach().cpu()
        
        for ind in range(len(batch)):
            torch.save(results[ind].clone(), batch[ind]["audio_filepath"].replace(".flac", "") + "ref_emb.pt")
        
    new_data = [
        {
            "audio_emb_path": sample['audio_emb_path'],
            "text_emb_path": sample['audio_emb_path'].replace(".pt", "") + "ref_emb.pt",
            "duration": sample['duration'],
            "text": sample['text']
        }
        for sample in new_data
    ]

    json_object = json.dumps(new_data, indent=4)
    with open(args.path_to_new_manifest, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
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
        '--path_to_dir_with_text_vae',
        type=str,
        help='path to dir with checkpoint and config'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="batch size for embedding model"
    )
    parser.add_argument(
        '--device_number',
        type=str,
        default="0",
        help="device number for concat with cuda:"
    )
    args = parser.parse_args()

    main(args)