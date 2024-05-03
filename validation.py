import soundfile as sf
import json
import torch

from tqdm import tqdm
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from text_vae import BARTForConditionalGenerationLatent
from transformers import AutoTokenizer
from diffusers import DDPMScheduler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput
from hyperpyyaml import load_hyperpyyaml

from model import NoisePredictor


sample = {
        "audio_filepath": "/diplom/LibriSpeech/train-clean-100/446/123502/446-123502-0000.flac",
        "text_emb_path": "/diplom/LibriSpeech/train-clean-100/446/123502/446-123502-0000_ref_emb.pt",
        "duration": 14.645,
        "text": "CHAPTER FOUR A PRISONER WE HAD GONE PERHAPS TEN MILES WHEN THE GROUND BEGAN TO RISE VERY RAPIDLY WE WERE AS I WAS LATER TO LEARN NEARING THE EDGE OF ONE OF MARS LONG DEAD SEAS IN THE BOTTOM OF WHICH\n"
    }

# читаем аудио

audio = [sf.read(sample["audio_filepath"])[0]]

model_name = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
audio_embedder = Wav2Vec2Model.from_pretrained(model_name).to("cuda")

features = feature_extractor(raw_speech=audio, padding="longest", sampling_rate=16000, return_tensors="pt")['input_values']

features = features.to("cuda")
with torch.no_grad():
    audio_embedding = audio_embedder(features).last_hidden_state.detach()

del feature_extractor
del audio_embedder


# деноизим

config = load_hyperpyyaml(open("/diplom/diplom/config.yaml", "r"))
path_to_noise_predictor_state_dict = "/diplom/diplom/checkpoints/noise_predictor_epoch_65"
model_weights = torch.load(path_to_noise_predictor_state_dict)
noise_predictor = NoisePredictor(config)
noise_predictor.load_state_dict(model_weights)

noise_predictor = noise_predictor.to("cuda")
noise_predictor.training = False
x = torch.rand((1, 32, 64)).to("cuda")
mask = torch.ones(1, 32, dtype=torch.bool).to("cuda")
noise_schedule = DDPMScheduler(num_train_timesteps=200)
noise_schedule.set_timesteps(200)

for t in tqdm(noise_schedule.timesteps):
    t = torch.tensor([t]).to("cuda")
    with torch.no_grad():
        noise_predict = noise_predictor(x=x, t=t, cond_emb=audio_embedding, mask=mask)
    #print(noise_predict.device, t.device, x.device)
    x = noise_schedule.step(noise_predict.detach().cpu(), t.detach().cpu(), x.detach().cpu()).prev_sample.to("cuda")
    


# кастим в текса
path_to_dir_with_text_vae = "/diplom/diplom/latent-diffusion-for-language/saved_latent_models/librispeech/2024-03-18_09-10-09"

text_vae_args = json.load(open(path_to_dir_with_text_vae + "/args.json"))
bart_model = BartForConditionalGeneration.from_pretrained(
            text_vae_args['enc_dec_model'])
config = bart_model.config

lm = BARTForConditionalGenerationLatent.from_pretrained(
    text_vae_args['enc_dec_model'], config=config, num_encoder_latents=text_vae_args['num_encoder_latents'], 
    num_decoder_latents=text_vae_args['num_decoder_latents'], dim_ae=text_vae_args['dim_ae'], num_layers=text_vae_args['num_layers'], 
    l2_normalize_latents=text_vae_args['l2_normalize_latents'], _fast_init=False)

model_weights = torch.load(path_to_dir_with_text_vae + "/model.pt")
lm.load_state_dict(model_weights['model'])

lm = lm.to("cuda")
lm.eval()

bart_model = bart_model.to("cuda")
bart_model.eval()

tokenizer = AutoTokenizer.from_pretrained(text_vae_args['enc_dec_model'])
encoder_output = BaseModelOutput(last_hidden_state=lm.get_decoder_input(x.clone()))
sample_ids = bart_model.generate(encoder_outputs=encoder_output, attention_mask=None)

print(sample_ids)
print(tokenizer.decode(sample_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
print(sample['text'])
