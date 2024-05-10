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
        "audio_filepath": "./446-123502-0000.pt",
        "text_emb_path": "./446-123502-0000ref_emb.pt",
        "duration": 14.645,
        "text": "CHAPTER FOUR A PRISONER WE HAD GONE PERHAPS TEN MILES WHEN THE GROUND BEGAN TO RISE VERY RAPIDLY WE WERE AS I WAS LATER TO LEARN NEARING THE EDGE OF ONE OF MARS LONG DEAD SEAS IN THE BOTTOM OF WHICH\n"
    }

# читаем аудио

audio_embedding = torch.load(sample['audio_filepath']).reshape((1, -1, 768)).float().to("cuda")
print(audio_embedding.shape)

# деноизим

config = load_hyperpyyaml(open("/diplom/diplom/config.yaml", "r"))
path_to_noise_predictor_state_dict = "/diplom/diplom/first_exp_new_vae_batch_768/noise_predictor_epoch_1890"
model_weights = torch.load(path_to_noise_predictor_state_dict)
for key in list(model_weights.keys()):
    model_weights[key.replace("module.", "")] = model_weights[key]
    model_weights.pop(key)
#print(model_weights.keys())
noise_predictor = NoisePredictor(config)
noise_predictor.load_state_dict(model_weights)

noise_predictor.to("cuda")
noise_predictor.training = False
x = torch.rand((1, 32, 64)).to("cuda")
mask = torch.ones(1, 32, dtype=torch.bool).to("cuda")
noise_schedule = DDPMScheduler(num_train_timesteps=1000)
noise_schedule.set_timesteps(1000)
for t in tqdm(noise_schedule.timesteps):
    t = torch.tensor([t]).to("cuda")
    cond_padding_mask=torch.zeros(audio_embedding.shape[1]).reshape((1, -1)).bool().to("cuda")
    with torch.no_grad():
        noise_predict = noise_predictor(x=x, t=t, cond_emb=audio_embedding, mask=mask, cond_padding_mask=cond_padding_mask)
    #print(noise_predict.device, t.device, x.device)
    x = noise_schedule.step(noise_predict.detach().cpu(), t.detach().cpu(), x.detach().cpu()).prev_sample.to("cuda")
    


# кастим в текса
path_to_dir_with_text_vae = "/train_vae/diplom/latent-diffusion-for-language/saved_latent_models/librispeech/2024-05-05_22-24-35"

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
sample_ids = bart_model.generate(encoder_outputs=encoder_output, attention_mask=None, max_length=200)

print(tokenizer.decode(sample_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
print(sample['text'])