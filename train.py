import argparse
from hyperpyyaml import load_hyperpyyaml
from transformers import get_scheduler
from train_loop import train_loop
from torch.utils.data import DataLoader
from model import NoisePredictor
from text_vae import BARTForConditionalGenerationLatent
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def train(config):
    model = config['model'](cfg=config)
    optimizer = config['optimizer'](model.parameters())
    dataset = config['dataset']
    lr_scheduler = get_scheduler(name=config['lr_scheduler_type'], optimizer=optimizer, num_warmup_steps=config['warmup_steps'], 
    num_training_steps=len(dataset) * config['num_epochs'] // config['batch_size'])

    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=config['collate_fn'])

    model_name = "facebook/wav2vec2-base-960h"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    audio_embedder = Wav2Vec2Model.from_pretrained(model_name).to("cuda")
    
    train_loop(
        config=config,
        model=model,
        feature_extractor=feature_extractor,
        audio_embedder=audio_embedder,
        noise_scheduler=config['noise_schedule'],
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My example explanation')
    parser.add_argument(
        '--path_to_config',
        type=str,
        help='path to train config'
    )
    args = parser.parse_args()

    config = load_hyperpyyaml(open(args.path_to_config, "r"))

    train(config)
