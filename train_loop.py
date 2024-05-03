import os
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from tqdm import tqdm


def train_loop(config, model, feature_extractor, audio_embedder, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'], 
        log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], "logs")
    )

    if accelerator.is_main_process:
        if config['output_dir'] is not None:
            os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler, feature_extractor, audio_embedder = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, feature_extractor, audio_embedder
    )
    
    model.to(accelerator.device, dtype=config['dtype'])
    audio_embedder.to(accelerator.device, dtype=config['dtype'])


    global_step = 0

    # Now you train the model
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            audio = batch['audio']
            text_embedding = batch['text_embedding']
            text_inds = batch['text_inds']

            audio = [i.cpu().numpy() for i in audio]
            features = feature_extractor(raw_speech=audio, padding="longest", sampling_rate=16000, return_tensors="pt")['input_values']

            features = features.to(accelerator.device)
            with torch.no_grad():
                audio_embedding = audio_embedder(features).last_hidden_state.detach()

            noise = torch.randn(text_embedding.shape).to(text_embedding.device)
            bs = audio_embedding.shape[0]
            mask = torch.ones(text_embedding.shape[0], config['num_encoder_latents'], dtype=torch.bool).to(accelerator.device)
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=text_embedding.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_text_embedding = noise_scheduler.add_noise(text_embedding, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(x=noisy_text_embedding, t=timesteps, cond_emb=audio_embedding, mask=mask)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            # в оригинале здесь валидация и сохранение чекпоинтов
            if (epoch % config['save_checkpoint_every_epoch'] == 0) and (epoch != 0):
                torch.save(model.state_dict(), f"{config['output_dir']}/noise_predictor_epoch_{epoch}")