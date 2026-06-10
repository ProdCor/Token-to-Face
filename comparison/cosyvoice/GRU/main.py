import argparse
import os
import pickle
import shutil
import sys

import pandas as pd
import torch
import numpy as np

sys.path.append("../")
from dataloader_adapt import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm

from model_adapt import FaceDiffBeatCosyVoice2
from utils_adapt import plot_losses, create_gaussian_diffusion, get_arkit_mask


def trainer_diff(args, train_loader, dev_loader, model, diffusion, optimizer, 
                 epoch=100, device="cuda", blendshape_mask=None):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0

    for e in range(epoch + 1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            
            # Load vertices
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            if blendshape_mask is not None:
                mask_np = blendshape_mask.cpu().numpy()
                vertice = vertice * mask_np
            vertice = torch.from_numpy(vertice)

            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            # CHANGED: Handle CosyVoice2 tokens
            if args.use_cosyvoice_tokens:
                # audio is already LongTensor of token indices
                audio = audio.to(device=device, dtype=torch.long)
            else:
                # Original: audio is FloatTensor of HuBERT features
                audio = audio.to(device=device, dtype=torch.float)
            
            vertice = vertice.to(device=device)
            template = template.to(device=device)
            one_hot = one_hot.to(device=device)

            loss = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )['loss']

            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())
            
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))

        train_losses.append(np.mean(loss_log))

        valid_loss_log = []
        model.eval()
        
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # To gpu
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            if blendshape_mask is not None:
                mask_np = blendshape_mask.cpu().numpy()
                vertice = vertice * mask_np
            vertice = torch.from_numpy(vertice)

            # For vocaset reduce the frame rate from 60 to 30
            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

            train_subject = file_name[0].split("_")[0]
            
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]

                loss = diffusion.training_losses(
                    model,
                    x_start=vertice,
                    t=t,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    }
                )['loss']

                loss = torch.mean(loss)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter, :]
                    loss = diffusion.training_losses(
                        model,
                        x_start=vertice,
                        t=t,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        }
                    )['loss']

                    loss = torch.mean(loss)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        val_losses.append(current_loss)
        
        # Save model checkpoints
        if e == args.max_epoch or e % 25 == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{args.dataset}_{e}.pth'))
            plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))
        
        print("epoch: {}, current loss:{:.8f}".format(e + 1, current_loss))

    plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))

    return model

@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, f'{args.model}_{args.dataset}_{epoch}.pth')))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        
        if args.dataset == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)

        audio, vertice = audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        train_subject = file_name[0].split("_")[0]
        vertice_path = os.path.split(vertice_path)[-1][:-4]
        print(vertice_path)

        # For BEAT2, handle test split specially
        if args.dataset in ['beat', 'beat2']:
            # For BEAT/BEAT2, we don't use subject conditioning
            # Just use the first one-hot (they're all the same anyway)
            one_hot = one_hot_all[:, 0, :]
            one_hot = one_hot.to(device=device)

            for sample_idx in range(1, args.num_samples + 1):
                sample = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=args.skip_steps,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                sample = sample.squeeze()
                sample = sample.detach().cpu().numpy()

                out_path = f"{vertice_path}.npy"
                
                if 'damm' in args.dataset:
                    sample = RIG_SCALER.inverse_transform(sample)
                    np.save(os.path.join(args.result_path, out_path), sample)
                    df = pd.DataFrame(sample)
                    df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path, out_path), sample)

        else:
            # Original logic for other datasets
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]
                one_hot = one_hot.to(device=device)

                for sample_idx in range(1, args.num_samples + 1):
                    sample = diffusion.p_sample_loop(
                        model,
                        shape,
                        clip_denoised=False,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        },
                        skip_timesteps=args.skip_steps,
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        device=device
                    )
                    sample = sample.squeeze()
                    sample = sample.detach().cpu().numpy()

                    if args.num_samples != 1:
                        out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
                    else:
                        out_path = f"{vertice_path}_condition_{condition_subject}.npy"
                    
                    if 'damm' in args.dataset:
                        sample = RIG_SCALER.inverse_transform(sample)
                        np.save(os.path.join(args.result_path, out_path), sample)
                        df = pd.DataFrame(sample)
                        df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                    else:
                        np.save(os.path.join(args.result_path, out_path), sample)

            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:, iter, :]
                    one_hot = one_hot.to(device=device)

                    sample_cond = diffusion.p_sample_loop(
                        model,
                        shape,
                        clip_denoised=False,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        },
                        skip_timesteps=args.skip_steps,
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        device=device
                    )
                    prediction_cond = sample_cond.squeeze()
                    prediction_cond = prediction_cond.detach().cpu().numpy()

                    prediction = prediction_cond
                    
                    if 'damm' in args.dataset:
                        prediction = RIG_SCALER.inverse_transform(prediction)
                        df = pd.DataFrame(prediction)
                        df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                    else:
                        np.save(os.path.join(args.result_path, f"{vertice_path}_condition_{condition_subject}.npy"), prediction)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="beat2", help='Name of the dataset folder. eg: BIWI, beat2')
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--vertice_dim", type=int, default=51, help='number of vertices - 23370*3 for BIWI, 103 for BEAT2 FLAME')
    parser.add_argument("--feature_dim", type=int, default=256, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=256, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder number of layers')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy_arkit", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates.pkl",
                        help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save/cben_100ep_GRU_COSY", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="train")
    parser.add_argument("--val_subjects", type=str, default="val")
    parser.add_argument("--test_subjects", type=str, default="test")
    parser.add_argument("--output_fps", type=int, default=30,
                        help='fps of the visual data, BIWI was captured in 25 fps, BEAT2 is 30 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=1, help='number of samples to generate per audio')
    parser.add_argument("--use_cosyvoice_tokens", action='store_true',
                    help='Use pre-extracted CosyVoice2 tokens instead of HuBERT')
    parser.add_argument("--token_path", type=str, default="train_split_tokens/splits/",
                        help='Path to directory containing token .pt files')
    parser.add_argument("--token_embedding_dim", type=int, default=2*768,
                        help='Embedding dimension for CosyVoice2 tokens')
    parser.add_argument("--dropout", type=float, default=0.3,
                    help='Dropout rate for regularization (default: 0.3)')
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='Weight decay (L2 penalty) for optimizer (default: 0.01)')
    parser.add_argument("--use_masking", action='store_true',
                    help='Apply blendshape masking (jaw+mouth only)')
    parser.add_argument("--mask_categories", type=str, nargs='+', 
                        default=['jaw', 'mouth'],
                        help='ARKit categories to train on')
    args = parser.parse_args()

    # Validate CUDA availability
    assert torch.cuda.is_available(), "CUDA is required for training"
    
    # Create save and result directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Vertex Dimension: {args.vertice_dim}")
    print(f"Feature Dimension: {args.feature_dim}")
    print(f"GRU Dimension: {args.gru_dim}")
    print(f"GRU Layers: {args.gru_layers}")
    print(f"Output FPS: {args.output_fps}")
    print(f"Diffusion Steps: {args.diff_steps}")
    print(f"Max Epochs: {args.max_epoch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Train Subjects: {args.train_subjects}")
    print(f"Val Subjects: {args.val_subjects}")
    print(f"Test Subjects: {args.test_subjects}")
    print("="*60 + "\n")

    # Create diffusion
    diffusion = create_gaussian_diffusion(args)

    # Select model based on dataset
    if 'damm' in args.dataset:
        print("Using FaceDiffDamm model")
        model = FaceDiffDamm(args)
    elif args.dataset == 'beat2':
            if args.use_cosyvoice_tokens:
                model = FaceDiffBeatCosyVoice2(
                    args,
                    vertice_dim=args.vertice_dim,
                    latent_dim=args.feature_dim,
                    diffusion_steps=args.diff_steps,
                    gru_latent_dim=args.gru_dim,
                    num_layers=args.gru_layers,
                    dropout=args.dropout,
                )
    elif args.dataset == 'beat':
        print("Using FaceDiffBeat model (for original BEAT dataset)")
        model = FaceDiffBeat(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )
    else:
        print("Using FaceDiff model")
        model = FaceDiff(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Move model to device
    cuda = torch.device(args.device)
    model = model.to(cuda)

    # Load data
    print("\nLoading datasets...")
    dataset = get_dataloaders(args)
    
    optimizer = torch.optim.AdamW(  
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay  
    )

    if args.use_masking:
        blendshape_mask = get_arkit_mask(
            categories=args.mask_categories,
            as_tensor=True
        ).to(args.device)
        print(f"\n Using blendshape masking: {blendshape_mask.sum()}/51 active")
    else:
        blendshape_mask = None
        print("\n No masking - all 51 blendshapes")

    # Train model
    print("\nStarting training...\n")
    model = trainer_diff(
        args, 
        dataset["train"], 
        dataset["valid"], 
        model, 
        diffusion, 
        optimizer,
        epoch=args.max_epoch, 
        device=args.device,
        blendshape_mask=blendshape_mask
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {args.save_path}")
    print(f"Results saved to: {args.result_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()