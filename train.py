import argparse
import time, os
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from concurrent.futures import ProcessPoolExecutor
import queue

from model_data_utils import PDBDataset, load_pdb_pt, worker_init_fn, \
        get_pdbs, batch_featurize, get_std_opt, Batches
from model import loss_smoothed, loss_nll, ProteinMPNN


def main(args):
    scaler = GradScaler()
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolder = os.path.join(base_folder, 'model_weights')
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    PATH = args.previous_checkpoint

    logfile = os.path.join(base_folder, 'log.txt')
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    data_path = args.path_for_training_data

    train_data_path = os.path.join(data_path, "train")
    train = list(filter(lambda x: x.endswith('.pt'), os.listdir(train_data_path)))
    train_set = PDBDataset(train, train_data_path, load_pdb_pt)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn)

    valid_data_path = os.path.join(data_path, "valid")
    valid = list(filter(lambda x: x.endswith('.pt'), os.listdir(valid_data_path)))
    valid_set = PDBDataset(valid, valid_data_path, load_pdb_pt)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn)

    model = ProteinMPNN(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        augment_eps=args.backbone_noise,
                        dropout=args.dropout,
                        model_type=args.model_type,
                        atom_context_num=args.atom_context_num,
                        device=device)
    model.to(device)
    
    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.model_type == "protein_mpnn":
        use_atom_context = False
        atom_context_num = 1
    elif args.model_type.startswith("ligand_mpnn"):
        use_atom_context = True
        atom_context_num = args.atom_context_num

    torch.multiprocessing.set_start_method('spawn')
    with ProcessPoolExecutor(max_workers=args.cpus_per_task) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for _ in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, args.max_protein_length, args.num_examples_per_epoch))

        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.

            if e % args.reload_data_every_n_epochs == 0:
                pdb_dict_train = q.get().result()
                pdb_dict_valid = p.get().result()
                train_batch_list = Batches(pdb_dict_train, max_length=args.max_protein_length, batch_size=args.batch_size)
                valid_batch_list = Batches(pdb_dict_valid, max_length=args.max_protein_length, batch_size=args.batch_size)
                if e > epoch:
                    q.put_nowait(executor.submit(get_pdbs, train_loader, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, args.max_protein_length, args.num_examples_per_epoch))

            for batch in train_batch_list:
                start_batch = time.time()
                batch_feature_dict = batch_featurize(batch, device, \
                        use_atom_context=use_atom_context, number_of_ligand_atoms=atom_context_num)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                S = batch_feature_dict["S"]
                mask_for_loss = batch_feature_dict["mask"]
                # mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(batch_feature_dict)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                    scaler.scale(loss_av_smoothed).backward()

                    if args.gradient_norm > 0.0:
                        total_norm = clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(batch_feature_dict)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for batch in valid_batch_list:
                    batch_feature_dict = batch_featurize(batch, device, \
                            use_atom_context=use_atom_context, number_of_ligand_atoms=atom_context_num)
                    log_probs = model(batch_feature_dict)
                    S = batch_feature_dict["S"]
                    mask_for_loss = batch_feature_dict["mask"]
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            
            checkpoint_filename_last = os.path.join(base_folder, 'model_weights', 'epoch_last.pt')
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'atom_context_num': args.atom_context_num,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = os.path.join(base_folder, 'model_weights', 'epoch{}_step{}.pt'.format(e+1, total_step))
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'atom_context_num': args.atom_context_num,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--cpus_per_task", type=int, default=12, help="number of CPUs allocated to the training task")
    argparser.add_argument("--path_for_training_data", type=str, default="training", help="path for loading training data")
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=32, help="number of neighbors for the sparse graph")
    argparser.add_argument("--model_type", type=str, choices=["protein_mpnn", "ligand_mpnn", "ligand_mpnn_new"], default="ligand_mpnn_new", help="model type")
    argparser.add_argument("--atom_context_num", type=int, default=25, help="number of context atom neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    # argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")

    args = argparser.parse_args()    
    main(args)
