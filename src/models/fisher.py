import copy
import itertools
import os

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial


from tqdm.auto import tqdm

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.modeling import ImageClassifier
from src.models import utils
from src.models.utils import LabelSmoothing, fisher_save

import src.datasets as datasets


def save_activations(
    saved_activations,
    module_name,
    module,
    input,
    output,
) -> None:
    """PyTorch Forward hook to save inputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    saved_activations[module_name] = input[0].float()


def save_gradients(
    saved_gradients,
    module_name,
    module,
    grad_input,
    grad_output,
) -> None:
    """PyTorch Backward hook to save inputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    saved_gradients[module_name] = grad_output[0].float()



###############################################################################
# TODO: Make these args
#######################################
_TRAIN_PREPROCESSING = False
# Set this to a positive integer for faster testing.
_N_EXAMPLES_PER_EPOCH = None
###############################################################################


def compute_fisher(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    # assert args.fisher is not None, "Please provide a path to save the Fisher to through --fisher."

    # save_path, = args.fisher
    # save_path = os.path.expanduser(save_path)
    save_path = args.load.replace(".pt", f"_{args.train_dataset}_fisher.pt")
    assert save_path != args.load

    # Copy the args so we can force the batch size to be 1 without affecting
    # other parts of the code base.
    args = copy.deepcopy(args)
    args.batch_size = 1

    model = ImageClassifier.load(os.path.expanduser(args.load))
    model.process_images = True

    if _TRAIN_PREPROCESSING:
        preprocess_fn = model.train_preprocess
    else:
        preprocess_fn = model.val_preprocess

    input_key = 'images'

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        # TODO: See if this needs to be set to 1.
        batch_size=args.batch_size
    )

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # NOTE: Not sure if label smoothing makes sense for Fisher
    # computation.
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the Fisher accumulators.
    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p.data)
        p.grad_counter = 0

    
    model.train()
    data_loader = get_dataloader(
        dataset, is_train=_TRAIN_PREPROCESSING, args=args, image_encoder=None)

    data_loader = itertools.islice(data_loader, 1000)    

    for i, batch in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        logits = utils.get_logits(inputs, model)
        target = torch.multinomial(F.softmax(logits, dim=-1), 1).detach().view(-1)
        loss = loss_fn(logits, target)

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad2_acc += p.grad.data ** 2
                p.grad_counter += 1

    fisher = {}

    for name, p in model.named_parameters():
        if name.startswith('module.'):
            name = name[len('module.'):]
        if p.grad_counter == 0:
            print(f'No gradients found for parameter: {name}')
            del p.grad2_acc
        else:
            p.grad2_acc /= p.grad_counter
            fisher[name] = p.grad2_acc

    fisher_save(fisher, save_path)

def compute_covariance(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
  
    inputGramMatrix_savePath = args.load.replace(".pt", f"_{args.train_dataset}_input_gram_matrix.pt")
    outputGramMatrix_savePath = args.load.replace(".pt", f"_{args.train_dataset}_output_gram_matrix.pt")
    assert inputGramMatrix_savePath != args.load
    assert outputGramMatrix_savePath != args.load


    # Copy the args so we can force the batch size to be 1 without affecting
    # other parts of the code base.
    args = copy.deepcopy(args)
    args.batch_size = 1

    model = ImageClassifier.load(os.path.expanduser(args.load))
    model.process_images = True

    if _TRAIN_PREPROCESSING:
        preprocess_fn = model.train_preprocess
    else:
        preprocess_fn = model.val_preprocess

    input_key = 'images'

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        # TODO: See if this needs to be set to 1.
        batch_size=args.batch_size
    )

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # NOTE: Not sure if label smoothing makes sense for Fisher
    # computation.
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    def computeGramMatrix_forBERT(module_name, matrix, mask):
            if len(matrix.shape) > 2:
                # [batch_size * num_tokens, input_dim]
                masked_activations = (
                    matrix.flatten(0, 1) * mask.flatten(0, 1).to(matrix.device)[:, None]
                )
            else:
                masked_activations = matrix
            return torch.matmul(masked_activations.T, masked_activations)

    stored_inputActivations = {}
    stored_outputActivationsGradients = {}
    new_model = copy.deepcopy(model)
    for module_name, module in new_model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(
                partial(
                    save_activations,
                    stored_inputActivations,
                    module_name,
                )
            )
            # output activation for lm_head is vocab_size, which is too large to store
            if "lm_head" not in module_name:
                module.register_full_backward_hook(
                    partial(
                        save_gradients,
                        stored_outputActivationsGradients,
                        module_name,
                    )
                )

    inputActivations_gramMatrix = {}
    outputActivationGradient_gramMatrix = {}
   
    model.train()
    data_loader = get_dataloader(
        dataset, is_train=_TRAIN_PREPROCESSING, args=args, image_encoder=None)

    data_loader = itertools.islice(data_loader, 1000)    

    for i, batch in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        logits = utils.get_logits(inputs, model)
        target = torch.multinomial(F.softmax(logits, dim=-1), 1).detach().view(-1)
        loss = loss_fn(logits, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for module_name, activations in stored_inputActivations.items():
                mask = inputs["attention_mask"]

                gram_matrix = computeGramMatrix_forBERT(module_name, activations, mask)
                if module_name not in inputActivations_gramMatrix:
                    inputActivations_gramMatrix[module_name] = gram_matrix.detach()
                else:
                    inputActivations_gramMatrix[module_name] += gram_matrix.detach()

                if "lm_head" not in module_name:
                    gradients = stored_outputActivationsGradients[module_name]
                    gram_matrix = computeGramMatrix_forBERT(
                        module_name, gradients, mask
                    )
                    if module_name not in outputActivationGradient_gramMatrix:
                        outputActivationGradient_gramMatrix[
                            module_name
                        ] = gram_matrix.detach()
                    else:
                        outputActivationGradient_gramMatrix[
                            module_name
                        ] += gram_matrix.detach()
    
    fisher_save(inputActivations_gramMatrix, save_path)
    fisher_save(outputActivationGradient_gramMatrix, save_path)


if __name__ == '__main__':
    args = parse_arguments()
    compute_fisher(args)
