import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings
from llm_attacks import get_workers

import pdb
import operator

import multiprocessing

from configs.template import get_config as default_config

from ml_collections import config_flags

# Read the configuration file
_CONFIG = default_config()

import random

import threading
import time
from queue import Queue

from scipy.stats import spearmanr
from scipy.stats import rankdata

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    return one_hot.grad.clone()

    
class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice
        )


class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices


        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def small_work_all(self, control_cands, verbose, batch_size, target_weight, control_weight, result_queue):
        main_device = self.models[0].device
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)

        for j, cand in enumerate(control_cands):
            progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
            for i in progress:
                for k, worker in enumerate(self.small_workers):
                    worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                
                logits, ids = zip(*[worker.results.get() for worker in self.small_workers])

                loss[j*batch_size:(j+1)*batch_size] += sum([
                    target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                    for k, (logit, id) in enumerate(zip(logits, ids))
                ])
                if control_weight != 0:
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                del logits, ids ; gc.collect()
                
                if verbose:
                    progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
            
        result_queue.put(['small',loss])
        

    def big_work_random(self, control_cands, verbose, probe_set, batch_size, target_weight, control_weight, result_queue):
        random_number = probe_set
        random_index = random.sample(range(batch_size), random_number)
        random_control_cands = [[operator.itemgetter(i)(control_cands[0]) for i in random_index]]
        batch_size =random_number
        main_device = self.models[0].device
        loss_large = torch.zeros(len(random_control_cands) * batch_size).to(main_device)


        for j, cand in enumerate(random_control_cands):
            progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
            for i in progress:
                for k, worker in enumerate(self.workers):
                    worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                logits, ids = zip(*[worker.results.get() for worker in self.workers])
                loss_large[j*batch_size:(j+1)*batch_size] += sum([
                    target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                    for k, (logit, id) in enumerate(zip(logits, ids))
                ])
                if control_weight != 0:
                    loss_large[j*batch_size:(j+1)*batch_size] += sum([
                        control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                del logits, ids ; gc.collect()
                
                if verbose:
                    progress.set_description(f"loss={loss_large[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")


        result_queue.put(['large', [loss_large, random_index, random_control_cands]])

    def kendall_tau(self, x, y):
        n = len(x)

        a = np.subtract.outer(x, x)
        b = np.subtract.outer(y, y)

        concordant = np.sum(np.logical_and(a > 0, b > 0))
        discordant = np.sum(np.logical_and(a < 0, b > 0)) + np.sum(np.logical_and(a > 0, b < 0))

        tau = (concordant - discordant) / np.sqrt((concordant + discordant) * (concordant + discordant + n * (n - 1) / 2))
        return tau 


    def goodman_kruskal_gamma(self, x, y):
        n = len(x)
        rx = rankdata(x)
        ry = rankdata(y)
        numerator = np.sum((rx - np.mean(rx)) * (ry - np.mean(ry)))
        denominator = np.sqrt(np.sum((rx - np.mean(rx))**2) * np.sum((ry - np.mean(ry))**2))
        gamma = numerator / denominator
        return gamma


    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True,
             probe_set = 64,
             filtered_set = 32,):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        
        with torch.no_grad():

            result_queue = Queue()

            thread1 = threading.Thread(target=self.small_work_all,  args=(control_cands, verbose, batch_size, target_weight, control_weight, result_queue))
            thread2 = threading.Thread(target=self.big_work_random, args=(control_cands,verbose, probe_set, batch_size, target_weight, control_weight, result_queue))
            
            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            results = []
            for _ in range(2):
                result = result_queue.get()
                results.append(result)
            
            if results[0][0]== 'small':
                loss = results[0][1]
                loss_large = results[1][1][0] 
                random_index = results[1][1][1]
                random_control_cands = results[1][1][2]
            else:
                loss = results[1][1]
                loss_large = results[0][1][0] 
                random_index = results[0][1][1]
                random_control_cands = results[0][1][2]

            print("Parallel completed")


            small_loss_random = [loss[i].item() for i in random_index]

            min_idx_random = loss_large.argmin()
            model_idx = min_idx_random // batch_size
            batch_idx = min_idx_random % batch_size
            next_control_random, cand_loss_random = random_control_cands[model_idx][batch_idx], loss_large[min_idx_random]

            large_loss_random = loss_large.tolist()


            # difference = (1 - np.corrcoef(small_loss_random, large_loss_random)[0,1])/2

            # difference = (1 - self.kendall_tau(small_loss_random, large_loss_random))/2

            # difference = (1 - self.goodman_kruskal_gamma(small_loss_random, large_loss_random))/2

            corr, _ = spearmanr(small_loss_random, large_loss_random)
            difference = (1 - corr)/2


            try:
                small_k = max(int(difference * filtered_set), 1)
            except:
                small_k = filtered_set
            _, indices = torch.topk(loss, k=small_k, largest=False)
            new_control_cands = [[operator.itemgetter(i.item())(control_cands[0]) for i in indices]]
            batch_size =small_k
            loss = torch.zeros(len(new_control_cands) * batch_size).to(main_device)


            for j, cand in enumerate(new_control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size

            if cand_loss_random < loss[min_idx]:
                next_control, cand_loss = next_control_random, cand_loss_random
            else:
                next_control, cand_loss = new_control_cands[model_idx][batch_idx], loss[min_idx]



               

        
        del control_cands, loss ; gc.collect()

        print('Sample Number:', small_k)

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
