import gc
import json
import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import time
import logging
from accelerate.utils import find_executable_batch_size
from attacks.utils import sample_control, get_nonascii_toks
from attacks.base_gcg import BaseGCG
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial


class GCG(BaseGCG):
    def __init__(self,
                 model,
                 tokenizer,
                 num_steps=500,
                 adv_string_init=" ! ! ! ! !",
                 search_width=512,
                 use_prefix_cache=True,
                 num_workers=None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.search_width = search_width
        self.search_batch_size = search_width
        self.not_allowed_tokens = get_nonascii_toks(tokenizer, device=self.device)
        self.use_prefix_cache = use_prefix_cache
        self.num_workers = num_workers or max(1, mp.cpu_count() // 2)

    def train_adv_string(self, data_dict, result_file, with_eos=False):
        """Original single-case training method"""
        model = self.model
        device = self.device
        tokenizer = self.tokenizer

        case_id = data_dict['CaseID']
        target = data_dict['Target'] + tokenizer.eos_token if with_eos else data_dict['Target']
        before_adv_string = data_dict['before_adv_string'] + ' '
        after_adv_string = data_dict['after_adv_string']

        logging.info(f'GCG || Case: {case_id} || Target: {target}')

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        search_width = self.search_width

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))

        # ========== setup optim_ids and cached components ========== #
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])

        cache_input_ids = tokenizer([tokenizer.bos_token + before_adv_string, after_adv_string, target], padding=False,
                                    add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids]
        before_adv_string_ids, after_adv_string_ids, target_ids = cache_input_ids
        before_adv_string_embeds, after_adv_string_embeds, target_embeds = [embed_layer(input_ids) for input_ids in
                                                                            cache_input_ids]

        if self.use_prefix_cache:
            with torch.no_grad():
                outputs = model(inputs_embeds=before_adv_string_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        visited_optim_ids = []
        for i in tqdm(range(num_steps)):
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            if self.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, after_adv_string_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat(
                    [before_adv_string_embeds, optim_embeds, after_adv_string_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)

            logits = outputs.logits

            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp - 1:-1, :].contiguous()
            shift_labels = target_ids
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256,
                                                 temp=1, not_allowed_tokens=self.not_allowed_tokens)

            # ========= Filter sampled_top_indices - PARALLELIZED ========= #
            new_sampled_top_indices = self._parallel_filter_candidates(
                sampled_top_indices,
                before_adv_string_ids[0],
                after_adv_string_ids[0],
                visited_optim_ids,
                tokenizer,
                device
            )

            count = len(sampled_top_indices) - len(new_sampled_top_indices)
            logging.info(f'Filtered {count} tokens from {len(sampled_top_indices)}')

            if len(new_sampled_top_indices) == 0:
                logging.info('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)

            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat([sampled_top_embeds,
                                          after_adv_string_embeds.repeat(new_search_width, 1, 1),
                                          target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds = torch.cat([before_adv_string_embeds.repeat(new_search_width, 1, 1),
                                          sampled_top_embeds,
                                          after_adv_string_embeds.repeat(new_search_width, 1, 1),
                                          target_embeds.repeat(new_search_width, 1, 1)], dim=1)

            loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds,
                                                                                                    target_ids)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()]
            visited_optim_ids.append(optim_ids)
            optim_ids = optim_ids.unsqueeze(0)

            input_ids = torch.cat([before_adv_string_ids, optim_ids, after_adv_string_ids], dim=1)
            decoded_text = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=False)
            decoded_adv_string = tokenizer.decode(optim_ids[0], clean_up_tokenization_spaces=False)
            logging.info(f'Step {i} || Loss: {loss.min().item()} || Adv String: {decoded_adv_string}')

            with open(result_file, 'a') as f:
                f.write(json.dumps({'step': i, 'loss': loss.min().item(), 'adv_string': decoded_adv_string,
                                    'input': decoded_text}) + '\n')

            del input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()

    def _parallel_filter_candidates(self, sampled_top_indices, before_ids, after_ids, visited_optim_ids, tokenizer,
                                    device):
        """Parallel filtering of candidate tokens"""

        def filter_single(j):
            input_ids = torch.cat([before_ids, sampled_top_indices[j], after_ids])
            decoded_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
            retokenized_ids = \
            tokenizer(decoded_text, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]

            if not torch.equal(input_ids, retokenized_ids) or any(
                    torch.equal(sampled_top_indices[j], v) for v in visited_optim_ids):
                return None
            return sampled_top_indices[j]

        # Use ThreadPoolExecutor for I/O-bound tokenization tasks
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(filter_single, range(len(sampled_top_indices))))

        return [r for r in results if r is not None]

    def train_adv_string_batch(self, data_dicts, result_files, with_eos=False):
        """
        Train multiple adversarial strings in parallel (multiple cases on same GPU).

        Args:
            data_dicts: List of data dictionaries
            result_files: List of result file paths
            with_eos: Whether to append EOS token
        """
        batch_size = len(data_dicts)
        model = self.model
        device = self.device
        tokenizer = self.tokenizer

        # Prepare batch data
        targets = [d['Target'] + (tokenizer.eos_token if with_eos else '') for d in data_dicts]
        before_strs = [d['before_adv_string'] + ' ' for d in data_dicts]
        after_strs = [d['after_adv_string'] for d in data_dicts]
        case_ids = [d['CaseID'] for d in data_dicts]

        logging.info(f'GCG Batch || Processing {batch_size} cases')

        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        search_width = self.search_width

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))

        # Initialize optim_ids for all cases
        optim_ids_batch = [
            tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
            for _ in range(batch_size)]
        num_optim_tokens = len(optim_ids_batch[0][0])

        # Cache embeddings for all cases
        all_embeds = []
        all_ids = []
        prefix_caches = []

        for i in range(batch_size):
            cache_input_ids = tokenizer([tokenizer.bos_token + before_strs[i], after_strs[i], targets[i]],
                                        padding=False, add_special_tokens=False)['input_ids']
            cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids]
            before_ids, after_ids, target_ids = cache_input_ids
            before_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]

            all_ids.append((before_ids, after_ids, target_ids))
            all_embeds.append((before_embeds, after_embeds, target_embeds))

            if self.use_prefix_cache:
                with torch.no_grad():
                    outputs = model(inputs_embeds=before_embeds, use_cache=True)
                    prefix_caches.append(outputs.past_key_values)

        visited_optim_ids_batch = [[] for _ in range(batch_size)]

        for step in tqdm(range(num_steps)):
            # Process each case in the batch
            for case_idx in range(batch_size):
                optim_ids = optim_ids_batch[case_idx]
                before_ids, after_ids, target_ids = all_ids[case_idx]
                before_embeds, after_embeds, target_embeds = all_embeds[case_idx]

                # Forward pass
                optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
                optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
                optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

                if self.use_prefix_cache:
                    input_embeds = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)
                    outputs = model(inputs_embeds=input_embeds, past_key_values=prefix_caches[case_idx])
                else:
                    input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds], dim=1)
                    outputs = model(inputs_embeds=input_embeds)

                logits = outputs.logits
                tmp = input_embeds.shape[1] - target_embeds.shape[1]
                shift_logits = logits[..., tmp - 1:-1, :].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
                token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

                # Sample candidates
                sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0),
                                                     search_width, topk=256, temp=1,
                                                     not_allowed_tokens=self.not_allowed_tokens)

                # Filter candidates (parallelized)
                new_sampled_top_indices = self._parallel_filter_candidates(
                    sampled_top_indices, before_ids[0], after_ids[0],
                    visited_optim_ids_batch[case_idx], tokenizer, device
                )

                if len(new_sampled_top_indices) > 0:
                    sampled_top_indices = torch.stack(new_sampled_top_indices)
                    new_search_width = len(new_sampled_top_indices)
                else:
                    new_search_width = search_width

                # Compute loss for candidates
                sampled_top_embeds = embed_layer(sampled_top_indices)
                if self.use_prefix_cache:
                    input_embeds = torch.cat([sampled_top_embeds,
                                              after_embeds.repeat(new_search_width, 1, 1),
                                              target_embeds.repeat(new_search_width, 1, 1)], dim=1)
                else:
                    input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                              sampled_top_embeds,
                                              after_embeds.repeat(new_search_width, 1, 1),
                                              target_embeds.repeat(new_search_width, 1, 1)], dim=1)

                loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds,
                                                                                                        target_ids)

                # Update
                best_optim_ids = sampled_top_indices[loss.argmin()]
                visited_optim_ids_batch[case_idx].append(best_optim_ids)
                optim_ids_batch[case_idx] = best_optim_ids.unsqueeze(0)

                # Log and save
                input_ids = torch.cat([before_ids, best_optim_ids.unsqueeze(0), after_ids], dim=1)
                decoded_text = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=False)
                decoded_adv_string = tokenizer.decode(best_optim_ids, clean_up_tokenization_spaces=False)

                with open(result_files[case_idx], 'a') as f:
                    f.write(json.dumps({'step': step, 'loss': loss.min().item(),
                                        'adv_string': decoded_adv_string, 'input': decoded_text}) + '\n')

                del input_embeds, sampled_top_embeds, logits, shift_logits, loss

            torch.cuda.empty_cache()
            gc.collect()