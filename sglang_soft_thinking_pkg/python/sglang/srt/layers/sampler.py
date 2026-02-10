import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

# ==========
# begin of soft thinking
# ==========
import triton
import triton.language as tl
# ==========
# end of soft thinking
# ==========

from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")

def calculate_js_divergence(logits, probs, log_input=False):
    if log_input:
        probs = torch.softmax(probs, dim=-1)
        
    # Post process logits
    tmp_probs = torch.softmax(logits, dim=-1)
    log_avg_prob = torch.log((probs+tmp_probs) * 0.5 + 1e-10)
    js_div = 0.5 * (F.kl_div(log_avg_prob, tmp_probs + 1e-10, reduction='none').sum(dim=-1) + 
        F.kl_div(log_avg_prob, probs + 1e-10, reduction='none').sum(dim=-1))
    
    return js_div

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of soft thinking
        # ==========
        enable_soft_thinking: bool = False,
        use_gumbel_randomness: bool = False,
        calculate_top1_input_kl: bool = False,
        # ==========
        # end of soft thinking
        # ==========
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits
        # ==========
        # begin of soft thinking
        # ==========
        probs_clone = None
        # ==========
        # end of soft thinking
        # ==========

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            # ==========
            # begin of soft thinking
            # ==========
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            batch_next_token_ids = torch.argmax(logits, -1)
            if enable_soft_thinking:
                # logits_output.topk_probs, logits_output.topk_indices 
                # logits.div_(sampling_info.temperatures)
                # logits[:] = torch.softmax(logits, dim=-1)
                # probs = logits
                # del logits
                # # determine how many top-k to keep (at least 1)
                # # 只用argmax，不用topk以提升速度
                # max_k = max(1, sampling_info.max_topk if sampling_info.max_topk is not None else 1)
                # logits_output.topk_probs = torch.zeros(probs.shape[0], max_k, dtype=probs.dtype, device=probs.device)
                # logits_output.topk_indices = torch.zeros(probs.shape[0], max_k, dtype=torch.long, device=probs.device)
                
                # # 取对应位置的概率，其余为0
                # logits_output.topk_probs[:, 0] = torch.gather(probs, 1, batch_next_token_ids.unsqueeze(1)).squeeze(1)
                logits_output.topk_probs[:, 0] = 1
                logits_output.topk_indices[:, 0] = batch_next_token_ids
            # ==========
            # end of soft thinking
            # ==========
            else:
                pass
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
                # ==========
                # begin of soft thinking
                # ==========
                # if enable_soft_thinking:
                #     batch_next_token_ids = top_k_top_p_min_p_dirichlet_alpha_sampling_from_probs_torch_optimized(
                #         probs,
                #         logits_output,
                #         sampling_info
                #     )
                # ==========
                # end of soft thinking
                # ==========

                # ==========
                # begin of soft thinking
                # ==========
                max_top_k_round, batch_size = 32, probs.shape[0]
                if enable_soft_thinking:

                    # 计算熵
                    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                    entropy = entropy / torch.log(torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device))
                    # nor
                    soft_mask = sampling_info.soft_thinking_modes # Shape (B,)

                    # 计算只使用top1输入的KL散度
                    if calculate_top1_input_kl and logits_output.token_input_next_token_logits is not None:
                        top1_divergence = calculate_js_divergence(
                            logits_output.token_input_next_token_logits[0].div_(sampling_info.temperatures),
                            probs
                            )
                        top2_divergence = calculate_js_divergence(
                            logits_output.token_input_next_token_logits[1].div_(sampling_info.temperatures),
                            probs
                            )
                        top1_top2_divergence = calculate_js_divergence(
                            logits_output.token_input_next_token_logits[0],
                            logits_output.token_input_next_token_logits[1],
                            log_input=True
                        )
                        logits_output.soft_top1_kl = torch.stack((top1_divergence, top2_divergence, top1_top2_divergence), dim=-1)
                    else:
                        B = soft_mask.shape[0]
                        logits_output.soft_top1_kl = torch.zeros((B,3)).to(entropy.device)
                    
                    top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
                    top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
                    min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
                    dirichlet_alphas = sampling_info.dirichlet_alphas

                    # top k top p renorm
                    probs = top_k_renorm_prob(probs, top_ks)
                    probs = top_p_renorm_prob(probs, top_ps)

                    # minp renorm
                    if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
                        max_prob = probs.max(dim=-1, keepdim=True).values
                        min_p_thresholds = max_prob * min_ps.view(-1, 1)
                        min_p_mask = probs < min_p_thresholds
                        probs.masked_fill_(min_p_mask, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    original_probs = probs.clone()
                    original_probs = top_k_renorm_prob(original_probs, sampling_info.max_topk)

                    # Dirichlet
                    if not sampling_info.is_all_no_noise: # slow

                        non_zero_probs_mask = (probs > 1e-7)
                        # dirichlet_alphas = (dirichlet_alphas - torch.tensor(0.5)) * entropy + torch.tensor(0.5)
                        conc = probs[soft_mask] * dirichlet_alphas[soft_mask].view(-1, 1)
                        gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                        gamma_samples = gamma_dist.sample()
                        probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                        probs_new = probs_new * non_zero_probs_mask[soft_mask]
                        probs[soft_mask] = probs_new / probs_new.sum(dim=-1, keepdim=True)

                    # max top k
                    topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
                    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))

                    if use_gumbel_randomness:
                        topk_logits = torch.log(topk_probs)
                        topk_probs = F.gumbel_softmax(topk_logits, tau=sampling_info.gumbel_temperature)
                    

                    sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                    topk_probs = sorted_weights
                    topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

                    if calculate_top1_input_kl:
                        new_probs = torch.zeros_like(original_probs)
                        new_probs.scatter_(1, topk_indices, topk_probs)

                        random_js_div = calculate_js_divergence(
                            new_probs,
                            original_probs,
                        )
                        logits_output.soft_top1_kl = torch.cat((logits_output.soft_top1_kl, random_js_div.unsqueeze(-1)), dim=-1)


                    # orginal sampling (after thinking)
                    non_soft_mask = ~soft_mask
                    if any(non_soft_mask):
                        sampled_token_ids = torch.multinomial(probs, num_samples=1)

                        # For rows where soft_thinking_modes is False
                        topk_probs[non_soft_mask] = 0.0
                        topk_indices[non_soft_mask] = 0

                        # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                        topk_probs[non_soft_mask, 0] = 1.0
                        topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)
                    

                    logits_output.topk_probs = topk_probs
                    logits_output.topk_indices = topk_indices
                    logits_output.entropy = entropy
                    batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
                # ==========
                # end of soft thinking
                # ========== 
                else:
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        # Check Nan will throw exception, only check when crash_on_warnings is True
                        check_nan = self.use_nan_detection and crash_on_warnings()
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=check_nan,
                        )

            elif global_server_args_dict["sampling_backend"] == "pytorch":
                raise NotImplementedError("Pytorch sampling backend is not implemented")
                # A slower fallback implementation with torch native operations.
                # ==========
                # begin of soft thinking
                # ==========
                if enable_soft_thinking:
                    batch_next_token_ids = top_k_top_p_min_p_dirichlet_alpha_sampling_from_probs_torch_optimized(
                        probs,
                        logits_output,
                        sampling_info
                    )
                # ==========
                # end of soft thinking
                # ==========
                else:
                    batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        sampling_info.min_ps,
                        sampling_info.need_min_p_sampling,
                    )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
                
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def top_k_top_p_min_p_dirichlet_alpha_sampling_from_probs_torch(
    probs: torch.Tensor,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    enable_entropy_threshold: bool = False,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    temperature = sampling_info.temperatures
    top_p = sampling_info.top_ps
    top_k = sampling_info.top_ks
    min_p = sampling_info.min_ps

    # replace with after thinking sampling params where soft_thinking_modes is False
    if any(~sampling_info.soft_thinking_modes):
        temperature[~sampling_info.soft_thinking_modes] = sampling_info.after_thinking_temperatures[~sampling_info.soft_thinking_modes]
        top_p[~sampling_info.soft_thinking_modes] = sampling_info.after_thinking_top_ps[~sampling_info.soft_thinking_modes]
        top_k[~sampling_info.soft_thinking_modes] = sampling_info.after_thinking_top_ks[~sampling_info.soft_thinking_modes]
        min_p[~sampling_info.soft_thinking_modes] = sampling_info.after_thinking_min_ps[~sampling_info.soft_thinking_modes]


    probs_sort, probs_idx = torch.topk(probs, k=sampling_info.max_topk, dim=-1, largest=True, sorted=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    top_k_mask = torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1) >= sampling_info.top_ks.view(-1, 1)
    probs_sort.masked_fill_(top_k_mask, 0.0)

    top_p_mask = (probs_sum - probs_sort) > sampling_info.top_ps.view(-1, 1)
    probs_sort.masked_fill_(top_p_mask, 0.0)

    if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * sampling_info.min_ps
        min_p_mask = probs_sort < min_p_thresholds.unsqueeze(1)
        probs_sort.masked_fill_(min_p_mask, 0.0)

    topk_probs = probs_sort # keep the original probs, will norm when do weighted embedding
    topk_indices = probs_idx  # (B, max_k)

    # Dirichlet sampling around existing topk_probs distribution
    # first normalize topk_probs to sum to 1 for proper concentration parameters
    normed = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))
    # concentration parameters = normalized probs * dirichlet_alphas
    conc = normed * sampling_info.dirichlet_alphas.view(-1, 1)
    gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
    gamma_samples = gamma_dist.sample()
    dirichlet_weights = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
    # 对 dirichlet_weights 按大小降序排序，并根据排序结果重排 topk_probs 和 topk_indices
    sorted_weights, sorted_idx = torch.sort(dirichlet_weights, dim=-1, descending=True)
    # 将 topk_probs 换成排序后的 dirichlet 权重
    dirichlet_topk_probs = sorted_weights
    # 根据排序索引重排原先的 topk_indices
    dirichlet_topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

    # # introduce gumbel randomness
    # dirichlet_topk_logits = torch.log(dirichlet_topk_probs)
    # dirichlet_topk_probs = F.gumbel_softmax(dirichlet_topk_logits, tau=sampling_info.gumbel_temperature)

    # Mode 0: one-hot 和采样索引
    sampled_indices = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
    one_hot_probs = torch.zeros_like(topk_probs, dtype=logits_output.hidden_states.dtype).scatter_(1, sampled_indices, 1.0)  # (B, max_k)
    sampled_token_ids = torch.gather(probs_idx, dim=1, index=sampled_indices)  # (B, 1)
    one_hot_indices = torch.zeros_like(topk_indices, dtype=topk_indices.dtype).scatter_(1, sampled_indices, sampled_token_ids)  # (B, max_k)

    # 根据 soft_thinking_modes 合并
    soft_thinking_modes = sampling_info.soft_thinking_modes
    conbined_probs = dirichlet_topk_probs.to(logits_output.hidden_states.dtype)
    conbined_indices = dirichlet_topk_indices.to(logits_output.hidden_states.dtype)

    combined_probs[~soft_thinking_modes] = 0.0
    combined_indices[~soft_thinking_modes] = 0

    combined_probs[~soft_thinking_modes, 0] = 1.0
    combined_indices[~soft_thinking_modes, 0] = sampled_token_ids[~soft_thinking_modes].view(-1)

    if enable_entropy_threshold:
        entropies = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
        entropy_mask = entropies < entropy_threshold
        
        conbined_probs[~soft_thinking_modes] = 0.0
        conbined_indices[~soft_thinking_modes, 0] = sampled_token_ids[~soft_thinking_modes].view(-1)
        


    # Store the results separately
    logits_output.topk_probs = combined_probs  # (B, K)
    logits_output.topk_indices = combined_indices  # (B, K)

    # 取第一个作为代表的token id
    # int32 range is enough to represent the token ids
    combined_indices = combined_indices.to(torch.int32)
    batch_next_token_ids = combined_indices[:, 0]
    return batch_next_token_ids


# --- Triton Kernel ---
@triton.jit
def _filter_probs_kernel(
    probs_ptr,          # Pointer to input top-k probabilities (B, max_k)
    probs_sum_ptr,      # Pointer to precomputed cumsum (B, max_k)
    filtered_probs_ptr, # Pointer to output filtered probabilities (B, max_k)
    top_p_ptr,          # Pointer to top_p values (B,)
    top_k_ptr,          # Pointer to top_k values (B,) - float but used as int limit
    min_p_ptr,          # Pointer to min_p values (B,)
    apply_min_p_mask_ptr, # Pointer to boolean mask indicating if min_p applies (B,)
    max_k,              # Dimension of the top-k probabilities
    stride_probs_batch, # Stride for probs/cumsum/output batch dim
    stride_probs_k,     # Stride for probs/cumsum/output k dim
    stride_params,      # Stride for param vectors (top_p, top_k, min_p, apply_min_p)
    BLOCK_SIZE_K: tl.constexpr, # Block size for the k dimension
    DTYPE: tl.constexpr # Match the dtype of probs (e.g., tl.float16)
):
    # --- Kernel Setup ---
    batch_idx = tl.program_id(axis=0)

    # --- Load per-row parameters ---
    param_offset = batch_idx * stride_params
    top_p_val = tl.load(top_p_ptr + param_offset).to(DTYPE)
    top_k_val = tl.load(top_k_ptr + param_offset).to(tl.int32) # Cast to int for comparison
    min_p_val = tl.load(min_p_ptr + param_offset).to(DTYPE)
    apply_min_p = tl.load(apply_min_p_mask_ptr + param_offset)

    # --- Process probabilities in blocks ---
    # Initialize min_p_threshold with the correct DTYPE
    min_p_threshold = tl.zeros((), dtype=DTYPE) # Correct way to initialize with DTYPE

    # Calculate min_p_threshold once before the loop if apply_min_p is true
    if apply_min_p:
         # Load original k=0 prob and cumsum
         # Ensure loading respects DTYPE
         first_prob_k0_orig = tl.load(probs_ptr + batch_idx * stride_probs_batch).to(DTYPE)
         first_prob_k0_cumsum = tl.load(probs_sum_ptr + batch_idx * stride_probs_batch).to(DTYPE)

         # Check if k=0 passed k and p filters
         first_prob_passes_k = 0 < top_k_val
         # Use '>' for original top-p masking condition
         first_prob_passes_p = ~((first_prob_k0_cumsum - first_prob_k0_orig) > top_p_val)

         if first_prob_passes_k and first_prob_passes_p:
             # If k=0 survived k/p, use its value for threshold
             # Calculation result should naturally be DTYPE if inputs are DTYPE
             min_p_threshold = first_prob_k0_orig * min_p_val
         # else: min_p_threshold remains 0.0 (of DTYPE)


    for k_start in range(0, tl.cdiv(max_k, BLOCK_SIZE_K)):
        offs_k = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < max_k # Boundary check for k dimension

        # Calculate pointers
        base_offset = batch_idx * stride_probs_batch
        offs_block = offs_k * stride_probs_k
        probs_block_ptr = probs_ptr + base_offset + offs_block
        probs_sum_block_ptr = probs_sum_ptr + base_offset + offs_block
        filtered_probs_block_ptr = filtered_probs_ptr + base_offset + offs_block

        # Load data, using 0.0 literal (Triton handles type promotion correctly here)
        # Ensure loaded data is cast to DTYPE if necessary, although loads usually infer correctly
        current_probs = tl.load(probs_block_ptr, mask=mask_k, other=0.0).to(DTYPE)
        current_probs_sum = tl.load(probs_sum_block_ptr, mask=mask_k, other=0.0).to(DTYPE)

        # --- Apply Filtering Masks ---
        # 1. Top-K Mask: Keep if index < top_k_val
        filter_mask_k = offs_k < top_k_val

        # 2. Top-P Mask: Keep if not ((probs_sum - current_probs) > top_p_val)
        filter_mask_p = ~((current_probs_sum - current_probs) > top_p_val) # Match original '>' logic

        # Combine k and p masks
        combined_filter_mask = filter_mask_k & filter_mask_p

        # 3. Min-P Mask (Conditional and based on pre-calculated threshold)
        final_filter_mask = combined_filter_mask # Start with k/p mask
        if apply_min_p:
            # Mask OUT condition: (prob < threshold)
            # We already checked min_p_val > 0 via apply_min_p flag
            mask_out_min_p = (current_probs < min_p_threshold)
            # Update final mask: must pass k/p AND NOT be masked by min_p
            final_filter_mask = final_filter_mask & (~mask_out_min_p)

        # Apply the final combined mask
        # Ensure the 'other' value in tl.where also matches DTYPE if needed, 0.0 literal is often fine.
        final_filtered_probs = tl.where(final_filter_mask, current_probs, 0.0)

        # Store the result
        tl.store(filtered_probs_block_ptr, final_filtered_probs, mask=mask_k)

# --- Triton Wrapper Function ---
def apply_triton_filtering(
    probs_sort: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    apply_min_p_mask: torch.Tensor,
    max_k: int
) -> torch.Tensor:
    """
    Applies top-k, top-p, and min-p filtering using a Triton kernel.

    Args:
        probs_sort: Tensor of shape (B, max_k), sorted top-k probabilities.
        top_p: Tensor of shape (B,), top-p values for each row.
        top_k: Tensor of shape (B,), top-k values (float) for each row.
        min_p: Tensor of shape (B,), min_p values for each row.
        apply_min_p_mask: Tensor of shape (B,), boolean indicating if min_p active.
        max_k: Integer, the size of the k dimension (must match probs_sort.shape[1]).

    Returns:
        Tensor of shape (B, max_k) with filtered probabilities.
    """
    # assert probs_sort.is_cuda, "Input tensor must be on CUDA"
    # assert probs_sort.dim() == 2, "Input probs must be 2D (B, max_k)"
    # assert top_p.dim() == 1 and top_k.dim() == 1 and min_p.dim() == 1
    # assert apply_min_p_mask.dim() == 1
    # assert probs_sort.shape[0] == top_p.shape[0] == top_k.shape[0] == min_p.shape[0] == apply_min_p_mask.shape[0]
    # assert probs_sort.shape[1] == max_k

    batch_size = probs_sort.shape[0]
    input_dtype = probs_sort.dtype # e.g., torch.float16

    # --- Precompute Cumsum (matches PyTorch logic) ---
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # --- Prepare Output Tensor ---
    filtered_probs = torch.empty_like(probs_sort)

    # --- Kernel Launch Setup ---
    # Ensure parameters are contiguous if needed, though Triton handles strides
    # top_p = top_p.contiguous()
    # top_k = top_k.contiguous() # Ensure float type matches kernel expectations if used directly
    # min_p = min_p.contiguous()
    # apply_min_p_mask = apply_min_p_mask.contiguous()

    # Define block size (tuneable)
    BLOCK_SIZE_K = 128 # Example, adjust based on max_k and GPU arch

    # Define grid size (one program instance per batch item)
    grid = (batch_size,)

    # Determine Triton dtype
    if input_dtype == torch.float16:
        TRITON_DTYPE = tl.float16
    elif input_dtype == torch.float32:
        TRITON_DTYPE = tl.float32
    elif input_dtype == torch.bfloat16:
        TRITON_DTYPE = tl.bfloat16
    else:
        raise TypeError(f"Unsupported dtype: {input_dtype}")

    # Launch the kernel
    _filter_probs_kernel[grid](
        probs_sort,         # Pass tensor directly, Triton handles pointers
        probs_sum,          # Pass precomputed cumsum
        filtered_probs,
        top_p,
        top_k,              # Pass float top_k
        min_p,
        apply_min_p_mask,   # Pass boolean tensor
        max_k,
        probs_sort.stride(0), probs_sort.stride(1), # Strides for probs/sum/output
        top_p.stride(0),    # Stride for param vectors (assuming contiguous)
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        DTYPE=TRITON_DTYPE,
        # num_warps=4,      # Optional: Tuning parameter
        # num_stages=3,     # Optional: Tuning parameter
    )

    return filtered_probs


# --- Updated Optimized Function using Triton ---
def top_k_top_p_min_p_dirichlet_alpha_sampling_from_probs_torch_optimized(
    probs: torch.Tensor,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
):
    """Optimized sampling using Triton kernel for filtering."""
    hidden_dtype = logits_output.hidden_states.dtype # Output embedding dtype

    # Opt 1: Parameter Selection
    soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
    top_p = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
    top_k = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
    min_p = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
    dirichlet_alpha = sampling_info.dirichlet_alphas

    max_k = max(1, sampling_info.max_topk)
    # top_k_val = torch.clamp(top_k, min=1, max=max_k).long() # Keep top_k as float for kernel

    # --- Perform Top-K in PyTorch ---
    # Ensure probs are on CUDA and potentially convert dtype if kernel expects specific type
    # For simplicity, assume input probs dtype is suitable (e.g., float16)
    probs_sort, probs_idx = torch.topk(probs, k=max_k, dim=-1, largest=True, sorted=True)

    # --- Apply Filtering using Triton Kernel ---
    # Determine if min_p needs applying for any row based on flags and values
    apply_min_p_flags = sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling
    apply_min_p_mask = (min_p > 0) & apply_min_p_flags

    final_filtered_probs = apply_triton_filtering(
        probs_sort, top_p, top_k, min_p, apply_min_p_mask, max_k
    )
    final_filtered_indices = probs_idx # Indices remain the same

    # --- Compute BOTH paths fully using PyTorch (post-Triton filtering) ---
    # Uses final_filtered_probs from Triton kernel

    # Dirichlet Path Components (Calculated fully)
    normed_probs_dir = final_filtered_probs / (final_filtered_probs.sum(dim=-1, keepdim=True))
    if not sampling_info.is_all_no_noise:
        # normed_probs_dir = torch.nan_to_num(normed_probs_dir, nan=0.0)
        conc = normed_probs_dir * dirichlet_alpha.view(-1, 1)
        # conc = torch.clamp(conc, min=1e-9)
        gamma_samples = torch._standard_gamma(conc)
        dirichlet_weights = gamma_samples / (gamma_samples.sum(dim=-1, keepdim=True))
        # dirichlet_weights = torch.nan_to_num(dirichlet_weights, nan=0.0)
        sorted_dirichlet_weights, sorted_dirichlet_rank_idx = torch.sort(dirichlet_weights, dim=-1, descending=True)
        dirichlet_path_probs = sorted_dirichlet_weights
        dirichlet_path_indices = torch.gather(final_filtered_indices, dim=1, index=sorted_dirichlet_rank_idx)
    else:
        dirichlet_path_probs = normed_probs_dir
        dirichlet_path_indices = final_filtered_indices

    # Standard Multinomial Path Components (Calculated fully)
    probs_for_sampling = normed_probs_dir
    # probs_for_sampling = torch.nan_to_num(probs_for_sampling, nan=0.0)
    all_zero_mask_std = (probs_for_sampling.sum(dim=-1) == 0)
    if torch.any(all_zero_mask_std):
       probs_for_sampling[all_zero_mask_std, 0] = 1.0
    sampled_indices_std = torch.multinomial(probs_for_sampling, num_samples=1) # Sampled for all rows
    sampled_token_ids_std = torch.gather(final_filtered_indices, dim=1, index=sampled_indices_std)
    std_path_probs = torch.zeros_like(final_filtered_probs, dtype=hidden_dtype).scatter_(1, sampled_indices_std, 1.0)
    std_path_indices = torch.zeros_like(final_filtered_indices).scatter_(1, sampled_indices_std, sampled_token_ids_std)

    # --- Combine based on soft_thinking_modes using torch.where ---
    soft_mask_expanded = soft_mask.view(-1, 1) # (B, 1)
    final_topk_probs = torch.where(soft_mask_expanded, dirichlet_path_probs, std_path_probs)
    final_topk_indices = torch.where(soft_mask_expanded, dirichlet_path_indices, std_path_indices)

    # Store results
    logits_output.topk_probs = final_topk_probs
    logits_output.topk_indices = final_topk_indices

    # Return value using original's logic
    batch_next_token_ids = final_topk_indices[:, 0]

    return batch_next_token_ids.to(torch.int32)
import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda
import torch.nn.functional as F

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of soft thinking
        # ==========
        enable_soft_thinking: bool = False,
        add_noise_dirichlet: bool = False,
        add_noise_gumbel_softmax: bool = False,
        # ==========
        # end of soft thinking
        # ==========
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits
        # ==========
        # begin of soft thinking
        # ==========
        probs_clone = None
        # ==========
        # end of soft thinking
        # ==========

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            # ==========
            # begin of soft thinking
            # ==========
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            batch_next_token_ids = torch.argmax(logits, -1)
            if enable_soft_thinking:
                # logits_output.topk_probs, logits_output.topk_indices 
                # logits.div_(sampling_info.temperatures)
                # logits[:] = torch.softmax(logits, dim=-1)
                # probs = logits
                # del logits
                # # determine how many top-k to keep (at least 1)
                # # 只用argmax，不用topk以提升速度
                # max_k = max(1, sampling_info.max_topk if sampling_info.max_topk is not None else 1)
                # logits_output.topk_probs = torch.zeros(probs.shape[0], max_k, dtype=probs.dtype, device=probs.device)
                # logits_output.topk_indices = torch.zeros(probs.shape[0], max_k, dtype=torch.long, device=probs.device)
                
                # # 取对应位置的概率，其余为0
                # logits_output.topk_probs[:, 0] = torch.gather(probs, 1, batch_next_token_ids.unsqueeze(1)).squeeze(1)
                logits_output.topk_probs[:, 0] = 1
                logits_output.topk_indices[:, 0] = batch_next_token_ids
            # ==========
            # end of soft thinking
            # ==========
            else:
                pass
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)

                # ==========
                # begin of soft thinking
                # ==========
                if enable_soft_thinking:
                    # calculate the entropy
                    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                    soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
                    top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
                    top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
                    min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
                    dirichlet_alphas = sampling_info.dirichlet_alphas

                    # top k top p renorm
                    probs = top_k_renorm_prob(probs, top_ks)
                    probs = top_p_renorm_prob(probs, top_ps)

                    # minp renorm
                    if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
                        max_prob = probs.max(dim=-1, keepdim=True).values
                        min_p_thresholds = max_prob * min_ps.view(-1, 1)
                        min_p_mask = probs < min_p_thresholds
                        probs.masked_fill_(min_p_mask, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    # dirichlet noise (not used in paper)
                    if add_noise_dirichlet: # slow
                        conc = probs[soft_mask] * dirichlet_alphas[soft_mask].view(-1, 1)
                        gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                        gamma_samples = gamma_dist.sample()
                        probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                        probs[soft_mask] = probs_new

                    # max top k
                    topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
                    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))

                    # gumbel softmax noise (not used in paper)
                    if add_noise_gumbel_softmax: # slow
                        topk_logits = torch.log(topk_probs)
                        gumbels = (
                            -torch.empty_like(topk_logits)
                            .exponential_()
                            .log()
                        )  # ~Gumbel(0,1)
                        gumbels = (topk_logits + gumbels) / sampling_info.gumbel_softmax_temperatures  # ~Gumbel(logits,tau)
                        topk_probs = gumbels.softmax(-1)
                        # topk_probs = F.gumbel_softmax(topk_logits, tau=sampling_info.dirichlet_alphas) # use dirichlet alpha as temperature
                        # sort the topk token according to new probability
                        sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                        topk_probs = sorted_weights
                        topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

                    # after thinking sampling
                    non_soft_mask = ~soft_mask
                    if any(non_soft_mask):
                        sampled_token_ids = torch.multinomial(probs, num_samples=1)

                        # For rows where soft_thinking_modes is False
                        topk_probs[non_soft_mask] = 0.0
                        topk_indices[non_soft_mask] = 0

                        # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                        topk_probs[non_soft_mask, 0] = 1.0
                        topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)

                    logits_output.topk_probs = topk_probs
                    logits_output.topk_indices = topk_indices
                    logits_output.entropy = entropy
                    batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
                # ==========
                # end of soft thinking
                # ========== 
                else:
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        # Check Nan will throw exception, only check when crash_on_warnings is True
                        check_nan = self.use_nan_detection and crash_on_warnings()
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=check_nan,
                        )

            elif global_server_args_dict["sampling_backend"] == "pytorch":
                raise NotImplementedError("Pytorch sampling backend is not implemented")
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
                
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx