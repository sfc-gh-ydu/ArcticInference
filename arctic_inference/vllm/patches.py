# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import vllm
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.worker.worker_base import WorkerBase

from arctic_inference.patching import ArcticPatch
from arctic_inference.utils import get_compatible_vllm_version
from arctic_inference.vllm.args import EngineArgsPatch, AsyncEngineArgsPatch
from arctic_inference.vllm.config import (ParallelConfigPatch,
                                          SpeculativeConfigPatch,
                                          VllmConfigPatch,
                                          MLPSpeculatorConfigPatch)
from arctic_inference.vllm.stats import (SpecDecodingStatsPatch, 
                                         SpecDecodingLoggingPatch)
from arctic_inference.vllm.structured_output import XgrammarBackendPatch
from arctic_inference.vllm.ulysses import apply_shift_parallel_patches


logger = init_logger(__name__)


class AsyncSchedulerPatch(ArcticPatch[AsyncScheduler]):
    """Patch AsyncScheduler to:
    1. Respect ``disable_by_batch_size`` when allocating spec token
       placeholders (the worker only drafts for the first N requests).
    2. Use the previous step's actual draft length for dynamic placeholder
       allocation, avoiding wasted verification compute when the real draft
       width (e.g. Arctic n_predict=3) is much smaller than
       num_speculative_tokens (e.g. 12).
    3. Store ``_scheduled_spec_count`` so that the post-fix in
       ``update_from_output`` can compensate for worker-side trimming.
    """

    _orig_update_after_schedule = AsyncScheduler._update_after_schedule

    @staticmethod
    def _is_dgp_enabled(scheduler: AsyncScheduler) -> bool:
        pc = getattr(scheduler.vllm_config, "parallel_config", None)
        return bool(
            getattr(pc, "enable_gearing_parallel", False)
            and int(getattr(pc, "ulysses_sequence_parallel_size", 1)) > 1
        )

    @staticmethod
    def _ensure_dgp_routing_state(scheduler: AsyncScheduler) -> None:
        if not hasattr(scheduler, "data_gearing_parallel_map"):
            scheduler.data_gearing_parallel_map = {}
        if not hasattr(scheduler, "data_gearing_req_to_group"):
            scheduler.data_gearing_req_to_group = {}
        if not hasattr(scheduler, "_data_gearing_group_refcount"):
            scheduler._data_gearing_group_refcount = {}
        if not hasattr(scheduler, "_data_gearing_next_rank"):
            scheduler._data_gearing_next_rank = 0

    @staticmethod
    def _extract_req_id(entry) -> str | None:
        return getattr(entry, "req_id", None)

    @staticmethod
    def _iter_scheduled_req_ids(scheduler_output) -> list[str]:
        req_ids: set[str] = set()
        req_ids.update(
            str(req_id)
            for req_id in getattr(scheduler_output, "num_scheduled_tokens", {}).keys()
        )
        for entry in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            req_id = AsyncSchedulerPatch._extract_req_id(entry)
            if req_id is not None:
                req_ids.add(str(req_id))
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is not None:
            if hasattr(cached_reqs, "req_ids"):
                req_ids.update(str(req_id) for req_id in getattr(cached_reqs, "req_ids", []))
            else:
                for entry in cached_reqs or []:
                    req_id = AsyncSchedulerPatch._extract_req_id(entry)
                    if req_id is not None:
                        req_ids.add(str(req_id))
        for entry in getattr(scheduler_output, "scheduled_running_reqs", []) or []:
            req_id = AsyncSchedulerPatch._extract_req_id(entry)
            if req_id is not None:
                req_ids.add(str(req_id))
        return list(req_ids)

    @staticmethod
    def _compute_prompt_group_key(request) -> tuple[int, ...]:
        num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if num_prompt_tokens > 0 and hasattr(request, "get_token_id"):
            return tuple(int(request.get_token_id(i)) for i in range(num_prompt_tokens))

        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = getattr(request, "all_token_ids", None)
            if token_ids is not None and num_prompt_tokens > 0:
                token_ids = token_ids[:num_prompt_tokens]
        if token_ids is None:
            raise RuntimeError(
                "DGP routing requires prompt token IDs to compute PromptGroupKey."
            )

        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, tuple):
            token_ids = list(token_ids)
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)

        if num_prompt_tokens > 0:
            token_ids = token_ids[:num_prompt_tokens]
        if len(token_ids) == 0:
            raise RuntimeError(
                "DGP routing encountered empty prompt token IDs for PromptGroupKey."
            )
        return tuple(int(token_id) for token_id in token_ids)

    @staticmethod
    def _assign_group_owner_rank(scheduler: AsyncScheduler, group_key: tuple[int, ...]) -> int:
        owner = scheduler.data_gearing_parallel_map.get(group_key)
        if owner is not None:
            return int(owner)
        pc = scheduler.vllm_config.parallel_config
        group_size = max(1, int(getattr(pc, "ulysses_sequence_parallel_size", 1)))
        owner = int(scheduler._data_gearing_next_rank % group_size)
        scheduler._data_gearing_next_rank = owner + 1
        scheduler.data_gearing_parallel_map[group_key] = owner
        return owner

    @staticmethod
    def _build_req_owner_rank(scheduler: AsyncScheduler, scheduler_output) -> dict[str, int]:
        AsyncSchedulerPatch._ensure_dgp_routing_state(scheduler)
        req_owner_rank: dict[str, int] = {}
        for req_id in AsyncSchedulerPatch._iter_scheduled_req_ids(scheduler_output):
            request = scheduler.requests.get(req_id)
            if request is None:
                raise RuntimeError(
                    f"DGP routing missing request state for req_id={req_id}."
                )
            group_key = scheduler.data_gearing_req_to_group.get(req_id)
            if group_key is None:
                group_key = AsyncSchedulerPatch._compute_prompt_group_key(request)
                scheduler.data_gearing_req_to_group[req_id] = group_key
                scheduler._data_gearing_group_refcount[group_key] = (
                    scheduler._data_gearing_group_refcount.get(group_key, 0) + 1
                )
            owner_rank = AsyncSchedulerPatch._assign_group_owner_rank(scheduler, group_key)
            req_owner_rank[req_id] = owner_rank
        return req_owner_rank

    @staticmethod
    def _cleanup_finished_routing_state(scheduler: AsyncScheduler, scheduler_output) -> None:
        if not AsyncSchedulerPatch._is_dgp_enabled(scheduler):
            return
        AsyncSchedulerPatch._ensure_dgp_routing_state(scheduler)
        for req_id in getattr(scheduler_output, "finished_req_ids", []) or []:
            group_key = scheduler.data_gearing_req_to_group.pop(req_id, None)
            if group_key is None:
                continue
            remaining = int(scheduler._data_gearing_group_refcount.get(group_key, 0)) - 1
            if remaining <= 0:
                scheduler._data_gearing_group_refcount.pop(group_key, None)
                scheduler.data_gearing_parallel_map.pop(group_key, None)
            else:
                scheduler._data_gearing_group_refcount[group_key] = remaining

    def _update_after_schedule(self, scheduler_output):
        # Call the base Scheduler._update_after_schedule (NOT the
        # AsyncScheduler override which we are replacing).
        Scheduler._update_after_schedule(self, scheduler_output)

        has_structured_output_requests = False
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Respect disable_by_batch_size: only add spec token placeholders
        # for the first N decode requests (matching the worker's draft_limit
        # in propose_draft_token_ids).
        spec_config = getattr(self.vllm_config, 'speculative_config', None)
        disable_bs = (
            spec_config.disable_by_batch_size if spec_config else None
        )
        decode_with_spec_count = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            has_structured_output_requests |= request.use_structured_output
            pending_structured_output_tokens |= (
                request.use_structured_output
                and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            # Store the originally-scheduled spec count so that
            # update_from_output can compensate for worker-side trimming.
            request._scheduled_spec_count = cur_num_spec_tokens

            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token + spec tokens.
                request.num_output_placeholders += 1 + cur_num_spec_tokens

                # Check if beyond the disable_by_batch_size limit.
                decode_with_spec_count += 1
                if disable_bs and decode_with_spec_count > disable_bs:
                    # Beyond limit: no spec token placeholders.
                    request.spec_token_ids = []
                    continue

                # Use previous step's actual draft length to size
                # placeholders.  When suffix had a good match (actual
                # > n_predict), allocate the full width so the next
                # step can verify all suffix tokens.  When suffix
                # didn't match (actual = n_predict from arctic),
                # allocate only that many to avoid wasting attention
                # compute on zero-padded positions.
                # Cold start: allocate full width (generous).
                prev_actual = getattr(
                    request, '_prev_actual_draft_len', None)
                if prev_actual is not None:
                    num_placeholders = min(
                        max(prev_actual, 1), self.num_spec_tokens)
                else:
                    num_placeholders = self.num_spec_tokens

                request.spec_token_ids = [-1] * num_placeholders

        scheduler_output.has_structured_output_requests = (
            has_structured_output_requests)
        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens)

        if AsyncSchedulerPatch._is_dgp_enabled(self):
            req_owner_rank = AsyncSchedulerPatch._build_req_owner_rank(
                self, scheduler_output
            )
            setattr(scheduler_output, "req_owner_rank", req_owner_rank)

    def update_from_output(self, scheduler_output, model_runner_output):
        """Wrap Scheduler.update_from_output to store actual draft counts.

        We infer the drafter's real capability from the acceptance
        results so that the next ``_update_after_schedule`` can size
        placeholders correctly.  This works even when the worker runs
        in a separate process (where scheduler_output._actual_draft_lens
        set by the worker doesn't survive serialisation back to the
        scheduler).

        Strategy:
        1. **Primary path**: read ``_actual_draft_lens`` from the
           ``model_runner_output`` object (attached by the model runner
           to the ``ModelRunnerOutput`` dataclass, which reliably
           survives the async pipeline).
        2. **Legacy path**: read from ``scheduler_output._actual_draft_lens``
           (works in same-process non-async mode).
        3. **Fallback**: infer from acceptance results with exponential
           growth — when all drafted tokens are accepted, double the
           allocation so suffix decoding reaches full capacity in
           O(log n) steps instead of O(n).
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index

        result = Scheduler.update_from_output(
            self, scheduler_output, model_runner_output)
        AsyncSchedulerPatch._cleanup_finished_routing_state(self, scheduler_output)

        # Primary path: read from model_runner_output (most reliable
        # for async scheduling — the ModelRunnerOutput object is
        # returned by get_output() and guaranteed to survive).
        actual_lens = getattr(
            model_runner_output, '_actual_draft_lens', None)

        # Legacy path: read from scheduler_output (works for non-async
        # or same-process setups where the attribute is preserved).
        if not actual_lens:
            actual_lens = getattr(
                scheduler_output, '_actual_draft_lens', None)

        if actual_lens:
            for req_id, actual_len in actual_lens.items():
                request = self.requests.get(req_id)
                if request is not None:
                    request._prev_actual_draft_len = actual_len
            return result

        # Fallback: infer from acceptance results (multi-process case).
        # Uses exponential growth when all drafted tokens are accepted
        # (indicating the drafter / suffix cache can handle more), so
        # the allocation converges to num_spec_tokens in O(log n)
        # steps:
        #   step 0: 1 position  → accept 1/1 → prev = 2
        #   step 1: 2 positions → accept 2/2 → prev = 4
        #   step 2: 4 positions → accept 4/4 → prev = 8
        #   ...
        # When not all are accepted (normal drafter), linear growth:
        #   step 0: 1 position  → accept 1 → prev = 2
        #   step 1: 2 positions → accept 2 → prev = 3
        #   step 2: 3 positions → steady state (n_predict = 3)
        if not sampled_token_ids:
            return result

        for req_id in scheduler_output.num_scheduled_tokens:
            scheduled_spec = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if not scheduled_spec:
                continue

            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue

            request = self.requests.get(req_id)
            if request is None:
                continue

            generated = sampled_token_ids[req_index]
            num_accepted = (len(generated) - 1) if generated else 0
            num_draft = len(scheduled_spec)

            prev = getattr(request, '_prev_actual_draft_len', None)

            if num_accepted > 0:
                if num_accepted >= num_draft and num_draft > 0:
                    # All drafted tokens accepted — the drafter (or
                    # suffix cache) could produce more if given room.
                    # Double the allocation for exponential ramp-up.
                    new_val = min(
                        num_draft * 2, self.num_spec_tokens)
                else:
                    # Partial acceptance: grow linearly.
                    new_val = min(
                        num_accepted + 1, self.num_spec_tokens)
                request._prev_actual_draft_len = max(
                    prev or 0, new_val)
            elif prev is None:
                # First step for this request, zero acceptance.
                # Seed with 1 so _update_after_schedule doesn't
                # fall back to the full num_spec_tokens next time.
                request._prev_actual_draft_len = 1

        return result


class EngineCoreProcPatch(ArcticPatch[EngineCoreProc]):

    _orig_run_engine_core = EngineCoreProc.run_engine_core

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # When starting the API server, it will spawn a new process to run the
        # EngineCore. We need to load the plugins in the new process before it
        # initializes the Executor.
        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_inference.vllm.model_runner import GPUModelRunnerPatch

        GPUModelRunnerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)


def apply_arctic_patches():

    from transformers import AutoConfig
    from arctic_inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry
    #from arctic_inference.vllm.swiftkv import LlamaSwiftKVForCausalLM

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model(
        "LlamaSwiftKVForCausalLM",
        "arctic_inference.vllm.swiftkv:LlamaSwiftKVForCausalLM")

    # Register ArcticSpeculator models to vLLM.
    from arctic_inference.vllm.spec_dec.arctic_speculator import (
        ArcticMLPSpeculator, ArcticLSTMSpeculator)
    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel",
                                 ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)
    # This name is currently used in corvo
    ModelRegistry.register_model("MLPVariantSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)

    # Patches that make later patches work properly.
    EngineCoreProcPatch.apply_patch()
    WorkerBasePatch.apply_patch()

    # Async scheduler patches for spec decode (disable_by_batch_size
    # interaction + dynamic draft width allocation).
    AsyncSchedulerPatch.apply_patch()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()
    SpecDecodingStatsPatch.apply_patch()
    SpecDecodingLoggingPatch.apply_patch()
    VllmConfigPatch.apply_patch()
    XgrammarBackendPatch.apply_patch()
    MLPSpeculatorConfigPatch.apply_patch()

    # Main optimization patches.
    apply_shift_parallel_patches()
