# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm GLM DSA shared indexer loading.

GLM-5 DSA checkpoints declare an ``indexer_types`` list where "full" layers
ship indexer weights and following "shared" layers reuse the full layer's
top-k indices. The current mlx-lm ``glm_moe_dsa`` model inherits DeepSeek V3.2
and instantiates an indexer for every layer, so strict loading asks for
``self_attn.indexer.*`` tensors that do not exist in shared layers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_ARGS_MARKER = "_omlx_glm_dsa_indexer_types_patch"
_MODEL_INIT_MARKER = "_omlx_glm_dsa_shared_indexer_patch"


def _is_full_indexer(indexer_type: Any) -> bool:
    return str(indexer_type).lower() == "full"


def _patch_model_args_from_dict(glm_module: Any) -> bool:
    model_args_cls = getattr(glm_module, "ModelArgs", None)
    if model_args_cls is None:
        return False

    current = getattr(model_args_cls, "from_dict", None)
    if getattr(current, _ARGS_MARKER, False):
        return False

    original_from_dict = current

    def patched_from_dict(cls, params):
        args = original_from_dict(params)
        for name in (
            "indexer_types",
            "index_topk_freq",
            "index_share_for_mtp_iteration",
        ):
            if name in params:
                setattr(args, name, params[name])
        return args

    patched_from_dict._omlx_glm_dsa_indexer_types_patch = True
    model_args_cls.from_dict = classmethod(patched_from_dict)
    return True


def _patch_model_init(glm_module: Any) -> bool:
    model_cls = getattr(glm_module, "Model", None)
    if model_cls is None:
        return False

    current_init = getattr(model_cls, "__init__", None)
    if getattr(current_init, _MODEL_INIT_MARKER, False):
        return False

    original_init = current_init

    def patched_init(self, config):
        original_init(self, config)

        indexer_types = getattr(config, "indexer_types", None)
        if not indexer_types:
            return

        layers = getattr(getattr(self, "model", None), "layers", [])
        full_count = 0
        shared_count = 0

        self.model._index_cache_state = {"last_topk_indices": None}
        state = self.model._index_cache_state
        for idx, layer in enumerate(layers):
            if layer is None:
                continue
            indexer_type = (
                indexer_types[idx] if idx < len(indexer_types) else "full"
            )
            is_full = _is_full_indexer(indexer_type)
            attn = layer.self_attn
            attn._ic_is_full = is_full
            attn._ic_state = state
            attn._glm_dsa_indexer_type = str(indexer_type)
            if is_full:
                full_count += 1
            else:
                attn.indexer = None
                shared_count += 1

        logger.info(
            "glm_moe_dsa_shared_indexer: configured %d full / %d shared "
            "indexer layers from config",
            full_count,
            shared_count,
        )

    patched_init._omlx_glm_dsa_shared_indexer_patch = True
    model_cls.__init__ = patched_init
    return True


def apply_glm_moe_dsa_shared_indexer_patch() -> bool:
    """Patch mlx-lm's GLM DSA model for config-declared shared indexers."""
    try:
        from mlx_lm.models import glm_moe_dsa
    except ImportError:
        logger.debug("glm_moe_dsa_shared_indexer: mlx_lm model not available")
        return False

    from .index_cache import apply_index_cache_call_patch

    changed = False
    changed = _patch_model_args_from_dict(glm_moe_dsa) or changed
    changed = _patch_model_init(glm_moe_dsa) or changed
    changed = apply_index_cache_call_patch() or changed

    if changed:
        logger.info("glm_moe_dsa_shared_indexer: pre-load patch applied")
    return changed


__all__ = ["apply_glm_moe_dsa_shared_indexer_patch"]
