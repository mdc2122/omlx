# SPDX-License-Identifier: Apache-2.0


def _tiny_glm_config():
    return {
        "model_type": "glm_moe_dsa",
        "vocab_size": 32,
        "hidden_size": 8,
        "index_head_dim": 2,
        "index_n_heads": 2,
        "index_topk": 2,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_shared_experts": None,
        "n_routed_experts": None,
        "routed_scaling_factor": 1.0,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "qk_nope_head_dim": 2,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 99,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0},
        "attention_bias": False,
        "indexer_types": ["full", "shared", "shared", "full"],
    }


def test_glm_shared_indexer_patch_removes_shared_layer_indexers():
    from mlx_lm.models import glm_moe_dsa

    from omlx.patches.glm_moe_dsa_shared_indexer import (
        apply_glm_moe_dsa_shared_indexer_patch,
    )

    apply_glm_moe_dsa_shared_indexer_patch()

    args = glm_moe_dsa.ModelArgs.from_dict(_tiny_glm_config())
    assert args.indexer_types == ["full", "shared", "shared", "full"]

    model = glm_moe_dsa.Model(args)
    layers = model.model.layers

    assert layers[0].self_attn._ic_is_full is True
    assert layers[0].self_attn.indexer is not None
    assert layers[1].self_attn._ic_is_full is False
    assert layers[1].self_attn.indexer is None
    assert layers[2].self_attn._ic_is_full is False
    assert layers[2].self_attn.indexer is None
    assert layers[3].self_attn._ic_is_full is True
    assert layers[3].self_attn.indexer is not None
