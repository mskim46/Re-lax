
from  __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.image as jim

@dataclass(fronzen =True)
class LPIPSConfig:
    input_size: Tuple[int, int] = (224, 224)
    layer_names: Tuple[str, ...] = ('conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2')
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

def _resize_bilinear(x: jnp.ndarray, size: Tuple[int, int]) -> jnp.ndarray:
    b, _, _, c = x.shape
    h, w = size
    return jimg.resize(x, (b, h, w, c), method = 'bilinear', antialias = True)


def _imagenent_normalize(x: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.asarray(mean, dtype = x.dtype).reshape(1, 1, 1, 3)
    std = jnp.asarray(std, dtype = x.dtype).reshape(1, 1, 1, 3)
    return (x - mean) / std

def _channel_norm(f: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    denom = jnp.sqrt(jnp.sum(f**2, axis = 1, keepdims = True) + eps)
    return f / denom

def _spatial_mse(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sum((a - b) ** 2, axis = -1), axis = (1, 2))

class LPIPS:
    def __init__(self, config: LPIPSConfig, per_layer_weights: Mapping[str, float] | None = None):
        self.config = config
        self.per_layer_weights = (
            {n: 1.0 for n in config.layer_names} if per_layer_weights is None
            else {n: float(per_layer_weights.get(n, 1.0)) for n in config.layer_names}
        )
    
    def preprocess(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = _resize_bilinear(x, self.config.input_size)
        x = _imagenent_normalize(x, self.config.imagenet_mean, self.config.imagenet_std)
        return x


    def __call__(self,
                backbone_apply: Collabe[..., Dict[str, jnp.ndarray]],
                backbone_params: Any,
                x_target: jnp.ndarray,
                x_recon: jnp.ndarray,
                ) -> jnp.ndarray:
        x_t = jax.lax.stop_gradient(x_target.astype(jnp.float32))
        x_r = x_recon.astype(jnp.float32)
        x_t = self.preprocess(x_t)
        x_r = self.preprocess(x_r)

        feats_t = backbone_apply(backbone_params, x_t, self.config.layer_names)
        feats_r = backbone_apply(backbone_params, x_r, self.config.layer_names)

        per_layer = []
        for name in self.config.layer_names:
            ft = _channel_norm(feats_t[name])
            fr = _channel_norm(feats_r[name])
            dl = _spatial_mse(ft, fr)
            wl = self.per_layer_weights[name]
            per_layer.append(wl * dl)

        return jnp.mean(jnp.stack(per_layer))
    