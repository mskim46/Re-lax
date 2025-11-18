from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.training import train_state
import optax

@dataclass(frozen=True)
#At now, we only use this class to atari environment.
#In the future, we will add more environments.
class TokenizerConfig:
    image_size : int = 96
    num_channels : int = 3
    latent_dim : int = 512
    channel_schedule : Tuple[int, ...] = (32, 64, 128, 256)
    dtype : Any = jnp.float32


class ConvBlock(nn.Module):
    channels : int
    kernel_size : Tuple[int, int] = (3, 3)
    stride : Tuple[int, int] = (1, 1)
    dtype : Any = jnp.float32

    @nn.compact
    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features = self.channels,
            kernel_size = self.kernel_size,
            strides = self.stride,
            padding = 'SAME',
            dtype = self.dtype,
            use_bias = True,
        )(x)
        x = nn.relu(x)
        return x

class Encoder(nn.Module):
    config : TokenizerConfig
    
    @nn.compact
    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        # B x H x W x C -> (B, latent_dim)
        config__ = self.config
        h = x
        # Downsampling
        for channels in config__.channel_schedule:
            h = ConvBlock(channels = channels, stride = (2, 2), dtype = config__.dtype)(h)
            h = ConvBlock(channels = channels, stride = (1, 1), dtype = config__.dtype)(h)
        h = h.reshape((h.shape[0], -1))
        z = nn.Dense(config__.latent_dim , dtype = config__.dtype)(h)
        return z


class Decoder(nn.Module):
    config : TokenizerConfig

    @nn.compact
    def __call__(self, z : jnp.ndarray) -> jnp.ndarray :
        # (B, latent_dim) -> B x H x W x C 
        config__ = self.config
        #Infer the spatial size
        num_down = len(config__.channel_schedule)
        spatial = config__.image_size // (2 ** num_down)
        hidden_channels = config__.channel_schedule[-1]

        h = nn.Dense(spatial * spatial * hidden_channels, dtype = config__.dtype)(z)
        h = nn.relu(h)
        h = h.reshape((h.shape[0], spatial, spatial, hidden_channels))

        # Upsampling
        for i, channels in enumerate(reversed(config__.channel_schedule)):
            h = nn.ConvTranspose(
                features = channels,
                kernel_size = (4, 4),
                strides = (2, 2),
                padding = 'SAME',
                dtype = config__.dtype,
                use_bias = True,
            )(h)

            h = nn.relu(h)
            # One extra Conv for refinement
            h = ConvBlock(channels = channels, stride = (1, 1), dtype = config__.dtype)(h)

        x_hat = nn.Conv(
            features = config__.num_channels,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = 'SAME',
            dtype = config__.dtype,
            use_bias = True,
        )(h)

        x_hat = jnp.clip(x_hat, 0.0, 1.0)
        return x_hat

class AutoEncoder(nn.Module):
    config__ : TokenizerConfig

    def setup(self) -> None:
        self.encoder = Encoder(self.config__)
        self.decoder = Decoder(self.config__)


    def __call__(self, x : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Return reconstruction and latent 
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def mse_loss(x: jnp.ndarray, x_hat: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - x_hat) ** 2)

def psnr_from_mse(mse_value: jnp.ndarray, max_value: float = 1.0) -> jnp.ndarray:
    eps = 1e-8
    return 20.0 * jnp.log10(max_value) - 10.0 * jnp.log10(jnp.maximum(mse_value, eps))

def lpips_loss(x: jnp.ndarray, x_hat: jnp.ndarray) -> jnp.ndarray:
    return lpips.compute_loss(x, x_hat)

class TokenizerTrainState(train_state.TrainState):
    # Mark non-array fields as static so JAX/JIT won't trace them.
    model_apply : Callable[..., Any] = struct.field(pytree_node=False)
    config__ : TokenizerConfig = struct.field(pytree_node=False)

def create_tokenizer_state(
    rng: jax.Array,
    config__: TokenizerConfig,
    learning_rate: float = 3e-4,
) -> TokenizerTrainState:
# Init model params and opt stats
    model = AutoEncoder(config__)

    dummy = jnp.zeros((1, config__.image_size, config__.image_size, config__.num_channels), dtype = config__.dtype)
    params = model.init(rng, dummy)["params"]

    tx = optax.adamw(learning_rate = learning_rate)
    state = TokenizerTrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = tx,
        model_apply = lambda p, x: model.apply({"params": p}, x),
        config__ = config__,
    )
    return state


@jax.jit
def train_step(
    state: TokenizerTrainState,
    batch_images: jnp.ndarray,
) -> Tuple[TokenizerTrainState, Dict[str, jnp.ndarray]]:
    # Single opt step: reconstruction loss on images in [0,1]
    def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x_hat, _ = state.model_apply(params, batch_images)
        loss = mse_loss(batch_images, x_hat)
        metrics = {
            "loss": loss,
            "psnr": psnr_from_mse(loss),
        }
        return loss, metrics

    grads, metrics = jax.grad(loss_fn, has_aux = True)(state.params)
    new_state = state.apply_gradients(grads = grads)
    return new_state, metrics

def normalize_uint8_to_float(image_uint8: jnp.ndarray) -> jnp.ndarray:
    # [0, 255] -> [0, 1]
    return image_uint8.astype(jnp.float32) / 255.0

def denormalize_float_to_uint8(image_float: jnp.ndarray) -> jnp.ndarray:
    # [0, 1] -> [0, 255]
    return jnp.clip(jnp.round(image_float * 255.0), 0, 255).astype(jnp.uint8)



    #tx = optax.adamw


