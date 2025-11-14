import jax, jax.numpy as jnp
import minari
from src.tokenizer import TokenizerConfig, create_tokenizer_state, train_step

config__ = TokenizerConfig(image_size = 96, num_channels = 1, latent_dim = 512)
key = jax.random.PRNGKey(0)
state = create_tokenizer_state(key, config__, learning_rate = 3e-4)

#Dummy batch, B x H x W x C
batch = jnp.zeros((8, config__.image_size, config__.image_size, config__.num_channels))
state, metrics = train_step(state, batch)

print(metrics)

#dataset = minari.load_dataset('atari/breakout/expert-v0')

