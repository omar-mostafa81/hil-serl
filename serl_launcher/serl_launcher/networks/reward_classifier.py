import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List, Optional
import requests
import os
from tqdm import tqdm
import torch
import numpy as np
from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper

class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class NWayClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x


def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    n_way: int = 2,
):
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoders = {
        image_key: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            pretrained_encoder=pretrained_encoder,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        # enable_stacking=True,
        enable_stacking=False,
        image_keys=image_keys,
    )
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    params = classifier_def.init(key, sample)["params"]
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    file_name = "resnet10_params.pkl"
    # Construct the full path to the file
    file_path = os.path.expanduser("~/.serl/")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, file_name)
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"The ResNet-10 weights already exist at '{file_path}'.")
    else:
        url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
        print(f"Downloading file from {url}")

        # Streaming download with progress bar
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise Exception("Error, something went wrong with the download")
        except Exception as e:
            raise RuntimeError(e)
        print("Download complete!")

    with open(file_path, "rb") as f:
        encoder_params = pkl.load(f)
            
    param_count = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )
    new_params = classifier.params
    for image_key in image_keys:
        if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
            for k in new_params["encoder_def"][f"encoder_{image_key}"][
                "pretrained_encoder"
            ]:
                if k in encoder_params:
                    new_params["encoder_def"][f"encoder_{image_key}"][
                        "pretrained_encoder"
                    ][k] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")

    classifier = classifier.replace(params=new_params)
    return classifier

def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
) -> Callable[[Dict], jnp.ndarray]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(key, sample, image_keys, n_way=n_way)
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
    )
    func = lambda obs: classifier.apply_fn(
        {"params": classifier.params}, obs, train=False
    )
    func = jax.jit(func)
    return func

def load_torch_reward_fn(
    checkpoint_path: str,
    n_way: int = 2,
):
    """
    Returns a reward function that:
      - Lazily initializes a JAX/Flax classifier from `checkpoint_path` on first call,
      - Infers input shape and uses a default image key,
      - Accepts a single image (HWC ndarray or CHW/HWC torch.Tensor or dict),
      - Runs the classifier and returns 0/1 based on sigmoid threshold.
    Usage:
        reward_fn = load_torch_reward_fn("path/to/ckpt/")
        reward = reward_fn(obs)  # obs: np.ndarray or torch.Tensor or {"image": ...}
    """
    jax_fn: Optional[Callable] = None
    params_initialized = False
    default_key = "image"

    def reward_fn(
        obs
    ) -> int:
        nonlocal jax_fn, params_initialized
        # Extract raw image
        if isinstance(obs, dict):
            raw = list(obs.values())[0]
        else:
            raw = obs

        # Convert to numpy
        if isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.array(raw)

        # On first call, initialize jax_fn using this shape
        if not params_initialized:
            # Ensure HWC -> batch NHWC
            if arr.ndim == 3:
                h, w, c = arr.shape
                sample = {default_key: jnp.ones((1, h, w, c), jnp.float32)}
            elif arr.ndim == 4 and arr.shape[-1] == arr.shape[1]:
                # NHWC batch
                sample = {default_key: jnp.ones(arr.shape, jnp.float32)}
            else:
                raise ValueError(f"Cannot infer sample shape from {arr.shape}")
            # Build jax_fn now
            jax_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=sample,
                image_keys=[default_key],
                checkpoint_path=checkpoint_path,
                n_way=n_way,
            )
            params_initialized = True

        # Prepare input batch NHWC
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                batch = arr[None, ...]
            elif arr.shape[0] == 3:
                batch = arr.transpose(1,2,0)[None, ...]
            else:
                raise ValueError(f"Unrecognized image shape {arr.shape}")
        elif arr.ndim == 4:
            # detect NCHW vs NHWC
            if arr.shape[1] == 3:
                batch = arr.transpose(0,2,3,1)
            else:
                batch = arr
        else:
            raise ValueError(f"Unsupported ndarray ndim={arr.ndim}")

        # Run through JAX classifier
        jax_in = {default_key: jax.device_put(batch.astype(np.float32))}
        jax_out = jax_fn(jax_in)
        logits = np.array(jax.device_get(jax_out)).squeeze(0)

        # Sigmoid + threshold
        prob = torch.sigmoid(torch.from_numpy(logits))
        return prob.item()

    return reward_fn
