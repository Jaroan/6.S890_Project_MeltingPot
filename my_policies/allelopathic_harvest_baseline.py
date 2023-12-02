from typing import Tuple

import cv2
import dm_env
from ray.rllib.policy.policy import Policy

from meltingpot.utils.policies import policy

def downsample_88_to_11(array):
  assert array.shape[:2] == (88, 88)
  return cv2.resize(array, (11, 11), interpolation=cv2.INTER_AREA)

class AllelopathicHarvestPolicy(policy.Policy):
  """ Policy wrapping an RLLib model for inference. """

  def __init__(self, policy_id: str) -> None:
    """Initialize a policy instance.

      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    agent_name = policy_id.replace('player_', 'agent_')
    ckpt_dir = 'my_policies/rllib_checkpoints/torch_untrained/'
    substrate_ckpt = "al_harvest/PPO_meltingpot_c1cc0_00000_0_2023-08-29_18-37-18/checkpoint_000001/"
    policy_ckpt = f'{ckpt_dir}/{substrate_ckpt}/policies/{agent_name}/'

    self.policy = Policy.from_checkpoint(policy_ckpt)
    self._prev_action = 0

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    state = self.policy.get_initial_state()
    self.prev_state = state
    return state


  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key and 'INTERACTION_INVENTORIES' not in key
    }
    observations['RGB'] = downsample_88_to_11(observations['RGB'])
    action, state, _ = self.policy.compute_single_action(observations,
                                                       self.prev_state,
                                                       prev_action=self._prev_action, 
                                                       prev_reward=timestep.reward)

    self._prev_action = action
    self.prev_state = state
    return action, state


  def close(self) -> None:
    """See base class."""