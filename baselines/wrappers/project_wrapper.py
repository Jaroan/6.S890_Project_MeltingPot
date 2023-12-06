import dmlab2d
from gymnasium import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
from baselines.wrappers import meltingpot_wrapper
from baselines.train import utils



class ProjectEnv(meltingpot_wrapper.MeltingPotEnv):
  """Interfacing Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    """Initializes the instance.

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
    """
    print("ProjectEnv init---------------------", env)
    super().__init__(env)
    self.alpha = 0.05
    self.beta = 0.5
    self.gamma = 0.99
    self.lambda_value = 0.95
    self.reward_traces = {}


  def reset(self, *args, **kwargs):
    """See base class."""
    timestep = self._env.reset()
    self.reward_traces = {agent_id: np.zeros_like(timestep.reward) for agent_id in self._ordered_agent_ids}
    return utils.timestep_to_observations(timestep), {}

  def step(self, action_dict):
    """See base class."""
    # Extract actions for each agent as per _ordered_agent_ids
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]

    #Step the ennvironment with actions list
    timestep = self._env.step(actions)

    #Empty dict to store rewards
    rewards = {}

    #Iterate over each agent to calculate individual reward
    for index, agent_id in enumerate(self._ordered_agent_ids):
        #Immediate reward
        ri = timestep.reward[index]
        #uUdate reward trace for temporal smoothing
        et_i = self.reward_traces[agent_id]
        et_i = self.gamma * self.lambda_value * et_i + ri
        self.reward_traces[agent_id] = et_i

        # Calculate inequity aversion
        inequity_aversion = 0
        # Initialize to 0
        ia_1, ia_2 = 0, 0
        for j_agent_id in self._ordered_agent_ids:
            if j_agent_id != agent_id:
                #Calculate differences in reward traces, add
                max_diff_1 = np.maximum(self.reward_traces[j_agent_id] - et_i, 0)
                max_diff_2 = np.maximum(et_i - self.reward_traces[j_agent_id], 0)
                inequity_aversion += max_diff_1 + max_diff_2
                ia_1 += max_diff_1
                ia_2 += max_diff_2

        agents_count = len(self._ordered_agent_ids) - 1
        
        #Reward adjustment based on IA
        rewards[agent_id] = ri - (self.alpha / agents_count) * ia_1 - \
                              (self.beta / agents_count) * ia_2
        rewards[agent_id] = np.mean(rewards[agent_id])

    done = {'__all__': timestep.last()}
    info = {}

    observations = utils.timestep_to_observations(timestep)
    return observations, rewards, done, done, info

  def close(self):
    """See base class."""

    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""

    return self._env

  # Metadata is required by the gym `Env` class that we are extending, to show
  # which modes the `render` method supports.
  metadata = {'render.modes': ['rgb_array']}

  def render(self) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """

    observation = self._env.observation()
    world_rgb = observation[0]['WORLD.RGB']

    # RGB mode is used for recording videos
    return world_rgb

  def _convert_spaces_tuple_to_dict(
      self,
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """

    return spaces.Dict({
        agent_id: (utils.remove_unrequired_observations_from_space(input_tuple[i])
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })
