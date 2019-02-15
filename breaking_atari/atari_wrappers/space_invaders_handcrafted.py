import gym
from breaking_atari.atari_wrappers.utils import compute_bunker_health, compute_agent_x_position, Template

class SpaceInvadersHandcrafted(gym.env):
    def __init__(self, sprites_dir):
        self._env = gym.make('SpaceInvadersDeterministic-v4')
        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(1 + 3 + 6*6) # agent-x, bunker health and enemy grid
        self.bunker_template = Template(sprites_dir + 'defense.png', 'barrier')
        self.agent_template = Template(sprites_dir + 'my_sprite.png', 'agent')
        self.enemy_templates = [
            Template(sprites_dir + 'enemy_0_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_0_b.png', 'enemy'),
            Template(sprites_dir + 'enemy_1_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_1_b.png', 'enemy'),
            Template(sprites_dir + 'enemy_2_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_2_b.png', 'enemy'),
            Template(sprites_dir + 'enemy_3_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_3_b.png', 'enemy'),
            Template(sprites_dir + 'enemy_4_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_4_b.png', 'enemy'),
            Template(sprites_dir + 'enemy_5_a.png', 'enemy'),
            Template(sprites_dir + 'enemy_5_b.png', 'enemy')
        ]

    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        obs = self._env.reset()

    def _get_features(self, obs):
        bunkers = compute_bunker_health(self.bunker_template.image, obs)
        agent_x_pos = compute_agent_x_position(self.agent_template.image, obs)
        enemies_locations = None