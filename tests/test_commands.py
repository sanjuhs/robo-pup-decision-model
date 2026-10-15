import numpy as np
import pytest

from robopup.commands import interpret
from robopup.env import RoboPupEnv


def test_ball_commands_have_honest_replies_and_explicit_waypoints():
    command = interpret("Miso, go eat the ball!", [1.8,-1,.12], [0,0,.4])
    assert command.mode == "approach" and len(command.waypoints) == 1
    assert "don't eat" in command.reply
    command = interpret("go behind the ball", [1.8,-1,.12], [0,0,.4])
    assert len(command.waypoints) == 3
    assert command.waypoints[-1][0] > 1.8


def test_unknown_commands_do_not_trigger_actions():
    assert interpret("stop please", None, [0,0,.4]).mode == "stop"
    with pytest.raises(ValueError):
        interpret("fly to the moon", None, [0,0,.4])
    with pytest.raises(ValueError):
        interpret("go to ball", None, [0,0,.4])


def test_ball_is_a_dynamic_physical_body():
    env = RoboPupEnv(obstacles=True)
    env.reset(seed=1)
    joint = env.model.body_jntadr[env.ball]
    dof = env.model.jnt_dofadr[joint]
    env.data.qvel[dof] = 1.
    initial = env.data.xpos[env.ball,0]
    for _ in range(30):
        env.step(np.zeros(12))
    assert env.data.xpos[env.ball,0] > initial+.1
