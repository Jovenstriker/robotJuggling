from juggling.env import SoccerJuggleEnv
from juggling.walk_env import SoccerJuggleWalkEnv

TASKS = {"static": SoccerJuggleEnv, "walk": SoccerJuggleWalkEnv}


def make_env(task="static", **kwargs):
    """`static`: fixed torso (kwarg `legs`=1|2). `walk`: free-standing planar robot."""
    return TASKS[task](**kwargs)


__all__ = ["SoccerJuggleEnv", "SoccerJuggleWalkEnv", "TASKS", "make_env"]
