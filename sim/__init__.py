"""SOS-Answered MuJoCo Gymnasium environment for Unitree G1 rescue training."""


def __getattr__(name):
    if name == "SOSRescueEnv":
        from sim.env import SOSRescueEnv
        return SOSRescueEnv
    raise AttributeError(f"module 'sim' has no attribute {name!r}")


__all__ = ["SOSRescueEnv"]
