import jax
import jax.numpy as jnp
import numpy as np

from numpyro.primitives import Messenger


class condition(Messenger):
    """
    Same as numpyro.handlers.condition except that it conditions on
    both sample and deterministic sites.
    """

    def __init__(self, fn=None, data=None, condition_fn=None):
        self.condition_fn = condition_fn
        self.data = data
        if sum((x is not None for x in (data, condition_fn))) != 1:
            raise ValueError(
                "Only one of `data` or `condition_fn` " "should be provided."
            )
        super(condition, self).__init__(fn)

    def process_message(self, msg):
        if (msg["type"] not in ["sample", "deterministic"]) or msg.get("_control_flow_done", False):
            if msg["type"] == "control_flow":
                if self.data is not None:
                    msg["kwargs"]["substitute_stack"].append(("condition", self.data))
                if self.condition_fn is not None:
                    msg["kwargs"]["substitute_stack"].append(
                        ("condition", self.condition_fn)
                    )
            return

        if self.data is not None:
            value = self.data.get(msg["name"])
        else:
            value = self.condition_fn(msg)

        if value is not None:
            msg["value"] = value
            msg["is_observed"] = True

