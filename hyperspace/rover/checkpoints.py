import os
from skopt.utils import dump


class CheckpointSaver(object):
    """
    Save current state after each iteration with `skopt.dump`.

    Example usage:
        import skopt
        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])

    Parameters
    ----------
    * `checkpoint_path`: 
        location where checkpoint will be saved to;

    * `dump_options`: 
        options to pass on to `skopt.dump`, like `compress=9`
    """
    def __init__(self, checkpoint_path, filename, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.filename = filename 
        self.savefile = os.path.join(self.checkpoint_path, self.filename)
        self.dump_options = dump_options

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        dump(res, self.savefile, **self.dump_options)
