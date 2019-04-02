import os
import json
import numbers

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


class JsonCheckpointSaver(object):
    """
    Save current state after each iteration with JSON format.

    Example usage:
        import skopt
        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])

    Parameters
    ----------
    * `checkpoint_path` : str
        location where checkpoint will be saved to;

    * `filename` : str
        Name of the file to save.
    """
    def __init__(self, checkpoint_path, filename):
        self.checkpoint_path = checkpoint_path
        self.filename = filename
        self.savefile = os.path.join(self.checkpoint_path, self.filename)

    def _convert_fields(self, result_field):
        """
        Convert numpy types to python objects.
        Necessary to serialize results.

        Parameters
        ----------
        * `result_field` : list
            Field consisting of numpy types to be converted.
        """
        converted_field = []
        for dim in result_field:
            if isinstance(dim, numbers.Integral):
                converted_field.append(int(dim))
            elif isinstance(dim, numbers.Real):
                converted_field.append(float(dim))
            else:
                converted_field.append(dim)

        return converted_field

    def __call__(self, res):
        """
        Save results to disk in JSON format.

        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        data = {}
        data['fun'] = float(res.fun)
        data['x'] = self._convert_fields(res.x)
        data['func_vals'] = self._convert_fields(res.func_vals.tolist())
        data['x_iters'] = [self._convert_fields(x) for x in res.x_iters]

        with open(self.savefile, 'w') as outfile:
            json.dump(data, outfile)
