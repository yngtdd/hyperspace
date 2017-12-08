# HyperSpace

### Welcome to Hyperspace!

If you have a complicated model with many hyperparameters, there is a lot to explore here.
A combinatorial explosion, in fact. Here is a *Hitchhiker's Guide*:

### Preparations

You can't just waltz out into space without the proper preparations.

```
git clone https://code-int.ornl.gov/ygx/hyperspace/
cd hyperspace

# Get you gear!
pip install .
```

### Modules

A Hubble height view of the library. For details, see our [wiki](https://code-int.ornl.gov/ygx/hyperspace/wikis/home).

### Space

> _"Space," it says, "is big. Really big. You just won't believe how vastly, hugely,
mindbogglingly big it is. I mean, you may think it's a long way down the road to the
chemist's, but that's just peanuts to space." - The Hitchhiker's Guide to the Galaxy_

In [space](https://code-int.ornl.gov/ygx/hyperspace/tree/ygx_hyperV2/hyperspace/space/space.py)
you will find the various classes that define hyperparameter search spaces.

### Mapping Space

In [mapping_space](https://code-int.ornl.gov/ygx/hyperspace/blob/ygx_hyperV2/hyperspace/space/mapping_space.py)
we have functions that define hyperspaces, the many subregions
of our hyperparameter search space to be distributed across cluster resources.


### Hyperdrive

In [hyperdrive](https://code-int.ornl.gov/ygx/hyperspace/tree/ygx_hyperV2/hyperspace/hyperdrive/hyperdrive.py)
we have various methods for distributing our optimization procedure.
