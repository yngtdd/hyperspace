===========================
Hyperparameter Optimization
===========================

Machine learning (ML) models often contain numerous hyperparameters, free parameters that must be set before the models can be trained. Optimal settings for these hyperparameters are rarely known a priori, though their settings often dictate our algorithms' ability to learn from data.
The challenge of hyperparameter tuning for ML algorithms can be attributed to several factors: (1) heterogeneity in the  types of hyperparameters, (2) the potentially complex interactions between hyperparameters, and (3) the computational expense inherit to hyperparameter optimization. Though there has been considerable progress in hyperparameter optimization, optimization in this space remains hard.

As :citet:`snoek2012practical` note, hyperparameters are often considered nuisances. However, this belies the fact that hyperparameters are design choices made by ML practitioners. When models consist of many hyperparameters, the differences between models presented across the literature could boil down to small changes in model architectures. If it is reasonable to expect that ML practitioners should be able to defend their choice of one model over others, it is also reasonable to expect that their choices for hyperparameter settings be defensible. It could even be the case that hyperparameter optimization could change a seemingly poor model choice into the best performing model for a given problem. We believe that careful study of our models' hyperparameters allows us to make more informed choices when modeling. It pays to know how our models' behavior changes as we vary our hyperparameters.


As the number of model hyperparameters increases, their optimization becomes significantly more challenging as we face a combinatorial increase in potential model configurations. Similarly, there is an increased chance that our models' hyperparameters interact in complex ways. Modeling these interactions in high-dimensional search spaces quickly becomes challenging and can often defy our intuition. Even experts find it difficult to manually configure hyperparameters :cite:`snoek2012practical`.


While optimizing model performance with respect to hyperparameters is an obvious goal, we posit that hyperparameter optimization techniques must provide insight into how ML models' behavior changes across a range of possible model configurations. We argue that simply finding an optimal setting for our hyperparameters over a given training and validation set is rather uninformative. We may wonder, how much would our models' performance change if we just slightly perturbed our model hyperparameters? Would a small change in our hyperparameters mean a significant change in our models' behavior? After running through some fixed number of optimization iterations, do we believe that a particular setting of our hyperparameters is unique in that it enables our models to generalize well, or could there be several settings of hyperparameters that perform reasonably well? If we found that there are several good settings of hyperparameters, would they have anything in common? We believe these are important questions and that their answers could lead us to a better understanding of the ML models we employ.

In order to take on these challenges, we developed HyperSpace, a parallel 
Bayesian model based optimization library.


.. rubric:: References

.. bibliography:: refs_results.bib
   :style: plain
   :cited:
