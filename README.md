# Robust-Contrastive-Representation-Learning-on-Bipartite-Graphs-with-Dynamic-Adversarial-Perturbation
This repo employs an adversarial mechanism to directly perturb bipartite node embeddings in the embedding space used for contrastive learning.


How to use:
1- Install the requirements using:
~~~
conda env create -f requirements.yml -n newname
~~~

2- To run recommendation for ML-100K dataset use adgclrec.py

3- To run link prediction for Wiki dataset use adgcllp.py

***NOTE***:
The code will become more modular in future updates. To alternate between Wiki splits or tuning hyperparameters, you have to change them in the script.
At the moment, the scripts may generate a "build" warning. This happens when the main optimizer sees the epsilon trainable parameters, but it can't update them during training. You can ignore this as the epsilon values are updated using the generator optimizer.
