# Analyzing the Quality of Concepts Learned by Self-Explainable and Foundation Models
This code allows to train explainable models (GlanceNet, CBM) and Visual-Language Concept Bottleneck Models on two possible datasets: Shapes3d and CelebA. It also allows to collect the metrics relevant to assess concept quality: disentanglement, completeness, leakage.
## Instructions
# Train models
Run `train_bash.sh` 

You can edit the file to change the training parameters. It starts training all the different models that are added to the file.
# Evaluate
Run `show_results.py`.

The output is a preformatted latex table. All the metrics are also stored in the model checkpoint's folder.

## Master Thesis
This is the work for my Master thesis at the University of Trento in 2024.

## References
Part of the code was taken from other sources. I will list them here:

[GlanceNet](https://arxiv.org/abs/2205.15612), [Emanuele Marconato](https://github.com/ema-marconato)

[Language in a Bottle](https://arxiv.org/abs/2211.11158)

[Label-free Concept Bottleneck Models](https://openreview.net/pdf?id=FlCg47MNvBA)

[DCI-ES: An Extended Disentanglement Framework with Connections to Identifiability](https://arxiv.org/abs/2210.00364)




