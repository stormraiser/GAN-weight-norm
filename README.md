# Weight Normalized GAN
Code for the paper "[On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks]( https://arxiv.org/abs/1704.03971)".

## About the two different versions
Here two versions are provided, one for torch and one for PyTorch.

The code used for the experiments in the paper was in torch and was a bit messy, with hand written backward pass of weight normalized layers. So we decided to clean up the code and port it to PyTorch. However, as of now we are not yet able to exactly reproduce the results in the paper with the PyTorch code. In particular, it consistantly gives better reconstruction loss on CelebA but generate samples with worse visual quality. We've checked all the implementation details, including those not mentioned in the paper (will be added), but found no difference.

It could be a numerical issue since the gradient are not computed in exactly the same way. Or I might have made stupid mistakes as I have been doing machine learning for only half a year. We are still investigating.

In short, to reproduce the results in the paper, use the torch version.

## Usage
The two versions accept the exact same set of arguments. Check the details in the source code.

Before training, you need to prepare the data. For torch you need [lmdb.torch](https://github.com/eladhoffer/lmdb.torch) for LSUN and [cifar.torch](https://github.com/soumith/cifar.torch) for CIFAR-10. Split the dataset into training data and test data with `split_data.py`.

The LSUN loader creates a cache if there isn't one. It takes some time. The loader for custom dataset from a image folder requires images of each class to be in one subfolder, so if you use say CelebA where there is no classes you need to manually create a dummy class.

To train, run `main.py`. See the code for a explanation of arguments. The only ones you must specify are the `--dataset`, `--dataroot`, `--save_path` and `--image_size`. By default it trains a vanilla model. Use `--norm batch` or `--norm weight` to try different normalizations.

Give `--load_path` to continue a saved training.

To test a trained model, use `--final_test`. Make sure to also use a larger `--test_steps` since the default value is for the running test during training.

Use `plot.py` to plot the loss curves. The PyTorch version uses [PyGnuplot](https://pypi.python.org/pypi/PyGnuplot).

## Notes
The WN model might fail in the first handful of iterations. This happens especially often if the network is deeper (on LSUN). Just restart training. If it get past iteration 5 it should continue to train without trouble. This effect could be reduced by using a much smaller learning rate for the first say 100 iterations.
