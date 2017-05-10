# Weight Normalized GAN
PyTorch code for the paper "[On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks]( https://arxiv.org/abs/1704.03971)".

The code used for the experiments in the paper was written in torch and was messy, and this is a clean version ported to PyTorch. There are some implementation details not mentioned in the paper (will be added) and we are still making sure that everything works exactly the same as before, but this should be able to reproduce the results.

## Usage notes
Before training, you need to split the dataset into training data and test data. Running `split_data.py` will create an index in the root directory of the dataset.

The PyTorch LSUN loader creates a cache in the working directory if there isn't one. It takes some time. The loader for custom dataset from a image folder requires images of each class to be in one subfolder, so if you use say CelebA where there is no classes you need to manually create a dummy class.

To train, run `main.py`. See the code for a explanation of arguments. The only ones you must specify are the `--dataset`, `--dataroot`, `--save_path` and `--image_size`. By default it trains a vanilla model. Use `--norm batch` or `--norm weight` to try different normalizations.

Give `--load_path` to continue a saved training.

To test a trained model, use `--final_test`. Make sure to also use a larger `--test_steps` since the default value is for the running test during training.

Use `plot.py` to plot the loss curves. Requires PyGnuplot.
