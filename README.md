# mnistExample

The MNIST handwritten digits classification problem is one of the most commonly used benchmarks for introducing machine learning concepts. We present here an implementation of the mnist problem using tensorflow with a couple of different models.

We implement it first using plain tensorflow and python which you can find in the `tensorflow` directory. Secondly, we implement the same code on the C3 platform in the `C3` directory. We use the KerasPipe Type to encapsulate the tensorflow models, and create an IDXFile type to read in the idx files containin the mnist training and test data.

## Python Implementation

To execute the python implementation, first use conda to install a new environment with the packages contained in `env.yaml`. (Use `env-osx.yaml` If you're a mac user.)

```
conda create -p ./venv -f ./env.yaml
```

Then, load the environment, and launch jupyter notebook.

```
conda activate ./venv
jupyter notebook
```

Finally, open `MNIST_Example.ipynb` and execute its cells!

## C3 Implementation

To execute the C3 implementation, first provision the code in the directory `c3/dtitraining` to your C3.ai tag. Once finished, open the jupyter notebook `c3/mnistExample.ipynb` and connect to your c3 session if necessary. Then execute the notebook cells!
