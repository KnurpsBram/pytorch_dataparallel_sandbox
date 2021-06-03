# Pytorch DistributedDataParallel Sandbox
How does `nn.DistributedDataParallel` work? What batch size and learning rate does it use internally. If I can increase my effective batch size, how should the learning rate follow? This repo may help you try some stuff out.

### Solving a Toy Problem on a Multi-GPU machine

In this example we train a neural network with a single neuron to map inputs of 1 to targets of 0. The neuron is initialised as 1. We pretend that we know the optimal hyperparameters to solve this problem are `batch_size=128` and `lr=1e-3`.

```
$ python single_gpu.py --lr 1e-3 --batch_size 128
my_net.w:              tensor(0.9980, device='cuda:0')
update step variance:  tensor(0., device='cuda:0')
```

Imagine that `batch_size=128` is too big to fit on a single gpu. In order to make it happen, we need to run this experiment on a multi-gpu machine. We want to make sure that the model's learning trajectory is equivalent to what it would be on a hypothetical single-gpu machine that could fit the full batch size of 128. The `batch_size` we submit to `nn.DistributedDataParallel` is the batch size per GPU. The effective batchsize is this `batch_size` times the amount of GPU's on this machine.

```
python multi_gpu.py --lr 1e-3 --batch_size 16
World Size:  8
my_net.w:              tensor(0.9980, device='cuda:0')
update step variance:  tensor(0., device='cuda:0')
```

Not only the value for the model's weight after training, but also the variance in update steps during training is equivalent between these runs. In order to see the variance, we evaluate a non-deterministic version of our toy problem. We add random uncorrelated gaussian noise to our inputs and targets. We take the average over many experiments.

```
$ python single_gpu.py --lr 1e-3 --batch_size 128 --deterministic False --n_experiments 10000
my_net.w:              tensor(0.9981, device='cuda:0')
update step variance:  tensor(8.3733e-09, device='cuda:0')

$ python multi_gpu.py --lr 1e-3 --batch_size 16 --deterministic False --n_experiments 10000
World Size:  8
my_net.w:              tensor(0.9981, device='cuda:0')
update step variance:  tensor(8.4037e-09, device='cuda:0')
```

It's wonderful if you know what your optimal learning rate is for the hypothetical situation of `batch_size=128` on a single-gpu machine, but it's likely that you only know the optimal learning rate for `batch_size=16` on a single-GPU. Let's reset our assumptions and pretend that we know the optimal hyperparameters to solve our problem are `batch_size=16` and `lr=1e-3`.
```
$ python single_gpu.py --lr 1e-3 --batch_size 16
my_net.w:              tensor(0.9840, device='cuda:0')
update step variance:  tensor(0., device='cuda:0')
```
Imagine that we've now got our hands on an esteemed multi-gpu machine so that we can pump the effective batch size up to 128 and really speed up training. How should the learning rate be adjusted for that larger batch size? In our toy example, a linear scaling rule would get us to the same model weight after training.
```
$ python multi_gpu.py --lr 8e-3 --batch_size 16
World Size:  8
my_net.w:              tensor(0.9840, device='cuda:0')
update step variance:  tensor(0., device='cuda:0')
```
There's a catch though.  If we scale the learning rate linearly we can expect a more noisy learning trajectory, because the variance of the update step doesn't increase linearly with an increase in batch size.
```
$ python single_gpu.py --lr 1e-3 --batch_size 16 --deterministic False --n_experiments 1000
my_net.w:              tensor(0.9847, device='cuda:0')
update step variance:  tensor(6.7332e-08, device='cuda:0')

$ python multi_gpu.py --lr 8e-3 --batch_size 16 --deterministic False --n_experiments 1000
World Size:  8
my_net.w:              tensor(0.9848, device='cuda:0')
update step variance:  tensor(5.5736e-07, device='cuda:0')
```
In order to keep the expected variance of the update steps at roughly the same value, we should scale the learning rate with the square root of the upscaling factor in effective batch size, which is `sqrt(8)=2.8284` in our case.
```
python multi_gpu.py --lr 2.8284e-3 --batch_size 16 --deterministic False --n_experiments 1000
World Size:  8
my_net.w:              tensor(0.9946, device='cuda:0')
update step variance:  tensor(6.5220e-08, device='cuda:0')
```
This keeps the variance in check, but you won't converge to the same model weight. You'll need more epochs to get there.

There's good arguments for linear scaling of the learning rate and there's good arguments for a square root scaling rule. Other sources may yet propose other scaling rules. The rules also appear to be different if you're using an optimizer that's more complicated than SGD, like the ADAM optimizer. As of yet no one appears to have the definitive answer.

It's important you read up on the scaling of learning rate with batch size, for instance [here](https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change/53046624), but it's most important that you try a few hyperparameter settings to find out which work best for your specific problem, model and data.

If you've decided what you want your learning rate to be for a specific effective batch size, you should use `nn.DistributedDataParallel` with that learning rate and a `batch_size` that's your effective batch size divided by the amount of GPU's on your machine.
