# Introduction 

## Aim

This template is intented to quickly bootstrap a PyTorch template with all the **necessary**
boilerplate that one typically writes in every project.

### Why this one?

While [this big boi]() is beautifully written, mature, highly functional and comes with all the
features you could ever want, I find that because of all the abstraction it actually becomes a
hindrance when writing research code. Unless you know the template on the tip of your fingers, it's
too much hassle to look for the right file, the right method where you have to put your code. And
PyTorch Lightning? Great for building stuff, not so much for researching new Deep Learning
architectures...

This template wants to write the **necessary boilerplate** for you, while **staying out of your
way**.


## Structure

TODO: Actually come up with a proper project structure
```
conf/
    experiment/
        base.yaml
    config.yaml
model/
    my_module.py
dataset/
    my_dataset.py
utils/
    plotting.py
scripts/
    resize_dataset.py
train.py 
test.py
```
