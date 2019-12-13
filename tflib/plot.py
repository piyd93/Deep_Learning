import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle

_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_gen_begin=collections.defaultdict(lambda: {})
_gen_last_flush=collections.defaultdict(lambda: {})

_iter = [0]

output_dir = '.'

def tick():
    _iter[0] += 1

    
def plot_generator(name,gen_value):
    _gen_last_flush[name][_iter[0]] = gen_value
    
    
def flush_generator():
    prints = []
    print(_gen_last_flush.items())

    for name, vals in _gen_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _gen_begin[name].update(vals)

        x_vals = np.sort(list(_gen_begin[name].keys()))
        y_vals = [_gen_begin[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(output_dir, name.replace(' ', '_')+'.jpg'))

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _gen_last_flush.clear()

    with open(os.path.join(output_dir, 'log_gen.pkl'), 'wb') as f:
        pickle.dump(dict(_gen_begin), f, pickle.HIGHEST_PROTOCOL)
        
        
def plot(name, value):
    _since_last_flush[name][_iter[0]] = value
    

def flush():
    prints = []

    print(_since_last_flush.items())

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _beginning[name].update(vals)

        x_vals = np.sort(list(_beginning[name].keys()))
        y_vals = [_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(output_dir, name.replace(' ', '_')+'.jpg'))

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    with open(os.path.join(output_dir, 'log.pkl'), 'wb') as f:
        pickle.dump(dict(_beginning), f, pickle.HIGHEST_PROTOCOL)
