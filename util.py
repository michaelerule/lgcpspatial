#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

"""
`util.py`: Miscellaneous utility functions used in the notebooks
"""

# For progress bar and code timing
import time

# Load a Matlab-like namespace
from pylab import *

# For the statistical summary function
from scipy.stats import pearsonr

# used by the cinv function
import scipy.linalg.lapack

# Jax's fft is slow
# Numpy's fft pathologically casts to float64
# Scipy's seems ok
from scipy.fft import *

ttic = None
def tic(msg=''):
    '''
    ----------------------------------------------------------------------------
    Timer routine to track performance
    '''
    global ttic
    t = time.time()*1000
    if ttic:
        elapsed = t-ttic
    else:
        elapsed = None
    if ttic and msg: 
        print(('Δt = %d ms'%elapsed).ljust(14)\
              +'elapsed for '+msg)
    ttic = t
    return elapsed

def progress_bar(x,N=None):
    '''
    ----------------------------------------------------------------------------
    Progress bar wrapper for loops
    '''
    if N is None:
        x = list(x)
        N = len(x)
    K = int(np.floor(np.log10(N)))+1
    pattern = ' %%%dd/%d'%(K,N)
    wait_til_ms = time.time()*1000
    for i,x in enumerate(x):
        time_ms = time.time()*1000
        if time_ms>=wait_til_ms:
            r = i*50/N
            k = int(r)
            q = ' ▏▎▍▌▋▊▉'[int((r-k)*8)]
            print('\r['+('█'*k)+q+(' '*(50-k-1))+']%3d%%'%(i*100//N)+(pattern%i),end='',flush=True)
            wait_til_ms = time_ms+250
        yield x
    print('\r'+' '*70+'\r',end='',flush=True)

def zgrid(L):
    '''
    ----------------------------------------------------------------------------
    2D grid coordinates as complex numbers
    '''
    c = arange(L)-L//2
    return 1j*c[:,None]+c[None,:]

def printstats(a,b,message='',mask=None):
    '''
    ----------------------------------------------------------------------------
    Print RMSE and correlation between two rate maps
    '''
    a,b = a.ravel(),b.ravel()
    if not mask is None:
        a = a[mask.ravel()]
        b = b[mask.ravel()]
    NMSE = mean((a-b)**2)/mean(a**2)#sqrt(mean(a**2)*mean(b**2))
    print(message+':')
    print('∙ Normalized MSE: %0.1f%%'%(100*NMSE))
    print('∙ Pearson correlation: %0.2f'%pearsonr(a,b)[0])

def find(a):
    '''
    ----------------------------------------------------------------------------
    The old "find" syntax is cleaner!
    
    Parameters
    ----------
    '''
    return np.where(np.array(a).ravel())[0]

# For jupyter notebooks: trigger browser to notify when done
def speak(text):
    '''
    ----------------------------------------------------------------------------
    
    Parameters
    ----------
    '''
    from IPython.display import Javascript as js, clear_output
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js('''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''.format(text=text)))
    # Clear the JS so that the notebook doesn't speak again when reopened/refreshed
    # clear_output(False)
    
def notify(what='attention'):
    '''
    ----------------------------------------------------------------------------
    
    Parameters
    ----------
    '''
    #os.system("echo -n '\a'")
    speak(what+'!')

