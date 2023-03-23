***This is still very much under construction; please check back later!***


# lgcpspatial

Log-Gaussian Cox process Python library designed for analyzing grid cells (and perhaps other periodic, densely sampled 2D spatial datasets).
[Please see the documentation on Github pages.](https://michaelerule.github.io/lgcpspatial/index.html)



If you're brave enough to test it, please get in touch. I need feedback on
 
 - Does this actually work outside of the datasets its been tested on?
 - Does this actually perform well on other hardware configurations?
 - How hard is it to start using? Is it missing dependencies? 
 - Are there simple ways to improve the API? 

There will be bugs and issues; Let's work them out before this thing goes to peer review. I don't have much time these days, but will be able to make a few more passes to tidy things up before we reach a "version of record".


One common pitfall of code like this: The kernels have only two parameters, but the optimization routines have about two-dozen configurable values, which could affect speed, accuracy, and stability for your particular use-case. I've dialed these in for the test data I have, but they may not be universal. If you have other data that seems to benefit from different defaults, let me know. It would be good to make all of this as painless as possible.
