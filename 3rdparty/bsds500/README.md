*bsds500* Module
----------------
This directory contains a cleaned/optimized version of the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) boundary detection evaluation utilities; originally provided with no license. These utilities are optionally linked into the *datasets* module via a CMake option. If left out, a more naïve evaluation approach is used directly in *datasets* which gives comparable results, but way faster.

The files here are been mostly cleaned from their original versions to get rid of warnings and useless constructs, and should provide identical behavior to their original counterparts.
