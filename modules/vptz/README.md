LITIV *vptz* Module
-------------------
This module contains a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers. It builds as a standalone dynamic library (or DLL with proper export symbols) if specified via CMake option. This module is only available if all OpenGL dependencies are found by CMake.

It was originally developed by Gengjie Chen as part of an intership at the LITIV in the summer of 2014, and presented in [ICIP2015](http://www.polymtl.ca/litiv/doc/ChenetalICIP2015.pdf); is it now kept up-to-date with bugfixes as part of the LITIV framework.

For examples on how the various classes defined here should be used, see the applications in the [*apps/vptz*](../../apps/vptz) directory.
