*dataset_simple*
----------------
This sample demonstrates how to set up a custom dataset to be used with a litiv algorithm (in this case, an edge detector). All the data required here is already located in the samples' data directory. See the [source code](./src/main.cpp) comments for more details.

The custom dataset created here does not include categories or groundtruth; examples which include these can be found in the pre-implemented dataset interface specializations in the datasets module (e.g. CDnet or BSDS500).

For an example on how to use a pre-implemented specialization from the module, see the 'changedet' or 'edges' projects in the 'apps' directory.
