*litiv2018viewer* sample
------------------------
This sample demonstrates how to parse and visualize the litiv-stcharles2018 dataset via [the interface provided in the datasets module](../../modules/datasets/include/litiv/datasets/impl/litiv-stcharles2018.hpp). Two modes of operation are supported in this sample; the first displays the unrectified data directly with its foreground segments as overlay, and the second displays the rectified data with the registered point pairs used for evaluation. If the dataset cannot be located at its default location on the system (determined via the CMake variable 'EXTERNAL_DATA_ROOT'), then the user will be asked to locate it.

For more examples of specialized dataset interfaces, see the [impl](../../modules/datasets/include/litiv/datasets/impl/) folder in the *datasets* module.
