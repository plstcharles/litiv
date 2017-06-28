
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////
//
// This sample demonstrates two things: how to set up a custom dataset to be
// used with a litiv algo (in this case, an edge detector), and how to create
// a dataset specialization with custom parsing routines. You can toggle
// between the two using the 'USE_MIDDLEBURY_SPECIALIZATION' define. All the
// data required here is already located in the 'samples/data' directory.
//
// By default, datasets created at run-time cannot parse ground truth, but
// specialized datasets (such as the Middlebury2005 demo below) can, with
// your own code.
//
/////////////////////////////////////////////////////////////////////////////

#define USE_MIDDLEBURY_SPECIALIZATION 1

#include "litiv/datasets.hpp" // includes all datasets module utilities (along with pre-implemented dataset specializations)
#if USE_MIDDLEBURY_SPECIALIZATION
#include "middlebury2005.hpp" // includes a custom dataset specialization used to parse middlebury stereo 2005 two-views data
#else //!USE_MIDDLEBURY_SPECIALIZATION
#include "litiv/imgproc.hpp" // includes all edge detection algos, along with most core utility & opencv headers
#endif //!USE_MIDDLEBURY_SPECIALIZATION

int main(int, char**) { // this sample uses no command line argument
    try { // its always a good idea to scope your app's top level in some try/catch blocks!
        std::cout << "\nNote: a directory will be created at '" << lv::getCurrentWorkDirPath() << "'\n" << std::endl;

#if USE_MIDDLEBURY_SPECIALIZATION

        // The 'DatasetType' alias below is only used to simplify templating; the 'Dataset_' interface
        // has three enum template parameters, namely the dataset task type ('eDatasetTask'), the dataset
        // identifier ('eDataset'), and the implementation type ('eEvalImpl'). For this example, we ask for
        // the stereo registration task interface as well as our own specialized dataset implementation, using
        // our special predefined dataset ID. Finally, we only require a traditional evaluation approach
        // (i.e. not asynchronous), so we use 'NonParallel'.
        using DatasetType = lv::Dataset_<lv::DatasetTask_StereoReg,lv::Dataset_Middlebury2005_demo,lv::NonParallel>;

        // Next, creating the dataset will automatically create work batches, and parse the data for each using the specialized functions from 'middlebury2005.hpp'.
        DatasetType::Ptr pDataset = DatasetType::create("results_test",true); // 'results_test' is the name of the output folder where logs/results will be saved, and we set eval-result to true
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false); // returns a list of all work batches in the dataset without considering hierarchy
        cv::Ptr<cv::StereoBM> pStereoMatcher = cv::StereoBM::create(); // creates a simple stereo matcher algorithm to evaluate some disparity maps!
        lvAssert__(vpBatches.size()>0 && pDataset->getInputCount()>0,"Could not parse any data for dataset '%s'",pDataset->getName().c_str()); // check that data was indeed properly parsed
        for(auto& pBatch : vpBatches) { // loop over all batches (or over all image array sets, in this case)
            DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch); // cast the batch object for full task-specific interface accessibility
            std::cout << "\tProcessing batch '" << oBatch.getName() << "'" << std::endl;
            oBatch.startProcessing(); // initializes the internal pushed packet counter & timers to allow processing time computation (this would be before the packet loop, if we had one)
            lvAssert_(oBatch.getInputCount()==1 && oBatch.getGTCount()==1,"bad packet count"); // each work batch of the middlebury dataset has a single packet (i.e. a stereo array)
            const size_t nPacketIdx = 0; // this is the only packet index that will be used here, but if there were more, we could use a loop below
            const std::vector<cv::Mat>& vImages = oBatch.getInputArray(nPacketIdx); // will return the input array packet to be processed
            lvAssert_(vImages.size()==2,"bad packet array size"); // the array should only contain two matrices (one for each stereo head)
            // if we needed raw GT frames for our own evaluation, we could query them via oBatch.getGTArray(...)
            std::vector<cv::Mat> vGrayImages(vImages.size()); // will hold the images converted to grayscale for StereoBM
            for(size_t nImageIdx=0; nImageIdx<vImages.size(); ++nImageIdx)
                cv::cvtColor(vImages[nImageIdx],vGrayImages[nImageIdx],cv::COLOR_BGR2GRAY); // converts images to required input type by StereoBM
            std::vector<cv::Mat> vOutputMaps(vImages.size());
            pStereoMatcher->compute(vGrayImages[0],vGrayImages[1],vOutputMaps[0]); // computes & return left disparity map using the input images
            std::vector<cv::Mat> vFlippedGrayImages(vImages.size()); // used to hold flipped inputs to allow right disparity map computation
            for(size_t nImageIdx=0; nImageIdx<vImages.size(); ++nImageIdx)
                cv::flip(vGrayImages[nImageIdx],vFlippedGrayImages[nImageIdx],1); // flip with ID=1 flips on the vertical axis
            pStereoMatcher->compute(vFlippedGrayImages[1],vFlippedGrayImages[0],vOutputMaps[1]); // computes & return right disparity map using the flipped input images
            cv::flip(vOutputMaps[1],vOutputMaps[1],1); // flips right disparity map back to its original state
            for(size_t nImageIdx=0; nImageIdx<vImages.size(); ++nImageIdx)
                vOutputMaps[nImageIdx].convertTo(vOutputMaps[nImageIdx],CV_32F,1.0f/16); // scales disparity maps for proper evaluation/display in float format
            const std::vector<cv::Mat> vOutputDisplays = oBatch.getColoredMaskArray(vOutputMaps,nPacketIdx,100); // gets colored output masks to display (will use GT internally if available)
            for(size_t nStreamIdx=0; nStreamIdx<vImages.size(); ++nStreamIdx) {
                cv::imshow(oBatch.getInputStreamName(nStreamIdx),vImages[nStreamIdx]);
                cv::imshow(oBatch.getInputStreamName(nStreamIdx)+" disp",vOutputMaps[nStreamIdx]/100.0f); // for a rough 0-100 normalized disparity map
                cv::imshow(oBatch.getInputStreamName(nStreamIdx)+" eval",vOutputDisplays[nStreamIdx]);
            }
            oBatch.push(vOutputMaps,nPacketIdx); // pushes the output disparity maps to the dataset for automatic evaluation
            oBatch.stopProcessing(); // stops processing this work batch (if we had lots of packets, this would be outside the packet loop)
            std::cout << "\t\tdone in " << oBatch.getFinalProcessTime() << " seconds; press any key to continue..." << std::endl;
            cv::waitKey(0);
        }
        std::cout << "All done!\n" << std::endl;
        pDataset->writeEvalReport(); // writes the evaluation report for all pushed disparity maps (by default, will be located in the output directory given to the constructor)

#else //!USE_MIDDLEBURY_SPECIALIZATION

        // The 'DatasetType' alias below is only used to simplify templating; the 'Dataset_' interface
        // has three enum template parameters, namely the dataset task type ('eDatasetTask'), the dataset
        // identifier ('eDataset'), and the implementation type ('eEvalImpl'). For this example, we use
        // an edge detection algo, so we set the task type as 'DatasetTask_EdgDet'; we are also defining
        // a custom run-time dataset, so we set the dataset identifier as 'Dataset_Custom'; finally, we
        // only require a traditional evaluation approach (i.e. not asynchronous), so we use 'NonParallel'.
        using DatasetType = lv::Dataset_<lv::DatasetTask_EdgDet,lv::Dataset_Custom,lv::NonParallel>;

        // The line below creates an instance of the dataset for parsing/evaluation, using the same template
        // parameters we used above. Since rely on the built-in custom dataset parser, we have to respect
        // the directory structure expected by the parser to make sure all data can be found automatically:
        //
        //  <SAMPLES_DATA_ROOT>/custom_dataset_ex/    => this is the 'dataset root' folder, in which the named batch directories (provided in the constructor below) can be found
        //        |------batch1/                      =< this is the first work batch in the dataset; it could contain a set of training/testing images, or images of a specific category requiring independent evaluation
        //        |        |----- (some image).jpg    => all data packets (or images, in this case) will be assigned a packet index based on the order they are parsed in; this first image would have ID=1 in 'batch1' (and so on)
        //        |        |----- (some image).jpg    => images do not need to all be using the same container (e.g. jpg's can be mixed with png's)
        //        |        \----- (some image).jpg
        //        |
        //        |------batch2/                      => second work batch; we could split it into sub-batches by creating subdirectories here, in which the actual images would be
        //        |        \----- (some image).jpg    => since packet indices are unique at the batch-level, this image would also have ID=1, but it would be tied to 'batch2' for reference
        //        |
        //        \------batch3/                      => third and final work batch; note that these batches do not need to be the same size
        //                 |----- (some image).jpg
        //                 \----- (some image).jpg
        //
        // Note that it may be possible to simultaneously parse input and groundtruth data for some dataset
        // task types using the run-time custom dataset parser. However, building a specialized dataset
        // inferface offers a lot more flexibility, and it should be considered the only true solution. For
        // more information on automatic groundtruth parsing, you will have to dig in the datasets module
        // source code (check the classes derived from the IIDataLoader super-interface primarily).
        //
        // Besides, we need to provide arguments to the 'DatasetType::create' function to guide the automatic
        // parser; these arguments are sent to the constructor of the 'IDataset_' interface via pass-through
        // constructors, or caught by overrides in the interface of pre-implemented specializations (based on
        // the dataset identifier). These arguments (along with their parameter name) are described below:
        //
        //   1.   "Custom Dataset Example"                    => const std::string& sDatasetName => verbose name of the dataset, used for display/debug/logging purposes only
        //   2.   "<SAMPLES_DATA_ROOT>/custom_dataset_ex/"    => const std::string& sDatasetDirPath => full path to the dataset's top-level directory, where work batches can be found
        //   3.   "results_test"                              => const std::string& sOutputDirPath => full path to the output directory, where logs/evaluation results will be written
        //   4.   {"batch1","batch2","batch3"}                => const std::vector<std::string>& vsWorkBatchDirs => list of dataset directory names to be treated as work batches
        //   5.   {}                                          => const std::vector<std::string>& vsSkippedDirTokens => list of tokens which, if found in a directory/batch name, will remove it from the dataset
        //   6.   true                                        => bool bSaveOutput => defines whether the processed output should be automatically saved when pushed for evaluation
        //   7.   false                                       => bool bUseEvaluator => defines whether the processed output should be fully evaluated internally or not (for a custom dataset, it might still not produce anything useful without specialization)
        //   8.   false                                       => bool bForce4ByteDataAlign => defines whether data packets (typically images) should be 4-byte aligned or not --- this helps when uploading data to GPU, for example
        //   9.   1.0                                         => double dScaleFactor => defines the scaling factor to be applied to the data packets (if applicable, typically only useful for images)
        //
        // The dataset object then returned can finally be queried for data packets, and to evaluate output.
        // In our case, the data packets are simply images that we should apply edge detection on, and the
        // output is an edge detection mask.
        //
        DatasetType::Ptr pDataset = DatasetType::create(
            "Custom Dataset Example",
            lv::addDirSlashIfMissing(SAMPLES_DATA_ROOT)+"custom_dataset_ex/",
            "results_test",
            std::vector<std::string>{"batch1","batch2","batch3"},
            std::vector<std::string>(),
            true,
            false,
            false,
            1.0
        );

        // Below is the rest of the code needed to go through the entire dataset and process the data of all
        // work batches. Since the dataset does not contain groundtruth, the 'push' function called with the
        // processing result simply counts packets instead of doing the evaluation. In the end, this allows
        // a high-level report to still be generated and written to disk with the processing time and other
        // useful metadata on the session duration and framework version.

        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false); // returns a list of all work batches in the dataset without considering hierarchy
        lvAssert__(vpBatches.size()>0 && pDataset->getInputCount()>0,"Could not parse any data for dataset '%s'",pDataset->getName().c_str()); // check that data was indeed properly parsed
        std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<EdgeDetectorLBSP>(); // instantiate an edge detector algo with default parameters
        cv::Mat oEdgeMask; // no need to preallocate the output matrix (the algo will make sure it is allocated at some point)
        size_t nProcessedBatches = 0; // used to keep track of how many work batches have been processed (for display purposes only)
        for(auto pBatchIter = vpBatches.begin(); pBatchIter!=vpBatches.end(); ++pBatchIter) { // loop over all batches (or over all image sets, in this case)
            DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(**pBatchIter); // get rid of the iterator to pointer for cleanliness, and cast it for full interface accessibility
            std::cout << "\tProcessing batch '" << oBatch.getName() << "' [" << ++nProcessedBatches << "/" << vpBatches.size() << "]" << std::endl;
            const size_t nTotPackets = oBatch.getImageCount(); // get the total number of images to process in this batch (this function becomes available due to the edge detection task template specialization)
            size_t nProcessedPackets = 0; // used to keep track of how many packets have been processed in this work batch
            oBatch.startProcessing(); // will initialize real-time evaluation components (if any), and call the specialized dataset initialization routine (if available)
            while(nProcessedPackets<nTotPackets) { // loop over all data packets (or images, in this case)
                cv::Mat oImage = oBatch.getInput(nProcessedPackets++); // will return the 'input' data packet to be processed, based on its packet index (in this case, simply an image)
                std::cout << "\t\tProcessing packet [" << nProcessedPackets << "/" << nTotPackets << "]" << std::endl;
                pAlgo->apply(oImage,oEdgeMask); // apply the edge detector on an image, and fetch the result simultaneously
                oBatch.push(oEdgeMask,nProcessedPackets-1); // push the last edge detection result for evaluation and/or logging, if needed
            }
            oBatch.stopProcessing(); // releases all real-time evaluation components, and halts data precaching (if it was activated)
            const double dTimeElapsed = oBatch.getFinalProcessTime(); // returns the time elapsed between the 'startProcessing' and 'stopProcessing' calls for this work batch
            const double dProcessSpeed = (double)nProcessedPackets/dTimeElapsed; // evaluate the average processing speed of the algorithm for this work batch
            std::cout << "\tBatch '" << oBatch.getName() << "' done at ~" << dProcessSpeed << " Hz" << std::endl;
        }
        pDataset->writeEvalReport(); // will write a basic evaluation report listing processed packet counts, processing speed, session duration, and framework version
        std::cout << "All done!\n" << std::endl;

#endif //!USE_MIDDLEBURY_SPECIALIZATION

    }
    catch(const lv::Exception&) {std::cout << "\nmain caught lv::Exception (check stderr)\n" << std::endl; return -1;}
    catch(const cv::Exception&) {std::cout << "\nmain caught cv::Exception (check stderr)\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}
