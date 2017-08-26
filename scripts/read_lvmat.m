
%// This file is part of the LITIV framework; visit the original repository at
%// https://github.com/plstcharles/litiv for more information.
%//
%// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
%//
%// Licensed under the Apache License, Version 2.0 (the "License");
%// you may not use this file except in compliance with the License.
%// You may obtain a copy of the License at
%//     http://www.apache.org/licenses/LICENSE-2.0
%//
%// Unless required by applicable law or agreed to in writing, software
%// distributed under the License is distributed on an "AS IS" BASIS,
%// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%// See the License for the specific language governing permissions and
%// limitations under the License.
%//
%/////////////////////////////////////////////////////////////////////////////
%//
%// This function provides a way to load matrix data saved in C++ using the
%// LITIV framework OpenCV utility function 'lv::write(...)'. Fully supports
%// all CV mat types, and preserves element ordering.
%//
%/////////////////////////////////////////////////////////////////////////////

function [ output ] = read_lvmat( file_path, file_type, flip_dims )
    assert(nargin>=1 && ~isempty(file_path));
    if nargin<1
        error('too few input arguments, need at least archive file path');
    end
    if nargin>4
        error('too many input arguments (max=3)');
    end
    if nargin<2
        if(numel(file_path)>4 && strcmp(file_path(end-3:end),'.txt'))
            file_type = 'plaintext';
        else
            file_type = 'binary';
        end
    end
    if nargin<3
        flip_dims = false;
    end
    if strcmp(file_type,'plaintext') || strcmp(file_type,'text') || strcmp(file_type,'txt') || strcmp(file_type,'t')
        [fd,errmsg] = fopen(file_path,'rt');
        assert(fd>0,'could not open text file at "%s" for reading (%s)',file_path,errmsg);
        try
            assert(strcmp(fscanf(fd,'%s',1),'htag') && ~isempty(fgets(fd)),'could not parse "htag" field from archive');
            assert(strcmp(fscanf(fd,'%s',1),'date') && ~isempty(fgets(fd)),'could not parse "date" field from archive');
            nDataType = int32(fscanf(fd,'nDataType %d\n',1));
            nDataDepth = int32(fscanf(fd,'nDataDepth %d\n',1));
            nChannels = int32(fscanf(fd,'nChannels %d\n',1));
            nElemSize = uint64(fscanf(fd,'nElemSize %lu\n',1));
            nElemCount = uint64(fscanf(fd,'nElemCount %lu\n',1));
            nDims = int32(fscanf(fd,'nDims %d\n',1));
            assert(strcmp(fscanf(fd,'%s',1),'anSizes'),'could not parse "anSizes" field from archive');
            anSizes = int32(fscanf(fd,'%d',nDims));
            fgets(fd);
            fgets(fd);
            assert(nChannels==int32(1+floor(double(nDataType)/8)));
            if nChannels>1
                nDims = nDims+1;
                anSizes = [anSizes; nChannels];
            end
            output = fscanf(fd,'%f',prod(anSizes));
            if mod(nDataType,8)==0
                output = uint8(output);
            elseif mod(nDataType,8)==1
                output = int8(output);
            elseif mod(nDataType,8)==2
                output = uint16(output);
            elseif mod(nDataType,8)==3
                output = int16(output);
            elseif mod(nDataType,8)==4
                output = int32(output);
            elseif mod(nDataType,8)==5
                output = single(output);
            end
            fclose(fd);
        catch err
            fclose(fd);
            rethrow(err);
        end
    elseif strcmp(file_type,'binary') || strcmp(file_type,'bin') || strcmp(file_type,'b')
        [fd,errmsg] = fopen(file_path,'r');
        assert(fd>0,'could not open binary file at "%s" for reading (%s)',file_path,errmsg);
        try
            nDataType = fread(fd,1,'int32');
            assert(nDataType>=0 && nDataType<=30);
            nElemSize = fread(fd,1,'uint64');
            nElemCount = fread(fd,1,'uint64');
            assert(nElemSize*nElemCount>0);
            nDims = fread(fd,1,'int32');
            assert(nDims>1);
            anSizes = fread(fd,nDims,'int32');
            nChannels = int32(1+floor(double(nDataType)/8));
            if nChannels>1
                nDims = nDims+1;
                anSizes = [anSizes; nChannels];
            end
            if mod(nDataType,8)==0
                output = uint8(fread(fd,prod(anSizes),'uint8'));
            elseif mod(nDataType,8)==1
                output = int8(fread(fd,prod(anSizes),'int8'));
            elseif mod(nDataType,8)==2
                output = uint16(fread(fd,prod(anSizes),'uint16'));
            elseif mod(nDataType,8)==3
                output = int16(fread(fd,prod(anSizes),'int16'));
            elseif mod(nDataType,8)==4
                output = int32(fread(fd,prod(anSizes),'int32'));
            elseif mod(nDataType,8)==5
                output = single(fread(fd,prod(anSizes),'float32'));
            elseif mod(nDataType,8)==6
                output = fread(fd,prod(anSizes),'float64');
            end
            fclose(fd);
        catch err
            fclose(fd);
            rethrow(err);
        end
    else
        error('unrecognized file type flag');
    end
    output = reshape(output,fliplr(anSizes'));
    if flip_dims
        output = permute(output,fliplr(1:nDims));
    end
end