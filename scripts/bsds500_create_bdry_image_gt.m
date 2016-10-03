clear all; close all;

if(isdir('../bench/benchmarks/'))
    addpath('../bench/benchmarks/')
else
    error 'cannot find benchmark script folder'
end
if(isdir('./images/') && isdir('./groundTruth/'))
    gtBaseDir = './groundTruth/';
else
    error 'cannot find image/gt folders'
end
outBaseDir = './groundTruth_bdry_images/';
if(~isdir(outBaseDir))
    mkdir(outBaseDir);
end

subDirs = {'train/','val/','test/'};

for s=1:numel(subDirs)
    gtDir = [gtBaseDir subDirs{s}];
    outDir = [outBaseDir subDirs{s}];
    if(~isdir(outDir))
        mkdir(outDir);
    end
    iids = dir(fullfile(gtDir,'*.mat'));
    for i = 1:numel(iids),
        gtFile = fullfile(gtDir, iids(i).name);
        clear groundTruth;
        load(gtFile);
        if isempty(groundTruth),
            error('could not open gt file');
        elseif numel(groundTruth)==0
            error('no gt boundaries found')
        elseif numel(groundTruth)>255
            error('too many gt boundaries for current impl')
        end
        outCurrDir = [outDir iids(i).name(1:end-4) '/'];
        if(~isdir(outCurrDir))
            mkdir(outCurrDir);
        end
        for g=1:numel(groundTruth)
            outFile = fullfile(outCurrDir,strcat(iids(i).name(1:end-4),'_',int2str(g-1),'.png'));
            imwrite(uint8(groundTruth{g}.Boundaries>0)*255,outFile);
        end
    end
end