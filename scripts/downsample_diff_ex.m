clear all;
close all;
str_base = '/shared/datasets/CDNet2014/dataset/dynamicBackground/fall/';
loop = 1:1000;
windowsize = 100;

str = [str_base 'input/in' sprintf('%0.6d',1) '.jpg'];
img_curr = imread(str);
img_fullsize = zeros([loop(end) size(img_curr)]);
img_downsampled = zeros([loop(end) size(imresize(img_curr,0.125,'bilinear'))]);
orig_dims = [size(img_curr,1) size(img_curr,2)];

for i=loop
    disp(sprintf('frame = %d of %d\n',i,loop(end)))
    str = [str_base 'input/in' sprintf('%0.6d',i) '.jpg'];
    img_curr = imread(str);
    img_fullsize(i,:,:,:) = img_curr;
    img_downsampled(i,:,:,:) = imresize(img_curr,0.125,'bilinear');
    if i>windowsize
        if windowsize==1
            img_mean1 = squeeze(img_downsampled(i,:,:,:));
            img_mean2 = squeeze(img_downsampled(i-1,:,:,:));
        else
            img_mean1 = squeeze(uint8(mean(img_downsampled(i-(windowsize-1):i,:,:,:),1)));
            img_mean2 = squeeze(uint8(mean(img_downsampled(i-(windowsize/4):i,:,:,:),1)));
        end
        img_curr_diff = imabsdiff(img_mean1,img_mean2);
        img_curr_diff_zoom = imresize(img_curr_diff,orig_dims,'nearest');
        img_final_row1 = cat(2,img_curr,img_curr,imresize(imresize(img_curr,0.125,'bilinear'),orig_dims,'nearest'));
        img_final_row2 = cat(2,img_curr_diff_zoom,imresize(img_mean1,orig_dims,'nearest'),imresize(img_mean2,orig_dims,'nearest'));
        imshow(cat(1,img_final_row1,img_final_row2));
        pause(0.03)
        %waitforbuttonpress
    end
end
