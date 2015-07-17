clear all;
close all;
clc;

lin_rel = 0.3;
range_min = 0;
range_max = 255;
input_min = 0;
input_max = 255;
output_min = 5;
output_max = input_max*lin_rel;
logst_min = -6;
logst_max = 6;

tot_iters = (range_max-range_min+1);
tot_rows = floor(sqrt(tot_iters));
tot_cols = floor(sqrt(tot_iters));
tot_diff = tot_iters-tot_rows*tot_cols;

assert(input_min>=range_min && output_min>=range_min);
assert(input_max<=range_max && output_max<=range_max);
assert(input_min<=input_max && output_min<=output_max);
x = input_min:((input_max-input_min)/(range_max-range_min)):input_max;
x_logst = ((x-input_min)./(input_max-input_min)).*(logst_max-logst_min)+logst_min;

% -------------------------------------------------------------------------

y_logst = (1./(1+exp(-x_logst*0.60))).*(output_max-output_min) + output_min;
figure();
plot(x,y_logst);
title('y_logst');
axis([range_min range_max range_min range_max]);
fprintf('y_logst = {\n');
idx = 0;
for i=1:tot_rows
    fprintf('\t');
    for j=1:tot_cols
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_logst(idx))));
    end
    fprintf('\n');
end
if tot_diff>0
    fprintf('\t');
    for i=1:tot_diff
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_logst(idx))));
    end
    fprintf('\n');
end
fprintf('};\n\n');

% -------------------------------------------------------------------------

y_lin = x.*lin_rel;
figure();
plot(x,y_lin);
title('y_lin');
axis([range_min range_max range_min range_max]);
fprintf('y_lin = {\n');
idx = 0;
for i=1:tot_rows
    fprintf('\t');
    for j=1:tot_cols
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_lin(idx))));
    end
    fprintf('\n');
end
if tot_diff>0
    fprintf('\t');
    for i=1:tot_diff
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_lin(idx))));
    end
    fprintf('\n');
end
fprintf('};\n\n');

% -------------------------------------------------------------------------

[~,idxs] = findpeaks(-abs(y_logst-x.*lin_rel));
[~,idx] = min(abs(idxs-tot_iters/2));
y_logst_lin_smooth = [y_logst(1:idxs(idx)) x(idxs(idx)+1:tot_iters).*lin_rel];
figure();
plot(x,y_logst_lin_smooth);
title('y_logst_lin_smooth');
axis([range_min range_max range_min range_max]);
fprintf('y_logst_lin_smooth = {\n');
idx = 0;
for i=1:tot_rows
    fprintf('\t');
    for j=1:tot_cols
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_logst_lin_smooth(idx))));
    end
    fprintf('\n');
end
if tot_diff>0
    fprintf('\t');
    for i=1:tot_diff
        idx = idx+1;
        fprintf('% 4d, ',uint8(floor(y_logst_lin_smooth(idx))));
    end
    fprintf('\n');
end
fprintf('};\n\n');
