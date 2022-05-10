clc;clear all;close all;
% Load the info .mat variable
load('data/info_test.mat');

% Load the global feature shape
load('../../../features/input/LSVID/previous/test/test_glofeat.mat');
test_feature = squeeze(mean(glofeat, 2));

% Load the feature that have been extracted
load('/home/oracl4/work_dir/mahdi/torch_videoreid/features/output/LSVID/new_featex/epoch_0.mat');

test_feature = feat;
info_query = info_test(query,:);

query_feature = test_feature(query,:);
distance = pdist2(query_feature, test_feature, 'euclidean');
[row, col] = size(query_feature);

for i=1:row
    good_index = intersect(find(info_test(:,3) == info_query(i,3)), find(info_test(:,4) ~= info_query(i,4)))';
    junk_index = intersect(find(info_test(:,3) == info_query(i,3)), find(info_test(:,4) == info_query(i,4)));
    [~,  sort_index1] = sort(distance(i,:));    
    [ap(i), CMC(i, :)] = compute_AP(good_index, junk_index, sort_index1);
end
ap = ap';

CMC = mean(CMC(:,:));
map = mean(ap);
r1_precision =  CMC(1);

fprintf('mAP = %f, r1 = %f, r5 = %f, r10 = %f, r20 = %f\n', map, CMC(1), CMC(5), CMC(10), CMC(20));