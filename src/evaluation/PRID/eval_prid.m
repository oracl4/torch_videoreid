name = '/home/oracl4/work_dir/mahdi/scrf/videoreid/data/outputs_new/prid/scrf_test_prid.mat';

box_feature_test = load(name);
box_feature_test = box_feature_test.feat;
box_feature_test = squeeze(box_feature_test)';
video_feat_test = process_box_feat(box_feature_test);
video_feat_test = box_feature_test;
feat_gallery = video_feat_test(:,1:2:end);
feat_query   = video_feat_test(:, 2:2:end);
distance = EuclidDist(feat_gallery',feat_query');
CMC = calc_CMC(distance);