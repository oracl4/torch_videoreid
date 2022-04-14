function [map, r1_precision] = test_func(feat_path)
  
  addpath 'utils/'
  load('info/query_IDX.mat');  % load pre-defined query index
  track_test = importdata('info/tracks_test_info.mat');
  
  % train, gallery, and query labels
  label_gallery = track_test(:, 3);
  label_query = label_gallery(query_IDX);
  cam_gallery = track_test(:, 4);
  cam_query = cam_gallery(query_IDX);

  name = feat_path;

  box_feature_test = load(name); 
  box_feature_test = box_feature_test.feat;
  box_feature_test = squeeze(box_feature_test)';
  box_feature_test = box_feature_test(1:512,:);
  video_feat_test1 = process_box_feat(box_feature_test);

  video_feat_test = video_feat_test1;

  feat_gallery = video_feat_test;
  feat_query = video_feat_test(:, query_IDX);

  distance = pdist2(feat_gallery',feat_query','euclidean');

  [CMC, map, r1_pairwise, ap_pairwise] = evaluation_mars(distance, label_gallery, label_query, cam_gallery, cam_query);
  
  r1_precision = CMC(1,1)
  
  fprintf('single query:   mAP = %f, r1 precision = %f\n', map, CMC(1,1));
  
 endfunction