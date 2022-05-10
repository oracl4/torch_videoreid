function [R1, R5, R20] = eval_func(feat_path)

    box_feature_test = load(feat_path);
    box_feature_test = box_feature_test.feat;
    box_feature_test = squeeze(box_feature_test)';
    video_feat_test = process_box_feat(box_feature_test);
    video_feat_test = box_feature_test;
    feat_gallery = video_feat_test(:,1:2:end);
    feat_query   = video_feat_test(:, 2:2:end);
    distance = EuclidDist(feat_gallery',feat_query');
    CMC = calc_CMC(distance);
    
    R1 = CMC(1);
    R5 = CMC(5);
    R20 = CMC(20);

endfunction