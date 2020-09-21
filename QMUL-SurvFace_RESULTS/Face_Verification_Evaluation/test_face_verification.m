function test_face_verification();
  
%[TARs, FARs, mean_accuracy] 

%%% for verification, we use two-metrics: TARs@FARs and mean-accuracy for mean-accuracy, we follow the 10-fold validation 

    initial_threshold = 1.1;
    range = 0.9;
    step = 0.001;
    

    %% load the feature map
    N = load('../../../QMUL-SurvFace/Face_Verification_Test_Set/mixed_models_012/neg_pairs.mat');
   
    neg_pair_scores = N.scores.';
    
    P = load('../../../QMUL-SurvFace/Face_Verification_Test_Set/mixed_models_012/pos_pairs.mat');
    
    pos_pair_scores = P.scores.'; 
    
    
  
    % we assume (1). you have extracted the feature map for both positive and negative pairs 
    % following the order of positve/negative-pairs-names we give; 
    % (2). you have calculated the distance (score) between each pairs, and
    % save them as pos_pair_scores (positive) & neg_pair_scores (negative)
    % (both should be 5320_by_1 matrix)
    
    %% calculate the 1st metric: TAR@FAR
    thresholds = [];
    FARs = [];
    t = 1;
    for threshold = initial_threshold - range:step:initial_threshold + range
        current_far = (sum(neg_pair_scores<threshold)) ...
            /(length(neg_pair_scores)); % we use "<" here, because we assume you use the features DISTANCE as "scores": smaller means more similar
        thresholds(t,1) = threshold;
        FARs(t,1) = current_far;
        t = t + 1;
    end

    TARs = [];
    for threshold_index = 1:length(thresholds)
        %current_tar = 1;
        
        %TARs(threshold_index,1) = current_tar;
        
        current_tar = (sum(pos_pair_scores<thresholds(threshold_index))) /(length(pos_pair_scores)) ;
        TARs(threshold_index,1) = current_tar ;
    end

    % draw result
    [FARs,far_index] = sort(FARs,'ascend');
    TARs = TARs(far_index);
    thresholds = thresholds(far_index);
    %figure, 
    %plot(FARs,TARs);

    % show the TAR@FAR=0.3/0.1/0.01/0.001 results & AUC (area under the curve)
    FARs_03 = abs(FARs-0.3);
    [~, FARs_03_idx] = sort(FARs_03);
    TAR_FAR_03 = TARs(FARs_03_idx(1));
    FARs_01 = abs(FARs-0.1);
    [~, FARs_01_idx] = sort(FARs_01);
    TAR_FAR_01 = TARs(FARs_01_idx(1));
    FARs_001 = abs(FARs-0.01);
    [~, FARs_001_idx] = sort(FARs_001);
    TAR_FAR_001 = TARs(FARs_001_idx(1));
    FARs_0001 = abs(FARs-0.001);
    [~, FARs_0001_idx] = sort(FARs_0001);
    TAR_FAR_0001 = TARs(FARs_0001_idx(1));
    AUC = trapz(FARs,TARs);

    display(['TAR@FAR=0.3/0.1/0.01/0.001:   ' num2str(TAR_FAR_03) ' /  ' num2str(TAR_FAR_01) '  / ' num2str(TAR_FAR_001) '  / ' num2str(TAR_FAR_0001)]);
    display(['AUC:  ' num2str(AUC)])


    %% calculate the 2nd metric: mean accuracy (10-folds validation)
    accuracy = zeros(10,1);
    fold_length = size(pos_pair_scores,1)/10;
    for i = 1:10
        test_fold_p = pos_pair_scores((i-1)*fold_length+1:i*fold_length,1);
        test_fold_n = neg_pair_scores((i-1)*fold_length+1:i*fold_length,1);
        test_index = zeros(length(pos_pair_scores),1);
        test_index((i-1)*fold_length+1:i*fold_length,1) = 1;
        test_index = logical(test_index);
        train_fold_p = pos_pair_scores(test_index == 0);
        train_fold_n = neg_pair_scores(test_index == 0);
        temp_accuracy = 0;
        optimal_threshold = initial_threshold;
        for threshold = initial_threshold - range:step:initial_threshold + range
            current_accuracy = (sum(train_fold_p<threshold)+sum(train_fold_n>threshold)) ...
                /(length(train_fold_p)+length(train_fold_n));
            if current_accuracy > temp_accuracy
                temp_accuracy = current_accuracy;
                optimal_threshold = threshold;
            end
        end
        accuracy(i,1) = (sum(test_fold_p<optimal_threshold)+sum(test_fold_n>optimal_threshold)) ...
                /(length(test_fold_p)+length(test_fold_n));
    end
    mean_accuracy = mean(accuracy);
    display(['mean accuracy:  ' num2str(mean_accuracy)]);
    
end
