% Input arguments
actions = ["bend" "jack" "jump" "pjump" "run" "side" "skip" "walk" "wave1" "wave2"];
subjects = ["daria" "denis" "eli" "ido" "ira" "lena" "lyova" "moshe" "shahar"];
results_file = "Results/results.csv";
data_folder = "Data/Weizmann/";
random_seed = 1;

rng(random_seed);

actions_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

% Construct LMP descriptors for each video sequence
for i=1:length(actions)
    action = actions(i);
    subjects_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for j=1:length(subjects)
        subject = subjects(j);
        postfix = "";
        if (strcmp(subject, "lena") == 1) && (strcmp(action, "run") == 1 || strcmp(action, "skip") == 1 || strcmp(action, "walk") == 1)
            postfix = "1";
        end
        video_name = strcat(subject, '_', action, postfix);
        
        % Read video sequence
        video_path = char(strcat(data_folder, action, '/', video_name, '.avi'));
        video = VideoReader(video_path);
        video_sequence = [];
        while hasFrame(video)
            video_frame = double(rgb2gray(readFrame(video)));
            video_sequence = cat(3, video_sequence, video_frame);
        end
        
        % Construct LMP descriptors
        features = LMP(video_sequence, 8, 24, 2);
        reduced_descriptors = RandomProjection(features, 128);
        
        subjects_map(char(subject)) = reduced_descriptors;
    end
    actions_map(char(action)) = subjects_map;
end

count = 0;
true_positives = 0;

output_file = fopen(results_file, 'w');

% Perform leave-one-out classification
for i=1:length(actions)
    action = actions(i);
    subjects_map = actions_map(char(action));
    for j=1:length(subjects)
        subject = subjects(j);
        test_descriptor = subjects_map(char(subject));
        test_name = strcat(subject, '_', action);
        
        dictionaries = [];
        
        for k=1:length(actions)
            training_descriptors = [];
            for m=1:length(subjects)
                map = actions_map(char(actions(k)));
                descriptors = map(char(subjects(m)));
                training_descriptors = [training_descriptors descriptors];
            end
            
            % Learn dictionary for each class
            D = rand(128, 256);
            D = KSVD(D, training_descriptors, 20, 12);
            dictionaries = [dictionaries {D}];
        end

        % Use RSR to estimate class of video sequence
        predicted_class = RSR(test_descriptor, dictionaries, 100000000000, 12, actions);
        
        fprintf(output_file, "%s;%s\n", test_name, predicted_class);
        
        count = count + 1;
        if predicted_class == action
            true_positives = true_positives + 1;
        end
    end
end

acc = true_positives / count

fclose(output_file);
