function descriptors = LMP(video_sequence, temporal_resolution, eta, sigma)
    
    descriptors = [];
    segments = [];
    
    % Segment video
    segment = [];
    for i=1:size(video_sequence, 3)
        frame = video_sequence(:,:,i);
        segment = cat(3, segment, frame);
        if mod(i, temporal_resolution) == 0
            segments = [segments; {segment}];
            segment = [];
        end
    end
    
    % Align frames in video and find keypoints in first frame
    for i=1:length(segments)
        segment = segments{i};
        % Prealignement of frames
        first_frame = segment(:,:,1);
        aligned_frames = [];
        for j=2:temporal_resolution
            next_frame = segment(:,:,j);
            [h, aligned] = imagesAlign(next_frame, first_frame);
            aligned_frames = cat(3, aligned_frames, aligned);
        end
        % Use 2D keypoint detector
        points = kp_harrislaplace(first_frame);
        points = points(:,[1 2]);

        padding_left = eta / 2.0;
        padding_right = (eta / 2.0) - 1;
        
        % Extract and process cubes from segment
        for j=1:size(points,1)
            point = points(j,:);
            x = point(1);
            y = point(2);
            cube = [];
            for k=1:temporal_resolution
                % For each frame, extract cube-frame. Pad with zero where
                % necessary
                frame = segment(:,:,k);
                height = size(frame, 1);
                width = size(frame, 2);
                if (x - padding_left) < 1
                    if (y - padding_left) < 1
                        extract = frame(1:x+padding_right, 1:y+padding_right);
                        extract = padarray(extract, [24-size(extract,1) 24-size(extract,2)], 'pre');
                    elseif (y + padding_right) > width
                        extract = frame(1:x+padding_right, y-padding_left:width);
                        extract = padarray(extract, [24-size(extract,1) 0], 'pre');
                        extract = padarray(extract, [0 24-size(extract,2)], 'post');
                    else
                        extract = frame(1:x+padding_right, y-padding_left:y+padding_right);
                        extract = padarray(extract, [24-size(extract,1) 0], 'pre');
                    end
                elseif (x - padding_right) > height
                    if (y - padding_left) < 1
                        extract = frame(x-padding_right:height, 1:y+padding_right);
                        extract = padarray(extract, [0 24-size(extract,2)], 'pre');
                        extract = padarray(extract, [24-size(extract,1) 0], 'post');
                    elseif (y + padding_right) > width
                        extract = frame(x-padding_right:height, y-padding_left:width);
                        extract = padarray(extract, [24-size(extract,1) 24-size(extract,2)], 'post');
                    else
                        extract = frame(x-padding_right:height, y-padding_left:y+padding_right);
                        extract = padarray(extract, [24-size(extract,1) 0], 'post');
                    end
                else
                    if (y - padding_left) < 1
                        extract = frame(x-padding_left:x+padding_right, 1:y+padding_right);
                        extract = padarray(extract, [0 24-size(extract,2)], 'pre');
                    elseif (y + padding_right) > width
                        extract = frame(x-padding_left:x+padding_right, y-padding_left:width);
                        extract = padarray(extract, [0 24-size(extract,2)], 'post');
                    else
                        extract = frame(x-padding_left:x+padding_right, y-padding_left:y+padding_right);
                    end
                end
                % Perform Gaussian blur
                blurred = imgaussfilt(extract, sigma);
                cube = cat(3, cube, blurred);
            end
            
            % Compute Moment matrix, M
            descriptor = [];
            for moment=2:4
                for k=1:eta
                    for m=1:eta
                        sum = 0;
                        for n=1:temporal_resolution
                            sum = sum + cube(k, m, n)^moment;
                        end
                        sum = sum / temporal_resolution;
                        descriptor = [descriptor; sum];
                    end
                end
            end
            descriptors = [descriptors descriptor];
        end
    end
end

