% Plotting 
clear; close all;

confidence = readNPY('C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\Poster_and_paper_prep\windowing_example\postprocess_confidence.npy');
image = readNPY('C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\Poster_and_paper_prep\windowing_example\postprocess_example1.npy');
slice_width = 64;

figure
imshow(image); hold on;
for i=1:length(confidence)
    annotation('textbox', [(i)/21+0.027, 0.35, 0.05, 0.1], 'String', sprintf('%.2f', confidence(i)), 'EdgeColor', 'none', 'Interpreter','latex')
    if confidence(i) > 0.5
        if i == 1
            pos = [(i-1)*slice_width+1, 1, ...
                slice_width-1 ,62];
        else
            pos = [(i-1)*slice_width, 1, ...
                   slice_width ,62];
        end
        rectangle('Position',pos,'EdgeColor','r','LineWidth',2)
    else
        continue
    end
end
rectangle('Position',[1, 25, 660 , 30],'EdgeColor','b', 'LineWidth',1)

figure
imshow(image); hold on;
filtered_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
for i=1:length(confidence)
    annotation('textbox', [(i)/21+0.027, 0.35, 0.05, 0.1], 'String', sprintf('%.2f', confidence(i)), 'EdgeColor', 'none', 'Interpreter','latex')
    if ismember(i, filtered_list)
        if i == 1
            pos = [(i-1)*slice_width+1, 1, ...
                slice_width-1 ,62];
        else
            pos = [(i-1)*slice_width, 1, ...
                   slice_width ,62];
        end
        rectangle('Position',pos,'EdgeColor','r','LineWidth',2)
    else
        continue
    end
end
rectangle('Position',[1, 25, 660 , 30],'EdgeColor','b', 'LineWidth',1)