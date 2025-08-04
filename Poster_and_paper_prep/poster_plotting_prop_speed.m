% Poster plotting - propogation speed plots

close all; %clear

% First load everything from the propogation classification plots folder
% into the workspace (just do manually)

% The frames below are from Run33, frames 119 to 143

slice_width = 64;
FR = 285e3;
dt = 1/FR;

% % Make the histogram
% figure
% subplot(1,3,1)
% histogram(Run33_convect_speeds)
% xlim([770, 870])
% xline(mean(Run33_convect_speeds), '-')
% xline(mean(Run33_convect_speeds)-std(Run33_convect_speeds), '--')
% xline(mean(Run33_convect_speeds)+std(Run33_convect_speeds), '--')
% ylabel('Frequency, counts','Interpreter','latex')
% xlabel('Meas. prop. speed, m/s','Interpreter','latex')
% title('Re97', 'Interpreter','latex')
% ax = gca;
% ax.TickLabelInterpreter = 'latex';
% 
% subplot(1,3,2)
% histogram(Run34_convect_speeds)
% xlim([770, 870])
% xline(mean(Run34_convect_speeds), '-')
% xline(mean(Run34_convect_speeds)-std(Run34_convect_speeds), '--')
% xline(mean(Run34_convect_speeds)+std(Run34_convect_speeds), '--')
% ylabel('Frequency, counts','Interpreter','latex')
% xlabel('Meas. prop. speed, m/s','Interpreter','latex')
% title('Re79', 'Interpreter','latex')
% ax = gca;
% ax.TickLabelInterpreter = 'latex';
% 
% subplot(1,3,3)
% histogram(Run38_convect_speeds)
% xlim([725, 925])
% xline(mean(Run38_convect_speeds), '-')
% xline(mean(Run38_convect_speeds)-std(Run38_convect_speeds), '--')
% xline(mean(Run38_convect_speeds)+std(Run38_convect_speeds), '--')
% ylabel('Frequency, counts','Interpreter','latex')
% xlabel('Meas. prop. speed, m/s','Interpreter','latex')
% title('Re66', 'Interpreter','latex')
% ax = gca;
% ax.TickLabelInterpreter = 'latex';

% Make the schlierien propogation plot
figure
subplot(7,1,1); hold on
imshow(Run33_frame119)
pos = [Run33_frame119WPs(1)*slice_width, 1, ...
    (Run33_frame119WPs(2)-Run33_frame119WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0$', 'Interpreter','latex', "Rotation",0, 'FontSize',16)

subplot(7,1,2); hold on
imshow(Run33_frame123)
pos = [Run33_frame123WPs(1)*slice_width, 1, ...
    (Run33_frame123WPs(2)-Run33_frame123WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+14.0 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)

subplot(7,1,3); hold on
imshow(Run33_frame127)
pos = [Run33_frame127WPs(1)*slice_width, 1, ...
    (Run33_frame127WPs(2)-Run33_frame127WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+28.0 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)

subplot(7,1,4); hold on
imshow(Run33_frame131)
pos = [Run33_frame131WPs(1)*slice_width, 1, ...
    (Run33_frame131WPs(2)-Run33_frame131WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+42.1 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)

subplot(7,1,5); hold on
imshow(Run33_frame135)
pos = [Run33_frame135WPs(1)*slice_width, 1, ...
    (Run33_frame135WPs(2)-Run33_frame135WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+56.1 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)

subplot(7,1,6); hold on
imshow(Run33_frame139)
pos = [Run33_frame139WPs(1)*slice_width, 1, ...
    (Run33_frame139WPs(2)-Run33_frame139WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+70.2 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)

subplot(7,1,7); hold on
imshow(Run33_frame143)
pos = [Run33_frame143WPs(1)*slice_width, 1, ...
    (Run33_frame143WPs(2)-Run33_frame143WPs(1))*slice_width ,62];
rectangle('Position',pos,'EdgeColor','r', 'LineWidth',2)
ylabel('$t_0+84.2 \mu$s',"Rotation",0, 'Interpreter','latex', 'FontSize',16)
