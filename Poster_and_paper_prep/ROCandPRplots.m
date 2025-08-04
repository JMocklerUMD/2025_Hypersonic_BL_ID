main_folder_path = 'C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\Poster_and_paper_prep\Wave_packet_detection_data';
data_file_TPR = fullfile(main_folder_path, 'TPRs.txt');
fileID_TPR = fopen(data_file_TPR, "r");

data_file_FPR = fullfile(main_folder_path, 'FPRs.txt');
fileID_FPR = fopen(data_file_FPR, "r");

data_file_Pres = fullfile(main_folder_path, 'Press.txt');
fileID_Pres = fopen(data_file_Pres, "r");

TPR_CF33 = [];
TPR_CF45 = [];
TPR_LR33 = [];
TPR_LR34 = [];
TPR_LR38 = [];
TPR_T9 = [];

FPR_CF33 = [];
FPR_CF45 = [];
FPR_LR33 = [];
FPR_LR34 = [];
FPR_LR38 = [];
FPR_T9 = [];

Pres_CF33 = [];
Pres_CF45 = [];
Pres_LR33 = [];
Pres_LR34 = [];
Pres_LR38 = [];
Pres_T9 = [];

i = 0;

while i < 50
    lineTPR = fgetl(fileID_TPR);
    lineFPR = fgetl(fileID_FPR);
    linePres = fgetl(fileID_Pres);
    i = i +1;
    datTPR  = strsplit(lineTPR, '\t');
    datFPR  = strsplit(lineFPR, '\t');
    datPres  = strsplit(linePres, '\t');
    
    TPR_CF33(i) = str2double(datTPR{1});
    TPR_CF45(i) = str2double(datTPR{2});
    TPR_LR33(i) = str2double(datTPR{3});
    TPR_LR34(i) = str2double(datTPR{4});
    TPR_LR38(i) = str2double(datTPR{5});
    TPR_T9(i) = str2double(datTPR{6});

    FPR_CF33(i) = str2double(datFPR{1});
    FPR_CF45(i) = str2double(datFPR{2});
    FPR_LR33(i) = str2double(datFPR{3});
    FPR_LR34(i) = str2double(datFPR{4});
    FPR_LR38(i) = str2double(datFPR{5});
    FPR_T9(i) = str2double(datFPR{6});

    Pres_CF33(i) = str2double(datPres{1});
    Pres_CF45(i) = str2double(datPres{2});
    Pres_LR33(i) = str2double(datPres{3});
    Pres_LR34(i) = str2double(datPres{4});
    Pres_LR38(i) = str2double(datPres{5});
    Pres_T9(i) = str2double(datPres{6});
end

%% Correct the weird NaN's
TPR_CF33(end) = 0;
TPR_CF45(end) = 0;
TPR_LR33(end) = 0;
TPR_LR34(end) = 0;
TPR_LR38(end) = 0;
TPR_T9(end) = 0;

Pres_CF33(end) = 1;
Pres_CF45(end) = 1;
Pres_LR33(end) = 1;
Pres_LR34(end) = 1;
Pres_LR38(end) = 1;
Pres_T9(end) = 1;

FPR_CF33(end) = 0;
FPR_CF45(end) = 0;
FPR_LR33(end) = 0;
FPR_LR34(end) = 0;
FPR_LR38(end) = 0;
FPR_T9(end) = 0;


%% 
figure
subplot(1,3,1); hold on
plot(TPR_T9, FPR_T9, 'b-o') %NOTE THESE ARE SWITCHED
plot(0:0.01:1, 0:0.01:1, 'k--')
ylabel('True Positive Rate','Interpreter','latex')
xlabel('False Positive Rate','Interpreter','latex')
%title('ROC Curves', 'Interpreter','latex')
legend('T9', 'interpreter', 'latex', 'location','southeast')
grid minor

subplot(1,3,2)
plot(FPR_CF33, TPR_CF33, 'b-o'); hold on;
plot(FPR_CF45, TPR_CF45, 'r-o');
plot(0:0.01:1, 0:0.01:1, 'k--')
%ylabel('True Positive Rate','Interpreter','latex')
xlabel('False Positive Rate','Interpreter','latex')
legend('CF 3.3', 'CF 4.5', 'Interpreter','latex','location','southeast')
grid minor

subplot(1,3,3); hold on;
plot(FPR_LR38, TPR_LR38, 'b-o');
plot(FPR_LR34, TPR_LR34, 'r-o');
plot(FPR_LR33, TPR_LR33, '-o', 'Color',[0.9290    0.6940    0.1250]); 
plot(0:0.01:1, 0:0.01:1, 'k--')
%ylabel('True Positive Rate','Interpreter','latex')
xlabel('False Positive Rate','Interpreter','latex')
legend('Lang 6.6', 'Lang 7.9', 'Lang 9.7', 'Interpreter','latex', 'location','southeast')
grid minor

ax = gca;
ax.TickLabelInterpreter = 'latex';

set(gcf, 'units', 'centimeters')
set(gcf, 'Position', [1, 1, 40, 10])


figure
subplot(1,3,1); hold on
plot(FPR_T9, Pres_T9, 'b-o') %NOTE THESE ARE SWITCHED
plot(1:-0.01:0, 0:0.01:1, 'k--')
ylabel('Precision','Interpreter','latex')
xlabel('Recall','Interpreter','latex')
%title('PR Curves', 'Interpreter','latex')
legend('T9', 'interpreter', 'latex', 'Location','southwest')
grid minor

subplot(1,3,2)
plot(TPR_CF33, Pres_CF33, 'b-o'); hold on;
plot(TPR_CF45, Pres_CF45, 'r-o');
plot(1:-0.01:0, 0:0.01:1, 'k--')
%ylabel('Precision','Interpreter','latex')
xlabel('Recall','Interpreter','latex')
legend('CF 3.3', 'CF 4.5', 'Interpreter','latex', 'Location','southwest')
grid minor

subplot(1,3,3)
plot(TPR_LR38, Pres_LR38, 'b-o'); hold on;
plot(TPR_LR34, Pres_LR34, 'r-o');
plot(TPR_LR33, Pres_LR33, '-o', 'Color',[0.9290    0.6940    0.1250]);
plot(1:-0.01:0, 0:0.01:1, 'k--')
%ylabel('Precision','Interpreter','latex')
xlabel('Recall','Interpreter','latex')
legend('Lang 6.6', 'Lang 7.9', 'Lang 9.7', 'Interpreter','latex', 'Location','southwest')
grid minor

ax = gca;
ax.TickLabelInterpreter = 'latex';

set(gcf, 'units', 'centimeters')
set(gcf, 'Position', [1, 1, 40, 10])

% figure;
% plot(FT9FPRs, FT9TPRs, 'b-o', 'LineWidth', 1.5)
% hold on;
% plot(NFT9FPRs, NFT9TPRs,'r-o', 'LineWidth', 1.5)
% plot(0:1, 0:1, 'k--')
% legend('Filtered Tunnel 9', 'Filtered/Normalized Tunnel 9', 'Random Classifier', 'Location','southeast', 'Interpreter','latex')
% ylabel('True Positive Rate','Interpreter','latex')
% xlabel('False Positive Rate','Interpreter','latex')
% title('Filtered Model Deployment ROC Curve', 'Interpreter','latex')
% grid minor
% ax = gca;
% ax.TickLabelInterpreter = 'latex';
% 
% 
% x = 0:0.1:1;
% y = -x + 1;
% figure;
% plot(FT9TPRs, FT9Pres, 'b-o', 'LineWidth', 1.5)
% hold on;
% plot(NFT9TPRs, NFT9Pres,'r-o', 'LineWidth', 1.5)
% plot(x, y, 'k--')
% legend('Filtered Tunnel 9', 'Filtered/Normalized Tunnel 9', 'Random Classifier', 'Location','southeast', 'Interpreter','latex')
% ylabel('Precision','Interpreter','latex')
% xlabel('Recall','Interpreter','latex')
% title('Filtered Model Deployment PR Curve', 'Interpreter','latex')
% grid minor
% ax = gca;
% ax.TickLabelInterpreter = 'latex';



