clc;
clear;

% A = importdata('A_mat.data');
% A=[10 36 7 1 4; 
%     45 15 -60 15 32;
%     47 -13 -2 10 17;
%     -1 25 4 6 24;
%     4 21 55 12 6;];
% A_rot1 = [
%     39 -44 -1 -36 25;
% 18 5 -14 -47 -9;
% -13 40 -17 11 17;
% -4 8 45 21 -49;
% -26 5 -25 21 18;
% 
% ];
A = [
    10 3 7 1;
    4 9 -6 15;
    47 -13 -2 1;
    -1 35 4 6;];

[U,S,V] = svd(A);

A_recon= U*S*(V');
%error in each matrix seperately
%Our_U = importdata('U_mat.data');
% Our_S = importdata('S_mat.data');
%Our_V = importdata('V_mat.data');

