clear; clc;

% Parameters
T = 1;
M = 10000;
N = 5;
h = T / M;

% Grid
x = linspace(0, T, M+1).';

% Model parameters
r = 0.76129;
s = 0.03119;
k = 2;
delta = 5;
beta = 5;
ggamma = 1;

% Initial condition
Y = zeros(N+1, M+1);
Y(1, :) = delta / beta;

% Precompute A[i], C[i] for i = 1..M-1
i_int = (1:M-1).';
A = 1/h^2 + k ./ (2*h*x(i_int+1));  
C = 1/h^2 - k ./ (2*h*x(i_int+1));   
C_M = -ggamma / h;                  

% Iterations n = 1..N
for n_idx = 2:(N+1)
    B = zeros(M,1);
    Ym1_i = Y(n_idx-1, 2); 
    B(1) = -2/h^2 - r*s / ((Ym1_i + s)^2) + C(1);
    Ym1_mid = Y(n_idx-1, 3:M); 
    B(2:M-1) = -2/h^2 - r*s ./ ((Ym1_mid + s).^2);
    B(M) = beta + ggamma / h;
    P = zeros(M,1);
    Yi = Y(n_idx-1, 2:M);                
    P(1:M-1) = r * ((Yi ./ (Yi + s)).^2);
    P(M) = delta;
    mainDiag = B;                          
    superDiag_full = [A; 0];              
    subDiag_full   = [0; C(2:end); C_M];  

    F = spdiags([subDiag_full, mainDiag, superDiag_full], [-1, 0, 1], M, M);

    %Solve F * W = P
    W = F \ P;
    Y(n_idx, 2:end) = W(:).';
    Y(n_idx, 1)     = Y(n_idx, 2);
end

%Residuals R[n,i] for i=1..M-1 (interior)
R = NaN(N, M-1);
for n = 1:N
    ni = n + 1;      
    cols = 2:M;     

    Ym = Y(ni, cols-1);
    Yc = Y(ni, cols);
    Yp = Y(ni, cols+1);

    xi = (x(cols)).';    

    term1 = (Yp - 2*Yc + Ym) / h^2;
    term2 = k * (Yp - Ym) ./ (2*h) ./ xi; 
    term3 = r * Yc ./ (s + Yc);

    R(n, :) = term1 + term2 - term3;
end

% RR[n] = max_i |R[n,i]|
RR = max(abs(R), [], 2);

% Print RR values
fprintf('RR values:\n');
for n = 1:N
    fprintf('RR[%d] = %.16g\n', n, RR(n));
end

% Plot: (x[i], Y[N, i]) for i=0..M
figure;
plot(x, Y(end, :), '.-');
xlabel('x'); ylabel(sprintf('Y(N=%d, x)', N));
title('Solution at iteration N');

