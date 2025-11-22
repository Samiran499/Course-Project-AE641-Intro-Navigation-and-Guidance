%% Van der Pol Oscillator - Extended Kalman Filter
clear; clc; close all;

%% Parameters
Ts = 1;          % Sampling time
Q_scalar = 4;    % Process noise variance
R = 1;           % Measurement noise variance
N = 50;          % Number of time steps

% Process noise covariance matrix
L = [1; -1];     % Process noise input matrix
Q = L * Q_scalar * L';  % 2x2 matrix

% Measurement matrix (linear part)
H = [0, 1];

%% Initial Conditions
x_hat = [1; 1];           % Initial state estimate
P = [1, 0; 0, 1];         % Initial covariance
y_meas = 1;               % Measurement at k=1

%% Storage for plotting
x_est_history = zeros(2, N+1);
P_history = zeros(2, 2, N+1);
x_true_history = zeros(2, N+1);
y_meas_history = zeros(1, N+1);

% Store initial values
x_est_history(:, 1) = x_hat;
P_history(:, :, 1) = P;

%% Generate True States (for comparison)
% Initialize true state (you can choose different initial conditions)
x_true = [1; 1];
x_true_history(:, 1) = x_true;

% Generate process and measurement noise
rng(42);  % For reproducibility
w = sqrt(Q_scalar) * randn(1, N);  % Process noise
v = sqrt(R) * randn(1, N);          % Measurement noise

%% Main EKF Loop
for k = 1:N
    %% ===== GENERATE TRUE STATE (Simulation) =====
    if k > 1
        % True state propagation with process noise
        x_true = vanderPolDynamics(x_true, Ts) + L * w(k-1);
        x_true_history(:, k) = x_true;
    end
    
    % Generate measurement from true state
    y_meas = H * x_true + v(k);
    y_meas_history(k) = y_meas;
    
    %% ===== EKF PREDICTION STEP =====
    % Predict state using nonlinear dynamics
    x_hat_pred = vanderPolDynamics(x_hat, Ts);
    
    % Compute Jacobian F at current estimate
    F = computeJacobianF(x_hat, Ts);
    
    % Predict covariance
    P_pred = F * P * F' + Q;
    
    %% ===== EKF UPDATE STEP =====
    % Innovation (measurement residual)
    y_pred = H * x_hat_pred;  % Predicted measurement
    innovation = y_meas - y_pred;
    
    % Innovation covariance
    S = H * P_pred * H' + R;
    
    % Kalman Gain
    K = P_pred * H' / S;  % Or: K = P_pred * H' * inv(S)
    
    % Update state estimate
    x_hat = x_hat_pred + K * innovation;
    
    % Update covariance
    P = (eye(2) - K * H) * P_pred;
    
    %% Store results
    x_est_history(:, k+1) = x_hat;
    P_history(:, :, k+1) = P;
end

%% Define Nonlinear Dynamics Function
function x_next = vanderPolDynamics(x, Ts)
    % Van der Pol oscillator dynamics
    x1 = x(1);
    x2 = x(2);
    
    x_next = [x1;
              x2 + Ts * ((1 - x1^2) * x2 - x1)];
end

%% Compute Jacobian F
function F = computeJacobianF(x, Ts)
    % Jacobian of state transition function
    x1 = x(1);
    x2 = x(2);
    
    F = [1,                          0;
         Ts*(-2*x1*x2 - 1),    1 + Ts*(1 - x1^2)];
end

%% Plotting
time = 0:N;

figure('Position', [100, 100, 1200, 800]);

% Plot x1
subplot(3,1,1);
plot(time, x_true_history(1,:), 'b-', 'LineWidth', 2); hold on;
plot(time, x_est_history(1,:), 'r--', 'LineWidth', 2);
% Add uncertainty bounds
P11 = squeeze(P_history(1,1,:));
fill([time, fliplr(time)], ...
     [x_est_history(1,:) + 2*sqrt(P11)', fliplr(x_est_history(1,:) - 2*sqrt(P11)')], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time Step k');
ylabel('x_1(k)');
title('State x_1: True vs Estimated');
legend('True State', 'EKF Estimate', '95% Confidence', 'Location', 'best');
grid on;

% Plot x2
subplot(3,1,2);
plot(time, x_true_history(2,:), 'b-', 'LineWidth', 2); hold on;
plot(time, x_est_history(2,:), 'r--', 'LineWidth', 2);
plot(1:N, y_meas_history, 'go', 'MarkerSize', 4);  % Show measurements
% Add uncertainty bounds
P22 = squeeze(P_history(2,2,:));
fill([time, fliplr(time)], ...
     [x_est_history(2,:) + 2*sqrt(P22)', fliplr(x_est_history(2,:) - 2*sqrt(P22)')], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time Step k');
ylabel('x_2(k)');
title('State x_2: True vs Estimated (with measurements)');
legend('True State', 'EKF Estimate', 'Measurements', '95% Confidence', 'Location', 'best');
grid on;

% Plot estimation errors
subplot(3,1,3);
errors = x_true_history - x_est_history;
plot(time, errors(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(time, errors(2,:), 'r-', 'LineWidth', 1.5);
xlabel('Time Step k');
ylabel('Estimation Error');
title('Estimation Errors: x_{true} - x_{estimate}');
legend('Error in x_1', 'Error in x_2');
grid on;

%% Phase Portrait
figure('Position', [100, 100, 600, 600]);
plot(x_true_history(1,:), x_true_history(2,:), 'b-', 'LineWidth', 2); hold on;
plot(x_est_history(1,:), x_est_history(2,:), 'r--', 'LineWidth', 2);
plot(x_true_history(1,1), x_true_history(2,1), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
plot(x_est_history(1,1), x_est_history(2,1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1');
ylabel('x_2');
title('Phase Portrait: Van der Pol Oscillator');
legend('True Trajectory', 'EKF Estimate', 'Start (True)', 'Start (Est)');
grid on;
axis equal;

%% Display Final Results
fprintf('=== EKF Results at k=1 ===\n');
fprintf('Estimated State: x̂(1|1) = [%.4f, %.4f]''\n', x_est_history(:,2));
fprintf('True State:      x(1)    = [%.4f, %.4f]''\n', x_true_history(:,2));
fprintf('Estimation Error:         = [%.4f, %.4f]''\n', errors(:,2));
fprintf('\nFinal Covariance P(1|1):\n');
disp(P_history(:,:,2));