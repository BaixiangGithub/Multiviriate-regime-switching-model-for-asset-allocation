function [expect_liklihood] = EL2(alpha)
%[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, w_h ,w_l,  beta2, beta3]
data = load('Data_for_opti2.mat');
Hidden_phi = data.Hidden_phi;
T = data.T;
Pi = data.Pi;
y = data.y;
%[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4]
u_y_l = alpha(1); gamma_y = alpha(2); u_pi_l = alpha(3); gamma_pi = alpha(4);
epsilon_y_H= alpha(5); epsilon_pi_H = alpha(6); sigma_yL = alpha(7); sigma_piL = alpha(8);
beta1 = alpha(9); beta2 = alpha(10); beta3= alpha(11); beta4 = alpha(12);

for t = 1: T
    
    eta(1,1,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL-epsilon_y_H^2, sigma_piL+epsilon_pi_H^2,beta1,y(t), Pi(t), 1,1);
    eta(1,2,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL-epsilon_y_H^2, sigma_piL,beta2,y(t), Pi(t), 1,0);
    eta(1,3,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL, sigma_piL+epsilon_pi_H^2,beta3,y(t), Pi(t), 0,1);
    eta(1,4,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL, sigma_piL,beta4,y(t), Pi(t), 0,0);
    
    
end

expect_liklihood = 0;
for t = 1:T
    
    expect_liklihood = expect_liklihood + log(eta(:,:,t))*Hidden_phi(:,:,t,T);
end
expect_liklihood = -expect_liklihood;
