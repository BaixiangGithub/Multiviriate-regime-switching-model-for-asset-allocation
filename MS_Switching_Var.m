load Data_USEconVECModel
rng(200);
%% Get GDP and inflation data
%data = Quandl.get('FRED/GDP','start_date','1951-01-01','end_date','2018-08-01','type','data');
filename = 'RATEINF-INFLATION_USA.csv';
Inflation = readtable('Inflation_return-2.csv'); GDP = readtable('GDP.csv');
Pi = Inflation{:,2};  y = GDP{:,2};
%% Biviriate normal with regime-dependent variance and covariance
%ini_alph_val = num2cell([-0.6371,    1.2496,    0.6384,    1.3528,    0.42,    0.46,    0.7,    0.4,    0.2,-0.1  -0.1,    0.2]);
%ini_alph_val = num2cell([-0.1,1.6^0.5, 0.5, 1.2093,    0.5230,   0.4,    0.7939,    0.6123,    0.0001,   -0.0398,   -0.1400,    0.0957]); %CHANGE Initial Points
                       %[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4];
%ini_alph_val = num2cell([-0.6288,    1.2204,    0.6152 ,   1.2599 ,  -0.3408 ,   0.4897 ,   0.7795 ,   0.4890 ,  -0.5009 ,  -0.0909,   -0.0481,    0.4362]);   
ini_alph_val =  num2cell([-0.6904,    1.3093,    0.4824,    1.2114,   -0.2895,    0.7223,    0.6885,   0.4207,   -0.1024,    0.0355,    0.2696,    2.1302]);
[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_y_H, epsilon_pi_H, sigma_yL, sigma_piL, beta1, beta2, beta3, beta4] = deal(ini_alph_val{:});
alpha = [u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_y_H, epsilon_pi_H, sigma_yL, sigma_piL, beta1, beta2, beta4, beta4];
P = [0.9, 0.025, 0.025, 0.05;
    0.05 0.9,0.025,0.025;
    0.025,0.025,0.9,0.05;
    0.05,0.025,0.025,0.9];

T = length(y);
Hidden_phi = zeros(4,1,T,T); %% Hidden_phi(i,j) = ¦Îi|j
ini_hidden_phi =[1;0;0;0];
Step = 50; %%EM iteration
eta = zeros(4,T); %%% 4 States
%likelihood = [];alphas = []; 
Ps = [];
Prob_prev = -Inf;
thresh = 0.6;
Prob = -Inf;
criterion = 0;
%step = 1;
S_t = [[1,1];[1,0];[0,1];[0,0]];
num_ini = 1;
ini_points = [alpha];
like_diff_points = [];

for m=1:num_ini
    likelihood = []; alphas = []; hessians = zeros(12,12,1);step = 1;
    while step<=Step && Prob >= Prob_prev %- 0.001*abs(Prob_prev)
        Prob_prev = Prob;

        %covs14 = [state1cov,state4cov];
        for t = 1: T                                                    % sigma_y                 sigma_pi                beta             sty stpi
            eta(1,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL-epsilon_y_H^2, sigma_piL+epsilon_pi_H^2,beta1,y(t), Pi(t), 1,1);
            eta(2,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL-epsilon_y_H^2, sigma_piL,beta2,y(t), Pi(t), 1,0);
            eta(3,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL, sigma_piL+epsilon_pi_H^2,beta3,y(t), Pi(t), 0,1);
            eta(4,t) = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_yL, sigma_piL,beta4,y(t), Pi(t), 0,0);
            if t == 1
                %ini_hidden_phi .* eta(:,t))/(sum(ini_hidden_phi .* eta(:,t))
                Hidden_phi(:,:,t,t) = (ini_hidden_phi .* eta(:,t))/(sum(ini_hidden_phi .* eta(:,t)));
                Hidden_phi(:,:,t+1,t) = P' * Hidden_phi(:,:,t,t);
            end
            if t~= 1 && t ~=T
                Hidden_phi(:,:,t,t) = (Hidden_phi(:,:,t,t-1) .* eta(:,t) ) / sum(Hidden_phi(:,:,t,t-1) .* eta(:,t));
                Hidden_phi(:,:,t+1,t) = P' * Hidden_phi(:,:,t,t);
            end
            if t == T
                Hidden_phi(:,:,t,t) = (Hidden_phi(:,:,t,t-1) .* eta(:,t) ) / sum(Hidden_phi(:,:,t,t-1) .* eta(:,t));
            end

        end
        %%% Kim Smooth
        t = T-1;
        while t >=1
            %Hidden_phi(:,:,t+1,T)./ Hidden_phi(:,:,t+1,t) 
            Hidden_phi(:,:,t,T) = Hidden_phi(:,:,t,t) .* ( P * (Hidden_phi(:,:,t+1,T)./Hidden_phi(:,:,t+1,t)) ) ;
            t = t-1;
        end
        %%% Expected Log Likelihood, M-step:
        save('Data_for_opti2.mat','Hidden_phi','T','y','Pi');
        %%% Update Transition Probability
        [r,l] = size(P);
        summ = 0;
        P_hat = zeros(r,l);
        summ2 = 0;
        for i = 1:r
            for j = 1:l
                summ = 0; summ2 = 0;
                for t = 2:T
                    %summ  = summ + Hidden_phi(j,1,t,T) * ( (Hidden_phi(i,1,t-1,t-1)*P(i,j)) /Hidden_phi(j,1,t,t-1) ) ;
                    %summ2 = summ2 + sum( ( Hidden_phi(i,1,t-1,t-1)* Hidden_phi(:,1,t,T)./Hidden_phi(:,1,t,t-1)).*P(i,:)' );
                    summ = summ+Hidden_phi(j,1,t,T)*P(i,j)*Hidden_phi(i,1,t-1,t-1)/Hidden_phi(j,1,t,t-1);
                    summ2 = summ2 + Hidden_phi(i,1,t-1,T);
                    
                end
                %for k = 1:r
                P_hat(i,j) = summ/summ2;
                %P_hat(i,j) = summ/summ2;
            end
            %P_hat(i,:) = P_hat(i,:)/sum(P_hat(i,:));
        end    
        P = P_hat;
        %Exp = EL(alpha);%,Hidden_phi,T,y,Pi);
        %%% Update alpha
        alpha0 = alpha;
        %    [u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4]
        lb = [min(y),-3,min(Pi),-3, -Inf, -Inf, 0.0001, 0.0001,  -Inf,  -3,  -3,  -3];
        ub = [1,   3,  1,  3,  Inf, Inf,  2,    2,          Inf,      3,     3,      3];
        s = size(alpha); 
        A= zeros(s(2),s(2)); b = ones(s(2),1);
        Aeq = zeros(s(2),s(2)); beq = zeros(s(2),1);
        %options = optimoptions(@fminunc,'Display','notify-detailed' );
        options = optimoptions(@fmincon,'Algorithm','sqp','MaxIterations',1600);
        [alpha,fval,exitflag,output,lambda,grad,hessian] = fmincon(@EL2,alpha0,A,b,Aeq,beq,lb,ub,@mycon2,options);
        ini_alph_val1 = num2cell(alpha);
        [u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_y_H, epsilon_pi_H, sigma_yL, sigma_piL,beta1,beta2,beta3,beta4]  = deal(ini_alph_val1{:});
        %likelihood = [likelihood; fval];

        likelihood = [likelihood;-fval];
        alphas = [alphas;alpha];
        hessians(:,:,step) = hessian;
        %Ps = [Ps;P];
        step = step + 1;
        s  = -fval;
        Prob = -fval;


    end
    'YES'
    Prob_prev =-Inf;Prob = -1e7;
    like_diff_points = [like_diff_points; likelihood(end-1)]; %Store Likelihood for different initial points. 
    %Print like_diff_points so you know what are the results you get based
    %on Likelihood
    
    err = sqrt(diag(inv(hessians(:,:,end-1)))); % Standard error of parameters
    err
    
    alpha0 = err.*(rand(12,1)-0.5) + alphas(end-1,:)';  %CHANGE generate new start points within a range of Standard error. 
    ini_alph_val = num2cell(alpha0');
                       %[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4];
                     
    [u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4]= deal(ini_alph_val{:});
    alpha = [u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4];
    
    ini_points = [ini_points;alpha0']; % Store all initial points. 
    
end



if Hidden_phi(1,1,T,T) ~= 0
    Optimizations = containers.Map;
    Optimizations('log_likelihood') = likelihood;
    Optimizations('alphas') =alphas;
    Optimizations('Trainsitions')  = Ps; 
    figure(1);
    plot(Optimizations('log_likelihood')); % Plot log_likelihood of each step. 
    xlabel('EM-Steps')
    ylabel('Log Likelihood')
    title(strcat(num2str(Step),' Steps'))

    figure(2);
    dateplot = datenum(Inflation{:,1});
    smooth_hh = Hidden_phi(1,1,:,T);
    plot(dateplot,smooth_hh(:));
    ax = gca;
    ax.XTick = dateplot(1:1:end);
    datetick('x','yyyy')
    xlabel('Time');
    ylabel('Probability');
    axis tight;
    title('Smooth probability of the regime (GDP^H,Inflation^H)');

    figure(3);
    dateplot = datenum(Inflation{:,1});
    smooth_hh = Hidden_phi(2,1,:,T);
    plot(dateplot,smooth_hh(:));
    ax = gca;
    ax.XTick = dateplot(1:1:end);
    datetick('x','yyyy')
    xlabel('Time');
    ylabel('Probability');
    axis tight;
    title('Smooth probability of the regime (GDP^H,Inflation^L)');

    figure(4);
    dateplot = datenum(Inflation{:,1});
    smooth_hh = Hidden_phi(3,1,:,T);
    plot(dateplot,smooth_hh(:));
    ax = gca;
    ax.XTick = dateplot(1:1:end);
    datetick('x','yyyy')
    xlabel('Time');
    ylabel('Probability');
    axis tight;
    title('Smooth probability of the regime (GDP^L,Inflation^H)');

    figure(5);
    dateplot = datenum(Inflation{:,1});
    smooth_hh = Hidden_phi(4,1,:,T);
    plot(dateplot,smooth_hh(:));
    ax = gca;
    ax.XTick = dateplot(1:1:end);
    datetick('x','yyyy')
    xlabel('Time');
    ylabel('Probability');
    axis tight;
    title('Smooth probability of the regime (GDP^L,Inflation^L)');
end

% save('Data_for_hidden_unit.mat','Hidden_phi');
        
        
%-0.6642    1.2946    0.0085   -0.0000   -0.0076   -0.6520   -0.0037        
        
% ini: -0.1000    1.2649    0.5000    1.2093    0.5230    0.4000    0.7939    0.6123    0.0001   -0.0398    0.0957    0.0957

%1) 1 std deviation search
% ini -0.6371    1.2496    0.6384    1.3528    0.4535    0.0194    0.9954    0.5115    0.1941   -0.1255   -0.5682    1.3473
%result -0.6680    1.2450    0.6405    1.3319    0.1435   -0.0000    0.8027    0.5122    0.0001   -0.1022   -0.4017    1.4045
%P
%{
P=
    0.6600    0.1068    0.1897    0.0435
    0.0115    0.9747    0.0000    0.0138
    0.5154    0.0000    0.4846    0.0000
    0.0000    0.3105    0.0000    0.6895
ini_points =

   -0.6371    1.2496    0.6384    1.3528    0.4200    0.4600    0.7000    0.4000    0.2000   -0.1000    0.2000    0.2000
   -0.6288    1.2204    0.6152    1.2599   -0.3408    0.4897    0.7795    0.4890   -0.5009   -0.0909   -0.0481    0.4362
   -0.6157    1.1932    0.5942    1.1186    0.5354    0.6364    0.9008    0.4407   -0.5290   -0.0559    0.9263    1.6122


P =

    0.7913    0.0385    0.1702    0.0000
    0.0159    0.9651    0.0059    0.0131
    0.1620    0.1611    0.6401    0.0368
    0.0000    0.3854    0.0014    0.6133

alpha = -0.5025    1.1909    0.5745    1.1633    0.4274    0.6099    0.9246    0.4343    0.0002   -0.0478    0.0001    1.6636

%}
% ini: -0.4187    1.1665    0.5885    1.1852    0.4022    0.6232    0.8323    0.4141    0.0181   -0.0210    0.8681    1.5596
%Likelihood -495.16 significant improvement

%-495: -0.4611    1.2265    0.5517    1.2010    0.4766    0.5333    0.8760    0.4365    0.2065   -0.0087    0.2062    1.8295
% -488.4233: -0.4162    1.1717    0.5124    1.0961   -0.6090    0.6848    0.7937    0.4197   -0.0713   -0.0306    0.2972    2.3057
% -484.89  -0.6904    1.3093    0.4824    1.2114   -0.2895    0.7223    0.6885    0.4207   -0.1024    0.0355    0.2696    2.1302

