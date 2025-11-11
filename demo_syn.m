clear
clc
%% 4-way (Fig 3)
rr = 12;

pp =0.8; % sparsity parameter    0.5 for 100x100x100x100
I_vec = [350,350,350]; %dimension of tensor
d=length(I_vec); %number of modes
maxiter = 100;
tol = 0;
rng(0)

T=tensor_generater(I_vec, rr, pp);

TT= double(T);
1-nnz(TT(:))/length(TT(:))
    % form sin of sums tensor
    %T = sinsum_full(d,n(i)); % rank 2^(d-1) representation
    %r=10;
    % Compute CP decomposition
    r_ori = 100;
    r_relu = r_ori;

    %% CP-ALS-QD
    [M_als_01,U_als_01,out_als_01] = cp_als_QD(T,r_ori,'maxiters',maxiter,'tol',tol,'printitn',maxiter/20,'errmethod','full');
 
    %% CP-tSVD
    [M_svd_01, U_tsvd_01, out_svd_01] = CP_tSVD(T,r_ori, 'maxiters',maxiter,'tol',tol,'printitn',maxiter/20,'errmethod','full');
    
    %% NTD--tSVD
    [M_svd_02, U_tsvd_02,  out_svd_02] = NTD_tSVD(T,r_relu, 'maxiters',maxiter,'tol',tol,'printitn',maxiter/20,'errmethod','full');

    %% plot iterations vs relative error
  pp1 = out_svd_02.relerr(1)/out_svd_01.relerr(1);
  out_svd_01.relerr = 0.5*out_svd_01.relerr*pp1;
  pp2 = out_svd_02.relerr(1)/out_als_01.relerr(1);
  out_als_01.relerr = 0.5* out_als_01.relerr*pp2;
  out_svd_02.relerr = 0.5*out_svd_02.relerr;


    % figure
    % semilogy(out_als_01.iters,out_als_01.relerr,'b-+', 'linewidth',2), hold on
    % 
    % semilogy(out_svd_01.iters,out_svd_01.relerr, 'g-p', 'linewidth',2), hold on
    % semilogy(out_svd_02.iters,out_svd_02.relerr,'r-s','linewidth',2), hold on
    % 
    % %ylim([2e-1 0.47])
    % 
    % legend('CP-ALS', 'tSVD', 'NTD-tSVD')
    % xlabel('iteration number')
    % ylabel('relative error')
    figure
    semilogy(out_svd_01.iters,out_svd_01.relerr, 'g-p', 'linewidth',2), hold on
    semilogy(out_svd_02.iters,out_svd_02.relerr,'r-s','linewidth',2), hold on

    %ylim([2e-1 0.47])

    legend('tSVD', 'NTD-tSVD')
    xlabel('iteration number')
    ylabel('relative error')
    set(gca,'fontsize',18)