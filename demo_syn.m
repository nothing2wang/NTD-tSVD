clear;clc;

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

    r_ori = 100;
    r_relu = r_ori;
    %% tSVD
    [M_svd_01, U_tsvd_01, out_svd_01] = tSVD(T,r_ori, 'maxiters',maxiter,'tol',tol,'printitn',maxiter/20,'errmethod','full');
    
    %% NTD--tSVD
    [M_svd_02, U_tsvd_02,  out_svd_02] = NTD_tSVD(T,r_relu, 'maxiters',maxiter,'tol',tol,'printitn',maxiter/20,'errmethod','full');



    figure
    semilogy(out_svd_01.iters,out_svd_01.relerr, 'g-p', 'linewidth',2), hold on
    semilogy(out_svd_02.iters,out_svd_02.relerr,'r-s','linewidth',2), hold on


    legend('tSVD', 'NTD-tSVD')
    xlabel('iteration number')
    ylabel('relative error')
    set(gca,'fontsize',18)