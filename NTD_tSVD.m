function [P,Uinit, output] = NTD_tSVD(X,R,varargin)




%% Extract number of dimensions and norm of X.
N = ndims(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParameter('printitn',1,@isscalar);
% added option for calculating error
params.addParameter('errmethod','fast', @(x) ismember(x,{'fast','full','lowmem'}));
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
errmethod = params.Results.errmethod;

%% Error checking 

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end)
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = rand(size(X,n),R);
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;
%% Initializing M
XX= double(X);
idx0=(XX==0);
idx1=(XX>0);
M=zeros(size(XX));
M(idx1)=XX(idx1);
% Store the last MTTKRP result to accelerate fitness computation.
U_mttkrp = zeros(size(X, dimorder(end)), R);

if printitn>0
  fprintf('\n NTD-tSVD:\n');
end

%% Main Loop: Iterate until convergence

if (isa(X,'sptensor') || isa(X,'tensor')) && (exist('cpals_core','file') == 3)
 
    %fprintf('Using C++ code\n');
    [lambda,U] = cpals_core(X, Uinit, fitchangetol, maxiters, dimorder);
    P = ktensor(lambda,U);
    
else
    
    UtU = zeros(R,R,N);
    tic;
    for n = 1:N
        if ~isempty(U{n})
            UtU(:,:,n) = U{n}'*U{n};
        end
    end
    t_gram = toc;  % Grams of factor matrices
    
    for iter = 1:maxiters
        
        % initialize timings
        t_mt = 0; % MTTKRP
        t_back = 0; % Backsolving
        t_lamb = 0; % Normalizng
        t_err = 0; % Error
        
        fitold = fit;
        
        % Iterate over all N modes of the tensor
        [U,S,V] = tsvd(double(M));
        
        PP_01 = tprod(U(:,1:R,:), S(1:R, 1:R, :));

        P = tprod(PP_01, tran(V(:,1:R,:)));


        M=min(0,double(P).*idx0);
        M=M+XX.*idx1;

        tic; 
        if normX == 0
            iprod = sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
            fit = norm(P)^2 - 2 * iprod;
        elseif mod(iter, printitn)==0 || iter ==  1
            switch errmethod
                case 'fast'
                    % This is equivalent to innerprod(X,P).
                    iprod = sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
                    normresidual = sqrt(abs( normX^2 + norm(P)^2 - 2 * iprod) );
                case 'full'
                    normresidual = norm(full(X) - full(max(0,double(P))) );
                case 'lowmem'
                    normresidual = normdiff(X,max(0,double(P)));
            end
            fit = 1 - (normresidual / normX); %fraction explained by model
            rel_err(iter,:) = normresidual / normX;
        end
        fitchange = abs(fitold - fit); t = toc; t_err = t_err + t;
        
        % Check for convergence
        if (iter > 1) && (fitchange < fitchangetol)
            flag = 0;
        else
            flag = 1;
        end
        
        if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
            fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
        end
        
        % Check for convergence
        if (flag == 0)
            break;
        end        
        times(iter,:) = [t_mt, t_gram, t_back, t_lamb, t_err];
    end  
end


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
%P = arrange(P);
% Fix the signs
%P = fixsigns(P);

if printitn>0
    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(X,P);
    else
        switch errmethod
            case 'fast'
                iprod = sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
                normresidual = sqrt(abs( normX^2 + norm(P)^2 - 2 * iprod) );
            case 'full'
                normresidual = norm(full(X) - full(max(0,double(P))));
            case 'lowmem'
                normresidual = normdiff(X,max(0,P));
        end
        fit = 1 - (normresidual / normX); %fraction explained by model
        rel_err(iter,:) = normresidual / normX;       
    end
  fprintf(' Final f = %e \n', fit);
end

output = struct;
output.params = params.Results;
output.iters = find(rel_err>0);
output.relerr = rel_err(rel_err>0);
output.fit = fit;
output.times = times;
output.P = P; 
